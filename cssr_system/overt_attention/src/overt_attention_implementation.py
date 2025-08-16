import os
import math
import yaml
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import numpy as np
import cv2
from cv_bridge import CvBridge

# ROS 2 messages / services
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D, Twist
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed

from cssr_system.srv import SetMode, GetMode    
from cssr_system.msg import FaceDetection      

# -----------------------------------------------------------------------------
# Constants / Config
# -----------------------------------------------------------------------------

CONFIG_DEFAULTS: Dict = {
    "camera":                 "RealSenseCamera",  # "RealSenseCamera" | "PepperCamera"
    "realignment_threshold":  50,                 # degrees
    "x_offset_to_head_yaw":   0.0,
    "y_offset_to_head_pitch": 0.0,
    "social_attention_mode":  "random",          # "random" | "saliency"
    "num_faces_social_att":   3,
    "engagement_timeout":     12.0,
    "use_sound":              False,
    "use_compressed_images":  False,
    "verbose_mode":           False,
}

TOPIC_DEFAULTS: Dict = {
    "RealSenseCamera":                "/camera/color/image_raw",
    "RealSenseCameraCompressed":      "/camera/color/image_raw/compressed",
    "RealSenseCameraDepth":           "/camera/depth/image_raw",
    "RealSenseCameraDepthCompressed": "/camera/depth/image_raw/compressed",
    "FrontCamera":                    "/naoqi_driver/camera/front/image_raw",
    "DepthCamera":                    "/naoqi_driver/camera/depth/image_raw",
    "JointAngles":                    "/joint_angles",
    "Wheels":                         "/cmd_vel",
    "JointStates":                    "/joint_states",
    "RobotPose":                      "/robotLocalization/pose",
    "FaceDetection":                  "/faceDetection/data",
    "SoundLocalization":              "/soundDetection/direction",
    "SetMode":                        "/overtAttention/set_mode",
    "GetMode":                        "/overtAttention/get_mode",
}

HEAD_LIMITS = {
    "yaw":   (-2.0857,  2.0857),
    "pitch": (-0.7068,  0.6371),
}
SCANNING_LIMITS = {
    "yaw":   (-0.58353, 0.58353),
    "pitch": (-0.3,     0.0),
}

CAMERAS = {
    "pepper_front": {"vfov": 44.30, "hfov": 55.20, "width": 640, "height": 480},
    "realsense":    {"vfov": 42.50, "hfov": 69.50, "width": 640, "height": 480},
}

DEFAULT_HEAD_PITCH = -0.2
DEFAULT_HEAD_YAW   =  0.0

PATCH_RADIUS = 15
HABITUATION_RATE = 0.1
IOR_LIMIT = 50  # iterations

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0

def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
class Mode(Enum):
    DISABLED = auto()
    SOCIAL   = auto()
    SCANNING = auto()
    SEEKING  = auto()
    LOCATION = auto()

@dataclass
class AngleChange:
    d_yaw: float
    d_pitch: float

@dataclass
class CameraSpec:
    hfov_deg: float
    vfov_deg: float
    width: int
    height: int

    @classmethod
    def from_name(cls, name: str) -> "CameraSpec":
        if name.lower() in ("peppercamera", "pepper_front", "frontcamera"):
            spec = CAMERAS["pepper_front"]
        else:
            spec = CAMERAS["realsense"]
        return cls(hfov_deg=spec["hfov"], vfov_deg=spec["vfov"], width=spec["width"], height=spec["height"])

    def pixel_to_angle(self, x: float, y: float) -> AngleChange:
        cx, cy = self.width / 2.0, self.height / 2.0
        x_prop = (x - cx) / self.width
        y_prop = (y - cy) / self.height
        d_yaw   = x_prop * deg2rad(self.hfov_deg) * -1.0
        d_pitch = y_prop * deg2rad(self.vfov_deg)
        return AngleChange(d_yaw=d_yaw, d_pitch=d_pitch)

    def angle_to_pixel(self, d_yaw: float, d_pitch: float) -> Tuple[int, int]:
        x_prop = (-d_yaw) / deg2rad(self.hfov_deg)
        y_prop = ( d_pitch) / deg2rad(self.vfov_deg)
        x = int(self.width  / 2.0 + x_prop * self.width)
        y = int(self.height / 2.0 + y_prop * self.height)
        return x, y

@dataclass(slots=True)
class AttentionState:
    mode: Mode = Mode.DISABLED

    # flags / inputs
    face_detected: bool = False
    sound_detected: bool = False
    mutual_gaze_detected: bool = False
    face_within_range: bool = False
    sound_angle: float = 0.0  # radians

    # kinematics
    robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)     # x, y, theta
    head_joint_states: Tuple[float, float] = (0.0, 0.0)          # pitch, yaw
    commanded_head: Tuple[float, float] = (DEFAULT_HEAD_PITCH, DEFAULT_HEAD_YAW)

    yaw_limits: Tuple[float, float] = (HEAD_LIMITS["yaw"][0], HEAD_LIMITS["yaw"][1])
    pitch_limits: Tuple[float, float] = (HEAD_LIMITS["pitch"][0], HEAD_LIMITS["pitch"][1])
    kp_head: float = 0.5

    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)
    _scan_dir: int = 1
    _scan_step: float = 0.08
    _seek_cooldown: int = 0
    _seek_period: Tuple[int, int] = (8, 18)

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode

    def set_pose(self, x: float, y: float, theta: float) -> None:
        self.robot_pose = (x, y, theta)

    def set_head(self, pitch: float, yaw: float) -> None:
        self.head_joint_states = (pitch, yaw)

    def _apply_limits(self, pitch: float, yaw: float) -> Tuple[float, float]:
        return (
            clamp(pitch, self.pitch_limits[0], self.pitch_limits[1]),
            clamp(yaw,   self.yaw_limits[0],   self.yaw_limits[1]),
        )

    def _nudge(self, target_pitch: float, target_yaw: float) -> None:
        cp, cy = self.commanded_head
        np_ = cp + self.kp_head * (target_pitch - cp)
        ny_ = cy + self.kp_head * (target_yaw   - cy)
        self.commanded_head = self._apply_limits(np_, ny_)

    # Behavior stubs (you can extend as you wish)
    def location_attention(self, px: float, py: float, pz: float) -> int:
        xy = math.hypot(px, py)
        if xy == 0.0 and pz == 0.0: return 0
        target_yaw   = math.atan2(py, px)
        target_pitch = math.atan2(-pz, max(1e-6, xy))
        target_pitch, target_yaw = self._apply_limits(target_pitch, target_yaw)
        self._nudge(target_pitch, target_yaw)
        return 1

    def social_attention(self, realignment_threshold_deg: int, social_control: int) -> int:
        deadband = deg2rad(realignment_threshold_deg)
        if self.face_detected and self.face_within_range:
            tp, ty = 0.0, 0.0
        elif self.sound_detected:
            tp, ty = 0.0, clamp(self.sound_angle, *self.yaw_limits)
        else:
            return 0
        cp, cy = self.commanded_head
        if abs(tp - cp) < deadband and abs(ty - cy) < deadband:
            return 0
        self._nudge(tp, ty)
        return 1

    def scanning_attention(self, center_yaw: float, center_pitch: float) -> int:
        center_yaw   = clamp(center_yaw,   *self.yaw_limits)
        center_pitch = clamp(center_pitch, *self.pitch_limits)
        next_y = self.commanded_head[1] + self._scan_dir * self._scan_step
        if next_y >= self.yaw_limits[1] or next_y <= self.yaw_limits[0]:
            self._scan_dir *= -1
            next_y = clamp(next_y, *self.yaw_limits)
        self._nudge(center_pitch, next_y)
        return 1

    def seeking_attention(self, realignment_threshold_deg: int) -> int:
        if self._seek_cooldown <= 0:
            ty = self.rng.uniform(*self.yaw_limits)
            tp = self.rng.uniform(*self.pitch_limits)
            self._pending = (tp, ty)
            self._seek_cooldown = self.rng.randint(*self._seek_period)
        else:
            self._seek_cooldown -= 1
        tp, ty = getattr(self, "_pending", self.commanded_head)
        cp, cy = self.commanded_head
        deadband = deg2rad(realignment_threshold_deg)
        if abs(tp - cp) < deadband and abs(ty - cy) < deadband:
            return 0
        self._nudge(tp, ty)
        return 1

# -----------------------------------------------------------------------------
# Saliency
# -----------------------------------------------------------------------------
class SaliencyProcessor:
    def __init__(self, cam: CameraSpec):
        self.cam = cam
        self.faces_map: Optional[np.ndarray] = None
        self.previous_locations: List[Tuple[float, float, int]] = []  # (yaw, pitch, t)
        self.face_locations: List[Tuple[float, float, int]] = []

    def compute_saliency_map(self, image: np.ndarray) -> np.ndarray:
        # Placeholder: use OpenCV fine-grained saliency if available; else gradient magnitude
        try:
            sal = cv2.saliency.StaticSaliencyFineGrained_create()
            success, salmap = sal.computeSaliency(image)
            if success:
                salmap = (salmap * 255).astype(np.uint8)
                return salmap.astype(np.float32)
        except Exception:
            pass
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return mag

    def winner_takes_all(self, saliency_map: np.ndarray) -> Tuple[int, int]:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(saliency_map)
        return int(max_loc[0]), int(max_loc[1])

    def pixel_to_angle(self, x: float, y: float) -> AngleChange:
        return self.cam.pixel_to_angle(x, y)

    def _apply_gaussian(self, mat: np.ndarray, x: int, y: int, radius: int, delta: float):
        h, w = mat.shape[:2]
        xs, ys = np.meshgrid(
            np.arange(max(0, x - radius), min(w, x + radius + 1)),
            np.arange(max(0, y - radius), min(h, y + radius + 1))
        )
        cx = x; cy = y
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (radius * 0.5) ** 2))
        mat[ys, xs] = np.clip(mat[ys, xs] + delta * g, 0, np.max(mat) if mat.size else 255)

    def habituation(self, saliency_map: np.ndarray,
                    wta_map: np.ndarray,
                    previous_locations: List[Tuple[float, float, int]]) -> Tuple[np.ndarray, List[Tuple[float, float, int]]]:
        out_prev: List[Tuple[float, float, int]] = []
        for yaw, pitch, t in previous_locations:
            x, y = self.cam.angle_to_pixel(d_yaw=yaw, d_pitch=pitch)
            if 0 <= x < self.cam.width and 0 <= y < self.cam.height:
                # reduce saliency at previous winners
                self._apply_gaussian(saliency_map, x, y, PATCH_RADIUS, -HABITUATION_RATE * 255.0)
                cv2.circle(wta_map, (x, y), PATCH_RADIUS, int(0.3 * 255), -1)
            out_prev.append((yaw, pitch, t + 1))
        return saliency_map, out_prev

    def inhibition_of_return(self, saliency_map: np.ndarray,
                             wta_map: np.ndarray,
                             previous_locations: List[Tuple[float, float, int]]) -> Tuple[np.ndarray, List[Tuple[float, float, int]]]:
        out_prev: List[Tuple[float, float, int]] = []
        for yaw, pitch, t in previous_locations:
            if IOR_LIMIT < t < IOR_LIMIT + 50:
                x, y = self.cam.angle_to_pixel(d_yaw=yaw, d_pitch=pitch)
                if 0 <= x < self.cam.width and 0 <= y < self.cam.height:
                    # hard suppress region
                    cv2.circle(saliency_map, (x, y), PATCH_RADIUS, 0, -1)
                    cv2.circle(wta_map, (x, y), PATCH_RADIUS, 0, -1)
            else:
                out_prev.append((yaw, pitch, t))
        return saliency_map, out_prev

# -----------------------------------------------------------------------------
# Main Node
# -----------------------------------------------------------------------------
class OvertAttentionSystem(Node):
    def __init__(self):
        super().__init__("overt_attention")
        self.bridge = CvBridge()
        self.camera_image: Optional[np.ndarray] = None

        self.state = AttentionState()
        self.verbose = False

        ok = self.initialize()
        if not ok:
            self.get_logger().error("Initialization failed; shutting down.")
            rclpy.shutdown()

    def load_yaml(self, path: str, defaults: dict) -> dict:
        data = defaults.copy()
        try:
            with open(path, "r") as f:
                override = yaml.safe_load(f) or {}
            data.update(override)
        except Exception as e:
            self.get_logger().error(f"Could not load YAML {path}: {e}")
        return data

    def initialize(self) -> bool:
        self.get_logger().info("Initializing OvertAttentionSystem...")

        pkg_share = get_package_share_directory("cssr_system")
        cfg_path = os.path.join(pkg_share, "overt_attention", "config", "overtAttentionConfiguration.yaml")
        cfg = self.load_yaml(cfg_path, CONFIG_DEFAULTS)

        self.verbose = bool(cfg.get("verbose_mode", False))

        camera_key_raw = str(cfg["camera"])
        use_compr = bool(cfg["use_compressed_images"])
        # normalize for topics map
        if camera_key_raw.lower() == "realsensecamera":
            camera_topic_key = "RealSenseCameraCompressed" if use_compr else "RealSenseCamera"
            cam_name = "realsense"
        elif camera_key_raw.lower() in ("peppercamera", "frontcamera", "pepper_front"):
            camera_topic_key = "FrontCamera"
            cam_name = "pepper_front"
        else:
            self.get_logger().error(f"Unsupported camera type: {camera_key_raw}")
            return False

        topics_path = os.path.join(pkg_share, "overt_attention", "data", "pepperTopics.yaml")
        topics = self.load_yaml(topics_path, TOPIC_DEFAULTS)

        # pubs/subs
        self.sub_img = self.create_subscription(Image, topics[camera_topic_key], self.camera_callback, 10)
        self.sub_js  = self.create_subscription(JointState, topics["JointStates"], self.joint_states_callback, 10)
        self.sub_pose= self.create_subscription(Pose2D, topics["RobotPose"], self.robot_pose_callback, 10)
        self.sub_snd = self.create_subscription(Float32, topics["SoundLocalization"], self.sound_callback, 10)
        # Optional FaceDetection
        try:
            self.sub_face = self.create_subscription(FaceDetection, topics["FaceDetection"], self.face_callback, 10)
        except Exception:
            self.sub_face = None

        self.pub_cmdvel = self.create_publisher(Twist, topics["Wheels"], 10)
        self.pub_head = self.create_publisher(JointAnglesWithSpeed, topics["JointAngles"], 10)

        # services (only register if srv types exist)
        try:
            self.srv_set = self.create_service(SetMode, topics["SetMode"], self.set_mode_callback)
            self.srv_get = self.create_service(GetMode, topics["GetMode"], self.get_mode_callback)
        except Exception:
            self.srv_set = self.srv_get = None

        # camera spec & saliency
        self.cam_spec = CameraSpec.from_name(cam_name)
        self.saliency = SaliencyProcessor(self.cam_spec)

        # internal buffers
        self.attention_head_pitch: List[float] = []
        self.attention_head_yaw:   List[float] = []
        self.face_labels:          List[int]   = []
        self.sound_count: int = 0

        self.get_logger().info("Initialization complete.")
        return True

    # ------------------ Callbacks ------------------
    def camera_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if img is None or img.size == 0:
                self.get_logger().error("Camera image is empty")
                return
            self.camera_image = img
        except Exception as e:
            self.get_logger().error(f"CV bridge conversion failed: {e}")

    def joint_states_callback(self, msg: JointState):
        try:
            idx_pitch = msg.name.index("HeadPitch")
            idx_yaw   = msg.name.index("HeadYaw")
            pitch = msg.position[idx_pitch]
            yaw   = msg.position[idx_yaw]
            self.state.set_head(pitch, yaw)
        except ValueError:
            self.get_logger().warning("HeadPitch or HeadYaw not in JointState")
        except IndexError:
            self.get_logger().error("JointState missing positions for head joints")

    def robot_pose_callback(self, msg: Pose2D):
        self.state.set_pose(msg.x, msg.y, msg.theta)

    def sound_callback(self, msg: Float32):
        if math.isnan(abs(msg.data)):
            return
        self.state.sound_detected = True
        self.state.sound_angle = deg2rad(msg.data)  # incoming assumed degrees
        self.sound_count += 1

    def face_callback(self, msg: FaceDetection):
        # This assumes msg has fields:
        #   centroids: array of geometry_msgs/Point (x,y,z in pixels or meters?)
        #   mutual_gaze: array<bool>
        # Adjust to your actual FaceDetection definition.
        if not hasattr(msg, "centroids"):
            return
        self.attention_head_pitch.clear()
        self.attention_head_yaw.clear()
        self.face_labels.clear()

        self.state.face_detected = len(msg.centroids) > 0
        self.state.face_within_range = False
        gaze_any = False

        faces_map = np.zeros((self.cam_spec.height, self.cam_spec.width), dtype=np.float32)

        for i, c in enumerate(msg.centroids):
            # If c are in pixel units: use as-is; if in meters, convert to pixel via projection.
            x = int(c.x); y = int(c.y)
            if 0 <= x < self.cam_spec.width and 0 <= y < self.cam_spec.height:
                if hasattr(msg, "mutual_gaze") and i < len(msg.mutual_gaze):
                    gaze_any = gaze_any or bool(msg.mutual_gaze[i])
                # set face saliency (closer -> stronger if z is distance)
                scale = 1.0 / max(1e-3, float(getattr(c, "z", 1.0)))
                faces_map[y, x] = max(faces_map[y, x], 1.0 * scale)

                # convert pixel to delta angles
                ang = self.cam_spec.pixel_to_angle(x, y)
                self.attention_head_pitch.append(ang.d_pitch)
                self.attention_head_yaw.append(ang.d_yaw)
                self.face_labels.append(i + 1)

                if getattr(c, "z", 10.0) <= 2.0:
                    self.state.face_within_range = True

        self.saliency.faces_map = faces_map
        self.state.mutual_gaze_detected = self.state.mutual_gaze_detected or gaze_any

    # ------------------ Services ------------------
    def set_mode_callback(self, request, response):
        # Adjust to your SetMode.srv fields
        # Expecting something like: string state; float64 location_x/y/z
        try:
            state_str: str = request.state.lower()
            if   state_str == "disabled": self.state.set_mode(Mode.DISABLED)
            elif state_str == "social":   self.state.set_mode(Mode.SOCIAL)
            elif state_str == "scanning": self.state.set_mode(Mode.SCANNING)
            elif state_str == "seeking":  self.state.set_mode(Mode.SEEKING)
            elif state_str == "location":
                self.state.set_mode(Mode.LOCATION)
                # store desired 3D point (meters)
                self._location_target = (request.location_x, request.location_y, request.location_z)
            else:
                raise ValueError(f"Invalid state: {request.state}")
            # If your response has a boolean/int `mode_set_success`
            if hasattr(response, "mode_set_success"):
                response.mode_set_success = 1
        except Exception as e:
            self.get_logger().error(f"set_mode failed: {e}")
            if hasattr(response, "mode_set_success"):
                response.mode_set_success = 0
        return response

    def get_mode_callback(self, request, response):
        # Fill response.mode or similar according to your srv definition
        try:
            if hasattr(response, "mode"):
                response.mode = self.state.mode.name
        except Exception as e:
            self.get_logger().error(f"get_mode failed: {e}")
        return response

    # ------------------ Control loop hook ------------------
    def run_once(self):
        """
        Call this periodically from a timer. Reads inputs and updates state.commanded_head.
        You can publish a head command here.
        """
        if self.state.mode == Mode.LOCATION:
            px, py, pz = getattr(self, "_location_target", (0.0, 0.0, 0.0))
            self.state.location_attention(px, py, pz)

        elif self.state.mode == Mode.SOCIAL:
            # choose random/saliency/etc. For now, prefer sound if fresh:
            self.state.social_attention(realignment_threshold_deg=50, social_control=0)

        elif self.state.mode == Mode.SCANNING:
            # use scanning limits
            self.state.yaw_limits   = SCANNING_LIMITS["yaw"]
            self.state.pitch_limits = SCANNING_LIMITS["pitch"]
            self.state.scanning_attention(center_yaw=0.0, center_pitch=DEFAULT_HEAD_PITCH)
        elif self.state.mode == Mode.SEEKING:
            self.state.yaw_limits   = HEAD_LIMITS["yaw"]
            self.state.pitch_limits = HEAD_LIMITS["pitch"]
            self.state.seeking_attention(realignment_threshold_deg=8)
        else:
            # disabled -> gently center
            self.state._nudge(DEFAULT_HEAD_PITCH, DEFAULT_HEAD_YAW)

        # Example: publish head angles using JointAnglesWithSpeed (Pepper/naoqi_bridge)
        pitch_cmd, yaw_cmd = self.state.commanded_head
        msg = JointAnglesWithSpeed()
        msg.joint_names = ["HeadPitch", "HeadYaw"]
        msg.joint_angles = [float(pitch_cmd), float(yaw_cmd)]
        msg.speed = 0.2
        msg.relative = False
        self.pub_head.publish(msg)

    # Optional: basic saliency step (if you want to use the camera image)
    def saliency_step(self):
        if self.camera_image is None:
            return
        salmap = self.saliency.compute_saliency_map(self.camera_image)
        wta = np.full_like(salmap, 255, dtype=np.float32)
        salmap, self.saliency.previous_locations = self.saliency.habituation(
            salmap, wta, self.saliency.previous_locations
        )
        salmap, self.saliency.previous_locations = self.saliency.inhibition_of_return(
            salmap, wta, self.saliency.previous_locations
        )
        x, y = self.saliency.winner_takes_all(salmap)
        # convert to angles and optionally use as a target
        ang = self.saliency.pixel_to_angle(x, y)
        # Example: set seeking target based on saliency peak
        if self.state.mode in (Mode.SOCIAL, Mode.SEEKING):
            self.state._nudge(ang.d_pitch, ang.d_yaw)