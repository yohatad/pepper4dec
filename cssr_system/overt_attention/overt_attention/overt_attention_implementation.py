import os
import math
import yaml
import random
import rclpy
import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

# ROS 2 messages / services
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D, Twist
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from cssr_interfaces.srv import OvertAttentionSetMode    
from cssr_interfaces.msg import FaceDetection, OvertAttentionStatus      

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
    "OvertAttentionStatus":           "/overtAttention/mode",

    "SetMode":                        "/overtAttention/set_mode",
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
    "peppercamera": {"vfov": 44.30, "hfov": 55.20, "width": 640, "height": 480},
    "realsense":    {"vfov": 42.50, "hfov": 69.50, "width": 640, "height": 480},
}

DEFAULT_HEAD_PITCH = -0.2
DEFAULT_HEAD_YAW   =  0.0

PATCH_RADIUS = 15
HABITUATION_RATE = 0.1
IOR_LIMIT = 50  # iterations

def clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(v, hi))

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
        if name.lower() == "peppercamera":
            spec = CAMERAS["peppercamera"]
        else:
            spec = CAMERAS["realsense"]
        return cls(hfov_deg=spec["hfov"], vfov_deg=spec["vfov"], width=spec["width"], height=spec["height"])

    def pixel_to_angle(self, x: float, y: float) -> AngleChange:
        cx, cy = self.width / 2.0, self.height / 2.0
        x_prop = (x - cx) / self.width
        y_prop = (y - cy) / self.height
        d_yaw   = x_prop * math.radians(self.hfov_deg) * -1.0
        d_pitch = y_prop * math.radians(self.vfov_deg)
        return AngleChange(d_yaw=d_yaw, d_pitch=d_pitch)

    def angle_to_pixel(self, d_yaw: float, d_pitch: float) -> Tuple[int, int]:
        x_prop = (-d_yaw) / math.radians(self.hfov_deg)
        y_prop = ( d_pitch) / math.radians(self.vfov_deg)
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
    scan_dir: int = 1
    scan_step: float = 0.08
    seek_cooldown: int = 0
    seek_period: Tuple[int, int] = (8, 18)

    
    
    def set_mode(self, mode: Mode) -> None:
        self.mode = mode

    def set_pose(self, x: float, y: float, theta: float) -> None:
        self.robot_pose = (x, y, theta)

    def set_head(self, pitch: float, yaw: float) -> None:
        self.head_joint_states = (pitch, yaw)

    def apply_limits(self, pitch: float, yaw: float) -> Tuple[float, float]:
        pmin, pmax = self.pitch_limits
        ymin, ymax = self.yaw_limits
        return (max(pmin, min(pitch, pmax)),
                max(ymin, min(yaw,   ymax)))

    def nudge(self, target_pitch: float, target_yaw: float) -> None:
        cp, cy = self.commanded_head
        np_ = cp + self.kp_head * (target_pitch - cp)
        ny_ = cy + self.kp_head * (target_yaw   - cy)
        self.commanded_head = self.apply_limits(np_, ny_)

    def location_attention(self, px: float, py: float, pz: float) -> int:
        xy = math.hypot(px, py)
        if xy == 0.0 and pz == 0.0: return 0
        target_yaw   = math.atan2(py, px)
        target_pitch = math.atan2(-pz, max(1e-6, xy))
        target_pitch, target_yaw = self.apply_limits(target_pitch, target_yaw)
        self.nudge(target_pitch, target_yaw)
        return 1

    def social_attention(self, realignment_threshold_deg: int, social_control: str) -> int:
        deadband = math.radians(realignment_threshold_deg)
        
        if self.face_detected and self.face_within_range:
            tp, ty = 0.0, 0.0
        
        elif self.sound_detected:
            tp, ty = 0.0, clamp(self.sound_angle, *self.yaw_limits)
        
        else:
            return 0
        
        cp, cy = self.commanded_head
        
        if abs(tp - cp) < deadband and abs(ty - cy) < deadband:
            return 0
        
        self.nudge(tp, ty)
        return 1

    def scanning_attention(self, center_yaw: float, center_pitch: float) -> int:
        center_yaw   = clamp(center_yaw,   *self.yaw_limits)
        center_pitch = clamp(center_pitch, *self.pitch_limits)
        
        next_y = self.commanded_head[1] + self.scan_dir * self.scan_step
        if next_y >= self.yaw_limits[1] or next_y <= self.yaw_limits[0]:
            self.scan_dir *= -1
            next_y = clamp(next_y, *self.yaw_limits)
        self.nudge(center_pitch, next_y)
        return 1

    def seeking_attention(self, realignment_threshold_deg: int) -> int:
        if self.seek_cooldown <= 0:
            ty = self.rng.uniform(*self.yaw_limits)
            tp = self.rng.uniform(*self.pitch_limits)
            self.pending = (tp, ty)
            self.seek_cooldown = self.rng.randint(*self.seek_period)
        else:
            self.seek_cooldown -= 1
        tp, ty = getattr(self, "pending", self.commanded_head)
        cp, cy = self.commanded_head
        deadband = math.radians(realignment_threshold_deg)
        if abs(tp - cp) < deadband and abs(ty - cy) < deadband:
            return 0
        self.nudge(tp, ty)
        return 1

# -----------------------------------------------------------------------------
# Saliency
# -----------------------------------------------------------------------------
class SaliencyProcessor:
    def __init__(self, cam: CameraSpec, verbose: bool = False):
        self.cam = cam
        self.verbose = verbose
        self.faces_map: Optional[np.ndarray] = None        # H×W float mask (0/1 or weights)
        self.previous_locations: List[Tuple[float, float, int]] = []  # (abs_yaw, abs_pitch, t)

    # ---- Base saliency (fine-grained if available; fallback Sobel magnitude) ----
    def base_saliency(self, image_bgr: np.ndarray) -> np.ndarray:
        try:
            sal = cv2.saliency.StaticSaliencyFineGrained_create()
            ok, salmap = sal.computeSaliency(image_bgr)
            if ok:
                return salmap.astype(np.float32)   # OpenCV returns float32 in [0,1]
        except Exception:
            pass
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return mag.astype(np.float32)  # keep raw; no normalization (C++ parity)

 
    def winner_takes_all(self, saliency_map: np.ndarray) -> Tuple[int, int]:
        _, _, _, max_loc = cv2.minMaxLoc(saliency_map)
        return int(max_loc[0]), int(max_loc[1])

    def habituation(self,
                    saliency_map: np.ndarray,
                    wta_map: np.ndarray,
                    prev: List[Tuple[float, float, int]],
                    head_pitch: float,
                    head_yaw: float) -> List[Tuple[float, float, int]]:
        out: List[Tuple[float, float, int]] = []
        for abs_yaw, abs_pitch, t in prev:
            d_yaw   = abs_yaw   - head_yaw
            d_pitch = abs_pitch - head_pitch
            x, y = self.cam.angle_to_pixel(d_yaw, d_pitch)
            if 0 <= x < self.cam.width and 0 <= y < self.cam.height:
                x0 = max(0, x - PATCH_RADIUS); x1 = min(self.cam.width,  x + PATCH_RADIUS + 1)
                y0 = max(0, y - PATCH_RADIUS); y1 = min(self.cam.height, y + PATCH_RADIUS + 1)
                saliency_map[y0:y1, x0:x1] = saliency_map[y0:y1, x0:x1] - (t * HABITUATION_RATE)
                cv2.circle(wta_map, (x, y), PATCH_RADIUS, int(t * PATCH_RADIUS), -1)  # gray blob
            out.append((abs_yaw, abs_pitch, t + 1))
        return out

    def inhibition_of_return(self,
                             saliency_map: np.ndarray,
                             wta_map: np.ndarray,
                             prev: List[Tuple[float, float, int]],
                             head_pitch: float,
                             head_yaw: float) -> List[Tuple[float, float, int]]:
        out: List[Tuple[float, float, int]] = []
        for abs_yaw, abs_pitch, t in prev:
            if IOR_LIMIT < t < IOR_LIMIT + 50:
                d_yaw   = abs_yaw   - head_yaw
                d_pitch = abs_pitch - head_pitch
                x, y = self.cam.angle_to_pixel(d_yaw, d_pitch)
                if 0 <= x < self.cam.width and 0 <= y < self.cam.height:
                    x0 = max(0, x - PATCH_RADIUS); x1 = min(self.cam.width,  x + PATCH_RADIUS + 1)
                    y0 = max(0, y - PATCH_RADIUS); y1 = min(self.cam.height, y + PATCH_RADIUS + 1)
                    saliency_map[y0:y1, x0:x1] = 0.0
                    cv2.circle(wta_map, (x, y), PATCH_RADIUS, 0, -1)
                # drop this entry
            else:
                out.append((abs_yaw, abs_pitch, t))
        return out

    def compute_saliency_features(self,
                                  camera_image: np.ndarray,
                                  head_pitch: float,
                                  head_yaw: float,
                                  debug: bool = False) -> Tuple[int, int, int, np.ndarray]:
        """
        Return (status, centre_x, centre_y, wta_map)
        status: 0 on success, -1 on failure (C++ parity)
        """
        wta = np.full_like(camera_image, 255, dtype=np.uint8)

        saliency_map = self.base_saliency(camera_image)
        if saliency_map is None or saliency_map.size == 0:
            print("Error computing saliency map")
            return -1, -1, -1, wta

        # Face boost: for any positive faces_map pixel, force saliency to 2.0 (C++ parity)
        if self.faces_map is not None:
            ys, xs = np.where(self.faces_map > 0)
            saliency_map[ys, xs] = 2.0

        if debug or self.verbose:
            cv2.imshow("Saliency Map", saliency_map); cv2.waitKey(1)

        # Apply habituation then IOR using current head pose
        self.previous_locations = self.habituation(saliency_map, wta, self.previous_locations, head_pitch, head_yaw)
        self.previous_locations = self.inhibition_of_return(saliency_map, wta, self.previous_locations, head_pitch, head_yaw)

        cx, cy = self.winner_takes_all(saliency_map)

        if debug or self.verbose:
            cv2.line(wta, (cx - 15, cy - 15), (cx + 15, cy + 15), (0, 255, 0), 5)
            cv2.line(wta, (cx - 15, cy + 15), (cx + 15, cy - 15), (0, 255, 0), 5)
            cv2.imshow("WTA", wta); cv2.waitKey(1)

        return 0, int(cx), int(cy), wta

# -----------------------------------------------------------------------------
# Main Node
# -----------------------------------------------------------------------------
class OvertAttentionSystem(Node):
    def __init__(self):
        super().__init__("overt_attention")
        
        self.pkg_share = get_package_share_directory("overt_attention")
        self.cfg_path  = os.path.join(self.pkg_share, "config", "overt_attention_configuration.yaml")
        self.bridge = CvBridge()
        self.camera_image: Optional[np.ndarray] = None

        self.cfg = self.load_yaml(self.cfg_path, CONFIG_DEFAULTS)
        self.state = AttentionState()
        self.verbose = bool(self.cfg.get("verbose_mode", False))
        
        # Start in SCANNING mode for immediate visualization
        self.state.set_mode(Mode.SCANNING)

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

        self.verbose = bool(self.cfg.get("verbose_mode", False))
        camera_key_raw = str(self.cfg["camera"])
        use_compr = bool(self.cfg["use_compressed_images"])

        # normalize for topics map
        if camera_key_raw.lower() == "realsense":
            camera_topic_key = "RealSenseCameraCompressed" if use_compr else "RealSenseCamera"
            cam_name = "realsense"
        elif camera_key_raw.lower() == "peppercamera":
            camera_topic_key = "FrontCamera"
            cam_name = "peppercamera"
        else:
            self.get_logger().error(f"Unsupported camera type: {camera_key_raw}")
            return False

        topics_path = os.path.join(self.pkg_share, "data", "pepper_topics.yaml")
        topics = self.load_yaml(topics_path, TOPIC_DEFAULTS)

        # pubs/subs
        self.image_subscription                 = self.create_subscription(Image, topics[camera_topic_key], self.camera_callback, 10)
        self.joint_state_subscription           = self.create_subscription(JointState, topics["JointStates"], self.joint_states_callback, 10)
        
        self.robot_pose_subscription            = self.create_subscription(Pose2D, topics["RobotPose"], self.robot_pose_callback, 10)
        self.sound_localization_subscription    = self.create_subscription(Float32, topics["SoundLocalization"], self.sound_callback, 10)
        self.face_detection_subscription        = self.create_subscription(FaceDetection, topics["FaceDetection"], self.face_callback, 10)

        self.cmd_vel_publisher                  = self.create_publisher(Twist, topics["Wheels"], 10)
        self.joint_angles_publisher             = self.create_publisher(JointAnglesWithSpeed, topics["JointAngles"], 10)
        self.overt_attention_publisher          = self.create_publisher(OvertAttentionStatus, topics["OvertAttentionStatus"], 10)
        self.debug_image_publisher              = self.create_publisher(Image, "/overt_attention/debug_image", 10)

        # services (only register if srv types exist)
        self.set_mode_service = self.create_service(OvertAttentionSetMode, topics["SetMode"], self.set_mode_callback)

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
        try:
            val = float(msg.data)
        except Exception:
            return
        if math.isnan(val):
            return
        self.state.sound_detected = True
        self.state.sound_angle = math.radians(val)  # assuming degrees on the topic
        self.sound_count += 1


    def face_callback(self, msg: FaceDetection):
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
                self.location_target = (request.location_x, request.location_y, request.location_z)
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

    # ------------------ Control loop hook ------------------
    def run_once(self):
        """
        Call this periodically from a timer. Reads inputs and updates state.commanded_head.
        You can publish a head command here.
        """
        if self.state.mode == Mode.LOCATION:
            px, py, pz = getattr(self, "location_target", (0.0, 0.0, 0.0))
            self.state.location_attention(px, py, pz)

        elif self.state.mode == Mode.SOCIAL:
            # choose random/saliency/etc. For now, prefer sound if fresh:
            self.state.social_attention(realignment_threshold_deg=self.cfg["realignment_threshold"], 
                                        social_control=self.cfg["social_attention_mode"])

        elif self.state.mode == Mode.SCANNING:
            self.state.yaw_limits   = SCANNING_LIMITS["yaw"]
            self.state.pitch_limits = SCANNING_LIMITS["pitch"]

            if self.camera_image is not None:
                hp, hy = self.state.head_joint_states
                status, cx, cy, _ = self.saliency.compute_saliency_features(
                    self.camera_image, head_pitch=hp, head_yaw=hy, debug=self.verbose
                )
                if status == 0:
                    d = self.cam_spec.pixel_to_angle(cx, cy)  # radians
                    control_pitch = d.d_pitch + hp
                    control_yaw   = d.d_yaw   + hy
                    self.saliency.previous_locations.append((control_yaw, control_pitch, 1))
                    self.state.scanning_attention(center_yaw=control_yaw, center_pitch=control_pitch)
                else:
                    self.state.scanning_attention(center_yaw=0.0, center_pitch=DEFAULT_HEAD_PITCH)
            else:
                self.state.scanning_attention(center_yaw=0.0, center_pitch=DEFAULT_HEAD_PITCH)

        
        elif self.state.mode == Mode.SEEKING:
            self.state.yaw_limits   = HEAD_LIMITS["yaw"]
            self.state.pitch_limits = HEAD_LIMITS["pitch"]
            self.state.seeking_attention(realignment_threshold_deg=8)

        elif self.state.mode == Mode.DISABLED:
            self.state.nudge(DEFAULT_HEAD_PITCH, DEFAULT_HEAD_YAW)

        else:
            # ROS ERROR log
            self.get_logger().error(f"Unknown state: {self.state.mode}")

        status_msg = OvertAttentionStatus()
        # pick one mapping that matches your .msg type
        status_msg.state = self.state.mode.name.lower()   # or an int enum if your msg uses uint8
        status_msg.mutual_gaze = bool(self.state.mutual_gaze_detected)
        self.overt_attention_publisher.publish(status_msg)

        # Example: publish head angles using JointAnglesWithSpeed (Pepper/naoqi_bridge)
        pitch_cmd, yaw_cmd = self.state.commanded_head
        msg = JointAnglesWithSpeed()
        msg.joint_names = ["HeadPitch", "HeadYaw"]
        msg.joint_angles = [float(pitch_cmd), float(yaw_cmd)]
        msg.speed = 0.2
        msg.relative = False
        self.joint_angles_publisher.publish(msg)

        # Publish debug visualization if camera image is available
        if self.camera_image is not None and self.verbose:
            self.publish_debug_visualization()

    def publish_debug_visualization(self):
        """Create visualization matching exactly the C++ implementation"""
        if self.camera_image is None:
            return
            
        # Show the original camera image (matching C++ "Image" window)
        cv2.imshow("Image", self.camera_image)
        cv2.waitKey(1)
        
        # Publish debug image as ROS topic for external tools
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(self.camera_image, encoding="bgr8")
            self.debug_image_publisher.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")
