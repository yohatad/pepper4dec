#!/usr/bin/env python3
"""
Overt Attention System Implementation
Core implementation for robot head attention control system.
"""

import math
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

# ROS 2 messages / services
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from cssr_interfaces.srv import OvertAttentionSetMode    
from cssr_interfaces.msg import FaceDetection, OvertAttentionStatus


class AttentionMode(Enum):
    """Robot attention modes"""
    DISABLED = "disabled"
    SOCIAL = "social"
    SCANNING = "scanning"
    SEEKING = "seeking"
    LOCATION = "location"

    @classmethod
    def from_string(cls, mode_str: str) -> 'AttentionMode':
        """Convert string to AttentionMode, case-insensitive"""
        for mode in cls:
            if mode.value.lower() == mode_str.lower():
                return mode
        raise ValueError(f"Invalid attention mode: {mode_str}")

@dataclass
class RobotLimits:
    """Robot joint limits and constraints"""
    HEAD_YAW_RANGE: Tuple[float, float] = (-2.0857, 2.0857)
    HEAD_PITCH_RANGE: Tuple[float, float] = (-0.7068, 0.6371)
    SCANNING_YAW_RANGE: Tuple[float, float] = (-0.58353, 0.58353)
    SCANNING_PITCH_RANGE: Tuple[float, float] = (-0.3, 0.0)
    
    DEFAULT_HEAD_PITCH: float = -0.2
    DEFAULT_HEAD_YAW: float = 0.0

@dataclass
class SaliencyConfig:
    """Saliency computation parameters"""
    PATCH_RADIUS: int = 15
    HABITUATION_RATE: float = 0.1
    IOR_LIMIT: int = 50  # iterations
    FACE_BOOST_VALUE: float = 2.0

@dataclass
class CameraSpec:
    """Camera specifications and pixel-to-angle conversion"""
    hfov_deg: float
    vfov_deg: float
    width: int
    height: int

    @classmethod
    def create_realsense(cls) -> 'CameraSpec':
        return cls(hfov_deg=69.50, vfov_deg=42.50, width=640, height=480)
    
    @classmethod
    def create_pepper(cls) -> 'CameraSpec':
        return cls(hfov_deg=55.20, vfov_deg=44.30, width=640, height=480)

    def pixel_to_angle(self, x: float, y: float) -> Tuple[float, float]:
        """Convert pixel coordinates to angle changes (d_yaw, d_pitch)"""
        cx, cy = self.width / 2.0, self.height / 2.0
        x_prop = (x - cx) / self.width
        y_prop = (y - cy) / self.height
        d_yaw = x_prop * math.radians(self.hfov_deg) * -1.0
        d_pitch = y_prop * math.radians(self.vfov_deg)
        return d_yaw, d_pitch

    def angle_to_pixel(self, d_yaw: float, d_pitch: float) -> Tuple[int, int]:
        """Convert angle changes to pixel coordinates"""
        x_prop = (-d_yaw) / math.radians(self.hfov_deg)
        y_prop = (d_pitch) / math.radians(self.vfov_deg)
        x = int(self.width / 2.0 + x_prop * self.width)
        y = int(self.height / 2.0 + y_prop * self.height)
        return x, y


class ConfigManager:
    """Handles configuration loading and validation"""
    
    DEFAULT_CONFIG = {
        "camera": "RealSenseCamera",
        "use_compressed_images": False,
        "verbose_mode": False,
    }
    
    DEFAULT_TOPICS = {
        "RealSenseCamera": "/camera/color/image_raw",
        "RealSenseCameraCompressed": "/camera/color/image_raw/compressed",
        "FrontCamera": "/naoqi_driver/camera/front/image_raw",
        "JointAngles": "/joint_angles",
        "JointStates": "/joint_states",
        "FaceDetection": "/faceDetection/data",
        "SoundLocalization": "/soundDetection/direction",
        "OvertAttentionStatus": "/overtAttention/mode",
        "SetMode": "/overtAttention/set_mode",
    }

    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        self.config = self.load_config()
        self.topics = self.load_topics()
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file"""
        config_path = self.package_path / "config" / "overt_attention_configuration.yaml"
        return self.load_yaml_with_defaults(config_path, self.DEFAULT_CONFIG)
    
    def load_topics(self) -> Dict:
        """Load topic mappings from YAML file"""
        topics_path = self.package_path / "data" / "pepper_topics.yaml"
        return self.load_yaml_with_defaults(topics_path, self.DEFAULT_TOPICS)
    
    def load_yaml_with_defaults(self, path: Path, defaults: Dict) -> Dict:
        """Load YAML file with fallback to defaults"""
        data = defaults.copy()
        try:
            if path.exists():
                with open(path, "r") as f:
                    override = yaml.safe_load(f) or {}
                data.update(override)
        except Exception as e:
            print(f"Warning: Could not load YAML {path}: {e}")
        return data

    def get_camera_topic(self) -> str:
        """Get the appropriate camera topic based on configuration"""
        camera_key = self.config["camera"].lower()
        use_compressed = self.config["use_compressed_images"]
        
        if camera_key == "realsense":
            return self.topics["RealSenseCameraCompressed" if use_compressed else "RealSenseCamera"]
        elif camera_key == "peppercamera":
            return self.topics["FrontCamera"]
        else:
            raise ValueError(f"Unsupported camera type: {camera_key}")

    def get_camera_spec(self) -> CameraSpec:
        """Get camera specifications based on configuration"""
        camera_key = self.config["camera"].lower()
        if camera_key == "realsense":
            return CameraSpec.create_realsense()
        elif camera_key == "peppercamera":
            return CameraSpec.create_pepper()
        else:
            raise ValueError(f"Unsupported camera type: {camera_key}")


class HeadController:
    """Manages robot head positioning and control"""
    
    def __init__(self, limits: RobotLimits, kp: float = 0.5):
        self.limits = limits
        self.kp = kp
        self.current_pitch = limits.DEFAULT_HEAD_PITCH
        self.current_yaw = limits.DEFAULT_HEAD_YAW
        self.commanded_pitch = limits.DEFAULT_HEAD_PITCH
        self.commanded_yaw = limits.DEFAULT_HEAD_YAW
        
        # Mode-specific limits
        self.active_pitch_range = limits.HEAD_PITCH_RANGE
        self.active_yaw_range = limits.HEAD_YAW_RANGE

    def set_current_position(self, pitch: float, yaw: float):
        """Update current head position from joint states"""
        self.current_pitch = pitch
        self.current_yaw = yaw

    def set_limits_for_mode(self, mode: AttentionMode):
        """Set joint limits based on attention mode"""
        if mode == AttentionMode.SCANNING:
            self.active_pitch_range = self.limits.SCANNING_PITCH_RANGE
            self.active_yaw_range = self.limits.SCANNING_YAW_RANGE
        else:
            self.active_pitch_range = self.limits.HEAD_PITCH_RANGE
            self.active_yaw_range = self.limits.HEAD_YAW_RANGE

    def clamp_to_limits(self, pitch: float, yaw: float) -> Tuple[float, float]:
        """Clamp joint angles to current limits"""
        pitch = max(self.active_pitch_range[0], min(pitch, self.active_pitch_range[1]))
        yaw = max(self.active_yaw_range[0], min(yaw, self.active_yaw_range[1]))
        return pitch, yaw

    def move_towards(self, target_pitch: float, target_yaw: float):
        """Move head towards target using proportional control"""
        # Apply proportional control
        new_pitch = self.commanded_pitch + self.kp * (target_pitch - self.commanded_pitch)
        new_yaw = self.commanded_yaw + self.kp * (target_yaw - self.commanded_yaw)
        
        # Apply limits
        self.commanded_pitch, self.commanded_yaw = self.clamp_to_limits(new_pitch, new_yaw)

    def get_command(self) -> Tuple[float, float]:
        """Get current commanded position"""
        return self.commanded_pitch, self.commanded_yaw


class SaliencyProcessor:
    """Handles saliency computation with habituation and inhibition of return"""
    
    def __init__(self, camera_spec: CameraSpec, config: SaliencyConfig):
        self.camera_spec = camera_spec
        self.config = config
        self.faces_map: Optional[np.ndarray] = None
        self.previous_locations: List[Tuple[float, float, int]] = []

    def compute_base_saliency(self, image_bgr: np.ndarray) -> np.ndarray:
        """Compute base saliency map using OpenCV or fallback to Sobel"""
        try:
            sal = cv2.saliency.StaticSaliencyFineGrained_create()
            ok, salmap = sal.computeSaliency(image_bgr)
            if ok:
                return salmap.astype(np.float32)
        except Exception:
            pass
        
        # Fallback to Sobel magnitude
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy).astype(np.float32)

    def apply_face_boost(self, saliency_map: np.ndarray):
        """Boost saliency at face locations"""
        if self.faces_map is not None:
            mask = self.faces_map > 0
            saliency_map[mask] = self.config.FACE_BOOST_VALUE

    def apply_habituation_and_ior(self, saliency_map: np.ndarray, wta_map: Optional[np.ndarray], head_pitch: float, head_yaw: float, debug: bool = False):
        """Apply habituation and inhibition of return"""
        updated_locations = []
        
        for abs_yaw, abs_pitch, age in self.previous_locations:
            # Convert absolute angles to relative pixel coordinates
            d_yaw = abs_yaw - head_yaw
            d_pitch = abs_pitch - head_pitch
            x, y = self.camera_spec.angle_to_pixel(d_yaw, d_pitch)
            
            if 0 <= x < self.camera_spec.width and 0 <= y < self.camera_spec.height:
                # Apply habituation (gradual reduction)
                if age <= self.config.IOR_LIMIT:
                    self.apply_patch_modification(saliency_map, x, y, -age * self.config.HABITUATION_RATE)
                    
                    # Visualize habituation areas in gray
                    if debug and wta_map is not None:
                        intensity = int(255 * (1.0 - age / self.config.IOR_LIMIT))
                        cv2.circle(wta_map, (x, y), self.config.PATCH_RADIUS, (intensity, intensity, intensity), -1)
                
                # Apply inhibition of return (complete suppression)
                elif self.config.IOR_LIMIT < age < self.config.IOR_LIMIT + 50:
                    self.apply_patch_modification(saliency_map, x, y, -saliency_map.max())
                    
                    # Visualize IOR areas in red
                    if debug and wta_map is not None:
                        cv2.circle(wta_map, (x, y), self.config.PATCH_RADIUS, (0, 0, 255), -1)
                    continue  # Don't keep this location
            
            updated_locations.append((abs_yaw, abs_pitch, age + 1))
        
        self.previous_locations = updated_locations

    def apply_patch_modification(self, saliency_map: np.ndarray, x: int, y: int, delta: float):
        """Apply modification to a circular patch around (x, y)"""
        r = self.config.PATCH_RADIUS
        x0, x1 = max(0, x - r), min(self.camera_spec.width, x + r + 1)
        y0, y1 = max(0, y - r), min(self.camera_spec.height, y + r + 1)
        saliency_map[y0:y1, x0:x1] = np.maximum(0, saliency_map[y0:y1, x0:x1] + delta)

    def compute_attention_point(self, image: np.ndarray, head_pitch: float, head_yaw: float, debug: bool = False) -> Tuple[int, int, Optional[np.ndarray]]:
        """Compute the most salient point for attention"""
        # Create visualization image (copy of original for debugging)
        wta_map = image.copy() if debug else None
        
        saliency_map = self.compute_base_saliency(image)
        
        # Show base saliency if debugging
        if debug:
            cv2.imshow("Base Saliency Map", saliency_map)
            cv2.waitKey(1)
        
        self.apply_face_boost(saliency_map)
        
        # Show face-boosted saliency if debugging
        if debug and self.faces_map is not None:
            face_boosted = saliency_map.copy()
            face_boosted[self.faces_map > 0] = 1.0  # Highlight face areas
            cv2.imshow("Face Boosted Saliency", face_boosted)
            cv2.waitKey(1)
        
        self.apply_habituation_and_ior(saliency_map, wta_map, head_pitch, head_yaw, debug)
        
        # Winner-takes-all
        _, _, _, max_loc = cv2.minMaxLoc(saliency_map)
        cx, cy = int(max_loc[0]), int(max_loc[1])
        
        # Draw attention point on visualization
        if debug and wta_map is not None:
            # Draw green cross at attention point
            cv2.line(wta_map, (cx - 15, cy - 15), (cx + 15, cy + 15), (0, 255, 0), 3)
            cv2.line(wta_map, (cx - 15, cy + 15), (cx + 15, cy - 15), (0, 255, 0), 3)
            
            # Show final result
            cv2.imshow("Winner Takes All", wta_map)
            cv2.waitKey(1)
        
        return cx, cy, wta_map

    def update_faces(self, faces_map: np.ndarray):
        """Update face locations for saliency boosting"""
        self.faces_map = faces_map

    def add_attention_location(self, abs_yaw: float, abs_pitch: float):
        """Add a new attention location to track"""
        self.previous_locations.append((abs_yaw, abs_pitch, 1))


class AttentionBehaviors:
    """Implements different attention behaviors"""
    
    def __init__(self, head_controller: HeadController, limits: RobotLimits):
        self.head_controller = head_controller
        self.limits = limits
        self.rng = random.Random()
        
        # Scanning state
        self.scan_direction = 1
        self.scan_step = 0.08
        
        # Seeking state
        self.seek_cooldown = 0
        self.seek_period = (8, 18)
        self.seek_target: Optional[Tuple[float, float]] = None

    def location_attention(self, target_x: float, target_y: float, target_z: float):
        """Point head towards a 3D location"""
        distance_xy = math.hypot(target_x, target_y)
        if distance_xy == 0.0 and target_z == 0.0:
            return
        
        target_yaw = math.atan2(target_y, target_x)
        target_pitch = math.atan2(-target_z, max(1e-6, distance_xy))
        
        self.head_controller.move_towards(target_pitch, target_yaw)

    def social_attention(self, face_detected: bool, face_in_range: bool, sound_detected: bool, sound_angle: float):
        """Social attention behavior - prioritize faces and sounds"""
        if face_detected and face_in_range:
            # Look at center when face is detected and in range
            self.head_controller.move_towards(0.0, 0.0)
        elif sound_detected:
            # Look towards sound source
            sound_yaw = max(self.head_controller.active_yaw_range[0], 
                          min(sound_angle, self.head_controller.active_yaw_range[1]))
            self.head_controller.move_towards(0.0, sound_yaw)

    def scanning_attention(self, saliency_center_yaw: float = 0.0, saliency_center_pitch: float = None):
        """Scanning behavior with saliency-guided center"""
        if saliency_center_pitch is None:
            saliency_center_pitch = self.limits.DEFAULT_HEAD_PITCH
        
        # Scan left-right around the saliency center
        current_yaw = self.head_controller.commanded_yaw
        next_yaw = current_yaw + self.scan_direction * self.scan_step
        
        # Reverse direction at limits
        if next_yaw >= self.head_controller.active_yaw_range[1] or next_yaw <= self.head_controller.active_yaw_range[0]:
            self.scan_direction *= -1
            next_yaw = max(self.head_controller.active_yaw_range[0], 
                          min(next_yaw, self.head_controller.active_yaw_range[1]))
        
        self.head_controller.move_towards(saliency_center_pitch, next_yaw)

    def seeking_attention(self):
        """Random seeking behavior"""
        if self.seek_cooldown <= 0:
            # Generate new random target
            target_yaw = self.rng.uniform(*self.head_controller.active_yaw_range)
            target_pitch = self.rng.uniform(*self.head_controller.active_pitch_range)
            self.seek_target = (target_pitch, target_yaw)
            self.seek_cooldown = self.rng.randint(*self.seek_period)
        else:
            self.seek_cooldown -= 1
        
        if self.seek_target:
            target_pitch, target_yaw = self.seek_target
            self.head_controller.move_towards(target_pitch, target_yaw)

    def disabled_attention(self):
        """Return to default position"""
        self.head_controller.move_towards(self.limits.DEFAULT_HEAD_PITCH, self.limits.DEFAULT_HEAD_YAW)


class OvertAttentionSystem(Node):
    """Main ROS2 node for overt attention system"""
    
    def __init__(self):
        super().__init__("overt_attention")
        
        # Initialize components
        self.bridge = CvBridge()
        package_path = get_package_share_directory("overt_attention")
        self.config_manager = ConfigManager(package_path)
        self.limits = RobotLimits()
        self.head_controller = HeadController(self.limits)
        
        camera_spec = self.config_manager.get_camera_spec()
        saliency_config = SaliencyConfig()
        self.saliency_processor = SaliencyProcessor(camera_spec, saliency_config)
        self.behaviors = AttentionBehaviors(self.head_controller, self.limits)
        
        # State
        self.current_mode = AttentionMode.SCANNING  # Start in scanning mode
        self.camera_image: Optional[np.ndarray] = None
        self.wta_visualization: Optional[np.ndarray] = None  # For debug visualization
        self.location_target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        
        # Sensor inputs
        self.face_detected = False
        self.face_in_range = False
        self.mutual_gaze = False
        self.sound_detected = False
        self.sound_angle = 0.0
        
        self.setup_ros_interfaces()
        self.get_logger().info("OvertAttentionSystem initialized")

    def setup_ros_interfaces(self):
        """Initialize ROS publishers, subscribers, and services"""
        topics = self.config_manager.topics
        
        # Subscribers
        camera_topic = self.config_manager.get_camera_topic()
        self.create_subscription(Image, camera_topic, self.camera_callback, 10)
        self.create_subscription(JointState, topics["JointStates"], self.joint_states_callback, 10)
        self.create_subscription(Float32, topics["SoundLocalization"], self.sound_callback, 10)
        self.create_subscription(FaceDetection, topics["FaceDetection"], self.face_callback, 10)
        
        # Publishers
        self.joint_angles_pub = self.create_publisher(JointAnglesWithSpeed, topics["JointAngles"], 10)
        self.status_pub = self.create_publisher(OvertAttentionStatus, topics["OvertAttentionStatus"], 10)
        self.debug_image_pub = self.create_publisher(Image, "/overt_attention/debug_image", 10)
        
        # Services
        self.create_service(OvertAttentionSetMode, topics["SetMode"], self.set_mode_callback)

    def camera_callback(self, msg: Image):
        """Handle camera image updates"""
        try:
            self.camera_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Camera conversion failed: {e}")

    def joint_states_callback(self, msg: JointState):
        """Handle joint state updates"""
        try:
            pitch_idx = msg.name.index("HeadPitch")
            yaw_idx = msg.name.index("HeadYaw")
            pitch = msg.position[pitch_idx]
            yaw = msg.position[yaw_idx]
            self.head_controller.set_current_position(pitch, yaw)
        except (ValueError, IndexError) as e:
            self.get_logger().warning(f"Joint state parsing error: {e}")

    def sound_callback(self, msg: Float32):
        """Handle sound detection updates"""
        try:
            angle_deg = float(msg.data)
            if not math.isnan(angle_deg):
                self.sound_detected = True
                self.sound_angle = math.radians(angle_deg)
        except Exception:
            pass

    def face_callback(self, msg: FaceDetection):
        """Handle face detection updates"""
        if not hasattr(msg, "centroids"):
            return
        
        self.face_detected = len(msg.centroids) > 0
        self.face_in_range = False
        self.mutual_gaze = False
        
        # Create face saliency map
        camera_spec = self.config_manager.get_camera_spec()
        faces_map = np.zeros((camera_spec.height, camera_spec.width), dtype=np.float32)
        
        for i, centroid in enumerate(msg.centroids):
            x, y = int(centroid.x), int(centroid.y)
            
            if 0 <= x < camera_spec.width and 0 <= y < camera_spec.height:
                # Check distance for in-range determination
                distance = getattr(centroid, "z", 10.0)
                if distance <= 2.0:
                    self.face_in_range = True
                
                # Check mutual gaze
                if hasattr(msg, "mutual_gaze") and i < len(msg.mutual_gaze):
                    self.mutual_gaze = self.mutual_gaze or bool(msg.mutual_gaze[i])
                
                # Add to saliency map
                scale = 1.0 / max(1e-3, distance)
                faces_map[y, x] = max(faces_map[y, x], scale)
        
        self.saliency_processor.update_faces(faces_map)

    def set_mode_callback(self, request, response):
        """Handle mode change requests"""
        try:
            new_mode = AttentionMode.from_string(request.state)
            self.current_mode = new_mode
            self.head_controller.set_limits_for_mode(new_mode)
            
            if new_mode == AttentionMode.LOCATION:
                self.location_target = (request.location_x, request.location_y, request.location_z)
            
            if hasattr(response, "mode_set_success"):
                response.mode_set_success = 1
                
            self.get_logger().info(f"Mode changed to: {new_mode.value}")
            
        except ValueError as e:
            self.get_logger().error(f"Invalid mode request: {e}")
            if hasattr(response, "mode_set_success"):
                response.mode_set_success = 0
        
        return response

    def run_once(self):
        """Main control loop - called by timer from application"""
        # Execute behavior based on current mode
        if self.current_mode == AttentionMode.LOCATION:
            self.behaviors.location_attention(*self.location_target)
            
        elif self.current_mode == AttentionMode.SOCIAL:
            self.behaviors.social_attention(
                self.face_detected, self.face_in_range, 
                self.sound_detected, self.sound_angle
            )
            
        elif self.current_mode == AttentionMode.SCANNING:
            if self.camera_image is not None:
                # Use saliency to guide scanning center
                head_pitch, head_yaw = self.head_controller.current_pitch, self.head_controller.current_yaw
                verbose = self.config_manager.config.get("verbose_mode", False)
                
                cx, cy, wta_visualization = self.saliency_processor.compute_attention_point(
                    self.camera_image, head_pitch, head_yaw, debug=verbose
                )
                
                # Store visualization for debug publishing
                if verbose and wta_visualization is not None:
                    self.wta_visualization = wta_visualization
                
                # Convert to angle and add to tracking
                camera_spec = self.config_manager.get_camera_spec()
                d_yaw, d_pitch = camera_spec.pixel_to_angle(cx, cy)
                abs_yaw = head_yaw + d_yaw
                abs_pitch = head_pitch + d_pitch
                self.saliency_processor.add_attention_location(abs_yaw, abs_pitch)
                
                self.behaviors.scanning_attention(abs_yaw, abs_pitch)
            else:
                self.behaviors.scanning_attention()
                
        elif self.current_mode == AttentionMode.SEEKING:
            self.behaviors.seeking_attention()
            
        elif self.current_mode == AttentionMode.DISABLED:
            self.behaviors.disabled_attention()
        
        # Reset transient states
        self.sound_detected = False
        
        # Publish commands and status
        self.publish_head_command()
        self.publish_status()
        
        # Debug visualization
        if self.config_manager.config.get("verbose_mode", False):
            self.publish_debug_visualization()

    def publish_head_command(self):
        """Publish head joint commands"""
        pitch, yaw = self.head_controller.get_command()
        
        msg = JointAnglesWithSpeed()
        msg.joint_names = ["HeadPitch", "HeadYaw"]
        msg.joint_angles = [float(pitch), float(yaw)]
        msg.speed = 0.2
        msg.relative = False
        
        self.joint_angles_pub.publish(msg)

    def publish_status(self):
        """Publish system status"""
        msg = OvertAttentionStatus()
        msg.state = self.current_mode.value
        msg.mutual_gaze = self.mutual_gaze
        self.status_pub.publish(msg)

    def publish_debug_visualization(self):
        """Publish debug visualization with rich saliency information"""
        if self.camera_image is None:
            return
            
        try:
            # Show original camera image
            cv2.imshow("Camera Image", self.camera_image)
            
            # If we have WTA visualization from scanning mode, show it
            if hasattr(self, 'wta_visualization') and self.wta_visualization is not None:
                # Publish the WTA visualization as debug image
                debug_msg = self.bridge.cv2_to_imgmsg(self.wta_visualization, encoding="bgr8")
                self.debug_image_pub.publish(debug_msg)
            else:
                # Publish original camera image if no WTA available
                debug_msg = self.bridge.cv2_to_imgmsg(self.camera_image, encoding="bgr8")
                self.debug_image_pub.publish(debug_msg)
                
            # Show current head target as overlay on camera image
            self.show_head_target_overlay()
            
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Debug visualization failed: {e}")

    def show_head_target_overlay(self):
        """Show current head target as overlay on camera image"""
        if self.camera_image is None:
            return
            
        overlay = self.camera_image.copy()
        
        # Get current head position and target
        current_pitch, current_yaw = self.head_controller.current_pitch, self.head_controller.current_yaw
        target_pitch, target_yaw = self.head_controller.get_command()
        
        # Convert current and target positions to pixel coordinates (relative to current head pose)
        camera_spec = self.config_manager.get_camera_spec()
        
        # Show center crosshair (current head direction)
        center_x, center_y = self.camera_image.shape[1] // 2, self.camera_image.shape[0] // 2
        cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 2)
        cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 2)
        
        # Show target direction if different from current
        d_pitch = target_pitch - current_pitch
        d_yaw = target_yaw - current_yaw
        
        if abs(d_pitch) > 0.01 or abs(d_yaw) > 0.01:  # Only show if there's significant difference
            target_x, target_y = camera_spec.angle_to_pixel(d_yaw, d_pitch)
            if 0 <= target_x < self.camera_image.shape[1] and 0 <= target_y < self.camera_image.shape[0]:
                cv2.circle(overlay, (target_x, target_y), 10, (0, 255, 255), 2)  # Yellow circle for target
                cv2.line(overlay, (center_x, center_y), (target_x, target_y), (0, 255, 255), 1)  # Yellow line to target
        
        # Add mode information
        mode_text = f"Mode: {self.current_mode.value.upper()}"
        cv2.putText(overlay, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add head position information
        head_info = f"Head: P:{current_pitch:.2f} Y:{current_yaw:.2f}"
        cv2.putText(overlay, head_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add face/sound status
        status_info = f"Face:{self.face_detected} Sound:{self.sound_detected}"
        cv2.putText(overlay, status_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Head Control Overlay", overlay)