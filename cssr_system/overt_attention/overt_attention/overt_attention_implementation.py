#!/usr/bin/env python3
"""
Complete Enhanced Overt Attention System Implementation
Core implementation for robot head attention control system with advanced social attention
and smooth movement filtering.
"""

import math
import yaml
import random
import time
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


class SocialControlMode(Enum):
    """Social attention control modes"""
    RANDOM = "random"
    SALIENCY = "saliency"

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
class FaceInfo:
    """Information about a detected face"""
    x: int                    # Pixel x coordinate
    y: int                    # Pixel y coordinate  
    distance: float           # Distance from robot
    confidence: float         # Detection confidence
    label: int               # Face ID/label
    mutual_gaze: bool = False # Whether mutual gaze detected
    last_seen: float = 0.0   # Timestamp when last seen

@dataclass
class SocialAttentionConfig:
    """Configuration for social attention behavior"""
    SOUND_COUNT_THRESHOLD: int = 20      # Switch to sound after N detections
    FACE_HABITUATION_RATE: float = 0.3  # How quickly faces become less interesting
    SOUND_BOOST_VALUE: float = 2.0      # Saliency boost for sound locations
    FACE_BOOST_BASE: float = 2.0        # Base saliency boost for faces
    DISTANCE_SCALING: bool = True       # Scale attention by face distance
    MAX_FACE_DISTANCE: float = 3.0      # Maximum distance to pay attention to faces
    MIN_FACE_DISTANCE: float = 0.5      # Minimum face distance (avoid division by zero)

@dataclass
class MovementConfig:
    """Configuration for smooth movement behavior"""
    MIN_MOVEMENT_THRESHOLD: float = 0.1    # Minimum angle change to trigger movement (radians ~5.7 degrees)
    STABILITY_TIME: float = 0.5             # Time target must be stable before moving (seconds)
    MOVEMENT_TIMEOUT: float = 2.0           # Maximum time between movements (seconds)
    MAX_MOVEMENT_SPEED: float = 0.3         # Maximum movement speed (rad/s)
    CONFIDENCE_THRESHOLD: float = 0.7       # Minimum confidence to trigger movement
    INTEREST_DECAY_TIME: float = 3.0        # Time after which interest in target decays

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
        "camera": "realsense",  # Options: "realsense", "peppercamera"
        "realignment_threshold": 50,
        "x_offset_to_head_yaw": 0,
        "y_offset_to_head_pitch": 0,
        "social_attention_mode": "random",
        "num_faces_social_att": 3,
        "engagement_timeout": 12.0,
        "use_sound": False,
        "use_compressed_images": False,
        "verbose_mode": True,

        "movement_filtering": {
            "min_movement_degrees": 5.0,
            "movement_cooldown": 1.0,
            "confidence_threshold": 0.8,
            "stability_time": 0.5,
            "focus_duration_min": 2.0,
            "focus_duration_max": 8.0
        }
    }
    
    DEFAULT_TOPICS = {
        "RealSenseCamera": "/camera/color/image_raw",
        "RealSenseCameraCompressed": "/camera/color/image_raw/compressed",
        "RealSenseCameraDepth": "/camera/depth/image_raw",
        "RealSenseCameraDepthCompressed": "/camera/depth/image_raw/compressed",
        
        "FrontCamera": "/naoqi_driver/camera/front/image_raw",
        "DepthCamera": "/naoqi_driver/camera/depth/image_raw",
        
        "JointAngles": "/joint_angles",
        "Wheels": "/cmd_vel",
        "JointStates": "/joint_states",
        
        "RobotPose": "/robotLocalization/pose",
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
                self._deep_update(data, override)
        except Exception as e:
            print(f"Warning: Could not load YAML {path}: {e}")
        return data
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update nested dictionaries"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

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

class SmoothedHeadController:
    """Head controller with movement filtering to avoid jerky behavior"""
    
    def __init__(self, base_controller, config: MovementConfig = None):
        self.base_controller = base_controller
        self.config = config or MovementConfig()
        
        # Movement state tracking
        self.current_target: Optional[Tuple[float, float]] = None
        self.target_first_seen: float = 0.0
        self.target_confidence: float = 0.0
        self.last_movement_time: float = 0.0
        self.is_moving: bool = False
        self.movement_start_time: float = 0.0
        
        # Target stability tracking
        self.stable_target: Optional[Tuple[float, float]] = None
        self.stable_since: float = 0.0
        
        # Interest tracking
        self.current_interest_level: float = 0.0
        self.interest_peak_time: float = 0.0
    
    def update_target(self, target_pitch: float, target_yaw: float, confidence: float = 1.0):
        """Update target with confidence and stability filtering"""
        current_time = time.time()
        new_target = (target_pitch, target_yaw)
        
        # Check if target is significantly different from current
        if self.current_target is None:
            self.current_target = new_target
            self.target_first_seen = current_time
            self.target_confidence = confidence
            self.stable_target = new_target
            self.stable_since = current_time
            return
        
        # Calculate distance from current target
        distance = self._calculate_distance(self.current_target, new_target)
        
        if distance < self.config.MIN_MOVEMENT_THRESHOLD:
            # Target is close to current - update stability
            if self._calculate_distance(self.stable_target or new_target, new_target) < 0.05:
                # Target is stable, update confidence
                self.target_confidence = max(self.target_confidence, confidence)
            else:
                # New stable target
                self.stable_target = new_target
                self.stable_since = current_time
        else:
            # Significant new target
            self.current_target = new_target
            self.target_first_seen = current_time
            self.target_confidence = confidence
            self.stable_target = new_target
            self.stable_since = current_time
    
    def should_move_to_target(self) -> bool:
        """Determine if we should move to the current target"""
        if not self.current_target or not self.stable_target:
            return False
        
        current_time = time.time()
        
        # Check confidence threshold
        if self.target_confidence < self.config.CONFIDENCE_THRESHOLD:
            return False
        
        # Check if we're already moving
        if self.is_moving:
            return False
        
        # Check minimum time between movements
        time_since_last_movement = current_time - self.last_movement_time
        if time_since_last_movement < self.config.MOVEMENT_TIMEOUT:
            return False
        
        # Check target stability
        stability_time = current_time - self.stable_since
        if stability_time < self.config.STABILITY_TIME:
            return False
        
        # Check if movement is significant enough
        current_pos = (self.base_controller.commanded_pitch, self.base_controller.commanded_yaw)
        distance = self._calculate_distance(current_pos, self.stable_target)
        
        return distance >= self.config.MIN_MOVEMENT_THRESHOLD
    
    def update_interest_level(self, new_interest: float):
        """Update interest level in current target"""
        current_time = time.time()
        
        if new_interest > self.current_interest_level:
            self.current_interest_level = new_interest
            self.interest_peak_time = current_time
        else:
            # Decay interest over time
            time_since_peak = current_time - self.interest_peak_time
            decay_factor = max(0.0, 1.0 - (time_since_peak / self.config.INTEREST_DECAY_TIME))
            self.current_interest_level *= decay_factor
    
    def execute_movement(self) -> bool:
        """Execute movement if conditions are met"""
        if not self.should_move_to_target():
            return False
        
        current_time = time.time()
        target_pitch, target_yaw = self.stable_target
        
        # Calculate movement speed based on distance and interest
        current_pos = (self.base_controller.commanded_pitch, self.base_controller.commanded_yaw)
        distance = self._calculate_distance(current_pos, self.stable_target)
        
        # Adjust speed based on interest level and distance
        speed_factor = min(1.0, self.current_interest_level + 0.3)  # Minimum 30% speed
        movement_speed = self.config.MAX_MOVEMENT_SPEED * speed_factor
        
        # Start smooth movement
        self.is_moving = True
        self.movement_start_time = current_time
        self.last_movement_time = current_time
        
        # Use gradual movement instead of direct movement
        self._start_gradual_movement(target_pitch, target_yaw, movement_speed)
        
        return True
    
    def _start_gradual_movement(self, target_pitch: float, target_yaw: float, speed: float):
        """Start gradual movement to target"""
        current_pitch = self.base_controller.commanded_pitch
        current_yaw = self.base_controller.commanded_yaw
        
        # Calculate step size based on speed
        pitch_diff = target_pitch - current_pitch
        yaw_diff = target_yaw - current_yaw
        total_diff = math.sqrt(pitch_diff**2 + yaw_diff**2)
        
        if total_diff > 0:
            # Calculate step size for smooth movement
            step_size = min(0.05, speed * 0.1)  # Small steps for smoothness
            step_pitch = current_pitch + (pitch_diff / total_diff) * step_size
            step_yaw = current_yaw + (yaw_diff / total_diff) * step_size
            
            self.base_controller.move_towards(step_pitch, step_yaw)
    
    def update_movement_state(self):
        """Update movement state - call this in main loop"""
        if not self.is_moving:
            return
        
        current_time = time.time()
        movement_duration = current_time - self.movement_start_time
        
        # Check if movement is complete
        if self.stable_target:
            current_pos = (self.base_controller.commanded_pitch, self.base_controller.commanded_yaw)
            distance_to_target = self._calculate_distance(current_pos, self.stable_target)
            
            if distance_to_target < 0.02 or movement_duration > 3.0:  # Close enough or timeout
                self.is_moving = False
    
    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_movement_status(self) -> dict:
        """Get current movement status for debugging"""
        return {
            'is_moving': self.is_moving,
            'has_stable_target': self.stable_target is not None,
            'target_confidence': self.target_confidence,
            'interest_level': self.current_interest_level,
            'time_since_last_movement': time.time() - self.last_movement_time,
            'stability_time': time.time() - self.stable_since if self.stable_target else 0
        }


class SelectiveAttentionManager:
    """Manages what deserves attention to avoid constant head movements"""
    
    def __init__(self):
        self.attention_history: List[Tuple[float, float, float]] = []  # (yaw, pitch, timestamp)
        self.current_focus: Optional[Tuple[str, float, float]] = None  # (type, yaw, pitch)
        self.focus_start_time: float = 0.0
        
        # Attention parameters
        self.min_focus_duration = 1.5  # Minimum time to focus on something
        self.max_focus_duration = 5.0  # Maximum time to focus without new stimulus
        self.attention_radius = 0.3    # Radius around current focus to ignore new targets
    
    def evaluate_attention_worthiness(self, target_type: str, yaw: float, pitch: float, 
                                    confidence: float, novelty: float) -> float:
        """
        Evaluate if a target is worth attention
        Returns attention score (0.0 to 1.0)
        """
        current_time = time.time()
        base_score = confidence * 0.5 + novelty * 0.5
        
        # Check if we're currently focused on something
        if self.current_focus:
            focus_type, focus_yaw, focus_pitch = self.current_focus
            focus_duration = current_time - self.focus_start_time
            
            # Calculate distance from current focus
            distance = math.sqrt((yaw - focus_yaw)**2 + (pitch - focus_pitch)**2)
            
            # Reduce score if too close to current focus
            if distance < self.attention_radius:
                base_score *= 0.3
            
            # Require higher score to break current focus
            if focus_duration < self.min_focus_duration:
                base_score *= 0.5  # Much harder to break early focus
            elif focus_duration < self.max_focus_duration:
                base_score *= 0.8  # Somewhat harder to break established focus
        
        # Boost score for certain target types
        type_multipliers = {
            'face_with_gaze': 1.5,
            'face_close': 1.3,
            'face_familiar': 1.2,
            'face_general': 1.0,
            'sound_speech': 1.4,
            'sound_general': 0.8,
            'visual_salient': 0.7,
            'motion': 0.9
        }
        
        multiplier = type_multipliers.get(target_type, 1.0)
        final_score = min(1.0, base_score * multiplier)
        
        return final_score
    
    def update_focus(self, target_type: str, yaw: float, pitch: float, attention_score: float):
        """Update current focus if attention score is high enough"""
        current_time = time.time()
        
        # Decision threshold - higher when already focused
        threshold = 0.6 if self.current_focus is None else 0.8
        
        if attention_score >= threshold:
            self.current_focus = (target_type, yaw, pitch)
            self.focus_start_time = current_time
            self.attention_history.append((yaw, pitch, current_time))
            return True
        
        return False
    
    def should_break_focus(self) -> bool:
        """Check if current focus should be broken due to timeout"""
        if not self.current_focus:
            return False
        
        focus_duration = time.time() - self.focus_start_time
        return focus_duration > self.max_focus_duration


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


class EnhancedSocialAttention:
    """Enhanced social attention with multi-modal fusion and smooth movement control"""
    
    def __init__(self, head_controller, camera_spec, config: Optional[SocialAttentionConfig] = None):
        self.head_controller = head_controller
        self.camera_spec = camera_spec
        self.config = config or SocialAttentionConfig()
        
        # Social control state
        self.control_mode = SocialControlMode.SALIENCY
        self.sound_count = 0
        self.last_seen_face_label = -1
        
        # Face tracking
        self.current_faces: List[FaceInfo] = []
        self.face_attention_history: Dict[int, float] = {}  # label -> last attention time
        
        # Random number generator for random mode
        self.rng = random.Random()
        
        # Saliency maps for fusion
        self.social_saliency_map: Optional[np.ndarray] = None
        
        # Add smooth movement control
        movement_config = MovementConfig()
        self.smooth_controller = SmoothedHeadController(head_controller, movement_config)
        self.attention_manager = SelectiveAttentionManager()
        
    def set_control_mode(self, mode: SocialControlMode):
        """Set the social attention control mode"""
        self.control_mode = mode
        
    def update_faces(self, faces: List[FaceInfo]):
        """Update current face detections"""
        self.current_faces = faces.copy()
        
        # Update last seen times
        current_time = time.time()
        for face in faces:
            face.last_seen = current_time
    
    def create_social_saliency_map(self, sound_detected: bool, sound_angle: float) -> np.ndarray:
        """Create combined saliency map from faces and sound"""
        # Initialize saliency map
        saliency_map = np.zeros((self.camera_spec.height, self.camera_spec.width), dtype=np.float32)
        
        # Add faces to saliency map
        for face in self.current_faces:
            if (0 <= face.x < self.camera_spec.width and 
                0 <= face.y < self.camera_spec.height and
                face.distance <= self.config.MAX_FACE_DISTANCE):
                
                # Calculate face saliency based on distance and other factors
                face_saliency = self.calculate_face_saliency(face)
                
                # Apply habituation if this face was recently attended to
                if face.label in self.face_attention_history:
                    time_since_attention = time.time() - self.face_attention_history[face.label]
                    habituation_factor = max(0.1, 1.0 - (self.config.FACE_HABITUATION_RATE * 
                                                        max(0, 5.0 - time_since_attention)))
                    face_saliency *= habituation_factor
                
                # Boost for mutual gaze
                if face.mutual_gaze:
                    face_saliency *= 1.5
                
                saliency_map[face.y, face.x] = max(saliency_map[face.y, face.x], face_saliency)
        
        # Add sound to saliency map if detected
        if sound_detected:
            # Convert sound angle to pixel coordinates (assuming sound at head level)
            sound_x, sound_y = self.sound_angle_to_pixel(sound_angle)
            if (0 <= sound_x < self.camera_spec.width and 
                0 <= sound_y < self.camera_spec.height):
                saliency_map[sound_y, sound_x] = max(saliency_map[sound_y, sound_x], 
                                                   self.config.SOUND_BOOST_VALUE)
        
        self.social_saliency_map = saliency_map
        return saliency_map
    
    def calculate_face_saliency(self, face: FaceInfo) -> float:
        """Calculate saliency value for a face based on multiple factors"""
        base_saliency = self.config.FACE_BOOST_BASE
        
        # Distance-based scaling (closer faces are more salient)
        if self.config.DISTANCE_SCALING and face.distance > 0:
            distance_factor = 1.0 / max(self.config.MIN_FACE_DISTANCE, face.distance)
            base_saliency *= min(3.0, distance_factor)  # Cap the boost
        
        # Confidence-based scaling
        base_saliency *= face.confidence
        
        return base_saliency
    
    def sound_angle_to_pixel(self, sound_angle: float) -> Tuple[int, int]:
        """Convert sound angle to pixel coordinates"""
        # Assume sound is at center height of image
        center_y = self.camera_spec.height // 2
        
        # Convert angle to pixel x coordinate
        # Assuming sound_angle is relative to current head yaw
        d_yaw = sound_angle
        d_pitch = 0.0  # Sound assumed at head level
        
        sound_x, sound_y = self.camera_spec.angle_to_pixel(d_yaw, d_pitch)
        return sound_x, sound_y
    
    def select_face_randomly(self) -> Optional[FaceInfo]:
        """Select a face randomly, avoiding the last seen face if possible"""
        if not self.current_faces:
            return None
        
        if len(self.current_faces) == 1:
            self.last_seen_face_label = -1  # Reset if only one face
            return self.current_faces[0]
        
        # Try to select a different face from the last one
        available_faces = [f for f in self.current_faces if f.label != self.last_seen_face_label]
        
        if available_faces:
            selected_face = self.rng.choice(available_faces)
        else:
            selected_face = self.rng.choice(self.current_faces)
        
        self.last_seen_face_label = selected_face.label
        return selected_face
    
    def select_face_by_saliency(self, sound_detected: bool, sound_angle: float) -> Tuple[Optional[FaceInfo], Tuple[int, int]]:
        """Select face/location using saliency-based winner-takes-all"""
        # Create combined saliency map
        saliency_map = self.create_social_saliency_map(sound_detected, sound_angle)
        
        # Find winner-takes-all location
        _, _, _, max_loc = cv2.minMaxLoc(saliency_map)
        winner_x, winner_y = int(max_loc[0]), int(max_loc[1])
        
        # Check if winner corresponds to a face
        winner_face = None
        for face in self.current_faces:
            # Check if winner location is close to this face
            distance = math.sqrt((face.x - winner_x)**2 + (face.y - winner_y)**2)
            if distance < 50:  # Within 50 pixels
                winner_face = face
                break
        
        return winner_face, (winner_x, winner_y)
    
    def social_attention(self, face_detected: bool, face_in_range: bool, 
                        sound_detected: bool, sound_angle: float) -> bool:
        """
        Enhanced social attention behavior with multi-modal fusion and smooth movement
        Returns True if attention command was issued
        """
        
        if not (face_detected or sound_detected):
            return False
        
        # Determine best target using existing logic
        target_info = self._evaluate_targets(face_detected, face_in_range, sound_detected, sound_angle)
        
        if not target_info:
            return False
        
        target_type, target_pitch, target_yaw, confidence, novelty = target_info
        
        # Evaluate if this target deserves attention
        attention_score = self.attention_manager.evaluate_attention_worthiness(
            target_type, target_yaw, target_pitch, confidence, novelty
        )
        
        # Update smooth controller
        self.smooth_controller.update_target(target_pitch, target_yaw, confidence)
        self.smooth_controller.update_interest_level(attention_score)
        
        # Update focus if score is high enough
        focus_updated = self.attention_manager.update_focus(
            target_type, target_yaw, target_pitch, attention_score
        )
        
        # Execute movement only if conditions are met
        movement_executed = self.smooth_controller.execute_movement()
        
        # Update movement state
        self.smooth_controller.update_movement_state()
        
        return movement_executed or focus_updated
    
    def _evaluate_targets(self, face_detected, face_in_range, sound_detected, sound_angle):
        """Evaluate and select the best target"""
        best_target = None
        best_score = 0.0
        
        # Increment sound count for mode switching logic
        if sound_detected:
            self.sound_count += 1
        
        if face_detected and sound_detected:
            # Both face and sound detected - complex fusion logic
            if self.control_mode == SocialControlMode.RANDOM:
                if self.sound_count > self.config.SOUND_COUNT_THRESHOLD:
                    # Switch to sound after threshold
                    target_type = 'sound_general'
                    target_pitch = 0.0  # Assume sound at head level
                    target_yaw = sound_angle
                    self.sound_count = 0  # Reset counter
                    return (target_type, target_pitch, target_yaw, 0.8, 1.0)
                else:
                    # Look at randomly selected face
                    selected_face = self.select_face_randomly()
                    if selected_face:
                        target_pitch, target_yaw = self.face_to_head_angles(selected_face)
                        self.record_face_attention(selected_face)
                        target_type = self._get_face_type(selected_face)
                        return (target_type, target_pitch, target_yaw, selected_face.confidence, 0.8)
            
            else:  # SALIENCY mode
                # Use saliency-based selection
                winner_face, winner_location = self.select_face_by_saliency(sound_detected, sound_angle)
                
                if winner_face:
                    target_pitch, target_yaw = self.face_to_head_angles(winner_face)
                    self.record_face_attention(winner_face)
                    target_type = self._get_face_type(winner_face)
                    return (target_type, target_pitch, target_yaw, winner_face.confidence, 0.9)
                else:
                    # Winner was sound location or other salient point
                    target_pitch, target_yaw = self.pixel_to_head_angles(winner_location[0], winner_location[1])
                    return ('sound_general', target_pitch, target_yaw, 0.8, 1.0)
        
        elif face_detected and not sound_detected:
            # Only face detected
            if self.control_mode == SocialControlMode.RANDOM:
                selected_face = self.select_face_randomly()
            else:
                # Use saliency without sound
                winner_face, winner_location = self.select_face_by_saliency(False, 0.0)
                selected_face = winner_face
            
            if selected_face:
                target_pitch, target_yaw = self.face_to_head_angles(selected_face)
                self.record_face_attention(selected_face)
                target_type = self._get_face_type(selected_face)
                return (target_type, target_pitch, target_yaw, selected_face.confidence, 0.7)
        
        elif not face_detected and sound_detected:
            # Only sound detected
            target_pitch = 0.0  # Assume sound at head level
            target_yaw = sound_angle
            self.sound_count = 0  # Reset counter
            return ('sound_general', target_pitch, target_yaw, 0.8, 1.0)
        
        return None
    
    def _get_face_type(self, face: FaceInfo) -> str:
        """Determine face type for attention evaluation"""
        if face.mutual_gaze:
            return 'face_with_gaze'
        elif face.distance < 1.5:
            return 'face_close'
        else:
            return 'face_general'
    
    def face_to_head_angles(self, face: FaceInfo) -> Tuple[float, float]:
        """Convert face pixel location to head angles"""
        return self.pixel_to_head_angles(face.x, face.y)
    
    def pixel_to_head_angles(self, x: int, y: int) -> Tuple[float, float]:
        """Convert pixel coordinates to head angles"""
        d_yaw, d_pitch = self.camera_spec.pixel_to_angle(x, y)
        
        # Convert relative angles to absolute head positions
        current_pitch = self.head_controller.current_pitch
        current_yaw = self.head_controller.current_yaw
        
        target_pitch = current_pitch + d_pitch
        target_yaw = current_yaw + d_yaw
        
        return target_pitch, target_yaw
    
    def record_face_attention(self, face: FaceInfo):
        """Record that we're paying attention to this face"""
        self.face_attention_history[face.label] = time.time()
    
    def get_debug_info(self) -> Dict:
        """Get debug information about social attention state"""
        return {
            'control_mode': self.control_mode.value,
            'sound_count': self.sound_count,
            'num_faces': len(self.current_faces),
            'last_seen_face': self.last_seen_face_label,
            'face_history_size': len(self.face_attention_history),
            'has_saliency_map': self.social_saliency_map is not None,
            'movement_status': self.smooth_controller.get_movement_status()
        }


class AttentionBehaviors:
    """Implements different attention behaviors with enhanced social attention"""
    
    def __init__(self, head_controller: HeadController, limits: RobotLimits, camera_spec: CameraSpec):
        self.head_controller = head_controller
        self.limits = limits
        self.camera_spec = camera_spec
        self.rng = random.Random()
        
        # Enhanced social attention system
        self.social_attention_system = EnhancedSocialAttention(
            head_controller, camera_spec
        )
        
        # Scanning state
        self.scan_direction = 1
        self.scan_step = 0.08
        
        # Seeking state
        self.seek_cooldown = 0
        self.seek_period = (8, 18)
        self.seek_target: Optional[Tuple[float, float]] = None

    def update_face_detections(self, faces: List[FaceInfo]):
        """Update face detections for social attention"""
        self.social_attention_system.update_faces(faces)
    
    def set_social_control_mode(self, mode: SocialControlMode):
        """Set social attention control mode"""
        self.social_attention_system.set_control_mode(mode)

    def location_attention(self, target_x: float, target_y: float, target_z: float):
        """Point head towards a 3D location"""
        distance_xy = math.hypot(target_x, target_y)
        if distance_xy == 0.0 and target_z == 0.0:
            return
        
        target_yaw = math.atan2(target_y, target_x)
        target_pitch = math.atan2(-target_z, max(1e-6, distance_xy))
        
        self.head_controller.move_towards(target_pitch, target_yaw)

    def social_attention(self, face_detected: bool, face_in_range: bool, sound_detected: bool, sound_angle: float) -> bool:
        """Enhanced social attention using the new system"""
        return self.social_attention_system.social_attention(
            face_detected, face_in_range, sound_detected, sound_angle
        )

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
    """Main ROS2 node for enhanced overt attention system"""
    
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
        
        # Use enhanced behaviors
        self.behaviors = AttentionBehaviors(self.head_controller, self.limits, camera_spec)
        
        # Set default social mode from config
        social_mode = self.config_manager.config.get("social_mode", "saliency")
        if social_mode.lower() == "random":
            self.behaviors.set_social_control_mode(SocialControlMode.RANDOM)
        else:
            self.behaviors.set_social_control_mode(SocialControlMode.SALIENCY)
        
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
        self.get_logger().info("Enhanced OvertAttentionSystem initialized")

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
        """Enhanced face callback with detailed face information"""
        if not hasattr(msg, "centroids"):
            return
        
        self.face_detected = len(msg.centroids) > 0
        self.face_in_range = False
        self.mutual_gaze = False
        
        # Create face saliency map (for backward compatibility with existing saliency processor)
        camera_spec = self.config_manager.get_camera_spec()
        faces_map = np.zeros((camera_spec.height, camera_spec.width), dtype=np.float32)
        
        # Create enhanced face information list
        enhanced_faces = []
        
        for i, centroid in enumerate(msg.centroids):
            x, y = int(centroid.x), int(centroid.y)
            
            if 0 <= x < camera_spec.width and 0 <= y < camera_spec.height:
                # Extract face information
                distance = getattr(centroid, "z", 10.0)
                confidence = getattr(centroid, "confidence", 1.0)
                
                # Check if face is in range
                if distance <= 2.0:
                    self.face_in_range = True
                
                # Check mutual gaze for this face
                face_mutual_gaze = False
                if hasattr(msg, "mutual_gaze") and i < len(msg.mutual_gaze):
                    face_mutual_gaze = bool(msg.mutual_gaze[i])
                    self.mutual_gaze = self.mutual_gaze or face_mutual_gaze
                
                # Create FaceInfo object
                face_info = FaceInfo(
                    x=x,
                    y=y,
                    distance=distance,
                    confidence=confidence,
                    label=i + 1,  # Simple labeling (could be more sophisticated)
                    mutual_gaze=face_mutual_gaze,
                    last_seen=time.time()
                )
                enhanced_faces.append(face_info)
                
                # Add to saliency map for backward compatibility
                scale = 1.0 / max(1e-3, distance)
                faces_map[y, x] = max(faces_map[y, x], scale)
        
        # Update both systems
        self.saliency_processor.update_faces(faces_map)  # Existing system
        self.behaviors.update_face_detections(enhanced_faces)  # New enhanced system

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
        """Main control loop with enhanced social attention"""
        # Execute behavior based on current mode
        if self.current_mode == AttentionMode.LOCATION:
            self.behaviors.location_attention(*self.location_target)
            
        elif self.current_mode == AttentionMode.SOCIAL:
            # Use enhanced social attention
            attention_issued = self.behaviors.social_attention(
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
        """Enhanced debug visualization with social attention info"""
        if self.camera_image is None:
            return
            
        try:
            # Show original camera image
            cv2.imshow("Camera Image", self.camera_image)
            
            # If we have WTA visualization from scanning mode, show it
            if hasattr(self, 'wta_visualization') and self.wta_visualization is not None:
                debug_msg = self.bridge.cv2_to_imgmsg(self.wta_visualization, encoding="bgr8")
                self.debug_image_pub.publish(debug_msg)
            else:
                debug_msg = self.bridge.cv2_to_imgmsg(self.camera_image, encoding="bgr8")
                self.debug_image_pub.publish(debug_msg)
            
            # Show social attention debug info
            if self.current_mode == AttentionMode.SOCIAL:
                self.show_social_attention_debug()
            
            # Show current head target as overlay on camera image
            self.show_head_target_overlay()
            
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Debug visualization failed: {e}")

    def show_social_attention_debug(self):
        """Show debug information for social attention system"""
        if self.behaviors.social_attention_system.social_saliency_map is not None:
            # Show social saliency map
            saliency_map = self.behaviors.social_attention_system.social_saliency_map
            # Normalize for display
            normalized = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colored_saliency = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            cv2.imshow("Social Saliency Map", colored_saliency)
        
        # Get debug info
        debug_info = self.behaviors.social_attention_system.get_debug_info()
        debug_text = f"Social Mode: {debug_info['control_mode']}, Faces: {debug_info['num_faces']}, Sound Count: {debug_info['sound_count']}"
        
        # Overlay debug text on camera image
        overlay = self.camera_image.copy()
        cv2.putText(overlay, debug_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add movement status
        movement_status = debug_info['movement_status']
        movement_text = f"Moving: {movement_status['is_moving']}, Confidence: {movement_status['target_confidence']:.2f}"
        cv2.putText(overlay, movement_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        cv2.imshow("Social Debug", overlay)

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