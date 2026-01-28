#!/usr/bin/env python3
"""
MiVOLO ROS2 Node for Age and Gender Estimation (v2)

Enhanced version with:
- Custom message support
- Better ByteTrack integration
- Temporal smoothing for age/gender estimates
- Separate face and person tracking association
"""

import rclpy
import torch
import numpy as np
import cv2
import time
import threading

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from vision_msgs.msg import Detection2DArray, Detection2D
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

from timm.data import resolve_data_config
from data_processing.misc import prepare_classification_images

# Try to import custom message, fall back to String if not available
try:
    from mivolo_ros2.msg import PersonAgeGender
    USE_CUSTOM_MSG = True
except ImportError:
    from std_msgs.msg import String
    import json
    USE_CUSTOM_MSG = False


class EstimationState(Enum):
    """State of age/gender estimation for a tracked person."""
    PENDING = "pending"      # Waiting for face association
    ESTIMATED = "estimated"  # Has valid estimation
    FAILED = "failed"        # Estimation failed


@dataclass
class BoundingBox:
    """Bounding box with track ID and metadata."""
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    class_name: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside bbox."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def contains(self, other: 'BoundingBox', threshold: float = 0.8) -> bool:
        """Check if this bbox contains another (with overlap threshold)."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        intersection = (x2 - x1) * (y2 - y1)
        return (intersection / other.area) >= threshold if other.area > 0 else False


@dataclass 
class PersonProfile:
    """Stores and smooths age/gender estimates for a tracked person."""
    track_id: int
    state: EstimationState = EstimationState.PENDING
    
    # Raw estimates history for temporal smoothing
    age_history: deque = field(default_factory=lambda: deque(maxlen=5))
    gender_history: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Smoothed values
    age: Optional[float] = None
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    
    # Metadata
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    estimation_count: int = 0
    last_bbox: Optional[BoundingBox] = None
    
    def add_estimate(self, age: float, gender: str, gender_conf: float):
        """Add new estimate and update smoothed values."""
        self.age_history.append(age)
        self.gender_history.append((gender, gender_conf))
        self.estimation_count += 1
        self.last_updated = time.time()
        self.state = EstimationState.ESTIMATED
        
        # Compute smoothed age (median)
        self.age = float(np.median(list(self.age_history)))
        
        # Compute smoothed gender (weighted voting)
        male_score = sum(conf if g == "male" else 0 for g, conf in self.gender_history)
        female_score = sum(conf if g == "female" else 0 for g, conf in self.gender_history)
        
        if male_score > female_score:
            self.gender = "male"
            self.gender_confidence = male_score / (male_score + female_score)
        else:
            self.gender = "female"
            self.gender_confidence = female_score / (male_score + female_score)
    
    def to_dict(self) -> dict:
        return {
            'track_id': self.track_id,
            'age': self.age,
            'gender': self.gender,
            'gender_confidence': self.gender_confidence,
            'estimation_count': self.estimation_count,
            'state': self.state.value,
            'bbox': {
                'x1': self.last_bbox.x1,
                'y1': self.last_bbox.y1,
                'x2': self.last_bbox.x2,
                'y2': self.last_bbox.y2
            } if self.last_bbox else None
        }


class MiVOLOInference:
    """MiVOLO model wrapper with GPU warmup."""
    
    def __init__(self, model_path: str, device: str = "cuda", logger=None):
        self.device = device
        self.logger = logger
        
        self._log(f"Loading MiVOLO model from {model_path}")
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        self.input_size = 224
        self.data_config = resolve_data_config(model=self.model, use_test_size=True)
        
        # Age decoding parameters
        self.min_age = 0
        self.max_age = 95
        self.avg_age = 48
        
        # Warmup
        self._warmup()
    
    def _log(self, msg: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(f"[MiVOLO] {msg}")
    
    def _warmup(self, num_runs: int = 3):
        """Run dummy inference to load model into GPU and stabilize performance."""
        self._log("Warming up model on GPU...")
        dummy_face = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        dummy_body = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        dummy_input = torch.cat((dummy_face, dummy_body), dim=1)
        
        # Run multiple times to ensure CUDA kernels are compiled
        for i in range(num_runs):
            with torch.no_grad():
                _ = self.model(dummy_input)
            if self.device == "cuda":
                torch.cuda.synchronize()
        
        self._log("Model warmup complete")
    
    def _resize_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize crop to model input size."""
        if crop.shape[0] == 0 or crop.shape[1] == 0:
            return None
        return cv2.resize(crop, (self.input_size, self.input_size))
    
    def prepare_input(self, face_crop: np.ndarray, body_crop: np.ndarray) -> Optional[torch.Tensor]:
        """Prepare face and body crops for model input."""
        # Resize crops
        face_resized = self._resize_crop(face_crop)
        body_resized = self._resize_crop(body_crop)
        
        if face_resized is None or body_resized is None:
            return None
        
        face_input = prepare_classification_images(
            [face_resized], self.input_size,
            self.data_config["mean"], self.data_config["std"],
            device=self.device
        )
        body_input = prepare_classification_images(
            [body_resized], self.input_size,
            self.data_config["mean"], self.data_config["std"],
            device=self.device
        )
        
        if face_input is None or body_input is None:
            return None
        
        return torch.cat((face_input, body_input), dim=1)
    
    def predict(self, face_crop: np.ndarray, body_crop: np.ndarray) -> Tuple[Optional[float], Optional[str], Optional[float]]:
        """
        Run age/gender estimation.
        
        Returns:
            Tuple of (age, gender, gender_confidence) or (None, None, None) on failure
        """
        model_input = self.prepare_input(face_crop, body_crop)
        if model_input is None:
            return None, None, None
        
        with torch.no_grad():
            output = self.model(model_input)
        
        # Parse output: [gender_male, gender_female, age]
        age_output = output[:, 2]
        gender_output = output[:, :2].softmax(-1)
        gender_probs, gender_idx = gender_output.topk(1)
        
        # Decode age
        age = age_output[0].item()
        age = age * (self.max_age - self.min_age) + self.avg_age
        age = round(age, 2)
        
        # Decode gender
        gender = "male" if gender_idx[0].item() == 0 else "female"
        gender_confidence = gender_probs[0].item()
        
        return age, gender, gender_confidence


class MiVOLONode(Node):
    """ROS2 Node for MiVOLO age/gender estimation."""
    
    def __init__(self):
        super().__init__('mivolo_node')
        
        # Declare parameters
        self.declare_parameter('mivolo_model_path', '/models/mivolo/mivolo_model.pt')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('face_topic', 'facedetection/data')
        self.declare_parameter('person_topic', 'objectdetection/data')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('output_topic', 'mivolo/results')
        self.declare_parameter('face_containment_threshold', 0.7)
        self.declare_parameter('max_cache_age_sec', 2.0)
        self.declare_parameter('min_estimate_interval_sec', 1.0)
        self.declare_parameter('publish_all_tracked', True)
        
        # Get parameters
        model_path = self.get_parameter('mivolo_model_path').value
        device = self.get_parameter('device').value
        
        self.face_topic = self.get_parameter('face_topic').value
        self.person_topic = self.get_parameter('person_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        
        self.face_containment_threshold = self.get_parameter('face_containment_threshold').value
        self.max_cache_age = self.get_parameter('max_cache_age_sec').value
        self.min_estimate_interval = self.get_parameter('min_estimate_interval_sec').value
        self.publish_all_tracked = self.get_parameter('publish_all_tracked').value
        
        # Initialize MiVOLO model
        self.mivolo = MiVOLOInference(model_path, device, self.get_logger())
        
        # State tracking
        self.known_person_ids: Set[int] = set()
        self.person_profiles: Dict[int, PersonProfile] = {}
        self.recent_persons: Dict[int, BoundingBox] = {}
        self.recent_faces: List[BoundingBox] = []
        
        # Image buffer
        self.latest_image: Optional[np.ndarray] = None
        self.image_timestamp: Optional[float] = None
        self.image_lock = threading.Lock()
        
        # Processing lock to prevent concurrent estimation
        self.processing_lock = threading.Lock()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # QoS profile
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.person_sub = self.create_subscription(
            Detection2DArray, self.person_topic,
            self.person_callback, sensor_qos
        )
        
        self.face_sub = self.create_subscription(
            Detection2DArray, self.face_topic,
            self.face_callback, sensor_qos
        )
        
        self.image_sub = self.create_subscription(
            Image, self.image_topic,
            self.image_callback, sensor_qos
        )
        
        # Publisher
        if USE_CUSTOM_MSG:
            self.result_pub = self.create_publisher(PersonAgeGender, self.output_topic, 10)
        else:
            self.result_pub = self.create_publisher(String, self.output_topic, 10)
        
        # Timers
        self.cleanup_timer = self.create_timer(1.0, self.cleanup_stale_data)
        
        self.get_logger().info(
            f"MiVOLO node initialized\n"
            f"  - Person topic: {self.person_topic}\n"
            f"  - Face topic: {self.face_topic}\n"
            f"  - Image topic: {self.image_topic}\n"
            f"  - Output topic: {self.output_topic}"
        )
    
    def image_callback(self, msg: Image):
        """Store latest image."""
        with self.image_lock:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_timestamp = time.time()
    
    def _parse_detection(self, det: Detection2D, default_class: str) -> BoundingBox:
        """Parse Detection2D message to BoundingBox."""
        bbox = det.bbox
        x_center = bbox.center.x
        y_center = bbox.center.y
        width = bbox.size_x
        height = bbox.size_y
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Extract track ID - try multiple possible locations
        track_id = 0
        
        # Method 1: Direct id field
        if hasattr(det, 'id') and det.id:
            try:
                track_id = int(det.id)
            except (ValueError, TypeError):
                pass
        
        # Method 2: From tracking_id in results (common in ByteTrack output)
        if track_id == 0 and det.results:
            for result in det.results:
                # Check for tracking_id attribute
                if hasattr(result, 'tracking_id'):
                    track_id = int(result.tracking_id)
                    break
                # Check hypothesis id (sometimes used for track ID)
                if hasattr(result.hypothesis, 'id'):
                    try:
                        track_id = int(result.hypothesis.id)
                        break
                    except (ValueError, TypeError):
                        pass
        
        # Get class name and confidence
        class_name = default_class
        confidence = 0.0
        if det.results:
            class_name = det.results[0].hypothesis.class_id or default_class
            confidence = det.results[0].hypothesis.score
        
        return BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            track_id=track_id,
            class_name=class_name,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def person_callback(self, msg: Detection2DArray):
        """Process person detections and trigger estimation for new IDs."""
        new_persons = []
        
        for det in msg.detections:
            person_bbox = self._parse_detection(det, 'person')
            
            # Skip detections without valid track ID
            if person_bbox.track_id == 0:
                continue
            
            self.recent_persons[person_bbox.track_id] = person_bbox
            
            # Check if new person
            if person_bbox.track_id not in self.known_person_ids:
                self.known_person_ids.add(person_bbox.track_id)
                self.person_profiles[person_bbox.track_id] = PersonProfile(
                    track_id=person_bbox.track_id
                )
                new_persons.append(person_bbox.track_id)
                self.get_logger().info(f"New person detected: track_id={person_bbox.track_id}")
            
            # Update last bbox
            if person_bbox.track_id in self.person_profiles:
                self.person_profiles[person_bbox.track_id].last_bbox = person_bbox
        
        # Trigger estimation for new persons
        for track_id in new_persons:
            self._schedule_estimation(track_id)
    
    def face_callback(self, msg: Detection2DArray):
        """Process face detections."""
        current_time = time.time()
        
        # Clear old faces and add new ones
        self.recent_faces = [
            f for f in self.recent_faces 
            if current_time - f.timestamp < self.max_cache_age
        ]
        
        for det in msg.detections:
            face_bbox = self._parse_detection(det, 'face')
            self.recent_faces.append(face_bbox)
    
    def _schedule_estimation(self, person_track_id: int):
        """Schedule age/gender estimation for a person."""
        # Run in a separate thread to not block callbacks
        thread = threading.Thread(
            target=self._run_estimation,
            args=(person_track_id,),
            daemon=True
        )
        thread.start()
    
    def _run_estimation(self, person_track_id: int):
        """Run estimation for a person (called from thread)."""
        with self.processing_lock:
            self._estimate_for_person(person_track_id)
    
    def _find_best_face(self, person_bbox: BoundingBox) -> Optional[BoundingBox]:
        """Find best matching face for a person bbox."""
        best_face = None
        best_score = 0.0
        
        for face_bbox in self.recent_faces:
            # Check if face center is inside person bbox
            face_cx, face_cy = face_bbox.center
            if not person_bbox.contains_point(face_cx, face_cy):
                continue
            
            # Check containment ratio
            if person_bbox.contains(face_bbox, self.face_containment_threshold):
                # Score by face size (larger faces are better)
                score = face_bbox.area
                if score > best_score:
                    best_score = score
                    best_face = face_bbox
        
        return best_face
    
    def _estimate_for_person(self, person_track_id: int):
        """Perform age/gender estimation for a person."""
        if person_track_id not in self.recent_persons:
            self.get_logger().debug(f"Person {person_track_id} no longer in recent detections")
            return
        
        profile = self.person_profiles.get(person_track_id)
        if profile is None:
            return
        
        # Check minimum interval
        if profile.estimation_count > 0:
            time_since_last = time.time() - profile.last_updated
            if time_since_last < self.min_estimate_interval:
                return
        
        person_bbox = self.recent_persons[person_track_id]
        face_bbox = self._find_best_face(person_bbox)
        
        if face_bbox is None:
            self.get_logger().debug(f"No face found for person {person_track_id}")
            return
        
        # Get image
        with self.image_lock:
            if self.latest_image is None:
                self.get_logger().warn("No image available")
                return
            image = self.latest_image.copy()
        
        # Extract crops
        h, w = image.shape[:2]
        
        # Person crop with bounds checking
        px1 = max(0, int(person_bbox.x1))
        py1 = max(0, int(person_bbox.y1))
        px2 = min(w, int(person_bbox.x2))
        py2 = min(h, int(person_bbox.y2))
        
        if px2 <= px1 or py2 <= py1:
            return
        person_crop = image[py1:py2, px1:px2]
        
        # Face crop
        fx1 = max(0, int(face_bbox.x1))
        fy1 = max(0, int(face_bbox.y1))
        fx2 = min(w, int(face_bbox.x2))
        fy2 = min(h, int(face_bbox.y2))
        
        if fx2 <= fx1 or fy2 <= fy1:
            return
        face_crop = image[fy1:fy2, fx1:fx2]
        
        # Run inference
        try:
            age, gender, gender_conf = self.mivolo.predict(face_crop, person_crop)
            
            if age is not None:
                profile.add_estimate(age, gender, gender_conf)
                
                self.get_logger().info(
                    f"Person {person_track_id}: "
                    f"age={profile.age:.1f} (raw={age:.1f}), "
                    f"gender={profile.gender} ({profile.gender_confidence*100:.1f}%)"
                )
                
                self._publish_result(profile)
            else:
                profile.state = EstimationState.FAILED
                
        except Exception as e:
            self.get_logger().error(f"Estimation failed for person {person_track_id}: {e}")
            profile.state = EstimationState.FAILED
    
    def _publish_result(self, profile: PersonProfile):
        """Publish estimation result."""
        if USE_CUSTOM_MSG:
            msg = PersonAgeGender()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.track_id = profile.track_id
            msg.age = profile.age or 0.0
            msg.gender = profile.gender or ""
            msg.gender_confidence = profile.gender_confidence or 0.0
            msg.estimation_count = profile.estimation_count
            
            if profile.last_bbox:
                msg.bbox_x1 = profile.last_bbox.x1
                msg.bbox_y1 = profile.last_bbox.y1
                msg.bbox_x2 = profile.last_bbox.x2
                msg.bbox_y2 = profile.last_bbox.y2
        else:
            msg = String()
            msg.data = json.dumps(profile.to_dict())
        
        self.result_pub.publish(msg)
    
    def cleanup_stale_data(self):
        """Clean up old detections and profiles."""
        current_time = time.time()
        
        # Clean old person detections
        stale_ids = [
            tid for tid, bbox in self.recent_persons.items()
            if current_time - bbox.timestamp > self.max_cache_age
        ]
        for tid in stale_ids:
            del self.recent_persons[tid]
        
        # Clean old faces
        self.recent_faces = [
            f for f in self.recent_faces
            if current_time - f.timestamp < self.max_cache_age
        ]


def main(args=None):
    rclpy.init(args=args)
    node = MiVOLONode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()