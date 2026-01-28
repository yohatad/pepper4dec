#!/usr/bin/env python3
"""
MiVOLO ROS2 Node for Age and Gender Estimation

Supports both face-only (3 channel) and face+body (6 channel) models.
Fully self-contained - no external MiVOLO dependencies required.

Subscribes to:
- objectdetection/data (ObjectDetection.msg) - person detections with track IDs
- facedetection/data (FaceDetection.msg) - face detections with matching label IDs
- camera image topic

Publishes age/gender estimates when new person track IDs are detected.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import torch
import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import json
import time
import threading

# Import custom messages
from cssr_interfaces.msg import ObjectDetection, FaceDetection


# ============================================================================
# Preprocessing functions (inlined from MiVOLO data_processing/misc.py)
# ============================================================================

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def class_letterbox(im: np.ndarray, new_shape: Tuple[int, int] = (640, 640), 
                    color: Tuple[int, int, int] = (0, 0, 0), scaleup: bool = True) -> np.ndarray:
    """Resize and pad image while maintaining aspect ratio."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    if im.shape[0] == new_shape[0] and im.shape[1] == new_shape[1]:
        return im

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im


def prepare_classification_images(
    img_list: List[Optional[np.ndarray]],
    target_size: int = 224,
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
    device: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Prepare images for MiVOLO classification model."""
    prepared_images: List[torch.Tensor] = []

    for img in img_list:
        if img is None:
            img_tensor = torch.zeros((3, target_size, target_size), dtype=torch.float32)
            for c in range(3):
                img_tensor[c] = (img_tensor[c] - mean[c]) / std[c]
            img_tensor = img_tensor.unsqueeze(0)
            prepared_images.append(img_tensor)
            continue
        
        img = class_letterbox(img, new_shape=(target_size, target_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = (img - mean) / std
        img = img.astype(dtype=np.float32)
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.unsqueeze(0)
        prepared_images.append(img_tensor)

    if len(prepared_images) == 0:
        return None

    prepared_input = torch.cat(prepared_images)

    if device:
        prepared_input = prepared_input.to(device)

    return prepared_input


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class BoundingBox:
    """Bounding box with track ID."""
    x1: float
    y1: float
    x2: float
    y2: float
    label_id: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    mutual_gaze: bool = False  # True when person is looking at camera
    depth: float = 0.0  # Depth in meters (z component of centroid)
    
    @classmethod
    def from_centroid(cls, centroid: Point, width: float, height: float, 
                      label_id: str, confidence: float = 1.0, 
                      mutual_gaze: bool = False) -> 'BoundingBox':
        """Create bbox from centroid and dimensions."""
        return cls(
            x1=centroid.x - width / 2,
            y1=centroid.y - height / 2,
            x2=centroid.x + width / 2,
            y2=centroid.y + height / 2,
            label_id=label_id,
            confidence=confidence,
            mutual_gaze=mutual_gaze,
            depth=centroid.z  # Store depth from z component
        )
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass 
class PersonProfile:
    """Stores age/gender estimates with temporal smoothing."""
    label_id: str
    
    age_history: deque = field(default_factory=lambda: deque(maxlen=5))
    gender_history: deque = field(default_factory=lambda: deque(maxlen=5))
    
    age: Optional[float] = None
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    estimation_count: int = 0
    has_valid_estimate: bool = False
    
    last_person_bbox: Optional[BoundingBox] = None
    last_face_bbox: Optional[BoundingBox] = None
    
    def add_estimate(self, age: float, gender: str, gender_conf: float):
        """Add new estimate and update smoothed values."""
        self.age_history.append(age)
        self.gender_history.append((gender, gender_conf))
        self.estimation_count += 1
        self.last_updated = time.time()
        self.has_valid_estimate = True
        
        self.age = float(np.median(list(self.age_history)))
        
        male_score = sum(conf if g == "male" else 0 for g, conf in self.gender_history)
        female_score = sum(conf if g == "female" else 0 for g, conf in self.gender_history)
        total = male_score + female_score
        
        if total > 0:
            if male_score > female_score:
                self.gender = "male"
                self.gender_confidence = male_score / total
            else:
                self.gender = "female"
                self.gender_confidence = female_score / total
    
    def to_dict(self) -> dict:
        return {
            'label_id': self.label_id,
            'age': round(self.age, 1) if self.age else None,
            'gender': self.gender,
            'gender_confidence': round(self.gender_confidence, 3) if self.gender_confidence else None,
            'estimation_count': self.estimation_count,
            'person_bbox': {
                'x1': self.last_person_bbox.x1,
                'y1': self.last_person_bbox.y1,
                'x2': self.last_person_bbox.x2,
                'y2': self.last_person_bbox.y2
            } if self.last_person_bbox else None
        }


# ============================================================================
# MiVOLO Inference
# ============================================================================

class MiVOLOInference:
    """MiVOLO model wrapper supporting both face-only and face+body models."""
    
    def __init__(self, model_path: str, device: str = "cuda", face_only: bool = False, logger=None):
        self.device = device
        self.logger = logger
        self.face_only = face_only
        
        self._log(f"Loading MiVOLO model from {model_path}")
        self._log(f"Face-only mode: {face_only}")
        
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.model.to(self.device)
        
        self.input_size = 224
        self.mean = IMAGENET_DEFAULT_MEAN
        self.std = IMAGENET_DEFAULT_STD
        
        # Age decoding parameters
        self.min_age = 0
        self.max_age = 95
        self.avg_age = 48
        
        self._warmup()
    
    def _log(self, msg: str, level: str = "info"):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(f"[MiVOLO] {msg}")
    
    def _warmup(self, num_runs: int = 3):
        """Run dummy inference to load model into GPU memory."""
        self._log("Warming up model on GPU...")
        
        # Face-only model: [B, 3, 224, 224]
        # Face+body model: [B, 6, 224, 224] (face channels first, then body)
        channels = 3 if self.face_only else 6
        dummy_input = torch.randn(1, channels, self.input_size, self.input_size).to(self.device)
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(dummy_input)
            if self.device == "cuda":
                torch.cuda.synchronize()
        
        self._log("Model warmup complete")
    
    def predict(self, face_crop: np.ndarray, body_crop: np.ndarray = None) -> Tuple[Optional[float], Optional[str], Optional[float]]:
        """
        Run age/gender estimation.
        
        For face+body model: requires both face_crop and body_crop
        For face-only model: only requires face_crop
        
        Args:
            face_crop: BGR image crop of face
            body_crop: BGR image crop of person body (required for face+body model)
            
        Returns:
            Tuple of (age, gender, gender_confidence) or (None, None, None) on failure
        """
        if face_crop is None or face_crop.size == 0:
            self._log("Face crop is None or empty", "warn")
            return None, None, None
        
        if not self.face_only:
            # Face+body model
            if body_crop is None or body_crop.size == 0:
                self._log("Body crop is None or empty (required for face+body model)", "warn")
                return None, None, None
            
            face_input = prepare_classification_images(
                [face_crop], self.input_size, self.mean, self.std, self.device
            )
            body_input = prepare_classification_images(
                [body_crop], self.input_size, self.mean, self.std, self.device
            )
            
            if face_input is None or body_input is None:
                return None, None, None
            
            # Concatenate: [B, 6, H, W] - face first, then body
            model_input = torch.cat((face_input, body_input), dim=1)
        else:
            # Face-only model
            face_input = prepare_classification_images(
                [face_crop], self.input_size, self.mean, self.std, self.device
            )
            if face_input is None:
                return None, None, None
            model_input = face_input
        
        with torch.no_grad():
            output = self.model(model_input)
        
        # Parse output: [gender_male_logit, gender_female_logit, age_normalized]
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


# ============================================================================
# ROS2 Node
# ============================================================================

class MiVOLONode(Node):
    """ROS2 Node for MiVOLO age/gender estimation."""
    
    def __init__(self):
        super().__init__('mivolo_node')
        
        # Declare parameters
        self.declare_parameter('mivolo_model_path', '/home/yoha/ros2_ws/src/cssr4africa/cssr_system/face_detection/models/mivolo_agegender_faceonly_422.torchscript')
        self.declare_parameter('device', 'cuda')
        # NOTE: Despite the filename saying "faceonly", this model actually requires face+body (6 channels)
        self.declare_parameter('face_only', False)
        self.declare_parameter('face_topic', '/faceDetection/data')
        self.declare_parameter('person_topic', '/objectDetection/data')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('output_topic', 'mivolo/results')
        self.declare_parameter('max_cache_age_sec', 2.0)
        self.declare_parameter('min_estimate_interval_sec', 0.5)
        self.declare_parameter('re_estimate_interval_sec', 30.0)
        self.declare_parameter('person_class_name', 'person')
        self.declare_parameter('max_depth_m', 4.0)  # Maximum depth for estimation (meters)
        
        # Get parameters
        model_path = self.get_parameter('mivolo_model_path').value
        device = self.get_parameter('device').value
        face_only = self.get_parameter('face_only').value
        
        self.face_topic = self.get_parameter('face_topic').value
        self.person_topic = self.get_parameter('person_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        
        self.max_cache_age = self.get_parameter('max_cache_age_sec').value
        self.min_estimate_interval = self.get_parameter('min_estimate_interval_sec').value
        self.re_estimate_interval = self.get_parameter('re_estimate_interval_sec').value
        self.person_class_name = self.get_parameter('person_class_name').value
        self.max_depth_m = self.get_parameter('max_depth_m').value
        
        # Initialize MiVOLO model
        self.mivolo = MiVOLOInference(model_path, device, face_only, self.get_logger())
        
        # State tracking
        self.known_label_ids: Set[str] = set()
        self.person_profiles: Dict[str, PersonProfile] = {}
        
        # Recent detections cache
        self.recent_persons: Dict[str, BoundingBox] = {}
        self.recent_faces: Dict[str, BoundingBox] = {}
        
        # Image buffer
        self.latest_image: Optional[np.ndarray] = None
        self.image_timestamp: float = 0.0
        self.image_lock = threading.Lock()
        
        # Processing lock
        self.processing_lock = threading.Lock()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.person_sub = self.create_subscription(
            ObjectDetection, self.person_topic,
            self.person_callback, sensor_qos
        )
        
        self.face_sub = self.create_subscription(
            FaceDetection, self.face_topic,
            self.face_callback, sensor_qos
        )
        
        self.image_sub = self.create_subscription(
            Image, self.image_topic,
            self.image_callback, sensor_qos
        )
        
        # Publisher
        self.result_pub = self.create_publisher(String, self.output_topic, 10)
        
        # Cleanup timer
        self.cleanup_timer = self.create_timer(1.0, self.cleanup_stale_data)
        
        # Debug timer
        self.debug_timer = self.create_timer(5.0, self.debug_status)
        
        self.get_logger().info(
            f"MiVOLO node initialized\n"
            f"  Model: {model_path}\n"
            f"  Face-only mode: {face_only}\n"
            f"  Person topic: {self.person_topic}\n"
            f"  Face topic: {self.face_topic}\n"
            f"  Image topic: {self.image_topic}\n"
            f"  Output topic: {self.output_topic}\n"
            f"  Max depth: {self.max_depth_m}m"
        )
    
    def debug_status(self):
        """Print debug status periodically."""
        with self.image_lock:
            has_image = self.latest_image is not None
            image_age = time.time() - self.image_timestamp if has_image else -1
        
        # Count faces with mutual gaze and valid depth
        gaze_count = sum(1 for f in self.recent_faces.values() if f.mutual_gaze)
        gaze_depth_ok = sum(1 for f in self.recent_faces.values() 
                           if f.mutual_gaze and 0 < f.depth < self.max_depth_m)
        
        self.get_logger().info(
            f"[DEBUG] Image: {'YES' if has_image else 'NO'} (age={image_age:.1f}s), "
            f"Persons: {len(self.recent_persons)}, "
            f"Faces: {len(self.recent_faces)} ({gaze_count} gaze, {gaze_depth_ok} <{self.max_depth_m}m), "
            f"Known IDs: {len(self.known_label_ids)}"
        )
        
        if self.recent_persons:
            self.get_logger().info(f"[DEBUG] Person IDs: {list(self.recent_persons.keys())}")
        if self.recent_faces:
            face_info = [(lid, f"{f.depth:.2f}m", "GAZE" if f.mutual_gaze else "") 
                         for lid, f in self.recent_faces.items()]
            self.get_logger().info(f"[DEBUG] Faces: {face_info}")
    
    def image_callback(self, msg: Image):
        """Store latest image."""
        with self.image_lock:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_timestamp = time.time()
    
    def person_callback(self, msg: ObjectDetection):
        """Process person detections."""
        current_time = time.time()
        new_person_ids = []
        
        for i in range(len(msg.object_label_id)):
            if msg.class_names[i] != self.person_class_name:
                continue
            
            label_id = msg.object_label_id[i]
            
            person_bbox = BoundingBox.from_centroid(
                centroid=msg.centroids[i],
                width=msg.width[i],
                height=msg.height[i],
                label_id=label_id,
                confidence=msg.confidences[i]
            )
            
            self.recent_persons[label_id] = person_bbox
            
            if label_id not in self.known_label_ids:
                self.known_label_ids.add(label_id)
                self.person_profiles[label_id] = PersonProfile(label_id=label_id)
                new_person_ids.append(label_id)
                self.get_logger().info(f"New person detected: {label_id}")
            
            self.person_profiles[label_id].last_person_bbox = person_bbox
        
        # For new persons, check if we already have a face with mutual_gaze and valid depth
        for label_id in new_person_ids:
            face_bbox = self.recent_faces.get(label_id)
            if face_bbox and face_bbox.mutual_gaze:
                depth_ok = (face_bbox.depth > 0.0 and face_bbox.depth < self.max_depth_m)
                if depth_ok:
                    self.get_logger().info(
                        f"New person {label_id} has mutual gaze at {face_bbox.depth:.2f}m, triggering estimation"
                    )
                    self._schedule_estimation(label_id)
    
    def face_callback(self, msg: FaceDetection):
        """Process face detections. Only triggers estimation when mutual_gaze is True and depth < max_depth_m."""
        for i in range(len(msg.face_label_id)):
            label_id = msg.face_label_id[i]
            mutual_gaze = msg.mutual_gaze[i] if i < len(msg.mutual_gaze) else False
            depth = msg.centroids[i].z if i < len(msg.centroids) else 0.0
            
            face_bbox = BoundingBox.from_centroid(
                centroid=msg.centroids[i],
                width=msg.width[i],
                height=msg.height[i],
                label_id=label_id,
                confidence=1.0,
                mutual_gaze=mutual_gaze
            )
            
            self.recent_faces[label_id] = face_bbox
            
            if label_id in self.person_profiles:
                self.person_profiles[label_id].last_face_bbox = face_bbox
            
            # Only trigger estimation when:
            # 1. mutual_gaze is True
            # 2. depth is valid (> 0) and less than max_depth_m
            # 3. Person is already known and doesn't have valid estimate
            depth_ok = (depth > 0.0 and depth < self.max_depth_m)
            
            if mutual_gaze and depth_ok and label_id in self.known_label_ids:
                profile = self.person_profiles.get(label_id)
                if profile and not profile.has_valid_estimate:
                    self.get_logger().info(
                        f"Mutual gaze detected for {label_id} at {depth:.2f}m (< {self.max_depth_m}m), triggering estimation"
                    )
                    self._schedule_estimation(label_id)
    
    def _should_re_estimate(self, profile: PersonProfile, current_time: float) -> bool:
        if not profile.has_valid_estimate:
            return True
        return (current_time - profile.last_updated) > self.re_estimate_interval
    
    def _schedule_estimation(self, label_id: str):
        thread = threading.Thread(target=self._run_estimation, args=(label_id,), daemon=True)
        thread.start()
    
    def _run_estimation(self, label_id: str):
        with self.processing_lock:
            self._estimate_for_person(label_id)
    
    def _estimate_for_person(self, label_id: str):
        """Perform age/gender estimation for a person."""
        profile = self.person_profiles.get(label_id)
        if profile is None:
            return
        
        if profile.estimation_count > 0:
            if (time.time() - profile.last_updated) < self.min_estimate_interval:
                return
        
        # Get face bbox (required)
        face_bbox = self.recent_faces.get(label_id)
        if face_bbox is None:
            self.get_logger().debug(f"No face bbox for {label_id}")
            return
        
        # Get person bbox (required for face+body model)
        person_bbox = self.recent_persons.get(label_id)
        if person_bbox is None and not self.mivolo.face_only:
            self.get_logger().debug(f"No person bbox for {label_id}")
            return
        
        # Get image
        with self.image_lock:
            if self.latest_image is None:
                self.get_logger().warn("No image available")
                return
            image = self.latest_image.copy()
        
        # Extract crops
        face_crop = self._extract_crop(image, face_bbox)
        if face_crop is None:
            self.get_logger().warn(f"Invalid face crop for {label_id}")
            return
        
        person_crop = None
        if not self.mivolo.face_only and person_bbox is not None:
            person_crop = self._extract_crop(image, person_bbox)
            if person_crop is None:
                self.get_logger().warn(f"Invalid person crop for {label_id}")
                return
        
        # Run inference
        try:
            self.get_logger().info(f"Running inference for {label_id} (face: {face_crop.shape}, body: {person_crop.shape if person_crop is not None else 'N/A'})")
            age, gender, gender_conf = self.mivolo.predict(face_crop, person_crop)
            
            if age is not None:
                profile.add_estimate(age, gender, gender_conf)
                
                self.get_logger().info(
                    f"[{label_id}] age={profile.age:.1f} (raw={age:.1f}), "
                    f"gender={profile.gender} ({profile.gender_confidence*100:.1f}%)"
                )
                
                self._publish_result(profile)
            else:
                self.get_logger().warn(f"Inference returned None for {label_id}")
                
        except Exception as e:
            self.get_logger().error(f"Estimation failed for {label_id}: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _extract_crop(self, image: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2].copy()
    
    def _publish_result(self, profile: PersonProfile):
        msg = String()
        msg.data = json.dumps(profile.to_dict())
        self.result_pub.publish(msg)
        self.get_logger().info(f"Published result for {profile.label_id}")
    
    def cleanup_stale_data(self):
        current_time = time.time()
        
        stale_persons = [lid for lid, bbox in self.recent_persons.items() if current_time - bbox.timestamp > self.max_cache_age]
        for lid in stale_persons:
            del self.recent_persons[lid]
        
        stale_faces = [lid for lid, bbox in self.recent_faces.items() if current_time - bbox.timestamp > self.max_cache_age]
        for lid in stale_faces:
            del self.recent_faces[lid]


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