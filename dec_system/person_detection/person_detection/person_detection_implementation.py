"""
person_detection_implementation.py - Implementation code for running the Person Detection and Localization ROS2 node.

Supports configurable person detection with ByteTrack tracking using the bytetracker package.

Author: Yohannes Tadesse Haile
Date: December 07, 2025
Version: v1.0
"""

import cv2
import numpy as np
import rclpy
import os
import onnxruntime
import multiprocessing
import yaml
import random
import threading
import supervision as sv
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from typing import Tuple, List, Dict, Optional, Set
from dec_interfaces.msg import ObjectDetection

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def load_configuration() -> Dict:
    """
    Load configuration from the default YAML file location.
    
    Returns:
        Dict: Configuration data with defaults for missing values
    """
    config = {
        # Default values
        'camera': 'realsense',
        'useCompressed': False,
        'imageTimeout': 2.0,
        'verboseMode': True,
        'confidenceThreshold': 0.5,
        
        # ByteTrack parameters
        'trackThreshold': 0.45,    # Detection confidence threshold for tracking
        'trackBuffer': 30,         # Frames to keep lost tracks (max_age)
        'matchThreshold': 0.8,     # IoU threshold for matching
        'frameRate': 30,           # Expected frame rate
        
        # Object classes to track (can be names or indices)
        # If empty or 'all', track all detected objects
        'targetClasses': ['person'],
    }
    
    try:
        package_path = get_package_share_directory('person_detection')
        config_file = os.path.join(package_path, 'config', 'person_detection_configuration.yaml')
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as file:
                file_config = yaml.safe_load(file) or {}
                config.update(file_config)  # Update defaults with file values
                # print(f"Loaded configuration from {config_file}")
        else:
            print(f"Warning: Configuration file not found at {config_file}, using defaults")
            
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        print("Using default configuration values")
        
    return config

def get_class_indices(target_classes: List, class_names: List[str] = COCO_CLASSES) -> Set[int]:
    """
    Convert class names or indices to a set of valid class indices.
    
    Args:
        target_classes: List of class names (str) or indices (int)
        class_names: List of all class names
        
    Returns:
        Set of class indices to track
    """
    if not target_classes or target_classes == ['all'] or 'all' in target_classes:
        return set(range(len(class_names)))
    
    indices = set()
    for cls in target_classes:
        if isinstance(cls, int):
            if 0 <= cls < len(class_names):
                indices.add(cls)
            else:
                print(f"Warning: Class index {cls} out of range (0-{len(class_names)-1})")
        elif isinstance(cls, str):
            cls_lower = cls.lower().strip()
            try:
                idx = [name.lower() for name in class_names].index(cls_lower)
                indices.add(idx)
            except ValueError:
                print(f"Warning: Class name '{cls}' not found in COCO classes")
    
    return indices
class PersonDetectionNode(Node):
    def __init__(self, config: Dict, node_name: str = 'personDetection'):
        super().__init__(node_name)
        
        self.config = config
        self.pub_objects = self.create_publisher(ObjectDetection, "/personDetection/data", 10)
        self.debug_pub = self.create_publisher(Image, "/personDetection/debug", 1)
        self.depth_debug_pub = self.create_publisher(Image, "/personDetection/depth_debug", 1)

        self.bridge = CvBridge()
        self.depth_image: Optional[np.ndarray] = None
        self.color_image: Optional[np.ndarray] = None
        
        # Configuration values
        self.use_compressed = config['useCompressed']
        self.camera_type = config['camera']
        self.verbose_mode = config['verboseMode']
        self.image_timeout = config['imageTimeout']
        
        # Target classes to track
        self.target_class_indices = get_class_indices(config.get('targetClasses', ['person']))
        
        if self.verbose_mode:
            target_names = [COCO_CLASSES[i] for i in self.target_class_indices]
            self.get_logger().info(f"Tracking classes: {target_names}")
        
        self.node_name = self.get_name()
        self.timer_start = self.get_clock().now()
        self.last_image_time = None  # timestamp of the last received image

        # Thread safety for visualization
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None

        # Color tracking for object IDs
        self.object_colors: Dict[int, Tuple[int, int, int]] = {}
        
        # Store class_id mapping for tracked objects
        self.track_class_map: Dict[int, int] = {}

        # Start visualization timer (30 Hz)
        self.vis_timer = self.create_timer(1.0 / 30.0, self.visualization_callback)

        # Status logging timer (10 seconds)
        self.status_timer = self.create_timer(10.0, self.status_callback)

    def status_callback(self):
        """Log status every 10 seconds."""
        self.get_logger().info(f"{self.node_name}: running.")

    def update_latest_frame(self, frame: np.ndarray):
        """Update the latest frame safely for visualization."""
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def visualization_callback(self):
        """Timer callback for showing or publishing debug images."""
        color_frame = depth_vis = None
        with self.frame_lock:
            if self.latest_frame is not None:
                color_frame = self.latest_frame.copy()
            if self.depth_image is not None:
                depth_vis = self.make_depth_vis(self.depth_image)

        if color_frame is None and depth_vis is None:
            return

        if self.config.get("verboseMode", False) and os.environ.get("DISPLAY", "") != "":
            try:
                if color_frame is not None:
                    cv2.imshow("Person Detection Debug (RGB)", color_frame)
                if depth_vis is not None:
                    cv2.imshow("Person Detection Debug (Depth)", depth_vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info(f"{self.node_name}: User requested shutdown")
                    rclpy.shutdown()
            except Exception as e:
                self.get_logger().warn(f"imshow failed (likely headless): {e}")

            # Publish debug images when verboseMode=True
            try:
                if color_frame is not None:
                    msg = self.bridge.cv2_to_imgmsg(color_frame, encoding="bgr8")
                    self.debug_pub.publish(msg)
                if depth_vis is not None:
                    dmsg = self.bridge.cv2_to_imgmsg(depth_vis, encoding="bgr8")
                    self.depth_debug_pub.publish(dmsg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish debug images: {e}")

    def get_topic_names(self) -> Tuple[str, str]:
        """Get RGB and depth topic names based on camera type."""
        topic_mapping = {
            "realsense": ("RealSenseCameraRGB", "RealSenseCameraDepth"),
            "pepper": ("PepperFrontCamera", "PepperDepthCamera"),
            "video": ("RealSenseCameraRGB", "RealSenseCameraDepth"),
        }
        
        if self.camera_type not in topic_mapping:
            raise ValueError(f"Invalid camera type: {self.camera_type}")
            
        rgb_key, depth_key = topic_mapping[self.camera_type]
        rgb_topic = self.extract_topic(rgb_key)
        depth_topic = self.extract_topic(depth_key)
        
        if not rgb_topic or not depth_topic:
            raise ValueError("Failed to extract camera topics")
            
        return rgb_topic, depth_topic

    def extract_topic(self, image_topic: str) -> Optional[str]:
        """Extract topic name from configuration file."""
        try:
            package = get_package_share_directory('person_detection')
            config_path = os.path.join(package, 'data', 'pepper_topics.yaml')

            with open(config_path, 'r') as file:
                topics = yaml.safe_load(file)
                return topics.get(image_topic)
        except Exception as e:
            self.get_logger().error(f"Error extracting topic '{image_topic}': {e}")

        return None

    def wait_for_topics(self, color_topic: str, depth_topic: str) -> bool:
        """Wait indefinitely for topics to become available."""
        self.get_logger().info(f"Waiting for topics: {color_topic}, {depth_topic}")
        
        start_time = self.get_clock().now()
        warning_interval = 5.0
        last_warning_time = start_time
        
        while rclpy.ok():
            # Check if topics are available
            topic_names_and_types = self.get_topic_names_and_types()
            published_topics = [name for name, _ in topic_names_and_types]
            
            if color_topic in published_topics and depth_topic in published_topics:
                if self.verbose_mode:
                    self.get_logger().info("Both topics are available!")
                return True
            
            # Periodic warnings
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if (self.get_clock().now() - last_warning_time).nanoseconds / 1e9 >= warning_interval:
                missing = [t for t in [color_topic, depth_topic] if t not in published_topics]
                self.get_logger().warn(f"Still waiting for topics after {int(elapsed)}s: {missing}")
                last_warning_time = self.get_clock().now()
                
            rclpy.spin_once(self, timeout_sec=1.0)
        
        return False

    def subscribe_topics(self) -> bool:
        """Set up image subscribers based on camera configuration."""
        try:
            rgb_topic, depth_topic = self.get_topic_names()
            
            # Determine final topic names based on compression
            if self.use_compressed and self.camera_type == "realsense":
                color_topic = rgb_topic + "/compressed"
                depth_topic = depth_topic + "/compressedDepth"
                color_msg_type = CompressedImage
                depth_msg_type = CompressedImage
            elif self.use_compressed and self.camera_type == "pepper":
                self.get_logger().warn("Compressed images not available for Pepper cameras")
                color_topic = rgb_topic
                depth_topic = depth_topic
                color_msg_type = Image
                depth_msg_type = Image
            else:
                color_topic = rgb_topic
                depth_topic = depth_topic
                color_msg_type = Image
                depth_msg_type = Image

            # Wait for topics
            if not self.wait_for_topics(color_topic, depth_topic):
                return False

            # In your subscribe_topics() method, replace the subscriber creation with:
            self.color_sub = Subscriber(self, color_msg_type, color_topic, qos_profile=qos_profile_sensor_data)
            self.depth_sub = Subscriber(self, depth_msg_type, depth_topic, qos_profile=qos_profile_sensor_data)
            
            self.get_logger().info(f"Subscribed to {color_topic}")
            self.get_logger().info(f"Subscribed to {depth_topic}")

            # Set up synchronizer
            slop = 5.0 if self.camera_type == "pepper" else 0.1
            self.ats = ApproximateTimeSynchronizer(
                [self.color_sub, self.depth_sub], 
                queue_size=10, 
                slop=slop
            )
            self.ats.registerCallback(self.synchronized_callback)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to setup subscribers: {e}")
            return False

    def synchronized_callback(self, color_data, depth_data):
        """Process synchronized color and depth images."""
        self.last_image_time = self.get_clock().now().nanoseconds / 1e9
        
        try:
            # Check camera resolution if both images are available
            if self.depth_image is not None and self.color_image is not None:
                if not self.check_camera_resolution(self.color_image, self.depth_image) and self.camera_type != "pepper":
                    self.get_logger().error(f"{self.node_name}: Color camera and depth camera have different resolutions.")
                    rclpy.shutdown()
                    return
                    
            # Process color image
            if isinstance(color_data, CompressedImage):
                np_arr = np.frombuffer(color_data.data, np.uint8)
                self.color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                self.color_image = self.bridge.imgmsg_to_cv2(color_data, desired_encoding="bgr8")

            # Process depth image
            self.depth_image = self.process_depth_image(depth_data)

            if self.color_image is None or self.depth_image is None:
                self.get_logger().warn("Failed to decode images")
                return

            # Process the images
            self.process_images()
            
        except Exception as e:
            self.get_logger().error(f"Error in synchronized_callback: {e}")

    def process_depth_image(self, depth_data) -> Optional[np.ndarray]:
        """Process depth image data."""
        try:
            if isinstance(depth_data, CompressedImage):
                # Check if this is a compressedDepth format (ROS2 standard)
                if hasattr(depth_data, "format") and depth_data.format:
                    if "compressedDepth" in depth_data.format:
                        depth_header_size = 12
                        
                        # Extract actual image data
                        depth_img_data = depth_data.data[depth_header_size:]
                        np_arr = np.frombuffer(depth_img_data, np.uint8)
                        
                        # Decode as 16-bit depth image
                        depth_image = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)
                        
                        if depth_image is None:
                            self.get_logger().warn(f"Failed to decode compressedDepth image, format: {depth_data.format}")
                        return depth_image
                    else:
                        # Regular compressed image (JPEG/PNG)
                        np_arr = np.frombuffer(depth_data.data, np.uint8)
                        depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                        
                        if depth_image is None:
                            self.get_logger().warn(f"Failed to decode compressed image, format: {depth_data.format}")
                        return depth_image
            else:
                # Handle uncompressed Image message
                return self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth image processing error: {e}")
            return None

    def start_timeout_monitor(self):
        """Start the timeout monitor timer."""
        self.create_timer(1.0, self.check_timeout)

    def check_timeout(self):
        """Check if images have timed out."""
        if self.last_image_time is not None:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.last_image_time
            if elapsed > self.image_timeout:
                self.get_logger().warn(
                    f"No images received for {elapsed:.1f}s (timeout={self.image_timeout}s)")

    def check_camera_resolution(self, color_image: np.ndarray, depth_image: np.ndarray) -> bool:
        """Check if color and depth images have matching resolutions."""
        if color_image is None or depth_image is None:
            return False
        return color_image.shape[:2] == depth_image.shape[:2]

    def make_depth_vis(self, depth: np.ndarray) -> Optional[np.ndarray]:
        """Convert raw depth to a colorized BGR8 image for debug viewing/publishing."""
        if depth is None:
            return None
        try:
            depth_f32 = np.array(depth, dtype=np.float32)
            depth_f32 = np.nan_to_num(depth_f32, nan=0.0, posinf=0.0, neginf=0.0)

            # Detect likely units (heuristic): if max > 100, assume mm
            m = float(np.max(depth_f32)) if depth_f32.size else 0.0
            if m > 1000.0:  # looks like millimeters
                depth_f32 = depth_f32 / 1000.0
            m = float(np.max(depth_f32)) if depth_f32.size else 1.0
            if m <= 0.0:
                return None

            norm = (depth_f32 / m * 255.0).astype(np.uint8)
            return cv2.applyColorMap(norm, cv2.COLORMAP_JET)  # BGR8
        except Exception as e:
            self.get_logger().error(f"Depth visualization failed: {e}")
            return None

    def get_depth_in_region(self, centroid_x: float, centroid_y: float, 
                           box_width: float, box_height: float, 
                           region_scale: float = 0.1) -> Optional[float]:
        """Get average depth value in a region around the centroid."""
        if self.depth_image is None:
            return None

        # Calculate region dimensions
        region_width = max(5, int(box_width * region_scale))
        region_height = max(5, int(box_height * region_scale))

        # Calculate region bounds
        x_start = max(0, int(centroid_x - region_width / 2))
        y_start = max(0, int(centroid_y - region_height / 2))
        x_end = min(self.depth_image.shape[1], x_start + region_width)
        y_end = min(self.depth_image.shape[0], y_start + region_height)

        if x_start >= x_end or y_start >= y_end:
            self.get_logger().warn(f"Invalid region coordinates ({x_start}, {y_start}, {x_end}, {y_end}).")
            return None

        # Extract region and get valid depth values
        depth_roi = self.depth_image[y_start:y_end, x_start:x_end]
        valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]

        return np.mean(valid_depths) / 1000.0 if valid_depths.size > 0 else None

    def generate_dark_color(self) -> Tuple[int, int, int]:
        """Generate a dark color for visualization."""
        while True:
            color = (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            if brightness < 130:
                return color

    def process_images(self):
        """Process both color and depth images when available."""
        if self.color_image is not None and self.depth_image is not None:
            # Check if resolution matches
            if self.check_camera_resolution(self.color_image, self.depth_image) or self.camera_type == "pepper":
                # Process the image with the object detection algorithm
                frame = self.color_image.copy()
                boxes, scores, class_ids = self.detect_object(frame) if hasattr(self, 'detect_object') else ([], [], [])
                
                # Filter detections by target classes
                if len(boxes) > 0 and len(self.target_class_indices) < len(COCO_CLASSES):
                    mask = np.array([cid in self.target_class_indices for cid in class_ids])
                    boxes = boxes[mask]
                    scores = scores[mask]
                    class_ids = class_ids[mask]
                
                # If boxes are returned, process them with ByteTrack
                if hasattr(self, 'tracker') and len(boxes) > 0:
                    tracked_objects = self.update_tracker(boxes, scores, class_ids)
                    tracking_data = self.prepare_tracking_data(tracked_objects)
                    self.latest_frame = self.draw_tracked_objects(frame, tracked_objects, tracking_data)
                    self.update_latest_frame(self.latest_frame)
                    
                    # Publish tracking data
                    self.publish_object_detection(tracking_data)
                else:
                    self.update_latest_frame(frame)
            else:
                self.get_logger().warn(f"{self.node_name}: Color and depth image resolutions do not match")

    def prepare_tracking_data(self, tracked_objects: List[Dict]) -> List[Dict]:
        """Prepare tracking data for publishing."""
        tracking_data = []
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            width = x2 - x1
            height = y2 - y1
            centroid_x = (x1 + x2) / 2
            centroid_y = (y1 + y2) / 2
            
            # Get depth at the centroid or in the region
            depth = self.get_depth_in_region(centroid_x, centroid_y, width, height)
            
            point = Point(x=float(centroid_x), y=float(centroid_y), z=float(depth) if depth else 0.0)
            
            class_id = int(obj['class_id'])
            tracking_data.append({
                'track_id': str(int(obj['track_id'])),
                'class_id': class_id,
                'class_name': COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else 'unknown',
                'confidence': float(obj['confidence']),
                'centroid': point,
                'width': float(width),
                'height': float(height)
            })
        
        return tracking_data

    def publish_object_detection(self, tracking_data: List[Dict]):
        """Publish the detected objects to the topic."""
        if not tracking_data:
            return
            
        object_msg = ObjectDetection()
        object_msg.object_label_id = [data['track_id'] for data in tracking_data]
        object_msg.class_names = [data['class_name'] for data in tracking_data]
        object_msg.class_ids = [data['class_id'] for data in tracking_data]
        object_msg.confidences = [data['confidence'] for data in tracking_data]
        object_msg.centroids = [data['centroid'] for data in tracking_data]
        object_msg.width = [data['width'] for data in tracking_data]
        object_msg.height = [data['height'] for data in tracking_data]
        
        self.pub_objects.publish(object_msg)

    def draw_tracked_objects(self, frame: np.ndarray, tracked_objects: List[Dict], 
                            tracking_data: List[Dict]) -> np.ndarray:
        """Draw bounding boxes for each tracked object."""
        output_img = frame.copy()
        for i, obj in enumerate(tracked_objects):
            x1, y1, x2, y2 = obj['bbox']
            track_id = int(obj['track_id'])
            class_id = int(obj['class_id'])
            confidence = obj['confidence']
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else 'unknown'
            
            # Assign a unique color for each track ID
            if track_id not in self.object_colors:
                self.object_colors[track_id] = self.generate_dark_color()
            
            color = self.object_colors[track_id]
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(output_img, p1, p2, color, 2)

            # Label with class name, ID, and confidence
            label_str = f"{class_name} #{track_id} ({confidence:.2f})"
            cv2.putText(output_img, label_str, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Use depth from tracking data if available
            depth = None
            for data in tracking_data:
                if data['track_id'] == str(track_id):
                    depth = data['centroid'].z
                    break
                
            # Format and display depth info
            if depth is not None and depth > 0:
                depth_str = f"Depth: {depth:.2f} m"
            else:
                depth_str = "Depth: Unknown"
                
            cv2.putText(output_img, depth_str, (int(x1), int(y2) + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
        return output_img

    def cleanup(self):
        """Clean up resources."""
        try:
            cv2.destroyAllWindows()
            self.get_logger().info("Cleanup completed")
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")

class YOLOv11(PersonDetectionNode):
    def __init__(self, config: Dict):
        """
        Initializes the ROS2 node, loads configuration, and subscribes to necessary topics.
        """
        super().__init__(config)
        
        # Get YOLOv11 specific configuration values with defaults
        self.confidence_threshold = self.config.get('confidenceThreshold', 0.5)
        
        # ByteTrack parameters
        self.track_thresh = self.config.get('trackThreshold', 0.45)
        self.track_buffer = self.config.get('trackBuffer', 30)
        self.match_thresh = self.config.get('matchThreshold', 0.8)
        self.frame_rate = self.config.get('frameRate', 30)

        # Initialize model
        if not self.init_model():
            self.get_logger().error("Failed to initialize ONNX model")
            rclpy.shutdown()
            return

        # Initialize ByteTrack tracker
        if not self.init_tracker():
            self.get_logger().error("Failed to initialize ByteTrack tracker")
            rclpy.shutdown()
            return

        if self.verbose_mode:
            self.get_logger().info(f"{self.node_name}: Person Detection YOLOv11 node initialized with ByteTrack")
        
        # Subscribe to topics
        if not self.subscribe_topics():
            self.get_logger().error("Failed to subscribe to topics")
            return

        # Start timeout monitor
        self.start_timeout_monitor()

    def init_tracker(self) -> bool:
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=self.frame_rate
            )
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to initialize ByteTrack: {e}")
            return False

    def init_model(self) -> bool:
        """Loads the ONNX model and prepares the runtime session."""
        try:
            so = onnxruntime.SessionOptions()
            so.intra_op_num_threads = multiprocessing.cpu_count()
            so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

            try:
                package_path = get_package_share_directory('person_detection')
                model_path = os.path.join(package_path, 'models', 'person_detection_yolov11m.onnx')
            except Exception as e:
                self.get_logger().error(f"Failed to get package path: {e}")
                return False

            if not os.path.exists(model_path):
                self.get_logger().error(f"Model file not found: {model_path}")
                return False

            self.session = onnxruntime.InferenceSession(
                model_path, 
                sess_options=so, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

            active_providers = self.session.get_providers()
            if self.verbose_mode:
                self.get_logger().info(f"{self.node_name}: Active providers: {active_providers}")
            if "CUDAExecutionProvider" not in active_providers:
                self.get_logger().warn(f"{self.node_name}: CUDAExecutionProvider is not available. Running on CPU may slow down inference.")
            else:
                if self.verbose_mode:
                    self.get_logger().info(f"{self.node_name}: CUDAExecutionProvider is active. Running on GPU for faster inference.")

            input_shape = self.session.get_inputs()[0].shape  # [N, C, H, W]
            self.input_height, self.input_width = input_shape[2], input_shape[3]

            # Warmup run to load model weights into memory
            dummy_input = np.zeros(
                [1 if d is None or isinstance(d, str) else d for d in input_shape],
                dtype=np.float32
            )
            self.session.run(
                [o.name for o in self.session.get_outputs()],
                {self.session.get_inputs()[0].name: dummy_input}
            )

            if self.verbose_mode:
                self.get_logger().info(f"{self.node_name}: ONNX model loaded successfully.")

            return True
        except Exception as e:
            self.get_logger().error(f"{self.node_name}: Failed to initialize ONNX model: {e}")
            return False

    def update_tracker(self, boxes: np.ndarray, scores: np.ndarray, 
              class_ids: np.ndarray) -> List[Dict]:
        try:
            detections = sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=class_ids.astype(int)
            )
            
            tracked_detections = self.tracker.update_with_detections(detections)
            
            tracked_objects = []
            for i in range(len(tracked_detections)):
                track_id = tracked_detections.tracker_id[i]
                
                # Skip detections without assigned track IDs
                if track_id is None:
                    continue
                    
                tracked_objects.append({
                    'bbox': tracked_detections.xyxy[i],
                    'track_id': track_id,
                    'class_id': tracked_detections.class_id[i],
                    'confidence': tracked_detections.confidence[i]
                })
            
            return tracked_objects
            
        except Exception as e:
            self.get_logger().error(f"Error updating tracker: {e}")
            return []

    def detect_object(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares the image and runs inference on the ONNX model.
        
        Returns:
            (boxes, scores, class_ids)
            - boxes in shape Nx4 (xyxy format)
            - scores in shape N
            - class_ids in shape N
        """
        model_input = self.prepare_input(image)
        outputs = self.session.run(
            [o.name for o in self.session.get_outputs()],
            {self.session.get_inputs()[0].name: model_input}
        )
        return self.process_output(outputs)

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Converts the image to RGB, resizes, normalizes, and transposes it for inference."""
        self.orig_height, self.orig_width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height)).astype(np.float32)
        resized /= 255.0
        return resized.transpose(2, 0, 1)[None]

    def process_output(self, model_output) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interprets the raw model output to filter boxes, scores, classes, 
        apply NMS, and return detections.
        
        YOLOv11 output format: [1, 84, 8400] where:
        - 84 = 4 box coordinates + 80 class scores
        - 8400 = number of predictions (grid cells)
        - Box coordinates are: cx, cy, w, h (normalized to 0-1)
        - Class scores are logits that need sigmoid activation
        """
        # Output shape: [1, 84, 8400]
        output = model_output[0]
        
        # Transpose to [8400, 84] - each row is one prediction
        predictions = output[0].T  # [8400, 84]
        
        # Split into box coordinates and class scores
        boxes = predictions[:, :4]  # [8400, 4] - cx, cy, w, h (normalized 0-1)
        class_scores = predictions[:, 4:]  # [8400, 80] - class logits
        
        # Apply sigmoid to class scores to get probabilities
        class_probs = 1 / (1 + np.exp(-class_scores))
        
        # Get max class probability for each prediction
        conf_scores = np.max(class_probs, axis=1)  # [8400]
        
        # Filter by confidence threshold
        mask = conf_scores > self.confidence_threshold
        boxes = boxes[mask]
        class_probs = class_probs[mask]
        conf_scores = conf_scores[mask]
        
        if not len(conf_scores):
            return np.array([]), np.array([]), np.array([])
        
        # Get class IDs
        class_ids = np.argmax(class_probs, axis=1)  # [N]
        
        # Convert normalized box coordinates to pixel coordinates
        # boxes are in format: cx, cy, w, h (normalized 0-1)
        boxes = self.rescale_boxes(boxes)
        boxes = self.xywh2xyxy(boxes)
        
        # Apply NMS
        keep_idx = self.multiclass_nms(boxes, conf_scores, class_ids, self.confidence_threshold)
        boxes = boxes[keep_idx]
        conf_scores = conf_scores[keep_idx]
        class_ids = class_ids[keep_idx]
        
        return boxes, conf_scores, class_ids

    def rescale_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Convert from model input scale to the original image scale."""
        scale = np.array([
            self.orig_width / self.input_width,
            self.orig_height / self.input_height,
            self.orig_width / self.input_width,
            self.orig_height / self.input_height
        ], dtype=np.float32)
        boxes *= scale
        return boxes

    def xywh2xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert [x_center, y_center, w, h] -> [x1, y1, x2, y2]."""
        x, y, w, h = [boxes[:, i].copy() for i in range(4)]
        boxes[:, 0] = x - w / 2
        boxes[:, 1] = y - h / 2
        boxes[:, 2] = x + w / 2
        boxes[:, 3] = y + h / 2
        return boxes

    def multiclass_nms(self, boxes: np.ndarray, scores: np.ndarray, 
                       class_ids: np.ndarray, iou_threshold: float) -> List[int]:
        """Perform NMS per class, gather kept indices."""
        final_keep = []
        for cid in np.unique(class_ids):
            idx = np.where(class_ids == cid)[0]
            keep = self.nms(boxes[idx], scores[idx], iou_threshold)
            final_keep.extend(idx[k] for k in keep)
        return final_keep

    def nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Single-class NMS."""
        sorted_idx = np.argsort(scores)[::-1]
        keep = []
        while len(sorted_idx):
            curr = sorted_idx[0]
            keep.append(curr)
            if len(sorted_idx) == 1:
                break
            ious = self.compute_iou(boxes[curr], boxes[sorted_idx[1:]])
            sorted_idx = sorted_idx[1:][ious < iou_threshold]
        return keep

    def compute_iou(self, main_box: np.ndarray, other_boxes: np.ndarray) -> np.ndarray:
        """IoU between one box and an array of boxes."""
        x1 = np.maximum(main_box[0], other_boxes[:, 0])
        y1 = np.maximum(main_box[1], other_boxes[:, 1])
        x2 = np.minimum(main_box[2], other_boxes[:, 2])
        y2 = np.minimum(main_box[3], other_boxes[:, 3])

        inter_w = np.maximum(0, x2 - x1)
        inter_h = np.maximum(0, y2 - y1)
        inter_area = inter_w * inter_h

        box_area = (main_box[2] - main_box[0]) * (main_box[3] - main_box[1])
        other_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

        return inter_area / (box_area + other_area - inter_area + 1e-6)
