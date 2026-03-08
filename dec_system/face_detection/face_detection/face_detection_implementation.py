"""
face_detection_implementation.py Implementation code for running the Face and Mutual Gaze Detection and Localization ROS2 node.

Author: Yohannes Tadesse Haile
Date: April 18, 2025
Version: v1.0

This program comes with ABSOLUTELY NO WARRANTY.
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
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from math import cos, sin, pi
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from typing import Tuple, List, Dict, Optional
from dec_interfaces.msg import FaceDetection, PersonDetection
from scipy.optimize import linear_sum_assignment
import supervision as sv

def load_configuration() -> Dict:
    """
    Load configuration from the default YAML file location.
    
    Returns:
        Dict: Configuration data with defaults for missing values
    """
    config = {
        # Default values
        'algorithm': 'sixdrep',
        'useCompressed': False,
        'camera': 'realsense',
        'verboseMode': True,
        'imageTimeout': 2.0,
        'sixdrepnetConfidence': 0.65,
        'sixdrepnetHeadposeAngle': 10,
        'requirePersonDetection': True,
        'personDetectionTimeout': 0.5,  # Max age for person detection data
        'prioritizeFaceDepth': True  # Prioritize face depth over person depth
    }
    
    try:
        package_path = get_package_share_directory('face_detection')
        config_file = os.path.join(package_path, 'config', 'face_detection_configuration.yaml')
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as file:
                file_config = yaml.safe_load(file) or {}
                config.update(file_config)  # Update defaults with file values
                print(f"Loaded configuration from {config_file}")
        else:
            print(f"Warning: Configuration file not found at {config_file}, using defaults")
            
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        print("Using default configuration values")
        
    return config

class FaceDetectionNode(Node):
    def __init__(self, config: Dict, node_name: str = 'faceDetection'):
        super().__init__(node_name)
        
        self.config = config
        self.pub_gaze = self.create_publisher(FaceDetection, "/faceDetection/data", 10)
        self.debug_pub = self.create_publisher(Image, "/faceDetection/debug", 1)
        self.depth_debug_pub = self.create_publisher(Image, "/faceDetection/depth_debug", 1)

        self.bridge = CvBridge()
        self.depth_image: Optional[np.ndarray] = None
        self.color_image: Optional[np.ndarray] = None
        
        # Configuration values
        self.use_compressed = config['useCompressed']
        self.camera_type = config['camera']
        self.verbose_mode = config['verboseMode']
        self.image_timeout = config['imageTimeout']
        self.require_person_detection = config.get('requirePersonDetection', True)
        self.person_detection_timeout = config.get('personDetectionTimeout', 0.5)
        self.prioritize_face_depth = config.get('prioritizeFaceDepth', True)
        
        self.node_name = self.get_name()
        self.timer_start = self.get_clock().now()
        self.last_image_time = None   # timestamp of the last received image

        # Thread safety for visualization (protects latest_frame and latest_depth)
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None

        # Person detection related attributes
        self.latest_person_detections = None
        self.latest_person_detections_timestamp = None
        self.person_detections_lock = threading.Lock()

        # Face colors keyed by person tracking ID (from person detection)
        self.face_colors: Dict[str, Tuple[int, int, int]] = {}
        
        # ByteTrack tracker for standalone mode face tracking
        self.face_tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=30,
            minimum_matching_threshold=0.3,
            frame_rate=15
        )
        
        # Only subscribe to person detection if required
        if self.require_person_detection:
            self.person_detection_sub = self.create_subscription(PersonDetection, '/personDetection/data', self.person_detection_callback, 10)

            if self.verbose_mode:
                self.get_logger().info(f"{self.node_name}: Person detection ENABLED - subscribed to /personDetection/data")
                self.get_logger().info(f"{self.node_name}: Using person detection tracking IDs for face identification")
        else:
            if self.verbose_mode:
                self.get_logger().info(f"{self.node_name}: Person detection DISABLED - running standalone face detection")

        # Start visualization timer (30 Hz)
        self.vis_timer = self.create_timer(1.0 / 30.0, self.visualization_callback)

    def update_latest_frame(self, frame: np.ndarray):
        """Update the latest frame and depth safely for visualization."""
        with self.frame_lock:
            self.latest_frame = frame.copy()
            self.latest_depth = self.depth_image.copy() if self.depth_image is not None else None

    def visualization_callback(self):
        """Timer callback for showing or publishing debug images."""
        color_frame = depth_vis = None
        with self.frame_lock:
            if self.latest_frame is not None:
                color_frame = self.latest_frame.copy()
            if self.latest_depth is not None:
                depth_vis = self.make_depth_vis(self.latest_depth)

        if color_frame is None and depth_vis is None:
            return

        if self.config.get("verboseMode", False) and os.environ.get("DISPLAY", "") != "":
            try:
                if color_frame is not None:
                    cv2.imshow("Face Detection Debug (RGB)", color_frame)
                if depth_vis is not None:
                    cv2.imshow("Face Detection Debug (Depth)", depth_vis)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn(f"imshow failed (likely headless): {e}")

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
        }
        
        if self.camera_type not in topic_mapping:
            raise ValueError(f"Invalid camera type: {self.camera_type}")
            
        rgb_key, depth_key = topic_mapping[self.camera_type]
        rgb_topic = self.extract_topic(rgb_key)
        depth_topic = self.extract_topic(depth_key)
        
        if not rgb_topic or not depth_topic:
            raise ValueError("Failed to extract camera topics")
            
        return rgb_topic, depth_topic

    def extract_topic(self, image_topic:str) -> Optional[str]:
        """Extract topic name from configuration file"""
        try:
            package = get_package_share_directory('face_detection')
            config_path = os.path.join(package, 'data', 'pepper_topics.yaml')

            with open(config_path, 'r') as file:
                topics = yaml.safe_load(file)
                return topics.get(image_topic)
        except Exception as e:
            self.get_logger().error(f"Error extracting topic '{image_topic}': {e}")

        return None

    def wait_for_topics(self, color_topic: str, depth_topic: str, person_detection_topic: Optional[str] = None, timeout: float = 30.0) -> bool:
        """Wait for topics to have active publishers."""
        topics_to_wait = [color_topic, depth_topic]
        if person_detection_topic and self.require_person_detection:
            topics_to_wait.append(person_detection_topic)
            
        self.get_logger().info(f"Waiting for topics: {topics_to_wait}")
        
        start_time = self.get_clock().now()
        warning_interval = 5.0
        last_warning_time = start_time
        
        while rclpy.ok():
            # Check publisher count for each topic
            missing_topics = []
            for topic in topics_to_wait:
                pub_count = self.count_publishers(topic)
                if pub_count == 0:
                    missing_topics.append(topic)
            
            if not missing_topics:
                if self.verbose_mode:
                    self.get_logger().info("All topics have active publishers!")
                return True
            
            # Check timeout
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > timeout:
                self.get_logger().error(f"Timeout waiting for publishers after {timeout}s. Missing: {missing_topics}")
                return False
            
            # Periodic warnings
            if (self.get_clock().now() - last_warning_time).nanoseconds / 1e9 >= warning_interval:
                self.get_logger().warn(f"Still waiting for publishers after {int(elapsed)}s: {missing_topics}")
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

            # Wait for topics (conditionally wait for person detection)
            person_detection_topic = "/personDetection/data" if self.require_person_detection else None
            if not self.wait_for_topics(color_topic, depth_topic, person_detection_topic):
                return False

            # Create subscribers
            self.color_sub = Subscriber(self, color_msg_type, color_topic, qos_profile=qos_profile_sensor_data)
            self.depth_sub = Subscriber(self, depth_msg_type, depth_topic, qos_profile=qos_profile_sensor_data)

            self.get_logger().info(f"Subscribed to {color_topic}")
            self.get_logger().info(f"Subscribed to {depth_topic}")
            if self.require_person_detection:
                self.get_logger().info(f"Subscribed to {person_detection_topic}")

            # Set up synchronizer
            slop = 5.0 if self.camera_type == "pepper" else 0.1
            self.ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=slop)
            self.ats.registerCallback(self.synchronized_callback)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to setup subscribers: {e}")
            return False

    def synchronized_callback(self, color_data, depth_data):
        """Process synchronized color and depth images."""
        self.last_image_time = self.get_clock().now().nanoseconds / 1e9
        
        try:
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

            # Check resolution match on the current frame
            if self.camera_type != "pepper" and not self.check_camera_resolution(self.color_image, self.depth_image):
                self.get_logger().error(f"{self.node_name}: Color camera and depth camera have different resolutions.")
                rclpy.shutdown()
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
                        
                        # Extract header values if needed
                        header_data = np.frombuffer(depth_data.data[:depth_header_size], np.float32)
     
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
        self.create_timer(1.0, self.check_timeout)

    def check_timeout(self):
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
            if m > 1000.0:             # looks like millimeters
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
        """Get median depth value in a region around the centroid."""
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
            return None

        # Extract region and get valid depth values
        depth_roi = self.depth_image[y_start:y_end, x_start:x_end]
        valid_depths = depth_roi[np.isfinite(depth_roi) & (depth_roi > 0)]

        return np.median(valid_depths) / 1000.0 if valid_depths.size > 0 else None

    def generate_dark_color(self) -> Tuple[int, int, int]:
        """Generate a dark color for visualization."""
        while True:
            color = (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            if brightness < 130:
                return color

    def publish_face_detection(self, tracking_data: List[Dict]):
        """Publish face detection results."""
        if not tracking_data:
            # Publish empty message
            face_msg = FaceDetection()
            self.pub_gaze.publish(face_msg)
            return

        face_msg = FaceDetection()
        face_msg.face_label_id = [str(d['face_id']) for d in tracking_data]
        face_msg.centroids = [d['centroid'] for d in tracking_data]
        face_msg.width = [float(d['width']) for d in tracking_data]
        face_msg.height = [float(d['height']) for d in tracking_data]
        face_msg.mutual_gaze = [bool(d['mutual_gaze']) for d in tracking_data]

        self.pub_gaze.publish(face_msg)

    def person_detection_callback(self, msg: PersonDetection):
        """Callback for person detection messages. Thread-safe storage with timestamp."""
        with self.person_detections_lock:
            self.latest_person_detections = {
                'person_label_id': list(msg.person_label_id),
                'class_names': list(msg.class_names),
                'centroids': [Point(x=c.x, y=c.y, z=c.z) for c in msg.centroids],
                'width': list(msg.width),
                'height': list(msg.height)
            }
            self.latest_person_detections_timestamp = self.get_clock().now()

    def cleanup(self):
        """Clean up resources."""
        try:
            cv2.destroyAllWindows()
            self.get_logger().info("Cleanup completed")
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")

class YOLOONNX:
    def __init__(self, model_path: str, class_score_th: float = 0.65,
        providers: List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider']):
        self.class_score_th = class_score_th
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        
        # Optimize ONNX Runtime session options
        session_option.intra_op_num_threads = multiprocessing.cpu_count()
        session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.onnx_session = onnxruntime.InferenceSession(
            model_path, sess_options=session_option, providers=providers)
        
        self.input_shape = self.onnx_session.get_inputs()[0].shape
        self.input_names = [inp.name for inp in self.onnx_session.get_inputs()]
        self.output_names = [out.name for out in self.onnx_session.get_outputs()]

        # Warmup run to load model weights into memory
        dummy_input = np.zeros(
            [1 if d is None or isinstance(d, str) else d for d in self.input_shape],
            dtype=np.float32
        )
        self.onnx_session.run(
            self.output_names,
            {name: dummy_input for name in self.input_names},
        )

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        resized_image = self.preprocess(image)
        inference_image = resized_image[np.newaxis, ...].astype(np.float32)
        boxes = self.onnx_session.run(
            self.output_names,
            {name: inference_image for name in self.input_names},
        )[0]
        return self.postprocess(image, boxes)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        resized_image = resized_image[:, :, ::-1] / 255.0  # BGR to RGB and normalize
        return resized_image.transpose(2, 0, 1)  # HWC to CHW

    def postprocess(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_h, img_w = image.shape[:2]
        result_boxes = []
        result_scores = []
        if boxes.size > 0:
            scores = boxes[:, 6]
            keep_idxs = scores > self.class_score_th
            boxes_keep = boxes[keep_idxs]
            for box in boxes_keep:
                x_min = int(max(box[2], 0) * img_w / self.input_shape[3])
                y_min = int(max(box[3], 0) * img_h / self.input_shape[2])
                x_max = int(min(box[4], self.input_shape[3]) * img_w / self.input_shape[3])
                y_max = int(min(box[5], self.input_shape[2]) * img_h / self.input_shape[2])
                result_boxes.append([x_min, y_min, x_max, y_max])
                result_scores.append(box[6])
        return np.array(result_boxes), np.array(result_scores)

class SixDrepNet(FaceDetectionNode):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Get SixDrepNet specific configuration values with defaults
        sixdrepnet_confidence = self.config.get('sixdrepnetConfidence', 0.65)
        self.sixdrep_angle = self.config.get('sixdrepnetHeadposeAngle', 10)
        
        if self.verbose_mode:
            self.get_logger().info(f"{self.node_name}: Initializing SixDrepNet ...")

        # Set up model paths
        try:
            package_path = get_package_share_directory('face_detection')
            yolo_model_path = os.path.join(package_path, 'models/face_detection_goldYOLO.onnx')
            sixdrepnet_model_path = os.path.join(package_path, 'models/face_detection_sixdrepnet360.onnx')
        except Exception as e:
            self.get_logger().error(f"{self.node_name}: Failed to get package path: {e}")
            return

        # Initialize YOLOONNX model for face detection
        try:
            self.yolo_model = YOLOONNX(model_path=yolo_model_path, class_score_th=sixdrepnet_confidence)
            if self.verbose_mode:
                self.get_logger().info(f"{self.node_name}: YOLOONNX model initialized successfully.")
        except Exception as e:
            self.yolo_model = None
            self.get_logger().error(f"{self.node_name}: Failed to initialize YOLOONNX model: {e}")
            return

        # Initialize SixDrepNet ONNX session
        try:
            session_option = onnxruntime.SessionOptions()
            session_option.log_severity_level = 3
            session_option.intra_op_num_threads = multiprocessing.cpu_count()
            session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sixdrepnet_session = onnxruntime.InferenceSession(
                sixdrepnet_model_path,
                sess_options=session_option,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )

            # Warmup run to load SixDrepNet weights into memory
            dummy_sixdrep = np.zeros((1, 3, 224, 224), dtype=np.float32)
            self.sixdrepnet_session.run(None, {'input': dummy_sixdrep})

            active_providers = self.sixdrepnet_session.get_providers()
            if self.verbose_mode:
                self.get_logger().info(f"{self.node_name}: Active providers: {active_providers}")
            if "CUDAExecutionProvider" not in active_providers:
                if self.verbose_mode:
                    self.get_logger().warn(f"{self.node_name}: CUDAExecutionProvider is not available. Running on CPU may slow down inference.")
            else:
                if self.verbose_mode:
                    self.get_logger().info(f"{self.node_name}: CUDAExecutionProvider is active. Running on GPU for faster inference.")
        except Exception as e:
            self.get_logger().error(f"{self.node_name}: Failed to initialize SixDrepNet ONNX session: {e}")
            return

        # Set up remaining attributes
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if self.verbose_mode:
            self.get_logger().info(f"{self.node_name} SixDrepNet initialization complete.")

        # Subscribe to topics
        if not self.subscribe_topics():
            self.get_logger().error(f"{self.node_name}: Failed to subscribe to required topics. Shutting down.")
            return

        self.start_timeout_monitor()
    
    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
        pitch = pitch * pi / 180
        yaw = -yaw * pi / 180
        roll = roll * pi / 180
        height, width = img.shape[:2]
        tdx = tdx if tdx is not None else width / 2
        tdy = tdy if tdy is not None else height / 2

        x1 = size * (cos(yaw) * cos(roll)) + tdx
        y1 = size * (cos(pitch) * sin(roll) + sin(pitch) * sin(yaw) * cos(roll)) + tdy
        x2 = size * (-cos(yaw) * sin(roll)) + tdx
        y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy
        x3 = size * sin(yaw) + tdx
        y3 = size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    def calculate_matching_cost(self, face: Dict, person: Dict, img_h: int, img_w: int) -> float:
        """
        Calculate the cost of matching a face to a person.
        Lower cost means better match.
        
        Args:
            face: Dictionary containing face information
            person: Dictionary containing person information
            img_h: Image height
            img_w: Image width
            
        Returns:
            float: Cost value (lower is better, np.inf if impossible match)
        """
        face_cx, face_cy = face['centroid']
        px1, py1, px2, py2 = person['box']
        
        # Check if face centroid is inside person box
        if not (px1 <= face_cx <= px2 and py1 <= face_cy <= py2):
            return np.inf  # Impossible match
        
        # Calculate person dimensions and center
        person_cx = (px1 + px2) / 2
        person_cy = (py1 + py2) / 2
        person_width = px2 - px1
        person_height = py2 - py1
        person_area = person_width * person_height
        
        if person_area == 0:
            return np.inf
        
        # 1. Calculate normalized distance from face to person center
        distance = np.sqrt((face_cx - person_cx)**2 + (face_cy - person_cy)**2)
        normalized_distance = distance / np.sqrt(person_area)
        
        # 2. Calculate vertical position penalty (faces should be in upper portion)
        person_upper_third_y = py1 + person_height * 0.33
        if face_cy < person_upper_third_y:
            vertical_penalty = 0.0
        elif face_cy < py1 + person_height * 0.5:
            vertical_penalty = 0.5
        else:
            vertical_penalty = 2.0
        
        # 3. Size consistency check
        face_height = face['size'][1]
        expected_face_height_ratio = face_height / person_height
        if 0.1 <= expected_face_height_ratio <= 0.4:
            size_penalty = 0.0
        else:
            size_penalty = 1.0
        
        # 4. Detection confidence
        confidence_cost = 1.0 - face['score']
        
        # Combine costs
        total_cost = (
            normalized_distance * 2.0 +
            vertical_penalty * 1.5 +
            size_penalty * 1.0 +
            confidence_cost * 0.5
        )
        
        return total_cost

    def match_faces_to_persons_hungarian(self, faces: List[Dict], persons: List[Dict], 
                                         img_h: int, img_w: int) -> List[Tuple[int, int]]:
        """
        Use Hungarian algorithm to optimally match faces to persons.
        Handles multiple faces per person by allowing multiple matches.
        
        Args:
            faces: List of face detections
            persons: List of person detections
            img_h: Image height
            img_w: Image width
            
        Returns:
            List of tuples (face_idx, person_idx) for matched pairs
        """
        if not faces or not persons:
            return []
        
        # Build cost matrix
        n_faces = len(faces)
        n_persons = len(persons)
        cost_matrix = np.full((n_faces, n_persons), np.inf)
        
        for f_idx, face in enumerate(faces):
            for p_idx, person in enumerate(persons):
                cost_matrix[f_idx, p_idx] = self.calculate_matching_cost(face, person, img_h, img_w)
        
        # Apply Hungarian algorithm
        try:
            face_indices, person_indices = linear_sum_assignment(cost_matrix)
            
            # Filter out impossible matches and collect valid ones
            matches = []
            for f_idx, p_idx in zip(face_indices, person_indices):
                if cost_matrix[f_idx, p_idx] < np.inf:
                    matches.append((f_idx, p_idx))
            
            # Additional pass: check for multiple faces in same person box
            # that weren't matched in the first pass
            matched_face_indices = set(f_idx for f_idx, _ in matches)
            for f_idx, face in enumerate(faces):
                if f_idx in matched_face_indices:
                    continue
                
                # Find best person box this face falls into
                best_cost = np.inf
                best_person_idx = None
                for p_idx, person in enumerate(persons):
                    cost = self.calculate_matching_cost(face, person, img_h, img_w)
                    if cost < best_cost and cost < np.inf:
                        best_cost = cost
                        best_person_idx = p_idx
                
                if best_person_idx is not None:
                    matches.append((f_idx, best_person_idx))
                    if self.verbose_mode:
                        self.get_logger().info(
                            f"Additional face {f_idx} matched to person {persons[best_person_idx]['id']} "
                            f"(multiple faces in same box)"
                        )
            
            return matches
        except Exception as e:
            self.get_logger().error(f"Hungarian algorithm failed: {e}")
            return []

    def get_best_depth_estimate(self, face_cx: float, face_cy: float, 
                                face_width: float, face_height: float,
                                person_depth: Optional[float]) -> float:
        """
        Get best depth estimate, prioritizing face-region depth over person depth.
        
        Args:
            face_cx, face_cy: Face centroid coordinates
            face_width, face_height: Face dimensions
            person_depth: Depth from person detection (may be None or 0)
            
        Returns:
            float: Best depth estimate in meters
        """
        # Try to get face-region depth first
        face_depth = self.get_depth_in_region(face_cx, face_cy, face_width, face_height)
        
        if self.prioritize_face_depth:
            # Prioritize face depth, use person depth as fallback
            if face_depth is not None and face_depth > 0:
                return face_depth
            elif person_depth is not None and person_depth > 0:
                return person_depth
            else:
                return 0.0
        else:
            # Use person depth if available, otherwise face depth
            if person_depth is not None and person_depth > 0:
                return person_depth
            elif face_depth is not None and face_depth > 0:
                return face_depth
            else:
                return 0.0

    def process_images(self):
        """Process synchronized RGB + depth images for SixDrepNet."""
        if self.color_image is None or self.depth_image is None:
            return

        if self.require_person_detection:
            frame = self.process_frame_with_person_detection(self.color_image)
        else:
            frame = self.process_frame_standalone(self.color_image)
            
        if frame is not None:
            self.update_latest_frame(frame)
    
    def process_frame_standalone(self, cv_image):
        """Process frame with standalone face detection using ByteTrack for persistent IDs."""
        debug_image = cv_image.copy()
        img_h, img_w = debug_image.shape[:2]
        tracking_data = []

        # Run face detection on full image
        face_boxes, face_scores = self.yolo_model(cv_image)

        if self.verbose_mode:
            self.get_logger().info(f"Standalone mode: {len(face_boxes)} faces detected")

        # Filter valid faces
        valid_boxes = []
        valid_scores = []
        for box, score in zip(face_boxes, face_scores):
            fx1, fy1, fx2, fy2 = box
            if (fx2 - fx1) < 20 or (fy2 - fy1) < 20:
                continue
            if not (0 <= fx1 < fx2 <= img_w and 0 <= fy1 < fy2 <= img_h):
                continue
            valid_boxes.append([fx1, fy1, fx2, fy2])
            valid_scores.append(score)

        # Run ByteTrack on face detections
        if valid_boxes:
            detections = sv.Detections(
                xyxy=np.array(valid_boxes, dtype=np.float32),
                confidence=np.array(valid_scores, dtype=np.float32)
            )
            tracked = self.face_tracker.update_with_detections(detections)
        else:
            tracked = sv.Detections.empty()

        active_face_ids = set()
        for i in range(len(tracked)):
            track_id = tracked.tracker_id[i]
            if track_id is None:
                continue

            fx1, fy1, fx2, fy2 = tracked.xyxy[i].astype(int)
            face_cx = (fx1 + fx2) // 2
            face_cy = (fy1 + fy2) // 2
            face_width = fx2 - fx1
            face_height = fy2 - fy1
            face_id = f"face_{track_id}"
            active_face_ids.add(face_id)

            if face_id not in self.face_colors:
                self.face_colors[face_id] = self.generate_dark_color()
            face_color = self.face_colors[face_id]

            # Crop face for head pose estimation
            face_image = cv_image[fy1:fy2, fx1:fx2]
            if face_image.size == 0:
                continue

            # Preprocess for SixDrepNet
            resized_image = cv2.resize(face_image, (224, 224))
            normalized_image = (resized_image[..., ::-1] / 255.0 - self.mean) / self.std
            input_tensor = normalized_image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

            # Run head pose estimation
            try:
                yaw_pitch_roll = self.sixdrepnet_session.run(None, {'input': input_tensor})[0][0]
                yaw_deg, pitch_deg, roll_deg = yaw_pitch_roll
            except Exception as e:
                self.get_logger().error(f"Head pose estimation failed: {e}")
                continue

            # Draw head pose axes
            self.draw_axis(debug_image, yaw_deg, pitch_deg, roll_deg, face_cx, face_cy, size=50)

            # Get depth
            cz = self.get_depth_in_region(face_cx, face_cy, face_width, face_height)
            cz = cz if cz is not None else 0.0

            # Mutual gaze check
            mutual_gaze = abs(yaw_deg) < self.sixdrep_angle and abs(pitch_deg) < self.sixdrep_angle

            tracking_data.append({
                'face_id': face_id,
                'centroid': Point(x=float(face_cx), y=float(face_cy), z=float(cz)),
                'width': float(face_width),
                'height': float(face_height),
                'mutual_gaze': mutual_gaze
            })

            # Draw visualizations
            cv2.rectangle(debug_image, (fx1, fy1), (fx2, fy2), face_color, 2)
            label = "Engaged" if mutual_gaze else "Not Engaged"
            cv2.putText(debug_image, label, (fx1 + 10, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
            cv2.putText(debug_image, f"ID: {face_id}", (fx1 + 10, fy1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
            if cz > 0:
                cv2.putText(debug_image, f"Depth: {cz:.2f}m", (fx1 + 10, fy2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)

        # Clean up colors for tracks that no longer exist
        stale_colors = set(self.face_colors.keys()) - active_face_ids
        for stale_id in stale_colors:
            del self.face_colors[stale_id]

        self.publish_face_detection(tracking_data)
        return debug_image

    def process_frame_with_person_detection(self, cv_image):
        """Process frame with person-constrained face detection using person detection tracking IDs."""
        debug_image = cv_image.copy()
        img_h, img_w = debug_image.shape[:2]
        tracking_data = []

        # Thread-safe retrieval of person detections with timestamp validation
        with self.person_detections_lock:
            if self.latest_person_detections is None or self.latest_person_detections_timestamp is None:
                if self.verbose_mode:
                    self.get_logger().warn("No person detection data available")
                self.publish_face_detection([])
                return debug_image

            # Check if person detection data is too old
            current_time = self.get_clock().now()
            detection_age = (current_time - self.latest_person_detections_timestamp).nanoseconds / 1e9

            if detection_age > self.person_detection_timeout:
                if self.verbose_mode:
                    self.get_logger().warn(
                        f"Person detection data is stale ({detection_age:.2f}s old, "
                        f"timeout={self.person_detection_timeout}s)"
                    )
                self.publish_face_detection([])
                return debug_image

            # Make efficient copy of only needed data
            person_detections = {
                'person_label_id': list(self.latest_person_detections['person_label_id']),
                'class_names': list(self.latest_person_detections['class_names']),
                'centroids': [Point(x=c.x, y=c.y, z=c.z) for c in self.latest_person_detections['centroids']],
                'width': list(self.latest_person_detections['width']),
                'height': list(self.latest_person_detections['height'])
            }

        # Validate array consistency including person_label_id
        n_detections = len(person_detections['person_label_id'])

        if self.verbose_mode and n_detections > 0:
            self.get_logger().info(
                f"Person detection: {n_detections} persons "
                f"(age: {detection_age:.2f}s)"
            )

        if n_detections == 0:
            if self.verbose_mode:
                self.get_logger().debug("No persons in person detection message")
            self.publish_face_detection([])
            # Clear face colors when no persons detected
            self.face_colors.clear()
            return debug_image

        # Validate all arrays have same length
        if not all(len(arr) == n_detections for arr in [
            person_detections['class_names'],
            person_detections['centroids'],
            person_detections['width'],
            person_detections['height']
        ]):
            self.get_logger().error("Inconsistent person detection array lengths")
            self.publish_face_detection([])
            return debug_image

        # Collect person detections with their tracking IDs
        persons = []
        active_person_ids = set()
        for i in range(n_detections):
            if person_detections['class_names'][i] == 'person':
                centroid = person_detections['centroids'][i]
                width = person_detections['width'][i]
                height = person_detections['height'][i]
                
                x1 = max(0, int(centroid.x - width / 2))
                y1 = max(0, int(centroid.y - height / 2))
                x2 = min(img_w, int(centroid.x + width / 2))
                y2 = min(img_h, int(centroid.y + height / 2))
                
                # Validate bounding box
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                if x2 <= x1 or y2 <= y1 or box_area < 100:
                    if self.verbose_mode:
                        self.get_logger().warn(f"Skipping invalid person box: area={box_area}")
                    continue
                
                if x1 >= img_w or y1 >= img_h or x2 <= 0 or y2 <= 0:
                    if self.verbose_mode:
                        self.get_logger().warn(f"Skipping out-of-frame person box")
                    continue
                
                # Use person detection tracking ID directly
                person_tracking_id = person_detections['person_label_id'][i]
                active_person_ids.add(str(person_tracking_id))
                
                if self.verbose_mode:
                    self.get_logger().info(
                        f"Person tracking_id={person_tracking_id}: box=({x1},{y1},{x2},{y2}), "
                        f"depth={centroid.z:.2f}m"
                    )
                
                persons.append({
                    'tracking_id': person_tracking_id,  # This is the key from person detection
                    'box': (x1, y1, x2, y2),
                    'depth': centroid.z,
                    'assigned_faces': []  # List to support multiple faces per person
                })

        # Clean up stale face colors for persons that are no longer tracked
        stale_ids = set(self.face_colors.keys()) - active_person_ids
        if stale_ids:
            if self.verbose_mode:
                self.get_logger().info(f"Cleaning up colors for lost person IDs: {stale_ids}")
            for stale_id in stale_ids:
                del self.face_colors[stale_id]

        # Early return if no valid persons
        if not persons:
            if self.verbose_mode:
                self.get_logger().info(f"No valid persons detected (total objects: {n_detections})")
            self.publish_face_detection([])
            return debug_image

        # Run face detection
        face_boxes, face_scores = self.yolo_model(cv_image)
        
        if self.verbose_mode:
            self.get_logger().info(f"Face detection: {len(face_boxes)} raw detections")

        # Build list of valid faces
        faces = []
        for idx, (box, score) in enumerate(zip(face_boxes, face_scores)):
            fx1, fy1, fx2, fy2 = box
            face_cx = (fx1 + fx2) // 2
            face_cy = (fy1 + fy2) // 2
            face_width = fx2 - fx1
            face_height = fy2 - fy1
            
            if face_width < 20 or face_height < 20:
                continue
            
            if not (0 <= fx1 < fx2 <= img_w and 0 <= fy1 < fy2 <= img_h):
                continue
                
            faces.append({
                'box': (fx1, fy1, fx2, fy2),
                'centroid': (face_cx, face_cy),
                'size': (face_width, face_height),
                'score': score,
                'original_idx': idx
            })

        if not faces:
            if self.verbose_mode:
                self.get_logger().info(f"No valid faces detected (persons: {len(persons)})")
            self.publish_face_detection([])
            if self.verbose_mode:
                for person in persons:
                    px1, py1, px2, py2 = person['box']
                    cv2.rectangle(debug_image, (px1, py1), (px2, py2), (128, 128, 128), 1)
                    cv2.putText(debug_image, f"P:{person['tracking_id']}", (px1 + 5, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            return debug_image

        # Hungarian matching
        matches = self.match_faces_to_persons_hungarian(faces, persons, img_h, img_w)
        
        if self.verbose_mode:
            self.get_logger().info(f"Hungarian matching: Persons={len(persons)}, Faces={len(faces)}, Matches={len(matches)}")

        if not matches:
            self.publish_face_detection([])
            if self.verbose_mode:
                self.get_logger().warn(f"Faces detected ({len(faces)}) but none matched to persons ({len(persons)})")
                for face in faces:
                    fx1, fy1, fx2, fy2 = face['box']
                    cv2.rectangle(debug_image, (fx1, fy1), (fx2, fy2), (0, 0, 255), 1)
                    cv2.putText(debug_image, f"Unmatched (score={face['score']:.2f})", 
                               (fx1, fy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                for person in persons:
                    px1, py1, px2, py2 = person['box']
                    cv2.rectangle(debug_image, (px1, py1), (px2, py2), (255, 0, 0), 2)
                    cv2.putText(debug_image, f"Person:{person['tracking_id']}", (px1 + 5, py1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            return debug_image

        # Process matched pairs - using person detection tracking IDs
        for face_idx, person_idx in matches:
            face = faces[face_idx]
            person = persons[person_idx]
            
            fx1, fy1, fx2, fy2 = face['box']
            face_cx, face_cy = face['centroid']
            face_width, face_height = face['size']
            
            # Track assigned faces (for multiple faces per person)
            person['assigned_faces'].append(face_idx)
            
            # Use person's tracking ID from person detection directly
            # For multiple faces per person, add suffix
            if len(person['assigned_faces']) == 1:
                face_id = str(person['tracking_id'])
            else:
                face_id = f"{person['tracking_id']}_f{len(person['assigned_faces'])}"
            
            # Assign consistent color based on tracking ID
            base_id = str(person['tracking_id'])
            if base_id not in self.face_colors:
                self.face_colors[base_id] = self.generate_dark_color()
            face_color = self.face_colors[base_id]

            # Crop face
            face_image = cv_image[fy1:fy2, fx1:fx2]
            if face_image.size == 0:
                continue

            # Preprocess for SixDrepNet
            resized_image = cv2.resize(face_image, (224, 224))
            normalized_image = (resized_image[..., ::-1] / 255.0 - self.mean) / self.std
            input_tensor = normalized_image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

            # Head pose estimation
            try:
                yaw_pitch_roll = self.sixdrepnet_session.run(None, {'input': input_tensor})[0][0]
                yaw_deg, pitch_deg, roll_deg = yaw_pitch_roll
            except Exception as e:
                self.get_logger().error(f"Head pose estimation failed: {e}")
                continue

            self.draw_axis(debug_image, yaw_deg, pitch_deg, roll_deg, face_cx, face_cy, size=50)

            # Get best depth estimate (prioritizing face depth)
            cz = self.get_best_depth_estimate(face_cx, face_cy, face_width, face_height, person['depth'])

            # Mutual gaze
            mutual_gaze = abs(yaw_deg) < self.sixdrep_angle and abs(pitch_deg) < self.sixdrep_angle

            tracking_data.append({
                'face_id': face_id,  # This is now based on person detection tracking ID
                'centroid': Point(x=float(face_cx), y=float(face_cy), z=float(cz)),
                'width': float(face_width),
                'height': float(face_height),
                'mutual_gaze': mutual_gaze
            })

            # Draw visualizations
            cv2.rectangle(debug_image, (fx1, fy1), (fx2, fy2), face_color, 2)
            label = "Engaged" if mutual_gaze else "Not Engaged"
            cv2.putText(debug_image, label, (fx1 + 10, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
            cv2.putText(debug_image, f"ID: {face_id}", (fx1 + 10, fy1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
            cv2.putText(debug_image, f"Depth: {cz:.2f}m", (fx1 + 10, fy2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)

        # Draw person boxes
        if self.verbose_mode:
            for person in persons:
                px1, py1, px2, py2 = person['box']
                color = (0, 255, 0) if person['assigned_faces'] else (128, 128, 128)
                cv2.rectangle(debug_image, (px1, py1), (px2, py2), color, 1)
                label = f"P:{person['tracking_id']}"
                if person['assigned_faces']:
                    label += f" ({len(person['assigned_faces'])} faces)"
                cv2.putText(debug_image, label, (px1 + 5, py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        self.publish_face_detection(tracking_data)
        return debug_image