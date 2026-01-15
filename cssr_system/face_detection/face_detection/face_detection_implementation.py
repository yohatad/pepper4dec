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
from rclpy.node import Node
from math import cos, sin, pi
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from typing import Tuple, List, Dict, Optional
from cssr_interfaces.msg import FaceDetection, ObjectDetection

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
        'sixdrepnetHeadposeAngle': 10
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
        
        self.node_name = self.get_name()
        self.timer_start = self.get_clock().now()
        self.last_image_time = None   # timestamp of the last received image

        # Thread safety for visualization
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None

        # Object detection related attributes
        self.latest_object_detections = None
        self.object_detections_lock = threading.Lock()
        
        # Create subscriber for object detection messages
        self.object_detection_sub = self.create_subscription(ObjectDetection, '/objectDetection/data', self.object_detection_callback, 10)
        
        if self.verbose_mode:
            self.get_logger().info(f"{self.node_name}: Subscribed to /objectDetection/data")

        # Start visualization timer (30 Hz)
        self.vis_timer = self.create_timer(1.0 / 30.0, self.visualization_callback)

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
            # depth_image is updated in sync callback; safe to read without copy for view,
            # but copy if you plan heavy ops
            if self.depth_image is not None:
                depth_vis = self.make_depth_vis(self.depth_image)

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

            # Publish only when verboseMode=True (as per your intent)
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

    def wait_for_topics(self, color_topic: str, depth_topic: str, object_detection_topic: str, timeout: float = 30.0) -> bool:
        """Wait for topics to have active publishers."""
        self.get_logger().info(f"Waiting for topics: {color_topic}, {depth_topic}, {object_detection_topic}")
        topics_to_wait = [color_topic, depth_topic, object_detection_topic]
        
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

            # Wait for topics including object detection
            object_detection_topic = "/objectDetection/data"
            if not self.wait_for_topics(color_topic, depth_topic, object_detection_topic):
                return False

            # Create subscribers
            self.color_sub = Subscriber(self, color_msg_type, color_topic)
            self.depth_sub = Subscriber(self, depth_msg_type, depth_topic)
            
            self.get_logger().info(f"Subscribed to {color_topic}")
            self.get_logger().info(f"Subscribed to {depth_topic}")
            self.get_logger().info(f"Subscribed to {object_detection_topic}")

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
            # check if the depth camera and color camera have the same resolution.
            if self.depth_image is not None:
                if not self.check_camera_resolution(self.color_image, self.depth_image) and self.camera_type != "pepper":
                    self.get_logger().error(f"{self.node_name}: Color camera and depth camera have different resolutions.")
                    rclpy.shutdown()
                    
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
                    f"No images received for {elapsed:.1f}s (timeout={self.image_timeout}s)"
                )

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

            # Optional: clamp far distances to improve local contrast, e.g., 0..3m
            # depth_f32 = np.clip(depth_f32, 0.0, 3.0)

            # Normalize to 0..255; if your depth is in millimeters, scale first
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
            return

        face_msg = FaceDetection()
        face_msg.face_label_id = [str(d['face_id']) for d in tracking_data]
        face_msg.centroids = [d['centroid'] for d in tracking_data]
        face_msg.width = [float(d['width']) for d in tracking_data]
        face_msg.height = [float(d['height']) for d in tracking_data]
        face_msg.mutual_gaze = [bool(d['mutual_gaze']) for d in tracking_data]

        self.pub_gaze.publish(face_msg)

    def object_detection_callback(self, msg: ObjectDetection):
        """Callback for object detection messages. Default implementation stores the message."""
        with self.object_detections_lock:
            self.latest_object_detections = msg

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

        # Initialize YOLOONNX model for face detection within person ROIs
        try:
            self.yolo_model = YOLOONNX(model_path=yolo_model_path, class_score_th=sixdrepnet_confidence)
            if self.verbose_mode:
                self.get_logger().info(f"{self.node_name}: YOLOONNX model initialized successfully.")
        except Exception as e:
            self.yolo_model = None
            self.get_logger().error(f"{self.node_name}: Failed to initialize YOLOONNX model: {e}")
            return  # Exit early if initialization fails

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
            return  # Exit early if initialization fails

        # Set up remaining attributes
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if self.verbose_mode:
            self.get_logger().info(f"{self.node_name} SixDrepNet initialization complete.")

        # Subscribe to topics (using parent class method)
        if not self.subscribe_topics():
            self.get_logger().error(f"{self.node_name}: Failed to subscribe to required topics. Shutting down.")
            # Don't proceed if we can't subscribe to required topics
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

    def process_images(self):
        """Process synchronized RGB + depth images for SixDrepNet."""
        if self.color_image is None or self.depth_image is None:
            return

        frame = self.process_frame(self.color_image)
        if frame is not None:
            self.update_latest_frame(frame)
            
    def process_frame(self, cv_image):
        """Process frame with one-face-per-person tracking."""
        debug_image = cv_image.copy()
        img_h, img_w = debug_image.shape[:2]
        tracking_data = []

        if not hasattr(self, "face_colors"):
            self.face_colors = {}

        with self.object_detections_lock:
            object_detections = self.latest_object_detections

        if object_detections is None:
            return debug_image

        # Collect person detections
        persons = []
        for i in range(len(object_detections.object_label_id)):
            if i < len(object_detections.class_names) and object_detections.class_names[i] == 'person':
                if i < len(object_detections.centroids):
                    centroid = object_detections.centroids[i]
                    width = object_detections.width[i] if i < len(object_detections.width) else 100
                    height = object_detections.height[i] if i < len(object_detections.height) else 100
                    
                    x1 = max(0, int(centroid.x - width / 2))
                    y1 = max(0, int(centroid.y - height / 2))
                    x2 = min(img_w, int(centroid.x + width / 2))
                    y2 = min(img_h, int(centroid.y + height / 2))
                    
                    persons.append({
                        'id': object_detections.object_label_id[i],
                        'box': (x1, y1, x2, y2),
                        'depth': centroid.z,
                        'assigned_face': None
                    })

        if not persons:
            return debug_image

        # Run face detection on full image
        face_boxes, face_scores = self.yolo_model(cv_image)
        
        if self.verbose_mode:
            self.get_logger().info(f"Persons: {len(persons)}, Faces: {len(face_boxes)}")

        # Build list of face detections
        faces = []
        for idx, (box, score) in enumerate(zip(face_boxes, face_scores)):
            fx1, fy1, fx2, fy2 = box
            face_cx = (fx1 + fx2) // 2
            face_cy = (fy1 + fy2) // 2
            face_width = fx2 - fx1
            face_height = fy2 - fy1
            
            if face_width < 20 or face_height < 20:
                continue
                
            faces.append({
                'box': (fx1, fy1, fx2, fy2),
                'centroid': (face_cx, face_cy),
                'size': (face_width, face_height),
                'score': score,
                'assigned': False
            })

        # Associate faces with persons (greedy: best score first)
        faces.sort(key=lambda f: f['score'], reverse=True)
        
        for face in faces:
            face_cx, face_cy = face['centroid']
            
            best_person_idx = None
            best_overlap = 0
            
            for p_idx, person in enumerate(persons):
                if person['assigned_face'] is not None:
                    continue  # Already has a face
                    
                px1, py1, px2, py2 = person['box']
                
                # Check if face centroid is inside person box
                if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
                    # Calculate how centered the face is in the upper portion of person box
                    # (faces should be in upper half of person)
                    person_upper_half_y = (py1 + py2) / 2
                    if face_cy < person_upper_half_y:
                        # Prefer faces in upper half
                        overlap = face['score'] + 0.5
                    else:
                        overlap = face['score']
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_person_idx = p_idx
            
            if best_person_idx is not None:
                persons[best_person_idx]['assigned_face'] = face
                face['assigned'] = True

        # Process assigned faces
        for person in persons:
            face = person['assigned_face']
            if face is None:
                continue
                
            fx1, fy1, fx2, fy2 = face['box']
            face_cx, face_cy = face['centroid']
            face_width, face_height = face['size']
            
            # Face ID = Person ID (stable tracking!)
            face_id = str(person['id'])
            
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
            yaw_pitch_roll = self.sixdrepnet_session.run(None, {'input': input_tensor})[0][0]
            yaw_deg, pitch_deg, roll_deg = yaw_pitch_roll

            # Draw head pose axes
            self.draw_axis(debug_image, yaw_deg, pitch_deg, roll_deg, face_cx, face_cy, size=50)

            # Get depth
            cz = person['depth'] if person['depth'] > 0 else 0.0
            if cz == 0.0:
                cz_depth = self.get_depth_in_region(face_cx, face_cy, face_width, face_height)
                cz = cz_depth if cz_depth is not None else 0.0

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
            cv2.putText(debug_image, f"Depth: {cz:.2f}m", (fx1 + 10, fy2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)

        # Draw person boxes
        if self.verbose_mode:
            for person in persons:
                px1, py1, px2, py2 = person['box']
                color = (0, 255, 0) if person['assigned_face'] else (128, 128, 128)
                cv2.rectangle(debug_image, (px1, py1), (px2, py2), color, 1)
                cv2.putText(debug_image, f"P:{person['id']}", (px1 + 5, py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        self.publish_face_detection(tracking_data)
        return debug_image