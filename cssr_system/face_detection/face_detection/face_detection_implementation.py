import cv2
import mediapipe as mp
import numpy as np
import rclpy
import os
import onnxruntime
import multiprocessing
import yaml
import random
import threading
import time
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from math import cos, sin, pi
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Point
from typing import Tuple, List, Dict, Optional
from cssr_interfaces.msg import FaceDetection
from cssr_interfaces.srv import GetDepthAtPixel  # Add service import
from .face_detection_tracking import Sort, CentroidTracker

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
        'mpFacedetConfidence': 0.5,
        'mpHeadposeAngle': 8,
        'centroidMaxDisappeared': 15,
        'centroidMaxDistance': 100,
        'sixdrepnetConfidence': 0.65,
        'sixdrepnetHeadposeAngle': 10,
        'sortMaxDisappeared': 5,
        'sortMinHits': 3,
        'sortIouThreshold': 0.3,
        'useDepthService': False,  # New config option
        'depthServiceTimeout': 0.1  # Timeout for service calls in seconds
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
        
        # Only publish depth debug if not using service mode
        if not config.get('useDepthService', False):
            self.depth_debug_pub = self.create_publisher(Image, "/faceDetection/depth_debug", 1)

        self.bridge = CvBridge()
        self.depth_image: Optional[np.ndarray] = None
        self.color_image: Optional[np.ndarray] = None
        
        # Configuration values
        self.use_compressed = config['useCompressed']
        self.camera_type = config['camera']
        self.verbose_mode = config['verboseMode']
        self.image_timeout = config['imageTimeout']
        self.use_depth_service = config.get('useDepthService', False)
        self.depth_service_timeout = config.get('depthServiceTimeout', 0.1)
        
        self.node_name = self.get_name()
        self.timer_start = self.get_clock().now()
        self.last_image_time = None   # timestamp of the last received image

        # Thread safety for visualization
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None

        # Initialize depth service client if using service mode
        if self.use_depth_service:
            self.depth_client = self.create_client(GetDepthAtPixel, self._get_depth_service_name()
            )
            self.get_logger().info(f"Using depth service mode: {self._get_depth_service_name()}")
            
            # Wait for service to be available
            if not self.depth_client.wait_for_service(timeout_sec=10.0):
                self.get_logger().error("Depth service not available after 10 seconds")
        else:
            self.depth_client = None
            self.get_logger().info("Using depth subscription mode")

        # Start visualization timer (30 Hz)
        self.vis_timer = self.create_timer(1.0 / 30.0, self.visualization_callback)

    def _get_depth_service_name(self) -> str:
        """Get the depth service name based on camera type."""
        if self.camera_type == "realsense":
            return "/realsense/depth_query"
        elif self.camera_type == "pepper":
            return "/pepper/depth_query"
        else:
            return "/depth_query"  # Default service name

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
            # Only visualize depth if not in service mode
            if not self.use_depth_service and self.depth_image is not None:
                depth_vis = self._make_depth_vis(self.depth_image)

        if color_frame is None and (not self.use_depth_service and depth_vis is None):
            return

        if self.config.get("verboseMode", False) and os.environ.get("DISPLAY", "") != "":
            try:
                if color_frame is not None:
                    cv2.imshow("Face Detection Debug (RGB)", color_frame)
                if depth_vis is not None and not self.use_depth_service:
                    cv2.imshow("Face Detection Debug (Depth)", depth_vis)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn(f"imshow failed (likely headless): {e}")

            # Publish only when verboseMode=True
            try:
                if color_frame is not None:
                    msg = self.bridge.cv2_to_imgmsg(color_frame, encoding="bgr8")
                    self.debug_pub.publish(msg)
                if depth_vis is not None and not self.use_depth_service:
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
        
        # Only get depth topic if not using service mode
        depth_topic = None if self.use_depth_service else self.extract_topic(depth_key)
        
        if not rgb_topic:
            raise ValueError("Failed to extract RGB camera topic")
        if not self.use_depth_service and not depth_topic:
            raise ValueError("Failed to extract depth camera topic")
            
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

    def wait_for_topics(self, color_topic: str, depth_topic: Optional[str], timeout: float = 30.0) -> bool:
        """Wait for topics to become available."""
        topics_to_wait = [color_topic]
        if depth_topic and not self.use_depth_service:
            topics_to_wait.append(depth_topic)
            
        self.get_logger().info(f"Waiting for topics: {topics_to_wait}")
        
        start_time = self.get_clock().now()
        warning_interval = 5.0
        last_warning_time = start_time
        
        while rclpy.ok():
            # Check if topics are available
            topic_names_and_types = self.get_topic_names_and_types()
            published_topics = [name for name, _ in topic_names_and_types]
            
            all_available = all(topic in published_topics for topic in topics_to_wait)
            if all_available:
                if self.verbose_mode:
                    self.get_logger().info("All required topics are available!")
                return True
            
            # Check timeout
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > timeout:
                self.get_logger().error(f"Timeout waiting for topics after {timeout}s")
                return False
            
            # Periodic warnings
            if (self.get_clock().now() - last_warning_time).nanoseconds / 1e9 >= warning_interval:
                missing = [t for t in topics_to_wait if t not in published_topics]
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
                color_msg_type = CompressedImage
                
                if not self.use_depth_service:
                    depth_topic = depth_topic + "/compressedDepth"
                    depth_msg_type = CompressedImage
            elif self.use_compressed and self.camera_type == "pepper":
                self.get_logger().warn("Compressed images not available for Pepper cameras")
                color_topic = rgb_topic
                color_msg_type = Image
                if not self.use_depth_service:
                    depth_msg_type = Image
            else:
                color_topic = rgb_topic
                color_msg_type = Image
                if not self.use_depth_service:
                    depth_msg_type = Image

            # Wait for topics
            if not self.wait_for_topics(color_topic, depth_topic if not self.use_depth_service else None):
                return False

            if self.use_depth_service:
                # Only subscribe to color topic
                self.color_sub = self.create_subscription(
                    color_msg_type,
                    color_topic,
                    self.color_only_callback,
                    10
                )
                self.get_logger().info(f"Subscribed to {color_topic} (service mode)")
            else:
                # Create subscribers for both color and depth
                self.color_sub = Subscriber(self, color_msg_type, color_topic)
                self.depth_sub = Subscriber(self, depth_msg_type, depth_topic)
                
                self.get_logger().info(f"Subscribed to {color_topic}")
                self.get_logger().info(f"Subscribed to {depth_topic}")

                # Set up synchronizer
                slop = 5.0 if self.camera_type == "pepper" else 0.1
                self.ats = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=slop)
                self.ats.registerCallback(self.synchronized_callback)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to setup subscribers: {e}")
            return False

    def color_only_callback(self, color_data):
        """Process color image only (when using depth service)."""
        self.last_image_time = self.get_clock().now().nanoseconds / 1e9
        
        try:
            # Process color image
            if isinstance(color_data, CompressedImage):
                np_arr = np.frombuffer(color_data.data, np.uint8)
                self.color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                self.color_image = self.bridge.imgmsg_to_cv2(color_data, desired_encoding="bgr8")

            if self.color_image is None:
                self.get_logger().warn("Failed to decode color image")
                return

            # Process the images (depth will be queried via service as needed)
            self.process_images()
            
        except Exception as e:
            self.get_logger().error(f"Error in color_only_callback: {e}")

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

            # Check resolution match
            if not self.check_camera_resolution(self.color_image, self.depth_image) and self.camera_type != "pepper":
                self.get_logger().error(f"{self.node_name}: Color camera and depth camera have different resolutions.")
                rclpy.shutdown()

            # Process the images
            self.process_images()
            
        except Exception as e:
            self.get_logger().error(f"Error in synchronized_callback: {e}")

    def process_depth_image(self, depth_data) -> Optional[np.ndarray]:
        """Process depth image data."""
        try:
            if isinstance(depth_data, CompressedImage):
                if hasattr(depth_data, "format") and depth_data.format and "compressedDepth png" in depth_data.format:
                    # Handle PNG compression
                    depth_header_size = 12
                    depth_img_data = depth_data.data[depth_header_size:]
                    np_arr = np.frombuffer(depth_img_data, np.uint8)
                    return cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)
                else:
                    # Regular compressed image
                    np_arr = np.frombuffer(depth_data.data, np.uint8)
                    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            else:
                return self.bridge.imgmsg_to_cv2(depth_data, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth image processing error: {e}")
            return None

    def query_depth_service(self, points: List[Tuple[float, float]], radii: Optional[List[int]] = None) -> Optional[List[float]]:
        """
        Query depth values using the depth service.
        
        Args:
            points: List of (x, y) coordinates
            radii: Optional list of radii for each point (0 for single point)
            
        Returns:
            List of depth values in meters, or None if service call fails
        """
        if not self.depth_client:
            return None
            
        request = GetDepthAtPixel.Request()
        
        # Handle multiple points
        request.centers_x = [int(x) for x, y in points]
        request.centers_y = [int(y) for x, y in points]
        
        if radii:
            request.radii = radii
        else:
            request.radii = [0] * len(points)  # Single point queries
        
        try:
            # Make synchronous service call with timeout
            future = self.depth_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=self.depth_service_timeout)
            
            if future.done():
                response = future.result()
                if response and response.success:
                    return list(response.depths)
                else:
                    if response:
                        self.get_logger().warn(f"Depth service failed: {response.message}")
                    return None
            else:
                self.get_logger().warn("Depth service call timed out")
                return None
                
        except Exception as e:
            self.get_logger().error(f"Depth service call failed: {e}")
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

    def _make_depth_vis(self, depth: np.ndarray) -> Optional[np.ndarray]:
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

    def get_depth_at_centroid(self, centroid_x: float, centroid_y: float) -> Optional[float]:
        """Get depth value at specific coordinates (either from image or service)."""
        if self.use_depth_service:
            # Use service to query single point
            depths = self.query_depth_service([(centroid_x, centroid_y)], [0])
            return depths[0] if depths else None
        else:
            # Use depth image directly
            if self.depth_image is None:
                return None

            height, width = self.depth_image.shape[:2]
            x, y = int(round(centroid_x)), int(round(centroid_y))

            if 0 <= x < width and 0 <= y < height:
                depth_value = self.depth_image[y, x]
                if np.isfinite(depth_value) and depth_value > 0:
                    return depth_value / 1000.0  # Convert to meters
        
        return None

    def get_depth_in_region(self, centroid_x: float, centroid_y: float, 
                           box_width: float, box_height: float, 
                           region_scale: float = 0.1) -> Optional[float]:
        """Get median depth value in a region around the centroid."""
        if self.use_depth_service:
            # Calculate radius based on region size
            radius = max(5, int(min(box_width, box_height) * region_scale / 2))
            depths = self.query_depth_service([(centroid_x, centroid_y)], [radius])
            return depths[0] if depths else None
        else:
            # Use depth image directly
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

    def get_batch_depths(self, face_centroids: List[Tuple[float, float]], 
                        face_sizes: Optional[List[Tuple[float, float]]] = None) -> List[Optional[float]]:
        """
        Get depth values for multiple faces in a single service call (if using service mode).
        
        Args:
            face_centroids: List of (x, y) centroid coordinates
            face_sizes: Optional list of (width, height) for each face
            
        Returns:
            List of depth values in meters (None for failed queries)
        """
        if self.use_depth_service:
            # Calculate radii based on face sizes if provided
            if face_sizes:
                radii = [max(5, int(min(w, h) * 0.1 / 2)) for w, h in face_sizes]
            else:
                radii = [10] * len(face_centroids)  # Default radius
            
            depths = self.query_depth_service(face_centroids, radii)
            if depths:
                return depths
            else:
                return [None] * len(face_centroids)
        else:
            # Fall back to individual queries for non-service mode
            result = []
            for i, (cx, cy) in enumerate(face_centroids):
                if face_sizes and i < len(face_sizes):
                    depth = self.get_depth_in_region(cx, cy, face_sizes[i][0], face_sizes[i][1])
                else:
                    depth = self.get_depth_at_centroid(cx, cy)
                result.append(depth)
            return result

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

    def cleanup(self):
        """Clean up resources."""
        try:
            cv2.destroyAllWindows()
            self.get_logger().info("Cleanup completed")
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")

    def process_images(self):
        """Base implementation - override in subclasses."""
        pass


class MediaPipe(FaceDetectionNode):
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Get MediaPipe specific configuration values with defaults
        mp_confidence = self.config.get('mpFacedetConfidence', 0.5)
        self.mp_angle = self.config.get('mpHeadposeAngle', 8)
        centroid_max_disappeared = self.config.get('centroidMaxDisappeared', 15)
        centroid_max_distance = self.config.get('centroidMaxDistance', 100)
        
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=mp_confidence, max_num_faces=10)

        # Initialize the CentroidTracker
        self.centroid_tracker = CentroidTracker(centroid_max_disappeared, centroid_max_distance)

        # Subscribe to the image topic
        self.subscribe_topics()

        self.start_timeout_monitor()

    def process_images(self):
        """Process synchronized RGB + depth images for MediaPipe."""
        if self.color_image is None:
            return

        frame = self.color_image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        self.process_face_mesh(frame, rgb_frame, img_h, img_w)

    def process_face_mesh(self, frame, rgb_frame, img_h, img_w):
        results = self.face_mesh.process(rgb_frame)
        centroids = []
        mutualGaze_list = []
        face_widths = []
        face_heights = []
        face_boxes = []  # Store bounding boxes for each face
        tracking_data = []  # Initialize tracking_data here
        
        # Dictionary to store face ID colors
        if not hasattr(self, "face_colors"):
            self.face_colors = {}
            
        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                face_2d, face_3d = [], []
                x_min, y_min, x_max, y_max = img_w, img_h, 0, 0  # Bounding box coordinates
                
                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                    # Expand bounding box
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Calculate width and height
                width = x_max - x_min
                height = y_max - y_min
                
                # Store bounding box
                face_boxes.append((x_min, y_min, x_max, y_max))
                face_widths.append(width)
                face_heights.append(height)
                
                centroid_x = np.mean([pt[0] for pt in face_2d])
                centroid_y = np.mean([pt[1] for pt in face_2d])
                centroids.append((centroid_x, centroid_y))
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                    [0, focal_length, img_h / 2],
                                    [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)
                
                success, rotation_vec, translation_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, distortion_matrix)
                
                rmat, jac = cv2.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                
                x_angle = angles[0] * 360
                y_angle = angles[1] * 360
                
                mutualGaze = abs(x_angle) <= self.mp_angle and abs(y_angle) <= self.mp_angle
                mutualGaze_list.append(mutualGaze)
            
            # Use the centroid tracker to match centroids with object IDs
            centroid_to_face_id = self.centroid_tracker.match_centroids(centroids)
            
            # Batch query depths if using service mode
            if self.use_depth_service and centroids:
                face_sizes = [(w, h) for w, h in zip(face_widths, face_heights)]
                depths = self.get_batch_depths(centroids, face_sizes)
            else:
                depths = [None] * len(centroids)
            
            for idx, (centroid, width, height, box) in enumerate(zip(centroids, face_widths, face_heights, face_boxes)):
                centroid_tuple = tuple(centroid)
                face_id = centroid_to_face_id.get(centroid_tuple, None)
                
                # Assign a new dark color for a new face or lost tracking
                if face_id is None or face_id not in self.face_colors:
                    self.face_colors[face_id] = self.generate_dark_color()
                
                face_color = self.face_colors[face_id]
                
                # Get depth (either from batch query or individual query)
                if self.use_depth_service:
                    cz = depths[idx] if idx < len(depths) else None
                else:
                    cz = self.get_depth_at_centroid(centroid[0], centroid[1])
                
                cz = cz if cz is not None else 0.0  # Default to 0.0 meters
                
                point = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(cz) if cz else 0.0)
                
                # Add width and height to tracking data
                tracking_data.append({
                    'face_id': str(face_id),
                    'centroid': point,
                    'width': float(width),
                    'height': float(height),
                    'mutual_gaze': bool(mutualGaze_list[idx])
                })
                
                # Unpack bounding box coordinates
                x_min, y_min, x_max, y_max = box
                
                # Draw bounding box with assigned color
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), face_color, 2)
                
                # Add label above bounding box
                label = "Engaged" if mutualGaze_list[idx] else "Not Engaged"
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
                cv2.putText(frame, f"Face: {face_id}", (x_min, y_min - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
                
                # Draw depth information below the box
                cv2.putText(frame, f"Depth: {cz:.2f}m", (x_min, y_max + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
            
        self.update_latest_frame(frame)
        
        # Publish the tracking data
        self.publish_face_detection(tracking_data)


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
        resized_image = self.__preprocess(image)
        inference_image = resized_image[np.newaxis, ...].astype(np.float32)
        boxes = self.onnx_session.run(
            self.output_names,
            {name: inference_image for name in self.input_names},
        )[0]
        return self.__postprocess(image, boxes)

    def __preprocess(self, image: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        resized_image = resized_image[:, :, ::-1] / 255.0  # BGR to RGB and normalize
        return resized_image.transpose(2, 0, 1)  # HWC to CHW

    def __postprocess(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        self.sort_max_disappeared = self.config.get('sortMaxDisappeared', 5)
        self.sort_min_hits = self.config.get('sortMinHits', 3)
        self.sort_iou_threshold = self.config.get('sortIouThreshold', 0.3)
        
        if self.verbose_mode:
            self.get_logger().info(f"{self.node_name}: Initializing SixDrepNet...")

        # Set up model paths
        try:
            package_path = get_package_share_directory('face_detection')
            yolo_model_path = os.path.join(package_path, 'models/face_detection_goldYOLO.onnx')
            sixdrepnet_model_path = os.path.join(package_path, 'models/face_detection_sixdrepnet360.onnx')
        except Exception as e:
            self.get_logger().error(f"{self.node_name}: Failed to get package path: {e}")
            return
            
        # Initialize YOLOONNX model early and check success
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

        self.sort_tracker = Sort(max_age=self.sort_max_disappeared, min_hits=self.sort_min_hits, iou_threshold=self.sort_iou_threshold)
        self.tracks = [] 
        
        if self.verbose_mode:
            self.get_logger().info(f"{self.node_name} SixDrepNet initialization complete.")

        self.subscribe_topics()

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
        if self.color_image is None:
            return

        frame = self.process_frame(self.color_image)
        if frame is not None:
            self.update_latest_frame(frame)
            
    def process_frame(self, cv_image):
        """
        Process the input frame for face detection and head pose estimation using SORT.
        Args: 
            cv_image: Input frame as a NumPy array (BGR format)
        """
        debug_image = cv_image.copy()
        img_h, img_w = debug_image.shape[:2]
        tracking_data = []

        # Dictionary to store face ID colors
        if not hasattr(self, "face_colors"):
            self.face_colors = {}

        # Object detection (YOLO)
        boxes, scores = self.yolo_model(debug_image)

        # Prepare detections for SORT ([x1, y1, x2, y2, score])
        detections = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, score])

        # Convert detections to NumPy array
        detections = np.array(detections)

        # Update SORT tracker with detections
        if detections.shape[0] > 0:
            self.tracks = self.sort_tracker.update(detections)
        else:
            self.tracks = []  # Reset tracks if no detections

        # Collect face data for batch depth query
        face_centroids = []
        face_sizes = []
        
        for track in self.tracks:
            x1, y1, x2, y2, face_id = map(int, track)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            face_centroids.append((cx, cy))
            face_sizes.append((width, height))
        
        # Batch query depths if using service mode
        if self.use_depth_service and face_centroids:
            depths = self.get_batch_depths(face_centroids, face_sizes)
        else:
            depths = [None] * len(face_centroids)

        # Process tracks
        for idx, track in enumerate(self.tracks):
            x1, y1, x2, y2, face_id = map(int, track)  # SORT returns face_id as the last value
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Calculate width and height
            width = x2 - x1
            height = y2 - y1

            # Assign a unique color for each face ID
            if face_id is None or face_id not in self.face_colors:
                self.face_colors[face_id] = self.generate_dark_color()

            face_color = self.face_colors[face_id]

            # Crop the face region for head pose estimation
            head_image = debug_image[max(y1, 0):min(y2, img_h), max(x1, 0):min(x2, img_w)]
            if head_image.size == 0:
                continue  # Skip if cropped region is invalid

            # Preprocess for SixDrepNet
            resized_image = cv2.resize(head_image, (224, 224))
            normalized_image = (resized_image[..., ::-1] / 255.0 - self.mean) / self.std
            input_tensor = normalized_image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

            # Run head pose estimation
            yaw_pitch_roll = self.sixdrepnet_session.run(None, {'input': input_tensor})[0][0]
            yaw_deg, pitch_deg, roll_deg = yaw_pitch_roll

            # Draw head pose axes
            self.draw_axis(debug_image, yaw_deg, pitch_deg, roll_deg, cx, cy, size=100)

            # Get depth (either from batch query or individual query)
            if self.use_depth_service:
                cz = depths[idx] if idx < len(depths) else None
            else:
                cz = self.get_depth_in_region(cx, cy, width, height)
            
            cz = cz if cz is not None else 0.0

            # Determine if the person is engaged
            mutual_gaze = abs(yaw_deg) < self.sixdrep_angle and abs(pitch_deg) < self.sixdrep_angle

            # Add width and height to tracking data
            tracking_data.append({
                'face_id': str(face_id),
                'centroid': Point(x=float(cx), y=float(cy), z=float(cz) if cz else 0.0),
                'width': float(width),
                'height': float(height),
                'mutual_gaze': mutual_gaze
            })

            # Draw bounding box with assigned color
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), face_color, 2)

            # Add labels above bounding box
            label = "Engaged" if mutual_gaze else "Not Engaged"
            cv2.putText(debug_image, label, (x1 + 10, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

            cv2.putText(debug_image, f"Face: {face_id}", (x1 + 10, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
            
            # Draw depth information below the box
            cv2.putText(debug_image, f"Depth: {cz:.2f}m", (x1 + 10, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

        # Publish tracking data
        self.publish_face_detection(tracking_data)
        return debug_image