#!/usr/bin/env python3
"""
Improved Visualization for Overt Attention System
Shows faces with tracking IDs, engagement status, depth, saliency peaks, and current head target
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from cssr_interfaces.msg import FaceDetection
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


def get_image_qos() -> QoSProfile:
    """Get QoS profile suitable for image transport over WiFi."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1
    )


def load_topics_config(package_name: str, relative_path: str) -> dict:
    """Load topics configuration from YAML file using ROS2 package path."""
    package_share = get_package_share_directory(package_name)
    config_path = Path(package_share) / relative_path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_color_from_id(face_id: str) -> tuple:
    """Generate a consistent color for a face ID using hash."""
    # Hash the ID to get a consistent number
    hash_val = hash(face_id)
    
    # Generate RGB values that are reasonably bright
    r = (hash_val & 0xFF)
    g = ((hash_val >> 8) & 0xFF)
    b = ((hash_val >> 16) & 0xFF)
    
    # Ensure minimum brightness
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    if brightness < 100:
        # Boost all channels proportionally
        scale = 150 / max(brightness, 1)
        r = min(255, int(r * scale))
        g = min(255, int(g * scale))
        b = min(255, int(b * scale))
    
    return (b, g, r)  # BGR for OpenCV


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        
        # Load topics configuration from YAML file using ROS2 package path
        try:
            self.topics_config = load_topics_config('overt_attention', 'data/pepper_topics.yaml')
            self.get_logger().info("Loaded topics configuration from overt_attention package")
        except Exception as e:
            self.get_logger().error(f"Failed to load topics configuration: {e}")
            raise
        
        # Parameters
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('image_topic_base', '/camera/color/image_raw_custom')
        self.declare_parameter('show_face_ids', True)
        self.declare_parameter('show_depth', True)
        self.declare_parameter('show_engagement', True)
        
        # Load parameters
        self.use_compressed = self.get_parameter('use_compressed').value
        image_base = self.get_parameter('image_topic_base').value
        self.show_face_ids = self.get_parameter('show_face_ids').value
        self.show_depth = self.get_parameter('show_depth').value
        self.show_engagement = self.get_parameter('show_engagement').value
        
        # Load topics from YAML config
        self.face_topic = self.topics_config['topics']['face']
        self.saliency_topic = self.topics_config['topics']['saliency']['peak']
        self.target_topic = self.topics_config['topics']['target_angles']
        self.camera_info_topic = self.topics_config['topics']['camera_info']
        
        # Construct image topic
        if self.use_compressed:
            self.image_topic = image_base + "/compressed"
        else:
            self.image_topic = image_base
        
        # QoS
        qos = get_image_qos()
        
        # Subscriptions
        if self.use_compressed:
            self.create_subscription(CompressedImage, self.image_topic, self.on_image_compressed, qos)
        else:
            self.create_subscription(Image, self.image_topic, self.on_image_raw, qos)
        
        self.get_logger().info(f"Subscribing to image: {self.image_topic}")
        self.create_subscription(FaceDetection, self.face_topic, self.on_faces, 10)
        self.create_subscription(Float32MultiArray, self.saliency_topic, self.on_saliency, 10)
        self.create_subscription(Vector3, self.target_topic, self.on_target, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_caminfo, qos)
        
        # Publisher
        self.pub_overlay = self.create_publisher(Image, '/attn/visualization', 10)
        
        # State
        self.bridge = CvBridge()
        self.faces = []
        self.saliency_peaks = []
        self.current_target = None
        self.target_face_id = None  # Track which face is currently targeted
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # Face color cache (for consistent colors across frames)
        self.face_colors = {}
        
        # Counters for debugging
        self.image_count = 0
        self.face_count = 0
        self.saliency_count = 0
        self.target_count = 0
        
        self.get_logger().info("Improved visualization ready (Faces w/ Tracking + Engagement + Saliency)")

    def on_caminfo(self, msg: CameraInfo):
        """Store camera intrinsics."""
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(
                f"Camera info: fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}"
            )

    def on_faces(self, msg: FaceDetection):
        """Store face detections with tracking IDs, engagement, and depth."""
        self.face_count += 1
        self.faces = []
        
        n = len(msg.centroids)
        for i in range(n):
            u = msg.centroids[i].x
            v = msg.centroids[i].y
            depth = msg.centroids[i].z
            w = msg.width[i] if i < len(msg.width) else 80
            h = msg.height[i] if i < len(msg.height) else 80
            face_id = msg.face_label_id[i] if i < len(msg.face_label_id) else f"unknown_{i}"
            engaged = msg.mutual_gaze[i] if i < len(msg.mutual_gaze) else False
            
            # Generate consistent color for this face ID
            if face_id not in self.face_colors:
                self.face_colors[face_id] = generate_color_from_id(face_id)
            
            self.faces.append({
                'u': int(u),
                'v': int(v),
                'w': int(w),
                'h': int(h),
                'depth': depth,
                'face_id': face_id,
                'engaged': engaged,
                'color': self.face_colors[face_id]
            })
        
        if self.face_count == 1:
            self.get_logger().info(f"First face message received: {len(self.faces)} faces")

    def on_saliency(self, msg: Float32MultiArray):
        """Store saliency peaks: [u1, v1, s1, u2, v2, s2, ...]"""
        self.saliency_count += 1
        self.saliency_peaks = []
        
        for i in range(0, len(msg.data) - 2, 3):
            u, v, score = msg.data[i], msg.data[i + 1], msg.data[i + 2]
            self.saliency_peaks.append({
                'u': int(u),
                'v': int(v),
                'score': score
            })
        
        if self.saliency_count == 1:
            self.get_logger().info(f"First saliency message received: {len(self.saliency_peaks)} peaks")

    def on_target(self, msg: Vector3):
        """Store current target angles and try to determine which face is targeted."""
        self.target_count += 1
        self.current_target = {
            'yaw': msg.x,
            'pitch': msg.y,
            'score': msg.z
        }
        
        # Try to determine which face is being targeted (if any)
        self.update_target_face()
        
        if self.target_count == 1:
            self.get_logger().info(
                f"First target received: yaw={np.degrees(msg.x):.1f}°, "
                f"pitch={np.degrees(msg.y):.1f}°"
            )

    def update_target_face(self):
        """Determine which face (if any) is currently being targeted."""
        if not self.current_target or not self.faces or self.fx is None:
            self.target_face_id = None
            return
        
        # Convert target angles to pixel coordinates
        yaw = self.current_target['yaw']
        pitch = self.current_target['pitch']
        
        x_norm = -np.tan(yaw)
        y_norm = np.tan(pitch)
        target_u = x_norm * self.fx + self.cx
        target_v = y_norm * self.fy + self.cy
        
        # Find closest face to target
        min_dist = float('inf')
        closest_face_id = None
        
        for face in self.faces:
            dist = np.sqrt((face['u'] - target_u)**2 + (face['v'] - target_v)**2)
            if dist < min_dist and dist < 100:  # Within 100 pixels
                min_dist = dist
                closest_face_id = face['face_id']
        
        self.target_face_id = closest_face_id

    def on_image_compressed(self, msg: CompressedImage):
        """Process compressed image."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                self.get_logger().warn("Failed to decode compressed image")
                return
            
            self.process_frame(frame, msg.header.stamp)
            
        except Exception as e:
            self.get_logger().error(f"Error processing compressed image: {e}")

    def on_image_raw(self, msg: Image):
        """Process raw image."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.process_frame(frame, msg.header.stamp)
            
        except Exception as e:
            self.get_logger().error(f"Error processing raw image: {e}")

    def process_frame(self, frame, stamp):
        """Process and visualize frame."""
        self.image_count += 1
        
        # Log first image
        if self.image_count == 1:
            h, w = frame.shape[:2]
            self.get_logger().info(f"First image received: {w}x{h}")
        
        # Draw visualizations
        vis = frame.copy()
        H, W = vis.shape[:2]
        
        # Draw saliency peaks first (so they're under faces)
        for i, peak in enumerate(self.saliency_peaks):
            self.draw_saliency_peak(vis, peak, i)
        
        # Draw faces (Priority 1)
        for face in self.faces:
            is_targeted = (face['face_id'] == self.target_face_id)
            self.draw_face(vis, face, is_targeted)
        
        # Draw current target
        if self.current_target and self.fx is not None:
            self.draw_target(vis, self.current_target, W, H)
        
        # Draw info overlay
        self.draw_info(vis)
        
        # Publish as RAW Image
        try:
            out_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            out_msg.header.stamp = stamp
            out_msg.header.frame_id = "camera_color_optical_frame"
            self.pub_overlay.publish(out_msg)
            
            # Log first publish
            if self.image_count == 1:
                self.get_logger().info("First visualization published to /attn/visualization")
                
        except Exception as e:
            self.get_logger().error(f"Error publishing visualization: {e}")

    def draw_face(self, vis, face, is_targeted=False):
        """Draw face bounding box with tracking ID, engagement, and depth."""
        u, v = face['u'], face['v']
        w2, h2 = face['w'] // 2, face['h'] // 2
        
        color = face['color']
        engaged = face['engaged']
        face_id = face['face_id']
        depth = face['depth']
        
        # Thicker box if targeted or engaged
        thickness = 5 if (is_targeted or engaged) else 3
        
        # Draw bounding box
        x1, y1 = u - w2, v - h2
        x2, y2 = u + w2, v + h2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        
        # Draw engagement indicator (filled circle) if engaged
        if engaged:
            # Draw a glowing effect for engaged faces
            cv2.circle(vis, (u, v), 25, (0, 255, 0), 3)
            cv2.circle(vis, (u, v), 15, (0, 255, 0), 2)
            cv2.putText(
                vis, "ENGAGED",
                (x1, y1 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2, cv2.LINE_AA
            )
        
        # Center cross
        cv2.drawMarker(
            vis, (u, v), color,
            markerType=cv2.MARKER_CROSS,
            markerSize=15, thickness=3
        )
        
        # Draw face ID if enabled
        if self.show_face_ids:
            id_text = f"ID: {face_id}"
            if is_targeted:
                id_text = ">>> " + id_text + " <<<"
            
            cv2.putText(
                vis, id_text,
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA
            )
        
        # Draw depth if enabled and available
        if self.show_depth and depth > 0:
            depth_text = f"{depth:.2f}m"
            cv2.putText(
                vis, depth_text,
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA
            )
        
        # Draw "FACE" label
        label = "FACE"
        if is_targeted:
            label = "TARGETED FACE"
        
        cv2.putText(
            vis, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            color, 2, cv2.LINE_AA
        )

    def draw_saliency_peak(self, vis, peak, rank):
        """Draw a saliency peak."""
        u, v = peak['u'], peak['v']
        score = peak['score']
        
        # Color by rank (brightest for highest)
        if rank == 0:
            color = (0, 255, 255)  # Yellow - highest
            radius = int(20 + score * 30)
        else:
            alpha = max(0.3, 1.0 - rank * 0.2)
            color = (int(alpha * 0), int(alpha * 200), int(alpha * 200))
            radius = int(15 + score * 20)
        
        # Draw circle
        cv2.circle(vis, (u, v), radius, color, 2)
        cv2.circle(vis, (u, v), 3, color, -1)
        
        # Label
        label = f"SAL #{rank+1}: {score:.2f}"
        cv2.putText(
            vis, label,
            (u + 25, v + 5 + rank * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1, cv2.LINE_AA
        )

    def draw_target(self, vis, target, W, H):
        """Draw current attention target."""
        yaw, pitch = target['yaw'], target['pitch']
        score = target.get('score', 0.0)
        
        # Project angles back to pixel coordinates
        x_norm = -np.tan(yaw)
        y_norm = np.tan(pitch)
        u = int(x_norm * self.fx + self.cx)
        v = int(y_norm * self.fy + self.cy)
        
        # Draw even if off-screen (with indicator)
        if not (0 <= u < W and 0 <= v < H):
            cv2.putText(
                vis, "TARGET OFF-SCREEN",
                (W//2 - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2, cv2.LINE_AA
            )
            return
        
        # Draw target reticle
        color = (0, 0, 255)  # Red
        size = 40
        
        # Crosshair
        cv2.line(vis, (u - size, v), (u + size, v), color, 3)
        cv2.line(vis, (u, v - size), (u, v + size), color, 3)
        cv2.circle(vis, (u, v), size, color, 3)
        cv2.circle(vis, (u, v), 5, color, -1)
        
        # Label
        label = f"TARGET: ({np.degrees(yaw):.1f}°, {np.degrees(pitch):.1f}°)"
        if score > 0:
            label += f" s={score:.2f}"
        
        # Add face ID if targeting a face
        if self.target_face_id:
            label += f" [{self.target_face_id}]"
        
        cv2.putText(
            vis, label,
            (u + 50, v - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            color, 2, cv2.LINE_AA
        )

    def draw_info(self, vis):
        """Draw info overlay."""
        H, W = vis.shape[:2]
        
        # Count engaged faces
        engaged_count = sum(1 for f in self.faces if f['engaged'])
        
        # Semi-transparent background
        overlay = vis.copy()
        info_height = 200 if engaged_count > 0 else 180
        cv2.rectangle(overlay, (10, 10), (450, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        # Info text
        y = 35
        info = [
            f"Faces: {len(self.faces)} ({engaged_count} engaged)",
            f"Saliency peaks: {len(self.saliency_peaks)}",
            f"Target: {self.target_face_id if self.target_face_id else 'Saliency' if self.current_target else 'None'}"
        ]
        
        # Add tracked face IDs
        if self.faces:
            face_ids = [f['face_id'] for f in self.faces]
            info.append(f"Tracked IDs: {', '.join(face_ids[:5])}")  # Show first 5
            if len(face_ids) > 5:
                info.append(f"  ... and {len(face_ids) - 5} more")
        
        for text in info:
            color = (0, 255, 0)
            # Highlight engaged faces in green
            if "engaged)" in text and engaged_count > 0:
                color = (0, 255, 0)
            
            cv2.putText(
                vis, text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA
            )
            y += 30
        
        # Legend
        y += 10
        cv2.putText(
            vis, "Legend:",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1, cv2.LINE_AA
        )
        y += 20
        
        legend_items = [
            ("Green Glow", "Engaged (mutual gaze)", (0, 255, 0)),
            ("Colored Box", "Tracked face", (255, 255, 255)),
            ("Red Cross", "Attention target", (0, 0, 255)),
            ("Yellow Circle", "Top saliency", (0, 255, 255))
        ]
        
        for label, desc, color in legend_items:
            # Draw small colored box
            cv2.rectangle(vis, (20, y - 10), (35, y + 5), color, -1)
            cv2.putText(
                vis, f"{label}: {desc}",
                (45, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (200, 200, 200), 1, cv2.LINE_AA
            )
            y += 18


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()