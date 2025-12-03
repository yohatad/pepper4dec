#!/usr/bin/env python3

"""
Visualization Node for Pepper Attention System
- Overlays attention targets on camera feed
- Publishes RViz markers for 3D visualization
- Real-time metrics dashboard
"""

import cv2
import numpy as np
import time
import collections
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import Vector3, Point, Pose, PoseStamped
from std_msgs.msg import Float32MultiArray, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from cssr_interfaces.msg import FaceDetection


# ============ Helper Functions ============
def get_image_topic(base_topic: str, use_compressed: bool, is_depth: bool = False) -> str:
    """
    Construct the full topic name based on compression setting.
    
    Args:
        base_topic: Base topic name (e.g., "/camera/color/image_raw")
        use_compressed: Whether to use compressed transport
        is_depth: Whether this is a depth image (uses /compressedDepth instead of /compressed)
    
    Returns:
        Full topic name with appropriate suffix
    """
    if use_compressed:
        suffix = "/compressedDepth" if is_depth else "/compressed"
        return base_topic + suffix
    return base_topic


def get_image_qos() -> QoSProfile:
    """Get QoS profile suitable for image transport over WiFi."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1
    )


class AttentionVisualization(Node):
    def __init__(self):
        super().__init__('attention_visualization')
        
        # Declare and load parameters
        self.declare_all_parameters()
        self.load_parameters()
        
        # QoS
        qos = get_image_qos()
        
        # Subscribers - image topic uses shared use_compressed
        image_topic = get_image_topic(self.image_topic_base, self.use_compressed, is_depth=False)
        if self.use_compressed:
            self.sub_img = self.create_subscription(CompressedImage, image_topic, self.on_image, qos)
        else:
            self.sub_img = self.create_subscription(Image, image_topic, self.on_image_raw, qos)
        
        self.get_logger().info(f"Subscribing to image: {image_topic}")
        
        self.sub_faces = self.create_subscription(FaceDetection, self.face_topic, self.on_faces, 10)
        self.sub_sal = self.create_subscription(Float32MultiArray, self.saliency_topic, self.on_saliency, 10)
        self.sub_target = self.create_subscription(Vector3, self.target_topic, self.on_target, 10)
        self.sub_caminfo = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_caminfo, qos)
        
        # Publishers
        if self.publish_overlay:
            self.pub_overlay = self.create_publisher(CompressedImage, '/attn/visualization/compressed', 10)
        
        if self.publish_markers:
            self.pub_markers = self.create_publisher(MarkerArray, '/attn/markers', 10)
            self.pub_target_pose = self.create_publisher(PoseStamped, '/attn/target_pose', 10)
        
        # State
        self.bridge = CvBridge()
        self.current_frame = None
        self.faces = []
        self.saliency_peaks = []  # Support multiple peaks
        self.current_target = None  # (yaw, pitch, score)
        self.fx = self.fy = self.cx = self.cy = None
        
        # Metrics
        self.frame_times = collections.deque(maxlen=30)
        self.last_frame_t = time.time()
        self.target_history = collections.deque(maxlen=100)
        self.switch_times = []
        self.last_target_id = None
        
        self.get_logger().info(
            f"Attention visualization ready "
            f"(use_compressed={self.use_compressed})"
        )

    def declare_all_parameters(self):
        """Declare all ROS parameters with defaults"""
        # Global shared parameter
        self.declare_parameter('use_compressed', True)
        
        # Image topic (base, without /compressed suffix)
        self.declare_parameter('image_topic_base', '/camera/color/image_raw')
        
        # Other topic parameters
        self.declare_parameter('face_topic', '/faceDetection/data')
        self.declare_parameter('saliency_topic', '/attn/saliency_peak')
        self.declare_parameter('target_topic', '/attn/target_angles')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        
        # Feature flags
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('show_metrics', True)

    def load_parameters(self):
        """Load all parameters into instance variables"""
        # Global shared parameter
        self.use_compressed = self.get_parameter('use_compressed').value
        
        # Topics
        self.image_topic_base = self.get_parameter('image_topic_base').value
        self.face_topic = self.get_parameter('face_topic').value
        self.saliency_topic = self.get_parameter('saliency_topic').value
        self.target_topic = self.get_parameter('target_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        
        # Feature flags
        self.publish_overlay = self.get_parameter('publish_overlay').value
        self.publish_markers = self.get_parameter('publish_markers').value
        self.show_metrics = self.get_parameter('show_metrics').value

    def on_caminfo(self, msg: CameraInfo):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def on_image(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self.process_visualization(frame, msg.header.stamp)

    def on_image_raw(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_visualization(frame, msg.header.stamp)

    def on_faces(self, msg: FaceDetection):
        self.faces = []
        n = len(msg.centroids)
        for i in range(n):
            fid = msg.face_label_id[i] if i < len(msg.face_label_id) else str(i)
            c = msg.centroids[i]
            w = msg.width[i] if i < len(msg.width) else 80.0
            h = msg.height[i] if i < len(msg.height) else 80.0
            mg = msg.mutual_gaze[i] if i < len(msg.mutual_gaze) else False
            
            self.faces.append({
                'id': fid,
                'u': int(c.x),
                'v': int(c.y),
                'w': int(w),
                'h': int(h),
                'mutual_gaze': mg
            })

    def on_saliency(self, msg: Float32MultiArray):
        """Handle multiple saliency peaks: [u1, v1, s1, u2, v2, s2, ...]"""
        self.saliency_peaks = []
        
        if len(msg.data) >= 3:
            for i in range(0, len(msg.data), 3):
                if i + 2 < len(msg.data):
                    self.saliency_peaks.append({
                        'u': int(msg.data[i]),
                        'v': int(msg.data[i + 1]),
                        'score': msg.data[i + 2]
                    })

    def on_target(self, msg: Vector3):
        self.current_target = {
            'yaw': msg.x,
            'pitch': msg.y,
            'score': msg.z
        }
        
        # Track switches
        current_id = f"{msg.x:.2f}_{msg.y:.2f}"
        if self.last_target_id and current_id != self.last_target_id:
            self.switch_times.append(time.time())
            # Keep only last 60s
            cutoff = time.time() - 60.0
            self.switch_times = [t for t in self.switch_times if t > cutoff]
        self.last_target_id = current_id
        
        self.target_history.append((msg.x, msg.y))

    def process_visualization(self, frame, stamp):
        """Main visualization pipeline"""
        now = time.time()
        dt = now - self.last_frame_t
        self.frame_times.append(dt)
        self.last_frame_t = now
        
        vis = frame.copy()
        H, W = vis.shape[:2]
        
        # Draw faces
        for face in self.faces:
            self._draw_face(vis, face)
        
        # Draw all saliency peaks
        for idx, sal in enumerate(self.saliency_peaks):
            self._draw_saliency(vis, sal, idx)
        
        # Draw current target
        if self.current_target and self.fx:
            self._draw_target(vis, self.current_target, W, H)
        
        # Draw metrics
        if self.show_metrics:
            self._draw_metrics(vis)
        
        # Publish overlay
        if self.publish_overlay and self.pub_overlay:
            enc = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
            msg = CompressedImage()
            msg.format = 'jpeg'
            msg.header.stamp = stamp
            msg.data = enc.tobytes()
            self.pub_overlay.publish(msg)
        
        # Publish RViz markers
        if self.publish_markers:
            self._publish_markers(stamp)

    def _draw_face(self, vis, face):
        """Draw face bounding box with labels"""
        u, v = face['u'], face['v']
        w2, h2 = face['w']//2, face['h']//2
        
        # Color based on mutual gaze
        color = (0, 255, 0) if face['mutual_gaze'] else (255, 100, 0)
        
        # Bounding box
        cv2.rectangle(
            vis,
            (u - w2, v - h2),
            (u + w2, v + h2),
            color, 2
        )
        
        # Center cross
        cv2.drawMarker(
            vis, (u, v), color,
            markerType=cv2.MARKER_CROSS,
            markerSize=12, thickness=2
        )
        
        # Label
        label = f"ID:{face['id']}"
        if face['mutual_gaze']:
            label += " [GAZE]"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            vis,
            (u - w2, v - h2 - th - 8),
            (u - w2 + tw + 8, v - h2),
            color, -1
        )
        cv2.putText(
            vis, label,
            (u - w2 + 4, v - h2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, cv2.LINE_AA
        )

    def _draw_saliency(self, vis, sal, rank=0):
        """Draw saliency peak with rank indicator"""
        u, v = sal['u'], sal['v']
        score = sal['score']
        
        # Colors by rank (top peak is brightest)
        colors = [
            (0, 255, 255),    # Yellow - highest
            (0, 200, 200),    # Darker yellow
            (0, 150, 150),    # Even darker
            (0, 100, 100),    # Dim
            (0, 80, 80)       # Dimmest
        ]
        color = colors[rank] if rank < len(colors) else (0, 60, 60)
        
        # Circle with score-based size
        radius = int(15 + score * 25) if rank == 0 else int(10 + score * 15)
        
        overlay = vis.copy()
        cv2.circle(overlay, (u, v), radius, color, 2)
        cv2.circle(overlay, (u, v), 3, color, -1)
        
        # Alpha decreases with rank
        alpha = max(0.2, 0.5 - rank * 0.1)
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        
        # Label (only for top peak or if high score)
        if rank == 0 or score > 0.5:
            label = f"SAL{rank+1}: {score:.2f}"
            cv2.putText(
                vis, label,
                (u + 20, v + rank * 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1, cv2.LINE_AA
            )

    def _draw_target(self, vis, target, W, H):
        """Draw current attention target"""
        yaw, pitch = target['yaw'], target['pitch']
        
        # Project back to pixel
        x_norm = np.tan(yaw)
        y_norm = np.tan(pitch)
        u = int(x_norm * self.fx + self.cx)
        v = int(y_norm * self.fy + self.cy)
        
        if 0 <= u < W and 0 <= v < H:
            # Target reticle
            color = (0, 0, 255)
            size = 30
            
            # Crosshair
            cv2.line(vis, (u - size, v), (u + size, v), color, 2)
            cv2.line(vis, (u, v - size), (u, v + size), color, 2)
            cv2.circle(vis, (u, v), size, color, 2)
            
            # Label
            label = f"TARGET ({yaw*180/np.pi:.1f}, {pitch*180/np.pi:.1f})"
            cv2.putText(
                vis, label,
                (u + 35, v - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2, cv2.LINE_AA
            )
        
        # Draw trajectory history
        if len(self.target_history) > 1:
            pts = []
            for y, p in self.target_history:
                x_n = np.tan(y)
                y_n = np.tan(p)
                pu = int(x_n * self.fx + self.cx)
                pv = int(y_n * self.fy + self.cy)
                if 0 <= pu < W and 0 <= pv < H:
                    pts.append((pu, pv))
            
            if len(pts) > 1:
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], False, (255, 0, 255), 1, cv2.LINE_AA)

    def _draw_metrics(self, vis):
        """Draw performance metrics overlay"""
        H, W = vis.shape[:2]
        
        # Semi-transparent background
        overlay = vis.copy()
        cv2.rectangle(overlay, (10, 10), (300, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
        
        # FPS
        fps = 1.0 / np.mean(self.frame_times) if len(self.frame_times) > 0 else 0
        
        # Switch rate (per minute)
        switch_rate = len(self.switch_times)
        
        # Metrics text
        y = 30
        metrics = [
            f"FPS: {fps:.1f}",
            f"Faces: {len(self.faces)}",
            f"Saliency peaks: {len(self.saliency_peaks)}",
            f"Switches/min: {switch_rate}",
            f"Target: {self.current_target['yaw']*180/np.pi:.1f}, {self.current_target['pitch']*180/np.pi:.1f}" if self.current_target else "Target: None"
        ]
        
        for text in metrics:
            cv2.putText(
                vis, text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA
            )
            y += 25

    def _publish_markers(self, stamp):
        """Publish RViz markers"""
        markers = MarkerArray()
        
        # Face markers
        for i, face in enumerate(self.faces):
            marker = Marker()
            marker.header.frame_id = "camera_color_optical_frame"
            marker.header.stamp = stamp
            marker.ns = "faces"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Position (approximate depth = 1.5m)
            if self.fx and self.cy:
                Z = 1.5
                X = (face['u'] - self.cx) / self.fx * Z
                Y = (face['v'] - self.cy) / self.fy * Z
                
                marker.pose.position.x = Z
                marker.pose.position.y = -X
                marker.pose.position.z = -Y
            
            marker.scale.x = marker.scale.y = marker.scale.z = 0.2
            
            # Color
            if face['mutual_gaze']:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            else:
                marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.6)
            
            markers.markers.append(marker)
        
        # Saliency markers (all peaks)
        for i, sal in enumerate(self.saliency_peaks):
            if self.fx:
                marker = Marker()
                marker.header.frame_id = "camera_color_optical_frame"
                marker.header.stamp = stamp
                marker.ns = "saliency"
                marker.id = i
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                Z = 2.0
                X = (sal['u'] - self.cx) / self.fx * Z
                Y = (sal['v'] - self.cy) / self.fy * Z
                
                marker.pose.position.x = Z
                marker.pose.position.y = -X
                marker.pose.position.z = -Y
                
                # Size decreases with rank
                scale = 0.3 - i * 0.05
                marker.scale.x = marker.scale.y = max(0.1, scale)
                marker.scale.z = 0.1
                
                # Alpha decreases with rank
                alpha = max(0.2, 0.6 - i * 0.1)
                marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=alpha)
                
                markers.markers.append(marker)
        
        # Target marker (arrow)
        if self.current_target:
            marker = Marker()
            marker.header.frame_id = "Head"
            marker.header.stamp = stamp
            marker.ns = "target"
            marker.id = 0
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Arrow from head origin in target direction
            yaw, pitch = self.current_target['yaw'], self.current_target['pitch']
            length = 0.5
            
            # Start point
            p1 = Point()
            p1.x = p1.y = p1.z = 0.0
            
            # End point
            p2 = Point()
            p2.x = length * np.cos(pitch) * np.cos(yaw)
            p2.y = length * np.cos(pitch) * np.sin(yaw)
            p2.z = length * np.sin(pitch)
            
            marker.points = [p1, p2]
            marker.scale.x = 0.02  # shaft diameter
            marker.scale.y = 0.04  # head diameter
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            
            markers.markers.append(marker)
        
        self.pub_markers.publish(markers)
        
        # Target pose for RViz camera plugin
        if self.current_target and self.pub_target_pose:
            pose = PoseStamped()
            pose.header.frame_id = "Head"
            pose.header.stamp = stamp
            
            yaw, pitch = self.current_target['yaw'], self.current_target['pitch']
            
            # Convert to quaternion (simplified - yaw around Z, pitch around Y)
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            
            pose.pose.orientation.w = cy * cp
            pose.pose.orientation.x = cy * sp
            pose.pose.orientation.y = sy * cp
            pose.pose.orientation.z = sy * sp
            
            self.pub_target_pose.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = AttentionVisualization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()