#!/usr/bin/env python3
"""
Simple Visualization for Pepper Attention System
Shows faces, saliency peaks, and current head target
"""

import cv2
import numpy as np
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32MultiArray
from cssr_interfaces.msg import FaceDetection
from cv_bridge import CvBridge


def get_image_qos() -> QoSProfile:
    """Get QoS profile suitable for image transport over WiFi."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1
    )


class SimpleVisualization(Node):
    def __init__(self):
        super().__init__('simple_visualization')
        
        # Parameters
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('image_topic_base', '/camera/color/image_raw_custom')
        self.declare_parameter('face_topic', '/faceDetection/data')
        self.declare_parameter('saliency_topic', '/attn/saliency_peak')
        self.declare_parameter('target_topic', '/attn/target_angles')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        
        # Load parameters
        self.use_compressed = self.get_parameter('use_compressed').value
        image_base = self.get_parameter('image_topic_base').value
        self.face_topic = self.get_parameter('face_topic').value
        self.saliency_topic = self.get_parameter('saliency_topic').value
        self.target_topic = self.get_parameter('target_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        
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
        self.pub_overlay = self.create_publisher(Image,'/attn/visualization', 10)
        
        # State
        self.bridge = CvBridge()
        self.faces = []
        self.saliency_peaks = []
        self.current_target = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # FPS tracking
        self.last_time = time.time()
        self.fps = 0.0
        
        # Counters for debugging
        self.image_count = 0
        self.face_count = 0
        self.saliency_count = 0
        self.target_count = 0
        
        self.get_logger().info("Simple visualization ready (Faces + Saliency)")

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
        """Store face detections."""
        self.face_count += 1
        self.faces = []
        
        n = len(msg.centroids)
        for i in range(n):
            u = msg.centroids[i].x
            v = msg.centroids[i].y
            w = msg.width[i] if i < len(msg.width) else 80
            h = msg.height[i] if i < len(msg.height) else 80
            
            self.faces.append({
                'u': int(u),
                'v': int(v),
                'w': int(w),
                'h': int(h)
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
        """Store current target angles."""
        self.target_count += 1
        self.current_target = {
            'yaw': msg.x,
            'pitch': msg.y,
            'score': msg.z
        }
        
        if self.target_count == 1:
            self.get_logger().info(
                f"First target received: yaw={np.degrees(msg.x):.1f}°, "
                f"pitch={np.degrees(msg.y):.1f}°"
            )

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
        
        # Calculate FPS
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_time = now
        
        # Draw visualizations
        vis = frame.copy()
        H, W = vis.shape[:2]
        
        # Draw faces (Priority 1)
        for face in self.faces:
            self._draw_face(vis, face)
        
        # Draw saliency peaks (Priority 2)
        for i, peak in enumerate(self.saliency_peaks):
            self._draw_saliency_peak(vis, peak, i)
        
        # Draw current target
        if self.current_target and self.fx is not None:
            self._draw_target(vis, self.current_target, W, H)
        
        # Draw info overlay
        self._draw_info(vis)
        
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

    def _draw_face(self, vis, face):
        """Draw face bounding box."""
        u, v = face['u'], face['v']
        w2, h2 = face['w'] // 2, face['h'] // 2
        
        color = (0, 255, 0)  # Green
        
        # Bounding box
        cv2.rectangle(
            vis,
            (u - w2, v - h2),
            (u + w2, v + h2),
            color, 3
        )
        
        # Center cross
        cv2.drawMarker(
            vis, (u, v), color,
            markerType=cv2.MARKER_CROSS,
            markerSize=15, thickness=3
        )
        
        # Label
        cv2.putText(
            vis, "FACE",
            (u - w2, v - h2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            color, 2, cv2.LINE_AA
        )

    def _draw_saliency_peak(self, vis, peak, rank):
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

    def _draw_target(self, vis, target, W, H):
        """Draw current attention target."""
        yaw, pitch = target['yaw'], target['pitch']
        score = target.get('score', 0.0)
        
        # Project angles back to pixel coordinates
        # NOTE: yaw is negated in pixel_to_angles, so negate it back here
        x_norm = -np.tan(yaw)  # ← ADD NEGATIVE HERE
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
        
        cv2.putText(
            vis, label,
            (u + 50, v - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            color, 2, cv2.LINE_AA
        )

    def _draw_info(self, vis):
        """Draw info overlay."""
        H, W = vis.shape[:2]
        
        # Semi-transparent background
        overlay = vis.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        # Info text
        y = 35
        info = [
            f"FPS: {self.fps:.1f}",
            f"Images: {self.image_count}",
            f"Faces: {len(self.faces)}",
            f"Saliency peaks: {len(self.saliency_peaks)}",
            f"Target: {'Yes' if self.current_target else 'No'}"
        ]
        
        for text in info:
            cv2.putText(
                vis, text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA
            )
            y += 30


def main(args=None):
    rclpy.init(args=args)
    node = SimpleVisualization()
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