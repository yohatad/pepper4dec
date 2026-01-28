#!/usr/bin/env python3
"""
Simple Attention Controller for Pepper Robot
Priority 1: Faces | Priority 2: Saliency (with cooldown + IOR)
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CameraInfo, JointState
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from geometry_msgs.msg import Vector3
from cssr_interfaces.msg import FaceDetection


def get_image_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1
    )


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def pixel_to_angles(u, v, fx, fy, cx, cy):
    """Convert pixel to camera-relative angles."""
    x, y = (u - cx) / fx, (v - cy) / fy
    return -math.atan2(x, 1.0), math.atan2(y, 1.0)


class SimpleAttention(Node):
    def __init__(self):
        super().__init__("simple_attention")
        
        # Topics
        self.declare_parameter("face_topic", "/faceDetection/data")
        self.declare_parameter("saliency_topic", "/attn/saliency_peak")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("head_command_topic", "/joint_angles")
        self.declare_parameter("target_topic", "/attn/target_angles")
        
        # Joint limits
        self.declare_parameter("yaw_lim", 1.8)
        self.declare_parameter("pitch_up", 0.4)
        self.declare_parameter("pitch_dn", -0.7)
        
        # Face parameters
        self.declare_parameter("face_timeout", 0.5)
        self.declare_parameter("saliency_min_score", 0.30)
        
        # Cooldown parameters
        self.declare_parameter("saliency_min_cooldown", 1.5)
        self.declare_parameter("saliency_max_dwell", 3.0)
        self.declare_parameter("switch_score_ratio", 1.4)
        self.declare_parameter("same_target_threshold_deg", 5.0)
        
        # IOR parameters
        self.declare_parameter("enable_ior", True)
        self.declare_parameter("ior_max_suppression", 0.9)
        self.declare_parameter("ior_half_life", 3.0)
        self.declare_parameter("ior_radius_deg", 15.0)
        self.declare_parameter("ior_cleanup_threshold", 0.05)
        self.declare_parameter("ior_max_locations", 20)
        
        # Load parameters
        self.face_topic = self.get_parameter("face_topic").value
        self.saliency_topic = self.get_parameter("saliency_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.head_cmd_topic = self.get_parameter("head_command_topic").value
        self.target_topic = self.get_parameter("target_topic").value
        
        self.yaw_lim = self.get_parameter("yaw_lim").value
        self.pitch_up = self.get_parameter("pitch_up").value
        self.pitch_dn = self.get_parameter("pitch_dn").value
        
        self.face_timeout = self.get_parameter("face_timeout").value
        self.saliency_min = self.get_parameter("saliency_min_score").value
        
        self.min_cooldown = self.get_parameter("saliency_min_cooldown").value
        self.max_dwell = self.get_parameter("saliency_max_dwell").value
        self.switch_ratio = self.get_parameter("switch_score_ratio").value
        self.same_target_threshold = math.radians(self.get_parameter("same_target_threshold_deg").value)
        
        self.enable_ior = self.get_parameter("enable_ior").value
        self.ior_max_suppression = self.get_parameter("ior_max_suppression").value
        self.ior_half_life = self.get_parameter("ior_half_life").value
        self.ior_radius = math.radians(self.get_parameter("ior_radius_deg").value)
        self.ior_cleanup_threshold = self.get_parameter("ior_cleanup_threshold").value
        self.ior_max_locations = self.get_parameter("ior_max_locations").value
        
        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        
        # Head state
        self._head_yaw = self._head_pitch = None
        
        # Face state
        self.last_face_time = 0.0
        
        # Saliency state (store in world angles)
        self.last_saliency_cmd_time = 0.0
        self.current_saliency_target = None  # (world_yaw, world_pitch)
        self.current_saliency_score = 0.0
        
        # IOR state: (world_yaw, world_pitch, timestamp)
        self.visited_locations = []
        
        # Subscriptions
        qos_img = get_image_qos()
        self.create_subscription(FaceDetection, self.face_topic, self.on_faces, 10)
        self.create_subscription(Float32MultiArray, self.saliency_topic, self.on_saliency, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_caminfo, qos_img)
        self.create_subscription(JointState, '/joint_states', self.on_joint_states, 10)
        
        
        # Publishers
        self.pub_head = self.create_publisher(JointAnglesWithSpeed, self.head_cmd_topic, 10)
        self.pub_target = self.create_publisher(Vector3, self.target_topic, 10) 
        
        self.get_logger().info("Attention controller ready (Faces + Saliency + IOR)")

    def on_caminfo(self, msg: CameraInfo):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(f"Camera: fx={self.fx:.1f}, fy={self.fy:.1f}")

    def on_joint_states(self, msg: JointState):
        try:
            self._head_yaw = msg.position[msg.name.index('HeadYaw')]
            self._head_pitch = msg.position[msg.name.index('HeadPitch')]
        except (ValueError, IndexError):
            pass

    def on_faces(self, msg: FaceDetection):
        """Priority 1: Face detection."""
        if self.fx is None or self._head_yaw is None:
            return
        
        if not msg.centroids:
            return
        
        # Pick face closest to center
        # FIX: min() returns a Point object, not a tuple
        best_centroid = min(msg.centroids, 
                            key=lambda c: (c.x - self.cx)**2 + (c.y - self.cy)**2)
        
        # Extract coordinates from the Point object
        u = best_centroid.x
        v = best_centroid.y
        
        self.last_face_time = time.time()
        
        # Convert to world angles
        cam_yaw, cam_pitch = pixel_to_angles(u, v, self.fx, self.fy, self.cx, self.cy)
        world_yaw = clamp(cam_yaw + self._head_yaw, -self.yaw_lim, self.yaw_lim)
        world_pitch = clamp(cam_pitch + self._head_pitch, self.pitch_dn, self.pitch_up)
        
        self.publish_head(world_yaw, world_pitch, score=1.0, source='face')

    def _calculate_ior_suppression(self, age_seconds):
        """Exponential decay: suppression = max * exp(-ln(2) * age / half_life)"""
        decay = math.log(2) / self.ior_half_life
        return self.ior_max_suppression * math.exp(-decay * age_seconds)

    def _apply_ior_filter(self, world_yaw, world_pitch, score):
        """Apply IOR suppression (expects world coordinates)."""
        if not self.enable_ior or not self.visited_locations:
            return score
        
        current_time = time.time()
        max_suppression = 0.0
        
        for v_yaw, v_pitch, timestamp in self.visited_locations:
            age = current_time - timestamp
            dist = math.sqrt((world_yaw - v_yaw)**2 + (world_pitch - v_pitch)**2)
            
            if dist < self.ior_radius:
                time_supp = self._calculate_ior_suppression(age)
                space_decay = 1.0 - (dist / self.ior_radius)
                max_suppression = max(max_suppression, time_supp * space_decay)
        
        return score * (1.0 - max_suppression)

    def _cleanup_weak_ior(self):
        """Remove locations with negligible suppression."""
        if not self.enable_ior:
            return
        
        current_time = time.time()
        self.visited_locations = [
            (yaw, pitch, ts) 
            for yaw, pitch, ts in self.visited_locations
            if self._calculate_ior_suppression(current_time - ts) >= self.ior_cleanup_threshold
        ][:self.ior_max_locations]

    def on_saliency(self, msg: Float32MultiArray):
        """Priority 2: Saliency with cooldown + IOR."""
        if self.fx is None or self._head_yaw is None:
            return
        
        if time.time() - self.last_face_time < self.face_timeout:
            return
        
        if len(msg.data) < 3:
            return
        
        self._cleanup_weak_ior()
        
        # Convert all candidates to world angles with IOR
        candidates = []
        for i in range(0, len(msg.data) - 2, 3):
            u, v, score = msg.data[i], msg.data[i + 1], msg.data[i + 2]
            
            if score < self.saliency_min:
                continue
            
            # Convert to world angles
            cam_yaw, cam_pitch = pixel_to_angles(u, v, self.fx, self.fy, self.cx, self.cy)
            world_yaw = cam_yaw + self._head_yaw
            world_pitch = cam_pitch + self._head_pitch
            
            # Apply IOR
            score = self._apply_ior_filter(world_yaw, world_pitch, score)
            
            if score >= self.saliency_min:
                candidates.append((world_yaw, world_pitch, score))
        
        if not candidates:
            return
        
        # Find best candidate
        best_yaw, best_pitch, best_score = max(candidates, key=lambda x: x[2])
        
        # Check if same target as current
        is_same = False
        if self.current_saliency_target:
            curr_yaw, curr_pitch = self.current_saliency_target
            dist = math.sqrt((best_yaw - curr_yaw)**2 + (best_pitch - curr_pitch)**2)
            is_same = dist < self.same_target_threshold
        
        # Cooldown logic
        time_on_target = time.time() - self.last_saliency_cmd_time
        
        if not self.current_saliency_target:
            should_switch = True
            reason = "initial"
        elif is_same:
            should_switch = True
            reason = "refresh"
        elif time_on_target < self.min_cooldown:
            should_switch = best_score > self.current_saliency_score * self.switch_ratio
            reason = "early" if should_switch else None
        elif time_on_target > self.max_dwell:
            should_switch = not is_same
            reason = "max_dwell" if should_switch else None
        else:
            should_switch = best_score > self.current_saliency_score * 1.15
            reason = "better" if should_switch else None
        
        if not should_switch:
            return
        
        # Update state
        self.current_saliency_target = (best_yaw, best_pitch)
        self.current_saliency_score = best_score
        self.last_saliency_cmd_time = time.time()
        
        # Add to IOR
        if self.enable_ior:
            self.visited_locations.append((best_yaw, best_pitch, time.time()))
        
        # Clamp and publish
        yaw = clamp(best_yaw, -self.yaw_lim, self.yaw_lim)
        pitch = clamp(best_pitch, self.pitch_dn, self.pitch_up)
        self.publish_head(yaw, pitch, score=best_score, source=f'saliency({reason})')

    def publish_head(self, yaw, pitch, score=0.0, source='unknown'):
        """Send absolute head command and publish camera-relative target for visualization."""
        msg = JointAnglesWithSpeed()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ['HeadYaw', 'HeadPitch']
        msg.joint_angles = [float(yaw), float(pitch)]
        msg.speed = 0.1
        msg.relative = False
        
        self.pub_head.publish(msg)
        
        # Publish target for visualization (convert back to camera-relative)
        if self._head_yaw is not None and self._head_pitch is not None:
            cam_relative_yaw = yaw - self._head_yaw
            cam_relative_pitch = pitch - self._head_pitch
        else:
            cam_relative_yaw = yaw
            cam_relative_pitch = pitch
        
        target_msg = Vector3()
        target_msg.x = float(cam_relative_yaw)
        target_msg.y = float(cam_relative_pitch)
        target_msg.z = float(score)
        self.pub_target.publish(target_msg)
        
        self.get_logger().info(
            f"[{source}] → yaw={math.degrees(yaw):.1f}°, pitch={math.degrees(pitch):.1f}°, score={score:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleAttention()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()