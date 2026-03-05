#!/usr/bin/env python3
"""
Improved Attention Controller for Robot Overt Attention
Priority 1: Engaged Faces | Priority 2: Detected Faces | Priority 3: Saliency (with cooldown + IOR)
"""

import math
import time
import yaml
from pathlib import Path
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import CameraInfo, JointState
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from geometry_msgs.msg import Vector3
from dec_interfaces.msg import FaceDetection
from std_srvs.srv import SetBool
from ament_index_python.packages import get_package_share_directory


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

def load_topics_config(package_name: str, relative_path: str) -> dict:
    """Load topics configuration from YAML file using ROS2 package path."""
    package_share = get_package_share_directory(package_name)
    config_path = Path(package_share) / relative_path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class OvertAttention(Node):
    def __init__(self):
        super().__init__("simple_attention")
        
        # Load topics configuration from YAML file using ROS2 package path
        try:
            self.topics_config = load_topics_config('overt_attention', 'data/pepper_topics.yaml')
            self.get_logger().info("Loaded topics configuration from overt_attention package")
        except Exception as e:
            self.get_logger().error(f"Failed to load topics configuration: {e}")
            raise
        
        # System parameters
        self.declare_parameter("start_enabled", True)  # Start with attention enabled
        self.declare_parameter("move_to_default_on_disable", True)  # Move to default position when disabled
        self.declare_parameter("default_yaw", 0.0)  # Default head yaw position (radians)
        self.declare_parameter("default_pitch", -0.2)  # Default head pitch position (radians)
        self.declare_parameter("default_move_speed", 0.1)  # Speed for moving to default position
        
        # Joint limits
        self.declare_parameter("yaw_lim", 1.8)
        self.declare_parameter("pitch_up", 0.4)
        self.declare_parameter("pitch_dn", -0.7)
        
        # Face parameters
        self.declare_parameter("face_timeout", 2.0)
        self.declare_parameter("engaged_priority_bonus", 2.0)
        self.declare_parameter("face_switch_cooldown", 1.0)
        self.declare_parameter("same_face_threshold_deg", 8.0)
        self.declare_parameter("prefer_closer_faces", True)
        self.declare_parameter("max_face_distance", 5.0)
        
        # Saliency parameters
        self.declare_parameter("saliency_min_score", 0.30)
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
        
        # Load topics from YAML config
        self.face_topic = self.topics_config['topics']['face']
        self.saliency_topic = self.topics_config['topics']['saliency']['peak']
        self.camera_info_topic = self.topics_config['topics']['camera_info']
        self.head_cmd_topic = self.topics_config['topics']['joint_angles']
        self.target_topic = self.topics_config['topics']['target_angles']
        
        # Load parameters
        self.move_to_default_on_disable = self.get_parameter("move_to_default_on_disable").value
        self.default_yaw = self.get_parameter("default_yaw").value
        self.default_pitch = self.get_parameter("default_pitch").value
        self.default_move_speed = self.get_parameter("default_move_speed").value
        
        self.yaw_lim = self.get_parameter("yaw_lim").value
        self.pitch_up = self.get_parameter("pitch_up").value
        self.pitch_dn = self.get_parameter("pitch_dn").value
        
        self.face_timeout = self.get_parameter("face_timeout").value
        self.engaged_bonus = self.get_parameter("engaged_priority_bonus").value
        self.face_switch_cooldown = self.get_parameter("face_switch_cooldown").value
        self.same_face_threshold = math.radians(self.get_parameter("same_face_threshold_deg").value)
        self.prefer_closer = self.get_parameter("prefer_closer_faces").value
        self.max_face_distance = self.get_parameter("max_face_distance").value
        
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
        
        # Enable/disable state
        self.attention_enabled = self.get_parameter("start_enabled").value
        
        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        
        # Head state
        self._head_yaw = self._head_pitch = None
        
        # Face state
        self.last_face_time = 0.0
        self.last_face_switch_time = 0.0
        self.current_face_id = None
        self.current_face_location = None
        
        # Saliency state
        self.last_saliency_cmd_time = 0.0
        self.current_saliency_target = None
        self.current_saliency_score = 0.0
        
        # IOR state
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
        
        # Services
        self.srv_enable = self.create_service(
            SetBool,
            '/attn/set_enabled',
            self.handle_set_enabled
        )
        
        status = "ENABLED" if self.attention_enabled else "DISABLED"
        default_mode = "move to default" if self.move_to_default_on_disable else "hold position"
        self.get_logger().info(f"Improved attention controller ready ({status})")
        self.get_logger().info(f"Service: /attn/set_enabled (std_srvs/SetBool)")
        self.get_logger().info(f"Disable mode: {default_mode} (yaw={math.degrees(self.default_yaw):.1f}°, pitch={math.degrees(self.default_pitch):.1f}°)")

    def handle_set_enabled(self, request, response):
        """Service callback to enable/disable attention system."""
        old_state = self.attention_enabled
        self.attention_enabled = request.data
        
        if old_state != self.attention_enabled:
            if self.attention_enabled:
                self.get_logger().info("Attention system ENABLED")
                response.success = True
                response.message = "Attention system enabled"
            else:
                # Clear all tracking state
                self.current_face_id = None
                self.current_face_location = None
                self.current_saliency_target = None
                self.visited_locations = []
                
                if self.move_to_default_on_disable:
                    # Move to default position
                    self.move_head_to_default()
                    self.get_logger().info(
                        f"Attention system DISABLED - moving to default position "
                        f"(yaw={math.degrees(self.default_yaw):.1f}°, pitch={math.degrees(self.default_pitch):.1f}°)"
                    )
                    response.success = True
                    response.message = (
                        f"Attention system disabled - moving to default position "
                        f"(yaw={math.degrees(self.default_yaw):.1f}°, pitch={math.degrees(self.default_pitch):.1f}°)"
                    )
                else:
                    # Hold current position
                    current_yaw = self._head_yaw if self._head_yaw is not None else 0.0
                    current_pitch = self._head_pitch if self._head_pitch is not None else 0.0
                    self.get_logger().info(
                        f"Attention system DISABLED - holding current position "
                        f"(yaw={math.degrees(current_yaw):.1f}°, pitch={math.degrees(current_pitch):.1f}°)"
                    )
                    response.success = True
                    response.message = (
                        f"Attention system disabled - holding position at "
                        f"yaw={math.degrees(current_yaw):.1f}°, pitch={math.degrees(current_pitch):.1f}°"
                    )
        else:
            status = "enabled" if self.attention_enabled else "disabled"
            response.success = True
            response.message = f"Attention system already {status}"
        
        return response

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

    def move_head_to_default(self):
        """Move head to default position."""
        msg = JointAnglesWithSpeed()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ['HeadYaw', 'HeadPitch']
        msg.joint_angles = [float(self.default_yaw), float(self.default_pitch)]
        msg.speed = float(self.default_move_speed)
        msg.relative = False
        
        self.pub_head.publish(msg)
        
        # Also publish to target topic for visualization
        target_msg = Vector3()
        target_msg.x = 0.0  # Camera-relative (default is centered)
        target_msg.y = 0.0
        target_msg.z = 0.0
        self.pub_target.publish(target_msg)

    def calculate_face_priority(self, centroid, mutual_gaze, face_id, is_current_face):
        """Calculate priority score for a face based on multiple factors."""
        # Base score
        score = 1.0
        
        # Factor 1: Engagement bonus
        if mutual_gaze:
            score *= self.engaged_bonus
        
        # Factor 2: Distance from center
        dist_from_center = math.sqrt((centroid.x - self.cx)**2 + (centroid.y - self.cy)**2)
        max_dist = math.sqrt(self.cx**2 + self.cy**2)
        center_score = 1.0 - (dist_from_center / max_dist)
        score *= (0.5 + 0.5 * center_score)
        
        # Factor 3: Depth bonus
        if self.prefer_closer and centroid.z > 0:
            if centroid.z <= self.max_face_distance:
                depth_bonus = 1.5 - (centroid.z / self.max_face_distance)
                score *= max(0.5, depth_bonus)
            else:
                score *= 0.3
        
        # Factor 4: Continuity bonus
        if is_current_face:
            time_since_switch = time.time() - self.last_face_switch_time
            if time_since_switch < self.face_switch_cooldown:
                score *= 1.5
            else:
                score *= 1.1
        
        return score

    def on_faces(self, msg: FaceDetection):
        """Priority 1: Face detection with engagement awareness."""
        # Check if attention is enabled
        if not self.attention_enabled:
            return
        
        if self.fx is None or self._head_yaw is None or self._head_pitch is None:
            return
        
        if not msg.centroids:
            # No faces detected - clear state
            if self.current_face_id is not None:
                self.get_logger().info(f"Lost face: {self.current_face_id}")
                self.current_face_id = None
                self.current_face_location = None
            return
        
        current_time = time.time()
        
        # Build candidate list with priorities
        candidates = []
        for i, centroid in enumerate(msg.centroids):
            face_id = msg.face_label_id[i] if i < len(msg.face_label_id) else f"unknown_{i}"
            mutual_gaze = msg.mutual_gaze[i] if i < len(msg.mutual_gaze) else False
            
            is_current = (face_id == self.current_face_id)
            priority = self.calculate_face_priority(centroid, mutual_gaze, face_id, is_current)
            
            cam_yaw, cam_pitch = pixel_to_angles(centroid.x, centroid.y, 
                                                 self.fx, self.fy, self.cx, self.cy)
            world_yaw = cam_yaw + self._head_yaw
            world_pitch = cam_pitch + self._head_pitch
            
            candidates.append({
                'face_id': face_id,
                'centroid': centroid,
                'world_yaw': world_yaw,
                'world_pitch': world_pitch,
                'mutual_gaze': mutual_gaze,
                'priority': priority,
                'is_current': is_current
            })
        
        # Select best face
        best_face = max(candidates, key=lambda f: f['priority'])
        
        # Check if we should switch faces
        should_switch = False
        switch_reason = ""
        
        if self.current_face_id is None:
            should_switch = True
            switch_reason = "initial"
        elif best_face['is_current']:
            should_switch = True
            switch_reason = "refresh"
        else:
            time_since_switch = current_time - self.last_face_switch_time
            current_face = next((f for f in candidates if f['is_current']), None)
            
            if current_face is None:
                should_switch = True
                switch_reason = "lost_current"
            elif time_since_switch < self.face_switch_cooldown:
                if best_face['priority'] > current_face['priority'] * 1.5:
                    should_switch = True
                    switch_reason = "much_better"
            else:
                if best_face['priority'] > current_face['priority'] * 1.1:
                    should_switch = True
                    switch_reason = "better"
        
        if not should_switch:
            self.last_face_time = current_time
            return
        
        # Update state
        is_new_face = (best_face['face_id'] != self.current_face_id)
        
        if is_new_face:
            self.last_face_switch_time = current_time
            self.get_logger().info(
                f"Switching to face: {best_face['face_id']} "
                f"(engaged={best_face['mutual_gaze']}, "
                f"depth={best_face['centroid'].z:.2f}m, "
                f"priority={best_face['priority']:.2f}, "
                f"reason={switch_reason})"
            )
        
        self.current_face_id = best_face['face_id']
        self.current_face_location = (best_face['world_yaw'], best_face['world_pitch'])
        self.last_face_time = current_time
        
        # Clamp and publish
        yaw = clamp(best_face['world_yaw'], -self.yaw_lim, self.yaw_lim)
        pitch = clamp(best_face['world_pitch'], self.pitch_dn, self.pitch_up)
        
        source = f"face[{best_face['face_id']}]"
        if best_face['mutual_gaze']:
            source += "_engaged"
        if switch_reason:
            source += f"({switch_reason})"
        
        self.publish_head(yaw, pitch, score=best_face['priority'], source=source)

    def calculate_ior_suppression(self, age_seconds):
        """Exponential decay: suppression = max * exp(-ln(2) * age / half_life)"""
        decay = math.log(2) / self.ior_half_life
        return self.ior_max_suppression * math.exp(-decay * age_seconds)

    def apply_ior_filter(self, world_yaw, world_pitch, score):
        """Apply IOR suppression (expects world coordinates)."""
        if not self.enable_ior or not self.visited_locations:
            return score
        
        current_time = time.time()
        max_suppression = 0.0
        
        for v_yaw, v_pitch, timestamp in self.visited_locations:
            age = current_time - timestamp
            dist = math.sqrt((world_yaw - v_yaw)**2 + (world_pitch - v_pitch)**2)
            
            if dist < self.ior_radius:
                time_supp = self.calculate_ior_suppression(age)
                space_decay = 1.0 - (dist / self.ior_radius)
                max_suppression = max(max_suppression, time_supp * space_decay)
        
        return score * (1.0 - max_suppression)

    def cleanup_weak_ior(self):
        """Remove locations with negligible suppression."""
        if not self.enable_ior:
            return
        
        current_time = time.time()
        self.visited_locations = [
            (yaw, pitch, ts) 
            for yaw, pitch, ts in self.visited_locations
            if self.calculate_ior_suppression(current_time - ts) >= self.ior_cleanup_threshold
        ][:self.ior_max_locations]

    def on_saliency(self, msg: Float32MultiArray):
        """Priority 3: Saliency with cooldown + IOR."""
        # Check if attention is enabled
        if not self.attention_enabled:
            return
        
        if self.fx is None or self._head_yaw is None:
            return
        
        if time.time() - self.last_face_time < self.face_timeout:
            return
        
        if self.current_face_id is not None:
            self.get_logger().info(f"No recent faces, switching to saliency (was tracking: {self.current_face_id})")
            self.current_face_id = None
            self.current_face_location = None
        
        if len(msg.data) < 3:
            return
        
        self.cleanup_weak_ior()
        
        # Convert all candidates to world angles with IOR
        candidates = []
        for i in range(0, len(msg.data) - 2, 3):
            u, v, score = msg.data[i], msg.data[i + 1], msg.data[i + 2]
            
            if score < self.saliency_min:
                continue
            
            cam_yaw, cam_pitch = pixel_to_angles(u, v, self.fx, self.fy, self.cx, self.cy)
            world_yaw = cam_yaw + self._head_yaw
            world_pitch = cam_pitch + self._head_pitch
            
            score = self.apply_ior_filter(world_yaw, world_pitch, score)
            
            if score >= self.saliency_min:
                candidates.append((world_yaw, world_pitch, score))
        
        if not candidates:
            return
        
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
        
        # Publish target for visualization
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
            f"[{source}] → yaw={math.degrees(yaw):.1f}°, pitch={math.degrees(pitch):.1f}°, score={score:.2f}")


def main(args=None):
    rclpy.init(args=args)
    node = OvertAttention()
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