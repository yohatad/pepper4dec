#!/usr/bin/env python3
"""
Improved Attention Controller for Pepper Robot
Priority 1: Engaged Faces | Priority 2: Detected Faces | Priority 3: Saliency (with cooldown + IOR)
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
        self.declare_parameter("face_timeout", 2.0)  # Increased: stay on faces longer
        self.declare_parameter("engaged_priority_bonus", 2.0)  # Bonus for engaged faces
        self.declare_parameter("face_switch_cooldown", 1.0)  # Min time before switching faces
        self.declare_parameter("same_face_threshold_deg", 8.0)  # Threshold to consider same face
        self.declare_parameter("prefer_closer_faces", True)  # Prefer faces that are closer
        self.declare_parameter("max_face_distance", 5.0)  # Maximum distance to consider (meters)
        
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
        
        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        
        # Head state
        self._head_yaw = self._head_pitch = None
        
        # Face state
        self.last_face_time = 0.0
        self.last_face_switch_time = 0.0
        self.current_face_id = None  # Track which face we're currently looking at
        self.current_face_location = None  # (world_yaw, world_pitch)
        
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
        
        self.get_logger().info("Improved attention controller ready (Engaged Faces > Faces > Saliency + IOR)")

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

    def calculate_face_priority(self, centroid, mutual_gaze, face_id, is_current_face):
        """
        Calculate priority score for a face based on multiple factors.
        
        Priority factors:
        1. Engagement (mutual gaze) - highest priority
        2. Distance from center - prefer centered faces
        3. Depth - prefer closer faces (if enabled)
        4. Continuity - prefer staying on current face
        
        Returns: priority score (higher is better)
        """
        # Base score
        score = 1.0
        
        # Factor 1: Engagement bonus (most important)
        if mutual_gaze:
            score *= self.engaged_bonus
        
        # Factor 2: Distance from center (normalized)
        # Closer to center = higher priority
        dist_from_center = math.sqrt((centroid.x - self.cx)**2 + (centroid.y - self.cy)**2)
        max_dist = math.sqrt(self.cx**2 + self.cy**2)  # Corner distance
        center_score = 1.0 - (dist_from_center / max_dist)
        score *= (0.5 + 0.5 * center_score)  # Scale: 0.5 to 1.0
        
        # Factor 3: Depth bonus (if enabled and valid depth)
        if self.prefer_closer and centroid.z > 0:
            if centroid.z <= self.max_face_distance:
                # Closer faces get higher scores (inverse distance)
                # Map 0-max_distance to 1.5-0.5 bonus
                depth_bonus = 1.5 - (centroid.z / self.max_face_distance)
                score *= max(0.5, depth_bonus)
            else:
                # Too far, reduce priority
                score *= 0.3
        
        # Factor 4: Continuity bonus (prefer staying on current face)
        if is_current_face:
            time_since_switch = time.time() - self.last_face_switch_time
            if time_since_switch < self.face_switch_cooldown:
                # Strong preference to stay on current face during cooldown
                score *= 1.5
            else:
                # Mild preference even after cooldown
                score *= 1.1
        
        return score

    def on_faces(self, msg: FaceDetection):
        """Priority 1: Face detection with engagement awareness."""
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
            
            # Check if this is the face we're currently looking at
            is_current = (face_id == self.current_face_id)
            
            # Calculate priority score
            priority = self.calculate_face_priority(centroid, mutual_gaze, face_id, is_current)
            
            # Convert to world angles
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
            # No current face - always switch
            should_switch = True
            switch_reason = "initial"
        elif best_face['is_current']:
            # Same face - always update (refresh position)
            should_switch = True
            switch_reason = "refresh"
        else:
            # Different face - check cooldown and priority
            time_since_switch = current_time - self.last_face_switch_time
            
            # Find current face in candidates
            current_face = next((f for f in candidates if f['is_current']), None)
            
            if current_face is None:
                # Current face no longer detected
                should_switch = True
                switch_reason = "lost_current"
            elif time_since_switch < self.face_switch_cooldown:
                # In cooldown - only switch if much better priority
                if best_face['priority'] > current_face['priority'] * 1.5:
                    should_switch = True
                    switch_reason = "much_better"
            else:
                # After cooldown - switch if better priority
                if best_face['priority'] > current_face['priority'] * 1.1:
                    should_switch = True
                    switch_reason = "better"
        
        if not should_switch:
            # Just update last_face_time to prevent saliency from taking over
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
        """Priority 3: Saliency with cooldown + IOR (only when no faces detected)."""
        if self.fx is None or self._head_yaw is None:
            return
        
        # Check if we have recent faces - if so, ignore saliency
        if time.time() - self.last_face_time < self.face_timeout:
            return
        
        # Clear face state if we're switching to saliency
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
            
            # Convert to world angles
            cam_yaw, cam_pitch = pixel_to_angles(u, v, self.fx, self.fy, self.cx, self.cy)
            world_yaw = cam_yaw + self._head_yaw
            world_pitch = cam_pitch + self._head_pitch
            
            # Apply IOR
            score = self.apply_ior_filter(world_yaw, world_pitch, score)
            
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
            f"[{source}] → yaw={math.degrees(yaw):.1f}°, pitch={math.degrees(pitch):.1f}°, score={score:.2f}")


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