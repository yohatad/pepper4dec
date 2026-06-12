#!/usr/bin/env python3
""" overt_attention_unified_attention.py

Entry point and lifecycle node implementation for the unified attention controller.
Fuses face detections and bottom-up saliency into a single gaze target and commands
Pepper's head to look at it.

This node arbitrates between three priority levels — engaged (mutual-gaze) faces,
detected faces, and salient regions — to decide where the robot should look next.
Face candidates are scored by engagement, centeredness, and proximity, with hysteresis
and cooldowns to avoid rapid target switching. When no face has been seen recently,
the node falls back to the highest-scoring saliency peak, applying inhibition-of-return
(IOR) so recently-attended locations are temporarily suppressed. The resulting gaze
direction is smoothed and published as a head joint command, along with the
camera-relative target angles used by the visualization node.

Subscribers:
    /face_detection/data (dec_interfaces/FaceDetection)
        Detected face centroids, tracking IDs, mutual-gaze flags, and depth.
    /overt_attention/saliency_peak (std_msgs/Float32MultiArray)
        Saliency peak candidates as flattened (u, v, score) triplets.
    /camera/color/camera_info (sensor_msgs/CameraInfo)
        Camera intrinsics used to convert pixel coordinates to angles.
    /joint_states (sensor_msgs/JointState)
        Current head joint positions (HeadYaw, HeadPitch).

Publishers:
    /joint_angles (naoqi_bridge_msgs/JointAnglesWithSpeed)
        Commanded head joint angles and motion speed.
    /overt_attention/target_angles (geometry_msgs/Vector3)
        Camera-relative target yaw/pitch and the attention score of the current target.

Services:
    /overt_attention/set_enabled (std_srvs/SetBool)
        Enable or disable the attention controller; optionally moves the head to its
        default pose on disable.

Parameters (loaded from overt_attention_configuration.yaml):
    start_enabled (bool, default: true)
    move_to_default_on_disable (bool, default: true)
    default_yaw (float, default: 0.0)
    default_pitch (float, default: -0.2)
    default_move_speed (float, default: 0.1)
    face_yaw_lim (float, default: 1.8)
    face_pitch_up (float, default: 0.4)
    face_pitch_dn (float, default: -0.7)
    saliency_yaw_lim (float, default: 1.2)
    saliency_pitch_up (float, default: 0.3)
    saliency_pitch_dn (float, default: -0.3)
    face_timeout (float, default: 2.0)
    engaged_priority_bonus (float, default: 2.0)
    face_switch_cooldown (float, default: 1.0)
    same_face_threshold_deg (float, default: 8.0)
    prefer_closer_faces (bool, default: true)
    max_face_distance (float, default: 5.0)
    min_angular_change_deg (float, default: 2.0)
    target_smoothing_alpha (float, default: 0.4)
    saliency_min_score (float, default: 0.30)
    saliency_min_cooldown (float, default: 1.5)
    saliency_max_dwell (float, default: 3.0)
    switch_score_ratio (float, default: 1.4)
    same_target_threshold_deg (float, default: 5.0)
    enable_ior (bool, default: true)
    ior_max_suppression (float, default: 0.9)
    ior_half_life (float, default: 3.0)
    ior_radius_deg (float, default: 15.0)
    ior_cleanup_threshold (float, default: 0.05)
    ior_max_locations (int, default: 20)

Lifecycle:
    configure  -> load topics YAML, declare/read parameters, init internal state,
                  and create the head/target lifecycle publishers and the
                  set_enabled service.
    activate   -> activate publishers and subscribe to face, saliency, camera_info,
                  and joint_state topics.
    deactivate -> destroy sensor subscriptions, clear tracking state, and
                  deactivate publishers.
    cleanup    -> destroy the lifecycle publishers and the set_enabled service.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: June 11, 2026
Version: v1.0
"""

import math
import time
import yaml
import rclpy
from pathlib import Path
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
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
    config_path   = Path(package_share) / relative_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class OvertAttention(LifecycleNode):
    """Lifecycle node that fuses face and saliency input into head gaze commands."""

    def __init__(self):
        super().__init__('simple_attention')

    # ── Lifecycle callbacks ─────────────────────────────────────────────────────

    def on_configure(self, _state) -> TransitionCallbackReturn:
        """Load the topics config, declare/read parameters, init state, and create publishers and the enable service."""
        try:
            self.topics_config = load_topics_config('overt_attention', 'data/pepper_topics.yaml')
            self.get_logger().info('Loaded topics configuration from overt_attention package')
        except Exception as e:
            self.get_logger().error(f'Failed to load topics configuration: {e}')
            return TransitionCallbackReturn.FAILURE

        # Declare parameters
        self.declare_parameter('start_enabled',            True)
        self.declare_parameter('move_to_default_on_disable', True)
        self.declare_parameter('default_yaw',              0.0)
        self.declare_parameter('default_pitch',            -0.2)
        self.declare_parameter('default_move_speed',       0.1)
        self.declare_parameter('face_yaw_lim',             1.8)
        self.declare_parameter('face_pitch_up',            0.4)
        self.declare_parameter('face_pitch_dn',            -0.7)
        self.declare_parameter('saliency_yaw_lim',         1.8)
        self.declare_parameter('saliency_pitch_up',        0.4)
        self.declare_parameter('saliency_pitch_dn',        -0.7)
        self.declare_parameter('face_timeout',             2.0)
        self.declare_parameter('engaged_priority_bonus',   2.0)
        self.declare_parameter('face_switch_cooldown',     1.0)
        self.declare_parameter('same_face_threshold_deg',  8.0)
        self.declare_parameter('prefer_closer_faces',      True)
        self.declare_parameter('max_face_distance',        5.0)
        self.declare_parameter('min_angular_change_deg',   2.0)
        self.declare_parameter('target_smoothing_alpha',   0.4)
        self.declare_parameter('saliency_min_score',       0.30)
        self.declare_parameter('saliency_min_cooldown',    1.5)
        self.declare_parameter('saliency_max_dwell',       3.0)
        self.declare_parameter('switch_score_ratio',       1.4)
        self.declare_parameter('same_target_threshold_deg', 5.0)
        self.declare_parameter('enable_ior',               True)
        self.declare_parameter('ior_max_suppression',      0.9)
        self.declare_parameter('ior_half_life',            3.0)
        self.declare_parameter('ior_radius_deg',           15.0)
        self.declare_parameter('ior_cleanup_threshold',    0.05)
        self.declare_parameter('ior_max_locations',        20)

        # Read topic names from YAML
        self.face_topic       = self.topics_config['topics']['face']
        self.saliency_topic   = self.topics_config['topics']['saliency']['peak']
        self.camera_info_topic = self.topics_config['topics']['camera_info']
        self.head_cmd_topic   = self.topics_config['topics']['joint_angles']
        self.target_topic     = self.topics_config['topics']['target_angles']

        # Read parameters
        self.move_to_default_on_disable = self.get_parameter('move_to_default_on_disable').value
        self.default_yaw       = self.get_parameter('default_yaw').value
        self.default_pitch     = self.get_parameter('default_pitch').value
        self.default_move_speed = self.get_parameter('default_move_speed').value
        self.face_yaw_lim      = self.get_parameter('face_yaw_lim').value
        self.face_pitch_up     = self.get_parameter('face_pitch_up').value
        self.face_pitch_dn     = self.get_parameter('face_pitch_dn').value
        self.saliency_yaw_lim  = self.get_parameter('saliency_yaw_lim').value
        self.saliency_pitch_up = self.get_parameter('saliency_pitch_up').value
        self.saliency_pitch_dn = self.get_parameter('saliency_pitch_dn').value
        self.face_timeout      = self.get_parameter('face_timeout').value
        self.engaged_bonus     = self.get_parameter('engaged_priority_bonus').value
        self.face_switch_cooldown = self.get_parameter('face_switch_cooldown').value
        self.same_face_threshold  = math.radians(self.get_parameter('same_face_threshold_deg').value)
        self.prefer_closer     = self.get_parameter('prefer_closer_faces').value
        self.max_face_distance = self.get_parameter('max_face_distance').value
        self.min_angular_change = math.radians(self.get_parameter('min_angular_change_deg').value)
        self.target_smoothing_alpha = self.get_parameter('target_smoothing_alpha').value
        self.saliency_min      = self.get_parameter('saliency_min_score').value
        self.min_cooldown      = self.get_parameter('saliency_min_cooldown').value
        self.max_dwell         = self.get_parameter('saliency_max_dwell').value
        self.switch_ratio      = self.get_parameter('switch_score_ratio').value
        self.same_target_threshold = math.radians(self.get_parameter('same_target_threshold_deg').value)
        self.enable_ior        = self.get_parameter('enable_ior').value
        self.ior_max_suppression = self.get_parameter('ior_max_suppression').value
        self.ior_half_life     = self.get_parameter('ior_half_life').value
        self.ior_radius        = math.radians(self.get_parameter('ior_radius_deg').value)
        self.ior_cleanup_threshold = self.get_parameter('ior_cleanup_threshold').value
        self.ior_max_locations = self.get_parameter('ior_max_locations').value
        self.attention_enabled = self.get_parameter('start_enabled').value

        # Internal state
        self.fx = self.fy = self.cx = self.cy = None
        self._head_yaw   = self._head_pitch   = None
        self._target_yaw = self._target_pitch = None
        self.last_face_time        = 0.0
        self.last_face_switch_time = 0.0
        self.current_face_id       = None
        self.current_face_location = None
        self.last_saliency_cmd_time  = 0.0
        self.current_saliency_target = None
        self.current_saliency_score  = 0.0
        self.visited_locations: list = []

        # Managed publishers
        self.pub_head   = self.create_lifecycle_publisher(JointAnglesWithSpeed, self.head_cmd_topic, 10)
        self.pub_target = self.create_lifecycle_publisher(Vector3, self.target_topic, 10)

        # Enable/disable service — available as long as node is configured
        self.srv_enable = self.create_service(SetBool, '/overt_attention/set_enabled', self.handle_set_enabled)

        status       = 'ENABLED' if self.attention_enabled else 'DISABLED'
        default_mode = 'move to default' if self.move_to_default_on_disable else 'hold position'
        self.get_logger().info(f'Attention controller configured ({status})')
        self.get_logger().info(f'Service: /overt_attention/set_enabled | disable mode: {default_mode}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, _state) -> TransitionCallbackReturn:
        """Activate the lifecycle publishers, then subscribe to face, saliency, camera_info, and joint_state topics."""
        super().on_activate(_state)

        qos_img = get_image_qos()
        self._sub_faces    = self.create_subscription(FaceDetection,     self.face_topic,        self.on_faces,       10)
        self._sub_saliency = self.create_subscription(Float32MultiArray,  self.saliency_topic,    self.on_saliency,    10)
        self._sub_caminfo  = self.create_subscription(CameraInfo,         self.camera_info_topic, self.on_caminfo,     qos_img)
        self._sub_joints   = self.create_subscription(JointState,         '/joint_states',        self.on_joint_states, 10)

        self.get_logger().info('Attention controller activated')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, _state) -> TransitionCallbackReturn:
        """Destroy sensor subscriptions, clear tracking state, and deactivate the lifecycle publishers."""
        for sub in (self._sub_faces, self._sub_saliency, self._sub_caminfo, self._sub_joints):
            self.destroy_subscription(sub)

        # Clear tracking state so stale data isn't used on re-activation
        self.current_face_id        = None
        self.current_face_location  = None
        self.current_saliency_target = None
        self.visited_locations       = []

        super().on_deactivate(_state)
        self.get_logger().info('Attention controller deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, _state) -> TransitionCallbackReturn:
        """Destroy the head/target lifecycle publishers and the set_enabled service."""
        self.destroy_lifecycle_publisher(self.pub_head)
        self.destroy_lifecycle_publisher(self.pub_target)
        self.destroy_service(self.srv_enable)
        self.get_logger().info('Attention controller cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state) -> TransitionCallbackReturn:
        """Log shutdown of the attention controller."""
        self.get_logger().info('Attention controller shutting down')
        return TransitionCallbackReturn.SUCCESS

    # ── Service ─────────────────────────────────────────────────────────────────

    def handle_set_enabled(self, request, response):
        old_state              = self.attention_enabled
        self.attention_enabled = request.data

        if old_state != self.attention_enabled:
            if self.attention_enabled:
                self.get_logger().info('Attention system ENABLED')
                response.success = True
                response.message = 'Attention system enabled'
            else:
                self.current_face_id        = None
                self.current_face_location  = None
                self.current_saliency_target = None
                self.visited_locations       = []

                if self.move_to_default_on_disable:
                    self.move_head_to_default()
                    self.get_logger().info(
                        f'Attention system DISABLED — moving to default '
                        f'(yaw={math.degrees(self.default_yaw):.1f}°, '
                        f'pitch={math.degrees(self.default_pitch):.1f}°)'
                    )
                    response.success = True
                    response.message = (
                        f'Attention disabled — moving to default '
                        f'(yaw={math.degrees(self.default_yaw):.1f}°, '
                        f'pitch={math.degrees(self.default_pitch):.1f}°)'
                    )
                else:
                    cy = self._head_yaw   if self._head_yaw   is not None else 0.0
                    cp = self._head_pitch if self._head_pitch is not None else 0.0
                    self.get_logger().info(
                        f'Attention system DISABLED — holding position '
                        f'(yaw={math.degrees(cy):.1f}°, pitch={math.degrees(cp):.1f}°)'
                    )
                    response.success = True
                    response.message = (
                        f'Attention disabled — holding at '
                        f'yaw={math.degrees(cy):.1f}°, pitch={math.degrees(cp):.1f}°'
                    )
        else:
            status           = 'enabled' if self.attention_enabled else 'disabled'
            response.success = True
            response.message = f'Attention system already {status}'

        return response

    # ── Sensor callbacks ─────────────────────────────────────────────────────────

    def on_caminfo(self, msg: CameraInfo):
        if self.fx is None:
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(f'Camera: fx={self.fx:.1f}, fy={self.fy:.1f}')

    def on_joint_states(self, msg: JointState):
        try:
            self._head_yaw   = msg.position[msg.name.index('HeadYaw')]
            self._head_pitch = msg.position[msg.name.index('HeadPitch')]
        except (ValueError, IndexError):
            pass

    def move_head_to_default(self):
        msg = JointAnglesWithSpeed()
        msg.header.stamp  = self.get_clock().now().to_msg()
        msg.joint_names   = ['HeadYaw', 'HeadPitch']
        msg.joint_angles  = [float(self.default_yaw), float(self.default_pitch)]
        msg.speed         = float(self.default_move_speed)
        msg.relative      = False
        self.pub_head.publish(msg)

        target_msg   = Vector3()
        target_msg.x = 0.0
        target_msg.y = 0.0
        target_msg.z = 0.0
        self.pub_target.publish(target_msg)

    # ── Face attention ───────────────────────────────────────────────────────────

    def calculate_face_priority(self, centroid, mutual_gaze, face_id, is_current_face):
        score = 1.0
        if mutual_gaze:
            score *= self.engaged_bonus
        dist_from_center = math.sqrt((centroid.x - self.cx)**2 + (centroid.y - self.cy)**2)
        max_dist         = math.sqrt(self.cx**2 + self.cy**2)
        center_score     = 1.0 - (dist_from_center / max_dist)
        score           *= (0.5 + 0.5 * center_score)
        if self.prefer_closer and centroid.z > 0:
            if centroid.z <= self.max_face_distance:
                depth_bonus = 1.5 - (centroid.z / self.max_face_distance)
                score      *= max(0.5, depth_bonus)
            else:
                score *= 0.3
        if is_current_face:
            time_since_switch = time.time() - self.last_face_switch_time
            score *= 1.5 if time_since_switch < self.face_switch_cooldown else 1.1
        return score

    def on_faces(self, msg: FaceDetection):
        if not self.attention_enabled:
            return
        if self.fx is None or self._head_yaw is None or self._head_pitch is None:
            return
        if not msg.centroids:
            if self.current_face_id is not None:
                self.get_logger().info(f'Lost face: {self.current_face_id}')
                self.current_face_id       = None
                self.current_face_location = None
            return

        current_time = time.time()
        candidates   = []
        for i, centroid in enumerate(msg.centroids):
            face_id     = msg.face_label_id[i] if i < len(msg.face_label_id) else f'unknown_{i}'
            mutual_gaze = msg.mutual_gaze[i]   if i < len(msg.mutual_gaze)   else False
            is_current  = (face_id == self.current_face_id)
            priority    = self.calculate_face_priority(centroid, mutual_gaze, face_id, is_current)
            cam_yaw, cam_pitch = pixel_to_angles(
                centroid.x, centroid.y, self.fx, self.fy, self.cx, self.cy
            )
            candidates.append({
                'face_id': face_id,
                'centroid': centroid,
                'world_yaw':   cam_yaw   + self._head_yaw,
                'world_pitch': cam_pitch + self._head_pitch,
                'mutual_gaze': mutual_gaze,
                'priority': priority,
                'is_current': is_current,
            })

        best_face    = max(candidates, key=lambda f: f['priority'])
        should_switch = False
        switch_reason = ''

        if self.current_face_id is None:
            should_switch = True
            switch_reason = 'initial'
        elif best_face['is_current']:
            should_switch = True
            switch_reason = 'refresh'
        else:
            time_since_switch = current_time - self.last_face_switch_time
            current_face      = next((f for f in candidates if f['is_current']), None)
            if current_face is None:
                should_switch = True
                switch_reason = 'lost_current'
            elif time_since_switch < self.face_switch_cooldown:
                if best_face['priority'] > current_face['priority'] * 1.5:
                    should_switch = True
                    switch_reason = 'much_better'
            else:
                if best_face['priority'] > current_face['priority'] * 1.1:
                    should_switch = True
                    switch_reason = 'better'

        if not should_switch:
            self.last_face_time = current_time
            return

        is_new_face = (best_face['face_id'] != self.current_face_id)
        if is_new_face:
            self.last_face_switch_time = current_time
            self._target_yaw           = None
            self._target_pitch         = None
            self.get_logger().info(
                f"Switching to face: {best_face['face_id']} "
                f"(engaged={best_face['mutual_gaze']}, "
                f"depth={best_face['centroid'].z:.2f}m, "
                f"priority={best_face['priority']:.2f}, reason={switch_reason})"
            )

        self.current_face_id       = best_face['face_id']
        self.current_face_location = (best_face['world_yaw'], best_face['world_pitch'])
        self.last_face_time        = current_time

        yaw   = clamp(best_face['world_yaw'],   -self.face_yaw_lim, self.face_yaw_lim)
        pitch = clamp(best_face['world_pitch'],  self.face_pitch_dn, self.face_pitch_up)
        source = f"face[{best_face['face_id']}]"
        if best_face['mutual_gaze']:
            source += '_engaged'
        if switch_reason:
            source += f'({switch_reason})'
        self.publish_head(yaw, pitch, score=best_face['priority'], source=source)

    # ── Saliency attention ───────────────────────────────────────────────────────

    def calculate_ior_suppression(self, age_seconds):
        decay = math.log(2) / self.ior_half_life
        return self.ior_max_suppression * math.exp(-decay * age_seconds)

    def apply_ior_filter(self, world_yaw, world_pitch, score):
        if not self.enable_ior or not self.visited_locations:
            return score
        current_time    = time.time()
        max_suppression = 0.0
        for v_yaw, v_pitch, timestamp in self.visited_locations:
            age  = current_time - timestamp
            dist = math.sqrt((world_yaw - v_yaw)**2 + (world_pitch - v_pitch)**2)
            if dist < self.ior_radius:
                time_supp       = self.calculate_ior_suppression(age)
                space_decay     = 1.0 - (dist / self.ior_radius)
                max_suppression = max(max_suppression, time_supp * space_decay)
        return score * (1.0 - max_suppression)

    def cleanup_weak_ior(self):
        if not self.enable_ior:
            return
        current_time = time.time()
        self.visited_locations = [
            (yaw, pitch, ts)
            for yaw, pitch, ts in self.visited_locations
            if self.calculate_ior_suppression(current_time - ts) >= self.ior_cleanup_threshold
        ][:self.ior_max_locations]

    def on_saliency(self, msg: Float32MultiArray):
        if not self.attention_enabled:
            return
        if self.fx is None or self._head_yaw is None:
            return
        if time.time() - self.last_face_time < self.face_timeout:
            return

        if self.current_face_id is not None:
            self.get_logger().info(
                f'No recent faces, switching to saliency (was tracking: {self.current_face_id})'
            )
            self.current_face_id        = None
            self.current_face_location  = None
            self._target_yaw            = None
            self._target_pitch          = None

            head_yaw   = self._head_yaw
            head_pitch = self._head_pitch if self._head_pitch is not None else 0.0
            clamped_yaw   = clamp(head_yaw,   -self.saliency_yaw_lim, self.saliency_yaw_lim)
            clamped_pitch = clamp(head_pitch,  self.saliency_pitch_dn, self.saliency_pitch_up)
            if clamped_yaw != head_yaw or clamped_pitch != head_pitch:
                self.publish_head(clamped_yaw, clamped_pitch, source='saliency_reenter', force=True)

        if len(msg.data) < 3:
            return

        self.cleanup_weak_ior()
        candidates = []
        for i in range(0, len(msg.data) - 2, 3):
            u, v, score = msg.data[i], msg.data[i + 1], msg.data[i + 2]
            if score < self.saliency_min:
                continue
            cam_yaw, cam_pitch = pixel_to_angles(u, v, self.fx, self.fy, self.cx, self.cy)
            world_yaw   = cam_yaw   + self._head_yaw
            world_pitch = cam_pitch + self._head_pitch
            if (world_yaw < -self.saliency_yaw_lim or world_yaw > self.saliency_yaw_lim or
                    world_pitch < self.saliency_pitch_dn or world_pitch > self.saliency_pitch_up):
                continue
            score = self.apply_ior_filter(world_yaw, world_pitch, score)
            if score >= self.saliency_min:
                candidates.append((world_yaw, world_pitch, score))

        if not candidates:
            return

        best_yaw, best_pitch, best_score = max(candidates, key=lambda x: x[2])

        is_same = False
        if self.current_saliency_target:
            curr_yaw, curr_pitch = self.current_saliency_target
            dist    = math.sqrt((best_yaw - curr_yaw)**2 + (best_pitch - curr_pitch)**2)
            is_same = dist < self.same_target_threshold

        time_on_target = time.time() - self.last_saliency_cmd_time
        if not self.current_saliency_target:
            should_switch = True
            reason = 'initial'
        elif is_same:
            should_switch = True
            reason = 'refresh'
        elif time_on_target < self.min_cooldown:
            should_switch = best_score > self.current_saliency_score * self.switch_ratio
            reason = 'early' if should_switch else None
        elif time_on_target > self.max_dwell:
            should_switch = not is_same
            reason = 'max_dwell' if should_switch else None
        else:
            should_switch = best_score > self.current_saliency_score * 1.15
            reason = 'better' if should_switch else None

        if not should_switch:
            return

        self.current_saliency_target = (best_yaw, best_pitch)
        self.current_saliency_score  = best_score
        self.last_saliency_cmd_time  = time.time()
        if self.enable_ior:
            self.visited_locations.append((best_yaw, best_pitch, time.time()))

        yaw   = clamp(best_yaw,   -self.saliency_yaw_lim, self.saliency_yaw_lim)
        pitch = clamp(best_pitch,  self.saliency_pitch_dn, self.saliency_pitch_up)
        self.publish_head(yaw, pitch, score=best_score, source=f'saliency({reason})')

    # ── Head command publisher ───────────────────────────────────────────────────

    def publish_head(self, yaw, pitch, score=0.0, source='unknown', force=False):
        if force:
            self._target_yaw   = yaw
            self._target_pitch = pitch
        else:
            if self._target_yaw is None:
                self._target_yaw   = yaw
                self._target_pitch = pitch
            else:
                alpha              = self.target_smoothing_alpha
                self._target_yaw   = alpha * yaw   + (1.0 - alpha) * self._target_yaw
                self._target_pitch = alpha * pitch + (1.0 - alpha) * self._target_pitch

            yaw   = self._target_yaw
            pitch = self._target_pitch

            if self._head_yaw is not None and self._head_pitch is not None:
                if (abs(yaw - self._head_yaw)   < self.min_angular_change and
                        abs(pitch - self._head_pitch) < self.min_angular_change):
                    return

        msg = JointAnglesWithSpeed()
        msg.header.stamp  = self.get_clock().now().to_msg()
        msg.joint_names   = ['HeadYaw', 'HeadPitch']
        msg.joint_angles  = [float(yaw), float(pitch)]
        msg.speed         = 0.1
        msg.relative      = False
        self.pub_head.publish(msg)

        if self._head_yaw is not None and self._head_pitch is not None:
            cam_relative_yaw   = yaw   - self._head_yaw
            cam_relative_pitch = pitch - self._head_pitch
        else:
            cam_relative_yaw   = yaw
            cam_relative_pitch = pitch

        target_msg   = Vector3()
        target_msg.x = float(cam_relative_yaw)
        target_msg.y = float(cam_relative_pitch)
        target_msg.z = float(score)
        self.pub_target.publish(target_msg)

        self.get_logger().info(
            f'[{source}] → yaw={math.degrees(yaw):.1f}°, '
            f'pitch={math.degrees(pitch):.1f}°, score={score:.2f}'
        )


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


if __name__ == '__main__':
    main()
