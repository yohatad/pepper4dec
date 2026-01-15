#!/usr/bin/env python3
"""
Unified Attention Controller for Pepper Robot
Priority: Faces → Saliency → Audio → Idle/Home
With unified IOR and scene change detection
"""

import math
import time
import random
import collections
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import CameraInfo, JointState, Image, CompressedImage
from threading import Lock

import cv2
from cv_bridge import CvBridge

from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from cssr_interfaces.msg import FaceDetection


# ============ Helpers ============

def get_image_topic(base_topic: str, use_compressed: bool, is_depth: bool = False) -> str:
    """Construct full topic name based on compression setting."""
    if use_compressed:
        suffix = "/compressedDepth" if is_depth else "/compressed"
        return base_topic + suffix
    return base_topic


def get_image_qos() -> QoSProfile:
    """QoS for image transport over WiFi."""
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1
    )


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def depth_bonus(Z, depth_min, depth_max):
    """Linear bonus favoring closer targets within valid range.
    
    Returns 1.0 at depth_min (close), 0.0 at depth_max (far).
    Returns 0.0 if Z is None.
    """
    if Z is None:
        return 0.0
    return (depth_max - Z) / (depth_max - depth_min)


def pixel_to_angles(u, v, fx, fy, cx, cy):
    """Convert pixel to angles (pinhole model)."""
    return math.atan2((u - cx) / fx, 1.0), math.atan2((v - cy) / fy, 1.0)


def uvZ_to_angles(u, v, Z, fx, fy, cx, cy):
    """Convert pixel + depth to angles."""
    Xc = (u - cx) / fx * Z
    Yc = (v - cy) / fy * Z
    return math.atan2(Xc, Z), math.atan2(Yc, Z)


# ============ Track Class ============
class Track:
    """Represents a face track."""
    
    def __init__(self, tid, u, v, w, h, mutual, smoothing=0.4):
        self.id = tid
        self.u, self.v = float(u), float(v)
        self.w, self.h = float(w), float(h)
        self.mutual = bool(mutual)
        self.last_seen = time.time()
        self.Z = None
        self.score = 0.0
        self._alpha = smoothing  # EMA smoothing factor (0 = no smoothing, 1 = no history)

    def update(self, u, v, w, h, mutual):
        """Update track with new detection, applying EMA smoothing."""
        self.u = self._alpha * u + (1 - self._alpha) * self.u
        self.v = self._alpha * v + (1 - self._alpha) * self.v
        self.w, self.h = float(w), float(h)
        self.mutual = bool(mutual)
        self.last_seen = time.time()

# ============ Main Node ============

class UnifiedAttention(Node):
    def __init__(self):
        super().__init__("pepper_unified_attention")
        
        self.cv_bridge = CvBridge()
        
        self._declare_parameters()
        self._load_parameters()
        
        qos_img = get_image_qos()
        
        # Subscriptions
        self.create_subscription(FaceDetection, self.face_topic, self._on_faces, 10)
        self.create_subscription(Float32MultiArray, self.saliency_topic, self._on_saliency, 10)
        self.create_subscription(Float32, self.audio_topic, self._on_audio, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_caminfo, qos_img)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 10)
        
        # Depth subscription
        depth_topic = get_image_topic(self.depth_image_topic, self.use_compressed, is_depth=True)
        if self.use_compressed:
            self.create_subscription(CompressedImage, depth_topic, self._on_depth_compressed, qos_img)
        else:
            self.create_subscription(Image, depth_topic, self._on_depth_raw, qos_img)
        self.get_logger().info(f"Subscribing to depth: {depth_topic}")
        
        # Publishers
        self.pub_js = self.create_publisher(JointAnglesWithSpeed, self.js_topic, 10)
        self.pub_dbg = self.create_publisher(Vector3, "/attn/target_angles", 10)
        
        # Thread-safe state
        self._depth_lock = Lock()
        self._depth_image = None
        self._depth_stamp = None
        
        self._tracks_lock = Lock()
        self._tracks = {}
        
        self._saliency_lock = Lock()
        self._saliency_peak = None  # (u, v, score) or None
        
        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        
        # Attention state
        self._current_face_id = None
        self._face_since = time.time()
        self._current_source = None  # 'face', 'saliency', 'audio', or None
        
        # IOR queue: (yaw, pitch, timestamp, source) - stored in robot base frame
        self._ior = collections.deque(maxlen=20)
        
        # Saliency tracking for IOR
        self._saliency_target = None  # (yaw, pitch, start_time) or None
        
        # Audio state
        self._audio_azimuth = None
        self._audio_until = 0.0
        
        # Head state
        self._head_yaw = 0.0
        self._head_pitch = 0.0
        self._head_velocity = 0.0
        self._head_settled_time = time.time()
        
        # Base odometry for IOR reset
        self._base_x = 0.0
        self._base_y = 0.0
        self._base_x_at_ior_reset = 0.0
        self._base_y_at_ior_reset = 0.0
        
        # Idle state
        self._idle_target = (self.home_yaw, self.home_pitch)
        self._next_idle_scan = 0.0
        self._next_micro_saccade = 0.0
        
        # Timers
        self.create_timer(1.0 / 20.0, self._tick)
        self.create_timer(2.0, self._cleanup_tracks)
        
        self.get_logger().info(
            f"Unified attention ready (IOR in base frame, "
            f"radius={math.degrees(self.ior_radius):.1f}°, "
            f"timeout={self.ior_timeout}s)"
        )

    def _declare_parameters(self):
        """Declare all ROS parameters."""
        # Common
        self.declare_parameter("use_compressed", True)
        self.declare_parameter("depth_image_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("face_topic", "/faceDetection/data")
        self.declare_parameter("saliency_topic", "/attn/saliency_peak")
        self.declare_parameter("audio_topic", "/audio/azimuth_rad")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("odom_topic", "/odom")  # For base movement detection
        
        # Command output
        self.declare_parameter("joint_names", ["HeadYaw", "HeadPitch"])
        self.declare_parameter("joint_state_topic", "/joint_angles")
        
        # Scoring
        self.declare_parameter("bias_face", 2.0)
        self.declare_parameter("mutual_gaze_bonus", 0.6)
        self.declare_parameter("saliency_min_score", 0.30)
        
        # Depth
        self.declare_parameter("depth_min", 0.3)   # Closer than this = sensor noise
        self.declare_parameter("depth_max", 3.0)   # Farther than this = not interacting
        self.declare_parameter("depth_roi_px", 5)
        self.declare_parameter("track_timeout_s", 5.0)
        
        # Hysteresis
        self.declare_parameter("hysteresis_delta", 1.2)
        self.declare_parameter("hysteresis_hold_s", 1.2)
        self.declare_parameter("dwell_cap_s", 8.0)
        
        # IOR (stored in robot base frame as yaw, pitch angles)
        self.declare_parameter("ior_radius_rad", 0.15)  # ~8.5 degrees
        self.declare_parameter("ior_timeout_s", 30.0)   # IOR entries expire after this
        self.declare_parameter("ior_penalty", 2.0)
        self.declare_parameter("saliency_dwell_for_ior", 2.5)
        self.declare_parameter("base_move_threshold", 0.1)  # meters - reset IOR if base moves
        
        # Audio
        self.declare_parameter("audio_hold_s", 2.0)
        
        # Motion gating
        self.declare_parameter("head_velocity_threshold", 0.08)
        self.declare_parameter("head_settle_time", 0.5)  # Reduced from 5.0
        self.declare_parameter("deadband_yaw", 0.06)
        self.declare_parameter("deadband_pitch", 0.06)
        
        # Idle
        self.declare_parameter("home_yaw", 0.0)
        self.declare_parameter("home_pitch", -0.05)
        self.declare_parameter("idle_scan_period_min", 3.0)
        self.declare_parameter("idle_scan_period_max", 6.0)
        self.declare_parameter("idle_scan_yaw_deg", 20.0)
        self.declare_parameter("idle_scan_pitch_deg", 8.0)
        self.declare_parameter("micro_saccade_deg", 0.4)
        self.declare_parameter("micro_period_min", 3.0)
        self.declare_parameter("micro_period_max", 5.0)
        
        # Limits
        self.declare_parameter("yaw_lim", 1.8)
        self.declare_parameter("pitch_up", 0.4)
        self.declare_parameter("pitch_dn", -0.7)

    def _load_parameters(self):
        """Load parameters into instance variables."""
        self.use_compressed = self.get_parameter("use_compressed").value
        self.depth_image_topic = self.get_parameter("depth_image_topic").value
        self.depth_scale = self.get_parameter("depth_scale").value
        self.face_topic = self.get_parameter("face_topic").value
        self.saliency_topic = self.get_parameter("saliency_topic").value
        self.audio_topic = self.get_parameter("audio_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.js_topic = self.get_parameter("joint_state_topic").value
        
        self.bias_face = self.get_parameter("bias_face").value
        self.mutual_bonus = self.get_parameter("mutual_gaze_bonus").value
        self.saliency_min = self.get_parameter("saliency_min_score").value
        
        self.depth_min = self.get_parameter("depth_min").value
        self.depth_max = self.get_parameter("depth_max").value
        self.depth_roi = self.get_parameter("depth_roi_px").value
        self.track_timeout = self.get_parameter("track_timeout_s").value
        
        self.hyst_delta = self.get_parameter("hysteresis_delta").value
        self.hyst_hold = self.get_parameter("hysteresis_hold_s").value
        self.dwell_cap = self.get_parameter("dwell_cap_s").value
        
        self.ior_radius = self.get_parameter("ior_radius_rad").value
        self.ior_timeout = self.get_parameter("ior_timeout_s").value
        self.ior_penalty = self.get_parameter("ior_penalty").value
        self.saliency_dwell_for_ior = self.get_parameter("saliency_dwell_for_ior").value
        self.base_move_threshold = self.get_parameter("base_move_threshold").value
        
        self.audio_hold = self.get_parameter("audio_hold_s").value
        
        self.vel_threshold = self.get_parameter("head_velocity_threshold").value
        self.settle_time = self.get_parameter("head_settle_time").value
        self.deadband_yaw = self.get_parameter("deadband_yaw").value
        self.deadband_pitch = self.get_parameter("deadband_pitch").value
        
        self.home_yaw = self.get_parameter("home_yaw").value
        self.home_pitch = self.get_parameter("home_pitch").value
        self.idle_min = self.get_parameter("idle_scan_period_min").value
        self.idle_max = self.get_parameter("idle_scan_period_max").value
        self.idle_yaw_deg = self.get_parameter("idle_scan_yaw_deg").value
        self.idle_pitch_deg = self.get_parameter("idle_scan_pitch_deg").value
        self.micro_deg = self.get_parameter("micro_saccade_deg").value
        self.micro_min = self.get_parameter("micro_period_min").value
        self.micro_max = self.get_parameter("micro_period_max").value
        
        self.yaw_lim = self.get_parameter("yaw_lim").value
        self.pitch_up = self.get_parameter("pitch_up").value
        self.pitch_dn = self.get_parameter("pitch_dn").value

    # ============ Depth Handling ============
    
    def _on_depth_raw(self, msg: Image):
        """Handle raw depth image."""
        try:
            depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self._depth_lock:
                self._depth_image = depth
                self._depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    def _on_depth_compressed(self, msg: CompressedImage):
        """Handle compressed depth image."""
        try:
            if len(msg.data) <= 12:
                return
            depth = cv2.imdecode(np.frombuffer(msg.data[12:], np.uint8), cv2.IMREAD_UNCHANGED)
            if depth is not None:
                with self._depth_lock:
                    self._depth_image = depth
                    self._depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f"Compressed depth failed: {e}")

    def _get_depth_at(self, u, v) -> float | None:
        """Get depth in meters at pixel, using ROI median."""
        with self._depth_lock:
            if self._depth_image is None:
                return None
            depth = self._depth_image.copy()
        
        h, w = depth.shape[:2]
        ui, vi = int(round(u)), int(round(v))
        
        if not (0 <= ui < w and 0 <= vi < h):
            return None
        
        # Extract ROI
        half = self.depth_roi // 2
        roi = depth[max(0, vi - half):min(h, vi + half + 1),
                    max(0, ui - half):min(w, ui + half + 1)]
        
        # Filter invalid
        if roi.dtype in (np.float32, np.float64):
            valid = roi[np.isfinite(roi) & (roi > 0)]
        else:
            valid = roi[roi > 0]
        
        if len(valid) == 0:
            return None
        
        Z = float(np.median(valid)) * self.depth_scale
        
        if not (self.depth_min <= Z <= self.depth_max):
            return None
        
        return Z

    # ============ IOR Management ============
    
    def _pixel_to_base_angles(self, u, v, Z=None) -> tuple[float, float]:
        """Convert pixel coordinates to angles in robot base frame.
        
        Returns (yaw, pitch) relative to robot base (head angles + camera angles).
        """
        if Z and self.depth_min <= Z <= self.depth_max:
            cam_yaw, cam_pitch = uvZ_to_angles(u, v, Z, self.fx, self.fy, self.cx, self.cy)
        else:
            cam_yaw, cam_pitch = pixel_to_angles(u, v, self.fx, self.fy, self.cx, self.cy)
        
        # Convert from camera frame to robot base frame
        # Camera angles are relative to head, so add current head position
        base_yaw = self._head_yaw + cam_yaw
        base_pitch = self._head_pitch + cam_pitch
        
        return base_yaw, base_pitch
    
    def _add_ior(self, yaw, pitch, source):
        """Add location to IOR (in robot base frame angles)."""
        self._ior.append((yaw, pitch, time.time(), source))
        self.get_logger().info(
            f"IOR added: {source} at ({math.degrees(yaw):.1f}°, {math.degrees(pitch):.1f}°), "
            f"total={len(self._ior)}"
        )
    
    def _add_ior_from_pixel(self, u, v, source, Z=None):
        """Convert pixel to base frame angles and add to IOR."""
        yaw, pitch = self._pixel_to_base_angles(u, v, Z)
        self._add_ior(yaw, pitch, source)

    def _is_suppressed_angles(self, yaw, pitch) -> tuple[bool, str | None]:
        """Check if base-frame angles are suppressed by IOR."""
        now = time.time()
        for (ior_yaw, ior_pitch, t, source) in self._ior:
            # Skip expired entries
            if now - t > self.ior_timeout:
                continue
            # Check angular distance
            dist = np.hypot(yaw - ior_yaw, pitch - ior_pitch)
            if dist < self.ior_radius:
                return True, source
        return False, None
    
    def _is_suppressed_pixel(self, u, v, Z=None) -> tuple[bool, str | None]:
        """Check if pixel location is suppressed by IOR."""
        yaw, pitch = self._pixel_to_base_angles(u, v, Z)
        return self._is_suppressed_angles(yaw, pitch)

    def _reset_ior(self, reason: str):
        """Clear IOR (scene/source changed)."""
        if self._ior:
            self.get_logger().info(f"IOR reset ({len(self._ior)} cleared): {reason}")
            self._ior.clear()
        self._base_x_at_ior_reset = self._base_x
        self._base_y_at_ior_reset = self._base_y
    
    def _set_attention_source(self, new_source: str):
        """Track attention source and reset IOR on source switch."""
        if self._current_source and self._current_source != new_source:
            self._reset_ior(f"Source switch: {self._current_source} → {new_source}")
        self._current_source = new_source

    # ============ Track Management ============
    
    def _cleanup_tracks(self):
        """Remove stale tracks."""
        now = time.time()
        with self._tracks_lock:
            stale = [tid for tid, t in self._tracks.items() if now - t.last_seen > self.track_timeout]
            for tid in stale:
                del self._tracks[tid]

    # ============ Callbacks ============
    
    def _on_caminfo(self, msg: CameraInfo):
        """Store camera intrinsics."""
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def _on_joint_states(self, msg: JointState):
        """Track head state for coordinate transforms."""
        try:
            yaw_idx = msg.name.index('HeadYaw')
            pitch_idx = msg.name.index('HeadPitch')
            
            self._head_yaw = msg.position[yaw_idx]
            self._head_pitch = msg.position[pitch_idx]
            
            # Get velocity (handle NaN)
            yaw_vel = msg.velocity[yaw_idx]
            pitch_vel = msg.velocity[pitch_idx]
            yaw_vel = 0.0 if math.isnan(yaw_vel) else yaw_vel
            pitch_vel = 0.0 if math.isnan(pitch_vel) else pitch_vel
            
            self._head_velocity = np.hypot(yaw_vel, pitch_vel)
            
            if self._head_velocity > self.vel_threshold:
                self._head_settled_time = time.time()
                
        except (ValueError, IndexError):
            pass
    
    def _on_odom(self, msg: Odometry):
        """Track base position and reset IOR if robot moves."""
        self._base_x = msg.pose.pose.position.x
        self._base_y = msg.pose.pose.position.y
        
        # Check if base moved significantly
        moved = np.hypot(
            self._base_x - self._base_x_at_ior_reset,
            self._base_y - self._base_y_at_ior_reset
        )
        if moved > self.base_move_threshold:
            self._reset_ior(f"Base moved {moved:.2f}m")

    def _on_audio(self, msg: Float32):
        """Store audio azimuth."""
        self._audio_azimuth = float(msg.data)
        self._audio_until = time.time() + self.audio_hold

    def _on_saliency(self, msg: Float32MultiArray):
        """Process saliency peaks (3 values per peak: u, v, score)."""
        if len(msg.data) < 3:
            with self._saliency_lock:
                self._saliency_peak = None
            return
        
        best = None
        best_score = 0.0
        
        # Parse peaks (stride of 3)
        for i in range(0, len(msg.data) - 2, 3):
            u, v, score = msg.data[i], msg.data[i + 1], msg.data[i + 2]
            
            # Check IOR suppression (convert pixel to base frame)
            suppressed, _ = self._is_suppressed_pixel(u, v)
            if suppressed:
                continue
            
            if score > best_score and score >= self.saliency_min:
                best = (u, v, score)
                best_score = score
        
        with self._saliency_lock:
            self._saliency_peak = best

    def _on_faces(self, msg: FaceDetection):
        """Process face detections (Priority 1)."""
        if self.fx is None:
            self.get_logger().warn_once("Waiting for camera info...")
            return
        
        now = time.time()
        n = len(msg.centroids)
        
        if n == 0:
            return
        
        # Update tracks
        with self._tracks_lock:
            for i in range(n):
                fid = msg.face_label_id[i] if i < len(msg.face_label_id) else str(i)
                c = msg.centroids[i]
                w = msg.width[i] if i < len(msg.width) else 80.0
                h = msg.height[i] if i < len(msg.height) else 80.0
                mg = msg.mutual_gaze[i] if i < len(msg.mutual_gaze) else False
                
                if fid not in self._tracks:
                    self._tracks[fid] = Track(fid, c.x, c.y, w, h, mg)
                else:
                    self._tracks[fid].update(c.x, c.y, w, h, mg)
            
            # Get candidates and copy their data for scoring outside lock
            candidates = []
            for t in self._tracks.values():
                if now - t.last_seen < 0.3:
                    candidates.append(t)
        
        if not candidates:
            return
        
        # Score candidates (outside lock to avoid blocking)
        for t in candidates:
            t.Z = self._get_depth_at(t.u, t.v)
            
            s = self.bias_face
            if t.mutual:
                s += self.mutual_bonus
            if t.Z:
                s += depth_bonus(t.Z, self.depth_min, self.depth_max)
            
            # Check IOR suppression (in base frame)
            suppressed, _ = self._is_suppressed_pixel(t.u, t.v, t.Z)
            if suppressed:
                s -= self.ior_penalty
            
            t.score = s
        
        # Select target with hysteresis
        target = self._select_face(candidates, now)
        if not target:
            return
        
        # Set attention source (will reset IOR if switching from saliency)
        self._set_attention_source('face')
        
        # Clear saliency tracking
        self._saliency_target = None
        
        # Compute angles using smoothed track position
        if target.Z:
            yaw, pitch = uvZ_to_angles(target.u, target.v, target.Z, self.fx, self.fy, self.cx, self.cy)
        else:
            yaw, pitch = pixel_to_angles(target.u, target.v, self.fx, self.fy, self.cx, self.cy)
        
        self._publish_head(yaw, pitch)

    def _select_face(self, candidates, now) -> Track | None:
        """Select best face with hysteresis and dwell cap."""
        best = max(candidates, key=lambda t: t.score)
        
        # Hysteresis: prefer current target
        if self._current_face_id and self._current_face_id != best.id:
            with self._tracks_lock:
                prev = self._tracks.get(self._current_face_id)
            if prev and prev in candidates:
                if best.score < prev.score + self.hyst_delta:
                    if now - self._face_since < self.hyst_hold:
                        return prev
        
        # Dwell cap: add to IOR if staring too long
        if self._current_face_id:
            if now - self._face_since > self.dwell_cap:
                with self._tracks_lock:
                    prev = self._tracks.get(self._current_face_id)
                if prev:
                    self._add_ior_from_pixel(prev.u, prev.v, "face_dwell", prev.Z)
                self._current_face_id = None
        
        # Switch target
        if self._current_face_id != best.id:
            # Add old target to IOR
            if self._current_face_id:
                with self._tracks_lock:
                    prev = self._tracks.get(self._current_face_id)
                if prev:
                    self._add_ior_from_pixel(prev.u, prev.v, "face_switch", prev.Z)
            
            self._current_face_id = best.id
            self._face_since = now
        
        return best

    def _tick(self):
        """Main attention loop for saliency/audio/idle."""
        if self.fx is None:
            return
        
        now = time.time()
        
        # Wait for head to settle
        if now - self._head_settled_time < self.settle_time:
            return
        
        # If recent face detection handled it, skip
        if self._current_face_id:
            with self._tracks_lock:
                t = self._tracks.get(self._current_face_id)
            if t and now - t.last_seen < 0.2:
                return
        
        # Priority 2: Saliency
        with self._saliency_lock:
            peak = self._saliency_peak
        
        if peak:
            u, v, score = peak
            
            # Get depth for angle calculation
            Z = self._get_depth_at(u, v)
            
            # Convert to base frame for tracking
            base_yaw, base_pitch = self._pixel_to_base_angles(u, v, Z)
            
            # Track saliency target for IOR (in base frame)
            if self._saliency_target:
                old_yaw, old_pitch, start = self._saliency_target
                dist = np.hypot(base_yaw - old_yaw, base_pitch - old_pitch)
                
                if dist > 0.1:  # ~6 degrees - new target
                    # Add old to IOR if dwelled
                    if now - start >= self.saliency_dwell_for_ior:
                        self._add_ior(old_yaw, old_pitch, "saliency")
                    self._saliency_target = (base_yaw, base_pitch, now)
            else:
                self._saliency_target = (base_yaw, base_pitch, now)
            
            # Set attention source (will reset IOR if switching from face)
            self._set_attention_source('saliency')
            
            # Compute camera-relative angles for head command
            if Z:
                yaw, pitch = uvZ_to_angles(u, v, Z, self.fx, self.fy, self.cx, self.cy)
            else:
                yaw, pitch = pixel_to_angles(u, v, self.fx, self.fy, self.cx, self.cy)
            
            self._publish_head(yaw, pitch)
            return
        
        # No saliency - add last target to IOR if dwelled
        if self._saliency_target:
            old_yaw, old_pitch, start = self._saliency_target
            if now - start >= self.saliency_dwell_for_ior:
                self._add_ior(old_yaw, old_pitch, "saliency")
            self._saliency_target = None
        
        # Priority 3: Audio
        if self._audio_azimuth is not None and now < self._audio_until:
            self._set_attention_source('audio')
            yaw = clamp(self._audio_azimuth, -self.yaw_lim, self.yaw_lim)
            pitch = clamp(self.home_pitch, self.pitch_dn, self.pitch_up)
            self._publish_head(yaw, pitch)
            return
        
        # Priority 4: Idle
        self._set_attention_source(None)
        yaw, pitch = self._idle_behavior(now)
        self._publish_head(yaw, pitch)

    def _idle_behavior(self, now) -> tuple[float, float]:
        """Generate idle scan with micro-saccades."""
        # Periodic scan
        if now >= self._next_idle_scan:
            dyaw = math.radians(random.uniform(-self.idle_yaw_deg, self.idle_yaw_deg))
            dpitch = math.radians(random.uniform(-self.idle_pitch_deg, self.idle_pitch_deg))
            
            self._idle_target = (
                clamp(self.home_yaw + dyaw, -self.yaw_lim, self.yaw_lim),
                clamp(self.home_pitch + dpitch, self.pitch_dn, self.pitch_up)
            )
            self._next_idle_scan = now + random.uniform(self.idle_min, self.idle_max)
        
        yaw, pitch = self._idle_target
        
        # Micro-saccade
        if now >= self._next_micro_saccade:
            myaw = math.radians(self.micro_deg) * random.choice([-1, 1])
            mpitch = math.radians(self.micro_deg) * random.choice([-1, 1])
            yaw = clamp(yaw + myaw, -self.yaw_lim, self.yaw_lim)
            pitch = clamp(pitch + mpitch, self.pitch_dn, self.pitch_up)
            self._next_micro_saccade = now + random.uniform(self.micro_min, self.micro_max)
        
        return yaw, pitch

    def _publish_head(self, yaw, pitch):
        """Publish head command with deadband."""
        yaw = clamp(float(yaw), -self.yaw_lim, self.yaw_lim)
        pitch = clamp(float(pitch), self.pitch_dn, self.pitch_up)
        
        if (abs(yaw - self._head_yaw) < self.deadband_yaw and 
            abs(pitch - self._head_pitch) < self.deadband_pitch):
            return
        
        js = JointAnglesWithSpeed()
        js.header.stamp = self.get_clock().now().to_msg()
        js.joint_names = self.joint_names
        js.joint_angles = [yaw, pitch]
        js.speed = 0.1
        js.relative = False
        self.pub_js.publish(js)
        
        # Debug
        v = Vector3(x=float(yaw), y=float(pitch), z=0.0)
        self.pub_dbg.publish(v)


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedAttention()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()