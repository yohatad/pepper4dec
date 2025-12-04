#!/usr/bin/env python3
"""
Unified Attention Controller for Pepper Robot
Priority: Faces → Saliency → Audio → Idle/Home
Fair exploration with unified IOR and scene change detection
"""

import math
import time
import random
import collections
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from builtin_interfaces.msg import Time, Duration
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import CameraInfo, JointState, Image, CompressedImage
from threading import Lock

from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from geometry_msgs.msg import Vector3

from cssr_interfaces.msg import FaceDetection

# For decompressing depth images
from cv_bridge import CvBridge
import cv2


# ============ Helper Functions ============
def get_image_topic(base_topic: str, use_compressed: bool, is_depth: bool = False) -> str:
    """
    Construct the full topic name based on compression setting.
    
    Args:
        base_topic: Base topic name (e.g., "/camera/depth/image_rect_raw")
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


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def depth_bonus(Z, lo, hi):
    """Triangular bonus for preferred depth range"""
    if Z is None:
        return 0.0
    mid, width = 0.5*(lo+hi), 0.5*(hi-lo)
    return max(0.0, 1.0 - abs(Z - mid)/width)


def pixel_to_angles(u, v, fx, fy, cx, cy):
    """Convert pixel to angles (simple pinhole)"""
    x = (u - cx)/fx
    y = (v - cy)/fy
    return math.atan2(x, 1.0), math.atan2(y, 1.0)


def uvZ_to_angles(u, v, Z, fx, fy, cx, cy):
    """Convert pixel + depth to angles"""
    Xc = (u - cx)/fx * Z
    Yc = (v - cy)/fy * Z
    Zc = Z
    return math.atan2(Xc, Zc), math.atan2(Yc, Zc)

# ============ Kalman Filter for Smoothing ============
class KF2D:
    """2D constant-velocity Kalman filter for pixel tracking"""
    def __init__(self):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1e3   
        self.last_t = None
        self.Q = np.eye(4, dtype=np.float32) * 1e-2  
        self.R = np.eye(2, dtype=np.float32) * 2.0   

    def update(self, u, v, now=None):
        t = now or time.time()
        if self.last_t is None:
            self.last_t = t
        dt = max(1e-3, t - self.last_t)
        self.last_t = t
        
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],  dtype=np.float32)
        
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        
        # Update
        z = np.array([[u], [v]], np.float32)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ H) @ self.P

    def predict(self, dt=0.05):
        """Predict position after dt seconds"""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        xp = F @ self.x
        return float(xp[0, 0]), float(xp[1, 0])

# ============ Track Class ============
class Track:
    """Represents a face track with depth and filtering"""
    def __init__(self, tid, u, v, w, h, mutual):
        self.id = tid
        self.u, self.v = float(u), float(v)
        self.w, self.h = float(w), float(h)
        self.mutual = bool(mutual)
        self.age = 0
        self.last_seen = time.time()
        self.Z = None
        self.validZ = False
        self.kf = KF2D()
        self.kf.update(self.u, self.v, self.last_seen)
        self.last_score = 0.0

# ============ Main Node ============
class UnifiedAttention(Node):
    def __init__(self):
        super().__init__("pepper_unified_attention")
        
        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Declare all parameters
        self.declare_all_parameters()  
        self.load_parameters()
        
        # QoS for camera/depth over Wi-Fi
        qos_img = get_image_qos()
        
        # Subscriptions
        self.create_subscription(FaceDetection, self.face_topic, self.on_faces, 10)
        self.create_subscription(Float32MultiArray, self.saliency_topic, self.on_saliency, 10)
        self.create_subscription(Float32, self.audio_topic, self.on_audio, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_caminfo, qos_img)
        self.create_subscription(JointState, '/joint_states', self.on_joint_states, 10)
        
        # Depth image subscription (compressed or raw) using utility
        depth_topic = get_image_topic(self.depth_image_topic, self.use_compressed, is_depth=True)
        if self.use_compressed:
            self.create_subscription(CompressedImage, depth_topic, self.on_depth_compressed, qos_img)
        else:
            self.create_subscription(Image, depth_topic, self.on_depth_raw, qos_img)
        self.get_logger().info(f"Subscribing to depth: {depth_topic}")
        
        # Publishers
        self.pub_js = self.create_publisher(JointAnglesWithSpeed, self.js_topic, 10)
        self.pub_dbg = self.create_publisher(Vector3, "/attn/target_angles", 10)
        
        # Depth image state (thread-safe)
        self.depth_image = None
        self.depth_image_stamp = None
        self.depth_lock = Lock()
        
        # State variables
        self.fx = self.fy = self.cx = self.cy = None
        self.tracks = {}
        self.tracks_lock = Lock()  # Thread safety for tracks
        self.prev_id = None
        self.prev_since = time.time()
        
        # ========================================
        # UNIFIED IOR QUEUE FOR ALL TARGETS
        # ========================================
        self.IOR = collections.deque(maxlen=20)  # (u, v, timestamp, source)
        self.frame_idx = 0
        
        # Saliency state
        self.saliency_peaks = []
        self.saliency_peak = None
        self.sal_depth_Z = None
        
        # Saliency tracking
        self.last_saliency_u = None
        self.last_saliency_v = None
        self.saliency_dwell_start = None
        
        # Audio state
        self.audio_azimuth = None
        self.audio_until = 0.0
        
        # ========================================
        # HEAD STATE TRACKING FOR SCENE CHANGE DETECTION
        # ========================================
        self.current_head_yaw = 0.0
        self.current_head_pitch = 0.0
        self.prev_head_yaw = None
        self.prev_head_pitch = None
        self.head_yaw_velocity = 0.0
        self.head_pitch_velocity = 0.0
        self.head_settled_time = time.time()
        self.head_last_reset_yaw = 0.0
        self.head_last_reset_pitch = 0.0
        
        # Idle state
        self.next_idle_scan_t = 0.0
        self.next_micro_t = 0.0
        self.current_idle_target = (self.home_yaw, self.home_pitch)
        
        # Timers
        self.create_timer(1.0/20.0, self.tick)
        self.create_timer(2.0, self.cleanup_stale_tracks)  # Track cleanup every 2s
        
        self.get_logger().info(
            f"Unified attention ready (mode={self.cmd_mode}, "
            f"unified IOR with scene change reset, "
            f"head_move_threshold={self.head_move_reset_threshold:.2f} rad, "
            f"depth_source={'compressed' if self.use_compressed else 'raw'})"
        )

    def declare_all_parameters(self):
        """Declare all ROS parameters with defaults"""
        # Global parameter (shared across nodes via /**:)
        self.declare_parameter("use_compressed", True)
        
        # Depth configuration
        self.declare_parameter("depth_image_topic", "/camera/depth/image_rect_raw")  # Base topic
        self.declare_parameter("depth_scale", 0.001)  # Scale factor to convert to meters
        
        self.declare_parameter("face_topic", "/faceDetection/data") 
        self.declare_parameter("saliency_topic", "/attn/saliency_peak")
        self.declare_parameter("audio_topic", "/audio/azimuth_rad")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        
        self.declare_parameter("command_mode", "joint_state")
        self.declare_parameter("joint_names", ["HeadYaw", "HeadPitch"])
        self.declare_parameter("joint_state_topic", "/joint_angles")
        
        self.declare_parameter("bias_face", 2.0)
        self.declare_parameter("mutual_gaze_bonus", 0.6)
        self.declare_parameter("saliency_min_score", 0.30)
        
        self.declare_parameter("depth_lo", 0.6)
        self.declare_parameter("depth_hi", 1.8)
        self.declare_parameter("depth_min", 0.35)
        self.declare_parameter("depth_max", 4.0)
        self.declare_parameter("depth_roi_px", 5)
        self.declare_parameter("track_timeout_s", 5.0)  # Stale track cleanup timeout

        self.declare_parameter("hysteresis_delta", 1.2)
        self.declare_parameter("hysteresis_hold_ms", 1200)
        self.declare_parameter("dwell_cap_s", 8.0)

        # Unified IOR parameters
        self.declare_parameter("ior_px_radius", 60.0)
        self.declare_parameter("ior_penalty", 2.0)  # Strong suppression
        
        # Scene change detection
        self.declare_parameter("head_move_reset_threshold", 0.5)  # radians (~30 degrees)
        
        self.declare_parameter("audio_hold_s", 2.0)
        
        # Motion gating parameters
        self.declare_parameter("head_velocity_threshold", 0.08)
        self.declare_parameter("head_settle_time", 5.0)
        self.declare_parameter("deadband_yaw", 0.06)
        self.declare_parameter("deadband_pitch", 0.06)
        
        # Saliency dwell time before adding IOR
        self.declare_parameter("saliency_dwell_time", 2.5)
        
        self.declare_parameter("home_yaw", 0.0)
        self.declare_parameter("home_pitch", -0.05)
        self.declare_parameter("idle_scan_period_min", 3.0)
        self.declare_parameter("idle_scan_period_max", 6.0)
        self.declare_parameter("idle_scan_yaw_deg", 20.0)
        self.declare_parameter("idle_scan_pitch_deg", 8.0)
        self.declare_parameter("micro_saccade_deg", 0.4)
        self.declare_parameter("micro_saccade_period_min", 3.0)
        self.declare_parameter("micro_saccade_period_max", 5.0)
        
        self.declare_parameter("yaw_lim", 1.8)
        self.declare_parameter("pitch_up", 0.4)
        self.declare_parameter("pitch_dn", -0.7)

    def load_parameters(self):
        """Load all parameters into instance variables"""
        self.use_compressed = self.get_parameter("use_compressed").value
        self.depth_image_topic = self.get_parameter("depth_image_topic").value
        self.depth_scale = self.get_parameter("depth_scale").value
        
        self.face_topic = self.get_parameter("face_topic").value
        self.saliency_topic = self.get_parameter("saliency_topic").value
        self.audio_topic = self.get_parameter("audio_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        
        self.cmd_mode = self.get_parameter("command_mode").value
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.js_topic = self.get_parameter("joint_state_topic").value
        
        self.bias_face = self.get_parameter("bias_face").value
        self.mutual_gaze_bonus = self.get_parameter("mutual_gaze_bonus").value
        self.saliency_min_score = self.get_parameter("saliency_min_score").value
        
        self.depth_lo = self.get_parameter("depth_lo").value
        self.depth_hi = self.get_parameter("depth_hi").value
        self.depth_min = self.get_parameter("depth_min").value
        self.depth_max = self.get_parameter("depth_max").value
        self.depth_roi_px = self.get_parameter("depth_roi_px").value
        self.track_timeout_s = self.get_parameter("track_timeout_s").value
        
        self.hysteresis_delta = self.get_parameter("hysteresis_delta").value
        self.hysteresis_hold_ms = self.get_parameter("hysteresis_hold_ms").value
        self.dwell_cap_s = self.get_parameter("dwell_cap_s").value
        
        self.ior_px_radius = self.get_parameter("ior_px_radius").value
        self.ior_penalty = self.get_parameter("ior_penalty").value
        
        self.head_move_reset_threshold = self.get_parameter("head_move_reset_threshold").value
        
        self.audio_hold_s = self.get_parameter("audio_hold_s").value
        
        # Motion gating parameters
        self.vel_threshold = self.get_parameter("head_velocity_threshold").value
        self.settle_time = self.get_parameter("head_settle_time").value
        self.deadband_yaw = self.get_parameter("deadband_yaw").value
        self.deadband_pitch = self.get_parameter("deadband_pitch").value
        
        # Saliency dwell time
        self.saliency_dwell_time = self.get_parameter("saliency_dwell_time").value
        
        self.home_yaw = self.get_parameter("home_yaw").value
        self.home_pitch = self.get_parameter("home_pitch").value
        self.idle_scan_min = self.get_parameter("idle_scan_period_min").value
        self.idle_scan_max = self.get_parameter("idle_scan_period_max").value
        self.idle_scan_yaw_deg = self.get_parameter("idle_scan_yaw_deg").value
        self.idle_scan_pitch_deg = self.get_parameter("idle_scan_pitch_deg").value
        self.micro_saccade_deg = self.get_parameter("micro_saccade_deg").value
        self.micro_period_min = self.get_parameter("micro_saccade_period_min").value
        self.micro_period_max = self.get_parameter("micro_saccade_period_max").value
        
        self.yaw_lim = self.get_parameter("yaw_lim").value
        self.pitch_up = self.get_parameter("pitch_up").value
        self.pitch_dn = self.get_parameter("pitch_dn").value

    # ============ Depth Image Callbacks ============
    def on_depth_raw(self, msg: Image):
        """Handle raw depth image"""
        try:
            # Convert to numpy array
            if msg.encoding == '16UC1':
                depth_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            elif msg.encoding == '32FC1':
                depth_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                depth_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            with self.depth_lock:
                self.depth_image = depth_img
                self.depth_image_stamp = msg.header.stamp
            
        except Exception as e:
            self.get_logger().warn(f"Failed to process raw depth image: {e}")

    def on_depth_compressed(self, msg: CompressedImage):
        """Handle compressed depth image (compressedDepth format)"""
        try:
            if 'compressedDepth' in msg.format:
                # compressedDepth format has a 12-byte header before the PNG data
                header_size = 12
                
                if len(msg.data) <= header_size:
                    self.get_logger().warn("Compressed depth data too short")
                    return
                
                # Decode the PNG/image data after header
                raw_data = np.frombuffer(msg.data[header_size:], dtype=np.uint8)
                depth_img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
                
                if depth_img is None:
                    self.get_logger().warn("Failed to decode compressed depth image")
                    return
                
            else:
                # Standard compressed image (not compressedDepth)
                raw_data = np.frombuffer(msg.data, dtype=np.uint8)
                depth_img = cv2.imdecode(raw_data, cv2.IMREAD_UNCHANGED)
                
                if depth_img is None:
                    self.get_logger().warn("Failed to decode compressed image")
                    return
            
            with self.depth_lock:
                self.depth_image = depth_img
                self.depth_image_stamp = msg.header.stamp
                
        except Exception as e:
            self.get_logger().warn(f"Failed to process compressed depth image: {e}")

    def get_depth_at_pixel(self, u, v, roi=None):
        """
        Get depth at pixel location with ROI median filtering.
        Returns depth in meters, or None if invalid.
        """
        if roi is None:
            roi = self.depth_roi_px
        
        # Thread-safe copy
        with self.depth_lock:
            if self.depth_image is None:
                return None
            depth_copy = self.depth_image.copy()
            stamp = self.depth_image_stamp
        
        # Check staleness (0.5s threshold)
        if stamp is not None:
            try:
                import rclpy.time
                age = (self.get_clock().now() - rclpy.time.Time.from_msg(stamp)).nanoseconds / 1e9
                if age > 0.5:
                    return None
            except Exception:
                pass
        
        h, w = depth_copy.shape[:2]
        
        # Clamp coordinates
        u_int = int(round(u))
        v_int = int(round(v))
        
        if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h:
            return None
        
        # Extract ROI
        half_roi = roi // 2
        u_min = max(0, u_int - half_roi)
        u_max = min(w, u_int + half_roi + 1)
        v_min = max(0, v_int - half_roi)
        v_max = min(h, v_int + half_roi + 1)
        
        roi_data = depth_copy[v_min:v_max, u_min:u_max]
        
        # Filter out invalid values (0 typically means no reading)
        if roi_data.dtype == np.float32 or roi_data.dtype == np.float64:
            valid_mask = np.isfinite(roi_data) & (roi_data > 0)
        else:
            valid_mask = roi_data > 0
        
        valid_depths = roi_data[valid_mask]
        
        if len(valid_depths) == 0:
            return None
        
        # Compute median and convert to meters
        median_depth = float(np.median(valid_depths))
        depth_meters = median_depth * self.depth_scale
        
        # Sanity check
        if depth_meters < self.depth_min or depth_meters > self.depth_max:
            return None
        
        return depth_meters

    # ============ Track Management ============
    def cleanup_stale_tracks(self):
        """Remove tracks not seen recently"""
        now = time.time()
        with self.tracks_lock:
            stale_ids = [
                tid for tid, t in self.tracks.items() 
                if (now - t.last_seen) > self.track_timeout_s
            ]
            for tid in stale_ids:
                del self.tracks[tid]
        
        if stale_ids:
            self.get_logger().debug(f"Cleaned up {len(stale_ids)} stale tracks")

    # ============ IOR Management ============
    def add_ior_site(self, u, v, source):
        """Add IOR site (permanent until scene change)"""
        now = time.time()
        self.IOR.append((u, v, now, source))
        self.get_logger().info(
            f"Added {source} IOR at ({u:.0f}, {v:.0f}). "
            f"Total IOR sites: {len(self.IOR)}"
        )
    
    def check_ior_suppression(self, u, v):
        """Check if location is suppressed by IOR (no decay)"""
        for (u_i, v_i, t_i, source) in list(self.IOR):
            dist = np.sqrt((u - u_i)**2 + (v - v_i)**2)
            if dist < self.ior_px_radius:
                return True, source
        return False, None
    
    def reset_ior(self, reason=""):
        """Clear all IOR sites (scene changed)"""
        if len(self.IOR) > 0:
            self.get_logger().info(
                f"Resetting IOR ({len(self.IOR)} sites cleared). "
                f"Reason: {reason}"
            )
            self.IOR.clear()
            # Reset head position tracking
            self.head_last_reset_yaw = self.current_head_yaw
            self.head_last_reset_pitch = self.current_head_pitch

    # ============ Callbacks ============
    def on_joint_states(self, msg: JointState):
        """Track head position and detect scene changes"""
        try:
            yaw_idx = msg.name.index('HeadYaw')
            pitch_idx = msg.name.index('HeadPitch')
            
            self.prev_head_yaw = self.current_head_yaw
            self.prev_head_pitch = self.current_head_pitch
            
            self.current_head_yaw = msg.position[yaw_idx]
            self.current_head_pitch = msg.position[pitch_idx]
            
            # Get velocities (handle NaN values)
            yaw_vel = msg.velocity[yaw_idx]
            pitch_vel = msg.velocity[pitch_idx]
            
            self.head_yaw_velocity = 0.0 if (yaw_vel != yaw_vel) else yaw_vel
            self.head_pitch_velocity = 0.0 if (pitch_vel != pitch_vel) else pitch_vel
            
            # Track when head becomes stationary
            if self.is_head_moving():
                self.head_settled_time = time.time()
            
            # ========================================
            # SCENE CHANGE DETECTION
            # ========================================
            if self.prev_head_yaw is not None:
                # Calculate movement since last IOR reset
                yaw_moved = abs(self.current_head_yaw - self.head_last_reset_yaw)
                pitch_moved = abs(self.current_head_pitch - self.head_last_reset_pitch)
                total_moved = np.sqrt(yaw_moved**2 + pitch_moved**2)
                
                # If head moved significantly, reset IOR
                if total_moved > self.head_move_reset_threshold:
                    self.reset_ior(
                        f"Head moved {math.degrees(total_moved):.1f}° "
                        f"(threshold: {math.degrees(self.head_move_reset_threshold):.1f}°)"
                    )
                
        except (ValueError, IndexError):
            pass

    def is_head_moving(self):
        """Check if head is currently moving above threshold"""
        speed = np.sqrt(self.head_yaw_velocity**2 + self.head_pitch_velocity**2)
        return speed > self.vel_threshold

    def is_head_settled(self):
        """Check if head has been still for settle_time"""
        return (time.time() - self.head_settled_time) > self.settle_time

    def on_caminfo(self, msg: CameraInfo):
        """Store camera intrinsics"""
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def on_audio(self, msg: Float32):
        """Store audio azimuth with timeout"""
        self.audio_azimuth = float(msg.data)
        self.audio_until = time.time() + self.audio_hold_s

    def on_saliency(self, msg: Float32MultiArray):
        """Process multiple saliency peaks with IOR suppression"""
        if len(msg.data) < 3:
            self.saliency_peaks = []
            self.saliency_peak = None
            return
        
        candidates = []
        
        # Parse all peaks from message
        for i in range(0, len(msg.data), 3):
            if i + 2 >= len(msg.data):
                break
            
            u = float(msg.data[i])
            v = float(msg.data[i+1])
            score = float(msg.data[i+2])
            
            # Check IOR suppression (no decay - permanent)
            suppressed, ior_source = self.check_ior_suppression(u, v)
            
            if suppressed:
                self.get_logger().debug(
                    f"Saliency peak at ({u:.0f}, {v:.0f}) suppressed "
                    f"by {ior_source} IOR"
                )
                final_score = 0.0  # Completely suppress
            else:
                final_score = score
            
            candidates.append((u, v, final_score, score))
        
        # Store all candidates
        self.saliency_peaks = candidates
        
        # Pick best non-suppressed peak
        valid_candidates = [(u, v, s, o) for u, v, s, o in candidates if s > 0]
        
        if valid_candidates:
            best = max(valid_candidates, key=lambda x: x[2])
            u_best, v_best, final_score, orig_score = best
            
            if final_score >= self.saliency_min_score:
                self.saliency_peak = (u_best, v_best, final_score)
                self.get_logger().debug(
                    f"Best saliency peak: ({u_best:.0f}, {v_best:.0f}) "
                    f"score={final_score:.2f}"
                )
            else:
                self.saliency_peak = None
        else:
            self.saliency_peak = None
            self.get_logger().debug("All saliency peaks suppressed by IOR")

    def on_faces(self, msg: FaceDetection):
        """Process face detections - Priority 1"""
        if self.fx is None:
            return
        
        self.frame_idx += 1
        now = time.time()
        
        # Update/create tracks
        cand = []
        n = len(msg.centroids)
        
        for i in range(n):
            fid = msg.face_label_id[i] if i < len(msg.face_label_id) else str(i)
            c = msg.centroids[i]
            w = msg.width[i] if i < len(msg.width) else 80.0
            h = msg.height[i] if i < len(msg.height) else 80.0
            mg = msg.mutual_gaze[i] if i < len(msg.mutual_gaze) else False
            
            if fid not in self.tracks:
                self.tracks[fid] = Track(fid, c.x, c.y, w, h, mg)
            
            T = self.tracks[fid]
            T.u, T.v = float(c.x), float(c.y)
            T.w, T.h = float(w), float(h)
            T.mutual = bool(mg)
            T.age += 1
            T.last_seen = now
            T.kf.update(T.u, T.v, now)
            cand.append(T)
        
        if not cand:
            return
        
        # Score faces with IOR suppression and update depth
        for T in cand:
            # Query depth from cached image
            depth_z = self.get_depth_at_pixel(T.u, T.v, self.depth_roi_px)
            if depth_z is not None:
                T.Z = depth_z
                T.validZ = True
            else:
                T.validZ = False
                T.Z = None
            
            s = self.bias_face
            
            if T.mutual:
                s += self.mutual_gaze_bonus
            
            if T.validZ and (self.depth_min <= T.Z <= self.depth_max):
                s += depth_bonus(T.Z, self.depth_lo, self.depth_hi)
            
            # Check IOR suppression (permanent)
            suppressed, ior_source = self.check_ior_suppression(T.u, T.v)
            if suppressed:
                s -= self.ior_penalty  # Heavy penalty
                self.get_logger().debug(
                    f"Face {T.id} at ({T.u:.0f}, {T.v:.0f}) "
                    f"suppressed by {ior_source} IOR"
                )
            
            T.last_score = s
        
        # Choose with hysteresis
        target, changed = self.choose(cand)
        if not target:
            return
        
        # If we switched away from saliency, add saliency to IOR
        if changed and self.last_saliency_u is not None:
            if self.saliency_dwell_start is not None:
                dwell = now - self.saliency_dwell_start
                if dwell >= self.saliency_dwell_time:
                    self.add_ior_site(
                        self.last_saliency_u,
                        self.last_saliency_v,
                        "saliency"
                    )
            self.last_saliency_u = None
            self.last_saliency_v = None
            self.saliency_dwell_start = None
        
        # Compute angles (KF prediction for pursuit)
        u_cmd, v_cmd = target.kf.predict(0.05)
        
        if target.validZ:
            yaw, pitch = uvZ_to_angles(u_cmd, v_cmd, target.Z,
                                      self.fx, self.fy, self.cx, self.cy)
        else:
            yaw, pitch = pixel_to_angles(u_cmd, v_cmd,
                                        self.fx, self.fy, self.cx, self.cy)
        
        self.publish_head(yaw, pitch)
        self.publish_debug(yaw, pitch)

    def tick(self):
        """General attention loop - handles saliency/audio/idle"""
        if self.fx is None:
            return
        
        now = time.time()
        
        # Only process saliency/audio when head is settled
        if not self.is_head_settled():
            return
        
        # If we have recent face, let on_faces handle it
        if self.prev_id is not None:
            tr = self.tracks.get(self.prev_id)
            if tr and (now - tr.last_seen) < 0.2:
                return
        
        # ========================================
        # Priority 2: Saliency
        # ========================================
        if self.saliency_peak and self.saliency_peak[2] >= self.saliency_min_score:
            u, v, score = self.saliency_peak
            
            # Check if new target
            is_new_target = False
            
            if self.last_saliency_u is not None:
                dist = np.sqrt((u - self.last_saliency_u)**2 + 
                              (v - self.last_saliency_v)**2)
                
                if dist > 50:  # New target threshold
                    is_new_target = True
                    
                    # Add OLD location to IOR if dwelled
                    if self.saliency_dwell_start is not None:
                        dwell = now - self.saliency_dwell_start
                        if dwell >= self.saliency_dwell_time:
                            self.add_ior_site(
                                self.last_saliency_u,
                                self.last_saliency_v,
                                "saliency"
                            )
                    
                    # Reset for new target
                    self.saliency_dwell_start = now
                    self.sal_depth_Z = None
                    
                    self.get_logger().info(
                        f"Switching saliency: "
                        f"({self.last_saliency_u:.0f}, {self.last_saliency_v:.0f}) → "
                        f"({u:.0f}, {v:.0f})"
                    )
            else:
                is_new_target = True
                self.saliency_dwell_start = now
            
            # Update tracking
            self.last_saliency_u = u
            self.last_saliency_v = v
            
            # Query depth from cached image
            self.sal_depth_Z = self.get_depth_at_pixel(u, v, self.depth_roi_px)
            
            # Compute angles
            if self.sal_depth_Z and (self.depth_min <= self.sal_depth_Z <= self.depth_max):
                yaw, pitch = uvZ_to_angles(u, v, self.sal_depth_Z,
                                          self.fx, self.fy, self.cx, self.cy)
            else:
                yaw, pitch = pixel_to_angles(u, v,
                                            self.fx, self.fy, self.cx, self.cy)
            
            self.publish_head(yaw, pitch)
            self.publish_debug(yaw, pitch)
            return
        
        # No valid saliency - add last one to IOR if dwelled
        if self.last_saliency_u is not None:
            if self.saliency_dwell_start is not None:
                dwell = now - self.saliency_dwell_start
                if dwell >= self.saliency_dwell_time:
                    self.add_ior_site(
                        self.last_saliency_u,
                        self.last_saliency_v,
                        "saliency"
                    )
        
        self.last_saliency_u = None
        self.last_saliency_v = None
        self.saliency_dwell_start = None
        
        # ========================================
        # Priority 3: Audio
        # ========================================
        if self.audio_azimuth is not None and now < self.audio_until:
            yaw = clamp(self.audio_azimuth, -self.yaw_lim, self.yaw_lim)
            pitch = clamp(self.home_pitch, self.pitch_dn, self.pitch_up)
            self.publish_head(yaw, pitch)
            self.publish_debug(yaw, pitch)
            return
        
        # ========================================
        # Priority 4: Idle
        # ========================================
        yaw, pitch = self.idle_behavior()
        self.publish_head(yaw, pitch)
        self.publish_debug(yaw, pitch)

    # ============ Selection Logic ============
    def choose(self, tracks):
        """Select best face track with hysteresis and dwell cap"""
        if not tracks:
            return None, False
        
        best = max(tracks, key=lambda t: t.last_score)
        now = time.time()
        
        # Hysteresis
        if self.prev_id is not None and best.id != self.prev_id:
            prev = self.tracks.get(self.prev_id)
            if prev:
                if best.last_score < prev.last_score + self.hysteresis_delta:
                    if (now - self.prev_since) < self.hysteresis_hold_ms/1000.0:
                        return prev, False
        
        # Dwell cap
        if self.prev_id is not None:
            prev = self.tracks.get(self.prev_id)
            if prev and (now - self.prev_since) > self.dwell_cap_s:
                self.add_ior_site(prev.u, prev.v, "face")
                self.prev_id = None
        
        changed = (self.prev_id is None) or (best.id != self.prev_id)
        
        # Add previous face to IOR when switching
        if changed and self.prev_id is not None:
            prev = self.tracks.get(self.prev_id)
            if prev:
                self.add_ior_site(prev.u, prev.v, "face")
        
        self.prev_id = best.id
        self.prev_since = now
        
        return best, changed

    # ============ Idle Behavior ============
    def idle_behavior(self):
        """Generate idle scan targets with micro-saccades"""
        now = time.time()
        
        if now >= self.next_idle_scan_t:
            dyaw = math.radians(
                random.uniform(-self.idle_scan_yaw_deg, self.idle_scan_yaw_deg)
            )
            dpit = math.radians(
                random.uniform(-self.idle_scan_pitch_deg, self.idle_scan_pitch_deg)
            )
            
            tgt_yaw = clamp(self.home_yaw + dyaw, -self.yaw_lim, self.yaw_lim)
            tgt_pitch = clamp(self.home_pitch + dpit, self.pitch_dn, self.pitch_up)
            
            self.current_idle_target = (tgt_yaw, tgt_pitch)
            self.next_idle_scan_t = now + random.uniform(
                self.idle_scan_min, self.idle_scan_max
            )
        
        yaw, pitch = self.current_idle_target
        
        if now >= self.next_micro_t:
            myaw = math.radians(self.micro_saccade_deg) * random.choice([-1, 1])
            mpit = math.radians(self.micro_saccade_deg) * random.choice([-1, 1])
            
            yaw = clamp(yaw + myaw, -self.yaw_lim, self.yaw_lim)
            pitch = clamp(pitch + mpit, self.pitch_dn, self.pitch_up)
            
            self.next_micro_t = now + random.uniform(
                self.micro_period_min, self.micro_period_max
            )
        
        return yaw, pitch

    # ============ Publishers ============
    def publish_head(self, yaw, pitch):
        """Publish head command with deadband"""
        yaw = clamp(float(yaw), -self.yaw_lim, self.yaw_lim)
        pitch = clamp(float(pitch), self.pitch_dn, self.pitch_up)
        
        error_yaw = abs(yaw - self.current_head_yaw)
        error_pitch = abs(pitch - self.current_head_pitch)
        
        if error_yaw < self.deadband_yaw and error_pitch < self.deadband_pitch:
            return
        
        js = JointAnglesWithSpeed()
        js.header.stamp = self.get_clock().now().to_msg()
        js.joint_names = self.joint_names
        js.joint_angles = [yaw, pitch]
        js.speed = 0.1
        js.relative = False
        
        self.pub_js.publish(js)
       
    def publish_debug(self, yaw, pitch):
        """Publish debug info"""
        v = Vector3()
        v.x, v.y = float(yaw), float(pitch)
        v.z = float(self.saliency_peak[2] if self.saliency_peak else 0.0)
        self.pub_dbg.publish(v)


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedAttention()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()