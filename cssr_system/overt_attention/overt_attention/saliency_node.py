#!/usr/bin/env python3
"""
Saliency Node - Computes bottom-up visual attention using Boolean Map Saliency (BMS)
Publishes peaks as /attn/saliency_peak
"""

import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray


def get_image_topic(base_topic: str, use_compressed: bool, is_depth: bool = False) -> str:
    """Construct the full topic name based on compression setting."""
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

class BooleanMapSaliency:
    """
    Boolean Map Saliency (BMS) - Frame-based
    
    - Threshold-based boolean maps (per BMS paper)
    - Spatial Gaussian smoothing
    - Output normalized to [0, 1]
    """

    def __init__(self, n_thresholds: int = 10):
        self.n_thresholds = n_thresholds
        self.thresholds = np.linspace(0, 1, n_thresholds + 1, endpoint=False)[1:]

    def activate_boolean_map(self, bool_map: np.ndarray) -> np.ndarray:
        """
        Activate boolean map using flood-fill to suppress background.
        Implements the core BMS region activation step.
        """
        activation = bool_map.astype(np.uint8)
        h, w = activation.shape
        ffill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # Flood-fill from image borders
        for y in range(h):
            if activation[y, 0]:
                cv2.floodFill(activation, ffill_mask, (0, y), 0)
            if activation[y, w - 1]:
                cv2.floodFill(activation, ffill_mask, (w - 1, y), 0)

        for x in range(w):
            if activation[0, x]:
                cv2.floodFill(activation, ffill_mask, (x, 0), 0)
            if activation[h - 1, x]:
                cv2.floodFill(activation, ffill_mask, (x, h - 1), 0)

        return activation

    def compute_saliency(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Compute saliency map.

        Args:
            frame_bgr: BGR image (H, W, 3), downsampled

        Returns:
            Saliency map (H, W), float32 in [0, 1]
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Global normalization
        lab_min, lab_max = lab.min(), lab.max()
        lab_range = lab_max - lab_min
        if lab_range < 1e-6:
            return np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.float32)
        lab = (lab - lab_min) / lab_range

        h, w = lab.shape[:2]
        saliency = np.zeros((h, w), dtype=np.float32)

        # Reorder to (C, H, W) for faster channel access
        lab_ch = lab.transpose(2, 0, 1)

        # Accumulate boolean activations
        for thresh in self.thresholds:
            for c in range(3):
                saliency += self.activate_boolean_map(lab_ch[c] > thresh)

        # Normalize accumulation
        saliency /= (self.n_thresholds * 3)

        return saliency.astype(np.float32)


class SaliencyNode(Node):
    def __init__(self):
        super().__init__('saliency_node')
        
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize dimensions
        self.W = 640
        self.H = 480
        
        # Thread-safe frame storage
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_stamp = None
        self._latest_size = None
        
        # Thread-safe depth storage
        self._depth_lock = threading.Lock()
        self._depth_small = None
        
        qos = get_image_qos()
        
        # Publishers
        self.pub_peak = self.create_publisher(Float32MultiArray, '/attn/saliency_peak', 10)
        
        self.pub_map = None
        if self.publish_map_flag:
            self.pub_map = self.create_publisher(
                CompressedImage, '/attn/saliency_map/compressed', 1)
        
        # Image subscriber
        image_topic = get_image_topic(self.image_topic_base, self.use_compressed, is_depth=False)
        if self.use_compressed:
            self.sub = self.create_subscription(
                CompressedImage, image_topic, self._on_image_compressed, qos)
        else:
            self.sub = self.create_subscription(
                Image, image_topic, self._on_image_raw, qos)
        
        self.get_logger().info(f"Subscribing to image: {image_topic}")
        
        # Depth subscriber
        if self.use_depth_weighting:
            depth_topic = get_image_topic(self.depth_topic_base, self.use_compressed, is_depth=True)
            if self.use_compressed:
                self.sub_depth = self.create_subscription(
                    CompressedImage, depth_topic, self._on_depth_compressed, qos)
            else:
                self.sub_depth = self.create_subscription(
                    Image, depth_topic, self._on_depth_raw, qos)
            self.get_logger().info(f"Subscribing to depth: {depth_topic}")
        
        self.bms = BooleanMapSaliency()
        
        # Processing timer
        timer_period = 1.0 / self.process_hz
        self.create_timer(timer_period, self._timer_callback)
        
        self.get_logger().info(
            f"Saliency node ready @ {self.process_hz} Hz "
            f"(depth_weighting={'ON' if self.use_depth_weighting else 'OFF'})"
        )

    def _declare_parameters(self):
        """Declare all ROS parameters."""
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('image_topic_base', '/camera/color/image_raw')
        self.declare_parameter('depth_topic_base', '/camera/depth/image_rect_raw')
        self.declare_parameter('use_depth_weighting', True)
        self.declare_parameter('depth_min_m', 0.3)
        self.declare_parameter('depth_max_m', 10.0)
        self.declare_parameter('depth_weight_min', 0.2)
        self.declare_parameter('publish_map', True)
        self.declare_parameter('down_w', 160)
        self.declare_parameter('down_h', 120)
        self.declare_parameter('min_peak', 0.25)
        self.declare_parameter('overlay_alpha', 0.4)
        self.declare_parameter('num_peaks', 10)
        self.declare_parameter('peak_min_distance_px', 50)
        self.declare_parameter('process_hz', 1.0)

    def _load_parameters(self):
        """Load parameters into instance variables."""
        self.use_compressed = self.get_parameter('use_compressed').value
        self.image_topic_base = self.get_parameter('image_topic_base').value
        self.depth_topic_base = self.get_parameter('depth_topic_base').value
        self.use_depth_weighting = self.get_parameter('use_depth_weighting').value
        self.depth_min_m = self.get_parameter('depth_min_m').value
        self.depth_max_m = self.get_parameter('depth_max_m').value
        self.depth_weight_min = self.get_parameter('depth_weight_min').value
        self.publish_map_flag = self.get_parameter('publish_map').value
        self.down_w = self.get_parameter('down_w').value
        self.down_h = self.get_parameter('down_h').value
        self.min_peak = self.get_parameter('min_peak').value
        self.overlay_alpha = self.get_parameter('overlay_alpha').value
        self.num_peaks = self.get_parameter('num_peaks').value
        self.peak_min_dist = self.get_parameter('peak_min_distance_px').value
        self.process_hz = self.get_parameter('process_hz').value

    # ============ Image Callbacks (just cache frames) ============
    
    def _on_image_raw(self, msg: Image):
        """Handle raw Image messages."""
        if msg.encoding.lower() in ('bgr8', 'rgb8'):
            bgr = np.frombuffer(msg.data, np.uint8).reshape((msg.height, msg.width, 3))
            if msg.encoding.lower() == 'rgb8':
                bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
            with self._frame_lock:
                self._latest_frame = bgr
                self._latest_stamp = msg.header.stamp
                self._latest_size = (msg.width, msg.height)

    def _on_image_compressed(self, msg: CompressedImage):
        """Handle CompressedImage messages."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            H, W = bgr.shape[:2]
            with self._frame_lock:
                self._latest_frame = bgr
                self._latest_stamp = msg.header.stamp
                self._latest_size = (W, H)

    def _on_depth_raw(self, msg: Image):
        """Handle raw depth messages."""
        if msg.encoding.lower() in ('16uc1', 'mono16'):
            depth = np.frombuffer(msg.data, np.uint16).reshape((msg.height, msg.width))
            depth_m = depth.astype(np.float32) / 1000.0
        elif msg.encoding.lower() == '32fc1':
            depth_m = np.frombuffer(msg.data, np.float32).reshape((msg.height, msg.width))
        else:
            return
        self._cache_depth(depth_m)

    def _on_depth_compressed(self, msg: CompressedImage):
        """Handle compressedDepth messages."""
        if len(msg.data) <= 12:
            return
        
        # Skip 12-byte header and decode PNG
        depth = cv2.imdecode(np.frombuffer(msg.data[12:], np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is not None:
            depth_m = depth.astype(np.float32) / 1000.0
            self._cache_depth(depth_m)

    def _cache_depth(self, depth_m: np.ndarray):
        """Downsample and cache depth."""
        depth_small = cv2.resize(
            depth_m, (self.down_w, self.down_h), 
            interpolation=cv2.INTER_NEAREST)
        with self._depth_lock:
            self._depth_small = depth_small

    # ============ Timer Callback ============
    
    def _timer_callback(self):
        """Process latest frame at fixed rate."""
        # Grab latest frame
        with self._frame_lock:
            if self._latest_frame is None:
                return
            bgr = self._latest_frame
            stamp = self._latest_stamp
            full_size = self._latest_size
        
        self._process_frame(bgr, full_size, stamp)

    # ============ Processing ============
    
    def _compute_depth_weight(self) -> np.ndarray:
        """Compute depth-based weighting (closer = higher weight)."""
        with self._depth_lock:
            if self._depth_small is None:
                return np.ones((self.down_h, self.down_w), dtype=np.float32)
            depth = self._depth_small.copy()
        
        # Handle invalid depth
        invalid = (depth <= 0) | (depth > 10.0)
        depth[invalid] = self.depth_max_m
        
        # Normalize and invert (close = high weight)
        depth_range = self.depth_max_m - self.depth_min_m
        if depth_range < 1e-6:
            return np.ones((self.down_h, self.down_w), dtype=np.float32)
        
        normalized = np.clip((depth - self.depth_min_m) / depth_range, 0, 1)
        weight = 1.0 - normalized * (1.0 - self.depth_weight_min)
        
        return weight.astype(np.float32)

    def _find_peaks(self, S: np.ndarray) -> list:
        """Find top N spatially separated peaks."""
        peaks = []
        S_work = S.copy()
        h, w = S_work.shape
        
        scale_x = self.W / float(self.down_w)
        scale_y = self.H / float(self.down_h)
        avg_scale = (scale_x + scale_y) / 2.0
        min_dist_down = max(1, self.peak_min_dist / avg_scale)
        
        # Edge padding
        pad = max(3, int(min(h, w) * 0.05))
        
        for _ in range(self.num_peaks):
            # Mask edges
            S_masked = S_work.copy()
            S_masked[:pad, :] = 0
            S_masked[-pad:, :] = 0
            S_masked[:, :pad] = 0
            S_masked[:, -pad:] = 0
            
            # Find max
            max_idx = np.argmax(S_masked)
            v_s, u_s = np.unravel_index(max_idx, S_masked.shape)
            score = float(S_masked[v_s, u_s])
            
            if score < self.min_peak:
                break
            
            # Convert to full resolution
            u = float(u_s * scale_x)
            v = float(v_s * scale_y)
            peaks.append([u, v, score])
            
            # Suppress region around peak
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - u_s)**2 + (yy - v_s)**2 <= min_dist_down**2
            S_work[mask] = 0
        
        return peaks

    def _process_frame(self, bgr: np.ndarray, full_size: tuple, stamp):
        """Main processing pipeline."""
        self.W, self.H = full_size
        
        # Downsample
        small_bgr = cv2.resize(bgr, (self.down_w, self.down_h), interpolation=cv2.INTER_AREA)
        
        # Compute BMS saliency
        S = self.bms.compute_saliency(small_bgr)
        
        # Apply depth weighting
        if self.use_depth_weighting:
            S = S * self._compute_depth_weight()
        
        # Smooth and normalize
        S = cv2.GaussianBlur(S, (5, 5), 0)
        s_min, s_max = S.min(), S.max()
        if (s_max - s_min) > 1e-6:
            S = (S - s_min) / (s_max - s_min)
        else:
            S = np.zeros_like(S)
        
        # Find peaks
        peaks = self._find_peaks(S)
        
        # Publish peaks (u, v, score) - 3 values per peak
        msg = Float32MultiArray()
        msg.data = []
        for u, v, score in peaks:
            msg.data.extend([u, v, score])
        self.pub_peak.publish(msg)
        
        # Visualization
        if self.pub_map is not None:
            self._publish_visualization(bgr, S, peaks, stamp)

    def _publish_visualization(self, bgr, S, peaks, stamp):
        """Publish saliency visualization."""
        # Resize saliency to full resolution
        vis = cv2.resize((S * 255).astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        vis_color = cv2.addWeighted(bgr, 1.0, vis_color, self.overlay_alpha, 0)
        
        # Draw peaks
        colors = [(255, 255, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)]
        for idx, (u, v, score) in enumerate(peaks):
            color = colors[idx] if idx < len(colors) else (200, 200, 200)
            cv2.drawMarker(vis_color, (int(u), int(v)), color,
                          markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
            cv2.putText(vis_color, f"{idx+1}:{score:.2f}", (int(u) + 12, int(v)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Publish
        out = CompressedImage()
        out.format = 'png'
        out.header.stamp = stamp
        out.data = cv2.imencode('.png', vis_color)[1].tobytes()
        self.pub_map.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = SaliencyNode()
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