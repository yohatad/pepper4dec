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
        # Pre-compute thresholds
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

    def compute_saliency(self, frame_bgr: np.ndarray, apply_blur: bool = True) -> np.ndarray:
        """
        Compute saliency map.

        Args:
            frame_bgr: BGR image (H, W, 3), downsampled
            apply_blur: Whether to apply Gaussian blur (set False if caller will blur)

        Returns:
            Saliency map (H, W), float32 in [0, 1]
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lab = lab.astype(np.float32)

        # Global normalization with proper handling of uniform images
        lab_min = lab.min()
        lab_max = lab.max()
        lab_range = lab_max - lab_min
        if lab_range < 1e-6:
            # Uniform image - return zero saliency
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

        # Spatial smoothing (optional - caller may want to do final blur)
        if apply_blur:
            saliency = cv2.GaussianBlur(saliency, (0, 0), 3)

        # Final normalization to [0, 1]
        s_min = saliency.min()
        s_max = saliency.max()
        s_range = s_max - s_min
        if s_range < 1e-6:
            return np.zeros_like(saliency, dtype=np.float32)
        saliency = (saliency - s_min) / s_range

        return saliency.astype(np.float32)

class SaliencyNode(Node):
    def __init__(self):
        super().__init__('saliency_node')
        
        # Declare and load parameters
        self.declare_all_parameters()
        self.load_parameters()
        
        # Initialize dimensions with sensible defaults
        self.W = 640
        self.H = 480
        
        # Thread lock for depth data access
        self._depth_lock = threading.Lock()
        self._depth_small = None  # Protected by _depth_lock
        
        # QoS for Wi-Fi
        qos = get_image_qos()
        
        # Publishers - create visualization publisher only if needed
        if self.visualize_components:
            self.pub_combined = self.create_publisher(
                CompressedImage, '/attn/saliency_combined/compressed', 1)
        else:
            self.pub_combined = None
        
        self.pub_peak = self.create_publisher(Float32MultiArray, '/attn/saliency_peak', 10)
        
        # Only create map publisher if flag is set
        self.pub_map = None
        if self.publish_map_flag:
            self.pub_map = self.create_publisher(
                CompressedImage, '/attn/saliency_map/compressed', 1)
        
        # Subscriber for images
        image_topic = get_image_topic(self.image_topic_base, self.use_compressed, is_depth=False)
        if self.use_compressed:
            self.sub = self.create_subscription(
                CompressedImage, image_topic, self.on_img_compressed, qos)
        else:
            self.sub = self.create_subscription(
                Image, image_topic, self.on_img_raw, qos)
        
        self.get_logger().info(f"Subscribing to image: {image_topic}")
        
        # Subscriber for depth images (for depth weighting)
        if self.use_depth_weighting:
            depth_topic = get_image_topic(self.depth_topic_base, self.use_compressed, is_depth=True)
            if self.use_compressed:
                self.sub_depth = self.create_subscription(
                    CompressedImage, depth_topic, self.on_depth_compressed, qos)
            else:
                self.sub_depth = self.create_subscription(
                    Image, depth_topic, self.on_depth_raw, qos)
            self.get_logger().info(f"Subscribing to depth: {depth_topic}")
        
        # Initialize BMS
        self.bms = BooleanMapSaliency()
        
        self.get_logger().info(
            f"Saliency node ready (Pure BMS) "
            f"(use_compressed={self.use_compressed}, "
            f"depth_weighting={'ON' if self.use_depth_weighting else 'OFF'})"
        )

    def declare_all_parameters(self):
        """Declare all ROS parameters with defaults."""
        # Global shared parameter
        self.declare_parameter('use_compressed', True)
        
        # Image topic (base, without /compressed suffix)
        self.declare_parameter('image_topic_base', '/camera/color/image_raw')
        self.declare_parameter('depth_topic_base', '/camera/depth/image_rect_raw')
        
        # Depth weighting parameters
        self.declare_parameter('use_depth_weighting', True)
        self.declare_parameter('depth_min_m', 0.3)  # Closer than this = max weight
        self.declare_parameter('depth_max_m', 4.0)  # Farther than this = min weight
        self.declare_parameter('depth_weight_min', 0.2)  # Weight at max distance
        
        # Processing parameters
        self.declare_parameter('publish_map', True)
        self.declare_parameter('down_w', 160)
        self.declare_parameter('down_h', 120)
        
        self.declare_parameter('min_peak', 0.25)
        
        # Visualization parameters
        self.declare_parameter('overlay_alpha', 0.4)
        self.declare_parameter('visualize_components', True)
        
        # Multi-peak parameters
        self.declare_parameter('num_peaks', 5)
        self.declare_parameter('peak_min_distance_px', 40)

    def load_parameters(self):
        """Load all parameters into instance variables."""
        # Global shared parameter
        self.use_compressed = self.get_parameter('use_compressed').value
        
        # Image topic base
        self.image_topic_base = self.get_parameter('image_topic_base').value
        self.depth_topic_base = self.get_parameter('depth_topic_base').value
        
        # Depth weighting
        self.use_depth_weighting = self.get_parameter('use_depth_weighting').value
        self.depth_min_m = self.get_parameter('depth_min_m').value
        self.depth_max_m = self.get_parameter('depth_max_m').value
        self.depth_weight_min = self.get_parameter('depth_weight_min').value
        
        # Processing parameters
        self.publish_map_flag = self.get_parameter('publish_map').value
        self.down_w = self.get_parameter('down_w').value
        self.down_h = self.get_parameter('down_h').value
        
        self.MIN_PEAK = self.get_parameter('min_peak').value
        
        # Visualization
        self.overlay_alpha = self.get_parameter('overlay_alpha').value
        self.visualize_components = self.get_parameter('visualize_components').value
        
        # Multi-peak
        self.num_peaks = self.get_parameter('num_peaks').value
        self.peak_min_dist = self.get_parameter('peak_min_distance_px').value

    def on_img_raw(self, msg: Image):
        """Handle raw Image messages."""
        if msg.encoding.lower() in ('bgr8', 'rgb8'):
            bgr = np.frombuffer(msg.data, np.uint8).reshape(
                (msg.height, msg.width, 3)
            )
            if msg.encoding.lower() == 'rgb8':
                bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        else:
            return
        
        self.process_frame(bgr, (msg.width, msg.height), msg.header.stamp)

    def on_img_compressed(self, msg: CompressedImage):
        """Handle CompressedImage messages."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        
        H, W = bgr.shape[:2]
        self.process_frame(bgr, (W, H), msg.header.stamp)

    def on_depth_raw(self, msg: Image):
        """Handle raw depth Image messages."""
        # RealSense depth is typically 16UC1 (millimeters)
        if msg.encoding.lower() in ('16uc1', 'mono16'):
            depth = np.frombuffer(msg.data, np.uint16).reshape(
                (msg.height, msg.width))
            depth_m = depth.astype(np.float32) / 1000.0  # Convert to meters
        elif msg.encoding.lower() == '32fc1':
            depth_m = np.frombuffer(msg.data, np.float32).reshape(
                (msg.height, msg.width))
        else:
            return
        
        self._process_depth(depth_m)

    def on_depth_compressed(self, msg: CompressedImage):
        """Handle CompressedImage depth messages (compressedDepth format)."""
        # compressedDepth uses PNG compression for 16-bit depth
        # First 12 bytes are a header with compression info
        if len(msg.data) < 12:
            return
        
        # Skip header and decode PNG
        png_data = msg.data[12:]
        depth = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_UNCHANGED)
        
        if depth is None:
            return
        
        # Convert to meters (RealSense depth is in millimeters)
        depth_m = depth.astype(np.float32) / 1000.0
        self._process_depth(depth_m)

    def _process_depth(self, depth_m: np.ndarray):
        """Downsample and cache depth for use in saliency computation."""
        # Downsample to match saliency resolution
        # Use INTER_NEAREST to avoid interpolating depth values
        depth_small = cv2.resize(
            depth_m, (self.down_w, self.down_h), 
            interpolation=cv2.INTER_NEAREST)
        
        # Thread-safe update
        with self._depth_lock:
            self._depth_small = depth_small

    def get_depth_small(self) -> np.ndarray:
        """Thread-safe getter for depth data."""
        with self._depth_lock:
            if self._depth_small is None:
                return None
            return self._depth_small.copy()

    def compute_depth_weight(self) -> np.ndarray:
        """
        Compute depth-based weighting map.
        Closer objects get higher weight.
        
        Returns:
            Weight map (down_h, down_w) with values in [depth_weight_min, 1.0]
        """
        depth_small = self.get_depth_small()
        if depth_small is None:
            # No depth available - return uniform weight
            return np.ones((self.down_h, self.down_w), dtype=np.float32)
        
        # depth_small is already a copy from get_depth_small()
        depth = depth_small
        
        # Handle invalid depth (0 or very large values)
        invalid_mask = (depth <= 0) | (depth > 10.0)
        depth[invalid_mask] = self.depth_max_m  # Treat invalid as far
        
        # Normalize depth to [0, 1] range
        # 0 = close (depth_min_m), 1 = far (depth_max_m)
        depth_range = self.depth_max_m - self.depth_min_m
        if depth_range < 1e-6:
            return np.ones((self.down_h, self.down_w), dtype=np.float32)
        
        depth_normalized = (depth - self.depth_min_m) / depth_range
        depth_normalized = np.clip(depth_normalized, 0, 1)
        
        # Invert: close = high weight, far = low weight
        # Scale to [depth_weight_min, 1.0]
        weight = 1.0 - depth_normalized * (1.0 - self.depth_weight_min)
        
        return weight.astype(np.float32)

    def compute_saliency_map(self, S_static: np.ndarray, depth_weight: np.ndarray) -> np.ndarray:
        """
        Compute final saliency map using depth-weighted BMS.
        
        Args:
            S_static: Static saliency map (BMS)
            depth_weight: Pre-computed depth weight map
            
        Returns:
            Final saliency map (values in [0, 1])
        """
        # Apply depth weighting to static saliency (closer objects score higher)
        if self.use_depth_weighting:
            S = S_static * depth_weight
        else:
            S = S_static
        
        return S

    def find_top_n_peaks(self, S: np.ndarray) -> list:
        """
        Find top N spatially separated peaks.
        
        Args:
            S: Final saliency map (downsampled resolution)
        
        Returns:
            List of [u, v, score] in full resolution coordinates
        """
        peaks = []
        S_copy = S.copy()
        h, w = S_copy.shape
        
        # Scale factors to full resolution
        scale_x = self.W / float(self.down_w)
        scale_y = self.H / float(self.down_h)
        
        # Convert min distance from full resolution to downsampled
        # Use average scale to handle non-uniform aspect ratios
        avg_scale = (scale_x + scale_y) / 2.0
        min_dist_downsampled = max(1, self.peak_min_dist / avg_scale)
        
        # Edge padding - use larger of 5% of image or 3 pixels
        pad = max(3, int(min(h, w) * 0.05))
        
        for i in range(self.num_peaks):
            # Mask edges to avoid boundary issues
            S_masked = S_copy.copy()
            S_masked[:pad, :] = 0
            S_masked[-pad:, :] = 0
            S_masked[:, :pad] = 0
            S_masked[:, -pad:] = 0
            
            # Find maximum
            max_idx = np.argmax(S_masked)
            v_s, u_s = np.unravel_index(max_idx, S_masked.shape)
            score = float(S_masked[v_s, u_s])
            
            # Stop if below threshold
            if score < self.MIN_PEAK:
                break
            
            # Convert to full resolution
            u = float(u_s * scale_x)
            v = float(v_s * scale_y)
            peaks.append([u, v, score])
            
            # Suppress circular region around this peak
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - u_s)**2 + (yy - v_s)**2 <= min_dist_downsampled**2
            S_copy[mask] = 0
        
        return peaks

    def process_frame(self, bgr: np.ndarray, full_size: tuple, stamp):
        """Main processing pipeline for pure BMS saliency."""
        
        # Update dimensions
        W, H = full_size
        self.W, self.H = W, H
        
        # Downsample
        small_bgr = cv2.resize(bgr, (self.down_w, self.down_h), interpolation=cv2.INTER_AREA)
        
        # Compute BMS saliency
        # Skip internal blur in BMS since we do final blur below
        S_static = self.bms.compute_saliency(small_bgr, apply_blur=False)
        
        # Compute depth weight for this frame
        depth_weight = self.compute_depth_weight()
        
        # Compute final saliency map (depth-weighted if enabled)
        S = self.compute_saliency_map(S_static, depth_weight)
        
        # Single smoothing pass on final result
        S = cv2.GaussianBlur(S, (5, 5), 0)
        
        # Normalize
        s_min, s_max = S.min(), S.max()
        s_range = s_max - s_min
        if s_range > 1e-6:
            S = (S - s_min) / s_range
        else:
            S = np.zeros_like(S)
        
        # Find peaks
        peaks = self.find_top_n_peaks(S)
        
        # ALWAYS publish - even empty list (so downstream knows we're alive)
        msg = Float32MultiArray()
        msg.data = []
        for peak in peaks:
            u, v, score = peak
            # For pure BMS, source is always static
            source_code = 0.0  # static
            msg.data.extend([u, v, score, source_code])
        self.pub_peak.publish(msg)
        
        # Visualization - only check pub_map existence (it's only created if flag was True)
        if self.pub_map is not None:
            self.publish_simple_visualization(bgr, S, peaks, W, H, stamp)
        
        # Component visualization
        if self.visualize_components and self.pub_combined is not None:
            self.publish_component_visualization(S, peaks, W, H, stamp, bgr)

    def publish_simple_visualization(self, bgr, S, peaks, W, H, stamp):
        """Simple saliency visualization with all peaks marked."""
        # Resize saliency to full resolution
        vis_small = (S * 255.0).astype(np.uint8)
        vis_full = cv2.resize(vis_small, (W, H), interpolation=cv2.INTER_LINEAR)
        saliency_colored = cv2.applyColorMap(vis_full, cv2.COLORMAP_JET)
        
        # Overlay on original image
        vis_color = cv2.addWeighted(bgr, 1.0, saliency_colored, self.overlay_alpha, 0)
        
        # Color scheme for peaks
        peak_color = (255, 200, 100)  # Light blue for BMS peaks
        marker_sizes = [20, 16, 14, 12, 10]
        
        for idx, peak in enumerate(peaks):
            u, v, score = peak
            color = peak_color
            size = marker_sizes[idx] if idx < len(marker_sizes) else 8
            
            # Draw marker
            cv2.drawMarker(vis_color, (int(u), int(v)), color,
                          markerType=cv2.MARKER_TILTED_CROSS,
                          markerSize=size, thickness=2)
            
            # Draw rank
            cv2.putText(vis_color, str(idx + 1), (int(u) + 12, int(v) - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw score
            cv2.putText(vis_color, f"{score:.2f}", (int(u) + 12, int(v) + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status text
        status = f"Peaks: {len(peaks)}/{self.num_peaks}"
        cv2.putText(vis_color, status, (10, H - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Publish
        enc = cv2.imencode('.png', vis_color)[1].tobytes()
        out = CompressedImage()
        out.format = 'png'
        out.header.stamp = stamp
        out.data = enc
        self.pub_map.publish(out)

    def publish_component_visualization(self, S_final, peaks, W, H, stamp, bgr_full):
        """Publish visualization of depth-weighted BMS saliency with consistent peaks."""
        # Resize saliency to full resolution
        vis_small = (S_final * 255.0).astype(np.uint8)
        vis_full = cv2.resize(vis_small, (W, H), interpolation=cv2.INTER_LINEAR)
        saliency_colored = cv2.applyColorMap(vis_full, cv2.COLORMAP_JET)
        
        # Overlay on original image
        vis_color = cv2.addWeighted(
            bgr_full.copy(), 1.0, saliency_colored, self.overlay_alpha, 0)
        
        # Draw peaks (use the same peaks from main pipeline for consistency)
        colors = [(255, 255, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)]
        marker_sizes = [20, 16, 14, 12, 10]
        
        for idx, peak in enumerate(peaks):
            u, v, score = peak[:3]
            color = colors[idx] if idx < len(colors) else (200, 200, 200)
            size = marker_sizes[idx] if idx < len(marker_sizes) else 8
            
            cv2.drawMarker(vis_color, (int(u), int(v)), color,
                          markerType=cv2.MARKER_TILTED_CROSS,
                          markerSize=size, thickness=2)
            cv2.putText(vis_color, str(idx + 1), (int(u) + 12, int(v) - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(vis_color, f"{score:.2f}", (int(u) + 12, int(v) + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add title
        title = "BMS Saliency" + (" + Depth" if self.use_depth_weighting else "")
        cv2.putText(vis_color, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Publish
        enc = cv2.imencode('.png', vis_color)[1].tobytes()
        msg = CompressedImage()
        msg.format = 'png'
        msg.header.stamp = stamp
        msg.data = enc
        self.pub_combined.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SaliencyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()