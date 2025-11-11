#!/usr/bin/env python3
"""
Saliency Node - Computes bottom-up visual attention
Uses Boolean Map Saliency (BMS) + motion (optical flow)
Publishes peak location as /attn/saliency_peak
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo, JointState
from std_msgs.msg import Float32MultiArray

class BooleanMapSaliency:
    """BMS - Fast boolean map based saliency with temporal filtering"""
    def __init__(self, quantization_level=12):
        self.prev_saliency = None
        self.alpha = 0.3  # Temporal smoothing factor
        self.quantization_level = quantization_level
        
    def compute_saliency(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        
        # Convert to LAB color space for better perceptual uniformity
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Quantize colors (key insight of BMS - reduces color space for efficiency)
        lab_quant = (lab // self.quantization_level).astype(np.int32)
        
        # Create boolean maps for each quantized color
        unique_colors = {}
        for y in range(h):
            for x in range(w):
                color = tuple(lab_quant[y, x])
                if color not in unique_colors:
                    unique_colors[color] = np.zeros((h, w), dtype=bool)
                unique_colors[color][y, x] = True
        
        # Compute attention map
        attention = np.zeros((h, w), dtype=np.float32)
        
        for color, bool_map in unique_colors.items():
            # Find connected components for this color
            num_labels, labels = cv2.connectedComponents(bool_map.astype(np.uint8))
            
            for label in range(1, num_labels):
                component = (labels == label)
                
                # Compute features
                area = component.sum()
                if area < 5:  # Skip tiny regions (noise)
                    continue
                
                # Centroid
                y_coords, x_coords = np.where(component)
                cy, cx = y_coords.mean(), x_coords.mean()
                
                # Compactness (how circular/blob-like the region is)
                contours, _ = cv2.findContours(
                    component.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )[-2:]
                
                if len(contours) > 0 and len(contours[0]) > 0:
                    perimeter = cv2.arcLength(contours[0], True)
                    compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
                else:
                    compactness = 0.0
                
                # Distance to border (center bias)
                border_dist = min(cy, h-cy, cx, w-cx)
                border_dist_norm = border_dist / (min(h, w) / 2 + 1e-6)
                
                # Saliency value: combines compactness, center bias, and size
                # Compact, centered, medium-sized regions are most salient
                sal_value = compactness * border_dist_norm * np.log(area + 1)
                attention[component] += sal_value
        
        # Smooth to create coherent regions
        attention = cv2.GaussianBlur(attention, (15, 15), 0)
        
        # Temporal filtering (exponential smoothing)
        if self.prev_saliency is not None:
            attention = self.alpha * attention + (1 - self.alpha) * self.prev_saliency
        self.prev_saliency = attention.copy()
        
        # Normalize to [0, 1]
        return (attention - attention.min()) / (attention.max() - attention.min() + 1e-6)


class SaliencyNode(Node):
    def __init__(self):
        super().__init__('saliency_node')
        
        # Parameters
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('publish_map', True)
        self.declare_parameter('down_w', 160)
        self.declare_parameter('down_h', 120)
        self.declare_parameter('alpha_static', 0.6)
        self.declare_parameter('beta_motion', 0.3)
        self.declare_parameter('gamma_center', 0.1)
        self.declare_parameter('min_peak', 0.25)
        self.declare_parameter('flow_method', 'farneback')
        
        # BMS-specific parameters
        self.declare_parameter('bms_quantization', 12)
        
        # Motion gating parameters
        self.declare_parameter('head_motion_threshold', 0.08)
        self.declare_parameter('skip_during_motion', True)
        
        # Visualization parameters
        self.declare_parameter('overlay_alpha', 0.4)
        self.declare_parameter('visualize_components', True)
        
        self.visualize_components = self.get_parameter('visualize_components').value
        
        # Multi-peak parameters
        self.declare_parameter('num_peaks', 5)
        self.declare_parameter('peak_min_distance_px', 40)

        # Load parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.publish_map_flag = self.get_parameter('publish_map').value
        self.down_w = self.get_parameter('down_w').value
        self.down_h = self.get_parameter('down_h').value
        self.ALPHA = self.get_parameter('alpha_static').value
        self.BETA = self.get_parameter('beta_motion').value
        self.GAMMA = self.get_parameter('gamma_center').value
        self.MIN_PEAK = self.get_parameter('min_peak').value
        self.flow_method = self.get_parameter('flow_method').value.lower()
        self.declare_parameter('motion_focus_threshold', 0.3)
        
        self.bms_quantization = self.get_parameter('bms_quantization').value
        
        self.head_threshold = self.get_parameter('head_motion_threshold').value
        self.skip_during_motion = self.get_parameter('skip_during_motion').value
        
        self.overlay_alpha = self.get_parameter('overlay_alpha').value

        # Multi-peak parameters
        self.num_peaks = self.get_parameter('num_peaks').value
        self.peak_min_dist = self.get_parameter('peak_min_distance_px').value

        self.motion_focus_threshold = self.get_parameter('motion_focus_threshold').value
        
        # Publishers
        if self.visualize_components:
            self.pub_static = self.create_publisher(CompressedImage, '/attn/saliency_static/compressed', 1)
            self.pub_motion = self.create_publisher(CompressedImage, '/attn/saliency_motion/compressed', 1)
            self.pub_center = self.create_publisher(CompressedImage, '/attn/saliency_center/compressed', 1)
            self.pub_combined = self.create_publisher(CompressedImage, '/attn/saliency_combined/compressed', 1)
        
        # QoS for Wi-Fi
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, 
                        durability=DurabilityPolicy.VOLATILE, 
                        history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.pub_peak = self.create_publisher(Float32MultiArray, '/attn/saliency_peak', 10)
        
        self.pub_map = None
        if self.publish_map_flag:
            self.pub_map = self.create_publisher(CompressedImage, '/attn/saliency_map/compressed', 1)
        
        # Subscriber for images
        if self.image_topic.endswith('/compressed'):
            self.sub = self.create_subscription(CompressedImage, self.image_topic, 
                                               self.on_img_compressed, qos)
        else:
            self.sub = self.create_subscription(Image, self.image_topic, 
                                               self.on_img_raw, qos)
        
        # Subscribe to joint states for motion gating
        self.create_subscription(JointState, '/joint_states', self.on_joint_states, 10)
        
        # Initialize BMS
        self.bms = BooleanMapSaliency(quantization_level=self.bms_quantization)
                
        # State
        self.prev_small = None
        self.prev_small_bgr = None
        self.dis = None
        if self.flow_method == 'dis':
            try:
                self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            except Exception:
                self.get_logger().warn("DIS not available, using Farneback")
                self.flow_method = 'farneback'
        
        self.W = self.H = None
        self.head_velocity = 0.0
        
        self.get_logger().info(
            f"Saliency node ready (BMS) on {self.image_topic} "
            f"(motion_gating={'ON' if self.skip_during_motion else 'OFF'}, "
            f"thresh={self.head_threshold:.3f} rad/s, "
            f"bms_quantization={self.bms_quantization})"
        )

    def publish_simple_visualization(self, bgr, S, peaks, W, H, stamp):
        """Simple saliency visualization with all peaks marked"""
        # Resize saliency to full resolution
        vis_small = (S * 255.0).astype(np.uint8)
        vis_full = cv2.resize(vis_small, (W, H), interpolation=cv2.INTER_LINEAR)
        saliency_colored = cv2.applyColorMap(vis_full, cv2.COLORMAP_JET)
        
        # Overlay on original image
        vis_color = cv2.addWeighted(bgr, 1.0, saliency_colored, self.overlay_alpha, 0)
        
        # Mark all peaks (different colors/sizes by rank)
        colors = [
            (255, 255, 255),  # White - highest
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 0),      # Green
            (255, 128, 0)     # Orange
        ]
        marker_sizes = [20, 16, 14, 12, 10]
        
        for idx, (u, v, score) in enumerate(peaks):
            color = colors[idx] if idx < len(colors) else (128, 128, 128)
            size = marker_sizes[idx] if idx < len(marker_sizes) else 8
            
            # Draw marker
            cv2.drawMarker(vis_color, (int(u), int(v)), color,
                        markerType=cv2.MARKER_TILTED_CROSS, 
                        markerSize=size, thickness=2)
            
            # Draw rank number
            cv2.putText(vis_color, str(idx + 1), (int(u) + 12, int(v) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw score near peak
            score_text = f"{score:.2f}"
            cv2.putText(vis_color, score_text, (int(u) + 12, int(v) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status text
        status = f"Peaks: {len(peaks)}/{self.num_peaks}"
        cv2.putText(vis_color, status, (10, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.skip_during_motion:
            motion_status = f"Head: {self.head_velocity:.3f} rad/s"
            color = (0, 255, 0) if not self.is_head_moving() else (0, 165, 255)
            cv2.putText(vis_color, motion_status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Publish
        enc = cv2.imencode('.png', vis_color)[1].tobytes()
        out = CompressedImage()
        out.format = 'png'
        out.header.stamp = stamp
        out.data = enc
        self.pub_map.publish(out)


    def publish_component_visualization(self, S_static, S_motion, C, S_final, W, H, stamp, bgr_full):
        """Publish visualization of individual saliency components"""
        if not self.visualize_components:
            return
        
        overlay_alpha = self.overlay_alpha
        
        def component_to_image(component, width, height, base_image):
            """Convert component to colorized overlay on real image and find top 3 peaks"""
            vis = (component * 255.0).astype(np.uint8)
            vis = cv2.resize(vis, (width, height), interpolation=cv2.INTER_LINEAR)
            saliency_colored = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            
            vis_color = cv2.addWeighted(base_image.copy(), 1.0, saliency_colored, overlay_alpha, 0)
            
            component_copy = component.copy()
            peaks = []
            min_dist = 10
            
            for rank in range(3):
                max_idx = np.argmax(component_copy)
                v_s, u_s = np.unravel_index(max_idx, component_copy.shape)
                peak_val = component_copy[v_s, u_s]
                
                scale_x = width / float(component_copy.shape[1])
                scale_y = height / float(component_copy.shape[0])
                u = int(u_s * scale_x)
                v = int(v_s * scale_y)
                
                peaks.append((u, v, peak_val))
                
                y_min = max(0, v_s - min_dist)
                y_max = min(component_copy.shape[0], v_s + min_dist + 1)
                x_min = max(0, u_s - min_dist)
                x_max = min(component_copy.shape[1], u_s + min_dist + 1)
                component_copy[y_min:y_max, x_min:x_max] = 0
            
            return vis_color, peaks
        
        vis_static, peaks_static = component_to_image(S_static, W, H, bgr_full)
        vis_motion, peaks_motion = component_to_image(S_motion, W, H, bgr_full)
        vis_center, peaks_center = component_to_image(C, W, H, bgr_full)
        vis_final, peaks_final = component_to_image(S_final, W, H, bgr_full)
        
        colors = [(255, 255, 255), (0, 255, 255), (255, 0, 255)]
        marker_sizes = [20, 16, 12]
        
        for vis, peaks in [(vis_static, peaks_static), 
                          (vis_motion, peaks_motion),
                          (vis_center, peaks_center),
                          (vis_final, peaks_final)]:
            for idx, (u, v, val) in enumerate(peaks):
                cv2.drawMarker(vis, (u, v), colors[idx],
                             markerType=cv2.MARKER_TILTED_CROSS, 
                             markerSize=marker_sizes[idx], thickness=2)
                cv2.putText(vis, str(idx + 1), (u + 12, v - 12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)
        
        cv2.putText(vis_static, "Static (BMS)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_motion, "Motion (Optical Flow)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_center, "Center Prior", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_final, f"Final (a={self.ALPHA:.1f} b={self.BETA:.1f} g={self.GAMMA:.1f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for vis, pub in [(vis_static, self.pub_static), 
                        (vis_motion, self.pub_motion),
                        (vis_center, self.pub_center),
                        (vis_final, self.pub_combined)]:
            enc = cv2.imencode('.png', vis)[1].tobytes()
            msg = CompressedImage()
            msg.format = 'png'
            msg.header.stamp = stamp
            msg.data = enc
            pub.publish(msg)

    def find_top_n_peaks(self, S):
        """Find top N spatially separated peaks
        
        Args:
            S: Saliency map (downsampled resolution)
        
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
        min_dist_downsampled = self.peak_min_dist / scale_x
        
        for i in range(self.num_peaks):
            # Find maximum in remaining map
            v_s, u_s = np.unravel_index(np.argmax(S_copy), S_copy.shape)
            score = float(S_copy[v_s, u_s])
            
            # Stop if below threshold
            if score < self.MIN_PEAK:
                break
            
            # Convert to full resolution
            u = float(u_s * scale_x)
            v = float(v_s * scale_y)
            peaks.append([u, v, score])
            
            # Suppress circular region around this peak
            # Use meshgrid for efficient circular mask
            yy, xx = np.ogrid[:h, :w]
            mask = (xx - u_s)**2 + (yy - v_s)**2 <= min_dist_downsampled**2
            S_copy[mask] = 0
        
        return peaks

    def on_joint_states(self, msg: JointState):
        """Track head motion for gating"""
        try:
            yaw_idx = msg.name.index('HeadYaw')
            pitch_idx = msg.name.index('HeadPitch')
            
            yaw_vel = msg.velocity[yaw_idx]
            pitch_vel = msg.velocity[pitch_idx]
            
            yaw_vel = 0.0 if (yaw_vel != yaw_vel) else yaw_vel
            pitch_vel = 0.0 if (pitch_vel != pitch_vel) else pitch_vel
            
            self.head_velocity = np.sqrt(yaw_vel**2 + pitch_vel**2)
            
        except (ValueError, IndexError):
            pass

    def is_head_moving(self):
        """Check if head is moving above threshold"""
        return self.head_velocity > self.head_threshold

    def on_img_raw(self, msg: Image):
        """Handle raw Image messages"""
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
        """Handle CompressedImage messages"""
        np_arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        
        H, W = bgr.shape[:2]
        self.process_frame(bgr, (W, H), msg.header.stamp)

    def motion_map(self, gray_small: np.ndarray) -> np.ndarray:
        """Compute motion saliency via optical flow"""
        if self.prev_small is None:
            return np.zeros_like(gray_small, dtype=np.float32)
        
        if self.flow_method == 'dis' and self.dis is not None:
            flow = self.dis.calc(self.prev_small, gray_small, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_small, gray_small,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
        
        h, w = flow.shape[:2]
        step = 8
        flow_samples = flow[::step, ::step].reshape(-1, 2)
        
        try:
            bins = 20
            hist_x, edges_x = np.histogram(flow_samples[:, 0], bins=bins)
            hist_y, edges_y = np.histogram(flow_samples[:, 1], bins=bins)
            
            mode_x = (edges_x[np.argmax(hist_x)] + edges_x[np.argmax(hist_x) + 1]) / 2
            mode_y = (edges_y[np.argmax(hist_y)] + edges_y[np.argmax(hist_y) + 1]) / 2
            
            ego_x, ego_y = mode_x, mode_y
        except:
            ego_x = np.median(flow[..., 0])
            ego_y = np.median(flow[..., 1])
        
        flow[..., 0] -= ego_x
        flow[..., 1] -= ego_y
        
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        motion_threshold = 0.5
        mag = np.where(mag < motion_threshold, 0, mag)
        
        motion_percentile_95 = np.percentile(mag, 95)
        motion_mean = np.mean(mag[mag > 0]) if np.any(mag > 0) else 0
        
        if motion_percentile_95 < 0.5 or motion_mean < 0.3:
            return np.zeros_like(gray_small, dtype=np.float32) * 0.1
        
        mag = cv2.GaussianBlur(mag, (5, 5), 0)
        
        kernel_size = 5
        mag_dilated = cv2.dilate(mag, np.ones((kernel_size, kernel_size)))
        mag = np.where(mag == mag_dilated, mag, mag * 0.3)
        
        m95 = np.percentile(mag, 95)
        
        if m95 < 1.0:
            scale_factor = m95 / 1.0
        else:
            scale_factor = 1.0
        
        M = np.clip(mag / (m95 + 1e-6), 0, 1).astype(np.float32)
        M = M * scale_factor
        
        M = np.where(M < 0.15, 0, M)
        
        if M.max() > 0:
            M = (M - M.min()) / (M.max() - M.min() + 1e-6)
        
        return M.astype(np.float32)

    def center_prior(self, w: int, h: int) -> np.ndarray:
        """Compute center bias"""
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w/2.0, h/2.0
        C = 1.0 - np.sqrt(((xx-cx)**2 + (yy-cy)**2)) / np.sqrt(cx**2 + cy**2)
        return np.clip(C, 0, 1).astype(np.float32)

    def process_frame(self, bgr: np.ndarray, full_size, stamp):
        """Main processing pipeline with multiple peaks"""
        
        # Set dimensions first
        W, H = full_size
        self.W, self.H = W, H
        
        # Motion gating
        if self.skip_during_motion and self.is_head_moving():
            return        
        
        # Downsample
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.down_w, self.down_h), interpolation=cv2.INTER_AREA)
        small_bgr = cv2.resize(bgr, (self.down_w, self.down_h), interpolation=cv2.INTER_AREA)
        
        # Compute components
        S_static = self.bms.compute_saliency(small_bgr)
        S_motion = self.motion_map(small)
        C = self.center_prior(self.down_w, self.down_h)
        
        # Check if there's significant motion
        motion_score = S_motion.max()
        has_significant_motion = motion_score > 0.3  # Threshold for "something is moving"
        
        # Fuse components
        S = self.ALPHA*S_static + self.BETA*S_motion + self.GAMMA*C
        S = cv2.GaussianBlur(S, (5, 5), 0)
        S = (S - S.min()) / (S.max() - S.min() + 1e-6)
        
        # ========================================
        # MOTION FOCUS: If motion detected, only return motion peak
        # ========================================
        if has_significant_motion:
            # Find single peak in motion map
            pad = 2
            crop = S_motion[pad:-pad, pad:-pad]
            v_s, u_s = np.unravel_index(np.argmax(crop), crop.shape)
            u_s += pad
            v_s += pad
            
            # Scale to full resolution
            scale_x = W / float(self.down_w)
            scale_y = H / float(self.down_h)
            u = float(u_s * scale_x)
            v = float(v_s * scale_y)
            
            # Use combined saliency score at motion location
            score = float(S[v_s, u_s])
            
            if score >= self.MIN_PEAK:
                # Publish ONLY the motion peak
                msg = Float32MultiArray()
                msg.data = [u, v, score]
                self.pub_peak.publish(msg)
                
                peaks = [[u, v, score]]  # For visualization
            else:
                peaks = []
        else:
            # No significant motion - find multiple static peaks
            peaks = self.find_top_n_peaks(S)
            
            # Publish all peaks as flat array: [u1, v1, s1, u2, v2, s2, ...]
            if peaks:
                msg = Float32MultiArray()
                msg.data = []
                for u, v, score in peaks:
                    msg.data.extend([u, v, score])
                self.pub_peak.publish(msg)
        
        # Visualization
        if self.pub_map and self.publish_map_flag and peaks:
            self.publish_simple_visualization(bgr, S, peaks, W, H, stamp)
        
        # Component visualization
        self.publish_component_visualization(S_static, S_motion, C, S, W, H, stamp, bgr)
        
        # Update state
        self.prev_small = small
        self.prev_small_bgr = small_bgr
    
def main(args=None):
    rclpy.init(args=args)
    node = SaliencyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()