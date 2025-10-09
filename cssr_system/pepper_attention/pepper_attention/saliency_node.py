#!/usr/bin/env python3
"""
Saliency Node - Computes bottom-up visual attention
Fuses static saliency (spectral residual) + motion (optical flow)
Publishes peak location as /attn/saliency_peak
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import Float32MultiArray


class SaliencyNode(Node):
    def __init__(self):
        super().__init__('saliency_node')
        
        # Parameters
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('publish_map', True)
        self.declare_parameter('down_w', 160)
        self.declare_parameter('down_h', 120)
        self.declare_parameter('alpha_static', 0.6)
        self.declare_parameter('beta_motion', 0.4)
        self.declare_parameter('gamma_center', 0.1)
        self.declare_parameter('min_peak', 0.25)
        self.declare_parameter('flow_method', 'farneback')
        
        self.image_topic = self.get_parameter('image_topic').value
        self.publish_map_flag = self.get_parameter('publish_map').value
        self.down_w = self.get_parameter('down_w').value
        self.down_h = self.get_parameter('down_h').value
        self.ALPHA = self.get_parameter('alpha_static').value
        self.BETA = self.get_parameter('beta_motion').value
        self.GAMMA = self.get_parameter('gamma_center').value
        self.MIN_PEAK = self.get_parameter('min_peak').value
        self.flow_method = self.get_parameter('flow_method').value.lower()
        
        # QoS for Wi-Fi
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE,history=HistoryPolicy.KEEP_LAST, depth=1)
        
        # Publishers
        self.pub_peak = self.create_publisher(Float32MultiArray, '/attn/saliency_peak', 10)
        
        self.pub_map = None
        if self.publish_map_flag:
            self.pub_map = self.create_publisher(CompressedImage,'/attn/saliency_map/compressed',1)
        
        # Subscriber (compressed or raw)
        if self.image_topic.endswith('/compressed'):
            self.sub = self.create_subscription(CompressedImage, self.image_topic, self.on_img_compressed, qos)
        else:
            self.sub = self.create_subscription(Image, self.image_topic, self.on_img_raw, qos)

        # State
        self.prev_small = None
        self.dis = None
        if self.flow_method == 'dis':
            try:
                self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            except Exception:
                self.get_logger().warn("DIS not available, using Farneback")
                self.flow_method = 'farneback'
        
        self.W = self.H = None
        self.get_logger().info(f"Saliency node ready on {self.image_topic}")

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

    def spectral_residual(self, gray_small: np.ndarray) -> np.ndarray:
        """Compute static saliency via spectral residual"""
        g = gray_small.astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        
        F = np.fft.fft2(g)
        A = np.abs(F)
        L = np.log(A + 1e-6)
        L_avg = cv2.blur(L, (3, 3))
        R = L - L_avg
        
        S = np.abs(np.fft.ifft2(np.exp(R + 1j*np.angle(F))))**2
        S = cv2.GaussianBlur(S, (3, 3), 0)
        S = (S - S.min()) / (S.max() - S.min() + 1e-6)

        return S.astype(np.float32)

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
        
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mag = cv2.GaussianBlur(mag, (3, 3), 0)
        
        # Robust normalize by 95th percentile
        m95 = np.percentile(mag, 95)
        M = np.clip(mag / (m95 + 1e-6), 0, 1).astype(np.float32)
        
        return M

    def center_prior(self, w: int, h: int) -> np.ndarray:
        """Compute center bias"""
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w/2.0, h/2.0
        C = 1.0 - np.sqrt(((xx-cx)**2 + (yy-cy)**2)) / np.sqrt(cx**2 + cy**2)
        return np.clip(C, 0, 1).astype(np.float32)

    def process_frame(self, bgr: np.ndarray, full_size, stamp):
        """Main processing pipeline"""
        W, H = full_size
        self.W, self.H = W, H
        
        # Downsample
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.down_w, self.down_h), interpolation=cv2.INTER_AREA)
        
        # Compute components
        S_static = self.spectral_residual(small)
        S_motion = self.motion_map(small)
        C = self.center_prior(self.down_w, self.down_h)
        
        # Fuse
        S = self.ALPHA*S_static + self.BETA*S_motion + self.GAMMA*C
        S = cv2.GaussianBlur(S, (5, 5), 0)
        S = (S - S.min()) / (S.max() - S.min() + 1e-6)
        
        # Find peak (avoid borders)
        pad = 2
        crop = S[pad:-pad, pad:-pad]
        v_s, u_s = np.unravel_index(np.argmax(crop), crop.shape)
        u_s += pad
        v_s += pad
        peak = float(S[v_s, u_s])
        
        # Scale to full resolution
        scale_x = W / float(self.down_w)
        scale_y = H / float(self.down_h)
        u = float(u_s * scale_x)
        v = float(v_s * scale_y)
        
        # Publish if strong enough
        if peak >= self.MIN_PEAK:
            msg = Float32MultiArray()
            msg.data = [u, v, peak]
            self.pub_peak.publish(msg)
        
        # Optional visualization
        if self.pub_map and self.publish_map_flag:
            vis_small = (S * 255.0).astype(np.uint8)
            vis_full = cv2.resize(
                vis_small, 
                (W, H), 
                interpolation=cv2.INTER_LINEAR
            )
            vis_color = cv2.applyColorMap(vis_full, cv2.COLORMAP_JET)
            
            # Mark peak
            cv2.drawMarker(
                vis_color, 
                (int(u), int(v)), 
                (255, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS, 
                markerSize=16, 
                thickness=2
            )
            
            enc = cv2.imencode('.png', vis_color)[1].tobytes()
            out = CompressedImage()
            out.format = 'png'
            out.header.stamp = stamp
            out.data = enc
            self.pub_map.publish(out)
        
        # Update state
        self.prev_small = small


def main(args=None):
    rclpy.init(args=args)
    node = SaliencyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()