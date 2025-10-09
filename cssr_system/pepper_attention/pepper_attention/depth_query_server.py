#!/usr/bin/env python3
"""
Depth Query Service for Jetson Nano
Maintains a small buffer of aligned depth frames and returns Z at (u,v) on request
"""
import rclpy
import numpy as np
import collections
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# Import your custom service (create this in srv/ folder)
from cssr_interfaces.srv import GetDepthAtPixel


class DepthQueryServer(Node):
    def __init__(self):
        super().__init__('depth_query_server')
        
        self.bridge = CvBridge()
        self.depth_buf = collections.deque(maxlen=6)  # ~200ms @ 30 FPS
        
        self.declare_parameter('depth_scale', 0.001)
        self.depth_scale = self.get_parameter('depth_scale').value
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, 
            depth=1
        )
        
        self.sub = self.create_subscription(
            Image, 
            '/camera/aligned_depth_to_color/image_raw', 
            self.depth_cb, 
            qos
        )
        
        self.srv = self.create_service(
            GetDepthAtPixel, 
            'get_depth_at_pixel', 
            self.handle_req
        )
        
        self.get_logger().info('Depth query server ready.')

    def depth_cb(self, msg: Image):
        """Store incoming depth frames with timestamp"""
        arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_buf.append((msg.header.stamp, arr, msg.encoding))

    @staticmethod
    def _ts(t: Time) -> float:
        """Convert Time to float seconds"""
        return t.sec + t.nanosec * 1e-9

    def handle_req(self, request, response):
        """Return median depth in ROI around (u,v)"""
        response.valid = False
        
        if not self.depth_buf:
            return response
        
        # Choose frame nearest to requested timestamp (or latest if t=0)
        if request.t.sec or request.t.nanosec:
            tgt = self._ts(request.t)
            stamp, depth, enc = min(
                self.depth_buf, 
                key=lambda it: abs(self._ts(it[0]) - tgt)
            )
        else:
            stamp, depth, enc = self.depth_buf[-1]
        
        h, w = depth.shape[:2]
        u = int(np.clip(request.u, 0, w-1))
        v = int(np.clip(request.v, 0, h-1))
        r = int(np.clip(request.roi, 0, 10))
        
        # Extract ROI
        win = depth[
            max(0, v-r):min(h, v+r+1),
            max(0, u-r):min(w, u+r+1)
        ]
        
        # Convert to meters
        if enc in ('16UC1', 'mono16') or str(win.dtype) == 'uint16':
            vals = win.astype(np.float32) * float(self.depth_scale)
        else:  # 32FC1 already meters
            vals = win.astype(np.float32)
        
        # Filter valid range
        good = vals[(vals > 0.2) & (vals < 6.0) & np.isfinite(vals)]
        
        if good.size == 0:
            return response
        
        response.z_mean = float(np.mean(good))
        response.z_median = float(np.median(good))
        response.valid = True
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DepthQueryServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()