#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge
import signal
import sys

class ColorCompressor(Node):
    def __init__(self):
        super().__init__('color_compressor')
        
        self.bridge = CvBridge()
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.compress_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            CompressedImage,
            '/camera/color/compressed',
            10
        )
        
        self.declare_parameter('jpeg_quality', 80)
        self.get_logger().info('Color compressor started')
    
    def compress_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            quality = self.get_parameter('jpeg_quality').value
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, compressed = cv2.imencode('.jpg', cv_image, encode_param)
            
            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header
            compressed_msg.format = "jpeg"
            compressed_msg.data = compressed.tobytes()
            
            self.publisher.publish(compressed_msg)
            
        except Exception as e:
            self.get_logger().error(f'Compression failed: {e}')

def signal_handler(sig, frame):
    print('\nShutting down color compressor...')
    rclpy.shutdown()
    sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    
    # Register signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    node = ColorCompressor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down color compressor')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()