#!/usr/bin/env python3
"""
Depth Service Provider Node
Provides a ROS2 service for querying depth values at specific image coordinates.
This reduces bandwidth by only sending depth data when requested.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from cssr_interfaces.srv import GetDepthAtPixel
from typing import Optional, List, Tuple
import threading


class DepthServiceProvider(Node):
    def __init__(self):
        super().__init__('depth_service_provider')
        
        # Parameters
        self.declare_parameter('camera_type', 'realsense')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('service_name', '/realsense/depth_query')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('buffer_time', 0.5)  # How long to keep depth images
        
        self.camera_type = self.get_parameter('camera_type').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.service_name = self.get_parameter('service_name').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.buffer_time = self.get_parameter('buffer_time').value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Thread-safe depth image buffer
        self.depth_lock = threading.Lock()
        self.latest_depth: Optional[np.ndarray] = None
        self.depth_timestamp: Optional[float] = None
        
        # Create service
        self.depth_service = self.create_service(GetDepthAtPixel, self.service_name,self.handle_depth_query)
        
        # Subscribe to depth topic
        if self.use_compressed:
            self.depth_sub = self.create_subscription(CompressedImage, self.depth_topic + '/compressedDepth', self.depth_compressed_callback, 10)
        else:
            self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        
        self.get_logger().info(f"Depth service provider started at {self.service_name}")
        self.get_logger().info(f"Subscribing to depth topic: {self.depth_topic}")
        
    def depth_callback(self, msg: Image):
        """Handle uncompressed depth image."""
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            with self.depth_lock:
                self.latest_depth = depth_image
                self.depth_timestamp = self.get_clock().now().nanoseconds / 1e9
                
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
    
    def depth_compressed_callback(self, msg: CompressedImage):
        """Handle compressed depth image."""
        try:
            if hasattr(msg, "format") and msg.format and "compressedDepth png" in msg.format:
                # Handle PNG compression
                depth_header_size = 12
                depth_img_data = msg.data[depth_header_size:]
                np_arr = np.frombuffer(depth_img_data, np.uint8)
                depth_image = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)
            else:
                # Regular compressed image
                np_arr = np.frombuffer(msg.data, np.uint8)
                depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            
            with self.depth_lock:
                self.latest_depth = depth_image
                self.depth_timestamp = self.get_clock().now().nanoseconds / 1e9
                
        except Exception as e:
            self.get_logger().error(f"Error processing compressed depth image: {e}")
    
    def handle_depth_query(self, request: GetDepthAtPixel.Request, response: GetDepthAtPixel.Response):
        """Handle depth query service request."""
        with self.depth_lock:
            # Check if we have recent depth data
            if self.latest_depth is None:
                response.success = False
                response.message = "No depth data available"
                return response
            
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - self.depth_timestamp > self.buffer_time:
                response.success = False
                response.message = f"Depth data too old ({current_time - self.depth_timestamp:.2f}s)"
                return response
            
            depth_image = self.latest_depth
        
        try:
            # Handle single point query
            if request.centers_x and len(request.centers_x) == 1:
                result = self._query_single_point(
                    depth_image,
                    request.centers_x[0],
                    request.centers_y[0],
                    request.radii[0] if request.radii else 0
                )
                
                response.success = True
                response.message = "Single point query successful"
                response.depth = result['depth']
                response.min_depth = result['min_depth']
                response.max_depth = result['max_depth']
                response.std_depth = result['std_depth']
                response.confidence = result['confidence']
                
            # Handle multiple points query
            elif request.centers_x:
                results = self._query_multiple_points(
                    depth_image,
                    request.centers_x,
                    request.centers_y,
                    request.radii if request.radii else [0] * len(request.centers_x)
                )
                
                response.success = True
                response.message = f"Queried {len(results)} points"
                response.depths = [r['depth'] for r in results]
                response.confidences = [r['confidence'] for r in results]
                
            else:
                response.success = False
                response.message = "No query points provided"
                
        except Exception as e:
            response.success = False
            response.message = f"Error processing query: {str(e)}"
            self.get_logger().error(f"Error in depth query: {e}")
        
        return response
    
    def _query_single_point(self, depth_image: np.ndarray, cx: int, cy: int, radius: int) -> dict:
        """Query depth at a single point or circular region."""
        height, width = depth_image.shape[:2]
        
        # Ensure coordinates are within image bounds
        cx = max(0, min(cx, width - 1))
        cy = max(0, min(cy, height - 1))
        
        if radius <= 0:
            # Single point query
            depth_value = depth_image[cy, cx]
            if np.isfinite(depth_value) and depth_value > 0:
                depth_m = depth_value / 1000.0  # Convert to meters
                return {
                    'depth': float(depth_m),
                    'min_depth': float(depth_m),
                    'max_depth': float(depth_m),
                    'std_depth': 0.0,
                    'confidence': 1.0
                }
            else:
                return {
                    'depth': 0.0,
                    'min_depth': 0.0,
                    'max_depth': 0.0,
                    'std_depth': 0.0,
                    'confidence': 0.0
                }
        else:
            # Circular region query
            x_start = max(0, cx - radius)
            y_start = max(0, cy - radius)
            x_end = min(width, cx + radius + 1)
            y_end = min(height, cy + radius + 1)
            
            # Extract region
            roi = depth_image[y_start:y_end, x_start:x_end]
            
            # Create circular mask
            y_indices, x_indices = np.ogrid[y_start:y_end, x_start:x_end]
            mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= radius**2
            
            # Apply mask and get valid depths
            masked_depths = roi[mask[y_start:y_end, x_start:x_end]]
            valid_depths = masked_depths[np.isfinite(masked_depths) & (masked_depths > 0)]
            
            if valid_depths.size > 0:
                valid_depths_m = valid_depths / 1000.0  # Convert to meters
                return {
                    'depth': float(np.median(valid_depths_m)),
                    'min_depth': float(np.min(valid_depths_m)),
                    'max_depth': float(np.max(valid_depths_m)),
                    'std_depth': float(np.std(valid_depths_m)),
                    'confidence': float(valid_depths.size / max(1, masked_depths.size))
                }
            else:
                return {
                    'depth': 0.0,
                    'min_depth': 0.0,
                    'max_depth': 0.0,
                    'std_depth': 0.0,
                    'confidence': 0.0
                }
    
    def _query_multiple_points(self, depth_image: np.ndarray, 
                              centers_x: List[int], centers_y: List[int], 
                              radii: List[int]) -> List[dict]:
        """Query depth at multiple points."""
        results = []
        
        for cx, cy, radius in zip(centers_x, centers_y, radii):
            result = self._query_single_point(depth_image, cx, cy, radius)
            results.append(result)
        
        return results


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = DepthServiceProvider()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()