#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
from dec_interfaces.srv import GetDepthROI

class DepthROIService(Node):
    def __init__(self):
        super().__init__('depth_roi_service')
        
        self.bridge = CvBridge()
        self.latest_depth = None
        self.depth_camera_info = None
        self.color_camera_info = None
        
        # Subscribe to depth
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        # Subscribe to camera info for intrinsics
        self.depth_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.depth_info_callback,
            10
        )
        
        self.color_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.color_info_callback,
            10
        )
        
        # Create service
        self.srv = self.create_service(
            GetDepthROI,
            'get_depth_roi',
            self.handle_depth_roi_request
        )
        
        self.get_logger().info('Depth ROI service started (supports single ROI, multiple points, multiple ROIs)')
    
    def depth_callback(self, msg):
        self.latest_depth = msg
    
    def depth_info_callback(self, msg):
        self.depth_camera_info = msg
    
    def color_info_callback(self, msg):
        self.color_camera_info = msg
    
    def project_color_to_depth(self, color_x, color_y):
        """Project color coordinates to depth coordinates."""
        if self.color_camera_info is None or self.depth_camera_info is None:
            return color_x, color_y
        
        color_width_img = self.color_camera_info.width
        color_height_img = self.color_camera_info.height
        depth_width_img = self.depth_camera_info.width
        depth_height_img = self.depth_camera_info.height
        
        scale_x = depth_width_img / color_width_img
        scale_y = depth_height_img / color_height_img
        
        depth_x = int(color_x * scale_x)
        depth_y = int(color_y * scale_y)
        
        return depth_x, depth_y
    
    def get_depth_at_point(self, depth_image, x, y, window_size=3):
        """
        Get depth at a point with small window average to handle noise.
        Uses median of surrounding pixels.
        """
        h, w = depth_image.shape
        
        # Clamp coordinates
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Extract small window around point
        half_window = window_size // 2
        y_min = max(0, y - half_window)
        y_max = min(h, y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(w, x + half_window + 1)
        
        window = depth_image[y_min:y_max, x_min:x_max]
        
        # Convert to meters and filter invalid values
        window_m = window.astype(np.float32) / 1000.0
        valid_depths = window_m[(window_m > 0.1) & (window_m < 10.0)]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Return median for robustness
        return float(np.median(valid_depths))
    
    def get_roi_stats(self, depth_image, x, y, width, height):
        """Get depth statistics for an ROI."""
        # Extract ROI
        depth_roi = depth_image[y:y+height, x:x+width]
        
        # Convert to meters
        depth_roi_m = depth_roi.astype(np.float32) / 1000.0
        
        # Filter invalid values
        valid_depths = depth_roi_m[(depth_roi_m > 0.1) & (depth_roi_m < 10.0)]
        
        if len(valid_depths) == 0:
            return None, None, None
        
        return (
            float(np.min(valid_depths)),
            float(np.max(valid_depths)),
            float(np.mean(valid_depths))
        )
    
    def handle_depth_roi_request(self, request, response):
        try:
            # Check if we have depth data
            if self.latest_depth is None:
                response.success = False
                response.message = "No depth data available"
                return response
            
            # Convert depth image to numpy array
            depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
            img_height, img_width = depth_image.shape
            
            # Determine request mode
            has_single_roi = request.width > 0 and request.height > 0
            has_points = len(request.points_x) > 0 and len(request.points_y) > 0
            has_multiple_rois = len(request.rois_x) > 0
            
            # Mode 1: Single ROI
            if has_single_roi and not has_points and not has_multiple_rois:
                depth_x, depth_y = self.project_color_to_depth(request.x, request.y)
                depth_w = int(request.width * (depth_image.shape[1] / self.color_camera_info.width if self.color_camera_info else 1))
                depth_h = int(request.height * (depth_image.shape[0] / self.color_camera_info.height if self.color_camera_info else 1))
                
                # Validate bounds
                if depth_x < 0 or depth_y < 0 or depth_x + depth_w > img_width or depth_y + depth_h > img_height:
                    response.success = False
                    response.message = f"ROI out of bounds"
                    return response
                
                min_d, max_d, mean_d = self.get_roi_stats(depth_image, depth_x, depth_y, depth_w, depth_h)
                
                if min_d is None:
                    response.success = False
                    response.message = "No valid depth values in ROI"
                    return response
                
                # Extract full depth array
                depth_roi = depth_image[depth_y:depth_y+depth_h, depth_x:depth_x+depth_w]
                depth_roi_m = depth_roi.astype(np.float32) / 1000.0
                
                response.success = True
                response.message = "Single ROI processed"
                response.depth_values = depth_roi_m.flatten().tolist()
                response.roi_width = depth_w
                response.roi_height = depth_h
                response.min_depth = min_d
                response.max_depth = max_d
                response.mean_depth = mean_d
                
                self.get_logger().info(f'Single ROI: mean={mean_d:.3f}m')
            
            # Mode 2: Multiple points
            elif has_points:
                if len(request.points_x) != len(request.points_y):
                    response.success = False
                    response.message = "points_x and points_y must have same length"
                    return response
                
                point_depths = []
                for px, py in zip(request.points_x, request.points_y):
                    depth_x, depth_y = self.project_color_to_depth(px, py)
                    
                    if 0 <= depth_x < img_width and 0 <= depth_y < img_height:
                        depth = self.get_depth_at_point(depth_image, depth_x, depth_y)
                        point_depths.append(depth)
                    else:
                        point_depths.append(0.0)
                
                response.success = True
                response.message = f"Processed {len(point_depths)} points"
                response.point_depths = point_depths
                
                self.get_logger().info(f'Multiple points: {len(point_depths)} depths computed')
            
            # Mode 3: Multiple ROIs
            elif has_multiple_rois:
                if not (len(request.rois_x) == len(request.rois_y) == len(request.rois_width) == len(request.rois_height)):
                    response.success = False
                    response.message = "All ROI arrays must have same length"
                    return response
                
                roi_mean_depths = []
                roi_min_depths = []
                roi_max_depths = []
                
                for rx, ry, rw, rh in zip(request.rois_x, request.rois_y, request.rois_width, request.rois_height):
                    depth_x, depth_y = self.project_color_to_depth(rx, ry)
                    depth_w = int(rw * (depth_image.shape[1] / self.color_camera_info.width if self.color_camera_info else 1))
                    depth_h = int(rh * (depth_image.shape[0] / self.color_camera_info.height if self.color_camera_info else 1))
                    
                    # Check bounds
                    if depth_x < 0 or depth_y < 0 or depth_x + depth_w > img_width or depth_y + depth_h > img_height:
                        roi_mean_depths.append(0.0)
                        roi_min_depths.append(0.0)
                        roi_max_depths.append(0.0)
                        continue
                    
                    min_d, max_d, mean_d = self.get_roi_stats(depth_image, depth_x, depth_y, depth_w, depth_h)
                    
                    roi_mean_depths.append(mean_d if mean_d is not None else 0.0)
                    roi_min_depths.append(min_d if min_d is not None else 0.0)
                    roi_max_depths.append(max_d if max_d is not None else 0.0)
                
                response.success = True
                response.message = f"Processed {len(roi_mean_depths)} ROIs"
                response.roi_mean_depths = roi_mean_depths
                response.roi_min_depths = roi_min_depths
                response.roi_max_depths = roi_max_depths
                
                self.get_logger().info(f'Multiple ROIs: {len(roi_mean_depths)} regions processed')
            
            else:
                response.success = False
                response.message = "Invalid request: must specify single ROI, points, or multiple ROIs"
            
            return response
            
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            self.get_logger().error(response.message)
            return response

def main(args=None):
    rclpy.init(args=args)
    node = DepthROIService()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down depth ROI service')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()