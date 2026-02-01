#!/usr/bin/env python3
"""
Test script for gesture execution visualization in RViz2

This script tests the deictic gesture visualization by:
1. Starting a ROS2 node that publishes marker messages
2. Simulating a deictic gesture call
3. Publishing visualization markers to RViz2

Run this script and then run rviz2 to see the visualization:
  ros2 run rviz2 rviz2

In RViz2, add a Marker display and set the topic to:
  /gesture_execution/visualization
"""

import rclpy
from rclpy.node import Node
import time
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Time

class VisualizationTestNode(Node):
    """Test node to verify gesture execution visualization in RViz2"""
    
    def __init__(self):
        super().__init__('visualization_test')
        
        # Create publisher for markers
        self.marker_pub = self.create_publisher(Marker, '/gesture_execution/visualization', 10)
        
        self.get_logger().info("Visualization Test Node started")
        self.get_logger().info("Publishing test markers to /gesture_execution/visualization")
        self.get_logger().info("Run 'ros2 run rviz2 rviz2' to visualize")
        
    def publish_test_markers(self):
        """Publish example deictic gesture markers"""
        
        # Simulate a pointing gesture to (1.0, 0.5, 0.3) meters from base_link
        target_x = 1000.0  # mm
        target_y = 500.0   # mm
        target_z = 300.0   # mm
        
        # Simulated shoulder position (right arm)
        shoulder_x = -57.0  # mm
        shoulder_y = -149.74  # mm (right arm is negative)
        shoulder_z = 86.82  # mm
        
        stamp = self.get_clock().now().to_msg()
        
        # 1. Target point marker (sphere) - bright red for better visibility
        target_marker = Marker()
        target_marker.header.stamp = stamp
        target_marker.header.frame_id = "base_link"
        target_marker.ns = "deictic_target"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = target_x / 1000.0  # meters
        target_marker.pose.position.y = target_y / 1000.0
        target_marker.pose.position.z = target_z / 1000.0
        target_marker.pose.orientation.w = 1.0
        target_marker.scale.x = 0.1  # 10cm sphere (increased from 5cm)
        target_marker.scale.y = 0.1
        target_marker.scale.z = 0.1
        target_marker.color.r = 1.0  # Bright red for target
        target_marker.color.g = 0.0
        target_marker.color.b = 0.0
        target_marker.color.a = 1.0  # Fully opaque
        # Set lifetime properly with both sec and nanosec
        target_marker.lifetime.sec = 10  # 10 seconds (increased from 5)
        target_marker.lifetime.nanosec = 0
        
        self.marker_pub.publish(target_marker)
        self.get_logger().info("Published target marker (bright red sphere)")
        
        # 2. Shoulder position marker - bright blue for better visibility
        shoulder_marker = Marker()
        shoulder_marker.header.stamp = stamp
        shoulder_marker.header.frame_id = "base_link"
        shoulder_marker.ns = "deictic_shoulder"
        shoulder_marker.id = 1
        shoulder_marker.type = Marker.SPHERE
        shoulder_marker.action = Marker.ADD
        shoulder_marker.pose.position.x = shoulder_x / 1000.0
        shoulder_marker.pose.position.y = shoulder_y / 1000.0
        shoulder_marker.pose.position.z = shoulder_z / 1000.0
        shoulder_marker.pose.orientation.w = 1.0
        shoulder_marker.scale.x = 0.06  # 6cm sphere (increased from 3cm)
        shoulder_marker.scale.y = 0.06
        shoulder_marker.scale.z = 0.06
        shoulder_marker.color.r = 0.0
        shoulder_marker.color.g = 0.0  # Bright blue for right arm
        shoulder_marker.color.b = 1.0
        shoulder_marker.color.a = 1.0  # Fully opaque
        shoulder_marker.lifetime.sec = 10
        shoulder_marker.lifetime.nanosec = 0
        
        self.marker_pub.publish(shoulder_marker)
        self.get_logger().info("Published shoulder marker (bright blue sphere)")
        
        # 3. Pointing line from shoulder to target - bright blue arrow
        line_marker = Marker()
        line_marker.header.stamp = stamp
        line_marker.header.frame_id = "base_link"
        line_marker.ns = "deictic_line"
        line_marker.id = 2
        line_marker.type = Marker.ARROW
        line_marker.action = Marker.ADD
        
        start_point = Point()
        start_point.x = shoulder_x / 1000.0
        start_point.y = shoulder_y / 1000.0
        start_point.z = shoulder_z / 1000.0
        end_point = Point()
        end_point.x = target_x / 1000.0
        end_point.y = target_y / 1000.0
        end_point.z = target_z / 1000.0
        
        line_marker.points.append(start_point)
        line_marker.points.append(end_point)
        
        line_marker.color.r = 0.0
        line_marker.color.g = 0.0  # Bright blue for right arm
        line_marker.color.b = 1.0
        line_marker.color.a = 0.8  # Slightly transparent for better visibility
        line_marker.scale.x = 0.03  # Shaft diameter (increased from 0.02)
        line_marker.scale.y = 0.06  # Head diameter (increased from 0.04)
        line_marker.scale.z = 0.12  # Head length (increased from 0.1)
        line_marker.lifetime.sec = 10
        line_marker.lifetime.nanosec = 0
        
        self.marker_pub.publish(line_marker)
        self.get_logger().info("Published pointing line (bright blue arrow)")
        
        # 4. Text label showing coordinates - larger and brighter
        text_marker = Marker()
        text_marker.header.stamp = stamp
        text_marker.header.frame_id = "base_link"
        text_marker.ns = "deictic_text"
        text_marker.id = 3
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = target_x / 1000.0
        text_marker.pose.position.y = target_y / 1000.0
        text_marker.pose.position.z = (target_z / 1000.0) + 0.15  # 15cm above target
        text_marker.pose.orientation.w = 1.0
        text_marker.text = f"Target: ({target_x/1000:.2f}, {target_y/1000:.2f}, {target_z/1000:.2f}) m"
        text_marker.scale.z = 0.07  # Text height (increased from 0.05)
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0  # White text for better visibility
        text_marker.color.a = 1.0  # Fully opaque
        text_marker.lifetime.sec = 10
        text_marker.lifetime.nanosec = 0
        
        self.marker_pub.publish(text_marker)
        self.get_logger().info("Published text label with coordinates")
        
        self.get_logger().info("\n===== VISUALIZATION TEST COMPLETE =====")
        self.get_logger().info("To view in RViz2:")
        self.get_logger().info("1. Run: ros2 run rviz2 rviz2")
        self.get_logger().info("2. Add a 'Marker' display")
        self.get_logger().info("3. Set topic to: /gesture_execution/visualization")
        self.get_logger().info("4. Make sure 'Global Options' -> 'Fixed Frame' is set to 'base_link'")
        self.get_logger().info("========================================")


def main(args=None):
    """Main test function"""
    rclpy.init(args=args)
    
    test_node = VisualizationTestNode()
    
    # Wait a moment for publisher to be ready
    time.sleep(1.0)
    
    # Publish test markers
    test_node.publish_test_markers()
    
    # Keep node alive for a bit so markers are published
    time.sleep(2.0)
    
    # Shutdown
    test_node.destroy_node()
    rclpy.shutdown()
    
    print("\nTest complete! Markers should be visible in RViz2 for 10 seconds.")
    print("You can run this test again to refresh the markers.")

if __name__ == '__main__':
    main()