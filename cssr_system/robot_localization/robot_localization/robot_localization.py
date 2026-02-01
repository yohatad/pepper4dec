#!/usr/bin/env python3
"""
robot_localization.py

Simple robot localization node that converts relative odometry to absolute position.
Uses initial position and integrates odometry data to provide absolute pose.

Author: Assistant
Date: 2025
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D
from transforms3d.euler import quat2euler


class RobotLocalization(Node):
    """
    Converts relative odometry to absolute robot pose
    Integrates odometry data with an initial position to provide global localization
    """
    
    def __init__(self):
        super().__init__('robot_localization')
        
        # Declare parameters with defaults
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_theta', 0.0)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('pose_topic', '/robotLocalization/pose')
        self.declare_parameter('publish_rate', 10.0)  # Hz
        
        # Get parameters
        initial_x = self.get_parameter('initial_x').value
        initial_y = self.get_parameter('initial_y').value
        initial_theta = self.get_parameter('initial_theta').value
        odom_topic = self.get_parameter('odom_topic').value
        pose_topic = self.get_parameter('pose_topic').value
        publish_rate = self.get_parameter('publish_rate').value
        
        # Initialize absolute pose
        self.absolute_x = initial_x
        self.absolute_y = initial_y
        self.absolute_theta = initial_theta
        
        # Store previous odometry for integration
        self.prev_odom_x = None
        self.prev_odom_y = None
        self.prev_odom_theta = None
        self.odom_initialized = False
        
        # Create subscriber to odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10
        )
        
        # Create publisher for absolute pose
        self.pose_pub = self.create_publisher(
            Pose2D,
            pose_topic,
            10
        )
        
        # Create timer for publishing at fixed rate
        self.timer = self.create_timer(
            1.0 / publish_rate,
            self.publish_pose
        )
        
        self.get_logger().info(
            f"Robot Localization started\n"
            f"  Initial position: ({initial_x:.3f}, {initial_y:.3f}, {math.degrees(initial_theta):.1f}°)\n"
            f"  Subscribing to: {odom_topic}\n"
            f"  Publishing to: {pose_topic}\n"
            f"  Publish rate: {publish_rate} Hz"
        )
    
    def odom_callback(self, msg: Odometry):
        """
        Process odometry messages and update absolute position
        Integrates relative odometry changes with the absolute pose
        """
        try:
            # Extract position from odometry
            odom_x = msg.pose.pose.position.x
            odom_y = msg.pose.pose.position.y
            
            # Extract orientation (yaw) from quaternion
            # Extract orientation (yaw) from quaternion
            orientation_q = msg.pose.pose.orientation
            # transforms3d uses [w, x, y, z] order (note the difference!)
            (roll, pitch, odom_theta) = quat2euler([orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z])
            
            # Initialize on first message
            if not self.odom_initialized:
                self.prev_odom_x = odom_x
                self.prev_odom_y = odom_y
                self.prev_odom_theta = odom_theta
                self.odom_initialized = True
                self.get_logger().info(
                    f"Odometry initialized at ({odom_x:.3f}, {odom_y:.3f}, {math.degrees(odom_theta):.1f}°)"
                )
                return
            
            # Calculate odometry deltas (in odometry frame)
            delta_odom_x = odom_x - self.prev_odom_x
            delta_odom_y = odom_y - self.prev_odom_y
            delta_odom_theta = self.normalize_angle(odom_theta - self.prev_odom_theta)
            
            # Transform delta to global frame using current absolute orientation
            delta_global_x = (delta_odom_x * math.cos(self.absolute_theta) - 
                             delta_odom_y * math.sin(self.absolute_theta))
            delta_global_y = (delta_odom_x * math.sin(self.absolute_theta) + 
                             delta_odom_y * math.cos(self.absolute_theta))
            
            # Update absolute position
            self.absolute_x += delta_global_x
            self.absolute_y += delta_global_y
            self.absolute_theta = self.normalize_angle(self.absolute_theta + delta_odom_theta)
            
            # Store current odometry for next iteration
            self.prev_odom_x = odom_x
            self.prev_odom_y = odom_y
            self.prev_odom_theta = odom_theta
            
        except Exception as e:
            self.get_logger().error(f"Error processing odometry: {e}")
    
    def publish_pose(self):
        """Publish the current absolute pose at fixed rate"""
        if not self.odom_initialized:
            return
        
        msg = Pose2D()
        msg.x = self.absolute_x
        msg.y = self.absolute_y
        msg.theta = self.absolute_theta
        
        self.pose_pub.publish(msg)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def reset_position(self, x: float, y: float, theta: float):
        """
        Reset absolute position (can be called from service or topic)
        Useful for manual corrections or known position updates
        """
        self.absolute_x = x
        self.absolute_y = y
        self.absolute_theta = theta
        self.get_logger().info(
            f"Position reset to ({x:.3f}, {y:.3f}, {math.degrees(theta):.1f}°)"
        )

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RobotLocalization()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in robot_localization: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()