#!/usr/bin/env python3

"""
gesture_execution_application.py
ROS2 Node for Gesture Execution.

Implements gesture execution logic.
Configuration is loaded from gesture_execution_configuration.yaml.

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohanneh@andrew.cmu.edu
Date: April 18, 2025
Version: v1.0
"""

import rclpy
from .animate_behavior_implementation import AnimateBehaviorSimple


def main(args=None):
    rclpy.init(args=args)
    
    try:
        action_server = AnimateBehaviorSimple()
        
        try:
            rclpy.spin(action_server)
        except KeyboardInterrupt:
            action_server.get_logger().info('Keyboard interrupt')
        finally:
            action_server._stop_execution()
            action_server.destroy_node()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()