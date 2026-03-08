#!/usr/bin/env python3

"""
gesture_execution_application.py
ROS2 Node for Gesture Execution.

Implements gesture execution logic.
Configuration is loaded from gesture_execution_configuration.yaml.

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: April 18, 2025
Version: v1.0
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from .animate_behavior_implementation import AnimateBehaviorServer


def main(args=None):
    rclpy.init(args=args)
    
    node = AnimateBehaviorServer()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()