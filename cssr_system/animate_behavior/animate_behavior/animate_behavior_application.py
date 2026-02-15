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
from rclpy.executors import MultiThreadedExecutor
from .animate_behavior_implementation import AnimateBehaviorServer


def main(args=None):
    rclpy.init(args=args)

    action_server = AnimateBehaviorServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass  # Don't log here - context may be shutting down
    finally:
        # Cleanup before destroying node
        try:
            action_server.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during shutdown

        # Destroy node and shutdown
        try:
            action_server.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()