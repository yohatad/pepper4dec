#!/usr/bin/env python3

"""
animate_behavior_application.py
ROS2 Node entry point for the Animate Behavior lifecycle action server.

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from .animate_behavior_implementation import AnimateBehaviorNode


def main(args=None):
    rclpy.init(args=args)

    node = AnimateBehaviorNode()

    # MultiThreadedExecutor lets the action server, timers, and
    # lifecycle state-machine callbacks run concurrently.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
