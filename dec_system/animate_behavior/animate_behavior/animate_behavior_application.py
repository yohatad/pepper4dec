#!/usr/bin/env python3

""" animate_behavior_application.py

Entry point for the AnimateBehaviorNode lifecycle node.
Running this node starts a lifecycle action server that drives Pepper's idle gestures,
body rotation, and face LED animations.

This node implements an action server on `/animate_behavior` that, once accepted, animates a
configurable subset of the robot's limbs (arms, hands, head, legs) with smooth random gestures
and periodic body rotation while reporting elapsed-time feedback. Joint commands are computed
from live `/joint_states` feedback and published as smoothed joint angle targets. An
`/animate_behavior/stop` service and goal cancellation allow the animation to be halted early.
While active, a cascading face-LED "breathing" animation is driven via the naoqi LED action
server, unless disabled.

All publishers, the action server, and the stop service are created in `on_configure`, the
joint-state subscription and animation/feedback timers are started in `on_activate`, and
everything is torn down symmetrically in `on_deactivate`/`on_cleanup`.

Subscribers:
    /joint_states (sensor_msgs/JointState)
        Current joint positions, used as the basis for smoothed gesture targets.

Publishers:
    /joint_angles (naoqi_bridge_msgs/JointAnglesWithSpeed)
        Smoothed target joint angles for the animated limbs.
    /cmd_vel (geometry_msgs/Twist)
        Periodic body rotation command issued while a behavior is active.

Services:
    /animate_behavior/stop (std_srvs/Trigger)
        Immediately stops the current animation and zeroes velocity/LEDs.

Actions:
    /animate_behavior (dec_interfaces/AnimateBehavior)
        Action server that runs a gesture/rotation/LED animation for a requested
        behavior type, range, and duration, reporting elapsed-time feedback.
    /naoqi_driver/run_led (naoqi_bridge_msgs/RunLed)
        Action client used to drive the cascading face-LED animation.

Parameters (loaded from animate_behavior_configuration.yaml):
    verbose_mode (bool, default: true)
    led_enabled (bool, default: true)
    led_white_step (double, default: 0.06)
    led_dark_step (double, default: 0.04)
    led_fade_duration (double, default: 0.10)
    led_white_hold (double, default: 2.0)
    led_dark_pause (double, default: 0.2)
    gesture_update_rate (double, default: 30.0)
    gesture_smoothing_factor (double, default: 0.15)
    gesture_motion_speed (double, default: 0.08)
    gesture_interval_min (double, default: 2.5)
    gesture_interval_max (double, default: 4.5)
    gesture_rotation_interval (double, default: 5.0)

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: April 27, 2026
Version: v1.0
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
