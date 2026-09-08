/* animate_behavior_application.cpp
 *
 * Entry point for the animate_behavior lifecycle node. Spins the node on a
 * MultiThreadedExecutor so the action server, timers, and lifecycle
 * state-machine callbacks can run concurrently; the AnimateBehaviorNode
 * class itself is implemented in animate_behavior_implementation.cpp.
 *
 * Subscribers:
 *   /joint_states (sensor_msgs/msg/JointState)
 *     Current joint positions, used as the basis for smoothed gesture targets.
 *
 * Publishers:
 *   /joint_angles (naoqi_bridge_msgs/msg/JointAnglesWithSpeed)
 *     Smoothed target joint angles for the animated limbs.
 *   /cmd_vel (geometry_msgs/msg/Twist)
 *     Periodic body rotation command issued while a behavior is active.
 *
 * Services:
 *   /animate_behavior/stop (std_srvs/srv/Trigger)
 *     Server that immediately stops the current animation and zeroes
 *     velocity and LEDs.
 *
 * Actions:
 *   /animate_behavior (dec_interfaces/action/AnimateBehavior)
 *     Server that runs a gesture/rotation/LED animation for a requested
 *     behavior type, range, and duration, reporting elapsed-time feedback.
 *   /naoqi_driver/run_led (naoqi_bridge_msgs/action/RunLed)
 *     Client used to drive the cascading face-LED animation.
 *
 * Parameters (config/animate_behavior_configuration.yaml, under
 * animate_behavior/ros__parameters):
 *   verbose_mode (bool, default: true)
 *   led_enabled (bool, default: true)
 *   led_white_step (double, default: 0.06)
 *   led_dark_step (double, default: 0.04)
 *   led_fade_duration (double, default: 0.10)
 *   led_white_hold (double, default: 2.0)
 *   led_dark_pause (double, default: 0.2)
 *   gesture_update_rate (double, default: 30.0)
 *   gesture_smoothing_factor (double, default: 0.15)
 *   gesture_motion_speed (double, default: 0.08)
 *   gesture_interval_min (double, default: 2.5)
 *   gesture_interval_max (double, default: 4.5)
 *   gesture_rotation_interval (double, default: 5.0)
 *
 * Lifecycle:
 *   configure  -> read parameters, create the joint/velocity publishers, the
 *                 action server, the stop service, and the LED action client
 *   activate   -> subscribe to /joint_states and start the animation timer
 *   deactivate -> stop the animation and LEDs, cancel the timer, destroy the
 *                 joint-state subscription
 *   cleanup    -> destroy the publishers, action server, service, and LED
 *                 action client
 *   shutdown   -> log that the node is shutting down
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 5, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include "animate_behavior/animate_behavior_interface.h"

#include "dec_common/node_runner.h"

int main(int argc, char** argv) {
    // 4 executor threads: the action server, timers, and lifecycle
    // state-machine callbacks run concurrently.
    return dec_common::runNode<AnimateBehaviorNode>(argc, argv, {nullptr, "animate_behavior", 4});
}
