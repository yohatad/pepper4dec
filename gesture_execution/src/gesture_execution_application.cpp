/* gesture_execution_application.cpp
 *
 * Entry point for the GestureExecutionNode lifecycle node. Spins the
 * node single-threaded; the class itself is implemented in
 * gesture_execution_implementation.cpp.
 *
 * Subscribers:
 *   /joint_states (sensor_msgs/msg/JointState)
 *     Current joint positions, used to track the robot's arm/head/leg state.
 *   /localization/pose (nav_msgs/msg/Odometry)
 *     Absolute map->base_footprint robot pose from fast_lio's
 *     fastlio_localization, used to compute pointing direction. The topic
 *     name comes from the RobotPose key in data/pepper_topics.yaml.
 *
 * Publishers:
 *   /joint_angles_trajectory (naoqi_bridge_msgs/msg/JointAnglesTrajectory)
 *     Joint angle trajectories sent to the robot to perform gestures.
 *   /gesture_execution/visualization (visualization_msgs/msg/Marker)
 *     Markers visualizing deictic gesture targets, shoulder, and pointing
 *     arrow.
 *
 * Actions:
 *   /gesture_execution (dec_interfaces/action/Gesture)
 *     Server that executes a named or typed gesture (deictic, iconic, bow,
 *     nod) with feedback on elapsed time and a success/failure result.
 *
 * Parameters (config/gesture_execution_configuration.yaml, under
 * gesture_action_server/ros__parameters):
 *   verbose_mode (bool, default: false)
 *   Gesture and topic data always load from fixed paths (data/gesture.yaml,
 *   data/pepper_topics.yaml) — not configurable via parameters.
 *
 * Lifecycle:
 *   configure  -> read parameters, load the gesture and topic YAML data, and
 *                 create the trajectory/marker publishers and action server
 *   activate   -> activate the publishers and subscribe to /joint_states and
 *                 /localization/pose
 *   deactivate -> destroy the joint-state and pose subscriptions and
 *                 deactivate the publishers
 *   cleanup    -> destroy the lifecycle publishers and the action server
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

#include "gesture_execution/gesture_execution_interface.h"

#include "dec_common/node_runner.h"

int main(int argc, char** argv) {
    return dec_common::runNode<GestureExecutionNode>(
        argc, argv,
        {"gesture_execution v1.0 — This program comes with ABSOLUTELY NO WARRANTY.", "gesture_execution"});
}
