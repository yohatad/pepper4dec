/* overt_attention_application.cpp
 *
 * Entry point for the OvertAttentionNode lifecycle node (Pepper's overt
 * attention controller). Constructs the node and spins it; the node class
 * itself is implemented in overt_attention_implementation.cpp.
 *
 * Subscribers:
 *   /face_detection/data (dec_interfaces/msg/FaceDetection)
 *     Detected/engaged faces competing for attention (topic from
 *     data/pepper_topics.yaml).
 *   /overt_attention/saliency_peak (std_msgs/msg/Float32MultiArray)
 *     Bottom-up saliency peaks published by SaliencyNode.
 *   <camera_info topic> (sensor_msgs/msg/CameraInfo)
 *     Camera intrinsics used to convert pixels to angles (topic resolved
 *     from data/pepper_topics.yaml for the configured camera_type).
 *   /joint_states (sensor_msgs/msg/JointState)
 *     Current head joint positions.
 *
 * Publishers:
 *   /joint_angles (naoqi_bridge_msgs/msg/JointAnglesWithSpeed)
 *     Head yaw/pitch commands moving gaze to the selected target.
 *   /overt_attention/target_angles (geometry_msgs/msg/Vector3)
 *     Current attention target as yaw/pitch angles, for downstream nodes
 *     and visualization.
 *
 * Services:
 *   /overt_attention/set_enabled (std_srvs/srv/SetBool)
 *     Server that enables or disables attention control (used to hand the
 *     head over to other behaviors).
 *
 * Parameters (config/overt_attention_configuration.yaml, under
 * overt_attention/ros__parameters, plus the shared wildcard block):
 *   camera_type (string, default: "pepper")
 *   start_enabled (bool, default: true)
 *   move_to_default_on_disable (bool, default: true)
 *   default_yaw (double, default: 0.0)
 *   default_pitch (double, default: -0.2)
 *   default_move_speed (double, default: 0.1)
 *   saliency_yaw_lim (double, default: 1.8)
 *   saliency_pitch_up (double, default: 0.4)
 *   saliency_pitch_dn (double, default: -0.7)
 *   face_timeout (double, default: 2.0)
 *   engaged_priority_bonus (double, default: 2.0)
 *   face_switch_cooldown (double, default: 1.0)
 *   prefer_closer_faces (bool, default: true)
 *   max_face_distance (double, default: 5.0)
 *   min_angular_change_deg (double, default: 2.0)
 *   target_smoothing_alpha (double, default: 0.4)
 *   saliency_min_score (double, default: 0.30)
 *   saliency_min_cooldown (double, default: 1.5)
 *   saliency_max_dwell (double, default: 3.0)
 *   switch_score_ratio (double, default: 1.4)
 *   same_target_threshold_deg (double, default: 5.0)
 *   enable_ior (bool, default: true)
 *   ior_max_suppression (double, default: 0.9)
 *   ior_half_life (double, default: 3.0)
 *   ior_radius_deg (double, default: 15.0)
 *
 * Lifecycle:
 *   configure  -> read parameters and topic names, create the head/target
 *                 publishers, the face/saliency/camera-info/joint-state
 *                 subscriptions, and the set_enabled service
 *   activate   -> activate the head and target publishers
 *   deactivate -> deactivate the head and target publishers
 *   cleanup    -> destroy the subscriptions, publishers, and service
 *   shutdown   -> inherited default (no node-specific teardown)
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: June 12, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include "overt_attention/overt_attention_interface.h"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<OvertAttentionNode>();
        rclcpp::spin(node->get_node_base_interface());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("overt_attention"), "Exception: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
