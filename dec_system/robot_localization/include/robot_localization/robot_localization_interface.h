/* robot_localization_interface.h
 *
 * Lifecycle node that converts relative odometry readings into an absolute
 * robot pose using a fixed initial position as the anchor.
 *
 * On configure, the node reads the initial pose and topic/rate parameters
 * and creates a managed pose publisher. On activate, it subscribes to
 * odometry and starts a timer that periodically publishes the absolute
 * pose. The first odometry reading received after activation is captured
 * as the odom-frame anchor; subsequent readings are transformed
 * (translated and rotated) into the global frame relative to that anchor
 * and the configured initial pose.
 *
 * Subscribers:
 *   <odom_topic> (nav_msgs/Odometry)
 *     Relative odometry readings used to compute displacement and rotation
 *     since the odom-frame anchor.
 *
 * Publishers:
 *   <pose_topic> (geometry_msgs/Pose2D)
 *     Absolute robot pose (x, y, theta) in the global frame, published at
 *     a fixed rate.
 *
 * Parameters:
 *   initial_x (double, default: 0.0)       Initial global x position.
 *   initial_y (double, default: 0.0)       Initial global y position.
 *   initial_theta (double, default: 0.0)   Initial global heading, radians.
 *   odom_topic (string, default: "/pepper_odom")
 *   pose_topic (string, default: "/robot_localization/pose")
 *   publish_rate (double, default: 10.0)   Pose publish rate, in Hz.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: Jul 05, 2026
 * Version: v1.0 - C++ port of robot_localization.py
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#ifndef ROBOT_LOCALIZATION_INTERFACE_H
#define ROBOT_LOCALIZATION_INTERFACE_H

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>

#include <string>

//=============================================================================
// RobotLocalization
//
// Lifecycle state machine:
//   UNCONFIGURED → on_configure:  read parameters, init pose/anchor state,
//                                 create the managed pose publisher
//   INACTIVE     → on_activate:   subscribe to odometry, start publish timer
//   ACTIVE       → on_deactivate: cancel timer, destroy odometry subscription
//   INACTIVE     → on_cleanup:    destroy publisher, reset odometry anchor
//=============================================================================

class RobotLocalization : public rclcpp_lifecycle::LifecycleNode {
public:
    using CallbackReturn =
        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

    RobotLocalization();

    // ── Lifecycle callbacks ─────────────────────────────────────────────────
    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_shutdown  (const rclcpp_lifecycle::State& state) override;

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void publishPose();
    static double normalizeAngle(double angle);

    std::string odom_topic_;
    std::string pose_topic_;
    double publish_rate_ = 10.0;

    /// Absolute pose, updated by odomCallback.
    double absolute_x_ = 0.0;
    double absolute_y_ = 0.0;
    double absolute_theta_ = 0.0;

    /// Anchor for the odom-frame -> global-frame transform.
    double initial_x_ = 0.0;
    double initial_y_ = 0.0;
    double initial_theta_ = 0.0;

    /// First odometry reading, captured once on startup.
    double first_odom_x_ = 0.0;
    double first_odom_y_ = 0.0;
    double first_odom_theta_ = 0.0;
    bool odom_initialized_ = false;

    /// Managed publisher — deactivated silently while node is INACTIVE.
    rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::Pose2D>::SharedPtr pose_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::TimerBase::SharedPtr pub_timer_;
};

#endif  // ROBOT_LOCALIZATION_INTERFACE_H
