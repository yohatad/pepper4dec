/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: July 05, 2026
Version: v1.0 - C++ port of robot_localization.py
*/

#include "robot_localization/robot_localization_interface.h"

#include <cmath>

RobotLocalization::RobotLocalization() : rclcpp_lifecycle::LifecycleNode("robot_localization") {
    // Declare parameters here so they are settable via launch/CLI before configure
    declare_parameter("initial_x", 0.0);
    declare_parameter("initial_y", 0.0);
    declare_parameter("initial_theta", 0.0);
    declare_parameter("odom_topic", std::string("/pepper_odom"));
    declare_parameter("pose_topic", std::string("/robot_localization/pose"));
    declare_parameter("publish_rate", 10.0);
}

RobotLocalization::CallbackReturn RobotLocalization::on_configure(const rclcpp_lifecycle::State&) {
    double initial_x = get_parameter("initial_x").as_double();
    double initial_y = get_parameter("initial_y").as_double();
    double initial_theta = get_parameter("initial_theta").as_double();
    odom_topic_ = get_parameter("odom_topic").as_string();
    pose_topic_ = get_parameter("pose_topic").as_string();
    publish_rate_ = get_parameter("publish_rate").as_double();

    // Absolute pose (updated by odomCallback)
    absolute_x_ = initial_x;
    absolute_y_ = initial_y;
    absolute_theta_ = initial_theta;

    // Anchor for the odom-frame -> global-frame transform
    initial_x_ = initial_x;
    initial_y_ = initial_y;
    initial_theta_ = initial_theta;

    odom_initialized_ = false;

    // Managed publisher — deactivated silently while node is INACTIVE
    pose_pub_ = create_publisher<geometry_msgs::msg::Pose2D>(pose_topic_, 10);

    RCLCPP_INFO(get_logger(),
        "RobotLocalization configured — initial=(%.3f, %.3f, %.1f deg) | odom=%s | pose=%s @ %.1f Hz",
        initial_x, initial_y, initial_theta * 180.0 / M_PI,
        odom_topic_.c_str(), pose_topic_.c_str(), publish_rate_);
    return CallbackReturn::SUCCESS;
}

RobotLocalization::CallbackReturn RobotLocalization::on_activate(const rclcpp_lifecycle::State& state) {
    LifecycleNode::on_activate(state);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 10, std::bind(&RobotLocalization::odomCallback, this, std::placeholders::_1));
    pub_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / publish_rate_)),
        std::bind(&RobotLocalization::publishPose, this));

    RCLCPP_INFO(get_logger(), "RobotLocalization activated");
    return CallbackReturn::SUCCESS;
}

RobotLocalization::CallbackReturn RobotLocalization::on_deactivate(const rclcpp_lifecycle::State& state) {
    pub_timer_->cancel();
    pub_timer_.reset();
    odom_sub_.reset();

    LifecycleNode::on_deactivate(state);
    RCLCPP_INFO(get_logger(), "RobotLocalization deactivated");
    return CallbackReturn::SUCCESS;
}

RobotLocalization::CallbackReturn RobotLocalization::on_cleanup(const rclcpp_lifecycle::State&) {
    pose_pub_.reset();
    odom_initialized_ = false;
    RCLCPP_INFO(get_logger(), "RobotLocalization cleaned up");
    return CallbackReturn::SUCCESS;
}

RobotLocalization::CallbackReturn RobotLocalization::on_shutdown(const rclcpp_lifecycle::State&) {
    RCLCPP_INFO(get_logger(), "RobotLocalization shutting down");
    return CallbackReturn::SUCCESS;
}

void RobotLocalization::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    double odom_x = msg->pose.pose.position.x;
    double odom_y = msg->pose.pose.position.y;

    // Extract yaw from quaternion (z-axis rotation only for 2D)
    const auto& q = msg->pose.pose.orientation;
    double odom_theta = std::atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z));

    // Capture the first reading to anchor the odom->global transform
    if (!odom_initialized_) {
        first_odom_x_ = odom_x;
        first_odom_y_ = odom_y;
        first_odom_theta_ = odom_theta;
        odom_initialized_ = true;
        RCLCPP_INFO(get_logger(), "Odometry anchor set at (%.3f, %.3f, %.1f deg)",
            odom_x, odom_y, odom_theta * 180.0 / M_PI);
        return;
    }

    // Displacement in the odom frame since the first reading
    double d_odom_x = odom_x - first_odom_x_;
    double d_odom_y = odom_y - first_odom_y_;

    // Fixed rotation: odom frame -> global frame
    double rot = initial_theta_ - first_odom_theta_;

    absolute_x_ = initial_x_ + d_odom_x * std::cos(rot) - d_odom_y * std::sin(rot);
    absolute_y_ = initial_y_ + d_odom_x * std::sin(rot) + d_odom_y * std::cos(rot);
    absolute_theta_ = normalizeAngle(initial_theta_ + (odom_theta - first_odom_theta_));
}

void RobotLocalization::publishPose() {
    geometry_msgs::msg::Pose2D msg;
    msg.x = absolute_x_;
    msg.y = absolute_y_;
    msg.theta = absolute_theta_;
    pose_pub_->publish(msg);
}

double RobotLocalization::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotLocalization>();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
