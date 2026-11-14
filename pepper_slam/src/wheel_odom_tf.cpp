/* wheel_odom_tf.cpp
 *
 * Publishes odom -> base_footprint from Pepper's wheel odometry, for the loc
 * stacks.
 *
 * The localization stacks (fastloc, pointloc) can broadcast REP-105
 * map -> odom instead of map -> base_footprint (publish.odom_frame, see
 * map_odom.hpp in fast_lio / point_lio). That needs a continuous
 * odom -> base_footprint, and in those stacks nothing else provides one:
 * there is no lio_odom_bridge, and the localizer's own filter IS the map pose.
 *
 * naoqi_driver2 already publishes /pepper_odom, but its own TF broadcast is off
 * on purpose (publish_wheel_odom_tf) and names the parent pepper_odom. This
 * node relays the topic as TF under the frame names given, stamped with the
 * message's own stamp so the localizer's lookup at the scan stamp
 * interpolates correctly.
 *
 * Do NOT run this in the mapping or AMCL stacks: lio_odom_bridge already owns
 * odom -> lio_init -> base_footprint there, and a second parent splits the
 * tree.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: October 11, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include <functional>
#include <memory>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

class WheelOdomTf : public rclcpp::Node
{
public:
  WheelOdomTf()
  : Node("wheel_odom_tf")
  {
    const auto topic = declare_parameter<std::string>("wheel_odom_topic", "/pepper_odom");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_footprint");

    broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    sub_ = create_subscription<nav_msgs::msg::Odometry>(
      topic, 50, std::bind(&WheelOdomTf::relay, this, std::placeholders::_1));
    RCLCPP_INFO(
      get_logger(), "%s -> TF %s -> %s", topic.c_str(), odom_frame_.c_str(),
      base_frame_.c_str());
  }

private:
  void relay(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    // The message's frame names (pepper_odom -> base_footprint) are
    // replaced, not trusted: the pose is what matters, and the odom frame
    // has to be the one the localizer composes against.
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = msg->header.stamp;
    t.header.frame_id = odom_frame_;
    t.child_frame_id = base_frame_;
    t.transform.translation.x = msg->pose.pose.position.x;
    t.transform.translation.y = msg->pose.pose.position.y;
    t.transform.translation.z = msg->pose.pose.position.z;
    // A degenerate quaternion would poison the whole TF tree, not just this
    // edge, so drop the sample instead of broadcasting it.
    const auto & q = msg->pose.pose.orientation;
    if (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w < 1e-12) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "%s sent a degenerate orientation; skipping this %s -> %s sample.",
        sub_->get_topic_name(), odom_frame_.c_str(), base_frame_.c_str());
      return;
    }
    t.transform.rotation = q;
    broadcaster_->sendTransform(t);
  }

  std::string odom_frame_;
  std::string base_frame_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WheelOdomTf>());
  rclcpp::shutdown();
  return 0;
}
