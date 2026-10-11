/* tf_nav_relay.cpp
 *
 * Republishes only the navigation edges of /tf on /tf_nav, for the Nav2 nodes.
 *
 * Why: naoqi_driver publishes Pepper's whole joint tree on /tf -- MEASURED 83
 * frames at 50 Hz, ~4150 transforms/s -- while Nav2 needs ~11/s (the
 * localizer's map edge). tf2 re-checks every pending transform request on
 * EVERY transform it receives, and each costmap sensor cloud waits briefly for
 * its TF, so the joint tree kept the costmap's TF thread saturated until the
 * local costmap stopped updating. MEASURED with this relay: 4 min live, no
 * freeze, 0 dropped clouds.
 *
 * C++ because all this node does is take every /tf message. MEASURED on a
 * synthetic joint tree at naoqi_driver's rate (i7-14700HX): the Python version
 * used 10.1% of a core, this one 0.8%, with identical /tf_nav output.
 *
 * The Nav2 nodes remap /tf -> /tf_nav in the launch files. /tf_static is NOT
 * relayed or remapped: static transforms arrive once and cost nothing. RViz
 * and everything else keep the full /tf, so the robot model still renders.
 *
 * Only remap nodes that LISTEN to TF. A node that also BROADCASTS on /tf
 * (amcl, rtabmap, the localizers) must not be remapped, or its broadcasts
 * would go to /tf_nav only and vanish from the real tree.
 *
 * Parameters:
 *   keep_parents (default: map odom lio_init base_footprint)
 *     A transform is relayed when its PARENT frame is one of these...
 *   drop_children (default: base_link)
 *     ...unless its child is one of these. base_footprint -> base_link is
 *     naoqi_driver's root edge; nothing in Nav2 uses it.
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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

class TfNavRelay : public rclcpp::Node
{
public:
  TfNavRelay()
  : Node("tf_nav_relay")
  {
    const auto keep = declare_parameter<std::vector<std::string>>(
      "keep_parents", {"map", "odom", "lio_init", "base_footprint"});
    const auto drop = declare_parameter<std::vector<std::string>>(
      "drop_children", {"base_link"});
    keep_.insert(keep.begin(), keep.end());
    drop_.insert(drop.begin(), drop.end());

    // Same QoS as tf2_ros's /tf publisher and listener.
    const auto qos = rclcpp::QoS(rclcpp::KeepLast(100)).reliable().durability_volatile();
    pub_ = create_publisher<tf2_msgs::msg::TFMessage>("/tf_nav", qos);
    sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf", qos, std::bind(&TfNavRelay::relay, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "relaying /tf -> /tf_nav for parents [%s], dropping children [%s]",
      join(keep_).c_str(), join(drop_).c_str());
  }

private:
  void relay(const tf2_msgs::msg::TFMessage::ConstSharedPtr msg)
  {
    auto out = std::make_unique<tf2_msgs::msg::TFMessage>();
    for (const auto & t : msg->transforms) {
      if (keep_.count(t.header.frame_id) && !drop_.count(t.child_frame_id)) {
        out->transforms.push_back(t);
      }
    }
    if (!out->transforms.empty()) {
      pub_->publish(std::move(out));
    }
  }

  static std::string join(const std::set<std::string> & names)
  {
    std::string s;
    for (const auto & n : names) {
      s += (s.empty() ? "" : ", ") + n;
    }
    return s;
  }

  // std::set, not unordered_set: a handful of names, and the startup log
  // lists them sorted.
  std::set<std::string> keep_;
  std::set<std::string> drop_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr pub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TfNavRelay>());
  rclcpp::shutdown();
  return 0;
}
