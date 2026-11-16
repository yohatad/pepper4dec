/* static_tf_publisher.cpp
 *
 * Publishes the sensor rig's whole static TF chain from a single node.
 *
 * Reads the transforms from a YAML list (config/sensor_tf.yaml, parsed by
 * rig_transforms.cpp) and emits them as one latched /tf_static message,
 * replacing N separate static_transform_publisher processes. Quaternions are
 * normalised on load; non-unit input is reported. A file it cannot use stops
 * the node with an error rather than publishing part of the rig.
 *
 * Parameters:
 *   transforms_file  path to the YAML list (required)
 *   scope            "mount" (default): only the rig's own edges, leaving the
 *                    RealSense's internal extrinsics to its driver.
 *                    "all": every edge, for bags recorded without /tf_static.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: November 17, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>

#include "pepper_slam/rig_transforms.h"

class StaticTfPublisher : public rclcpp::Node
{
public:
  StaticTfPublisher()
  : Node("static_tf_publisher")
  {
    const auto path = declare_parameter<std::string>("transforms_file", "");
    auto scope = declare_parameter<std::string>("scope", "mount");
    std::transform(
      scope.begin(), scope.end(), scope.begin(),
      [](unsigned char c) {return static_cast<char>(std::tolower(c));});

    // Throws RigTransformError; main() reports it and exits non-zero.
    const auto rig = pepper_slam::loadRigTransforms(path, scope);
    for (const auto & warning : rig.warnings) {
      RCLCPP_WARN(get_logger(), "%s", warning.c_str());
    }
    if (rig.skipped_driver_edges > 0) {
      RCLCPP_INFO(
        get_logger(),
        "scope=mount: publishing %zu rig transforms, leaving %zu camera-internal ones to "
        "the RealSense driver. Pass scope:=all when replaying a bag with no /tf_static.",
        rig.transforms.size(), rig.skipped_driver_edges);
    }

    const auto now = get_clock()->now();
    std::vector<geometry_msgs::msg::TransformStamped> msgs;
    std::string listing;
    for (const auto & t : rig.transforms) {
      geometry_msgs::msg::TransformStamped m;
      m.header.stamp = now;
      m.header.frame_id = t.parent;
      m.child_frame_id = t.child;
      m.transform.translation.x = t.xyz[0];
      m.transform.translation.y = t.xyz[1];
      m.transform.translation.z = t.xyz[2];
      m.transform.rotation.x = t.qxyzw[0];
      m.transform.rotation.y = t.qxyzw[1];
      m.transform.rotation.z = t.qxyzw[2];
      m.transform.rotation.w = t.qxyzw[3];
      msgs.push_back(m);
      listing += (listing.empty() ? "" : ", ") + t.parent + "->" + t.child;
    }

    // One latched message with the whole chain.
    broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    broadcaster_->sendTransform(msgs);
    RCLCPP_INFO(
      get_logger(), "Published %zu static transforms from '%s' in a single /tf_static "
      "message: %s", msgs.size(), path.c_str(), listing.c_str());
  }

private:
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> broadcaster_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  std::shared_ptr<StaticTfPublisher> node;
  try {
    node = std::make_shared<StaticTfPublisher>();
  } catch (const pepper_slam::RigTransformError & ex) {
    RCLCPP_FATAL(rclcpp::get_logger("static_tf_publisher"), "%s", ex.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
