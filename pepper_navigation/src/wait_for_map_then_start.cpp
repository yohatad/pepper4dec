/* wait_for_map_then_start.cpp
 *
 * Starts Nav2's lifecycle manager once the map frame actually exists.
 *
 * Why: Nav2 configures its costmaps at startup, and local_costmap blocks
 * waiting for a transform to its global_frame. With fastlio_localization that
 * transform appears only once ScanContext locks, and the lock waits on the
 * world rather than the clock -- enough scan overlap with the prior map,
 * init_agree_count estimates agreeing, then the candidate holding up over
 * lock_verify_scans live scans. The wait is unbounded even standing still, so
 * under autostart the bringup stalls and planner/controller/bt_navigator sit
 * inactive forever. No fixed delay is correct when the lock waits on the
 * robot. Instead the lifecycle manager is launched with autostart:=false and
 * this node calls its startup service the moment the transform appears.
 *
 * Note: a STATIC map -> lio_init anchor is published at startup so the map
 * frame exists immediately (RViz needs a Fixed Frame to draw the prior map
 * while searching). It deliberately stops there -- nothing publishes
 * lio_init -> base_footprint -- so the lookup below still waits for a real
 * lock. Do not "simplify" it to a frame-exists check.
 *
 * Parameters (config/pepper_navigation_configuration.yaml):
 *   target_frame / source_frame  the transform to wait for (map <- base_footprint)
 *   manager                      the lifecycle manager's manage_nodes service
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

#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <chrono>
#include <memory>
#include <string>

#include <nav2_msgs/srv/manage_lifecycle_nodes.hpp>
#include <rclcpp/rclcpp.hpp>

using ManageLifecycleNodes = nav2_msgs::srv::ManageLifecycleNodes;

class WaitForMapThenStart : public rclcpp::Node
{
public:
  WaitForMapThenStart()
  : Node("wait_for_map_then_start")
  {
    target_ = declare_parameter<std::string>("target_frame", "map");
    source_ = declare_parameter<std::string>("source_frame", "base_footprint");
    const auto manager = declare_parameter<std::string>(
      "manager", "/lifecycle_manager_navigation/manage_nodes");

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    client_ = create_client<ManageLifecycleNodes>(manager);
    timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(1.0), [this] {tick();});

    RCLCPP_INFO(
      get_logger(),
      "waiting for %s -> %s before starting nav2; with fastlio_localization this "
      "appears once ScanContext locks, which requires the robot to move",
      target_.c_str(), source_.c_str());
  }

private:
  void tick()
  {
    ++ticks_;
    // With an error string, tf2 reports a missing frame there instead of
    // printing a warning on every tick of what is an expected wait.
    std::string error;
    if (!tf_buffer_->canTransform(target_, source_, tf2::TimePointZero, &error)) {
      if (ticks_ % 15 == 0) {
        RCLCPP_INFO(
          get_logger(), "still waiting for %s (%ds) -- move the robot if it has not localized yet",
          target_.c_str(), ticks_);
      }
      return;
    }
    RCLCPP_INFO(get_logger(), "%s is up after %ds; starting nav2", target_.c_str(), ticks_);
    if (!client_->wait_for_service(std::chrono::seconds(10))) {
      RCLCPP_ERROR(get_logger(), "lifecycle manager service never appeared");
      return;  // retried on the next tick
    }
    auto request = std::make_shared<ManageLifecycleNodes::Request>();
    request->command = ManageLifecycleNodes::Request::STARTUP;
    client_->async_send_request(
      request, [this](rclcpp::Client<ManageLifecycleNodes>::SharedFuture future) {
        RCLCPP_INFO(
          get_logger(), "nav2 startup: %s", future.get()->success ? "OK" : "REPORTED FAILURE");
      });
    // Done with TF. Left subscribed, the listener would keep deserializing
    // naoqi_driver's joint tree on /tf (~4150 transforms/s) for the whole run.
    tf_listener_.reset();
    timer_->cancel();
  }

  std::string target_;
  std::string source_;
  int ticks_ = 0;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Client<ManageLifecycleNodes>::SharedPtr client_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WaitForMapThenStart>());
  rclcpp::shutdown();
  return 0;
}
