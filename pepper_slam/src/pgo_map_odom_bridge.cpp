/* pgo_map_odom_bridge.cpp
 *
 * Publishes the REP-105 map -> odom loop-closure correction from the pose-graph
 * optimizer (fastlio_lc_pgo).
 *
 * Composes the estimator's odom -> body with the optimizer's map -> body into
 * map -> odom, buffering the estimator's odometry and matching each optimized
 * keyframe to the nearest sample in time. Identity until the first
 * correction, then republished on every odometry tick, stamped with it.
 *
 * Optionally publishes a one-time static level_frame -> map_frame, so that
 * level_frame is gravity-aligned with z = 0 on the floor (octomap's
 * /projected_map hardcodes its grid origin to z = 0).
 *
 * Parameters (config/pepper_slam_configuration.yaml):
 *   map_frame, odom_frame, base_frame, level_frame
 *   odom_topic            the estimator's odometry (odom -> body)
 *   pgo_odom_topic        the optimizer's latest keyframe pose (map -> body)
 *   publish_level_frame   publish level_frame -> map_frame once
 *   level_frame_on_floor  put level_frame's z = 0 on the floor
 *   lidar_imu_frame       the static frame the estimator's body is; MUST match
 *                         its config's publish.body_frame (set by the launch)
 *   level_source          'calibration': the levelling rotation is the static
 *                         base_frame -> lidar_imu_frame mount, identical on
 *                         every run. 'odometry': the legacy runtime snapshot.
 *                         Calibration needs the estimator's world frame to start
 *                         at identity rotation (Point-LIO: gravity_align false).
 *   odom_buffer_size      recent odometry samples kept for matching
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

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/exceptions.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_ros/transform_listener.hpp>

#include "pepper_slam/transform_utils.h"

namespace
{

using pepper_slam::fill;
using pepper_slam::toSec;
using pepper_slam::toTransform;

struct OdomSample
{
  double t;
  tf2::Transform pose;
};

}  // namespace

class PgoMapOdomBridge : public rclcpp::Node
{
public:
  PgoMapOdomBridge()
  : Node("pgo_map_odom_bridge")
  {
    map_frame_ = declare_parameter<std::string>("map_frame", "pgo_init");
    odom_frame_ = declare_parameter<std::string>("odom_frame", "lio_init");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_footprint");
    level_frame_ = declare_parameter<std::string>("level_frame", "map");
    const auto odom_topic = declare_parameter<std::string>("odom_topic", "/odom_lio");
    const auto pgo_odom_topic = declare_parameter<std::string>("pgo_odom_topic", "/aft_pgo_odom");
    publish_level_ = declare_parameter<bool>("publish_level_frame", true);
    level_on_floor_ = declare_parameter<bool>("level_frame_on_floor", true);
    lidar_imu_frame_ = declare_parameter<std::string>("lidar_imu_frame", "l2lidar_frame_imu");
    level_source_ = declare_parameter<std::string>("level_source", "calibration");
    if (level_source_ != "calibration" && level_source_ != "odometry") {
      RCLCPP_ERROR(
        get_logger(), "level_source '%s' invalid; using 'calibration'.",
        level_source_.c_str());
      level_source_ = "calibration";
    }
    buffer_size_ = static_cast<std::size_t>(
      std::max<int64_t>(1, declare_parameter<int64_t>("odom_buffer_size", 1000)));

    // TF is read only for the one-time level frame, so the listener exists
    // only until that is published (see tryPublishLevel).
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    if (publish_level_) {
      tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    }
    broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    map_odom_.setIdentity();

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, 50, std::bind(&PgoMapOdomBridge::onOdom, this, std::placeholders::_1));
    sub_pgo_ = create_subscription<nav_msgs::msg::Odometry>(
      pgo_odom_topic, 50, std::bind(&PgoMapOdomBridge::onPgoOdom, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "pgo_map_odom_bridge: correction %s -> %s from %s vs %s; leveling %s -> %s %s.",
      map_frame_.c_str(), odom_frame_.c_str(), pgo_odom_topic.c_str(), odom_topic.c_str(),
      level_frame_.c_str(), map_frame_.c_str(), publish_level_ ? "ON" : "OFF");
  }

private:
  void onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    const double t = toSec(msg->header.stamp);
    // Insert keeping the buffer time-sorted (odometry usually arrives in order).
    const auto at = std::lower_bound(
      odom_.begin(), odom_.end(), t,
      [](const OdomSample & s, double value) {return s.t < value;});
    odom_.insert(
      at,
      OdomSample{t, toTransform(msg->pose.pose.position, msg->pose.pose.orientation)});
    if (odom_.size() > buffer_size_) {
      odom_.pop_front();
    }

    // Republish the latest correction, stamped with this odometry's time.
    geometry_msgs::msg::TransformStamped out;
    out.header.stamp = msg->header.stamp;
    out.header.frame_id = map_frame_;
    out.child_frame_id = odom_frame_;
    fill(out, map_odom_);
    broadcaster_->sendTransform(out);

    if (publish_level_ && !level_published_) {
      tryPublishLevel();
    }
  }

  // The buffered odometry pose nearest in time to t, or nullptr if empty.
  const tf2::Transform * nearestOdom(double t) const
  {
    if (odom_.empty()) {
      return nullptr;
    }
    const auto at = std::lower_bound(
      odom_.begin(), odom_.end(), t,
      [](const OdomSample & s, double value) {return s.t < value;});
    // On a tie the later sample wins, as in the Python bridge this replaced.
    auto best = at;
    if (at == odom_.end() ||
      (at != odom_.begin() && std::abs(std::prev(at)->t - t) < std::abs(at->t - t)))
    {
      best = std::prev(at);
    }
    return &best->pose;
  }

  void onPgoOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    // Reject degenerate poses instead of latching a bogus correction. The
    // optimizer once published a default-constructed Odometry (quaternion
    // 0,0,0,0, stamp 0) before any keyframe was optimized; that is fixed
    // upstream, but this node stays safe against any publisher that is not.
    const auto & q = msg->pose.pose.orientation;
    const double qn = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (!std::isfinite(qn) || std::abs(qn - 1.0) > 1e-3) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Ignoring /aft_pgo_odom with non-unit quaternion (norm %.4f); this is what an "
        "unoptimized or default-constructed pose looks like.", qn);
      return;
    }
    const double t = toSec(msg->header.stamp);
    if (t <= 0.0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "Ignoring /aft_pgo_odom with non-positive stamp.");
      return;
    }
    const tf2::Transform * odom_body = nearestOdom(t);
    if (odom_body == nullptr) {
      return;  // no odometry buffered yet
    }
    // map -> odom = (map -> body) * (odom -> body)^-1
    map_odom_ =
      toTransform(msg->pose.pose.position, msg->pose.pose.orientation) * odom_body->inverse();
  }

  // One-time static level_frame -> map_frame, so level_frame is z-up at t = 0.
  void tryPublishLevel()
  {
    tf2::Transform level;
    if (level_source_ == "calibration") {
      // The estimator's world frame starts at identity rotation, so map's axes
      // ARE the lidar IMU's axes: the levelling is the static mount.
      try {
        level = pepper_slam::levelFromCalibration(
          toTransform(
            tf_buffer_->lookupTransform(base_frame_, lidar_imu_frame_, tf2::TimePointZero)),
          level_on_floor_);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "Waiting for static %s -> %s to level %s: %s",
          base_frame_.c_str(), lidar_imu_frame_.c_str(), level_frame_.c_str(), ex.what());
        return;
      }
    } else {
      tf2::Transform odom_base;
      try {
        odom_base = toTransform(
          tf_buffer_->lookupTransform(odom_frame_, base_frame_, tf2::TimePointZero));
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "Waiting for %s -> %s to level %s: %s",
          odom_frame_.c_str(), base_frame_.c_str(), level_frame_.c_str(), ex.what());
        return;
      }
      // map -> base = (map -> odom) * (odom -> base); early on map -> odom ~ I.
      level = pepper_slam::levelFromPose(map_odom_ * odom_base, level_on_floor_);
    }
    const double level_z = level.getOrigin().z();

    geometry_msgs::msg::TransformStamped out;
    out.header.stamp = get_clock()->now();
    out.header.frame_id = level_frame_;
    out.child_frame_id = map_frame_;
    fill(out, level);
    static_broadcaster_->sendTransform(out);
    level_published_ = true;
    // Left subscribed, the listener would keep deserializing naoqi_driver's
    // joint tree on /tf (~4150 transforms/s) for the whole run: measured ~11%
    // of a core.
    tf_listener_.reset();
    RCLCPP_INFO(
      get_logger(),
      "Published static %s -> %s (one-time leveling from %s, z=0 on the %s, offset %+.3f m; "
      "use '%s' as RViz fixed frame).", level_frame_.c_str(), map_frame_.c_str(),
      level_source_.c_str(), level_on_floor_ ? "floor" : "LIO start pose", level_z,
      level_frame_.c_str());
  }

  std::string map_frame_, odom_frame_, base_frame_, level_frame_, lidar_imu_frame_,
    level_source_;
  bool publish_level_{true};
  bool level_on_floor_{true};
  bool level_published_{false};
  std::size_t buffer_size_{1000};

  std::deque<OdomSample> odom_;
  tf2::Transform map_odom_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_pgo_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PgoMapOdomBridge>());
  rclcpp::shutdown();
  return 0;
}
