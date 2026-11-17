/* lio_odom_bridge.cpp
 *
 * Publishes REP-105 odom -> base_footprint from a LIO estimator's odometry.
 *
 * The estimator emits lio_init -> <body frame> as Odometry (lio_init is its own
 * tilted world frame). This node closes the tree by publishing
 *
 *     odom -> base_footprint = (lio_init -> body) * (base_footprint -> body)^-1
 *
 * reading the first term from the message and the second from the static TF.
 * Output is stamped with the odometry message's own stamp. The estimator's own
 * TF broadcast must be disabled (publish_tf: false) or the tree splits.
 *
 * It also publishes a one-time static level_frame -> odom_frame, so level_frame
 * is gravity-aligned with z = 0 on the floor; optionally flattens the published
 * pose onto the floor (flatten_base_frame); and optionally gates it through the
 * divergence guard (guard_enable, lio_odom_guard.cpp), reporting the guard's
 * state on /diagnostics and a wheel-odometry trust scale.
 *
 * Parameters: config/pepper_slam_configuration.yaml, section lio_odom_bridge.
 * lidar_imu_frame MUST equal the estimator config's publish.body_frame; the
 * launch file reads it from that config.
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

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/exceptions.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/static_transform_broadcaster.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_ros/transform_listener.hpp>

#include "pepper_slam/lio_odom_guard.h"
#include "pepper_slam/transform_utils.h"

namespace
{

using pepper_slam::GuardVerdict;

using pepper_slam::fill;
using pepper_slam::toTransform;

// A rotation projected onto pure yaw about z, discarding roll and pitch.
tf2::Matrix3x3 yawOnly(const tf2::Matrix3x3 & r)
{
  const double yaw = std::atan2(r[1][0], r[0][0]);
  const double c = std::cos(yaw), s = std::sin(yaw);
  return tf2::Matrix3x3(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
}

}  // namespace

class LioOdomBridge : public rclcpp::Node
{
public:
  LioOdomBridge()
  : Node("lio_odom_bridge")
  {
    odom_frame_ = declare_parameter<std::string>("odom_frame", "lio_init");
    base_frame_ = declare_parameter<std::string>("base_frame", "base_footprint");
    lidar_imu_frame_ = declare_parameter<std::string>("lidar_imu_frame", "l2lidar_frame_imu");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odom_lio");
    level_frame_ = declare_parameter<std::string>("level_frame", "odom");
    // False when a higher layer owns odom (e.g. the pose-graph's map -> odom).
    publish_level_ = declare_parameter<bool>("publish_level_frame", true);
    // With a localizer publishing map -> odom, level_frame cannot be odom's
    // parent; publishing the inverse as a child keeps one parent per frame.
    level_as_child_ = declare_parameter<bool>("level_frame_as_child", false);
    level_on_floor_ = declare_parameter<bool>("level_frame_on_floor", true);
    // 'calibration': the levelling is the static base_frame -> lidar_imu_frame
    // mount, exact and identical on every run. 'odometry' (legacy): snapshot
    // from the first odometry message, which leaks in any motion before it.
    // Calibration needs the estimator's world frame to start at identity
    // rotation (Point-LIO: mapping.gravity_align false).
    level_source_ = declare_parameter<std::string>("level_source", "calibration");
    if (level_source_ != "calibration" && level_source_ != "odometry") {
      RCLCPP_ERROR(
        get_logger(), "level_source '%s' is not 'calibration' or 'odometry'; falling back "
        "to 'calibration'.", level_source_.c_str());
      level_source_ = "calibration";
    }
    // Zero the levelled z, roll and pitch every cycle: Pepper is flat-floor only.
    flatten_ = declare_parameter<bool>("flatten_base_frame", false);

    // Divergence guard, off by default: it changes what lands on
    // odom -> base_footprint.
    const bool guard_enable = declare_parameter<bool>("guard_enable", false);
    pepper_slam::GuardParams gp;
    gp.max_linear_speed = declare_parameter<double>("max_linear_speed", 0.7);
    gp.max_angular_speed = declare_parameter<double>("max_angular_speed", 1.0);
    const auto wheel_topic = declare_parameter<std::string>("wheel_odom_topic", "/pepper_odom");
    gp.max_hold_duration = declare_parameter<double>("max_hold_duration", 3.0);
    gp.zero_velocity_eps = declare_parameter<double>("zero_velocity_eps", 0.02);
    gp.zero_velocity_lio_eps = declare_parameter<double>("zero_velocity_lio_eps", 0.10);
    const auto trust_topic = declare_parameter<std::string>(
      "trust_scale_topic", "/localization/wheel_trust_scale");
    // scale(d) = degraded_trust_scale + (hold_drift_k * d)^2, capped.
    degraded_trust_scale_ = declare_parameter<double>("degraded_trust_scale", 0.5);
    hold_drift_k_ = declare_parameter<double>("hold_drift_k", 1.0);
    max_trust_scale_ = declare_parameter<double>("max_trust_scale", 10.0);

    if (flatten_ && !publish_level_) {
      RCLCPP_ERROR(
        get_logger(),
        "flatten_base_frame is true but publish_level_frame is false -- flattening needs "
        "that rotation snapshot to know which way is up in the raw (tilted) odom frame. "
        "Flattening will be skipped until publish_level_frame is also enabled.");
    }

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    // Only until the static base_frame -> lidar_imu_frame is cached.
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

    sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10, std::bind(&LioOdomBridge::onOdom, this, std::placeholders::_1));

    if (guard_enable) {
      guard_ = std::make_unique<pepper_slam::OdomGuard>(gp);
      wheel_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        wheel_topic, 50, std::bind(&LioOdomBridge::onWheelOdom, this, std::placeholders::_1));
      trust_pub_ = create_publisher<std_msgs::msg::Float64>(trust_topic, 10);
      diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
      RCLCPP_INFO(
        get_logger(),
        "guard ENABLED: rejecting poses above %.2f m/s / %.2f rad/s, dead reckoning on %s for "
        "up to %.1f s.", gp.max_linear_speed, gp.max_angular_speed, wheel_topic.c_str(),
        gp.max_hold_duration);
    }

    // With 'calibration' the levelling depends only on the static TF, so publish
    // it as soon as that resolves rather than at the first odometry message.
    if (publish_level_ && level_source_ == "calibration") {
      level_timer_ = rclcpp::create_timer(
        this, get_clock(), std::chrono::milliseconds(200), [this]() {publishLevelEarly();});
    }

    const std::string level_msg = publish_level_ ?
      " and (once) " + level_frame_ + " -> " + odom_frame_ :
      " (leveling " + level_frame_ + " DISABLED; a higher layer owns " + odom_frame_ + ")";
    RCLCPP_INFO(
      get_logger(),
      "lio_odom_bridge: consuming %s (%s -> %s), publishing %s -> %s%s. Ensure FAST-LIO's own "
      "TF broadcast (publish.publish_tf) is DISABLED.", odom_topic_.c_str(), odom_frame_.c_str(),
      lidar_imu_frame_.c_str(), odom_frame_.c_str(), base_frame_.c_str(), level_msg.c_str());
  }

private:
  void publishLevelEarly()
  {
    if (!level_published_ && lookupBaseImu()) {
      publishLevelFrame(std::nullopt);
    }
    if (level_published_) {
      level_timer_->cancel();
    }
  }

  // Static base_frame -> lidar_imu_frame, cached after the first success.
  const tf2::Transform * lookupBaseImu()
  {
    if (base_imu_) {
      return &*base_imu_;
    }
    try {
      const auto t = tf_buffer_->lookupTransform(
        base_frame_, lidar_imu_frame_, tf2::TimePointZero);
      base_imu_ = toTransform(t);
      // Nothing else reads TF. Left subscribed, the listener would keep
      // deserializing naoqi_driver's joint tree on /tf (~4150 transforms/s)
      // for the whole run: measured ~11% of a core.
      tf_listener_.reset();
      return &*base_imu_;
    } catch (const tf2::TransformException & ex) {
      const auto now = std::chrono::steady_clock::now();
      if (!base_imu_wait_start_) {
        base_imu_wait_start_ = now;
      }
      const double waited =
        std::chrono::duration<double>(now - *base_imu_wait_start_).count();
      if (waited > 10.0) {
        // Past a startup race: very likely a missing launch dependency.
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 15000,
          "Still no static %s -> %s after %.0fs: %s. This node cannot publish anything until "
          "that transform exists -- is 'pepper_sensor_tf.launch.py' (or equivalent) running? "
          "No output will appear until it is.", base_frame_.c_str(), lidar_imu_frame_.c_str(),
          waited, ex.what());
      } else {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "Waiting for static %s -> %s: %s",
          base_frame_.c_str(), lidar_imu_frame_.c_str(), ex.what());
      }
      return nullptr;
    }
  }

  // One-time static level_frame -> odom_frame (or its inverse as a child).
  void publishLevelFrame(const std::optional<tf2::Transform> & odom_base)
  {
    const tf2::Transform corr = level_source_ == "calibration" ?
      // The estimator's world frame starts at identity rotation, so odom's
      // axes ARE the lidar IMU's axes: the levelling is the static mount.
      pepper_slam::levelFromCalibration(*base_imu_, level_on_floor_) :
      pepper_slam::levelFromPose(*odom_base, level_on_floor_);
    level_rotation_ = corr.getBasis();  // for flatten_base_frame
    level_z_ = corr.getOrigin().z();

    geometry_msgs::msg::TransformStamped out;
    out.header.stamp = get_clock()->now();
    if (level_as_child_) {
      // The inverse, so the geometry is unchanged and only the roles swap.
      out.header.frame_id = odom_frame_;
      out.child_frame_id = level_frame_;
      fill(out, corr.inverse());
    } else {
      out.header.frame_id = level_frame_;
      out.child_frame_id = odom_frame_;
      fill(out, corr);
    }
    static_broadcaster_->sendTransform(out);
    level_published_ = true;
    RCLCPP_INFO(
      get_logger(),
      "Published static %s -> %s (one-time leveling from %s, z=0 on the %s, offset %+.3f m; "
      "use '%s' as RViz's fixed frame for an upright view).", out.header.frame_id.c_str(),
      out.child_frame_id.c_str(), level_source_.c_str(),
      level_on_floor_ ? "floor" : "LIO start pose", level_z_, level_frame_.c_str());
  }

  // Project odom -> base_footprint onto a flat floor: zero the LEVELLED z, roll
  // and pitch (the estimator's own drifting estimate of them), keep x, y, yaw.
  // A hard assumption, valid only because Pepper is always on a flat floor.
  tf2::Transform flattenToLevel(const tf2::Transform & odom_base) const
  {
    const tf2::Matrix3x3 & r_level = *level_rotation_;
    // Raw odom's axes are not up/forward/side; go via the levelled frame.
    tf2::Vector3 p_level = r_level * odom_base.getOrigin();
    // Clamp to the published level frame's datum, so "flat" means the floor.
    p_level.setZ(-level_z_);
    const tf2::Matrix3x3 r_flat = yawOnly(r_level * odom_base.getBasis());
    const tf2::Matrix3x3 r_level_inv = r_level.transpose();
    return tf2::Transform(r_level_inv * r_flat, r_level_inv * p_level);
  }

  void onOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    if (!msg->header.frame_id.empty() && msg->header.frame_id != odom_frame_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000, "Odometry frame_id '%s' != odom_frame '%s'.",
        msg->header.frame_id.c_str(), odom_frame_.c_str());
    }
    if (!msg->child_frame_id.empty() && msg->child_frame_id != lidar_imu_frame_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "Odometry child_frame_id '%s' != lidar_imu_frame '%s'.", msg->child_frame_id.c_str(),
        lidar_imu_frame_.c_str());
    }

    const tf2::Transform * base_imu = lookupBaseImu();
    if (base_imu == nullptr) {
      return;
    }

    // odom -> base = (odom -> imu) * (base -> imu)^-1
    tf2::Transform odom_base =
      toTransform(msg->pose.pose.position, msg->pose.pose.orientation) * base_imu->inverse();

    if (publish_level_ && !level_published_) {
      publishLevelFrame(odom_base);
    }
    if (flatten_ && level_rotation_) {
      odom_base = flattenToLevel(odom_base);
    }
    // Gate last, on the pose that will actually be broadcast: the one
    // chokepoint every downstream consumer passes through.
    if (guard_) {
      odom_base = applyGuard(odom_base, msg->header.stamp);
    }

    geometry_msgs::msg::TransformStamped out;
    out.header.stamp = msg->header.stamp;  // source time, not wall-now
    out.header.frame_id = odom_frame_;
    out.child_frame_id = base_frame_;
    fill(out, odom_base);
    broadcaster_->sendTransform(out);
  }

  void onWheelOdom(const nav_msgs::msg::Odometry::ConstSharedPtr msg)
  {
    wheel_pose_ = toTransform(msg->pose.pose.position, msg->pose.pose.orientation);
    const auto & v = msg->twist.twist.linear;
    wheel_speed_ = std::hypot(v.x, v.y);
  }

  tf2::Transform applyGuard(
    const tf2::Transform & odom_base, const builtin_interfaces::msg::Time & stamp)
  {
    const double t = pepper_slam::toSec(stamp);
    const auto result = guard_->check(odom_base, t, wheel_pose_, wheel_speed_);

    if (result.verdict == GuardVerdict::kFault) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "LIO pose rejected (%s); dead reckoning on wheel odometry.",
        guard_->lastReason().c_str());
    } else if (result.verdict == GuardVerdict::kHold) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "LIO pose rejected (%s); dead reckoning on wheel odometry.",
        guard_->lastReason().c_str());
    } else if (last_verdict_ != GuardVerdict::kAccept) {
      RCLCPP_INFO(get_logger(), "LIO pose accepted again; guard released.");
    }
    if (!wheel_pose_ && result.verdict != GuardVerdict::kAccept) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "No wheel odometry: the held pose is FROZEN, not dead reckoned.");
    }
    // An over-tight gate rejects exactly the measurements that would correct
    // it, so a high rate convicts the guard rather than the estimator.
    if (guard_->rejectionRate() > 0.5) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "Guard rejecting %.0f%% of poses -- suspect the thresholds, not the estimator "
        "(gate lockout).", guard_->rejectionRate() * 100.0);
    }
    last_verdict_ = result.verdict;
    publishHealth(result.verdict, stamp);
    return result.pose;
  }

  // Wheel covariance multiplier: ramps with the distance held on wheels, not
  // with the verdict alone. FAULT means "held too long"; a long hold that
  // covered no ground has added no error.
  double trustScale(GuardVerdict verdict) const
  {
    if (verdict == GuardVerdict::kAccept) {
      return 1.0;
    }
    const double drift = hold_drift_k_ * guard_->heldDistance();
    return std::min(degraded_trust_scale_ + drift * drift, max_trust_scale_);
  }

  void publishHealth(GuardVerdict verdict, const builtin_interfaces::msg::Time & stamp)
  {
    std_msgs::msg::Float64 scale;
    scale.data = trustScale(verdict);
    trust_pub_->publish(scale);

    using diagnostic_msgs::msg::DiagnosticStatus;
    using diagnostic_msgs::msg::KeyValue;
    DiagnosticStatus status;
    status.name = "lio_odom_bridge: LIO pose validation";
    status.hardware_id = odom_topic_;
    if (verdict == GuardVerdict::kAccept) {
      status.level = DiagnosticStatus::OK;
      status.message = "LIO pose accepted";
    } else if (verdict == GuardVerdict::kHold) {
      status.level = DiagnosticStatus::WARN;
      status.message = "holding on wheel odometry: " + guard_->lastReason();
    } else {
      status.level = DiagnosticStatus::ERROR;
      status.message = "held longer than max_hold_duration; pose is NOT trustworthy: " +
        guard_->lastReason();
    }
    auto kv = [](const std::string & key, const std::string & value) {
        KeyValue k;
        k.key = key;
        k.value = value;
        return k;
      };
    char buf[32];
    status.values.push_back(kv("verdict", pepper_slam::toString(verdict)));
    std::snprintf(buf, sizeof(buf), "%.3f", guard_->heldDistance());
    status.values.push_back(kv("held_distance_m", buf));
    std::snprintf(buf, sizeof(buf), "%.3f", scale.data);
    status.values.push_back(kv("trust_scale", buf));
    std::snprintf(buf, sizeof(buf), "%.3f", guard_->rejectionRate());
    status.values.push_back(kv("rejection_rate", buf));
    status.values.push_back(kv("wheel_odom", wheel_pose_ ? "yes" : "MISSING"));

    diagnostic_msgs::msg::DiagnosticArray diag;
    diag.header.stamp = stamp;
    diag.status.push_back(status);
    diag_pub_->publish(diag);
  }

  std::string odom_frame_, base_frame_, lidar_imu_frame_, odom_topic_, level_frame_,
    level_source_;
  bool publish_level_{true};
  bool level_as_child_{false};
  bool level_on_floor_{true};
  bool flatten_{false};
  bool level_published_{false};
  double degraded_trust_scale_{0.5};
  double hold_drift_k_{1.0};
  double max_trust_scale_{10.0};

  std::optional<tf2::Transform> base_imu_;
  std::optional<std::chrono::steady_clock::time_point> base_imu_wait_start_;
  std::optional<tf2::Matrix3x3> level_rotation_;
  double level_z_{0.0};

  std::unique_ptr<pepper_slam::OdomGuard> guard_;
  std::optional<tf2::Transform> wheel_pose_;
  std::optional<double> wheel_speed_;
  GuardVerdict last_verdict_{GuardVerdict::kAccept};

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr trust_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr level_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LioOdomBridge>());
  rclcpp::shutdown();
  return 0;
}
