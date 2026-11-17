/* sensor_guard.cpp
 *
 * Last gate in Nav2's velocity chain: stops translation while a sensor the
 * collision monitor depends on has gone silent.
 *
 *   controller_server -> cmd_vel_raw -> collision_monitor -> cmd_vel_safe
 *                     -> sensor_guard -> cmd_vel  (what Pepper drives on)
 *
 * Why: MEASURED on slam_20260823_aligned, Humble's collision_monitor HOLDS
 * ITS LAST VERDICT when a source goes stale -- fed a clear laser reading and
 * then silence, it passed 161 consecutive full-speed commands with no sensor
 * data at all. source_timeout only decides whether a reading is used, not
 * whether the robot may keep moving without one, and nothing else in the
 * chain catches it: naoqi_driver's cmd_vel_timeout fires only if cmd_vel
 * stops, and it does not. So required_topics are watched here, in the node
 * that owns cmd_vel and can enforce a stop. Lost clamps translation but
 * leaves rotation: the footprint is a 0.25 m circle, so turning in place
 * sweeps nothing new, and it is how an operator recovers.
 *
 * A topic never heard from counts as lost: a sensor that failed to connect at
 * startup is the case this most needs to catch.
 *
 * Bumper contacts are not handled here: naoqi_driver latches any press into a
 * full stop for every caller (its ~/clear_bumper resets it), so a second
 * latch in the Nav2 chain only duplicated it in one profile.
 *
 * Fails safe by dying: this node owns cmd_vel, so if it exits nothing
 * publishes cmd_vel and the driver's own command timeout halts the robot. It
 * never invents a command either -- it only republishes one it received, so an
 * upstream that goes quiet stops the robot rather than latching the last
 * velocity.
 *
 * Reports "sensor_guard: sensor liveness" on /diagnostics (ERROR naming the
 * lost topics), which localization_watchdog also watches, so a lost sensor
 * cancels the navigation goal instead of leaving Nav2 pushing against the
 * clamp.
 *
 * Parameters (config/pepper_navigation_configuration.yaml):
 *   cmd_vel_in_topic / cmd_vel_out_topic  default cmd_vel_safe -> cmd_vel
 *   required_topics   sensor topics that must stay fresh (default: none, off)
 *   required_timeout  s before a required topic counts as lost (default 1.0)
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: November 17, 2026
 * Version: v3.0 (was bumper_guard v2.0)
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

#include "pepper_navigation/topic_liveness.h"

namespace
{
constexpr char kLivenessName[] = "sensor_guard: sensor liveness";
}  // namespace

class SensorGuard : public rclcpp::Node
{
public:
  SensorGuard()
  : Node("sensor_guard")
  {
    const auto in_topic = declare_parameter<std::string>("cmd_vel_in_topic", "cmd_vel_safe");
    const auto out_topic = declare_parameter<std::string>("cmd_vel_out_topic", "cmd_vel");
    const auto required = declare_parameter<std::vector<std::string>>(
      "required_topics", std::vector<std::string>{});
    sensors_ = pepper_navigation::TopicLiveness(
      required, declare_parameter<double>("required_timeout", 1.0));

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(out_topic, 1);
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    // Depth 1: a stale velocity is worse than a dropped one.
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      in_topic, 1, [this](geometry_msgs::msg::Twist::SharedPtr msg) {onCmdVel(*msg);});

    // Bound on a timer, not here: a publisher that starts after this node
    // would otherwise never be found, and the type has to be discovered.
    bind_timer_ = create_wall_timer(
      std::chrono::milliseconds(500), [this] {bindSensors();});
    diag_timer_ = create_wall_timer(
      std::chrono::milliseconds(500), [this] {publishDiagnostics();});

    std::string topics;
    for (const auto & t : required) {
      topics += (topics.empty() ? "" : ", ") + t;
    }
    RCLCPP_INFO(
      get_logger(), "gating %s -> %s: translation stops while any of [%s] is silent for "
      "more than %.1f s", in_topic.c_str(), out_topic.c_str(), topics.c_str(),
      get_parameter("required_timeout").as_double());
  }

private:
  double nowSeconds() {return now().seconds();}

  /// Generic subscriptions, so any message type counts as a heartbeat without
  /// this node having to depend on it. SensorDataQoS deliberately: it is
  /// BEST_EFFORT, which is compatible with both reliable and best-effort
  /// publishers, and /points is best-effort -- a reliable subscription here
  /// would receive nothing and read as permanently stale.
  void bindSensors()
  {
    for (const auto & topic : sensors_.unseen()) {
      if (sensor_subs_.count(topic)) {continue;}
      const auto types = get_topic_names_and_types();
      auto it = types.find(topic);
      if (it == types.end() || it->second.empty()) {continue;}
      sensor_subs_[topic] = create_generic_subscription(
        topic, it->second.front(), rclcpp::SensorDataQoS(),
        [this, topic](std::shared_ptr<rclcpp::SerializedMessage>) {
          sensors_.mark(topic, nowSeconds());
        });
      RCLCPP_INFO(
        get_logger(), "watching %s (%s) for liveness",
        topic.c_str(), it->second.front().c_str());
    }
  }

  void onCmdVel(const geometry_msgs::msg::Twist & in)
  {
    geometry_msgs::msg::Twist out = in;
    if (sensors_.any_stale(nowSeconds())) {
      // Blind in every direction, so no translation at all. Rotation stays:
      // the footprint is a 0.25 m circle, so turning in place sweeps nothing new.
      out.linear.x = 0.0;
      out.linear.y = 0.0;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "required sensor topic(s) lost; holding translation");
    }
    cmd_pub_->publish(out);
  }

  void publishDiagnostics()
  {
    if (!sensors_.watching()) {return;}
    using diagnostic_msgs::msg::DiagnosticStatus;
    DiagnosticStatus s;
    s.name = kLivenessName;
    s.hardware_id = "pepper_safety_sources";
    const auto bad = sensors_.stale_topics(nowSeconds());
    if (bad.empty()) {
      s.level = DiagnosticStatus::OK;
      s.message = "all required sensor topics fresh";
    } else {
      s.level = DiagnosticStatus::ERROR;
      s.message = "lost: ";
      for (std::size_t i = 0; i < bad.size(); ++i) {
        s.message += (i ? ", " : "") + bad[i];
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = bad[i];
        kv.value = "stale";
        s.values.push_back(kv);
      }
    }
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    arr.status.push_back(s);
    diag_pub_->publish(arr);
  }

  pepper_navigation::TopicLiveness sensors_;
  std::map<std::string, rclcpp::GenericSubscription::SharedPtr> sensor_subs_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::TimerBase::SharedPtr bind_timer_;
  rclcpp::TimerBase::SharedPtr diag_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SensorGuard>());
  rclcpp::shutdown();
  return 0;
}
