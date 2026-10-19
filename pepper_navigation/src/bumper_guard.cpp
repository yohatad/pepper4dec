/* bumper_guard.cpp
 *
 * Last gate in the velocity chain: blocks the direction a bumper contact came
 * from, and nothing else.
 *
 *   controller_server -> cmd_vel_raw -> collision_monitor -> cmd_vel_safe
 *                     -> bumper_guard -> cmd_vel  (what Pepper drives on)
 *
 * Why a separate node and not a collision_monitor source: the monitor's whole
 * action vocabulary is isotropic -- DO_NOTHING, STOP, SLOWDOWN, APPROACH all
 * zero or scale the entire command. None can block forward while permitting
 * reverse. With the controller at vx_min 0.0, MPPI can never reverse, so a
 * contact that only stops the robot leaves it leaning on the obstacle with no
 * command that gets it off. Series gates each of which only ever subtracts
 * authority are safe to chain; two PARALLEL arbiters of cmd_vel are not,
 * which is why localization_watchdog cancels goals instead of touching this.
 *
 * Fails safe by dying: this node owns cmd_vel, so if it exits nothing
 * publishes cmd_vel and the driver's own command timeout halts the robot. It
 * never invents a command either -- it only ever republishes one it received,
 * so an upstream that goes quiet stops the robot rather than latching the
 * last velocity.
 *
 * /bumper is an EVENT topic with no header and no heartbeat (naoqi registers
 * RightBumperPressed / LeftBumperPressed / BackBumperPressed at frequency 0),
 * so silence cannot be told from a dead driver on its own. joint_states, which
 * the driver publishes at 50 Hz, is the liveness witness: while it is absent
 * a latch holds rather than ageing out.
 *
 * Parameters:
 *   cmd_vel_in_topic / cmd_vel_out_topic  default cmd_vel_safe -> cmd_vel
 *   clear_delay      s a latch outlives the switch release (default 1.0)
 *   stuck_timeout    s of continuous press before ERROR (default 10.0)
 *   liveness_topic / liveness_timeout     default /joint_states, 1.0 s
 *   block_lateral    also zero vy while latched (default false; vy_max is 0.0)
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: October 19, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include <algorithm>
#include <memory>
#include <string>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <naoqi_bridge_msgs/msg/bumper.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "pepper_navigation/bumper_latch.h"

namespace
{
constexpr char kStatusName[] = "bumper_guard: contact";
}  // namespace

class BumperGuard : public rclcpp::Node
{
public:
  BumperGuard()
  : Node("bumper_guard"),
    latch_(declare_parameter<double>("clear_delay", 1.0),
      declare_parameter<double>("stuck_timeout", 10.0))
  {
    const auto in_topic = declare_parameter<std::string>("cmd_vel_in_topic", "cmd_vel_safe");
    const auto out_topic = declare_parameter<std::string>("cmd_vel_out_topic", "cmd_vel");
    const auto liveness_topic = declare_parameter<std::string>("liveness_topic", "/joint_states");
    liveness_timeout_ = declare_parameter<double>("liveness_timeout", 1.0);
    block_lateral_ = declare_parameter<bool>("block_lateral", false);

    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(out_topic, 1);
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);

    // Depth 1: a stale velocity is worse than a dropped one.
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      in_topic, 1, [this](geometry_msgs::msg::Twist::SharedPtr msg) {onCmdVel(*msg);});

    // Events are rare but must never be dropped, so this queue is deep.
    bumper_sub_ = create_subscription<naoqi_bridge_msgs::msg::Bumper>(
      "/bumper", 50, [this](naoqi_bridge_msgs::msg::Bumper::SharedPtr msg) {onBumper(*msg);});

    liveness_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      liveness_topic, 1,
      [this](sensor_msgs::msg::JointState::SharedPtr) {last_liveness_ = nowSeconds();});

    clear_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/clear",
      [this](
        const std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr res) {onClear(res);});

    diag_timer_ = create_wall_timer(
      std::chrono::milliseconds(500), [this] {publishDiagnostics();});

    RCLCPP_INFO(
      get_logger(), "gating %s -> %s on /bumper (clear_delay %.1fs, liveness %s)",
      in_topic.c_str(), out_topic.c_str(), get_parameter("clear_delay").as_double(),
      liveness_topic.c_str());
  }

private:
  double nowSeconds() {return now().seconds();}

  /// Blind means joint_states has gone quiet, so /bumper's silence proves
  /// nothing. Never seen it at all counts as blind too.
  bool blind()
  {
    if (last_liveness_ == pepper_navigation::kNever) {return true;}
    return (nowSeconds() - last_liveness_) > liveness_timeout_;
  }

  void onBumper(const naoqi_bridge_msgs::msg::Bumper & msg)
  {
    const bool pressed = msg.state == naoqi_bridge_msgs::msg::Bumper::STATE_PRESSED;
    latch_.set(msg.bumper, pressed, nowSeconds());
    if (pressed) {
      RCLCPP_WARN(get_logger(), "bumper %u pressed -- blocking that direction", msg.bumper);
    }
  }

  void onCmdVel(const geometry_msgs::msg::Twist & in)
  {
    latch_.set_blind(blind());
    const auto clamp = latch_.clamp(nowSeconds());

    geometry_msgs::msg::Twist out = in;
    if (clamp.block_forward) {out.linear.x = std::min(out.linear.x, 0.0);}
    if (clamp.block_reverse) {out.linear.x = std::max(out.linear.x, 0.0);}
    // Rotation is always allowed: turning in place is how the robot gets its
    // nose off an obstacle when reversing is not enough.
    if (block_lateral_ && (clamp.block_forward || clamp.block_reverse)) {out.linear.y = 0.0;}

    if (out.linear.x != in.linear.x || out.linear.y != in.linear.y) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "bumper latched: vx %.2f -> %.2f", in.linear.x, out.linear.x);
    }
    cmd_pub_->publish(out);
  }

  void onClear(const std_srvs::srv::Trigger::Response::SharedPtr & res)
  {
    res->success = latch_.clear();
    res->message = res->success ?
      "bumper latches cleared" :
      "refused: a bumper is still pressed";
    RCLCPP_INFO(get_logger(), "%s", res->message.c_str());
  }

  void publishDiagnostics()
  {
    using diagnostic_msgs::msg::DiagnosticStatus;
    const double t = nowSeconds();
    latch_.set_blind(blind());
    const auto clamp = latch_.clamp(t);

    DiagnosticStatus s;
    s.name = kStatusName;
    s.hardware_id = "pepper_bumper";
    if (latch_.stuck(t)) {
      s.level = DiagnosticStatus::ERROR;
      s.message = "bumper held down -- stuck switch or debris";
    } else if (clamp.block_forward || clamp.block_reverse) {
      s.level = DiagnosticStatus::WARN;
      s.message = clamp.block_forward ?
        (clamp.block_reverse ? "contact front and back" : "contact front") :
        "contact back";
    } else if (latch_.blind()) {
      s.level = DiagnosticStatus::WARN;
      s.message = "no joint_states -- bumper state unknown, latches held";
    } else {
      s.level = DiagnosticStatus::OK;
      s.message = "clear";
    }
    auto add = [&s](const char * key, bool val) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = key;
        kv.value = val ? "true" : "false";
        s.values.push_back(kv);
      };
    add("right", latch_.latched(pepper_navigation::kRight, t));
    add("left", latch_.latched(pepper_navigation::kLeft, t));
    add("back", latch_.latched(pepper_navigation::kBack, t));
    add("block_forward", clamp.block_forward);
    add("block_reverse", clamp.block_reverse);
    add("blind", latch_.blind());

    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = now();
    arr.status.push_back(s);
    diag_pub_->publish(arr);
  }

  pepper_navigation::BumperLatch latch_;
  double liveness_timeout_;
  bool block_lateral_;
  double last_liveness_ = pepper_navigation::kNever;

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<naoqi_bridge_msgs::msg::Bumper>::SharedPtr bumper_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr liveness_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
  rclcpp::TimerBase::SharedPtr diag_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BumperGuard>());
  rclcpp::shutdown();
  return 0;
}
