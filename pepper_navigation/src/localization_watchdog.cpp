/* localization_watchdog.cpp
 *
 * Stops navigating when localization says it is lost.
 *
 * fastlio_localization reports its own health on /diagnostics (from the
 * periodic map-overlap check), but nothing downstream consumed it: a
 * wrong-but-confident lock left Nav2 planning against a costmap built on a bad
 * map -> base_footprint, driving toward a goal computed from a wrong pose.
 *
 * This node watches those statuses -- the pose lock, and in odom mode the
 * map -> odom jump gate, which reports ERROR after it force-adopts a step the
 * robot cannot have made -- and once any has been bad long enough to not be a
 * transient, cancels the active navigate_to_pose goal, and keeps cancelling
 * while still lost so a goal issued in that window does not slip through. It
 * does not touch cmd_vel: the collision monitor already owns the
 * safety-critical stop, and two arbiters of velocity are worse than one.
 *
 * The sustained-badness gate matters because overlap dips for a scan or two
 * whenever the robot corners into unmapped space or someone walks through the
 * scan. Silence counts as lost too (stale_timeout), since a frozen TF is as
 * dangerous as a wrong one. The decisions live in LocalizationHealth
 * (include/pepper_navigation/localization_health.h).
 *
 * Parameters (config/pepper_navigation_configuration.yaml):
 *   status_names        diagnostics statuses to watch; any one lost is lost
 *   lost_duration       s of continuous ERROR before acting
 *   stale_timeout       s of silence after a first report; 0 disables
 *   treat_warn_as_lost  count WARN as bad too
 *   cancel_goals        cancel every navigation goal while lost
 *   call_recovery       call recovery_service once per LOST episode
 *   nav_action          the navigation action whose goals are cancelled
 *   recovery_service    std_srvs/Trigger asking for a re-search
 *   check_period        s between checks
 *   recancel_period     s between cancels while still lost
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

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <action_msgs/srv/cancel_goal.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "pepper_navigation/localization_health.h"

class LocalizationWatchdog : public rclcpp::Node
{
public:
  LocalizationWatchdog()
  : Node("localization_watchdog"),
    health_(readParams())
  {
    const auto nav_action = declare_parameter<std::string>("nav_action", "/navigate_to_pose");
    const auto recovery_service = declare_parameter<std::string>(
      "recovery_service", "/localization_recover");
    const double check_period = declare_parameter<double>("check_period", 1.0);

    cancel_client_ = create_client<action_msgs::srv::CancelGoal>(
      nav_action + "/_action/cancel_goal");
    recover_client_ = create_client<std_srvs::srv::Trigger>(recovery_service);

    diag_sub_ = create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics", 10,
      [this](diagnostic_msgs::msg::DiagnosticArray::SharedPtr msg) {onDiagnostics(*msg);});

    // On the node's clock, so a bag replay on sim time is checked in sim time.
    timer_ = rclcpp::create_timer(
      this, get_clock(), rclcpp::Duration::from_seconds(check_period), [this] {tick();});

    std::string names;
    for (const auto & n : params_.status_names) {
      names += (names.empty() ? "'" : ", '") + n + "'";
    }
    RCLCPP_INFO(
      get_logger(),
      "watching [%s]; will cancel goals on %.0fs of bad health on any of them "
      "(cancel_goals=%s, call_recovery=%s)",
      names.c_str(), params_.lost_duration, params_.cancel_goals ? "true" : "false",
      params_.call_recovery ? "true" : "false");
  }

private:
  pepper_navigation::HealthParams readParams()
  {
    // A list: the pose lock and the map -> odom gate are separate failures,
    // and either one means the pose is not trustworthy.
    params_.status_names = declare_parameter<std::vector<std::string>>(
      "status_names", {"fastlio_localization: pose lock"});
    params_.lost_duration = declare_parameter<double>("lost_duration", 5.0);
    params_.stale_timeout = declare_parameter<double>("stale_timeout", 10.0);
    params_.warn_is_lost = declare_parameter<bool>("treat_warn_as_lost", false);
    params_.cancel_goals = declare_parameter<bool>("cancel_goals", true);
    params_.call_recovery = declare_parameter<bool>("call_recovery", false);
    params_.recancel_period = declare_parameter<double>("recancel_period", 3.0);
    return params_;
  }

  double now() {return get_clock()->now().seconds();}

  void onDiagnostics(const diagnostic_msgs::msg::DiagnosticArray & msg)
  {
    const double t = now();
    for (const auto & status : msg.status) {
      health_.report(status.name, status.level, t);
    }
  }

  void tick()
  {
    const auto step = health_.step(now());
    if (step.recovered) {
      RCLCPP_INFO(get_logger(), "localization healthy again; no longer holding navigation");
    }
    if (step.became_lost) {
      RCLCPP_ERROR(
        get_logger(),
        "localization LOST (%s) -- holding navigation. The pose is not "
        "trustworthy; goals will keep being cancelled until it recovers.",
        step.reason.c_str());
    }
    if (step.request_recovery) {
      requestRecovery();
    }
    if (step.cancel_goals) {
      cancelAllGoals();
    }
  }

  void cancelAllGoals()
  {
    if (!cancel_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 30000,
        "nav2 cancel service not available; cannot hold navigation");
      return;
    }
    // A zero goal_id and a zero stamp is the action-spec idiom for "cancel
    // everything", so this needs no goal handle of its own.
    cancel_client_->async_send_request(
      std::make_shared<action_msgs::srv::CancelGoal::Request>(),
      [this](rclcpp::Client<action_msgs::srv::CancelGoal>::SharedFuture future) {
        const auto n = future.get()->goals_canceling.size();
        if (n > 0) {
          RCLCPP_WARN(get_logger(), "cancelled %zu active navigation goal(s)", n);
        }
      });
  }

  void requestRecovery()
  {
    if (!recover_client_->service_is_ready()) {
      RCLCPP_WARN(
        get_logger(), "%s not available", recover_client_->get_service_name());
      return;
    }
    recover_client_->async_send_request(
      std::make_shared<std_srvs::srv::Trigger::Request>(),
      [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
        RCLCPP_INFO(get_logger(), "recovery: %s", future.get()->message.c_str());
      });
  }

  pepper_navigation::HealthParams params_;
  pepper_navigation::LocalizationHealth health_;
  rclcpp::Client<action_msgs::srv::CancelGoal>::SharedPtr cancel_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr recover_client_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalizationWatchdog>());
  rclcpp::shutdown();
  return 0;
}
