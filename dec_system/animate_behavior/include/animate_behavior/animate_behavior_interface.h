/* animate_behavior_interface.h
 *
 * Author: Yohannes Tadesse Haile
 * Date: Jul 05, 2026
 * Version: v1.0 - C++ port of animate_behavior_implementation.py / animate_behavior_application.py
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#ifndef ANIMATE_BEHAVIOR_INTERFACE_H
#define ANIMATE_BEHAVIOR_INTERFACE_H

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <naoqi_bridge_msgs/msg/joint_angles_with_speed.hpp>
#include <naoqi_bridge_msgs/action/run_led.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <dec_interfaces/action/animate_behavior.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_msgs/msg/color_rgba.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <functional>

// Per-limb joint definition: names + soft-limits + home pose + per-joint randomization factor.
struct JointDef {
    std::vector<std::string> names;
    std::vector<double> min;
    std::vector<double> max;
    std::vector<double> home;
    std::vector<double> factors;
};

//=============================================================================
// AnimateBehaviorNode
//
// Lifecycle state machine:
//   UNCONFIGURED -> on_configure:  read parameters, build joint definitions,
//                                  create the joint-angle/velocity lifecycle
//                                  publishers, the action server, the stop
//                                  service, and the LED action client.
//   INACTIVE     -> on_activate:   activate the lifecycle publishers,
//                                  subscribe to /joint_states, start the
//                                  animation/feedback timers, and kick off
//                                  the face-LED cascade animation if enabled.
//   ACTIVE       -> on_deactivate: stop any active animation, cancel and
//                                  destroy the animation/feedback timers,
//                                  destroy the joint-state subscription, and
//                                  deactivate the lifecycle publishers.
//   INACTIVE     -> on_cleanup:    destroy the lifecycle publishers, action
//                                  server, stop service, and LED client.
//=============================================================================

class AnimateBehaviorNode : public rclcpp_lifecycle::LifecycleNode {
public:
    using CallbackReturn =
        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
    using AnimateBehaviorAction = dec_interfaces::action::AnimateBehavior;
    using GoalHandleAnimateBehavior = rclcpp_action::ServerGoalHandle<AnimateBehaviorAction>;
    using RunLed = naoqi_bridge_msgs::action::RunLed;

    AnimateBehaviorNode();

    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_shutdown  (const rclcpp_lifecycle::State& state) override;

private:
    // Face-LED cascade layers — each entry is the set of LED names lit together.
    static const std::vector<std::vector<std::string>> kCascadeLayers;

    // ── Subscription callback ──────────────────────────────────────────────
    void jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg);

    // ── Action server callbacks ─────────────────────────────────────────────
    rclcpp_action::GoalResponse handleGoal(
        const rclcpp_action::GoalUUID& uuid, std::shared_ptr<const AnimateBehaviorAction::Goal> goal);
    rclcpp_action::CancelResponse handleCancel(
        const std::shared_ptr<GoalHandleAnimateBehavior> goal_handle);
    void handleAccepted(const std::shared_ptr<GoalHandleAnimateBehavior> goal_handle);
    void execute(const std::shared_ptr<GoalHandleAnimateBehavior> goal_handle);

    void stopServiceCallback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void stop();

    // ── Timer callbacks ──────────────────────────────────────────────────────
    void feedbackUpdate();
    void animationUpdate();

    // ── Motion helpers ───────────────────────────────────────────────────────
    void applyRotation();
    void applyGesture();
    void randomGesture(const std::string& limb_name);
    void publishJoints();

    // ── LED helpers ──────────────────────────────────────────────────────────
    static std_msgs::msg::ColorRGBA makeColor(double r, double g, double b);
    void sendRgbFade(double r, double g, double b, double duration, const std::string& target);
    void sendOff(const std::string& target);
    void scheduleLed(double delay_sec, std::function<void()> callback);
    void startLedAsync();
    void cascadeWave();
    void stopLeds();

    // ── Parameters ────────────────────────────────────────────────────────────
    bool verbose_mode_ = true;
    bool led_enabled_ = true;
    double led_white_step_ = 0.06;
    double led_dark_step_ = 0.04;
    double led_fade_duration_ = 0.10;
    double led_white_hold_ = 2.0;
    double led_dark_pause_ = 0.2;
    double gesture_interval_min_ = 2.5;
    double gesture_interval_max_ = 4.5;
    double rotation_interval_ = 5.0;
    double update_rate_ = 30.0;
    double smoothing_factor_ = 0.15;
    double motion_speed_ = 0.08;

    bool lifecycle_active_ = false;

    std::unordered_map<std::string, JointDef> joints_;

    // Animation state
    bool active_ = false;
    std::string behavior_;
    double range_ = 0.2;
    std::vector<std::string> limbs_to_animate_;
    double last_rotation_ = 0.0;
    double last_gesture_time_ = 0.0;
    int gesture_count_ = 0;
    std::shared_ptr<GoalHandleAnimateBehavior> current_goal_handle_;
    rclcpp::Time start_time_;
    double duration_ = 0.0;
    double rotation_sign_ = 1.0;
    rclcpp::TimerBase::SharedPtr rotation_stop_timer_;
    std::unordered_map<std::string, double> current_positions_;
    std::unordered_map<std::string, double> target_positions_;
    bool joint_states_received_ = false;
    double gesture_interval_ = 3.0;

    // Goal-completion signalling (mirrors Python's threading.Event)
    std::mutex goal_mutex_;
    std::condition_variable goal_cv_;
    bool goal_complete_ = false;

    rclcpp::CallbackGroup::SharedPtr callback_group_;

    rclcpp_lifecycle::LifecyclePublisher<naoqi_bridge_msgs::msg::JointAnglesWithSpeed>::SharedPtr joint_pub_;
    rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::Twist>::SharedPtr vel_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::TimerBase::SharedPtr animation_timer_;
    rclcpp::TimerBase::SharedPtr feedback_timer_;
    rclcpp_action::Server<AnimateBehaviorAction>::SharedPtr action_server_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_service_;

    rclcpp_action::Client<RunLed>::SharedPtr led_client_;
    bool led_active_ = false;
    std::vector<rclcpp::TimerBase::SharedPtr> led_scheduled_timers_;
};

#endif  // ANIMATE_BEHAVIOR_INTERFACE_H
