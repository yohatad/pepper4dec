/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: July 05, 2026
Version: v1.0 - C++ port of animate_behavior_implementation.py / animate_behavior_application.py
*/

#include "animate_behavior/animate_behavior_interface.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>

namespace {
double nowSeconds() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

double randomUniform(double lo, double hi) {
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(gen);
}
}  // namespace

const std::vector<std::vector<std::string>> AnimateBehaviorNode::kCascadeLayers = {
    {"FaceLedRight1", "FaceLedLeft1"},
    {"FaceLedRight0", "FaceLedRight2", "FaceLedLeft0", "FaceLedLeft2"},
    {"FaceLedRight7", "FaceLedRight3", "FaceLedLeft7", "FaceLedLeft3"},
    {"FaceLedRight6", "FaceLedRight4", "FaceLedLeft6", "FaceLedLeft4"},
    {"FaceLedRight5", "FaceLedLeft5"},
};

AnimateBehaviorNode::AnimateBehaviorNode() : rclcpp_lifecycle::LifecycleNode("animate_behavior") {
    // Declare parameters here so they are settable via launch before configure
    declare_parameter("verbose_mode", true);
    declare_parameter("led_enabled", true);
    declare_parameter("led_white_step", 0.06);
    declare_parameter("led_dark_step", 0.04);
    declare_parameter("led_fade_duration", 0.10);
    declare_parameter("led_white_hold", 2.0);
    declare_parameter("led_dark_pause", 0.2);
    declare_parameter("gesture_update_rate", 30.0);
    declare_parameter("gesture_smoothing_factor", 0.15);
    declare_parameter("gesture_motion_speed", 0.08);
    declare_parameter("gesture_interval_min", 2.5);
    declare_parameter("gesture_interval_max", 4.5);
    declare_parameter("gesture_rotation_interval", 5.0);
}

AnimateBehaviorNode::CallbackReturn AnimateBehaviorNode::on_configure(const rclcpp_lifecycle::State&) {
    verbose_mode_ = get_parameter("verbose_mode").as_bool();
    led_enabled_ = get_parameter("led_enabled").as_bool();
    led_white_step_ = get_parameter("led_white_step").as_double();
    led_dark_step_ = get_parameter("led_dark_step").as_double();
    led_fade_duration_ = get_parameter("led_fade_duration").as_double();
    led_white_hold_ = get_parameter("led_white_hold").as_double();
    led_dark_pause_ = get_parameter("led_dark_pause").as_double();
    gesture_interval_min_ = get_parameter("gesture_interval_min").as_double();
    gesture_interval_max_ = get_parameter("gesture_interval_max").as_double();
    rotation_interval_ = get_parameter("gesture_rotation_interval").as_double();
    update_rate_ = get_parameter("gesture_update_rate").as_double();
    smoothing_factor_ = get_parameter("gesture_smoothing_factor").as_double();
    motion_speed_ = get_parameter("gesture_motion_speed").as_double();

    if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "ANIMATE BEHAVIOR NODE (LIFECYCLE)");
    }

    callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    joints_["RArm"] = JointDef{
        {"RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"},
        {-2.09, -1.56, -2.09, 0.01, -1.82},
        {2.09, -0.01, 2.09, 1.56, 1.82},
        {1.7410, -0.09664, 1.6981, 0.09664, -0.05679},
        {0.6, 0.4, 0.6, 0.4, 0.5}};
    joints_["LArm"] = JointDef{
        {"LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"},
        {-2.09, 0.01, -2.09, -1.56, -1.82},
        {2.09, 1.56, 2.09, -0.01, 1.82},
        {1.7625, 0.09970, -1.7150, -0.1334, 0.06592},
        {0.6, 0.4, 0.6, 0.4, 0.5}};
    joints_["RHand"] = JointDef{{"RHand"}, {0.0}, {1.0}, {0.67}, {0.8}};
    joints_["LHand"] = JointDef{{"LHand"}, {0.0}, {1.0}, {0.67}, {0.8}};
    joints_["Leg"] = JointDef{
        {"HipPitch", "HipRoll", "KneePitch"},
        {-1.04, -0.51, -0.51},
        {1.04, 0.51, 0.51},
        {0.0107, -0.00766, 0.03221},
        {0.0, 0.2, 0.0}};

    // Animation state
    active_ = false;
    behavior_.clear();
    range_ = 0.2;
    limbs_to_animate_.clear();
    last_rotation_ = 0.0;
    last_gesture_time_ = 0.0;
    gesture_count_ = 0;
    current_goal_handle_.reset();
    duration_ = 0.0;
    rotation_sign_ = 1.0;
    rotation_stop_timer_.reset();
    current_positions_.clear();
    target_positions_.clear();
    joint_states_received_ = false;
    gesture_interval_ = randomUniform(gesture_interval_min_, gesture_interval_max_);

    // Managed publishers — silenced when node is INACTIVE
    joint_pub_ = create_publisher<naoqi_bridge_msgs::msg::JointAnglesWithSpeed>("/joint_angles", 10);
    vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

    // Action server — created once; handleGoal guards against inactive state
    action_server_ = rclcpp_action::create_server<AnimateBehaviorAction>(
        this,
        "/animate_behavior",
        std::bind(&AnimateBehaviorNode::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&AnimateBehaviorNode::handleCancel, this, std::placeholders::_1),
        std::bind(&AnimateBehaviorNode::handleAccepted, this, std::placeholders::_1),
        rcl_action_server_get_default_options(),
        callback_group_);

    // Stop service — always available after configure
    stop_service_ = create_service<std_srvs::srv::Trigger>(
        "/animate_behavior/stop",
        std::bind(&AnimateBehaviorNode::stopServiceCallback, this, std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default,
        callback_group_);

    // LED action client
    led_client_ = rclcpp_action::create_client<RunLed>(this, "/naoqi_driver/run_led");
    led_active_ = false;
    led_scheduled_timers_.clear();

    RCLCPP_INFO(get_logger(), "configured");
    return CallbackReturn::SUCCESS;
}

AnimateBehaviorNode::CallbackReturn AnimateBehaviorNode::on_activate(const rclcpp_lifecycle::State& state) {
    LifecycleNode::on_activate(state);
    lifecycle_active_ = true;

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10, std::bind(&AnimateBehaviorNode::jointStatesCallback, this, std::placeholders::_1));

    animation_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / update_rate_)),
        std::bind(&AnimateBehaviorNode::animationUpdate, this),
        callback_group_);
    feedback_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(0.5)),
        std::bind(&AnimateBehaviorNode::feedbackUpdate, this),
        callback_group_);

    // Start LED animation — server check runs in a thread to avoid blocking the lifecycle callback
    if (led_enabled_) {
        std::thread(&AnimateBehaviorNode::startLedAsync, this).detach();
    } else if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "LED animation disabled in config");
    }

    RCLCPP_INFO(get_logger(),
        "activated — animation@%.1fHz, feedback@2Hz | waiting for goals on /animate_behavior", update_rate_);
    return CallbackReturn::SUCCESS;
}

AnimateBehaviorNode::CallbackReturn AnimateBehaviorNode::on_deactivate(const rclcpp_lifecycle::State& state) {
    lifecycle_active_ = false;
    stop();
    stopLeds();

    if (animation_timer_) {
        animation_timer_->cancel();
        animation_timer_.reset();
    }
    if (feedback_timer_) {
        feedback_timer_->cancel();
        feedback_timer_.reset();
    }
    joint_sub_.reset();

    LifecycleNode::on_deactivate(state);
    RCLCPP_INFO(get_logger(), "deactivated");
    return CallbackReturn::SUCCESS;
}

AnimateBehaviorNode::CallbackReturn AnimateBehaviorNode::on_cleanup(const rclcpp_lifecycle::State&) {
    joint_pub_.reset();
    vel_pub_.reset();
    action_server_.reset();
    stop_service_.reset();
    led_client_.reset();
    RCLCPP_INFO(get_logger(), "cleaned up");
    return CallbackReturn::SUCCESS;
}

AnimateBehaviorNode::CallbackReturn AnimateBehaviorNode::on_shutdown(const rclcpp_lifecycle::State&) {
    RCLCPP_INFO(get_logger(), "shutting down");
    return CallbackReturn::SUCCESS;
}

// ── Subscription callback ────────────────────────────────────────────────────

void AnimateBehaviorNode::jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    for (size_t i = 0; i < msg->name.size(); ++i) {
        if (i < msg->position.size()) {
            current_positions_[msg->name[i]] = msg->position[i];
        }
    }
    if (!joint_states_received_) {
        joint_states_received_ = true;
        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "joint states received: %zu joints", current_positions_.size());
        }
    }
}

// ── Action server callbacks ──────────────────────────────────────────────────

rclcpp_action::GoalResponse AnimateBehaviorNode::handleGoal(
    const rclcpp_action::GoalUUID&, std::shared_ptr<const AnimateBehaviorAction::Goal> goal) {
    if (!lifecycle_active_) {
        RCLCPP_WARN(get_logger(), "rejecting goal — node is not active");
        return rclcpp_action::GoalResponse::REJECT;
    }
    if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "goal: %s, range=%.2f, duration=%d",
            goal->behavior_type.c_str(), goal->selected_range, goal->duration_seconds);
    }
    static const std::unordered_set<std::string> valid_behaviors = {
        "All", "body", "arms", "hands", "idle", "rotation", "home"};
    if (valid_behaviors.find(goal->behavior_type) == valid_behaviors.end()) {
        RCLCPP_ERROR(get_logger(), "invalid behavior: %s", goal->behavior_type.c_str());
        return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse AnimateBehaviorNode::handleCancel(
    const std::shared_ptr<GoalHandleAnimateBehavior>) {
    if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "cancel requested");
    }
    return rclcpp_action::CancelResponse::ACCEPT;
}

void AnimateBehaviorNode::handleAccepted(const std::shared_ptr<GoalHandleAnimateBehavior> goal_handle) {
    std::thread(&AnimateBehaviorNode::execute, this, goal_handle).detach();
}

void AnimateBehaviorNode::execute(const std::shared_ptr<GoalHandleAnimateBehavior> goal_handle) {
    current_goal_handle_ = goal_handle;
    start_time_ = get_clock()->now();
    auto goal = goal_handle->get_goal();
    duration_ = goal->duration_seconds;
    behavior_ = goal->behavior_type;
    range_ = goal->selected_range;

    if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "starting behavior: %s, range=%.2f", behavior_.c_str(), range_);
    }

    std::unordered_set<std::string> limbs;
    if (behavior_ == "All" || behavior_ == "body") {
        limbs.insert({"RArm", "LArm", "RHand", "LHand", "Leg"});
    }
    if (behavior_ == "All" || behavior_ == "arms") {
        limbs.insert({"RArm", "LArm"});
    }
    if (behavior_ == "All" || behavior_ == "hands") {
        limbs.insert({"RHand", "LHand"});
    }
    if (behavior_ == "home") {
        // Move all joints to neutral; auto-stop after smoothing settles
        for (const auto& [name, joint_def] : joints_) {
            (void)name;
            for (size_t i = 0; i < joint_def.names.size(); ++i) {
                target_positions_[joint_def.names[i]] = joint_def.home[i];
            }
        }
        if (duration_ == 0.0) duration_ = 1.5;
    }
    limbs_to_animate_.assign(limbs.begin(), limbs.end());

    active_ = true;
    gesture_count_ = 0;
    last_gesture_time_ = nowSeconds();
    last_rotation_ = nowSeconds();
    rotation_sign_ = 1.0;
    gesture_interval_ = randomUniform(gesture_interval_min_, gesture_interval_max_);

    {
        std::lock_guard<std::mutex> lock(goal_mutex_);
        goal_complete_ = false;
    }

    // Block until stop() signals completion (mirrors Python's threading.Event().wait())
    {
        std::unique_lock<std::mutex> lock(goal_mutex_);
        goal_cv_.wait(lock, [this] { return goal_complete_; });
    }

    double elapsed = (get_clock()->now() - start_time_).seconds();
    auto result = std::make_shared<AnimateBehaviorAction::Result>();
    result->total_duration = static_cast<float>(elapsed);
    if (goal_handle->is_canceling()) {
        result->success = false;
        result->message = "Cancelled";
        goal_handle->canceled(result);
    } else {
        result->success = true;
        result->message = "Completed";
        goal_handle->succeed(result);
    }
    current_goal_handle_.reset();
}

void AnimateBehaviorNode::stopServiceCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "stop service called");
    }
    stop();
    response->success = true;
    response->message = "Animation stopped";
}

void AnimateBehaviorNode::stop() {
    active_ = false;
    if (rotation_stop_timer_) {
        rotation_stop_timer_->cancel();
        rotation_stop_timer_.reset();
    }
    if (vel_pub_) {
        vel_pub_->publish(geometry_msgs::msg::Twist());
    }
    limbs_to_animate_.clear();
    last_gesture_time_ = 0.0;
    gesture_count_ = 0;
    target_positions_.clear();

    {
        std::lock_guard<std::mutex> lock(goal_mutex_);
        goal_complete_ = true;
    }
    goal_cv_.notify_all();
}

// ── Timer callbacks ──────────────────────────────────────────────────────────

void AnimateBehaviorNode::feedbackUpdate() {
    if (!current_goal_handle_ || !active_) return;
    double elapsed = (get_clock()->now() - start_time_).seconds();
    auto feedback = std::make_shared<AnimateBehaviorAction::Feedback>();
    feedback->elapsed_time = static_cast<float>(elapsed);
    feedback->current_limb = limbs_to_animate_.empty() ? "" : limbs_to_animate_.front();
    feedback->gestures_completed = gesture_count_;
    feedback->is_running = active_;
    current_goal_handle_->publish_feedback(feedback);

    if (duration_ > 0 && elapsed >= duration_) {
        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "duration reached: %.2fs", duration_);
        }
        stop();
    }
}

void AnimateBehaviorNode::animationUpdate() {
    if (!active_ || !joint_states_received_) return;

    double current_time = nowSeconds();

    if (behavior_ == "All" || behavior_ == "body" || behavior_ == "rotation") {
        if (current_time - last_rotation_ >= rotation_interval_) {
            applyRotation();
            last_rotation_ = current_time;
        }
    }

    if (!limbs_to_animate_.empty()) {
        if (current_time - last_gesture_time_ >= gesture_interval_) {
            applyGesture();
            last_gesture_time_ = current_time;
            gesture_count_++;
            gesture_interval_ = randomUniform(gesture_interval_min_, gesture_interval_max_);
        }
    }

    publishJoints();
}

// ── Motion helpers ───────────────────────────────────────────────────────────

void AnimateBehaviorNode::applyRotation() {
    double speed = 0.3;                        // rad/s
    double duration = (M_PI / 4.0) / speed;     // time to turn exactly 45 deg

    if (rotation_stop_timer_) {
        rotation_stop_timer_->cancel();
        rotation_stop_timer_.reset();
    }

    geometry_msgs::msg::Twist twist;
    twist.angular.z = speed * rotation_sign_;
    rotation_sign_ *= -1.0;
    vel_pub_->publish(twist);

    rotation_stop_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(duration)),
        [this]() {
            rotation_stop_timer_->cancel();
            rotation_stop_timer_.reset();
            vel_pub_->publish(geometry_msgs::msg::Twist());
        },
        callback_group_);
}

void AnimateBehaviorNode::applyGesture() {
    for (const auto& limb : limbs_to_animate_) {
        if (joints_.find(limb) != joints_.end()) {
            randomGesture(limb);
        }
    }
}

void AnimateBehaviorNode::randomGesture(const std::string& limb_name) {
    const auto& joint_def = joints_.at(limb_name);
    for (size_t i = 0; i < joint_def.names.size(); ++i) {
        double factor = joint_def.factors[i];
        double rand_val = randomUniform(-1.0, 1.0) * range_ * factor;
        double target = joint_def.home[i] + rand_val;
        target = std::max(joint_def.min[i], std::min(joint_def.max[i], target));
        target_positions_[joint_def.names[i]] = target;
    }
}

void AnimateBehaviorNode::publishJoints() {
    naoqi_bridge_msgs::msg::JointAnglesWithSpeed msg;
    msg.header.stamp = get_clock()->now();
    std::vector<std::string> names;
    std::vector<float> angles;
    for (auto& [name, target] : target_positions_) {
        double current = current_positions_.count(name) ? current_positions_[name] : target;
        double smoothed = current + smoothing_factor_ * (target - current);
        current_positions_[name] = smoothed;
        names.push_back(name);
        angles.push_back(static_cast<float>(smoothed));
    }
    msg.joint_names = names;
    msg.joint_angles = angles;
    msg.speed = static_cast<float>(motion_speed_);
    if (!names.empty()) {
        joint_pub_->publish(msg);
    }
}

// ── LED helpers ──────────────────────────────────────────────────────────────

std_msgs::msg::ColorRGBA AnimateBehaviorNode::makeColor(double r, double g, double b) {
    std_msgs::msg::ColorRGBA c;
    c.r = static_cast<float>(r);
    c.g = static_cast<float>(g);
    c.b = static_cast<float>(b);
    c.a = 1.0f;
    return c;
}

void AnimateBehaviorNode::sendRgbFade(double r, double g, double b, double duration, const std::string& target) {
    RunLed::Goal goal;
    goal.target = target;
    goal.mode = RunLed::Goal::MODE_RGB_FADE;
    goal.color = makeColor(r, g, b);
    goal.duration = static_cast<float>(duration);
    led_client_->async_send_goal(goal);
}

void AnimateBehaviorNode::sendOff(const std::string& target) {
    RunLed::Goal goal;
    goal.target = target;
    goal.mode = RunLed::Goal::MODE_OFF;
    led_client_->async_send_goal(goal);
}

void AnimateBehaviorNode::scheduleLed(double delay_sec, std::function<void()> callback) {
    auto box = std::make_shared<rclcpp::TimerBase::SharedPtr>();
    *box = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(delay_sec)),
        [this, box, callback]() {
            (*box)->cancel();
            auto it = std::find(led_scheduled_timers_.begin(), led_scheduled_timers_.end(), *box);
            if (it != led_scheduled_timers_.end()) {
                led_scheduled_timers_.erase(it);
            }
            if (led_active_) {
                callback();
            }
        },
        callback_group_);
    led_scheduled_timers_.push_back(*box);
}

void AnimateBehaviorNode::startLedAsync() {
    if (led_client_->wait_for_action_server(std::chrono::seconds(5))) {
        led_active_ = true;
        cascadeWave();
        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "LED animation enabled");
        }
    } else if (verbose_mode_) {
        RCLCPP_WARN(get_logger(), "LED server not available — disabled");
    }
}

void AnimateBehaviorNode::cascadeWave() {
    size_t n = kCascadeLayers.size();

    for (size_t i = 0; i < n; ++i) {
        const auto& names = kCascadeLayers[n - 1 - i];
        double t = static_cast<double>(i) * led_white_step_;
        for (const auto& name : names) {
            scheduleLed(t, [this, name]() {
                sendRgbFade(1.0, 1.0, 1.0, led_fade_duration_, name);
            });
        }
    }

    double t_hold_end = static_cast<double>(n - 1) * led_white_step_ + led_fade_duration_ + led_white_hold_;
    for (size_t i = 0; i < n; ++i) {
        const auto& names = kCascadeLayers[i];
        double t = t_hold_end + static_cast<double>(i) * led_dark_step_;
        for (const auto& name : names) {
            scheduleLed(t, [this, name]() {
                sendRgbFade(0.0, 0.0, 0.0, led_fade_duration_, name);
            });
        }
    }

    double last_dark = t_hold_end + static_cast<double>(n - 1) * led_dark_step_;
    scheduleLed(last_dark + led_fade_duration_ + led_dark_pause_, [this]() { cascadeWave(); });
}

void AnimateBehaviorNode::stopLeds() {
    led_active_ = false;
    for (auto& timer : led_scheduled_timers_) {
        if (timer) {
            timer->cancel();
        }
    }
    led_scheduled_timers_.clear();
    for (const auto& names : kCascadeLayers) {
        for (const auto& name : names) {
            sendOff(name);
        }
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<AnimateBehaviorNode>();

    // MultiThreadedExecutor lets the action server, timers, and
    // lifecycle state-machine callbacks run concurrently.
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, 4u);
    executor.add_node(node->get_node_base_interface());
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
