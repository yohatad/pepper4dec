/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: July 05, 2026
Version: v1.0 - C++ port of gesture_execution_implementation.py / gesture_execution_application.py
*/

#include "gesture_execution/gesture_execution_interface.h"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

namespace {
double nowSeconds() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

std::vector<std::string> armJointNames(const std::string& arm_name) {
    if (arm_name == "RArm") return {"RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"};
    if (arm_name == "LArm") return {"LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"};
    if (arm_name == "Leg") return {"HipPitch", "HipRoll", "KneePitch"};
    return {};
}
}  // namespace

std::unordered_map<std::string, GestureDescriptor> loadGestureDescriptors(const std::string& yaml_path) {
    std::unordered_map<std::string, GestureDescriptor> result;
    YAML::Node root;
    try {
        root = YAML::LoadFile(yaml_path);
    } catch (const std::exception&) {
        return result;
    }
    if (!root["gestures"]) return result;

    for (const auto& item : root["gestures"]) {
        std::string name = item.first.as<std::string>();
        YAML::Node data = item.second;

        GestureDescriptor desc;
        if (data["joints"]) {
            for (const auto& j : data["joints"]) desc.arms.push_back(j.as<std::string>());
        }

        YAML::Node joint_angles = data["joint_angles"];
        YAML::Node times = data["times"];
        for (const auto& arm_name : desc.arms) {
            if (!joint_angles || !joint_angles[arm_name]) continue;

            ArmWaypoints aw;
            aw.joint_names = armJointNames(arm_name);
            for (const auto& wp : joint_angles[arm_name]) {
                std::vector<double> row;
                for (const auto& v : wp) row.push_back(v.as<double>());
                aw.waypoints.push_back(row);
            }
            if (times && times[arm_name]) {
                for (const auto& t : times[arm_name]) aw.times.push_back(t.as<double>());
            }
            desc.per_arm[arm_name] = aw;
        }
        result[name] = desc;
    }
    return result;
}

RobotTopics loadRobotTopics(const std::string& yaml_path) {
    RobotTopics topics;
    try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        if (root["topics"]) {
            YAML::Node t = root["topics"];
            if (t["JointStates"]) topics.joint_states = t["JointStates"].as<std::string>();
            if (t["RobotPose"]) topics.robot_pose = t["RobotPose"].as<std::string>();
        }
    } catch (const std::exception&) {
        // Keep defaults, mirroring the Python ConfigManager's best-effort load.
    }
    return topics;
}

GestureExecutionSystem::GestureExecutionSystem() : rclcpp_lifecycle::LifecycleNode("gesture_action_server") {}

GestureExecutionSystem::CallbackReturn GestureExecutionSystem::on_configure(const rclcpp_lifecycle::State&) {
    std::string package_path;
    try {
        package_path = ament_index_cpp::get_package_share_directory("gesture_execution");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Configuration failed: %s", e.what());
        return CallbackReturn::FAILURE;
    }

    // gestureDescriptors/robotTopics keys in the configuration YAML are
    // unused (the data file paths below are fixed) — mirrors the reference
    // Python ConfigManager, which loads but never reads them.
    verbose_mode_ = false;
    try {
        YAML::Node cfg = YAML::LoadFile(package_path + "/config/gesture_execution_configuration.yaml");
        if (cfg["verboseMode"]) verbose_mode_ = cfg["verboseMode"].as<bool>();
    } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "Warning: could not load config: %s", e.what());
    }

    gestures_ = loadGestureDescriptors(package_path + "/data/gesture.yaml");
    topics_ = loadRobotTopics(package_path + "/data/pepper_topics.yaml");
    home_positions_ = HomePositions();

    robot_pose_ = RobotPose();
    joint_states_.clear();
    executing_ = false;
    feedback_stop_ = false;
    lifecycle_active_ = false;

    initJointStates();

    // Managed publishers
    joint_traj_pub_ = create_publisher<naoqi_bridge_msgs::msg::JointAnglesTrajectory>(
        "/joint_angles_trajectory", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/gesture_execution/visualization", 10);

    // Action server — handleGoal guards against inactive state
    action_server_ = rclcpp_action::create_server<GestureAction>(
        this,
        "/gesture_execution",
        std::bind(&GestureExecutionSystem::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&GestureExecutionSystem::handleCancel, this, std::placeholders::_1),
        std::bind(&GestureExecutionSystem::handleAccepted, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "GestureExecutionSystem configured");
    return CallbackReturn::SUCCESS;
}

GestureExecutionSystem::CallbackReturn GestureExecutionSystem::on_activate(const rclcpp_lifecycle::State& state) {
    LifecycleNode::on_activate(state);
    lifecycle_active_ = true;

    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        topics_.joint_states, 10, std::bind(&GestureExecutionSystem::jointStatesCallback, this, std::placeholders::_1));
    pose_sub_ = create_subscription<geometry_msgs::msg::Pose2D>(
        topics_.robot_pose, 10, std::bind(&GestureExecutionSystem::robotPoseCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "GestureExecutionSystem activated — ready for goals on /gesture_execution");
    return CallbackReturn::SUCCESS;
}

GestureExecutionSystem::CallbackReturn GestureExecutionSystem::on_deactivate(const rclcpp_lifecycle::State& state) {
    lifecycle_active_ = false;
    joint_sub_.reset();
    pose_sub_.reset();
    LifecycleNode::on_deactivate(state);
    RCLCPP_INFO(get_logger(), "GestureExecutionSystem deactivated");
    return CallbackReturn::SUCCESS;
}

GestureExecutionSystem::CallbackReturn GestureExecutionSystem::on_cleanup(const rclcpp_lifecycle::State&) {
    joint_traj_pub_.reset();
    marker_pub_.reset();
    action_server_.reset();
    RCLCPP_INFO(get_logger(), "GestureExecutionSystem cleaned up");
    return CallbackReturn::SUCCESS;
}

GestureExecutionSystem::CallbackReturn GestureExecutionSystem::on_shutdown(const rclcpp_lifecycle::State&) {
    RCLCPP_INFO(get_logger(), "GestureExecutionSystem shutting down");
    return CallbackReturn::SUCCESS;
}

// ── Internal helpers ──────────────────────────────────────────────────────

void GestureExecutionSystem::initJointStates() {
    joint_states_["HeadPitch"] = DEFAULT_HEAD_PITCH;
    joint_states_["HeadYaw"] = DEFAULT_HEAD_YAW;
    joint_states_["RShoulderPitch"] = home_positions_.r_arm[0];
    joint_states_["RShoulderRoll"] = home_positions_.r_arm[1];
    joint_states_["RElbowYaw"] = home_positions_.r_arm[2];
    joint_states_["RElbowRoll"] = home_positions_.r_arm[3];
    joint_states_["RWristYaw"] = home_positions_.r_arm[4];
    joint_states_["LShoulderPitch"] = home_positions_.l_arm[0];
    joint_states_["LShoulderRoll"] = home_positions_.l_arm[1];
    joint_states_["LElbowYaw"] = home_positions_.l_arm[2];
    joint_states_["LElbowRoll"] = home_positions_.l_arm[3];
    joint_states_["LWristYaw"] = home_positions_.l_arm[4];
    joint_states_["RHand"] = 0.67;
    joint_states_["LHand"] = 0.67;
    joint_states_["HipPitch"] = home_positions_.leg[0];
    joint_states_["HipRoll"] = home_positions_.leg[1];
    joint_states_["KneePitch"] = home_positions_.leg[2];
}

// ── Subscription callbacks ─────────────────────────────────────────────────

void GestureExecutionSystem::jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i) {
        auto it = joint_states_.find(msg->name[i]);
        if (it != joint_states_.end()) {
            it->second = msg->position[i];
        }
    }
}

void GestureExecutionSystem::robotPoseCallback(const geometry_msgs::msg::Pose2D::SharedPtr msg) {
    robot_pose_.x = msg->x;
    robot_pose_.y = msg->y;
    robot_pose_.theta = msg->theta;
}

// ── Action server callbacks ────────────────────────────────────────────────

rclcpp_action::GoalResponse GestureExecutionSystem::handleGoal(
    const rclcpp_action::GoalUUID&, std::shared_ptr<const GestureAction::Goal> goal) {
    if (!lifecycle_active_) {
        RCLCPP_WARN(get_logger(), "Rejecting goal — node is not active");
        return rclcpp_action::GoalResponse::REJECT;
    }
    if (executing_) {
        RCLCPP_WARN(get_logger(), "Gesture in progress — rejecting new goal");
        return rclcpp_action::GoalResponse::REJECT;
    }
    RCLCPP_INFO(get_logger(), "Accepting goal: type='%s' name='%s'",
        goal->gesture_type.c_str(), goal->gesture_name.c_str());
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse GestureExecutionSystem::handleCancel(const std::shared_ptr<GoalHandleGesture>) {
    RCLCPP_INFO(get_logger(), "Cancel requested (NAOqi motion cannot be stopped mid-way)");
    return rclcpp_action::CancelResponse::ACCEPT;
}

void GestureExecutionSystem::handleAccepted(const std::shared_ptr<GoalHandleGesture> goal_handle) {
    // Executed synchronously (single-threaded spin in main()) — mirrors the
    // reference implementation, where execute_callback blocks the executor
    // for the duration of the gesture while a background thread streams
    // feedback.
    execute(goal_handle);
}

void GestureExecutionSystem::execute(const std::shared_ptr<GoalHandleGesture> goal_handle) {
    executing_ = true;
    feedback_stop_ = false;

    auto goal = goal_handle->get_goal();
    auto result = std::make_shared<GestureAction::Result>();

    std::string gesture_type = goal->gesture_type;
    // Trim whitespace and lowercase, matching Python's `.strip().lower()`.
    gesture_type.erase(gesture_type.begin(), std::find_if(gesture_type.begin(), gesture_type.end(),
        [](unsigned char c) { return !std::isspace(c); }));
    gesture_type.erase(std::find_if(gesture_type.rbegin(), gesture_type.rend(),
        [](unsigned char c) { return !std::isspace(c); }).base(), gesture_type.end());
    std::transform(gesture_type.begin(), gesture_type.end(), gesture_type.begin(),
        [](unsigned char c) { return std::tolower(c); });

    double start_time = nowSeconds();

    if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "Executing gesture — type='%s', name='%s', duration=%ldms",
            gesture_type.c_str(), goal->gesture_name.c_str(), static_cast<long>(goal->gesture_duration));
    }

    std::thread feedback_thread(&GestureExecutionSystem::publishElapsedFeedback, this, goal_handle, start_time);

    bool success = false;
    std::string error_message;
    try {
        success = executeGesture(gesture_type, goal->gesture_name, goal->gesture_duration,
                                  goal->bow_nod_angle, goal->location_x, goal->location_y, goal->location_z);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Gesture failed: %s", e.what());
        error_message = e.what();
    }

    feedback_stop_ = true;
    if (feedback_thread.joinable()) feedback_thread.join();

    result->success = success;
    result->actual_duration_seconds = static_cast<float>(nowSeconds() - start_time);
    result->message = success ? "completed" : (error_message.empty() ? "failed" : error_message);

    if (success) {
        RCLCPP_INFO(get_logger(), "Gesture completed in %.2fs", result->actual_duration_seconds);
        goal_handle->succeed(result);
    } else {
        RCLCPP_ERROR(get_logger(), "Gesture execution failed");
        goal_handle->abort(result);
    }
    executing_ = false;
}

void GestureExecutionSystem::publishElapsedFeedback(
    const std::shared_ptr<GoalHandleGesture> goal_handle, double start_time) {
    while (!feedback_stop_) {
        auto feedback = std::make_shared<GestureAction::Feedback>();
        feedback->elapsed_seconds = static_cast<float>(nowSeconds() - start_time);
        try {
            goal_handle->publish_feedback(feedback);
        } catch (const std::exception&) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// ── Gesture execution ────────────────────────────────────────────────────

bool GestureExecutionSystem::executeGesture(const std::string& gesture_type, const std::string& gesture_name,
                                            int64_t gesture_duration_ms, int64_t bow_nod_angle,
                                            double location_x, double location_y, double location_z) {
    gesture_duration_ms = std::max(MIN_GESTURE_DURATION_MS, std::min(gesture_duration_ms, MAX_GESTURE_DURATION_MS));

    if (gesture_type == "deictic") {
        return executeDeicticGesture(location_x, location_y, location_z, gesture_duration_ms);
    } else if (gesture_type == "iconic") {
        return executeIconicGesture(gesture_name, gesture_duration_ms);
    } else if (gesture_type == "symbolic") {
        RCLCPP_WARN(get_logger(), "Symbolic gestures not implemented yet");
        return false;
    } else if (gesture_type == "bow") {
        return executeBowingGesture(bow_nod_angle, gesture_duration_ms);
    } else if (gesture_type == "nod") {
        return executeNoddingGesture(bow_nod_angle, gesture_duration_ms);
    } else {
        return executeIconicGesture(gesture_type, gesture_duration_ms);
    }
}

bool GestureExecutionSystem::executeDeicticGesture(double point_x, double point_y, double point_z, int64_t duration_ms) {
    try {
        double robot_x = robot_pose_.x * 1000.0;
        double robot_y = robot_pose_.y * 1000.0;
        double robot_theta = robot_pose_.theta;

        double relative_x = (point_x * 1000.0) - robot_x;
        double relative_y = (point_y * 1000.0) - robot_y;
        double pointing_x = relative_x * std::cos(-robot_theta) - relative_y * std::sin(-robot_theta);
        double pointing_y = relative_y * std::cos(-robot_theta) + relative_x * std::sin(-robot_theta);
        double pointing_z = point_z * 1000.0;

        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "Pointing coordinates: (%.1f, %.1f, %.1f)", pointing_x, pointing_y, pointing_z);
        }

        if (pointing_x < 0.0) {
            RCLCPP_ERROR(get_logger(), "Target behind robot: x=%.1fmm", pointing_x);
            return false;
        }

        int pointing_arm = (pointing_y <= 0.0) ? pepper_kinematics::RIGHT_ARM : pepper_kinematics::LEFT_ARM;
        double shoulder_x = SHOULDER_OFFSET_X;
        double shoulder_y = (pointing_arm == pepper_kinematics::LEFT_ARM) ? SHOULDER_OFFSET_Y : -SHOULDER_OFFSET_Y;
        double shoulder_z = SHOULDER_OFFSET_Z + TORSO_HEIGHT;

        double distance = std::sqrt(std::pow(pointing_x - shoulder_x, 2) + std::pow(pointing_y - shoulder_y, 2) +
                                     std::pow(pointing_z - shoulder_z, 2));
        double l_2 = distance - UPPER_ARM_LENGTH;

        double elbow_x = ((UPPER_ARM_LENGTH * pointing_x) + (l_2 * shoulder_x)) / distance;
        double elbow_y = ((UPPER_ARM_LENGTH * pointing_y) + (l_2 * shoulder_y)) / distance;
        double elbow_z = ((UPPER_ARM_LENGTH * pointing_z) + (l_2 * shoulder_z)) / distance;
        elbow_z -= TORSO_HEIGHT;

        auto angles = pepper_kinematics::getArmShoulderAngles(pointing_arm, elbow_x, elbow_y, elbow_z);
        double shoulder_pitch = angles[0];
        double shoulder_roll = angles[1];

        if (std::isnan(shoulder_pitch) || std::isnan(shoulder_roll)) {
            RCLCPP_ERROR(get_logger(), "Invalid joint angles (unreachable target)");
            return false;
        }

        if (!checkJointLimits(pointing_arm, shoulder_pitch, shoulder_roll)) {
            return false;
        }

        publishDeicticVisualization(pointing_x, pointing_y, pointing_z, shoulder_x, shoulder_y, shoulder_z, pointing_arm);

        return executePointingMotion(pointing_arm, shoulder_pitch, shoulder_roll, duration_ms,
                                      pointing_x, pointing_y, pointing_z);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Deictic gesture failed: %s", e.what());
        return false;
    }
}

bool GestureExecutionSystem::checkJointLimits(int arm, double shoulder_pitch, double shoulder_roll) {
    if (arm == pepper_kinematics::RIGHT_ARM) {
        if (shoulder_pitch < MIN_RSHOULDER_PITCH || shoulder_pitch > MAX_RSHOULDER_PITCH ||
            shoulder_roll < MIN_RSHOULDER_ROLL || shoulder_roll > MAX_RSHOULDER_ROLL) {
            RCLCPP_ERROR(get_logger(), "Right arm limits exceeded: pitch=%.1f deg, roll=%.1f deg",
                pepper_kinematics::radiansToDegrees(shoulder_pitch),
                pepper_kinematics::radiansToDegrees(shoulder_roll));
            return false;
        }
    } else {
        if (shoulder_pitch < MIN_LSHOULDER_PITCH || shoulder_pitch > MAX_LSHOULDER_PITCH ||
            shoulder_roll < MIN_LSHOULDER_ROLL || shoulder_roll > MAX_LSHOULDER_ROLL) {
            RCLCPP_ERROR(get_logger(), "Left arm limits exceeded: pitch=%.1f deg, roll=%.1f deg",
                pepper_kinematics::radiansToDegrees(shoulder_pitch),
                pepper_kinematics::radiansToDegrees(shoulder_roll));
            return false;
        }
    }
    return true;
}

bool GestureExecutionSystem::executePointingMotion(int arm, double shoulder_pitch, double shoulder_roll,
                                                    int64_t duration_ms, double pointing_x, double pointing_y,
                                                    double pointing_z) {
    try {
        double duration_sec = duration_ms / 1000.0;
        auto head_angles = calculateHeadAnglesToTarget(pointing_x, pointing_y, pointing_z);
        double head_pitch = head_angles[0];
        double head_yaw = head_angles[1];

        std::vector<std::string> arm_joint_names;
        std::vector<double> pointing_angles;
        std::vector<double> home_position;
        std::string hand_joint_name;

        if (arm == pepper_kinematics::RIGHT_ARM) {
            arm_joint_names = {"RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"};
            pointing_angles = {shoulder_pitch, shoulder_roll, 2.0857, 0.0, -1.0};
            home_position = home_positions_.r_arm;
            hand_joint_name = "RHand";
        } else {
            arm_joint_names = {"LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll", "LWristYaw"};
            pointing_angles = {shoulder_pitch, shoulder_roll, -1.5620, -0.0, -1.0};
            home_position = home_positions_.l_arm;
            hand_joint_name = "LHand";
        }

        std::vector<std::string> head_joint_names = {"HeadPitch", "HeadYaw"};
        std::vector<double> head_home = home_positions_.head;
        std::vector<double> head_pointing = {head_pitch, head_yaw};
        std::vector<double> hand_home_position = {0.67};
        std::vector<double> hand_open_position = {1.0};

        std::vector<std::string> joint_names = arm_joint_names;
        joint_names.insert(joint_names.end(), head_joint_names.begin(), head_joint_names.end());
        joint_names.push_back(hand_joint_name);

        auto concatRow = [](std::vector<double> a, const std::vector<double>& b, const std::vector<double>& c) {
            a.insert(a.end(), b.begin(), b.end());
            a.insert(a.end(), c.begin(), c.end());
            return a;
        };
        std::vector<double> home_waypoint = concatRow(home_position, head_home, hand_home_position);
        std::vector<double> point_waypoint = concatRow(pointing_angles, head_pointing, hand_open_position);
        std::vector<double> return_waypoint = concatRow(home_position, head_home, hand_home_position);

        moveJointsBezier(joint_names, {home_waypoint, point_waypoint, return_waypoint}, duration_sec * 2, true);
        std::this_thread::sleep_for(std::chrono::duration<double>(duration_sec * 2));
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Pointing motion failed: %s", e.what());
        return false;
    }
}

std::array<double, 2> GestureExecutionSystem::calculateHeadAnglesToTarget(
    double target_x, double target_y, double target_z) {
    double distance_xy = std::sqrt(target_x * target_x + target_y * target_y);
    double head_yaw = std::atan2(target_y, target_x);
    const double HEAD_HEIGHT = 300.0;
    double head_pitch = -std::atan2(target_z - HEAD_HEIGHT, distance_xy);
    head_yaw = std::max(HEAD_YAW_MIN, std::min(head_yaw, HEAD_YAW_MAX));
    head_pitch = std::max(HEAD_PITCH_MIN, std::min(head_pitch, HEAD_PITCH_MAX));
    return {head_pitch, head_yaw};
}

bool GestureExecutionSystem::executeIconicGesture(const std::string& gesture_name, int64_t duration_ms) {
    try {
        auto it = gestures_.find(gesture_name);
        if (it == gestures_.end()) {
            RCLCPP_ERROR(get_logger(), "Gesture '%s' not found", gesture_name.c_str());
            return false;
        }
        const GestureDescriptor& gesture = it->second;
        if (gesture.arms.empty() || gesture.per_arm.empty()) {
            RCLCPP_ERROR(get_logger(), "Invalid gesture data");
            return false;
        }

        std::vector<std::string> all_joint_names;
        std::vector<float> all_joint_angles;
        std::vector<float> all_times;
        double max_duration = 0.0;

        for (const auto& arm_name : gesture.arms) {
            auto arm_it = gesture.per_arm.find(arm_name);
            if (arm_it == gesture.per_arm.end()) continue;
            const ArmWaypoints& aw = arm_it->second;

            std::vector<double> arm_times = aw.times;
            double arm_duration;
            if (!arm_times.empty()) {
                arm_duration = arm_times.back();
            } else {
                arm_duration = duration_ms / 1000.0;
                size_t num_waypoints = aw.waypoints.size();
                arm_times.resize(num_waypoints);
                for (size_t i = 0; i < num_waypoints; ++i) {
                    arm_times[i] = arm_duration * static_cast<double>(i + 1) / static_cast<double>(num_waypoints);
                }
            }

            max_duration = std::max(max_duration, arm_duration);
            all_joint_names.insert(all_joint_names.end(), aw.joint_names.begin(), aw.joint_names.end());

            size_t num_joints = aw.joint_names.size();
            size_t num_waypoints = aw.waypoints.size();
            for (size_t joint_idx = 0; joint_idx < num_joints; ++joint_idx) {
                for (size_t wp_idx = 0; wp_idx < num_waypoints; ++wp_idx) {
                    all_joint_angles.push_back(static_cast<float>(aw.waypoints[wp_idx][joint_idx]));
                    all_times.push_back(static_cast<float>(arm_times[wp_idx]));
                }
            }
        }

        if (all_joint_names.empty()) {
            RCLCPP_ERROR(get_logger(), "No valid joints to execute");
            return false;
        }

        publishJointTrajectory(all_joint_names, all_joint_angles, all_times, true);
        std::this_thread::sleep_for(std::chrono::duration<double>(max_duration));
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Iconic gesture failed: %s", e.what());
        return false;
    }
}

bool GestureExecutionSystem::executeBowingGesture(int64_t bow_angle, int64_t duration_ms) {
    try {
        double duration_sec = duration_ms / 1000.0;
        double bow_angle_clamped = std::max(MIN_BOW_ANGLE, std::min(static_cast<double>(bow_angle), MAX_BOW_ANGLE));
        double bow_angle_rad = -pepper_kinematics::degreesToRadians(bow_angle_clamped);
        std::vector<std::string> joint_names = {"HipPitch", "HipRoll", "KneePitch"};
        std::vector<double> home_position = home_positions_.leg;
        std::vector<double> bow_position = {bow_angle_rad, -0.00766, 0.03221};
        moveJointsBezier(joint_names, {home_position, bow_position, home_position}, duration_sec * 2, true);
        std::this_thread::sleep_for(std::chrono::duration<double>(duration_sec * 2));
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Bowing gesture failed: %s", e.what());
        return false;
    }
}

bool GestureExecutionSystem::executeNoddingGesture(int64_t nod_angle, int64_t duration_ms) {
    try {
        double duration_sec = duration_ms / 1000.0;
        double nod_angle_clamped = std::max(MIN_NOD_ANGLE, std::min(static_cast<double>(nod_angle), MAX_NOD_ANGLE));
        double nod_angle_rad = pepper_kinematics::degreesToRadians(nod_angle_clamped);
        std::vector<std::string> joint_names = {"HeadPitch", "HeadYaw"};
        std::vector<double> home_position = home_positions_.head;
        std::vector<double> nod_position = {nod_angle_rad, 0.012271};
        moveJointsBezier(joint_names, {home_position, nod_position, home_position}, duration_sec * 2, true);
        std::this_thread::sleep_for(std::chrono::duration<double>(duration_sec * 2));
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Nodding gesture failed: %s", e.what());
        return false;
    }
}

void GestureExecutionSystem::moveJointsBezier(const std::vector<std::string>& joint_names,
                                              const std::vector<std::vector<double>>& waypoints,
                                              double total_duration, bool use_bezier) {
    size_t num_joints = joint_names.size();
    size_t num_waypoints = waypoints.size();
    for (const auto& wp : waypoints) {
        if (wp.size() != num_joints) {
            RCLCPP_ERROR(get_logger(), "Waypoint dimension mismatch");
            return;
        }
    }

    std::vector<double> times_list(num_waypoints);
    for (size_t i = 0; i < num_waypoints; ++i) {
        times_list[i] = total_duration * static_cast<double>(i + 1) / static_cast<double>(num_waypoints);
    }

    std::vector<float> msg_joint_angles;
    std::vector<float> msg_times;
    for (size_t joint_idx = 0; joint_idx < num_joints; ++joint_idx) {
        for (size_t wp_idx = 0; wp_idx < num_waypoints; ++wp_idx) {
            msg_joint_angles.push_back(static_cast<float>(waypoints[wp_idx][joint_idx]));
            msg_times.push_back(static_cast<float>(times_list[wp_idx]));
        }
    }

    publishJointTrajectory(joint_names, msg_joint_angles, msg_times, use_bezier);
}

void GestureExecutionSystem::publishJointTrajectory(const std::vector<std::string>& joint_names,
                                                    const std::vector<float>& joint_angles,
                                                    const std::vector<float>& times, bool use_bezier) {
    naoqi_bridge_msgs::msg::JointAnglesTrajectory msg;
    msg.header.stamp = get_clock()->now();
    msg.joint_names = joint_names;
    msg.relative = 0;
    msg.use_bezier = use_bezier;
    msg.joint_angles = joint_angles;
    msg.times = times;
    joint_traj_pub_->publish(msg);

    if (verbose_mode_) {
        size_t num_joints = joint_names.size();
        size_t num_wp = num_joints > 0 ? joint_angles.size() / num_joints : 0;
        double total_time = times.empty() ? 0.0 : times.back();
        RCLCPP_INFO(get_logger(), "Trajectory: %zu joints, %zu waypoints over %.2fs",
            num_joints, num_wp, total_time);
    }
}

void GestureExecutionSystem::publishDeicticVisualization(double target_x, double target_y, double target_z,
                                                         double shoulder_x, double shoulder_y, double shoulder_z,
                                                         int arm) {
    try {
        auto stamp = get_clock()->now();

        auto makeMarker = [&](const std::string& ns, int mid, int32_t mtype) {
            visualization_msgs::msg::Marker m;
            m.header.stamp = stamp;
            m.header.frame_id = "base_link";
            m.ns = ns;
            m.id = mid;
            m.type = mtype;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.orientation.w = 1.0;
            m.lifetime.sec = 10;
            m.color.a = 1.0;
            return m;
        };

        // Target sphere
        auto tm = makeMarker("deictic_target", 0, visualization_msgs::msg::Marker::SPHERE);
        tm.pose.position.x = target_x / 1000.0;
        tm.pose.position.y = target_y / 1000.0;
        tm.pose.position.z = target_z / 1000.0;
        tm.scale.x = tm.scale.y = tm.scale.z = 0.1;
        tm.color.r = 1.0;
        marker_pub_->publish(tm);

        // Shoulder sphere
        auto sm = makeMarker("deictic_shoulder", 1, visualization_msgs::msg::Marker::SPHERE);
        sm.pose.position.x = shoulder_x / 1000.0;
        sm.pose.position.y = shoulder_y / 1000.0;
        sm.pose.position.z = shoulder_z / 1000.0;
        sm.scale.x = sm.scale.y = sm.scale.z = 0.06;
        if (arm == pepper_kinematics::LEFT_ARM) sm.color.g = 1.0; else sm.color.b = 1.0;
        marker_pub_->publish(sm);

        // Arrow
        auto lm = makeMarker("deictic_line", 2, visualization_msgs::msg::Marker::ARROW);
        lm.color.a = 0.8;
        lm.scale.x = 0.03;
        lm.scale.y = 0.06;
        lm.scale.z = 0.12;
        geometry_msgs::msg::Point start;
        start.x = shoulder_x / 1000.0;
        start.y = shoulder_y / 1000.0;
        start.z = shoulder_z / 1000.0;
        geometry_msgs::msg::Point end;
        end.x = target_x / 1000.0;
        end.y = target_y / 1000.0;
        end.z = target_z / 1000.0;
        lm.points = {start, end};
        if (arm == pepper_kinematics::LEFT_ARM) lm.color.g = 1.0; else lm.color.b = 1.0;
        marker_pub_->publish(lm);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Visualization failed: %s", e.what());
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::cout << "gesture_execution v1.0\n"
              << "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY." << std::endl;

    auto node = std::make_shared<GestureExecutionSystem>();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
