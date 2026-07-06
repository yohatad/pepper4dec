/* gesture_execution_interface.h
 *
 * Lifecycle node implementing Pepper's gesture execution action server,
 * providing deictic, iconic, bowing, and nodding gestures with
 * elapsed-time feedback. Loads gesture descriptors and topic mappings
 * from YAML, computes the joint trajectories needed to perform a
 * requested gesture (using inverse kinematics for deictic pointing via
 * pepper_kinematics_utilities), and streams the resulting joint-angle
 * trajectory to the robot. While a gesture is executing, it periodically
 * reports elapsed time back to the action client. Deictic gestures also
 * publish RViz markers showing the target point, shoulder position, and
 * pointing vector.
 *
 * Subscribers:
 *   /joint_states (sensor_msgs/JointState)
 *     Current joint positions, used to track the robot's arm/head/leg state.
 *   /robot_localization/pose (geometry_msgs/Pose2D)
 *     Current robot pose in the world frame, used to compute pointing direction.
 *
 * Publishers:
 *   /joint_angles_trajectory (naoqi_bridge_msgs/JointAnglesTrajectory)
 *     Joint angle trajectories sent to the robot to perform gestures.
 *   /gesture_execution/visualization (visualization_msgs/Marker)
 *     Markers visualizing deictic gesture targets, shoulder, and pointing arrow.
 *
 * Actions:
 *   /gesture_execution (dec_interfaces/action/Gesture)
 *     Executes a named or typed gesture (deictic, iconic, bow, nod) with
 *     feedback on elapsed time and a success/failure result.
 *
 * Parameters (loaded from gesture_execution_configuration.yaml):
 *   verboseMode (bool, default: true)
 *   gestureDescriptors, robotTopics are loaded but unused — the gesture
 *   and topic data file paths are fixed (data/gesture.yaml,
 *   data/pepper_topics.yaml), mirroring the reference Python ConfigManager.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: Jul 05, 2026
 * Version: v1.0 - C++ port of gesture_execution_implementation.py / gesture_execution_application.py
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#ifndef GESTURE_EXECUTION_INTERFACE_H
#define GESTURE_EXECUTION_INTERFACE_H

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <naoqi_bridge_msgs/msg/joint_angles_trajectory.hpp>
#include <dec_interfaces/action/gesture.hpp>

#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>

#include "gesture_execution/pepper_kinematics_utilities.h"

// ── Constants ─────────────────────────────────────────────────────────────────
constexpr int64_t MIN_GESTURE_DURATION_MS = 1000;
constexpr int64_t MAX_GESTURE_DURATION_MS = 5000;

constexpr double UPPER_ARM_LENGTH = 150.0;
constexpr double SHOULDER_OFFSET_X = -57.0;
constexpr double SHOULDER_OFFSET_Y = 149.74;
constexpr double SHOULDER_OFFSET_Z = 86.82;
constexpr double TORSO_HEIGHT = 0.0;

constexpr double MIN_RSHOULDER_PITCH = -2.0857;
constexpr double MAX_RSHOULDER_PITCH = 2.0857;
constexpr double MIN_RSHOULDER_ROLL = -1.5621;
constexpr double MAX_RSHOULDER_ROLL = -0.0087;
constexpr double MIN_LSHOULDER_PITCH = -2.0857;
constexpr double MAX_LSHOULDER_PITCH = 2.0857;
constexpr double MIN_LSHOULDER_ROLL = 0.0087;
constexpr double MAX_LSHOULDER_ROLL = 1.5621;

constexpr double MIN_BOW_ANGLE = 5.0;
constexpr double MAX_BOW_ANGLE = 45.0;
constexpr double MIN_NOD_ANGLE = 5.0;
constexpr double MAX_NOD_ANGLE = 30.0;

constexpr double DEFAULT_HEAD_PITCH = -0.2;
constexpr double DEFAULT_HEAD_YAW = 0.0;
constexpr double HEAD_YAW_MIN = -2.0857;
constexpr double HEAD_YAW_MAX = 2.0857;
constexpr double HEAD_PITCH_MIN = -0.7068;
constexpr double HEAD_PITCH_MAX = 0.6371;

struct RobotPose {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
};

// One gesture's per-arm waypoint data, as loaded from data/gesture.yaml.
struct ArmWaypoints {
    std::vector<std::string> joint_names;
    std::vector<std::vector<double>> waypoints;  // one entry per waypoint, one value per joint_name
    std::vector<double> times;                   // one entry per waypoint
};

// A single named gesture descriptor (e.g. "welcome", "wave", "shake").
struct GestureDescriptor {
    std::vector<std::string> arms;  // e.g. {"LArm", "RArm", "Leg"}
    std::unordered_map<std::string, ArmWaypoints> per_arm;
};

// Home-pose joint angles per limb, used both to seed joint_states_ and as the
// resting position for deictic/bow/nod gestures.
struct HomePositions {
    std::vector<double> r_arm{1.7410, -0.09664, 1.6981, 0.09664, -0.05679};
    std::vector<double> l_arm{1.7625, 0.09970, -1.7150, -0.1334, 0.06592};
    std::vector<double> head{-0.2, 0.0};
    std::vector<double> leg{0.0, 0.0, 0.0};
};

/**
 * @brief Load gesture descriptors from a gesture.yaml file.
 * @param yaml_path Absolute path to the gesture.yaml file.
 * @return Map of gesture name to its per-arm waypoint data. Empty if the
 *         file is missing or malformed.
 */
std::unordered_map<std::string, GestureDescriptor> loadGestureDescriptors(const std::string& yaml_path);

// Loads the robot topic mapping from data/pepper_topics.yaml (falls back to
// hardcoded defaults for any missing key, mirroring the Python ConfigManager).
struct RobotTopics {
    std::string joint_states = "/joint_states";
    std::string robot_pose = "/robot_localization/pose";
};

/**
 * @brief Load the robot topic mapping from a pepper_topics.yaml file.
 * @param yaml_path Absolute path to the pepper_topics.yaml file.
 * @return RobotTopics with any keys present in the file overriding the
 *         hardcoded defaults.
 */
RobotTopics loadRobotTopics(const std::string& yaml_path);

//=============================================================================
// GestureExecutionSystem
//
// Lifecycle state machine:
//   UNCONFIGURED → on_configure:  load gesture descriptors + topic mapping,
//                                 init kinematics/state, create the
//                                 lifecycle publishers and the action server.
//   INACTIVE     → on_activate:   activate publishers, subscribe to
//                                 joint_states and robot_pose.
//   ACTIVE       → on_deactivate: destroy the joint_states/robot_pose
//                                 subscriptions and deactivate publishers.
//   INACTIVE     → on_cleanup:    destroy the lifecycle publishers and the
//                                 action server.
//=============================================================================

class GestureExecutionSystem : public rclcpp_lifecycle::LifecycleNode {
public:
    using CallbackReturn =
        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
    using GestureAction = dec_interfaces::action::Gesture;
    using GoalHandleGesture = rclcpp_action::ServerGoalHandle<GestureAction>;

    GestureExecutionSystem();

    // ── Lifecycle callbacks ─────────────────────────────────────────────────
    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_shutdown  (const rclcpp_lifecycle::State& state) override;

private:
    void initJointStates();

    // ── Subscription callbacks ──────────────────────────────────────────────
    void jointStatesCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void robotPoseCallback(const geometry_msgs::msg::Pose2D::SharedPtr msg);

    // ── Action server callbacks ─────────────────────────────────────────────
    rclcpp_action::GoalResponse handleGoal(
        const rclcpp_action::GoalUUID& uuid, std::shared_ptr<const GestureAction::Goal> goal);
    rclcpp_action::CancelResponse handleCancel(const std::shared_ptr<GoalHandleGesture> goal_handle);
    void handleAccepted(const std::shared_ptr<GoalHandleGesture> goal_handle);
    void execute(const std::shared_ptr<GoalHandleGesture> goal_handle);
    void publishElapsedFeedback(const std::shared_ptr<GoalHandleGesture> goal_handle, double start_time);

    // ── Gesture execution ────────────────────────────────────────────────────
    bool executeGesture(const std::string& gesture_type, const std::string& gesture_name,
                        int64_t gesture_duration_ms, int64_t bow_nod_angle,
                        double location_x, double location_y, double location_z);
    bool executeDeicticGesture(double point_x, double point_y, double point_z, int64_t duration_ms);
    bool checkJointLimits(int arm, double shoulder_pitch, double shoulder_roll);
    bool executePointingMotion(int arm, double shoulder_pitch, double shoulder_roll, int64_t duration_ms,
                                double pointing_x, double pointing_y, double pointing_z);
    std::array<double, 2> calculateHeadAnglesToTarget(double target_x, double target_y, double target_z);
    bool executeIconicGesture(const std::string& gesture_name, int64_t duration_ms);
    bool executeBowingGesture(int64_t bow_angle, int64_t duration_ms);
    bool executeNoddingGesture(int64_t nod_angle, int64_t duration_ms);

    // Publishes a JointAnglesTrajectory directly from already-flattened,
    // joint-major joint_angles/times (one entry per (joint, waypoint) pair).
    void publishJointTrajectory(const std::vector<std::string>& joint_names,
                                const std::vector<float>& joint_angles,
                                const std::vector<float>& times, bool use_bezier);

    // Reshapes a uniform waypoint grid (same joint_names for every waypoint)
    // into the flattened joint-major layout, splitting total_duration evenly
    // across waypoints, then publishes it.
    void moveJointsBezier(const std::vector<std::string>& joint_names,
                          const std::vector<std::vector<double>>& waypoints,
                          double total_duration, bool use_bezier = true);

    void publishDeicticVisualization(double target_x, double target_y, double target_z,
                                     double shoulder_x, double shoulder_y, double shoulder_z, int arm);

    // ── Configuration ────────────────────────────────────────────────────────
    bool verbose_mode_ = false;
    std::unordered_map<std::string, GestureDescriptor> gestures_;
    RobotTopics topics_;
    HomePositions home_positions_;

    RobotPose robot_pose_;
    std::unordered_map<std::string, double> joint_states_;
    std::atomic<bool> executing_{false};
    std::atomic<bool> feedback_stop_{false};
    bool lifecycle_active_ = false;

    rclcpp_lifecycle::LifecyclePublisher<naoqi_bridge_msgs::msg::JointAnglesTrajectory>::SharedPtr joint_traj_pub_;
    rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr pose_sub_;
    rclcpp_action::Server<GestureAction>::SharedPtr action_server_;
};

#endif  // GESTURE_EXECUTION_INTERFACE_H
