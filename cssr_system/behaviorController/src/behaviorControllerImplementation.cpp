/* behaviorControllerImplementation.cpp
 * Author: Yohannes Tadesse Haile
 * Date: February 09, 2026
 * Version: v1.0
 */

#include "behaviorController/behaviorControllerInterface.h"
#include <behaviortree_ros2/bt_service_node.hpp>
#include <behaviortree_ros2/bt_action_node.hpp>
#include <behaviortree_ros2/ros_node_params.hpp>
#include <behaviortree_ros2/plugins.hpp>

using namespace BT;

//=============================================================================
// AnimateBehaviorNode
// Action: cssr_interfaces::action::AnimateBehavior
//
// Goal fields:
//   string  behavior_type     – "All", "body", "hands", "rotation"
//   float32 selected_range    – 0.0 to 1.0
//   int32   duration_seconds  – 0 = run until cancelled
//
// Result fields:
//   bool    success
//   string  message
//   float32 total_duration
//
// Feedback fields:
//   string  current_limb
//   int32   gestures_completed
//   float32 elapsed_time
//   bool    is_running
//=============================================================================

BT::PortsList AnimateBehaviorNode::providedPorts()
{
    return {
        BT::InputPort<std::string> ("behavior_type",       "All", "all | body | hands | rotation"),
        BT::InputPort<float>       ("selected_range",       0.5f, "Movement range [0.0, 1.0]"),
        BT::InputPort<int>         ("duration_seconds",     0,    "Duration in seconds (0 = indefinite)"),
        BT::OutputPort<std::string>("message",                    "Result message from action server"),
        BT::OutputPort<std::string>("current_limb",               "Feedback: limb currently animating"),
        BT::OutputPort<int>        ("gestures_completed",         "Feedback: number of gestures completed"),
        BT::OutputPort<float>      ("elapsed_time",               "Feedback: elapsed time in seconds"),
        BT::OutputPort<bool>       ("is_running",                 "Feedback: whether animation is active"),
    };
}

bool AnimateBehaviorNode::setGoal(Goal& goal)
{
    auto behavior_type    = getInput<std::string>("behavior_type");
    auto selected_range   = getInput<float>("selected_range");
    auto duration_seconds = getInput<int>("duration_seconds");

    if (!behavior_type || !selected_range || !duration_seconds) {
        RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                     "[AnimateBehaviorNode] Missing required input port(s)");
        return false;
    }

    goal.behavior_type    = behavior_type.value();
    goal.selected_range   = selected_range.value();
    goal.duration_seconds = duration_seconds.value();

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[AnimateBehaviorNode] Goal → type=%s range=%.2f duration=%ds",
                    goal.behavior_type.c_str(), goal.selected_range, goal.duration_seconds);
    }
    return true;
}

BT::NodeStatus AnimateBehaviorNode::onFeedback(const std::shared_ptr<const Feedback> feedback)
{
    setOutput("current_limb",        feedback->current_limb);
    setOutput("gestures_completed",  feedback->gestures_completed);
    setOutput("elapsed_time",        feedback->elapsed_time);
    setOutput("is_running",          feedback->is_running);

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[AnimateBehaviorNode] Feedback: limb=%s completed=%d elapsed=%.2fs running=%s",
                    feedback->current_limb.c_str(), feedback->gestures_completed,
                    feedback->elapsed_time, feedback->is_running ? "true" : "false");
    }
    return BT::NodeStatus::RUNNING;
}

BT::NodeStatus AnimateBehaviorNode::onResultReceived(const WrappedResult& result)
{
    setOutput("message", result.result->message);

    if (!result.result->success) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "[AnimateBehaviorNode] Action failed: %s", result.result->message.c_str());
        return BT::NodeStatus::FAILURE;
    }

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[AnimateBehaviorNode] Succeeded in %.2fs: %s",
                    result.result->total_duration, result.result->message.c_str());
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus AnimateBehaviorNode::onFailure(BT::ActionNodeErrorCode error)
{
    RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                 "[AnimateBehaviorNode] Action error: %s", toStr(error));
    return BT::NodeStatus::FAILURE;
}

//=============================================================================
// GestureNode
// Action: cssr_interfaces::action::Gesture
//
// Goal fields:
//   string  gesture_type
//   int64   gesture_id
//   int64   gesture_duration
//   int64   bow_nod_angle
//   float64 location_x / y / z
//
// Result fields:
//   bool    success
//   string  message
//   float32 actual_duration_seconds
//
// Feedback fields:
//   float32 elapsed_seconds
//=============================================================================

BT::PortsList GestureNode::providedPorts()
{
    return {
        BT::InputPort<std::string> ("gesture_type",     "",  "Gesture type (e.g. wave, point)"),
        BT::InputPort<int64_t>     ("gesture_id",       0,   "Gesture ID"),
        BT::InputPort<int64_t>     ("gesture_duration", 0,   "Duration in ms"),
        BT::InputPort<int64_t>     ("bow_nod_angle",    0,   "Bow/nod angle in degrees"),
        BT::InputPort<double>      ("location_x",       0.0, "Target x (metres)"),
        BT::InputPort<double>      ("location_y",       0.0, "Target y (metres)"),
        BT::InputPort<double>      ("location_z",       0.0, "Target z (metres)"),
        BT::OutputPort<std::string>("message",               "Result message from action server"),
        BT::OutputPort<float>      ("elapsed_seconds",       "Feedback: elapsed time in seconds"),
    };
}

bool GestureNode::setGoal(Goal& goal)
{
    auto gesture_type     = getInput<std::string>("gesture_type");
    auto gesture_id       = getInput<int64_t>("gesture_id");
    auto gesture_duration = getInput<int64_t>("gesture_duration");
    auto bow_nod_angle    = getInput<int64_t>("bow_nod_angle");
    auto location_x       = getInput<double>("location_x");
    auto location_y       = getInput<double>("location_y");
    auto location_z       = getInput<double>("location_z");

    if (!gesture_type) {
        RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                     "[GestureNode] Missing required input port 'gesture_type'");
        return false;
    }

    goal.gesture_type     = gesture_type.value();
    goal.gesture_id       = gesture_id       ? gesture_id.value()       : 0;
    goal.gesture_duration = gesture_duration ? gesture_duration.value() : 0;
    goal.bow_nod_angle    = bow_nod_angle    ? bow_nod_angle.value()    : 0;
    goal.location_x       = location_x       ? location_x.value()       : 0.0;
    goal.location_y       = location_y       ? location_y.value()       : 0.0;
    goal.location_z       = location_z       ? location_z.value()       : 0.0;

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[GestureNode] Goal → type=%s id=%ld duration=%ldms target=(%.2f,%.2f,%.2f)",
                    goal.gesture_type.c_str(), goal.gesture_id, goal.gesture_duration,
                    goal.location_x, goal.location_y, goal.location_z);
    }
    return true;
}

BT::NodeStatus GestureNode::onFeedback(const std::shared_ptr<const Feedback> feedback)
{
    setOutput("elapsed_seconds", feedback->elapsed_seconds);

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[GestureNode] Feedback: elapsed=%.2fs", feedback->elapsed_seconds);
    }
    return BT::NodeStatus::RUNNING;
}

BT::NodeStatus GestureNode::onResultReceived(const WrappedResult& result)
{
    setOutput("message", result.result->message);

    if (!result.result->success) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "[GestureNode] Action failed: %s", result.result->message.c_str());
        return BT::NodeStatus::FAILURE;
    }

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[GestureNode] Succeeded in %.2fs: %s",
                    result.result->actual_duration_seconds, result.result->message.c_str());
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus GestureNode::onFailure(BT::ActionNodeErrorCode error)
{
    RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                 "[GestureNode] Action error: %s", toStr(error));
    return BT::NodeStatus::FAILURE;
}

//=============================================================================
// NavigateNode
// Action: cssr_interfaces::action::Navigation
//
// Goal fields:
//   float64 goal_x
//   float64 goal_y
//   float64 goal_theta
//
// Result fields:
//   bool navigation_goal_success
//
// Feedback fields:
//   int8 progress
//=============================================================================

BT::PortsList NavigateNode::providedPorts()
{
    return {
        BT::InputPort<double>("goal_x",     0.0, "Goal x position (metres)"),
        BT::InputPort<double>("goal_y",     0.0, "Goal y position (metres)"),
        BT::InputPort<double>("goal_theta", 0.0, "Goal heading (degrees)"),
    };
}

bool NavigateNode::setGoal(Goal& goal)
{
    auto goal_x     = getInput<double>("goal_x");
    auto goal_y     = getInput<double>("goal_y");
    auto goal_theta = getInput<double>("goal_theta");

    if (!goal_x || !goal_y || !goal_theta) {
        RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                     "[NavigateNode] Missing required input port(s)");
        return false;
    }

    goal.goal_x     = goal_x.value();
    goal.goal_y     = goal_y.value();
    goal.goal_theta = goal_theta.value();

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[NavigateNode] Goal → (%.3f, %.3f) θ=%.1f°",
                    goal.goal_x, goal.goal_y, goal.goal_theta);
    }
    return true;
}

BT::NodeStatus NavigateNode::onResultReceived(const WrappedResult& result)
{
    if (!result.result->navigation_goal_success) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "[NavigateNode] Navigation failed");
        return BT::NodeStatus::FAILURE;
    }

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[NavigateNode] Navigation succeeded");
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus NavigateNode::onFailure(BT::ActionNodeErrorCode error)
{
    RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                 "[NavigateNode] Action error: %s", toStr(error));
    return BT::NodeStatus::FAILURE;
}

//=============================================================================
// SpeechRecognitionNode
// Action: cssr_interfaces::action::SpeechRecognition
//
// Goal fields:
//   float32 wait          – seconds to wait for speech input
//
// Result fields:
//   string  transcription – recognised text
//
// Feedback fields:
//   string status – "waiting" | "speech" | "transcribing"
//=============================================================================

BT::PortsList SpeechRecognitionNode::providedPorts()
{
    return {
        BT::InputPort<float>       ("wait",          2.0f, "Seconds to wait for speech input"),
        BT::OutputPort<std::string>("transcription",       "Recognised speech text"),
        BT::OutputPort<std::string>("status",              "Feedback update from action server (waiting/speech/transcribing)"),
    };
}

bool SpeechRecognitionNode::setGoal(Goal& goal)
{
    auto wait = getInput<float>("wait");

    if (!wait) {
        RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                     "[SpeechRecognitionNode] Missing required input port 'wait'");
        return false;
    }

    goal.wait = wait.value();

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[SpeechRecognitionNode] Goal → wait=%.1fs", goal.wait);
    }
    return true;
}

BT::NodeStatus SpeechRecognitionNode::onFeedback(const std::shared_ptr<const Feedback> feedback)
{
    setOutput("status", feedback->status);

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[SpeechRecognitionNode] Feedback: %s", feedback->status.c_str());
    }
    return BT::NodeStatus::RUNNING;
}

BT::NodeStatus SpeechRecognitionNode::onResultReceived(const WrappedResult& result)
{
    if (result.result->transcription.empty()) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "[SpeechRecognitionNode] Empty transcription received");
        return BT::NodeStatus::FAILURE;
    }

    setOutput("transcription", result.result->transcription);

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[SpeechRecognitionNode] Transcription: \"%s\"",
                    result.result->transcription.c_str());
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus SpeechRecognitionNode::onFailure(BT::ActionNodeErrorCode error)
{
    RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                 "[SpeechRecognitionNode] Action error: %s", toStr(error));
    return BT::NodeStatus::FAILURE;
}

namespace behavior_controller {

BT::Tree initializeTree(const std::string& scenario,
                        std::shared_ptr<rclcpp::Node> node_handle)
{
    auto logger = node_handle->get_logger();

    // Locate the XML tree file
    std::string packagePath =
        ament_index_cpp::get_package_share_directory("behavior_controller");
    std::string xmlPath = packagePath + "/data/" + scenario + ".xml";

    if (!fileExists(xmlPath)) {
        throw std::runtime_error("Behavior tree XML not found: " + xmlPath);
    }

    // Build RosNodeParams shared by all RosActionNode wrappers
    BT::RosNodeParams params;
    params.nh              = node_handle;
    params.server_timeout  = std::chrono::milliseconds(5000);
    params.wait_for_server_timeout = std::chrono::milliseconds(10000);

    // Create factory and register nodes
    BT::BehaviorTreeFactory factory;

    factory.registerNodeType<AnimateBehaviorNode>   ("AnimateBehavior",   params);
    factory.registerNodeType<GestureNode>           ("Gesture",           params);
    factory.registerNodeType<NavigateNode>          ("Navigate",          params);
    factory.registerNodeType<SpeechRecognitionNode> ("SpeechRecognition", params);

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(logger, "[initializeTree] Registered nodes: AnimateBehavior, Gesture, Navigate, SpeechRecognition");
        RCLCPP_INFO(logger, "[initializeTree] Loading tree: %s", xmlPath.c_str());
    }

    BT::Tree tree = factory.createTreeFromFile(xmlPath);

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(logger, "[initializeTree] Tree loaded successfully");
    }

    return tree;
}

} // namespace behavior_controller
