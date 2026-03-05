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
// StopAnimateBehavior
// Service: animate_behavior/stop  (std_srvs::srv::Trigger)
//
// Calls the stop service on the animate_behavior node to immediately halt any
// ongoing animation. Returns SUCCESS if the service confirms the stop, FAILURE
// if the service reports an error or is unavailable.
//=============================================================================

BT::PortsList StopAnimateBehavior::providedPorts()
{
    return {
        BT::OutputPort<std::string>("message", "Response message from the stop service"),
    };
}

bool StopAnimateBehavior::setRequest(Request::SharedPtr& /*request*/)
{
    // std_srvs::srv::Trigger has an empty request — nothing to set
    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[StopAnimateBehavior] Sending stop request to animate_behavior/stop");
    }
    return true;
}

BT::NodeStatus StopAnimateBehavior::onResponseReceived(const Response::SharedPtr& response)
{
    setOutput("message", response->message);

    if (!response->success) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "[StopAnimateBehavior] Service returned failure: %s", response->message.c_str());
        return BT::NodeStatus::FAILURE;
    }

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[StopAnimateBehavior] Animation stopped: %s", response->message.c_str());
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus StopAnimateBehavior::onFailure(BT::ServiceNodeErrorCode error)
{
    RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                 "[StopAnimateBehavior] Service error: %s", toStr(error));
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
// Navigate
// Action: nav2_msgs::action::NavigateToPose  →  server: /navigate_to_pose
//
// Goal fields:
//   geometry_msgs/PoseStamped pose  – constructed from goal_x, goal_y, goal_theta
//   string                behavior_tree  – leave empty for Nav2 default BT
//
// Result fields:
//   (empty – success/failure determined by result.code)
//
// Feedback fields:
//   geometry_msgs/PoseStamped current_pose
//   float32                   distance_remaining
//   int16                     number_of_recoveries
//=============================================================================

BT::PortsList Navigate::providedPorts()
{
    return {
        BT::InputPort<double>      ("goal_x",                  "Goal x position (metres)"),
        BT::InputPort<double>      ("goal_y",                  "Goal y position (metres)"),
        BT::InputPort<double>      ("goal_theta",        0.0,  "Goal heading (radians)"),
        BT::InputPort<std::string> ("frame_id",          "map","Coordinate frame for the goal pose"),
        BT::OutputPort<float>      ("distance_remaining",      "Feedback: metres remaining to goal"),
        BT::OutputPort<int>        ("recoveries",              "Feedback: number of recovery attempts"),
    };
}

bool Navigate::setGoal(Goal& goal)
{
    auto goal_x     = getInput<double>("goal_x");
    auto goal_y     = getInput<double>("goal_y");
    auto goal_theta = getInput<double>("goal_theta");
    auto frame_id   = getInput<std::string>("frame_id");

    if (!goal_x || !goal_y) {
        RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                     "[Navigate] Missing required input port(s): goal_x, goal_y");
        return false;
    }

    const double x     = goal_x.value();
    const double y     = goal_y.value();
    const double theta = goal_theta ? goal_theta.value() : 0.0;

    goal.pose.header.frame_id = frame_id ? frame_id.value() : "map";
    goal.pose.header.stamp    = rclcpp::Clock().now();
    goal.pose.pose.position.x = x;
    goal.pose.pose.position.y = y;
    goal.pose.pose.position.z = 0.0;

    // Yaw (radians) → quaternion: roll=0, pitch=0
    goal.pose.pose.orientation.x = 0.0;
    goal.pose.pose.orientation.y = 0.0;
    goal.pose.pose.orientation.z = std::sin(theta / 2.0);
    goal.pose.pose.orientation.w = std::cos(theta / 2.0);

    // Empty string → Nav2 uses its default navigation BT
    goal.behavior_tree = "";

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[Navigate] Goal → frame=%s x=%.3f y=%.3f theta=%.3f rad",
                    goal.pose.header.frame_id.c_str(), x, y, theta);
    }
    return true;
}

BT::NodeStatus Navigate::onFeedback(const std::shared_ptr<const Feedback> feedback)
{
    setOutput("distance_remaining", feedback->distance_remaining);
    setOutput("recoveries",         static_cast<int>(feedback->number_of_recoveries));

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[Navigate] Feedback: %.2fm remaining, %d recoveries",
                    feedback->distance_remaining, feedback->number_of_recoveries);
    }
    return BT::NodeStatus::RUNNING;
}

BT::NodeStatus Navigate::onResultReceived(const WrappedResult& result)
{
    if (result.code != rclcpp_action::ResultCode::SUCCEEDED) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "[Navigate] Navigation did not succeed (code=%d)",
                    static_cast<int>(result.code));
        return BT::NodeStatus::FAILURE;
    }

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[Navigate] Navigation succeeded");
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus Navigate::onFailure(BT::ActionNodeErrorCode error)
{
    RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                 "[Navigate] Action error: %s", toStr(error));
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

//=============================================================================
// ConversationManagerNode
// Action: cssr_interfaces::action::ConversationManager
//
// Goal fields:
//   string prompt    – natural-language input to the conversation manager
//
// Result fields:
//   bool   success
//   string response  – the generated reply
//
// Feedback fields:
//   string status    – "searching" | "generating"
//=============================================================================

BT::PortsList ConversationManagerNode::providedPorts()
{
    return {
        BT::InputPort<std::string> ("prompt",   "", "Natural-language prompt to send"),
        BT::OutputPort<std::string>("response",     "Reply from the conversation manager"),
        BT::OutputPort<std::string>("status",       "Feedback: searching | generating"),
    };
}

bool ConversationManagerNode::setGoal(Goal& goal)
{
    auto prompt = getInput<std::string>("prompt");

    if (!prompt || prompt.value().empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                     "[ConversationManagerNode] Missing or empty 'prompt' input port");
        return false;
    }

    goal.prompt = prompt.value();

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[ConversationManagerNode] Goal → prompt=\"%s\"", goal.prompt.c_str());
    }
    return true;
}

BT::NodeStatus ConversationManagerNode::onFeedback(const std::shared_ptr<const Feedback> feedback)
{
    setOutput("status", feedback->status);

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[ConversationManagerNode] Feedback: %s", feedback->status.c_str());
    }
    return BT::NodeStatus::RUNNING;
}

BT::NodeStatus ConversationManagerNode::onResultReceived(const WrappedResult& result)
{
    setOutput("response", result.result->response);

    if (!result.result->success) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "[ConversationManagerNode] Action failed");
        return BT::NodeStatus::FAILURE;
    }

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[ConversationManagerNode] Response → \"%s\"", result.result->response.c_str());
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus ConversationManagerNode::onFailure(BT::ActionNodeErrorCode error)
{
    RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                 "[ConversationManagerNode] Action error: %s", toStr(error));
    return BT::NodeStatus::FAILURE;
}

//=============================================================================
// SpeechWithFeedbackNode
// Action: naoqi_bridge_msgs::action::SpeechWithFeedback
//
// Goal fields:
//   string say          – text to speak (supports \mrk=N\ bookmark syntax)
//
// Result fields:
//   bool success
//
// Feedback fields:
//   bool   started       – true once speech begins
//   int32  bookmark      – current bookmark ID (-1 if none)
//   string current_word  – word currently being spoken
//=============================================================================

BT::PortsList SpeechWithFeedbackNode::providedPorts()
{
    return {
        BT::InputPort<std::string> ("say",          "",    "Text to speak (supports \\mrk=N\\ bookmarks)"),
        BT::OutputPort<bool>       ("started",             "Feedback: true once speech begins"),
        BT::OutputPort<int>        ("bookmark",            "Feedback: current bookmark ID (-1 if none)"),
        BT::OutputPort<std::string>("current_word",        "Feedback: word currently being spoken"),
    };
}

bool SpeechWithFeedbackNode::setGoal(Goal& goal)
{
    auto say = getInput<std::string>("say");

    if (!say || say.value().empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                     "[SpeechWithFeedbackNode] Missing or empty 'say' input port");
        return false;
    }

    goal.say = say.value();

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[SpeechWithFeedbackNode] Goal → say=\"%s\"", goal.say.c_str());
    }
    return true;
}

BT::NodeStatus SpeechWithFeedbackNode::onFeedback(const std::shared_ptr<const Feedback> feedback)
{
    setOutput("started",      feedback->started);
    setOutput("bookmark",     static_cast<int>(feedback->bookmark));
    setOutput("current_word", feedback->current_word);

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[SpeechWithFeedbackNode] Feedback: started=%s bookmark=%d word=\"%s\"",
                    feedback->started ? "true" : "false",
                    feedback->bookmark,
                    feedback->current_word.c_str());
    }
    return BT::NodeStatus::RUNNING;
}

BT::NodeStatus SpeechWithFeedbackNode::onResultReceived(const WrappedResult& result)
{
    if (!result.result->success) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "[SpeechWithFeedbackNode] Speech action reported failure");
        return BT::NodeStatus::FAILURE;
    }

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[SpeechWithFeedbackNode] Speech completed successfully");
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus SpeechWithFeedbackNode::onFailure(BT::ActionNodeErrorCode error)
{
    RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                 "[SpeechWithFeedbackNode] Action error: %s", toStr(error));
    return BT::NodeStatus::FAILURE;
}

//=============================================================================
// SetOvertAttention
// Service: /attn/set_enabled  (std_srvs::srv::SetBool)
//
// Enables or disables the overt attention system.
//   enabled = true  → attention system starts tracking faces / saliency
//   enabled = false → attention system stops and returns head to default pose
//=============================================================================

BT::PortsList SetOvertAttention::providedPorts()
{
    return {
        BT::InputPort<bool>        ("enabled", true, "true = enable attention, false = disable"),
        BT::OutputPort<std::string>("message",       "Response message from the service"),
    };
}

bool SetOvertAttention::setRequest(Request::SharedPtr& request)
{
    auto enabled = getInput<bool>("enabled");

    if (!enabled) {
        RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                     "[SetOvertAttention] Missing required input port 'enabled'");
        return false;
    }

    request->data = enabled.value();

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[SetOvertAttention] Request → enabled=%s",
                    request->data ? "true" : "false");
    }
    return true;
}

BT::NodeStatus SetOvertAttention::onResponseReceived(const Response::SharedPtr& response)
{
    setOutput("message", response->message);

    if (!response->success) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "[SetOvertAttention] Service returned failure: %s", response->message.c_str());
        return BT::NodeStatus::FAILURE;
    }

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[SetOvertAttention] %s", response->message.c_str());
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus SetOvertAttention::onFailure(BT::ServiceNodeErrorCode error)
{
    RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                 "[SetOvertAttention] Service error: %s", toStr(error));
    return BT::NodeStatus::FAILURE;
}

//=============================================================================
// CheckFaceDetected
// Topic: /faceDetection/data  (cssr_interfaces::msg::FaceDetection)
//
// Blocks (returns RUNNING) until the face-detection condition is met, then
// returns SUCCESS. Runs indefinitely — never times out.
//
// Input ports:
//   bool require_mutual_gaze  – if true, succeed only when mutual gaze detected
//
// Output ports:
//   int    face_count   – number of faces in the latest message
//   bool   mutual_gaze  – true if any detected face has mutual gaze
//   string face_id      – label ID of the first detected face
//   double face_x       – centroid x of first face (pixels)
//   double face_y       – centroid y of first face (pixels)
//   double face_depth   – depth of first face (metres)
//=============================================================================

CheckFaceDetected::CheckFaceDetected(const std::string& name,
                                     const BT::NodeConfig& config,
                                     std::shared_ptr<rclcpp::Node> node)
    : BT::StatefulActionNode(name, config), node_(node)
{
    sub_ = node_->create_subscription<cssr_interfaces::msg::FaceDetection>(
        "/faceDetection/data", 10,
        [this](const cssr_interfaces::msg::FaceDetection::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(mutex_);
            latestMsg_ = msg;
        });
}

BT::PortsList CheckFaceDetected::providedPorts()
{
    return {
        BT::InputPort<bool>        ("require_mutual_gaze", true, "Succeed only when mutual gaze is detected"),
        BT::OutputPort<int>        ("face_count",                 "Number of faces in the latest message"),
        BT::OutputPort<bool>       ("mutual_gaze",                "True if any face has mutual gaze"),
        BT::OutputPort<std::string>("face_id",                    "Label ID of the first detected face"),
        BT::OutputPort<double>     ("face_x",                     "Centroid x of first face (pixels)"),
        BT::OutputPort<double>     ("face_y",                     "Centroid y of first face (pixels)"),
        BT::OutputPort<double>     ("face_depth",                 "Depth of first face (metres)"),
    };
}

// Shared evaluation logic used by both onStart and onRunning.
// Returns SUCCESS when conditions are met, RUNNING while still waiting.
BT::NodeStatus CheckFaceDetected::checkLatestMessage()
{
    cssr_interfaces::msg::FaceDetection::SharedPtr msg;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        msg = latestMsg_;
    }

    if (!msg) {
        return BT::NodeStatus::RUNNING;
    }

    const int faceCount = static_cast<int>(msg->face_label_id.size());
    bool anyMutualGaze = false;
    for (bool g : msg->mutual_gaze) {
        anyMutualGaze = anyMutualGaze || g;
    }

    setOutput("face_count",  faceCount);
    setOutput("mutual_gaze", anyMutualGaze);

    if (faceCount > 0) {
        setOutput("face_id",    msg->face_label_id[0]);
        setOutput("face_x",     static_cast<double>(msg->centroids[0].x));
        setOutput("face_y",     static_cast<double>(msg->centroids[0].y));
        setOutput("face_depth", static_cast<double>(msg->centroids[0].z));
    }

    if (faceCount == 0) {
        return BT::NodeStatus::RUNNING;
    }

    const bool requireGaze = getInput<bool>("require_mutual_gaze").value_or(false);
    if (requireGaze && !anyMutualGaze) {
        if (ConfigManager::instance().isVerbose()) {
            RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                        "[CheckFaceDetected] %d face(s) detected, waiting for mutual gaze...", faceCount);
        }
        return BT::NodeStatus::RUNNING;
    }

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[CheckFaceDetected] SUCCESS – %d face(s), mutual_gaze=%s, id=%s depth=%.2fm",
                    faceCount, anyMutualGaze ? "true" : "false",
                    msg->face_label_id[0].c_str(),
                    static_cast<double>(msg->centroids[0].z));
    }
    return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus CheckFaceDetected::onStart()
{
    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[CheckFaceDetected] Started – waiting for face detection data...");
    }
    return checkLatestMessage();
}

BT::NodeStatus CheckFaceDetected::onRunning()
{
    return checkLatestMessage();
}

void CheckFaceDetected::onHalted()
{
    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                    "[CheckFaceDetected] Halted");
    }
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

    factory.registerNodeType<AnimateBehaviorNode>      ("AnimateBehavior",      params);
    factory.registerNodeType<StopAnimateBehavior>      ("StopAnimateBehavior",  params);
    factory.registerNodeType<SetOvertAttention>        ("SetOvertAttention",    params);
    factory.registerNodeType<GestureNode>              ("Gesture",              params);
    factory.registerNodeType<Navigate>                 ("Navigate",             params);
    factory.registerNodeType<SpeechRecognitionNode>    ("SpeechRecognition",    params);
    factory.registerNodeType<ConversationManagerNode>  ("ConversationManager",  params);
    factory.registerNodeType<SpeechWithFeedbackNode>   ("SpeechWithFeedback",   params);

    // CheckFaceDetected requires the node handle at construction; use registerBuilder
    factory.registerBuilder<CheckFaceDetected>(
        "CheckFaceDetected",
        [node_handle](const std::string& name, const BT::NodeConfig& config) {
            return std::make_unique<CheckFaceDetected>(name, config, node_handle);
        });

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(logger, "[initializeTree] Registered nodes: AnimateBehavior, StopAnimateBehavior, SetOvertAttention, Gesture, Navigate, SpeechRecognition, ConversationManager, SpeechWithFeedback, CheckFaceDetected");
        RCLCPP_INFO(logger, "[initializeTree] Loading tree: %s", xmlPath.c_str());
    }

    BT::Tree tree = factory.createTreeFromFile(xmlPath);

    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(logger, "[initializeTree] Tree loaded successfully");
    }

    return tree;
}

} // namespace behavior_controller
