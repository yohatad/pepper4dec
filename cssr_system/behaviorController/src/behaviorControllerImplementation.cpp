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
// RosActionNode wrapper implementations (for Actions)
//=============================================================================

// ——————————————
// TTSRosAction - Uses cssr_interfaces/action/TTS
// ——————————————
class TTSRosAction : public RosActionNode<cssr_interfaces::action::TTS>
{
public:
    TTSRosAction(const std::string& name, 
                 const NodeConfig& conf,
                 const RosNodeParams& params)
        : RosActionNode<cssr_interfaces::action::TTS>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("action_name", "/tts", "Action server name"),
            InputPort<std::string>("phrase_key", "Key for utility phrase"),
            InputPort<std::string>("text", "Direct text to speak"),
            InputPort<std::string>("language", "Language code")
        });
    }

    bool setGoal(Goal& goal) override
    {
        // Try direct text first
        std::string text;
        if (getInput("text", text)) {
            goal.text = text;
            return true;
        }
        
        // Otherwise get from phrase key
        std::string phraseKey;
        if (!getInput("phrase_key", phraseKey)) {
            phraseKey = name();
        }
        
        std::string language;
        if (!getInput("language", language)) {
            language = ConfigManager::instance().getLanguage();
        }
        
        try {
            goal.text = KnowledgeManager::instance().getUtilityPhrase(phraseKey, language);
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(logger(), "Failed to get utility phrase: %s", e.what());
            return false;
        }
    }

    NodeStatus onResultReceived(const WrappedResult& result) override
    {
        if (result.result->success) {
            RCLCPP_INFO(logger(), "TTS succeeded: %s", result.result->message.c_str());
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "TTS failed: %s", result.result->message.c_str());
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ActionNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "TTS action error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }

    NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override
    {
        RCLCPP_DEBUG(logger(), "TTS status: %s", feedback->status.c_str());
        return NodeStatus::RUNNING;
    }
};

// ——————————————
// NavigateRosAction - Uses cssr_interfaces/action/Navigation
// ——————————————
class NavigateRosAction : public RosActionNode<cssr_interfaces::action::Navigation>
{
public:
    NavigateRosAction(const std::string& name, 
                      const NodeConfig& conf,
                      const RosNodeParams& params)
        : RosActionNode<cssr_interfaces::action::Navigation>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("action_name", "/navigation", "Action server name"),
            InputPort<double>("goal_x", "Goal X coordinate"),
            InputPort<double>("goal_y", "Goal Y coordinate"),
            InputPort<double>("goal_theta", "Goal theta orientation")
        });
    }

    bool setGoal(Goal& goal) override
    {
        RobotPose location;
        
        // Try input ports first
        if (getInput("goal_x", location.x) &&
            getInput("goal_y", location.y) &&
            getInput("goal_theta", location.theta)) 
        {
            goal.goal_x = location.x;
            goal.goal_y = location.y;
            goal.goal_theta = location.theta;
            return true;
        }
        
        // Fall back to blackboard
        if (config().blackboard->get("exhibitLocation", location)) {
            goal.goal_x = location.x;
            goal.goal_y = location.y;
            goal.goal_theta = location.theta;
            return true;
        }
        
        RCLCPP_ERROR(logger(), "Cannot get navigation goal from ports or blackboard");
        return false;
    }

    NodeStatus onResultReceived(const WrappedResult& result) override
    {
        if (result.result->navigation_goal_success) {
            RCLCPP_INFO(logger(), "Navigation succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "Navigation failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ActionNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "Navigation action error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }

    NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override
    {
        RCLCPP_DEBUG(logger(), "Navigation progress: %d", feedback->progress);
        return NodeStatus::RUNNING;
    }
};

// ——————————————
// GestureRosAction - Uses cssr_interfaces/action/Gesture
// ——————————————
class GestureRosAction : public RosActionNode<cssr_interfaces::action::Gesture>
{
public:
    GestureRosAction(const std::string& name, 
                     const NodeConfig& conf,
                     const RosNodeParams& params)
        : RosActionNode<cssr_interfaces::action::Gesture>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("action_name", "/gesture", "Action server name"),
            InputPort<std::string>("gesture_type", "deictic", "Gesture type"),
            InputPort<int>("gesture_id", Constants::DEICTIC_GESTURE_ID, "Gesture ID"),
            InputPort<int>("gesture_duration", Constants::GESTURE_DURATION_MS, "Duration in ms"),
            InputPort<int>("bow_nod_angle", 0, "Bow/nod angle"),
            InputPort<double>("location_x", "Target X coordinate"),
            InputPort<double>("location_y", "Target Y coordinate"),
            InputPort<double>("location_z", "Target Z coordinate")
        });
    }

    bool setGoal(Goal& goal) override
    {
        // Set defaults
        goal.gesture_type = "deictic";
        goal.gesture_id = Constants::DEICTIC_GESTURE_ID;
        goal.gesture_duration = Constants::GESTURE_DURATION_MS;
        goal.bow_nod_angle = 0;
        
        // Override from ports
        getInput("gesture_type", goal.gesture_type);
        getInput("gesture_id", goal.gesture_id);
        getInput("gesture_duration", goal.gesture_duration);
        getInput("bow_nod_angle", goal.bow_nod_angle);
        
        // Get target location
        Position3D target;
        if (getInput("location_x", target.x) &&
            getInput("location_y", target.y) &&
            getInput("location_z", target.z)) 
        {
            goal.location_x = target.x;
            goal.location_y = target.y;
            goal.location_z = target.z;
            return true;
        }
        
        // Try blackboard
        if (config().blackboard->get("exhibitGestureTarget", target)) {
            goal.location_x = target.x;
            goal.location_y = target.y;
            goal.location_z = target.z;
            return true;
        }
        
        RCLCPP_ERROR(logger(), "Cannot get gesture target from ports or blackboard");
        return false;
    }

    NodeStatus onResultReceived(const WrappedResult& result) override
    {
        if (result.result->success) {
            RCLCPP_INFO(logger(), "Gesture succeeded: %s (%.2fs)", 
                       result.result->message.c_str(),
                       result.result->actual_duration_seconds);
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "Gesture failed: %s", result.result->message.c_str());
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ActionNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "Gesture action error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }

    NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override
    {
        RCLCPP_DEBUG(logger(), "Gesture elapsed: %.2fs", feedback->elapsed_seconds);
        return NodeStatus::RUNNING;
    }
};

// ——————————————
// SpeechRecognitionRosAction - Uses cssr_interfaces/action/SpeechRecognition
// ——————————————
class SpeechRecognitionRosAction : public RosActionNode<cssr_interfaces::action::SpeechRecognition>
{
public:
    SpeechRecognitionRosAction(const std::string& name, 
                               const NodeConfig& conf,
                               const RosNodeParams& params)
        : RosActionNode<cssr_interfaces::action::SpeechRecognition>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("action_name", "/speech_recognition", "Action server name"),
            InputPort<float>("wait", 5.0, "Wait time in seconds")
        });
    }

    bool setGoal(Goal& goal) override
    {
        float wait = 5.0;
        getInput("wait", wait);
        goal.wait = wait;
        return true;
    }

    NodeStatus onResultReceived(const WrappedResult& result) override
    {
        if (!result.result->transcription.empty()) {
            RCLCPP_INFO(logger(), "Speech recognized: %s", result.result->transcription.c_str());
            
            // Store transcription in blackboard for processing
            config().blackboard->set("visitorSpeech", result.result->transcription);
            
            // Simple yes/no detection (you can make this more sophisticated)
            std::string lower = result.result->transcription;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            
            if (lower.find("yes") != std::string::npos || lower.find("yeah") != std::string::npos) {
                config().blackboard->set("visitorResponse", "yes");
            } else {
                config().blackboard->set("visitorResponse", "no");
            }
            
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "No speech recognized");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ActionNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "Speech recognition action error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }

    NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override
    {
        RCLCPP_DEBUG(logger(), "Speech recognition progress: %d", feedback->progress);
        return NodeStatus::RUNNING;
    }
};

// ——————————————
// AnimateBehaviorRosAction - Uses cssr_interfaces/action/AnimateBehavior
// ——————————————
// class AnimateBehaviorRosAction : public RosActionNode<cssr_interfaces::action::AnimateBehavior>
// {
// public:
//     AnimateBehaviorRosAction(const std::string& name, 
//                              const NodeConfig& conf,
//                              const RosNodeParams& params)
//         : RosActionNode<cssr_interfaces::action::AnimateBehavior>(name, conf, params)
//     {}

//     static PortsList providedPorts()
//     {
//         return providedBasicPorts({
//             InputPort<std::string>("action_name", "/animate_behavior", "Action server name"),
//             InputPort<std::string>("behavior_type", "All", "Behavior type"),
//             InputPort<float>("selected_range", 0.5, "Movement range 0-1"),
//             InputPort<int>("duration_seconds", 0, "Duration (0=infinite)")
//         });
//     }

//     bool setGoal(Goal& goal) override
//     {
//         goal.behavior_type = "All";
//         goal.selected_range = 0.5;
//         goal.duration_seconds = 0;
        
//         getInput("behavior_type", goal.behavior_type);
//         getInput("selected_range", goal.selected_range);
//         getInput("duration_seconds", goal.duration_seconds);
        
//         return true;
//     }

//     NodeStatus onResultReceived(const WrappedResult& result) override
//     {
//         if (result.result->success) {
//             RCLCPP_INFO(logger(), "AnimateBehavior succeeded: %s (%.2fs)", 
//                        result.result->message.c_str(),
//                        result.result->total_duration);
//             return NodeStatus::SUCCESS;
//         }
//         RCLCPP_WARN(logger(), "AnimateBehavior failed: %s", result.result->message.c_str());
//         return NodeStatus::FAILURE;
//     }

//     NodeStatus onFailure(ActionNodeErrorCode error) override
//     {
//         RCLCPP_ERROR(logger(), "AnimateBehavior action error: %d", static_cast<int>(error));
//         return NodeStatus::FAILURE;
//     }

//     NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override
//     {
//         RCLCPP_DEBUG(logger(), "AnimateBehavior: %s, gestures: %d, elapsed: %.2fs", 
//                     feedback->current_limb.c_str(),
//                     feedback->gestures_completed,
//                     feedback->elapsed_time);
//         return NodeStatus::RUNNING;
//     }
// };

//=============================================================================
// RosServiceNode wrapper implementations
//=============================================================================

// ——————————————
// SetOvertAttentionModeRosService - Uses cssr_interfaces/srv/OvertAttentionSetMode
// ——————————————
class SetOvertAttentionModeRosService : public RosServiceNode<cssr_interfaces::srv::OvertAttentionSetMode>
{
public:
    SetOvertAttentionModeRosService(const std::string& name, 
                                    const NodeConfig& conf,
                                    const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::OvertAttentionSetMode>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/overtAttention/set_mode", "Service name"),
            InputPort<std::string>("state", "Attention state"),
            InputPort<float>("location_x", "Location X"),
            InputPort<float>("location_y", "Location Y"),
            InputPort<float>("location_z", "Location Z")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string state;
        if (!getInput("state", state)) {
            state = name();
        }
        request->state = state;
        
        request->location_x = 0.0f;
        request->location_y = 0.0f;
        request->location_z = 0.0f;
        
        if (state == "location") {
            Position3D target;
            
            if (getInput("location_x", target.x) &&
                getInput("location_y", target.y) &&
                getInput("location_z", target.z)) 
            {
                request->location_x = static_cast<float>(target.x);
                request->location_y = static_cast<float>(target.y);
                request->location_z = static_cast<float>(target.z);
                return true;
            }
            
            if (config().blackboard->get("exhibitGestureTarget", target)) {
                request->location_x = static_cast<float>(target.x);
                request->location_y = static_cast<float>(target.y);
                request->location_z = static_cast<float>(target.z);
                return true;
            }
            
            RCLCPP_ERROR(logger(), "Cannot get location for SetOvertAttentionMode");
            return false;
        }
        
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->mode_set_success) {
            RCLCPP_INFO(logger(), "SetOvertAttentionMode succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "SetOvertAttentionMode failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "SetOvertAttentionMode service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// SetAnimateBehaviorRosService - Uses cssr_interfaces/srv/AnimateBehaviorSetActivation
// ——————————————
class SetAnimateBehaviorRosService : public RosServiceNode<cssr_interfaces::srv::AnimateBehaviorSetActivation>
{
public:
    SetAnimateBehaviorRosService(const std::string& name, 
                                 const NodeConfig& conf,
                                 const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::AnimateBehaviorSetActivation>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/animateBehaviour/setActivation", "Service name"),
            InputPort<std::string>("state", "Activation state")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string state;
        if (!getInput("state", state)) {
            state = name();
        }
        request->state = state;
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->success == "1") {
            RCLCPP_INFO(logger(), "SetAnimateBehavior succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "SetAnimateBehavior failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "SetAnimateBehavior service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// ConversationPromptRosService - Uses cssr_interfaces/srv/ConversationManagerPrompt
// ——————————————
class ConversationPromptRosService : public RosServiceNode<cssr_interfaces::srv::ConversationManagerPrompt>
{
public:
    ConversationPromptRosService(const std::string& name, 
                                 const NodeConfig& conf,
                                 const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::ConversationManagerPrompt>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/conversation/prompt", "Service name"),
            InputPort<std::string>("prompt", "Conversation prompt")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string prompt;
        if (!getInput("prompt", prompt)) {
            // Try to get visitor speech from blackboard
            if (!config().blackboard->get("visitorSpeech", prompt)) {
                RCLCPP_ERROR(logger(), "No prompt provided");
                return false;
            }
        }
        request->prompt = prompt;
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (!response->response.empty()) {
            RCLCPP_INFO(logger(), "Conversation response received");
            config().blackboard->set("conversationResponse", response->response);
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "Empty conversation response");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "Conversation service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// StartOfTree
// ——————————————

//=============================================================================
// Custom Action/Condition Nodes
//=============================================================================

class StartOfTree : public BT::SyncActionNode
{
public:
    StartOfTree(const std::string& name, const BT::NodeConfig& config)
        : BT::SyncActionNode(name, config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override 
    {
        auto& cfg = ConfigManager::instance();
        RCLCPP_INFO(rclcpp::get_logger("StartOfTree"), "=== START OF TREE ===");
        RCLCPP_INFO(rclcpp::get_logger("StartOfTree"), "Language: %s", cfg.getLanguage().c_str());
        return BT::NodeStatus::SUCCESS;
    }
};

class IsVisitorDiscovered : public BT::StatefulActionNode
{
public:
    IsVisitorDiscovered(const std::string& name,
                        const BT::NodeConfig& config,
                        const BT::RosNodeParams& params)
        : BT::StatefulActionNode(name, config)
        , discovered_(false)
    {
        node_ = params.nh.lock();
        if (!node_) {
            throw std::runtime_error("Failed to lock node pointer");
        }
        subscriber_ = node_->create_subscription<cssr_interfaces::msg::FaceDetection>(
            "/faceDetection/data", 10,
            [this](cssr_interfaces::msg::FaceDetection::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!msg->face_label_id.empty()) {
                    discovered_ = true;
                }
            });
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus onStart() override { return BT::NodeStatus::RUNNING; }
    
    BT::NodeStatus onRunning() override 
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return discovered_ ? BT::NodeStatus::SUCCESS : BT::NodeStatus::RUNNING;
    }
    
    void onHalted() override {}

private:
    bool discovered_;
    std::mutex mutex_;
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Subscription<cssr_interfaces::msg::FaceDetection>::SharedPtr subscriber_;
};

class SelectExhibit : public BT::SyncActionNode
{
public:
    SelectExhibit(const std::string& name,
                  const BT::NodeConfig& config,
                  const BT::RosNodeParams& params)
        : BT::SyncActionNode(name, config)
    {
        node_ = params.nh.lock();
        if (!node_) {
            throw std::runtime_error("Failed to lock node pointer");
        }
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override 
    {
        auto logger = node_->get_logger();
        RCLCPP_INFO(logger, "SelectExhibit Action Node");
        try {
            int visits = 0;
            if (!config().blackboard->get("visits", visits)) {
                RCLCPP_WARN(logger, "Unable to retrieve 'visits' from Blackboard");
                return BT::NodeStatus::FAILURE;
            }

            auto& knowledge = KnowledgeManager::instance();
            auto tour = knowledge.getTourSpecification();

            if (visits >= tour.getCurrentLocationCount()) {
                RCLCPP_WARN(logger, "Visit index out of range");
                return BT::NodeStatus::FAILURE;
            }

            auto locationInfo = knowledge.getLocationInfo(tour.locationIds[visits]);
            std::string lang = ConfigManager::instance().getLanguage();

            auto preIt = locationInfo.preMessages.find(lang);
            auto postIt = locationInfo.postMessages.find(lang);
            if (preIt == locationInfo.preMessages.end() ||
                postIt == locationInfo.postMessages.end())
            {
                RCLCPP_ERROR(logger, "Messages not found for language: %s", lang.c_str());
                return BT::NodeStatus::FAILURE;
            }

            config().blackboard->set("exhibitPreGestureMessage", preIt->second);
            config().blackboard->set("exhibitPostGestureMessage", postIt->second);
            config().blackboard->set("exhibitLocation", locationInfo.robotPose);
            config().blackboard->set("exhibitGestureTarget", locationInfo.gestureTarget);

            RCLCPP_INFO(logger, "Visiting: %s", locationInfo.description.c_str());
            config().blackboard->set("visits", ++visits);

            return BT::NodeStatus::SUCCESS;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(logger, "Exception in SelectExhibit: %s", e.what());
            return BT::NodeStatus::FAILURE;
        }
    }

private:
    std::shared_ptr<rclcpp::Node> node_;
};

class IsListWithExhibit : public BT::SyncActionNode
{
public:
    IsListWithExhibit(const std::string& name,
                      const BT::NodeConfig& config,
                      const BT::RosNodeParams& params)
        : BT::SyncActionNode(name, config)
    {
        node_ = params.nh.lock();
        if (!node_) {
            throw std::runtime_error("Failed to lock node pointer");
        }
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override 
    {
        auto logger = node_->get_logger();
        RCLCPP_INFO(logger, "IsListWithExhibit Condition Node");
        try {
            int visits = 0;
            if (!config().blackboard->get("visits", visits)) {
                RCLCPP_ERROR(logger, "Unable to retrieve 'visits' from Blackboard");
                return BT::NodeStatus::FAILURE;
            }

            auto tour = KnowledgeManager::instance().getTourSpecification();
            if (visits < tour.getCurrentLocationCount()) {
                return BT::NodeStatus::SUCCESS;
            }
            RCLCPP_INFO(logger, "ALL LOCATIONS VISITED");
            return BT::NodeStatus::FAILURE;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(logger, "Exception in IsListWithExhibit: %s", e.what());
            return BT::NodeStatus::FAILURE;
        }
    }

private:
    std::shared_ptr<rclcpp::Node> node_;
};

class RetrieveListOfExhibits : public BT::SyncActionNode
{
public:
    RetrieveListOfExhibits(const std::string& name,
                           const BT::NodeConfig& config,
                           const BT::RosNodeParams& params)
        : BT::SyncActionNode(name, config)
    {
        node_ = params.nh.lock();
        if (!node_) {
            throw std::runtime_error("Failed to lock node pointer");
        }
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override 
    {
        auto logger = node_->get_logger();
        RCLCPP_INFO(logger, "RetrieveListOfExhibits Action Node");
        try {
            auto tour = KnowledgeManager::instance().getTourSpecification();
            if (tour.getCurrentLocationCount() == 0) {
                RCLCPP_ERROR(logger, "No exhibits found");
                return BT::NodeStatus::FAILURE;
            }
            config().blackboard->set("visits", 0);
            return BT::NodeStatus::SUCCESS;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(logger, "Exception in RetrieveListOfExhibits: %s", e.what());
            return BT::NodeStatus::FAILURE;
        }
    }

private:
    std::shared_ptr<rclcpp::Node> node_;
};

class IsMutualGazeDiscovered : public BT::StatefulActionNode
{
public:
    IsMutualGazeDiscovered(const std::string& name,
                           const BT::NodeConfig& config,
                           const BT::RosNodeParams& params)
        : BT::StatefulActionNode(name, config)
        , gazeDetected_(false)
    {
        node_ = params.nh.lock();
        if (!node_) {
            throw std::runtime_error("Failed to lock node pointer");
        }
        subscriber_ = node_->create_subscription<cssr_interfaces::msg::OvertAttentionStatus>(
            "/overtAttention/status", 10,
            [this](const cssr_interfaces::msg::OvertAttentionStatus::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (msg->state == "mutual_gaze_detected") {
                    gazeDetected_ = true;
                }
            });
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus onStart() override { return BT::NodeStatus::RUNNING; }

    BT::NodeStatus onRunning() override 
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (gazeDetected_) {
            RCLCPP_INFO(node_->get_logger(), "Mutual gaze detected");
            gazeDetected_ = false;
            return BT::NodeStatus::SUCCESS;
        }
        return BT::NodeStatus::RUNNING;
    }
    
    void onHalted() override {}

private:
    bool gazeDetected_;
    std::mutex mutex_;
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Subscription<cssr_interfaces::msg::OvertAttentionStatus>::SharedPtr subscriber_;
};

class IsVisitorResponseYes : public BT::SyncActionNode
{
public:
    IsVisitorResponseYes(const std::string& name,
                         const BT::NodeConfig& config,
                         const BT::RosNodeParams& params)
        : BT::SyncActionNode(name, config)
    {
        node_ = params.nh.lock();
        if (!node_) {
            throw std::runtime_error("Failed to lock node pointer");
        }
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override 
    {
        auto logger = node_->get_logger();
        RCLCPP_INFO(logger, "IsVisitorResponseYes Condition Node");
        try {
            std::string visitorResponse;
            if (config().blackboard->get("visitorResponse", visitorResponse) &&
                visitorResponse == "yes")
            {
                return BT::NodeStatus::SUCCESS;
            }
            return BT::NodeStatus::FAILURE;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(logger, "Exception in IsVisitorResponseYes: %s", e.what());
            return BT::NodeStatus::FAILURE;
        }
    }

private:
    std::shared_ptr<rclcpp::Node> node_;
};

class HandleFallBack : public BT::SyncActionNode
{
public:
    HandleFallBack(const std::string& name,
                   const BT::NodeConfig& config,
                   const BT::RosNodeParams& params)
        : BT::SyncActionNode(name, config)
    {
        node_ = params.nh.lock();
        if (!node_) {
            throw std::runtime_error("Failed to lock node pointer");
        }
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override 
    {
        RCLCPP_INFO(node_->get_logger(), "HandleFallBack Action Node");
        return BT::NodeStatus::SUCCESS;
    }

private:
    std::shared_ptr<rclcpp::Node> node_;
};
//=============================================================================
// Tree Initialization Function
//=============================================================================

namespace behavior_controller {

BT::Tree initializeTree(const std::string& scenario,
                        std::shared_ptr<rclcpp::Node> node_handle)
{
    // Find XML file
    auto pkg = ament_index_cpp::get_package_share_directory("behavior_controller");
    std::string xml = pkg + "/data/" + scenario + ".xml";
    if (!std::ifstream(xml).good()) {
        throw std::runtime_error("Tree XML not found: " + xml);
    }

    // Create factory and ROS node params
    BT::BehaviorTreeFactory factory;
    BT::RosNodeParams params;
    params.nh = node_handle;
    params.default_port_value = "service_name";

    // Register Action nodes (ROS2 Actions)
    factory.registerNodeType<TTSRosAction>("TTSRosAction", params);
    factory.registerNodeType<NavigateRosAction>("NavigateRosAction", params);
    factory.registerNodeType<GestureRosAction>("GestureRosAction", params);
    factory.registerNodeType<SpeechRecognitionRosAction>("SpeechRecognitionRosAction", params);
    // factory.registerNodeType<AnimateBehaviorRosAction>("AnimateBehaviorRosAction", params);

    // Register Service nodes (ROS2 Services)
    factory.registerNodeType<SetOvertAttentionModeRosService>("SetOvertAttentionModeRosService", params);
    factory.registerNodeType<SetAnimateBehaviorRosService>("SetAnimateBehaviorRosService", params);
    factory.registerNodeType<ConversationPromptRosService>("ConversationPromptRosService", params);

    // Register custom action/condition nodes (WITHOUT params for StartOfTree)
    factory.registerNodeType<StartOfTree>("StartOfTree");
    
    // Register custom nodes WITH params (these need access to ROS node)
    factory.registerNodeType<IsVisitorDiscovered>("IsVisitorDiscovered", params);
    factory.registerNodeType<SelectExhibit>("SelectExhibit", params);
    factory.registerNodeType<IsListWithExhibit>("IsListWithExhibit", params);
    factory.registerNodeType<RetrieveListOfExhibits>("RetrieveListOfExhibits", params);
    factory.registerNodeType<IsMutualGazeDiscovered>("IsMutualGazeDiscovered", params);
    factory.registerNodeType<IsVisitorResponseYes>("IsVisitorResponseYes", params);
    factory.registerNodeType<HandleFallBack>("HandleFallBack", params);

    // Create and return tree
    return factory.createTreeFromFile(xml);
}

}  // namespace behavior_controller