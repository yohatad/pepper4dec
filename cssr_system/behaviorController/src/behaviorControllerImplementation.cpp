/* behaviorControllerImplementation.cpp
 * Author: Yohannes Tadesse Haile
 * Date: July 25, 2025
 * Version: v2.0 - Refactored to use BehaviorTree.ROS2
 */

#include "behaviorController/behaviorControllerInterface.h"
#include <behaviortree_ros2/bt_service_node.hpp>
#include <behaviortree_ros2/bt_action_node.hpp>
#include <behaviortree_ros2/ros_node_params.hpp>
#include <behaviortree_ros2/plugins.hpp>

using namespace BT;

//=============================================================================
// RosServiceNode wrapper implementations
//=============================================================================

// ——————————————
// SayTextRosService
// ——————————————
class SayTextRosService : public RosServiceNode<cssr_interfaces::srv::TextToSpeechSayText>
{
public:
    SayTextRosService(const std::string& name, 
                      const NodeConfig& conf,
                      const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::TextToSpeechSayText>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/textToSpeech/say_text", "Service name"),
            InputPort<std::string>("phrase_key", "Key for utility phrase"),
            InputPort<std::string>("language", "Language code")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string phraseKey;
        if (!getInput("phrase_key", phraseKey)) {
            phraseKey = name();
        }
        
        std::string language;
        if (!getInput("language", language)) {
            language = ConfigManager::instance().getLanguage();
        }
        
        try {
            request->language = language;
            request->message = KnowledgeManager::instance().getUtilityPhrase(phraseKey, language);
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(logger(), "Failed to get utility phrase: %s", e.what());
            return false;
        }
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->success) {
            RCLCPP_INFO(logger(), "SayText succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "SayText failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "SayText service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// NavigateRosService
// ——————————————
class NavigateRosService : public RosServiceNode<cssr_interfaces::srv::RobotNavigationSetGoal>
{
public:
    NavigateRosService(const std::string& name, 
                       const NodeConfig& conf,
                       const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::RobotNavigationSetGoal>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/robotNavigation/set_goal", "Service name"),
            InputPort<double>("goal_x", "Goal X coordinate"),
            InputPort<double>("goal_y", "Goal Y coordinate"),
            InputPort<double>("goal_theta", "Goal theta orientation")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        RobotPose location;
        
        // Try input ports first
        if (getInput("goal_x", location.x) &&
            getInput("goal_y", location.y) &&
            getInput("goal_theta", location.theta)) 
        {
            request->goal_x = location.x;
            request->goal_y = location.y;
            request->goal_theta = location.theta;
            return true;
        }
        
        // Fall back to blackboard
        if (config().blackboard->get("exhibitLocation", location)) {
            request->goal_x = location.x;
            request->goal_y = location.y;
            request->goal_theta = location.theta;
            return true;
        }
        
        RCLCPP_ERROR(logger(), "Cannot get navigation goal from ports or blackboard");
        return false;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->navigation_goal_success) {
            RCLCPP_INFO(logger(), "Navigate succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "Navigate failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "Navigate service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// PerformDeicticGestureRosService
// ——————————————
class PerformDeicticGestureRosService : public RosServiceNode<cssr_interfaces::srv::GestureExecutionPerformGesture>
{
public:
    PerformDeicticGestureRosService(const std::string& name, 
                                    const NodeConfig& conf,
                                    const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::GestureExecutionPerformGesture>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/gestureExecution/perform_gesture", "Service name"),
            InputPort<std::string>("gesture_type", "deictic", "Gesture type"),
            InputPort<int>("gesture_id", Constants::DEICTIC_GESTURE_ID, "Gesture ID"),
            InputPort<int>("gesture_duration", Constants::GESTURE_DURATION_MS, "Duration in ms"),
            InputPort<double>("location_x", "Target X coordinate"),
            InputPort<double>("location_y", "Target Y coordinate"),
            InputPort<double>("location_z", "Target Z coordinate"),
            InputPort<double>("bow_nod_angle", 0.0, "Bow/nod angle")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        // Set defaults
        request->gesture_type = "deictic";
        request->gesture_id = Constants::DEICTIC_GESTURE_ID;
        request->gesture_duration = Constants::GESTURE_DURATION_MS;
        request->bow_nod_angle = 0;
        
        // Override from ports
        getInput("gesture_type", request->gesture_type);
        getInput("gesture_id", request->gesture_id);
        getInput("gesture_duration", request->gesture_duration);
        getInput("bow_nod_angle", request->bow_nod_angle);
        
        // Get target location
        Position3D target;
        if (getInput("location_x", target.x) &&
            getInput("location_y", target.y) &&
            getInput("location_z", target.z)) 
        {
            request->location_x = target.x;
            request->location_y = target.y;
            request->location_z = target.z;
            return true;
        }
        
        // Try blackboard
        if (config().blackboard->get("exhibitGestureTarget", target)) {
            request->location_x = target.x;
            request->location_y = target.y;
            request->location_z = target.z;
            return true;
        }
        
        RCLCPP_ERROR(logger(), "Cannot get gesture target from ports or blackboard");
        return false;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->gesture_success) {
            RCLCPP_INFO(logger(), "PerformDeicticGesture succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "PerformDeicticGesture failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "PerformDeicticGesture service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// PerformIconicGestureRosService
// ——————————————
class PerformIconicGestureRosService : public RosServiceNode<cssr_interfaces::srv::GestureExecutionPerformGesture>
{
public:
    PerformIconicGestureRosService(const std::string& name, 
                                   const NodeConfig& conf,
                                   const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::GestureExecutionPerformGesture>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/gestureExecution/perform_gesture", "Service name"),
            InputPort<std::string>("gesture_type", "iconic", "Gesture type"),
            InputPort<int>("gesture_id", "Gesture ID (welcome/goodbye)"),
            InputPort<int>("gesture_duration", Constants::GESTURE_DURATION_MS, "Duration in ms"),
            InputPort<double>("bow_nod_angle", 0.0, "Bow/nod angle")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        request->gesture_type = "iconic";
        request->gesture_duration = Constants::GESTURE_DURATION_MS;
        request->bow_nod_angle = 0;
        request->location_x = request->location_y = request->location_z = 0.0;
        
        // Override from ports
        getInput("gesture_type", request->gesture_type);
        getInput("gesture_duration", request->gesture_duration);
        getInput("bow_nod_angle", request->bow_nod_angle);
        
        // Determine gesture ID
        int gestureId = 0;
        if (!getInput("gesture_id", gestureId)) {
            std::string nodeName = name();
            if (nodeName == "welcome") {
                gestureId = Constants::WELCOME_GESTURE_ID;
            } else if (nodeName == "goodbye") {
                gestureId = Constants::GOODBYE_GESTURE_ID;
            } else {
                RCLCPP_ERROR(logger(), "Undefined Iconic Gesture Type: %s", nodeName.c_str());
                return false;
            }
        }
        
        request->gesture_id = gestureId;
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->gesture_success) {
            RCLCPP_INFO(logger(), "PerformIconicGesture succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "PerformIconicGesture failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "PerformIconicGesture service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// DescribeExhibitSpeechRosService
// ——————————————
class DescribeExhibitSpeechRosService : public RosServiceNode<cssr_interfaces::srv::TextToSpeechSayText>
{
public:
    DescribeExhibitSpeechRosService(const std::string& name, 
                                    const NodeConfig& conf,
                                    const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::TextToSpeechSayText>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/textToSpeech/say_text", "Service name"),
            InputPort<std::string>("message", "Direct message to speak"),
            InputPort<std::string>("language", "Language code"),
            InputPort<std::string>("message_type", "Type: 'pre' or 'post'")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string language;
        if (!getInput("language", language)) {
            language = ConfigManager::instance().getLanguage();
        }
        request->language = language;
        
        // Try direct message first
        std::string message;
        if (getInput("message", message)) {
            request->message = message;
            return true;
        }
        
        // Determine message type
        std::string messageType;
        if (!getInput("message_type", messageType)) {
            std::string nodeName = name();
            if (nodeName == "1" || nodeName.find("pre") != std::string::npos) {
                messageType = "pre";
            } else if (nodeName == "2" || nodeName.find("post") != std::string::npos) {
                messageType = "post";
            }
        }
        
        // Get from blackboard
        if (messageType == "pre") {
            if (config().blackboard->get("exhibitPreGestureMessage", message)) {
                request->message = message;
                return true;
            }
        } else if (messageType == "post") {
            if (config().blackboard->get("exhibitPostGestureMessage", message)) {
                request->message = message;
                return true;
            }
        }
        
        RCLCPP_ERROR(logger(), "Cannot get message for DescribeExhibitSpeech");
        return false;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->success) {
            RCLCPP_INFO(logger(), "DescribeExhibitSpeech succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "DescribeExhibitSpeech failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "DescribeExhibitSpeech service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// SetSpeechEventRosService
// ——————————————
class SetSpeechEventRosService : public RosServiceNode<cssr_interfaces::srv::SpeechEventSetEnabled>
{
public:
    SetSpeechEventRosService(const std::string& name, 
                             const NodeConfig& conf,
                             const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::SpeechEventSetEnabled>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/speechEvent/set_enabled", "Service name"),
            InputPort<std::string>("status", "Enable/disable status")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string status;
        if (!getInput("status", status)) {
            status = name();
        }
        request->status = status;
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->response) {
            RCLCPP_INFO(logger(), "SetSpeechEvent succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "SetSpeechEvent failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "SetSpeechEvent service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// SetOvertAttentionModeRosService
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
            InputPort<double>("location_x", "Location X (for 'location' state)"),
            InputPort<double>("location_y", "Location Y (for 'location' state)"),
            InputPort<double>("location_z", "Location Z (for 'location' state)")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string state;
        if (!getInput("state", state)) {
            state = name();
        }
        request->state = state;
        
        if (state == "location") {
            Position3D target;
            
            if (getInput("location_x", target.x) &&
                getInput("location_y", target.y) &&
                getInput("location_z", target.z)) 
            {
                request->location_x = target.x;
                request->location_y = target.y;
                request->location_z = target.z;
                return true;
            }
            
            if (config().blackboard->get("exhibitGestureTarget", target)) {
                request->location_x = target.x;
                request->location_y = target.y;
                request->location_z = target.z;
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
// SetAnimateBehaviorRosService
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
// ResetRobotPoseRosService
// ——————————————
class ResetRobotPoseRosService : public RosServiceNode<cssr_interfaces::srv::RobotLocalizationResetPose>
{
public:
    ResetRobotPoseRosService(const std::string& name, 
                             const NodeConfig& conf,
                             const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::RobotLocalizationResetPose>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/robotLocalization/reset_pose", "Service name")
        });
    }

    bool setRequest(Request::SharedPtr& /*request*/) override
    {
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->success) {
            RCLCPP_INFO(logger(), "ResetRobotPose succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "ResetRobotPose failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "ResetRobotPose service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// PressYesNoDialogueRosService
// ——————————————
class PressYesNoDialogueRosService : public RosServiceNode<cssr_interfaces::srv::TabletEventPromptAndGetResponse>
{
public:
    PressYesNoDialogueRosService(const std::string& name, 
                                 const NodeConfig& conf,
                                 const RosNodeParams& params)
        : RosServiceNode<cssr_interfaces::srv::TabletEventPromptAndGetResponse>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("service_name", "/tabletEvent/prompt_and_get_response", "Service name"),
            InputPort<std::string>("message", "'Yes'|'No'", "Prompt message")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string message;
        if (!getInput("message", message)) {
            message = "'Yes'|'No'";
        }
        request->message = message;
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->success) {
            RCLCPP_INFO(logger(), "PressYesNoDialogue succeeded");
            return NodeStatus::SUCCESS;
        }
        RCLCPP_WARN(logger(), "PressYesNoDialogue failed");
        return NodeStatus::FAILURE;
    }

    NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "PressYesNoDialogue service error: %d", static_cast<int>(error));
        return NodeStatus::FAILURE;
    }
};

//=============================================================================
// Custom Action/Condition Nodes using BtActionNode base
//=============================================================================

// ——————————————
// StartOfTree
// ——————————————
class StartOfTree : public BtActionNode<>
{
public:
    StartOfTree(const std::string& name,
                const NodeConfig& config,
                const RosNodeParams& params)
        : BtActionNode<>(name, config, params)
    {}

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override 
    {
        RCLCPP_INFO(logger(), "=== START OF TREE ===");
        try {
            auto& cfg = ConfigManager::instance();

            if (cfg.isAsrEnabled()) {
                auto client = node_->create_client<cssr_interfaces::srv::SpeechEventSetLanguage>(
                    "/speechEvent/set_language");
                
                if (!client->wait_for_service(std::chrono::seconds(1))) {
                    RCLCPP_WARN(logger(), "Speech service not available - running in standalone mode");
                    return NodeStatus::SUCCESS;
                }

                auto request = std::make_shared<cssr_interfaces::srv::SpeechEventSetLanguage::Request>();
                request->language = cfg.getLanguage();

                auto future = client->async_send_request(request);
                if (rclcpp::spin_until_future_complete(node_, future, std::chrono::seconds(5)) ==
                    rclcpp::FutureReturnCode::SUCCESS) 
                {
                    return NodeStatus::SUCCESS;
                }
            }
            return NodeStatus::SUCCESS;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(logger(), "Exception in StartOfTree: %s", e.what());
            return NodeStatus::FAILURE;
        }
    }
};

// ——————————————
// IsVisitorDiscovered
// ——————————————
class IsVisitorDiscovered : public BtActionNode<>
{
public:
    IsVisitorDiscovered(const std::string& name,
                        const NodeConfig& config,
                        const RosNodeParams& params)
        : BtActionNode<>(name, config, params)
        , discovered_(false)
    {
        subscriber_ = node_->create_subscription<cssr_interfaces::msg::FaceDetectionData>(
            "/faceDetection/data", 10,
            [this](cssr_interfaces::msg::FaceDetectionData::SharedPtr msg) {
                if (!msg->face_label_id.empty()) {
                    discovered_ = true;
                }
            });
    }

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override 
    {
        return discovered_ ? NodeStatus::SUCCESS : NodeStatus::RUNNING;
    }

private:
    bool discovered_;
    rclcpp::Subscription<cssr_interfaces::msg::FaceDetectionData>::SharedPtr subscriber_;
};

// ——————————————
// SelectExhibit
// ——————————————
class SelectExhibit : public BtActionNode<>
{
public:
    SelectExhibit(const std::string& name,
                  const NodeConfig& config,
                  const RosNodeParams& params)
        : BtActionNode<>(name, config, params)
    {}

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override 
    {
        RCLCPP_INFO(logger(), "SelectExhibit Action Node");
        try {
            int visits = 0;
            if (!config().blackboard->get("visits", visits)) {
                RCLCPP_WARN(logger(), "Unable to retrieve 'visits' from Blackboard");
                return NodeStatus::FAILURE;
            }

            auto& knowledge = KnowledgeManager::instance();
            auto tour = knowledge.getTourSpecification();

            if (visits >= tour.getCurrentLocationCount()) {
                RCLCPP_WARN(logger(), "Visit index out of range");
                return NodeStatus::FAILURE;
            }

            auto locationInfo = knowledge.getLocationInfo(tour.locationIds[visits]);
            std::string lang = ConfigManager::instance().getLanguage();

            auto preIt = locationInfo.preMessages.find(lang);
            auto postIt = locationInfo.postMessages.find(lang);
            if (preIt == locationInfo.preMessages.end() ||
                postIt == locationInfo.postMessages.end())
            {
                RCLCPP_ERROR(logger(), "Messages not found for language: %s", lang.c_str());
                return NodeStatus::FAILURE;
            }

            config().blackboard->set("exhibitPreGestureMessage", preIt->second);
            config().blackboard->set("exhibitPostGestureMessage", postIt->second);
            config().blackboard->set("exhibitLocation", locationInfo.robotPose);
            config().blackboard->set("exhibitGestureTarget", locationInfo.gestureTarget);

            RCLCPP_INFO(logger(), "Visiting: %s", locationInfo.description.c_str());
            config().blackboard->set("visits", ++visits);

            return NodeStatus::SUCCESS;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(logger(), "Exception in SelectExhibit: %s", e.what());
            return NodeStatus::FAILURE;
        }
    }
};

// ——————————————
// IsListWithExhibit
// ——————————————
class IsListWithExhibit : public BtActionNode<>
{
public:
    IsListWithExhibit(const std::string& name,
                      const NodeConfig& config,
                      const RosNodeParams& params)
        : BtActionNode<>(name, config, params)
    {}

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override 
    {
        RCLCPP_INFO(logger(), "IsListWithExhibit Condition Node");
        try {
            int visits = 0;
            if (!config().blackboard->get("visits", visits)) {
                RCLCPP_ERROR(logger(), "Unable to retrieve 'visits' from Blackboard");
                return NodeStatus::FAILURE;
            }

            auto tour = KnowledgeManager::instance().getTourSpecification();
            if (visits < tour.getCurrentLocationCount()) {
                return NodeStatus::SUCCESS;
            }
            RCLCPP_INFO(logger(), "ALL LOCATIONS VISITED");
            return NodeStatus::FAILURE;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(logger(), "Exception in IsListWithExhibit: %s", e.what());
            return NodeStatus::FAILURE;
        }
    }
};

// ——————————————
// RetrieveListOfExhibits
// ——————————————
class RetrieveListOfExhibits : public BtActionNode<>
{
public:
    RetrieveListOfExhibits(const std::string& name,
                           const NodeConfig& config,
                           const RosNodeParams& params)
        : BtActionNode<>(name, config, params)
    {}

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override 
    {
        RCLCPP_INFO(logger(), "RetrieveListOfExhibits Action Node");
        try {
            auto tour = KnowledgeManager::instance().getTourSpecification();
            if (tour.getCurrentLocationCount() == 0) {
                RCLCPP_ERROR(logger(), "No exhibits found");
                return NodeStatus::FAILURE;
            }
            config().blackboard->set("visits", 0);
            return NodeStatus::SUCCESS;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(logger(), "Exception in RetrieveListOfExhibits: %s", e.what());
            return NodeStatus::FAILURE;
        }
    }
};

// ——————————————
// IsMutualGazeDiscovered
// ——————————————
class IsMutualGazeDiscovered : public BtActionNode<>
{
public:
    IsMutualGazeDiscovered(const std::string& name,
                           const NodeConfig& config,
                           const RosNodeParams& params)
        : BtActionNode<>(name, config, params)
        , seekingStatus_(0)
    {
        subscriber_ = node_->create_subscription<cssr_interfaces::msg::OvertAttentionMode>(
            "/overtAttention/mode", 10,
            [this](const cssr_interfaces::msg::OvertAttentionMode::SharedPtr msg) {
                if (msg->state == "seeking") {
                    if (msg->value == 2) seekingStatus_.store(1);
                    else if (msg->value == 3) seekingStatus_.store(2);
                }
            });
    }

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override 
    {
        auto status = seekingStatus_.load();
        if (status == 1) {
            RCLCPP_INFO(logger(), "Mutual gaze detected");
            return NodeStatus::SUCCESS;
        }
        if (status == 2) {
            RCLCPP_INFO(logger(), "Mutual gaze detection failed");
            return NodeStatus::FAILURE;
        }
        return NodeStatus::RUNNING;
    }

private:
    std::atomic<int> seekingStatus_;
    rclcpp::Subscription<cssr_interfaces::msg::OvertAttentionMode>::SharedPtr subscriber_;
};

// ——————————————
// IsVisitorResponseYes
// ——————————————
class IsVisitorResponseYes : public BtActionNode<>
{
public:
    IsVisitorResponseYes(const std::string& name,
                         const NodeConfig& config,
                         const RosNodeParams& params)
        : BtActionNode<>(name, config, params)
    {}

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override 
    {
        RCLCPP_INFO(logger(), "IsVisitorResponseYes Condition Node");
        try {
            std::string visitorResponse;
            if (config().blackboard->get("visitorResponse", visitorResponse) &&
                visitorResponse == "yes")
            {
                return NodeStatus::SUCCESS;
            }
            return NodeStatus::FAILURE;
        }
        catch (const std::exception& e) {
            RCLCPP_ERROR(logger(), "Exception in IsVisitorResponseYes: %s", e.what());
            return NodeStatus::FAILURE;
        }
    }
};

// ——————————————
// HandleFallBack
// ——————————————
class HandleFallBack : public BtActionNode<>
{
public:
    HandleFallBack(const std::string& name,
                   const NodeConfig& config,
                   const RosNodeParams& params)
        : BtActionNode<>(name, config, params)
    {}

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override 
    {
        RCLCPP_INFO(logger(), "HandleFallBack Action Node");
        return NodeStatus::SUCCESS;
    }
};

//=============================================================================
// Tree Initialization Function
//=============================================================================

namespace behavior_controller {

BT::Tree initializeTree(const std::string& scenario,
                        std::shared_ptr<rclcpp::Node> node_handle)
{
    // Find XML file
    auto pkg = ament_index_cpp::get_package_share_directory("cssr_interfaces");
    std::string xml = pkg + "/behaviorController/data/" + scenario + ".xml";
    if (!std::ifstream(xml).good()) {
        throw std::runtime_error("Tree XML not found: " + xml);
    }

    // Create factory and ROS node params
    BT::BehaviorTreeFactory factory;
    BT::RosNodeParams params;
    params.nh = node_handle;
    params.default_port_value = "service_name";

    // Register all RosServiceNode-based nodes
    factory.registerNodeType<SayTextRosService>("SayTextRosService", params);
    factory.registerNodeType<NavigateRosService>("NavigateRosService", params);
    factory.registerNodeType<PerformDeicticGestureRosService>("PerformDeicticGestureRosService", params);
    factory.registerNodeType<PerformIconicGestureRosService>("PerformIconicGestureRosService", params);
    factory.registerNodeType<DescribeExhibitSpeechRosService>("DescribeExhibitSpeechRosService", params);
    factory.registerNodeType<SetSpeechEventRosService>("SetSpeechEventRosService", params);
    factory.registerNodeType<SetOvertAttentionModeRosService>("SetOvertAttentionModeRosService", params);
    factory.registerNodeType<SetAnimateBehaviorRosService>("SetAnimateBehaviorRosService", params);
    factory.registerNodeType<ResetRobotPoseRosService>("ResetRobotPoseRosService", params);
    factory.registerNodeType<PressYesNoDialogueRosService>("PressYesNoDialogueRosService", params);

    // Register custom action/condition nodes
    factory.registerNodeType<StartOfTree>("StartOfTree", params);
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