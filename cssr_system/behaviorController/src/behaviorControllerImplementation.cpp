/* behaviorControllerImplementation.cpp

 * Author: Yohannes Tadesse Haile
 * Date: July 25, 2025
 * Version: v1.0
 */

#include "behaviorController/behaviorControllerInterface.h"

//=============================================================================
// Behavior Tree Node Implementations
//=============================================================================

#include <behaviortree_ros2/bt_service_node.hpp>
#include <behaviortree_ros2/ros_node_params.hpp>

using namespace BT;

// ——————————————
// SayTextRosService (RosServiceNode version)
// ——————————————
class SayTextRosService : public RosServiceNode<cssr_system::srv::TextToSpeechSayText>
{
public:
    SayTextRosService(const std::string& name, const NodeConfig& conf,
                      const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::TextToSpeechSayText>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("phrase_key"),
            InputPort<std::string>("language")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string phraseKey;
        if (!getInput("phrase_key", phraseKey)) {
            // Use node name as phrase key if not provided
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
        } else {
            RCLCPP_WARN(logger(), "SayText failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "SayText service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// NavigateRosService (RosServiceNode version)
// ——————————————
class NavigateRosService : public RosServiceNode<cssr_system::srv::RobotNavigationSetGoal>
{
public:
    NavigateRosService(const std::string& name, const NodeConfig& conf,
                       const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::RobotNavigationSetGoal>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<double>("goal_x"),
            InputPort<double>("goal_y"),
            InputPort<double>("goal_theta")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        RobotPose location;
        if (getInput("goal_x", location.x) &&
            getInput("goal_y", location.y) &&
            getInput("goal_theta", location.theta)) {
            // Use input ports if provided
            request->goal_x = location.x;
            request->goal_y = location.y;
            request->goal_theta = location.theta;
            return true;
        }
        
        // Otherwise try to get from blackboard
        auto blackboard = config().blackboard;
        if (blackboard->rootBlackboard()->get("exhibitLocation", location)) {
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
        } else {
            RCLCPP_WARN(logger(), "Navigate failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "Navigate service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// PerformDeicticGestureRosService (RosServiceNode version)
// ——————————————
class PerformDeicticGestureRosService : public RosServiceNode<cssr_system::srv::GestureExecutionPerformGesture>
{
public:
    PerformDeicticGestureRosService(const std::string& name, const NodeConfig& conf,
                                    const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::GestureExecutionPerformGesture>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("gesture_type"),
            InputPort<int>("gesture_id"),
            InputPort<int>("gesture_duration"),
            InputPort<double>("location_x"),
            InputPort<double>("location_y"),
            InputPort<double>("location_z"),
            InputPort<double>("bow_nod_angle")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        // Default values
        request->gesture_type = "deictic";
        request->gesture_id = Constants::DEICTIC_GESTURE_ID;
        request->gesture_duration = Constants::GESTURE_DURATION_MS;
        request->bow_nod_angle = 0;
        
        // Try to get target from blackboard first
        Position3D target;
        auto blackboard = config().blackboard;
        if (blackboard->rootBlackboard()->get("exhibitGestureTarget", target)) {
            request->location_x = target.x;
            request->location_y = target.y;
            request->location_z = target.z;
            return true;
        }
        
        // Otherwise try input ports
        if (getInput("location_x", request->location_x) &&
            getInput("location_y", request->location_y) &&
            getInput("location_z", request->location_z)) {
            // Use input ports if provided
            getInput("gesture_type", request->gesture_type);
            getInput("gesture_id", request->gesture_id);
            getInput("gesture_duration", request->gesture_duration);
            getInput("bow_nod_angle", request->bow_nod_angle);
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
        } else {
            RCLCPP_WARN(logger(), "PerformDeicticGesture failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "PerformDeicticGesture service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// PerformIconicGestureRosService (RosServiceNode version)
// ——————————————
class PerformIconicGestureRosService : public RosServiceNode<cssr_system::srv::GestureExecutionPerformGesture>
{
public:
    PerformIconicGestureRosService(const std::string& name, const NodeConfig& conf,
                                   const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::GestureExecutionPerformGesture>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("gesture_type"),
            InputPort<int>("gesture_id"),
            InputPort<int>("gesture_duration"),
            InputPort<double>("bow_nod_angle")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        request->gesture_type = "iconic";
        request->gesture_duration = Constants::GESTURE_DURATION_MS;
        request->bow_nod_angle = 0;
        request->location_x = request->location_y = request->location_z = 0.0;
        
        // Determine gesture ID based on node name or input port
        std::string gestureType = name();
        int gestureId = 0;
        
        if (gestureType == "welcome") {
            gestureId = Constants::WELCOME_GESTURE_ID;
        } else if (gestureType == "goodbye") {
            gestureId = Constants::GOODBYE_GESTURE_ID;
        } else {
            // Try to get from input port
            if (!getInput("gesture_id", gestureId)) {
                RCLCPP_ERROR(logger(), "Undefined Iconic Gesture Type: %s", gestureType.c_str());
                return false;
            }
        }
        
        request->gesture_id = gestureId;
        
        // Allow override from input ports
        getInput("gesture_type", request->gesture_type);
        getInput("gesture_duration", request->gesture_duration);
        getInput("bow_nod_angle", request->bow_nod_angle);
        
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->gesture_success) {
            RCLCPP_INFO(logger(), "PerformIconicGesture succeeded");
            return NodeStatus::SUCCESS;
        } else {
            RCLCPP_WARN(logger(), "PerformIconicGesture failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "PerformIconicGesture service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// DescribeExhibitSpeechRosService (RosServiceNode version)
// ——————————————
class DescribeExhibitSpeechRosService : public RosServiceNode<cssr_system::srv::TextToSpeechSayText>
{
public:
    DescribeExhibitSpeechRosService(const std::string& name, const NodeConfig& conf,
                                    const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::TextToSpeechSayText>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("message"),
            InputPort<std::string>("language"),
            InputPort<std::string>("message_type")  // "pre" or "post"
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string language;
        if (!getInput("language", language)) {
            language = ConfigManager::instance().getLanguage();
        }
        request->language = language;
        
        std::string message;
        if (getInput("message", message)) {
            // Use direct message from input port
            request->message = message;
            return true;
        }
        
        // Otherwise try to get from blackboard based on message_type or node name
        std::string messageType;
        if (!getInput("message_type", messageType)) {
            // Infer from node name
            std::string nodeName = name();
            if (nodeName == "1" || nodeName.find("pre") != std::string::npos) {
                messageType = "pre";
            } else if (nodeName == "2" || nodeName.find("post") != std::string::npos) {
                messageType = "post";
            }
        }
        
        auto blackboard = config().blackboard;
        if (messageType == "pre") {
            if (blackboard->get("exhibitPreGestureMessage", message)) {
                request->message = message;
                return true;
            }
        } else if (messageType == "post") {
            if (blackboard->get("exhibitPostGestureMessage", message)) {
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
        } else {
            RCLCPP_WARN(logger(), "DescribeExhibitSpeech failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "DescribeExhibitSpeech service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// SetSpeechEventRosService (RosServiceNode version)
// ——————————————
class SetSpeechEventRosService : public RosServiceNode<cssr_system::srv::SpeechEventSetEnabled>
{
public:
    SetSpeechEventRosService(const std::string& name, const NodeConfig& conf,
                             const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::SpeechEventSetEnabled>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("status")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string status;
        if (!getInput("status", status)) {
            // Use node name as status
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
        } else {
            RCLCPP_WARN(logger(), "SetSpeechEvent failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "SetSpeechEvent service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// SetOvertAttentionModeRosService (RosServiceNode version)
// ——————————————
class SetOvertAttentionModeRosService : public RosServiceNode<cssr_system::srv::OvertAttentionSetMode>
{
public:
    SetOvertAttentionModeRosService(const std::string& name, const NodeConfig& conf,
                                    const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::OvertAttentionSetMode>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("state"),
            InputPort<double>("location_x"),
            InputPort<double>("location_y"),
            InputPort<double>("location_z")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string state;
        if (!getInput("state", state)) {
            // Use node name as state
            state = name();
        }
        request->state = state;
        
        // If state is "location", need location coordinates
        if (state == "location") {
            Position3D target;
            
            // Try input ports first
            if (getInput("location_x", target.x) &&
                getInput("location_y", target.y) &&
                getInput("location_z", target.z)) {
                request->location_x = target.x;
                request->location_y = target.y;
                request->location_z = target.z;
                return true;
            }
            
            // Otherwise try blackboard
            auto blackboard = config().blackboard;
            if (blackboard->rootBlackboard()->get("exhibitGestureTarget", target)) {
                request->location_x = target.x;
                request->location_y = target.y;
                request->location_z = target.z;
                return true;
            }
            
            RCLCPP_ERROR(logger(), "Cannot get location for SetOvertAttentionMode with state 'location'");
            return false;
        }
        
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->mode_set_success) {
            RCLCPP_INFO(logger(), "SetOvertAttentionMode succeeded");
            return NodeStatus::SUCCESS;
        } else {
            RCLCPP_WARN(logger(), "SetOvertAttentionMode failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "SetOvertAttentionMode service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// SetAnimateBehaviorRosService (RosServiceNode version)
// ——————————————
class SetAnimateBehaviorRosService : public RosServiceNode<cssr_system::srv::AnimateBehaviorSetActivation>
{
public:
    SetAnimateBehaviorRosService(const std::string& name, const NodeConfig& conf,
                                 const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::AnimateBehaviorSetActivation>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("state")
        });
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        std::string state;
        if (!getInput("state", state)) {
            // Use node name as state
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
        } else {
            RCLCPP_WARN(logger(), "SetAnimateBehavior failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "SetAnimateBehavior service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// ResetRobotPoseRosService (RosServiceNode version)
// ——————————————
class ResetRobotPoseRosService : public RosServiceNode<cssr_system::srv::RobotLocalizationResetPose>
{
public:
    ResetRobotPoseRosService(const std::string& name, const NodeConfig& conf,
                             const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::RobotLocalizationResetPose>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({});
    }

    bool setRequest(Request::SharedPtr& request) override
    {
        // No request parameters needed
        return true;
    }

    NodeStatus onResponseReceived(const Response::SharedPtr& response) override
    {
        if (response->success) {
            RCLCPP_INFO(logger(), "ResetRobotPose succeeded");
            return NodeStatus::SUCCESS;
        } else {
            RCLCPP_WARN(logger(), "ResetRobotPose failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "ResetRobotPose service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// PressYesNoDialogueRosService (RosServiceNode version)
// ——————————————
class PressYesNoDialogueRosService : public RosServiceNode<cssr_system::srv::TabletEventPromptAndGetResponse>
{
public:
    PressYesNoDialogueRosService(const std::string& name, const NodeConfig& conf,
                                 const RosNodeParams& params)
        : RosServiceNode<cssr_system::srv::TabletEventPromptAndGetResponse>(name, conf, params)
    {}

    static PortsList providedPorts()
    {
        return providedBasicPorts({
            InputPort<std::string>("message")
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
        } else {
            RCLCPP_WARN(logger(), "PressYesNoDialogue failed");
            return NodeStatus::FAILURE;
        }
    }

    virtual NodeStatus onFailure(ServiceNodeErrorCode error) override
    {
        RCLCPP_ERROR(logger(), "PressYesNoDialogue service error: %s", toStr(error));
        return NodeStatus::FAILURE;
    }
};

// ——————————————
// StartOfTree
// ——————————————
class StartOfTree : public BT::SyncActionNode, public BaseTreeNode {
public:
    StartOfTree(const std::string &name, const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config) , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("=== START OF TREE ===");
        try {
            auto &cfg = ConfigManager::instance();

            if (cfg.isAsrEnabled()) {
                auto req = std::make_shared<cssr_system::srv::SpeechEventSetLanguage::Request>();
                auto res = std::make_shared<cssr_system::srv::SpeechEventSetLanguage::Response>();
                req->language = cfg.getLanguage();

                bool ok = callServiceSafely<cssr_system::srv::SpeechEventSetLanguage>(
                              "/speechEvent/set_language", req, res, "StartOfTree");
                if (!ok) {
                    logger_->warn("Speech service not available - running in standalone mode");
                }
            }
            return BT::NodeStatus::SUCCESS;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in StartOfTree: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// SayText
// ——————————————
class SayText : public BT::SyncActionNode, public BaseTreeNode {
public:
    SayText(const std::string &name,
            const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SayText Action Node");
        try {
            auto req = std::make_shared<cssr_system::srv::TextToSpeechSayText::Request>();
            auto res = std::make_shared<cssr_system::srv::TextToSpeechSayText::Response>();

            req->language = ConfigManager::instance().getLanguage();
            // Use the node's instance name as the phrase key
            req->message  = KnowledgeManager::instance().getUtilityPhrase(name());

            bool ok = callServiceSafely<cssr_system::srv::TextToSpeechSayText>(
                          "/textToSpeech/say_text", req, res, "SayText");
            return (ok && res->success)
                 ? BT::NodeStatus::SUCCESS
                 : BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in SayText: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


//=============================================================================

class Navigate : public BT::SyncActionNode, public BaseTreeNode {
public:
    Navigate(const std::string &name,
             const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)    // pulls node_ & logger_ from the Blackboard
    {}

    static BT::PortsList providedPorts() {
        // You could also define an input port for a PoseStamped
        return {};
    }

    BT::NodeStatus tick() override {
        logger_->info("Navigate Action Node");
        try {
            // Retrieve the target pose from the root Blackboard
            RobotPose location;
            if (!config().blackboard->rootBlackboard()->get("exhibitLocation", location)) {
                logger_->error("exhibitLocation not found on Blackboard");
                return BT::NodeStatus::FAILURE;
            }

            // Prepare navigation service request
            auto req = std::make_shared<cssr_system::srv::RobotNavigationSetGoal::Request>();
            auto res = std::make_shared<cssr_system::srv::RobotNavigationSetGoal::Response>();
            req->goal_x     = location.x;
            req->goal_y     = location.y;
            req->goal_theta = location.theta;

            // Call the service; success if navigation_goal_success == true
            bool ok = callServiceSafely<cssr_system::srv::RobotNavigationSetGoal>(
                          "/robotNavigation/set_goal", req, res, "Navigate");
            return (ok && res->navigation_goal_success)
                 ? BT::NodeStatus::SUCCESS
                 : BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in Navigate: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// GetVisitorResponse
// ——————————————

// class GetVisitorResponse : public BT::SyncActionNode, public BaseTreeNode {
// public:
//     GetVisitorResponse(const std::string &name,
//                        const BT::NodeConfiguration &config)
//       : BT::SyncActionNode(name, config)
//       , BaseTreeNode(config)
//     {}

//     static BT::PortsList providedPorts() { return {}; }

//     BT::NodeStatus tick() override {
//         logger_->info("GetVisitorResponse Action Node");
//         try {
//             // Prepare request/response for your "get_response" service
//             auto req = std::make_shared<cssr_system::srv::VisitorResponse::Request>();
//             auto res = std::make_shared<cssr_system::srv::VisitorResponse::Response>();

//             // Call the service once per tick
//             bool ok = callServiceSafely<cssr_system::srv::VisitorResponse>(
//                           "/visitorResponse/get_response", req, res, "GetVisitorResponse");

//             if (ok && res->response) {
//                 // Store the positive result if you like
//                 config().blackboard->rootBlackboard()->set("visitorResponse", "yes");
//                 return BT::NodeStatus::SUCCESS;
//             }
//             // Otherwise, keep spinning until the Timeout decorator cuts you off
//             return BT::NodeStatus::RUNNING;
//         }
//         catch (const std::exception &e) {
//             logger_->error("Exception in GetVisitorResponse: " + std::string(e.what()));
//             return BT::NodeStatus::FAILURE;
//         }
//     }
// };

// ——————————————
// IsVisitorDiscovered
// ——————————————

class IsVisitorDiscovered : public BT::ConditionNode,
                            public BaseTreeNode
{
    public:
    IsVisitorDiscovered(const std::string &name,
                        const BT::NodeConfiguration &config)
        : BT::ConditionNode(name, config)
        , BaseTreeNode(config)
        , discovered_(false)
    {
        subscriber_ = node_->create_subscription<cssr_system::msg::FaceDetectionData>(
            "/faceDetection/data", 10,
            [this](cssr_system::msg::FaceDetectionData::SharedPtr msg) {
                if (!msg->face_label_id.empty()) {
                    discovered_ = true;    // plain assignment, no .store()
                }
            });
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        return discovered_
        ? BT::NodeStatus::SUCCESS
        : BT::NodeStatus::RUNNING;
  }

private:
    bool discovered_{false};
    rclcpp::Subscription<cssr_system::msg::FaceDetectionData>::SharedPtr subscriber_;
};


// ——————————————
// SelectExhibit
// ——————————————
class SelectExhibit : public BT::SyncActionNode, public BaseTreeNode {
public:
    SelectExhibit(const std::string &name,
                  const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SelectExhibit Action Node");
        try {
            int visits = 0;
            if (!config().blackboard->get("visits", visits)) {
                logger_->warn("Unable to retrieve 'visits' from Blackboard");
                return BT::NodeStatus::FAILURE;
            }

            auto &knowledge = KnowledgeManager::instance();
            auto tour = knowledge.getTourSpecification();

            if (visits >= tour.getCurrentLocationCount()) {
                logger_->warn("Visit index out of range");
                return BT::NodeStatus::FAILURE;
            }

            // Fetch exhibit info
            auto locationInfo = knowledge.getLocationInfo(tour.locationIds[visits]);
            std::string lang = ConfigManager::instance().getLanguage();

            // Lookup messages
            auto preIt  = locationInfo.preMessages.find(lang);
            auto postIt = locationInfo.postMessages.find(lang);
            if (preIt  == locationInfo.preMessages.end() ||
                postIt == locationInfo.postMessages.end())
            {
                logger_->error("Messages not found for language: " + lang);
                return BT::NodeStatus::FAILURE;
            }
            if (preIt->second.empty() || postIt->second.empty()) {
                logger_->error("Empty messages for language: " + lang);
                return BT::NodeStatus::FAILURE;
            }

            // Store on blackboards
            config().blackboard->set("exhibitPreGestureMessage",  preIt->second);
            config().blackboard->set("exhibitPostGestureMessage", postIt->second);
            config().blackboard->rootBlackboard()->set("exhibitLocation",      locationInfo.robotPose);
            config().blackboard->rootBlackboard()->set("exhibitGestureTarget", locationInfo.gestureTarget);

            logger_->info("Visiting: " + locationInfo.description);
            config().blackboard->set("visits", ++visits);

            return BT::NodeStatus::SUCCESS;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in SelectExhibit: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// IsListWithExhibit
// ——————————————
class IsListWithExhibit : public BT::ConditionNode, public BaseTreeNode {
public:
    IsListWithExhibit(const std::string &name,
                      const BT::NodeConfiguration &config)
      : BT::ConditionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("IsListWithExhibit Condition Node");
        try {
            int visits = 0;
            if (!config().blackboard->get("visits", visits)) {
                logger_->error("Unable to retrieve 'visits' from Blackboard");
                return BT::NodeStatus::FAILURE;
            }

            auto tour = KnowledgeManager::instance().getTourSpecification();
            if (visits < tour.getCurrentLocationCount()) {
                return BT::NodeStatus::SUCCESS;
            } else {
                logger_->info("ALL LOCATIONS VISITED");
                return BT::NodeStatus::FAILURE;
            }
        }
        catch (const std::exception &e) {
            logger_->error("Exception in IsListWithExhibit: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// RetrieveListOfExhibits
// ——————————————
class RetrieveListOfExhibits : public BT::SyncActionNode, public BaseTreeNode {
public:
    RetrieveListOfExhibits(const std::string &name,
                           const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("RetrieveListOfExhibits Action Node");
        try {
            auto tour = KnowledgeManager::instance().getTourSpecification();
            if (tour.getCurrentLocationCount() == 0) {
                logger_->error("No exhibits found");
                return BT::NodeStatus::FAILURE;
            }
            // initialize visit count in blackboard
            config().blackboard->set("visits", 0);
            return BT::NodeStatus::SUCCESS;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in RetrieveListOfExhibits: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// PerformDeicticGesture
// ——————————————
class PerformDeicticGesture : public BT::SyncActionNode, public BaseTreeNode {
public:
    PerformDeicticGesture(const std::string &name,
                          const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("PerformDeicticGesture Action Node");
        try {
            auto req = std::make_shared<cssr_system::srv::GestureExecutionPerformGesture::Request>();
            auto res = std::make_shared<cssr_system::srv::GestureExecutionPerformGesture::Response>();

            // read target from root blackboard
            Position3D target;
            if (!config().blackboard->rootBlackboard()->get("exhibitGestureTarget", target)) {
                logger_->error("Unable to retrieve exhibitGestureTarget from Blackboard");
                return BT::NodeStatus::FAILURE;
            }

            // fill request
            req->gesture_type     = "deictic";
            req->gesture_id       = Constants::DEICTIC_GESTURE_ID;
            req->gesture_duration = Constants::GESTURE_DURATION_MS;
            req->bow_nod_angle    = 0;
            req->location_x       = target.x;
            req->location_y       = target.y;
            req->location_z       = target.z;

            bool ok = callServiceSafely<cssr_system::srv::GestureExecutionPerformGesture>(
                         "/gestureExecution/perform_gesture", req, res, "PerformDeicticGesture");
            return (ok && res->gesture_success)
                 ? BT::NodeStatus::SUCCESS
                 : BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in PerformDeicticGesture: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};

// ——————————————
// PerformIconicGesture
// ——————————————

class PerformIconicGesture : public BT::SyncActionNode, public BaseTreeNode {
public:
    PerformIconicGesture(const std::string &name,
                         const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)    // now pulls node_ from config.blackboard
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("PerformIconicGesture Action Node");
        try {
            auto req  = std::make_shared<cssr_system::srv::GestureExecutionPerformGesture::Request>();
            auto res  = std::make_shared<cssr_system::srv::GestureExecutionPerformGesture::Response>();
            const std::string gestureType = name();  // instance name in XML

            // Common parameters
            req->gesture_type     = "iconic";
            req->gesture_duration = Constants::GESTURE_DURATION_MS;
            req->bow_nod_angle    = 0;
            req->location_x = req->location_y = req->location_z = 0.0;

            // Choose gesture_id based on the node’s instance name
            if      (gestureType == "welcome") req->gesture_id = Constants::WELCOME_GESTURE_ID;
            else if (gestureType == "goodbye") req->gesture_id = Constants::GOODBYE_GESTURE_ID;
            else {
                logger_->error("Undefined Iconic Gesture Type: " + gestureType);
                return BT::NodeStatus::FAILURE;
            }

            // Call the service via BaseTreeNode’s helper
            if (callServiceSafely<cssr_system::srv::GestureExecutionPerformGesture>(
                    "/gestureExecution/perform_gesture", req, res, "PerformIconicGesture"))
            {
                return res->gesture_success
                     ? BT::NodeStatus::SUCCESS
                     : BT::NodeStatus::FAILURE;
            }
            return BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in PerformIconicGesture: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// DescribeExhibitSpeech
// ——————————————

class DescribeExhibitSpeech : public BT::SyncActionNode, public BaseTreeNode {
public:
    DescribeExhibitSpeech(const std::string &name,
                          const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)    // now pulls node_ from the Blackboard
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("DescribeExhibit Action Node");
        try {
            // Prepare TTS request/response
            auto req = std::make_shared<cssr_system::srv::TextToSpeechSayText::Request>();
            auto res = std::make_shared<cssr_system::srv::TextToSpeechSayText::Response>();
            req->language = ConfigManager::instance().getLanguage();

            // Determine which message to speak based on this node's instance name
            const std::string nodeInstance = name();
            std::string message;
            if (nodeInstance == "1") {
                if (!config().blackboard->get("exhibitPreGestureMessage", message)) {
                    logger_->error("Unable to retrieve pre-gesture message from Blackboard");
                    return BT::NodeStatus::FAILURE;
                }
            }
            else if (nodeInstance == "2") {
                if (!config().blackboard->get("exhibitPostGestureMessage", message)) {
                    logger_->error("Unable to retrieve post-gesture message from Blackboard");
                    return BT::NodeStatus::FAILURE;
                }
            }
            else {
                logger_->warn("Invalid node instance for DescribeExhibitSpeech: " + nodeInstance);
                return BT::NodeStatus::FAILURE;
            }

            if (message.empty()) {
                logger_->warn("Empty message for DescribeExhibitSpeech");
                return BT::NodeStatus::FAILURE;
            }

            req->message = message;

            // Call the TTS service
            if (callServiceSafely<cssr_system::srv::TextToSpeechSayText>(
                    "/textToSpeech/say_text", req, res, "DescribeExhibitSpeech"))
            {
                if (res->success) {
                    // small pause after speaking
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                    return BT::NodeStatus::SUCCESS;
                }
            }
            return BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in DescribeExhibitSpeech: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};

// ——————————————
// IsMutualGazeDiscovered
// ——————————————

class IsMutualGazeDiscovered : public BT::ConditionNode, public BaseTreeNode {
private:
    // 0 = not seen yet, 1 = success, 2 = failure
    std::atomic<int> seekingStatus_{0};

    rclcpp::Subscription<cssr_system::msg::OvertAttentionMode>::SharedPtr subscriber_;

public:
    IsMutualGazeDiscovered(const std::string &name, 
                           const BT::NodeConfiguration &config)
      : BT::ConditionNode(name, config)
      , BaseTreeNode(config)   
    {
        // Subscribe once at construction; callback just updates atomic status
        subscriber_ = node_->create_subscription<cssr_system::msg::OvertAttentionMode>(
            "/overtAttention/mode", 10,
            [this](const cssr_system::msg::OvertAttentionMode::SharedPtr msg) {
                if (msg->state == "seeking") {
                    if (msg->value == 2) seekingStatus_.store(1);
                    else if (msg->value == 3) seekingStatus_.store(2);
                }
            });
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        // Called every tree tick; just check the status flag
        auto status = seekingStatus_.load();
        if (status == 1) {
            logger_->info("Mutual gaze detected");
            return BT::NodeStatus::SUCCESS;
        }
        if (status == 2) {
            logger_->info("Mutual gaze detection failed");
            return BT::NodeStatus::FAILURE;
        }
        // Still waiting for a message
        return BT::NodeStatus::RUNNING;
    }
};

// ——————————————
// IsVisitorResponseYes 
// ——————————————

class IsVisitorResponseYes : public BT::ConditionNode, public BaseTreeNode {
public:
    IsVisitorResponseYes(const std::string &name,
                         const BT::NodeConfiguration &config)
      : BT::ConditionNode(name, config)
      , BaseTreeNode(config)    // pulls node_ & logger_ from the Blackboard
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("IsVisitorResponseYes Condition Node");
        try {
            // Get the *root* blackboard to read shared values:
            auto rootBB = config().blackboard->rootBlackboard();

            std::string visitorResponse;
            if (rootBB->get("visitorResponse", visitorResponse) &&
                visitorResponse == "yes")
            {
                return BT::NodeStatus::SUCCESS;
            }
            return BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in IsVisitorResponseYes: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// IsASREnabled
// ——————————————
class IsASREnabled : public BT::ConditionNode, public BaseTreeNode {
public:
    IsASREnabled(const std::string &name,
                 const BT::NodeConfiguration &config)
      : BT::ConditionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("IsASREnabled Condition Node");
        return ConfigManager::instance().isAsrEnabled()
             ? BT::NodeStatus::SUCCESS
             : BT::NodeStatus::FAILURE;
    }
};

// ——————————————
// SetSpeechEvent
// ——————————————
class SetSpeechEvent : public BT::SyncActionNode, public BaseTreeNode {
public:
    SetSpeechEvent(const std::string &name,
                   const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SetSpeechEvent Action Node");
        try {
            auto req  = std::make_shared<cssr_system::srv::SpeechEventSetEnabled::Request>();
            auto res  = std::make_shared<cssr_system::srv::SpeechEventSetEnabled::Response>();
            req->status = name();  // use node’s instance name as the status

            bool ok = callServiceSafely<cssr_system::srv::SpeechEventSetEnabled>(
                        "/speechEvent/set_enabled", req, res, "SetSpeechEvent");
            return (ok && res->response)
                 ? BT::NodeStatus::SUCCESS
                 : BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in SetSpeechEvent: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// SetOvertAttentionMode
// ——————————————
class SetOvertAttentionMode : public BT::SyncActionNode, public BaseTreeNode {
public:
    SetOvertAttentionMode(const std::string &name,
                          const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SetOvertAttentionMode Action Node");
        try {
            auto req  = std::make_shared<cssr_system::srv::OvertAttentionSetMode::Request>();
            auto res  = std::make_shared<cssr_system::srv::OvertAttentionSetMode::Response>();
            const std::string state = name();
            req->state = state;

            if (state == "location") {
                Position3D target;
                if (!config().blackboard->rootBlackboard()
                           ->get("exhibitGestureTarget", target))
                {
                    logger_->error("Cannot get exhibitGestureTarget from Blackboard");
                    return BT::NodeStatus::FAILURE;
                }
                req->location_x = target.x;
                req->location_y = target.y;
                req->location_z = target.z;
            }

            bool ok = callServiceSafely<cssr_system::srv::OvertAttentionSetMode>(
                        "/overtAttention/set_mode", req, res, "SetOvertAttentionMode");
            return (ok && res->mode_set_success)
                 ? BT::NodeStatus::SUCCESS
                 : BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in SetOvertAttentionMode: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// SetAnimateBehavior
// ——————————————
class SetAnimateBehavior : public BT::SyncActionNode, public BaseTreeNode {
public:
    SetAnimateBehavior(const std::string &name,
                       const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SetAnimateBehavior Action Node");
        try {
            auto req  = std::make_shared<cssr_system::srv::AnimateBehaviorSetActivation::Request>();
            auto res  = std::make_shared<cssr_system::srv::AnimateBehaviorSetActivation::Response>();
            req->state = name();

            bool ok = callServiceSafely<cssr_system::srv::AnimateBehaviorSetActivation>(
                        "/animateBehaviour/setActivation", req, res, "SetAnimateBehavior");
            return (ok && res->success == "1")
                 ? BT::NodeStatus::SUCCESS
                 : BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in SetAnimateBehavior: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


//=============================================================================

// ——————————————
// ResetRobotPose
// ——————————————
class ResetRobotPose : public BT::SyncActionNode, public BaseTreeNode {
public:
    ResetRobotPose(const std::string &name,
                   const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("ResetRobotPose Action Node");
        try {
            auto req = std::make_shared<cssr_system::srv::RobotLocalizationResetPose::Request>();
            auto res = std::make_shared<cssr_system::srv::RobotLocalizationResetPose::Response>();

            bool ok = callServiceSafely<cssr_system::srv::RobotLocalizationResetPose>(
                        "/robotLocalization/reset_pose", req, res, "ResetRobotPose");
            return (ok && res->success)
                 ? BT::NodeStatus::SUCCESS
                 : BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in ResetRobotPose: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};


// ——————————————
// PressYesNoDialogue
// ——————————————
class PressYesNoDialogue : public BT::SyncActionNode, public BaseTreeNode {
public:
    PressYesNoDialogue(const std::string &name,
                       const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("PressYesNoDialogue Action Node");
        try {
            auto req = std::make_shared<cssr_system::srv::TabletEventPromptAndGetResponse::Request>();
            auto res = std::make_shared<cssr_system::srv::TabletEventPromptAndGetResponse::Response>();
            req->message = "'Yes'|'No'";

            bool ok = callServiceSafely<cssr_system::srv::TabletEventPromptAndGetResponse>(
                        "/tabletEvent/prompt_and_get_response", req, res, "PressYesNoDialogue");
            return (ok && res->success)
                 ? BT::NodeStatus::SUCCESS
                 : BT::NodeStatus::FAILURE;
        }
        catch (const std::exception &e) {
            logger_->error("Exception in PressYesNoDialogue: " + std::string(e.what()));
            return BT::NodeStatus::FAILURE;
        }
    }
};

// ——————————————
// HandleFallBack
// ——————————————
class HandleFallBack : public BT::SyncActionNode, public BaseTreeNode {
public:
    HandleFallBack(const std::string &name,
                   const BT::NodeConfiguration &config)
      : BT::SyncActionNode(name, config)
      , BaseTreeNode(config)
    {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("HandleFallBack Action Node");
        return BT::NodeStatus::SUCCESS;
    }
};

//=============================================================================
// Tree Initialization Function
//=============================================================================

namespace behavior_controller {

BT::Tree initializeTree(const std::string &scenario,
                        std::shared_ptr<rclcpp::Node> node_handle)
{
    // — 1) Find the XML file —
    auto pkg = ament_index_cpp::get_package_share_directory("cssr_system");
    std::string xml = pkg + "/behaviorController/data/" + scenario + ".xml";
    if (!std::ifstream(xml).good()) {
        throw std::runtime_error("Tree XML not found: " + xml);
    }

    // — 2) Make a Blackboard and store the ROS node in it —
    auto blackboard = BT::Blackboard::create();
    blackboard->set("node", node_handle);

    // — 3) Build the factory, pull in ROS2 wrappers, register your nodes —
    BT::BehaviorTreeFactory factory;
    // registerRos2Nodes(factory);  // brings in <RosService/>, <NavigateToPose/>, etc.

    // Register RosServiceNode-based implementations
    factory.registerNodeType<SayTextRosService>("SayTextRosService");
    factory.registerNodeType<NavigateRosService>("NavigateRosService");
    factory.registerNodeType<PerformDeicticGestureRosService>("PerformDeicticGestureRosService");
    factory.registerNodeType<PerformIconicGestureRosService>("PerformIconicGestureRosService");
    factory.registerNodeType<DescribeExhibitSpeechRosService>("DescribeExhibitSpeechRosService");
    factory.registerNodeType<SetSpeechEventRosService>("SetSpeechEventRosService");
    factory.registerNodeType<SetOvertAttentionModeRosService>("SetOvertAttentionModeRosService");
    factory.registerNodeType<SetAnimateBehaviorRosService>("SetAnimateBehaviorRosService");
    factory.registerNodeType<ResetRobotPoseRosService>("ResetRobotPoseRosService");
    factory.registerNodeType<PressYesNoDialogueRosService>("PressYesNoDialogueRosService");
    
    // Keep original registrations for backward compatibility
    factory.registerNodeType<StartOfTree>("StartOfTree");
    factory.registerNodeType<SayText>("SayText");
    factory.registerNodeType<Navigate>("Navigate");
    // factory.registerNodeType<GetVisitorResponse>("GetVisitorResponse");
    factory.registerNodeType<IsVisitorDiscovered>("IsVisitorDiscovered");
    factory.registerNodeType<SelectExhibit>("SelectExhibit");
    factory.registerNodeType<IsListWithExhibit>("IsListWithExhibit");
    factory.registerNodeType<RetrieveListOfExhibits>("RetrieveListOfExhibits");
    factory.registerNodeType<PerformDeicticGesture>("PerformDeicticGesture");
    factory.registerNodeType<PerformIconicGesture>("PerformIconicGesture");
    factory.registerNodeType<DescribeExhibitSpeech>("DescribeExhibitSpeech");
    factory.registerNodeType<IsMutualGazeDiscovered>("IsMutualGazeDiscovered");
    factory.registerNodeType<IsVisitorResponseYes>("IsVisitorResponseYes");
    // factory.registerNodeType<IsASREnabled>("IsASREnabled");
    factory.registerNodeType<SetSpeechEvent>("SetSpeechEvent");
    factory.registerNodeType<SetOvertAttentionMode>("SetOvertAttentionMode");
    factory.registerNodeType<SetAnimateBehavior>("SetAnimateBehavior");
    factory.registerNodeType<ResetRobotPose>("ResetRobotPose");
    factory.registerNodeType<PressYesNoDialogue>("PressYesNoDialogue");
    factory.registerNodeType<HandleFallBack>("HandleFallBack");
    
    // — 4) Load and return the tree, passing in the blackboard —
    return factory.createTreeFromFile(xml, blackboard);
}

}  // namespace behavior_controller

