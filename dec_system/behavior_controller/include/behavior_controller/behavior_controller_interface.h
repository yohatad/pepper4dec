/* behaviorControllerInterface.h 
 *
 * Author: Yohannes Tadesse Haile
 * Date: Feb 07, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#ifndef BEHAVIOR_CONTROLLER_INTERFACE_H
#define BEHAVIOR_CONTROLLER_INTERFACE_H

// ROS includes
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <std_msgs/msg/string.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

// BehaviorTree.CPP includes
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>

// BehaviorTree.ROS2 includes
#include <behaviortree_ros2/bt_service_node.hpp>
#include <behaviortree_ros2/bt_action_node.hpp>
#include <behaviortree_ros2/ros_node_params.hpp>
#include <behaviortree_ros2/plugins.hpp>

// Standard includes
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <optional>
#include <csignal>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <thread>
#include <iomanip>

// YAML includes
#include <yaml-cpp/yaml.h>

// Custom message/service/action includes from dec_interfaces package
// Messages
#include "dec_interfaces/msg/face_detection.hpp"

// naoqi_bridge_msgs actions
#include "naoqi_bridge_msgs/action/speech_with_feedback.hpp"

// Actions
#include "dec_interfaces/action/tts.hpp"
#include "dec_interfaces/action/gesture.hpp"
#include "dec_interfaces/action/animate_behavior.hpp"
#include "dec_interfaces/action/speech_recognition.hpp"
#include "dec_interfaces/action/conversation_manager.hpp"

// Services
#include "dec_interfaces/srv/conversation_manager_prompt.hpp"
#include <std_srvs/srv/trigger.hpp>
#include <std_srvs/srv/set_bool.hpp>

//=============================================================================
// Data Structures
//=============================================================================
struct Position3D {
    double x = 0.0, y = 0.0, z = 0.0;
    Position3D() = default;
    Position3D(double x_val, double y_val, double z_val = 0.0) 
        : x(x_val), y(y_val), z(z_val) {}
};

struct RobotPose {
    double x = 0.0, y = 0.0, theta = 0.0;
    RobotPose() = default;
    RobotPose(double x_val, double y_val, double theta_val) 
        : x(x_val), y(y_val), theta(theta_val) {}
};

struct LocationInfo {
    std::string description;
    RobotPose robotPose;
    Position3D gestureTarget;
    std::string gestureMessage;
};

struct TourSpec {
    std::vector<std::string> locationIds;
    size_t getLocationCount() const { return locationIds.size(); }
};

//=============================================================================
// Core Managers (Singletons)
//=============================================================================

// Configuration Manager
class ConfigManager {
public:
    static ConfigManager& instance();
    
    [[nodiscard]] bool loadFromFile(const std::string& configPath);
    
    // Getters
    bool isVerbose() const;
    std::string getScenarioSpecification() const;
    std::string getCultureKnowledgeBasePath() const;
    std::string getEnvironmentKnowledgeBasePath() const;

private:
    ConfigManager() = default;
    bool verbose = false;
    std::string scenarioSpecification = "lab_tour";
    std::string cultureKnowledgeBasePath = "cultureKnowledgeBase.yaml";
    std::string environmentKnowledgeBasePath = "labEnvironmentKnowledgeBase.yaml";
    
    // Non-copyable
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
};

// Knowledge Base Manager
class KnowledgeManager {
public:
    static KnowledgeManager& instance();
    
    [[nodiscard]] bool loadFromPackage(const std::string& packagePath);

    std::string getUtilityPhrase(const std::string& phraseId);
    LocationInfo getLocationInfo(const std::string& locationId);
    TourSpec getTourSpecification();

private:
    KnowledgeManager() = default;
    std::unordered_map<std::string, std::string> utilityPhrases;
    std::unordered_map<std::string, LocationInfo> locations;
    std::optional<TourSpec> tourSpec;
    bool loaded = false;
    
    // Non-copyable
    KnowledgeManager(const KnowledgeManager&) = delete;
    KnowledgeManager& operator=(const KnowledgeManager&) = delete;
};

//=============================================================================
// Utility Classes
//=============================================================================

// Simplified Logger
class Logger {
public:
    explicit Logger(std::shared_ptr<rclcpp::Node> node);
    
    void info(const std::string& msg);
    void warn(const std::string& msg);
    void error(const std::string& msg);
    void debug(const std::string& msg);

private:
    std::shared_ptr<rclcpp::Node> node;
    std::string formatMessage(const std::string& msg);
};

// Service Manager (for non-BT service calls)
class ServiceManager {
public:
    explicit ServiceManager(std::shared_ptr<rclcpp::Node> node);

    [[nodiscard]] bool checkServicesAvailable(const std::vector<std::string>& services);
    [[nodiscard]] bool waitForService(const std::string& serviceName,
                                     std::chrono::seconds timeout = std::chrono::seconds(5));

private:
    std::shared_ptr<rclcpp::Node> node;
};

// Topic Monitor
class TopicMonitor {
public:
    explicit TopicMonitor(std::shared_ptr<rclcpp::Node> node);
    
    [[nodiscard]] bool isTopicAvailable(const std::string& topicName);
    [[nodiscard]] bool checkTopicsAvailable(const std::vector<std::string>& topics);
    [[nodiscard]] bool waitForTopic(const std::string& topicName,
                                   std::chrono::seconds timeout = std::chrono::seconds(5));

private:
    std::shared_ptr<rclcpp::Node> node;
};

// Text Utilities
class TextUtils {
public:
    static bool containsAnyWord(const std::string& text, const std::vector<std::string>& words);
    static std::string toLowerCase(const std::string& text);
    static std::vector<std::string> split(const std::string& text, char delimiter);
    static std::string trim(const std::string& text);
};

//=============================================================================
// BehaviorTree Action Nodes
//=============================================================================

// Wraps dec_interfaces::action::AnimateBehavior
class AnimateBehaviorNode
    : public BT::RosActionNode<dec_interfaces::action::AnimateBehavior>
{
public:
    AnimateBehaviorNode(const std::string& name,
                        const BT::NodeConfig& config,
                        const BT::RosNodeParams& params)
        : BT::RosActionNode<dec_interfaces::action::AnimateBehavior>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setGoal(Goal& goal) override;
    BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override;
    BT::NodeStatus onResultReceived(const WrappedResult& result) override;
    BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
};

// Wraps dec_interfaces::action::Gesture
class GestureNode
    : public BT::RosActionNode<dec_interfaces::action::Gesture>
{
public:
    GestureNode(const std::string& name,
                const BT::NodeConfig& config,
                const BT::RosNodeParams& params)
        : BT::RosActionNode<dec_interfaces::action::Gesture>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setGoal(Goal& goal) override;
    BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override;
    BT::NodeStatus onResultReceived(const WrappedResult& result) override;
    BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
};

// Wraps nav2_msgs::action::NavigateToPose  →  Nav2 /navigate_to_pose server
class Navigate
    : public BT::RosActionNode<nav2_msgs::action::NavigateToPose>
{
public:
    Navigate(const std::string& name,
                      const BT::NodeConfig& config,
                      const BT::RosNodeParams& params)
        : BT::RosActionNode<nav2_msgs::action::NavigateToPose>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setGoal(Goal& goal) override;
    BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override;
    BT::NodeStatus onResultReceived(const WrappedResult& result) override;
    BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
};

// Wraps dec_interfaces::action::SpeechRecognition
class SpeechRecognitionNode
    : public BT::RosActionNode<dec_interfaces::action::SpeechRecognition>
{
public:
    SpeechRecognitionNode(const std::string& name,
                          const BT::NodeConfig& config,
                          const BT::RosNodeParams& params)
        : BT::RosActionNode<dec_interfaces::action::SpeechRecognition>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setGoal(Goal& goal) override;
    BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override;
    BT::NodeStatus onResultReceived(const WrappedResult& result) override;
    BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
};

// Wraps dec_interfaces::action::ConversationManager
//
// Output ports written from the action result:
//   response   – full generated answer text
//   intent     – classified intent (ASK_EXHIBIT_QUESTION | NAVIGATION_REQUEST |
//                SOCIAL_SMALL_TALK | OFF_TOPIC | STOP | AFFIRMATIVE | NEGATIVE | …)
//   confidence – LLM confidence in the intent (0.0 – 1.0)
class ConversationManagerNode
    : public BT::RosActionNode<dec_interfaces::action::ConversationManager>
{
public:
    ConversationManagerNode(const std::string& name,
                            const BT::NodeConfig& config,
                            const BT::RosNodeParams& params)
        : BT::RosActionNode<dec_interfaces::action::ConversationManager>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setGoal(Goal& goal) override;
    BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override;
    BT::NodeStatus onResultReceived(const WrappedResult& result) override;
    BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
};

// Wraps naoqi_bridge_msgs::action::SpeechWithFeedback
class SpeechWithFeedbackNode
    : public BT::RosActionNode<naoqi_bridge_msgs::action::SpeechWithFeedback>
{
public:
    SpeechWithFeedbackNode(const std::string& name,
                           const BT::NodeConfig& config,
                           const BT::RosNodeParams& params)
        : BT::RosActionNode<naoqi_bridge_msgs::action::SpeechWithFeedback>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setGoal(Goal& goal) override;
    BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override;
    BT::NodeStatus onResultReceived(const WrappedResult& result) override;
    BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
};

// Wraps dec_interfaces::action::TTS
// Sends text to the /text_to_speech action server (text_to_speech node), which synthesises
// audio via Kokoro or ElevenLabs and plays it through the configured backend.
// Blocks until playback is complete.
class TTSNode
    : public BT::RosActionNode<dec_interfaces::action::TTS>
{
public:
    TTSNode(const std::string& name,
            const BT::NodeConfig& config,
            const BT::RosNodeParams& params)
        : BT::RosActionNode<dec_interfaces::action::TTS>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setGoal(Goal& goal) override;
    BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override;
    BT::NodeStatus onResultReceived(const WrappedResult& result) override;
    BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
};

// Calls /animate_behavior/stop (std_srvs::srv::Trigger) to immediately stop animation.
// Returns SUCCESS if the service reports success, FAILURE otherwise.
class StopAnimateBehavior
    : public BT::RosServiceNode<std_srvs::srv::Trigger>
{
public:
    StopAnimateBehavior(const std::string& name,
                        const BT::NodeConfig& config,
                        const BT::RosNodeParams& params)
        : BT::RosServiceNode<std_srvs::srv::Trigger>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setRequest(Request::SharedPtr& request) override;
    BT::NodeStatus onResponseReceived(const Response::SharedPtr& response) override;
    BT::NodeStatus onFailure(BT::ServiceNodeErrorCode error) override;
};

// Calls /overt_attention/set_enabled (std_srvs::srv::SetBool) to enable or disable overt attention.
// Input port 'enabled' (bool): true = enable, false = disable.
// Returns SUCCESS if the service confirms the change, FAILURE otherwise.
class SetOvertAttention
    : public BT::RosServiceNode<std_srvs::srv::SetBool>
{
public:
    SetOvertAttention(const std::string& name,
                      const BT::NodeConfig& config,
                      const BT::RosNodeParams& params)
        : BT::RosServiceNode<std_srvs::srv::SetBool>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setRequest(Request::SharedPtr& request) override;
    BT::NodeStatus onResponseReceived(const Response::SharedPtr& response) override;
    BT::NodeStatus onFailure(BT::ServiceNodeErrorCode error) override;
};

// Subscribes to /speech_event/text (standalone mode) and blocks (RUNNING) until
// a new transcription arrives after this node started, then returns SUCCESS.
// Requires speech_event running with action_server: false.
class ListenForSpeech : public BT::StatefulActionNode
{
public:
    ListenForSpeech(const std::string& name,
                    const BT::NodeConfig& config,
                    std::shared_ptr<rclcpp::Node> node);

    static BT::PortsList providedPorts();
    BT::NodeStatus onStart() override;
    BT::NodeStatus onRunning() override;
    void onHalted() override;

private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
    std::string latestText_;
    bool newTextAvailable_ = false;
    std::mutex mutex_;
};

// Calls /speech_event/set_enabled (std_srvs::srv::SetBool) to mute or unmute
// the speech recognition mic (e.g. disable during TTS, re-enable after).
// Input port 'enabled' (bool): true = listen, false = mute.
class SetSpeechListening
    : public BT::RosServiceNode<std_srvs::srv::SetBool>
{
public:
    SetSpeechListening(const std::string& name,
                       const BT::NodeConfig& config,
                       const BT::RosNodeParams& params)
        : BT::RosServiceNode<std_srvs::srv::SetBool>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setRequest(Request::SharedPtr& request) override;
    BT::NodeStatus onResponseReceived(const Response::SharedPtr& response) override;
    BT::NodeStatus onFailure(BT::ServiceNodeErrorCode error) override;
};

// Subscribes to /face_detection/data and blocks (RUNNING) until face(s) are present,
// then returns SUCCESS. Never times out — runs indefinitely until condition is met.
class CheckFaceDetected : public BT::StatefulActionNode
{
public:
    CheckFaceDetected(const std::string& name,
                      const BT::NodeConfig& config,
                      std::shared_ptr<rclcpp::Node> node);

    static BT::PortsList providedPorts();
    BT::NodeStatus onStart() override;
    BT::NodeStatus onRunning() override;
    void onHalted() override;

private:
    BT::NodeStatus checkLatestMessage();

    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Subscription<dec_interfaces::msg::FaceDetection>::SharedPtr sub_;
    dec_interfaces::msg::FaceDetection::SharedPtr latestMsg_;
    std::mutex mutex_;
};

// Blocks (RUNNING) until at least one face appears on /face_detection/data.
// Returns FAILURE when the "timeout" port (seconds) expires without a face.
class IsVisitorDiscovered : public BT::StatefulActionNode
{
public:
    IsVisitorDiscovered(const std::string& name,
                        const BT::NodeConfig& config,
                        std::shared_ptr<rclcpp::Node> node);

    static BT::PortsList providedPorts();
    BT::NodeStatus onStart() override;
    BT::NodeStatus onRunning() override;
    void onHalted() override;

private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Subscription<dec_interfaces::msg::FaceDetection>::SharedPtr sub_;
    dec_interfaces::msg::FaceDetection::SharedPtr latestMsg_;
    rclcpp::Time deadline_;
    std::mutex mutex_;
};

// Blocks (RUNNING) until mutual gaze is detected on /face_detection/data.
// Returns FAILURE when the "timeout" port (seconds) expires.
class IsMutualGazeDiscovered : public BT::StatefulActionNode
{
public:
    IsMutualGazeDiscovered(const std::string& name,
                           const BT::NodeConfig& config,
                           std::shared_ptr<rclcpp::Node> node);

    static BT::PortsList providedPorts();
    BT::NodeStatus onStart() override;
    BT::NodeStatus onRunning() override;
    void onHalted() override;

private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Subscription<dec_interfaces::msg::FaceDetection>::SharedPtr sub_;
    dec_interfaces::msg::FaceDetection::SharedPtr latestMsg_;
    rclcpp::Time deadline_;
    rclcpp::Time gazeStart_;   // when continuous mutual gaze began; zero if not currently gazing
    std::mutex mutex_;
};

// Blocks (RUNNING) until a new transcription arrives on /speech_event/text.
// Returns FAILURE when the "timeout" port (seconds) expires without speech.
// Writes the recognised text to the "visitor_response" output port.
class GetVisitorResponse : public BT::StatefulActionNode
{
public:
    GetVisitorResponse(const std::string& name,
                       const BT::NodeConfig& config,
                       std::shared_ptr<rclcpp::Node> node);

    static BT::PortsList providedPorts();
    BT::NodeStatus onStart() override;
    BT::NodeStatus onRunning() override;
    void onHalted() override;

private:
    std::shared_ptr<rclcpp::Node> node_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
    std::string latestText_;
    bool newTextAvailable_ = false;
    rclcpp::Time deadline_;
    std::mutex mutex_;
};

//=============================================================================
// Tour Loop Nodes  (SyncActionNode — no ROS, pure blackboard + KnowledgeManager)
//
// All four nodes share the blackboard key "exhibit_queue"
// (a std::vector<std::string> of location IDs in visit order).
//=============================================================================

// Reads KnowledgeManager::getTourSpecification().locationIds and writes the
// ordered vector to the "exhibit_list" output port (default key: exhibit_queue).
// Returns FAILURE if the tour specification is empty.
class RetrieveListOfExhibits : public BT::SyncActionNode
{
public:
    RetrieveListOfExhibits(const std::string& name, const BT::NodeConfig& config)
        : BT::SyncActionNode(name, config) {}

    static BT::PortsList providedPorts();
    BT::NodeStatus tick() override;
};

// Condition: returns SUCCESS if exhibit_queue is non-empty, FAILURE otherwise.
// Reads the queue via the "exhibit_list" input port (default key: exhibit_queue).
class IsListWithExhibit : public BT::SyncActionNode
{
public:
    IsListWithExhibit(const std::string& name, const BT::NodeConfig& config)
        : BT::SyncActionNode(name, config) {}

    static BT::PortsList providedPorts();
    BT::NodeStatus tick() override;
};

// Reads the front of exhibit_queue, looks up the corresponding LocationInfo in
// KnowledgeManager, and writes 7 blackboard keys consumed by NavigateToLocation
// and PresentExhibit:
//   {exhibit_speech}         ← LocationInfo.gestureMessage
//   {exhibit_goal_x/y/theta} ← LocationInfo.robotPose
//   {exhibit_location_x/y/z} ← LocationInfo.gestureTarget
// Does NOT pop the queue — that is deferred to PopExhibitFromList.
class SelectExhibit : public BT::SyncActionNode
{
public:
    SelectExhibit(const std::string& name, const BT::NodeConfig& config)
        : BT::SyncActionNode(name, config) {}

    static BT::PortsList providedPorts();
    BT::NodeStatus tick() override;
};

// Removes the front entry from exhibit_queue and writes the remaining count to
// the "remaining_count" output port. Always returns SUCCESS.
class PopExhibitFromList : public BT::SyncActionNode
{
public:
    PopExhibitFromList(const std::string& name, const BT::NodeConfig& config)
        : BT::SyncActionNode(name, config) {}

    static BT::PortsList providedPorts();
    BT::NodeStatus tick() override;
};

//=============================================================================
// Utility Nodes  (SyncActionNode — no ROS, pure blackboard/logging)
//=============================================================================

// Logs a message at the requested level (debug | info | warn | error).
// Always returns SUCCESS.
class LogEvent : public BT::SyncActionNode
{
public:
    LogEvent(const std::string& name, const BT::NodeConfig& config)
        : BT::SyncActionNode(name, config) {}

    static BT::PortsList providedPorts();
    BT::NodeStatus tick() override;
};

// Writes a string value to an arbitrary blackboard key.
// Always returns SUCCESS.
class SetBlackboardValue : public BT::SyncActionNode
{
public:
    SetBlackboardValue(const std::string& name, const BT::NodeConfig& config)
        : BT::SyncActionNode(name, config) {}

    static BT::PortsList providedPorts();
    BT::NodeStatus tick() override;
};

// Reads blackboard key "key" and compares its string value to "expected".
// Returns SUCCESS on match, FAILURE otherwise.
class CheckBlackboard : public BT::SyncActionNode
{
public:
    CheckBlackboard(const std::string& name, const BT::NodeConfig& config)
        : BT::SyncActionNode(name, config) {}

    static BT::PortsList providedPorts();
    BT::NodeStatus tick() override;
};

// Sends the raw visitor utterance to ConversationManager. The system prompt
// classifies AFFIRMATIVE utterances with answer="yes" and NEGATIVE with answer="no".
// Returns SUCCESS if response == "yes", FAILURE otherwise.
class IsVisitorResponseYes
    : public BT::RosActionNode<dec_interfaces::action::ConversationManager>
{
public:
    IsVisitorResponseYes(const std::string& name,
                         const BT::NodeConfig& config,
                         const BT::RosNodeParams& params)
        : BT::RosActionNode<dec_interfaces::action::ConversationManager>(name, config, params) {}

    static BT::PortsList providedPorts();
    bool setGoal(Goal& goal) override;
    BT::NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override;
    BT::NodeStatus onResultReceived(const WrappedResult& result) override;
    BT::NodeStatus onFailure(BT::ActionNodeErrorCode error) override;
};

//=============================================================================
// Function Declarations
//=============================================================================
namespace behavior_controller {

/**
 * @brief Build and register all ROS2-aware and custom BehaviorTree nodes,
 *        load the XML file for the given scenario, and return a ready-to-tick tree.
 *
 * This function creates a BehaviorTreeFactory, registers all custom nodes
 * (both RosServiceNode-based and BtActionNode-based), and loads the XML
 * behavior tree definition for the specified scenario.
 *
 * @param scenario      Base name (without ".xml") of the tree file under data/
 * @param node_handle   Shared pointer to your ROS2 node (used by all BT nodes)
 * @return BT::Tree     The fully constructed behavior tree
 * @throws std::runtime_error if the XML file cannot be found or loaded
 */
BT::Tree initializeTree(const std::string& scenario,
                        std::shared_ptr<rclcpp::Node> node_handle);

/**
 * @brief Validate the format of an environment knowledge base YAML file.
 *
 * Checks that all required top-level keys exist, that every location referenced
 * in tour_specification has a corresponding entry in locations, and that each
 * location entry contains all mandatory fields with legal values:
 *   - robot_location_description  (non-empty string)
 *   - robot_location_pose         (map: x, y numeric; theta in [0, 360])
 *   - gesture_target              (map: x, y, z numeric; z >= 0)
 *   - gesture_message_english     (string, may be empty)
 *   - cultural_knowledge          (non-empty sequence of non-empty strings)
 *
 * Errors are logged via RCLCPP_ERROR and the function returns false on the
 * first structural violation, or after collecting all field-level errors.
 *
 * @param filePath  Absolute path to the YAML file to validate
 * @return true if the file is fully valid, false otherwise
 */
[[nodiscard]] bool validateEnvironmentKnowledgeBase(const std::string& filePath);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Log system information (available services/topics and active config)
 * @param node The ROS2 node to query
 */
void logSystemInfo(std::shared_ptr<rclcpp::Node> node);

/**
 * @brief Check if a file exists at the given path
 * @param filepath Path to the file
 * @return true if file exists and is readable, false otherwise
 */
[[nodiscard]] bool fileExists(const std::string& filepath);

/**
 * @brief Get absolute path to a file relative to the package share directory
 * @param relativePath Path relative to the package data/ directory
 * @return Absolute path to the file
 */
std::string getPackageDataPath(const std::string& relativePath);

/**
 * @brief Print node name, namespace, and fully-qualified name to logs
 * @param node The ROS2 node
 */
void printNodeInfo(std::shared_ptr<rclcpp::Node> node);

/**
 * @brief Convert BehaviorTree NodeStatus to a human-readable string
 * @param status The NodeStatus to convert
 * @return String representation ("SUCCESS", "FAILURE", "RUNNING", "IDLE", "UNKNOWN")
 */
std::string nodeStatusToString(BT::NodeStatus status);

} // namespace behavior_controller

//=============================================================================
// BehaviorControllerLifecycleNode
//
// Lifecycle state machine:
//   UNCONFIGURED → on_configure:  load config + knowledge base, build BT tree
//   INACTIVE     → on_activate:   start 50 Hz tick timer
//   ACTIVE       → on_deactivate: cancel tick timer (tree stays built)
//   INACTIVE     → on_cleanup:    halt + destroy BT tree
//   any          → on_shutdown:   cancel timer + halt tree
//
// Why a companion bt_node_?
//   BT::RosNodeParams requires std::shared_ptr<rclcpp::Node>.
//   rclcpp_lifecycle::LifecycleNode does NOT inherit rclcpp::Node, so it
//   cannot be passed directly.  bt_node_ is a plain rclcpp::Node used
//   exclusively for BT action/service clients and topic subscriptions.
//   Both lc_node and bt_node_ are added to the MultiThreadedExecutor in main().
//=============================================================================

class BehaviorControllerLifecycleNode : public rclcpp_lifecycle::LifecycleNode
{
public:
    using CallbackReturn =
        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

    explicit BehaviorControllerLifecycleNode();

    /// Expose the companion node so main() can add it to the executor.
    rclcpp::Node::SharedPtr get_bt_node() const { return bt_node_; }

    // ── Lifecycle callbacks ─────────────────────────────────────────────────
    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_shutdown  (const rclcpp_lifecycle::State& state) override;

private:
    /// Companion plain node — used by all BT nodes (BT::RosNodeParams,
    /// StatefulActionNode subscriptions, etc.)
    rclcpp::Node::SharedPtr bt_node_;

    /// The live behavior tree (empty until on_configure succeeds).
    BT::Tree tree_;

    /// 50 Hz tick timer — created in on_activate, cancelled in on_deactivate.
    rclcpp::TimerBase::SharedPtr tick_timer_;

    /// Guard: true only after initializeTree() succeeds inside on_configure.
    bool tree_initialized_ = false;
};

#endif // BEHAVIOR_CONTROLLER_INTERFACE_H