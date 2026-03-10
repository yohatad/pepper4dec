/* behaviorControllerInterface.h 
 *
 * Author: Yohannes Tadesse Haile
 * Date: Feb 07, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#ifndef BEHAVIOR_CONTROLLER_INTERFACE_H
#define BEHAVIOR_CONTROLLER_INTERFACE_H

// ROS includes
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
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
    std::unordered_map<std::string, std::string> preMessages;  
    std::unordered_map<std::string, std::string> postMessages;
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
    std::string getLanguage() const;
    std::string getScenarioSpecification() const;
    std::string getCultureKnowledgeBasePath() const;
    std::string getEnvironmentKnowledgeBasePath() const;

private:
    ConfigManager() = default;
    bool verbose = false;
    std::string language = "English";
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
    
    std::string getUtilityPhrase(const std::string& phraseId, const std::string& language = "");
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

// Calls animate_behavior/stop (std_srvs::srv::Trigger) to immediately stop animation.
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

// Calls /attn/set_enabled (std_srvs::srv::SetBool) to enable or disable overt attention.
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

// Subscribes to /faceDetection/data and blocks (RUNNING) until face(s) are present,
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
 *   - pre_gesture_message_english / pre_gesture_message_kinyarwanda  (non-empty strings)
 *   - post_gesture_message_english / post_gesture_message_kinyarwanda (non-empty strings)
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
 * @brief Check if a language is supported
 * @param language The language name to check (e.g. "English", "Kinyarwanda")
 * @return true if supported, false otherwise
 */
[[nodiscard]] bool isValidLanguage(const std::string& language);

/**
 * @brief Get list of supported languages
 * @return Vector of supported language names
 */
std::vector<std::string> getSupportedLanguages();

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

#endif // BEHAVIOR_CONTROLLER_INTERFACE_H