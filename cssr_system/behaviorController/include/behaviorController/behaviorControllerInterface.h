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

// Custom message/service/action includes from cssr_interfaces package
// Messages
#include "cssr_interfaces/msg/face_detection.hpp"
#include "cssr_interfaces/msg/object_detection.hpp"
#include "cssr_interfaces/msg/overt_attention_status.hpp"
// Actions
#include "cssr_interfaces/action/tts.hpp"
#include "cssr_interfaces/action/gesture.hpp"
#include "cssr_interfaces/action/navigation.hpp"
#include "cssr_interfaces/action/animate_behavior.hpp"
#include "cssr_interfaces/action/speech_recognition.hpp"
// Services
#include "cssr_interfaces/srv/overt_attention_set_mode.hpp"
#include "cssr_interfaces/srv/animate_behavior_set_activation.hpp"
#include "cssr_interfaces/srv/conversation_manager_prompt.hpp"

//=============================================================================
// Constants
//=============================================================================
namespace Constants {
    constexpr int GESTURE_DURATION_MS = 3000;
    constexpr int SERVICE_TIMEOUT_SEC = 5;
    constexpr int RESPONSE_TIMEOUT_SEC = 10;
    constexpr int VISITOR_RESPONSE_TIMEOUT_SEC = 5;
    constexpr double LOOP_RATE_HZ = 10.0;
    constexpr int WELCOME_GESTURE_ID = 1;
    constexpr int GOODBYE_GESTURE_ID = 3;
    constexpr int DEICTIC_GESTURE_ID = 1;
}

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
    int getCurrentLocationCount() const { return locationIds.size(); }
};

//=============================================================================
// Core Managers (Singletons)
//=============================================================================

// Configuration Manager
class ConfigManager {
public:
    static ConfigManager& instance();
    
    bool loadFromFile(const std::string& configPath);
    
    // Getters
    bool isVerbose() const;
    bool isAsrEnabled() const;
    bool isTestMode() const;
    std::string getLanguage() const;
    std::string getNodeName() const;
    std::string getCultureKnowledgeBasePath() const;
    std::string getEnvironmentKnowledgeBasePath() const;

private:
    ConfigManager() = default;
    mutable std::mutex mutex_;
    bool verbose_ = false;
    bool asrEnabled_ = false;
    bool testMode_ = false;
    std::string language_ = "English";
    std::string nodeName_ = "behaviorController";
    
    // Non-copyable
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
};

// Knowledge Base Manager
class KnowledgeManager {
public:
    static KnowledgeManager& instance();
    
    bool loadFromPackage(const std::string& packagePath);
    
    std::string getUtilityPhrase(const std::string& phraseId, const std::string& language = "");
    LocationInfo getLocationInfo(const std::string& locationId);
    TourSpec getTourSpecification();

private:
    KnowledgeManager() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::string> utilityPhrases_;
    std::unordered_map<std::string, LocationInfo> locations_;
    std::optional<TourSpec> tourSpec_;
    bool loaded_ = false;
    
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
    std::shared_ptr<rclcpp::Node> node_;
    std::string formatMessage(const std::string& msg);
};

// Service Manager (for non-BT service calls)
class ServiceManager {
public:
    explicit ServiceManager(std::shared_ptr<rclcpp::Node> node);
    
    template<typename ServiceType>
    bool callService(const std::string& serviceName, 
                    typename ServiceType::Request::SharedPtr request,
                    typename ServiceType::Response::SharedPtr& response);
    
    bool checkServicesAvailable(const std::vector<std::string>& services);
    bool waitForService(const std::string& serviceName, 
                       std::chrono::seconds timeout = std::chrono::seconds(5));

private:
    std::shared_ptr<rclcpp::Node> node_;
    std::unordered_map<std::string, rclcpp::ClientBase::SharedPtr> clients_;
    std::mutex clientsMutex_;
    
    template<typename ServiceType>
    std::shared_ptr<rclcpp::Client<ServiceType>> getClient(const std::string& serviceName);
};

// Topic Monitor
class TopicMonitor {
public:
    explicit TopicMonitor(std::shared_ptr<rclcpp::Node> node);
    
    bool isTopicAvailable(const std::string& topicName);
    bool checkTopicsAvailable(const std::vector<std::string>& topics);
    bool waitForTopic(const std::string& topicName,
                     std::chrono::seconds timeout = std::chrono::seconds(5));

private:
    std::shared_ptr<rclcpp::Node> node_;
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

} // namespace behavior_controller

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get a configuration value from the YAML config file
 * @param key The configuration key to retrieve
 * @return The value as a string
 * @throws std::runtime_error if key not found or file cannot be read
 */
std::string getConfigValue(const std::string& key);

/**
 * @brief Validate the configuration file has all required fields
 * @param configPath Path to the configuration YAML file
 * @return true if valid, false otherwise
 */
bool validateConfigurationFile(const std::string& configPath);

/**
 * @brief Log system information (ROS distribution, available services/topics)
 * @param node The ROS2 node to query
 */
void logSystemInfo(std::shared_ptr<rclcpp::Node> node);

/**
 * @brief Check if a language is supported
 * @param language The language code to check
 * @return true if supported, false otherwise
 */
bool isValidLanguage(const std::string& language);

/**
 * @brief Get list of supported languages
 * @return Vector of supported language codes
 */
std::vector<std::string> getSupportedLanguages();

/**
 * @brief Check if a file exists
 * @param filepath Path to the file
 * @return true if file exists, false otherwise
 */
bool fileExists(const std::string& filepath);

/**
 * @brief Get absolute path to a file relative to package share directory
 * @param relativePath Path relative to package share directory
 * @return Absolute path to the file
 */
std::string getPackageDataPath(const std::string& relativePath);

/**
 * @brief Print node information to logs
 * @param node The ROS2 node
 */
void printNodeInfo(std::shared_ptr<rclcpp::Node> node);

/**
 * @brief Convert BehaviorTree NodeStatus to string
 * @param status The NodeStatus to convert
 * @return String representation of the status
 */
std::string nodeStatusToString(BT::NodeStatus status);

//=============================================================================
// Template Implementations
//=============================================================================

template<typename ServiceT>
bool ServiceManager::callService(
    const std::string& service_name,
    typename ServiceT::Request::SharedPtr request,
    typename ServiceT::Response::SharedPtr& response)
{
    // 1) Bail out early on shutdown
    if (!rclcpp::ok()) {
        RCLCPP_WARN(node_->get_logger(),
                    "Skipping service call (ROS is shutting down): %s",
                    service_name.c_str());
        return false;
    }

    // 2) Get or create the client
    auto client = getClient<ServiceT>(service_name);

    // 3) Wait for availability
    auto timeout = std::chrono::seconds(Constants::SERVICE_TIMEOUT_SEC);
    if (!client->wait_for_service(timeout)) {
        RCLCPP_ERROR(node_->get_logger(),
                     "Service not available: %s",
                     service_name.c_str());
        return false;
    }

    // 4) Send request & block until we get a response or timeout
    auto future = client->async_send_request(request);
    if (rclcpp::spin_until_future_complete(node_, future, timeout)
        != rclcpp::FutureReturnCode::SUCCESS)
    {
        RCLCPP_ERROR(node_->get_logger(),
                     "Service call to %s failed or timed out",
                     service_name.c_str());
        return false;
    }

    // 5) Grab the result
    response = future.get();
    return true;
}

template<typename ServiceType>
std::shared_ptr<rclcpp::Client<ServiceType>> 
ServiceManager::getClient(const std::string& serviceName) 
{
    std::lock_guard<std::mutex> lock(clientsMutex_);
    
    auto it = clients_.find(serviceName);
    if (it != clients_.end()) {
        return std::static_pointer_cast<rclcpp::Client<ServiceType>>(it->second);
    }
    
    auto client = node_->create_client<ServiceType>(serviceName);
    clients_[serviceName] = client;
    return client;
}

#endif // BEHAVIOR_CONTROLLER_INTERFACE_H