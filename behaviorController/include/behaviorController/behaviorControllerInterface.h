/* behaviorControllerInterface.h 
 *
 * Author: Yohannes Tadesse Haile
 * Date: July 25, 2025
 * Version: v1.0
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#ifndef BEHAVIOR_CONTROLLER_INTERFACE_H
#define BEHAVIOR_CONTROLLER_INTERFACE_H

// ROS includes
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

// BehaviorTree includes
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>

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
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <thread>
#include <iomanip>

// Custom message/service includes
#include "cssr_system/msg/face_detection_data.hpp"
#include "cssr_system/msg/overt_attention_mode.hpp"
#include "cssr_system/srv/animate_behavior_set_activation.hpp"
#include "cssr_system/srv/gesture_execution_perform_gesture.hpp"
#include "cssr_system/srv/overt_attention_set_mode.hpp"
#include "cssr_system/srv/robot_localization_reset_pose.hpp"
#include "cssr_system/srv/robot_navigation_set_goal.hpp"
#include "cssr_system/srv/speech_event_set_enabled.hpp"
#include "cssr_system/srv/speech_event_set_language.hpp"
#include "cssr_system/srv/tablet_event_prompt_and_get_response.hpp"
#include "cssr_system/srv/text_to_speech_say_text.hpp"

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

private:
    std::shared_ptr<rclcpp::Node> node_;
    std::string formatMessage(const std::string& msg);
};

// Service Manager
class ServiceManager {
public:
    explicit ServiceManager(std::shared_ptr<rclcpp::Node> node);
    
    template<typename ServiceType>
    bool callService(const std::string& serviceName, 
                    typename ServiceType::Request::SharedPtr request,
                    typename ServiceType::Response::SharedPtr& response);
    
    bool checkServicesAvailable(const std::vector<std::string>& services);

private:
    std::shared_ptr<rclcpp::Node> node_;
    std::unordered_map<std::string, rclcpp::ClientBase::SharedPtr> clients_;
    mutable std::mutex clientsMutex_;
    
    template<typename ServiceType>
    std::shared_ptr<rclcpp::Client<ServiceType>> getClient(const std::string& serviceName);
};

// Topic Monitor
class TopicMonitor {
public:
    explicit TopicMonitor(std::shared_ptr<rclcpp::Node> node);
    bool checkTopicsAvailable(const std::vector<std::string>& topics);

private:
    std::shared_ptr<rclcpp::Node> node_;
    bool isTopicAvailable(const std::string& topicName);
};

// Word Processing Utilities
class TextUtils {
public:
    static bool containsAnyWord(const std::string& text, const std::vector<std::string>& words);
    static std::string toLowerCase(const std::string& text);
};

//=============================================================================
// Base Node Class for Behavior Tree Nodes
//=============================================================================
class BaseTreeNode {
protected:
    std::shared_ptr<rclcpp::Node>   node_;
    std::unique_ptr<Logger>         logger_;
    std::unique_ptr<ServiceManager> serviceManager_;

public:
    /** 
     * Construct by retrieving the ROS2 node handle from the BT blackboard.
     * Throws if someone forgot to put "node" into the blackboard before tree creation.
     */
    explicit BaseTreeNode(const BT::NodeConfiguration &config);

    virtual ~BaseTreeNode() = default;

protected:
    template<typename ServiceType>
    bool callServiceSafely(const std::string                        &serviceName,
                           typename ServiceType::Request::SharedPtr  request,
                           typename ServiceType::Response::SharedPtr &response,
                           const std::string                        &treeNodeName);
};

//=============================================================================
// Function Declarations
//=============================================================================
namespace behavior_controller {

/**
 * @brief Build and register all ROS2‑aware and custom BehaviorTree.CPP nodes,
 *        load the XML file for the given scenario, and return a ready‑to‑tick tree.
 *
 * @param scenario      Base name (without “.xml”) of the tree file under data/
 * @param node_handle   Shared pointer to your ROS2 node (injected into every BT node)
 * @return BT::Tree     The fully constructed behavior tree
 * @throws std::runtime_error if the XML file cannot be found or loaded
 */
BT::Tree initializeTree(
    const std::string &scenario,
    std::shared_ptr<rclcpp::Node> node_handle);

}  // namespace behavior_controller

std::string getConfigValue(const std::string& key);

//=============================================================================
// Template Implementations
//=============================================================================
template<typename ServiceType>
bool ServiceManager::callService(const std::string& serviceName, 
                                typename ServiceType::Request::SharedPtr request,
                                typename ServiceType::Response::SharedPtr& response) {
    try {
        // Check if ROS is still OK before attempting service call
        if (!rclcpp::ok()) {
            RCLCPP_WARN(node_->get_logger(), "ROS is shutting down, skipping service call to %s", serviceName.c_str());
            return false;
        }
        
        auto client = getClient<ServiceType>(serviceName);
        auto timeout = std::chrono::seconds(Constants::SERVICE_TIMEOUT_SEC);
        
        if (!client->wait_for_service(timeout)) {
            RCLCPP_ERROR(node_->get_logger(), "Service %s not available", serviceName.c_str());
            return false;
        }
        
        // Check again before making the actual call
        if (!rclcpp::ok()) {
            RCLCPP_WARN(node_->get_logger(), "ROS is shutting down, aborting service call to %s", serviceName.c_str());
            return false;
        }
        
        auto future = client->async_send_request(request);
        auto status = rclcpp::spin_until_future_complete(node_, future, timeout);
        
        if (status == rclcpp::FutureReturnCode::SUCCESS) {
            response = future.get();
            return true;
        }
        
        RCLCPP_ERROR(node_->get_logger(), "Service call to %s failed", serviceName.c_str());
        return false;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Exception in service call to %s: %s", 
                     serviceName.c_str(), e.what());
        return false;
    }
}

template<typename ServiceType>
std::shared_ptr<rclcpp::Client<ServiceType>> ServiceManager::getClient(const std::string& serviceName) {
    std::lock_guard<std::mutex> lock(clientsMutex_);
    
    auto it = clients_.find(serviceName);
    if (it != clients_.end()) {
        return std::static_pointer_cast<rclcpp::Client<ServiceType>>(it->second);
    }
    
    auto client = node_->create_client<ServiceType>(serviceName);
    clients_[serviceName] = client;
    return client;
}

template<typename ServiceType>
bool BaseTreeNode::callServiceSafely(const std::string& serviceName,
                                     typename ServiceType::Request::SharedPtr request,
                                     typename ServiceType::Response::SharedPtr& response,
                                     const std::string& treeNodeName) {
    try {
        return serviceManager_->callService<ServiceType>(serviceName, request, response);
    } catch (const std::exception& e) {
        logger_->error("Exception in " + treeNodeName + " service call: " + e.what());
        return false;
    }
}

#endif // BEHAVIOR_CONTROLLER_INTERFACE_H
