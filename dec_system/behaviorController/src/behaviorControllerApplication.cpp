/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohanneh@andrew.cmu.edu
Date: February 09, 2026
Version: v2.0 - Updated to use BehaviorTree.ROS2 with valid cssr_interfaces
*/

#include "behaviorController/behaviorControllerInterface.h"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>

void displayStartupInfo(const rclcpp::Logger& logger) {
    std::string softwareVersion = "1.0";
    
    RCLCPP_INFO(logger, "\n"
         "**************************************************************************************************\n"
         "\t\tBehavior Controller v%s (BehaviorTree.ROS2)\n"
         "\t\tCopyright (C) 2025 CyLab Carnegie Mellon University Africa\n"
         "\t\tThis program comes with ABSOLUTELY NO WARRANTY.\n"
         "**************************************************************************************************\n",
         softwareVersion.c_str()
    );
}

bool initializeSystem(const rclcpp::Logger& logger) {
    // Load configuration
    std::string packagePath = ament_index_cpp::get_package_share_directory("behavior_controller");
    std::string configPath = packagePath + "/config/behaviorControllerConfiguration.yaml";
    
    if (!ConfigManager::instance().loadFromFile(configPath)) {
        RCLCPP_ERROR(logger, "Failed to load configuration from: %s", configPath.c_str());
        return false;
    }
    
    // Load knowledge base
    if (!KnowledgeManager::instance().loadFromPackage(packagePath)) {
        RCLCPP_ERROR(logger, "Failed to load knowledge base from: %s", packagePath.c_str());
        return false;
    }
    
    // Log configuration
    auto& config = ConfigManager::instance();
    RCLCPP_INFO(logger, "Configuration loaded successfully:");
    RCLCPP_INFO(logger, "  - Verbose Mode: %s", config.isVerbose() ? "Yes" : "No");
    RCLCPP_INFO(logger, "  - Language: %s", config.getLanguage().c_str());
    RCLCPP_INFO(logger, "  - Scenario: %s", config.getScenarioSpecification().c_str());

    return true;
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("behavior_controller");
    auto logger = node->get_logger();

    displayStartupInfo(logger);
    
    if (!initializeSystem(logger)) {
        RCLCPP_ERROR(logger, "System initialization failed. Shutting down.");
        rclcpp::shutdown();
        return 1;
    }

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}