/*
behaviorControllerApplication.cpp ROS2 Node for Robot Mission Behavior Tree Execution and Control.

Copyright (C) 2025 CyLab Carnegie Mellon University Africa

This program comes with ABSOLUTELY NO WARRANTY.
*/

/*
behaviorControllerApplication.cpp ROS2 node to execute robot missions using Behavior Tree framework.

The behavior controller is implemented as a ROS2 node that orchestrates complex robot behaviors through a behavior tree
architecture. It manages multi-language tour guide scenarios for the Pepper robot, including face detection integration,
speech recognition, text-to-speech, gesture execution, navigation. The system supports dynamic mission configuration 
through YAML files and XML behavior tree definitions. The behavior tree nodes are implemented as modular, reusable 
components that can be configured for different scenarios and languages (English, Kinyarwanda, IsiZulu). The system 
includes robust service discovery, topic monitoring, and knowledge base management for cultural and environmental information.

Libraries
Standard Libraries:
    - std::string, std::fstream, std::sstream
    - std::vector, std::unordered_map, std::optional
    - std::memory, std::mutex, std::atomic
    - std::chrono, std::thread, std::regex
    - std::algorithm, std::transform

ROS2 Libraries:
    - rclcpp/rclcpp.hpp
    - std_msgs/msg/string.hpp
    - geometry_msgs/msg/point.hpp
    - ament_index_cpp/get_package_share_directory.hpp

BehaviorTree Libraries:
    - behaviortree_cpp/bt_factory.h
    - behaviortree_cpp/loggers/groot2_publisher.h
    - behaviortree_ros2/bt_service_node.hpp
    - behaviortree_ros2/bt_action_node.hpp
    - behaviortree_ros2/ros_node_params.hpp

External Libraries:
    - yaml-cpp/yaml.h

Custom Message/Service/Action Libraries:
    - cssr_interfaces/msg/face_detection.hpp
    - cssr_interfaces/msg/overt_attention_status.hpp
    - cssr_interfaces/msg/object_detection.hpp
    - cssr_interfaces/srv/animate_behavior_set_activation.hpp
    - cssr_interfaces/srv/overt_attention_set_mode.hpp
    - cssr_interfaces/srv/conversation_manager_prompt.hpp
    - cssr_interfaces/action/tts.hpp
    - cssr_interfaces/action/navigation.hpp
    - cssr_interfaces/action/gesture.hpp
    - cssr_interfaces/action/speech_recognition.hpp
    - cssr_interfaces/action/animate_behavior.hpp

Parameters
Launch File Parameters:
    ros2 run behavior_controller behaviorController
        No command-line parameters required

Configuration File Parameters:
    Key                             Value                   Description
    scenario_specification          lab_tour                Mission scenario XML file name
    verbose_mode                    true/false              Enable/disable detailed logging
    asr_enabled                     true/false              Enable/disable Automatic Speech Recognition

Behavior Tree Node Types:
    Action Nodes (ROS2 Actions):
        - TTSRosAction: Execute text-to-speech synthesis
        - NavigateRosAction: Move robot to specified locations
        - GestureRosAction: Execute deictic, iconic, symbolic gestures
        - SpeechRecognitionRosAction: Listen and transcribe visitor speech
        - AnimateBehaviorRosAction: Execute idle animations and behaviors

    Service Nodes (ROS2 Services):
        - SetOvertAttentionModeRosService: Control robot attention system
        - SetAnimateBehaviorRosService: Activate/deactivate animations
        - ConversationPromptRosService: Query LLM for conversational responses

    Custom Logic Nodes:
        - StartOfTree: Initialize mission and configuration
        - SelectExhibit: Choose next tour location
        - RetrieveListOfExhibits: Load tour specification
        - HandleFallBack: Error recovery mechanism

    Condition Nodes:
        - IsVisitorDiscovered: Check face detection status
        - IsMutualGazeDiscovered: Verify eye contact establishment
        - IsVisitorResponseYes: Validate affirmative responses
        - IsListWithExhibit: Check tour completion status

Subscribed Topics and Message Types:
    Topic Name                      Message Type                                Description
    /faceDetection/data             cssr_interfaces::msg::FaceDetection        Face detection and visitor presence
    /overtAttention/status          cssr_interfaces::msg::OvertAttentionStatus Attention system status updates
    /objectDetection/data           cssr_interfaces::msg::ObjectDetection      Object detection for exhibits

Published Topics and Message Types:
    None (Behavior controller acts as orchestrator, not publisher)

Advertised Services:
    None (Behavior controller acts as service client, not server)

Services Invoked:
    Service Name                                    Service Type                                        Description
    /animateBehaviour/setActivation                 cssr_interfaces::srv::AnimateBehaviorSetActivation Activate/deactivate robot animations
    /overtAttention/set_mode                        cssr_interfaces::srv::OvertAttentionSetMode        Configure attention and gaze behavior
    /conversation/prompt                            cssr_interfaces::srv::ConversationManagerPrompt    Query LLM for conversational responses

Action Servers Used:
    Action Name                     Action Type                             Description
    /tts                            cssr_interfaces::action::TTS           Text-to-speech synthesis with feedback
    /navigation                     cssr_interfaces::action::Navigation    Navigate to goal with progress updates
    /gesture                        cssr_interfaces::action::Gesture       Execute gestures with duration feedback
    /speech_recognition             cssr_interfaces::action::SpeechRecognition Listen and transcribe speech
    /animate_behavior               cssr_interfaces::action::AnimateBehavior Execute idle animations

Input Data Files:
    - lab_tour.xml: Behavior tree definition for robotics lab tour scenario
    - dec_tour.xml: Behavior tree definition for Digital Experience Center tour
    - environmentKnowledgeBase.yaml: Location and tour information
    - cultureKnowledgeBase.yaml: Language-specific phrases and cultural data

Output Data Files:
    None

Configuration Files:
    - behaviorControllerConfiguration.yaml: Main configuration file
    - environmentKnowledgeBase.yaml: Spatial and content knowledge
    - cultureKnowledgeBase.yaml: Cultural and linguistic knowledge

Supported Languages:
    - English: Primary language with full feature support
    - Kinyarwanda: Local language support for Rwanda deployment
    
Example Instantiation of the Module:
    # Basic execution
    ros2 run behavior_controller behaviorController

    # With custom configuration
    ros2 run behavior_controller behaviorController --ros-args -p config_path:=/path/to/config.yaml

    # With specific scenario
    ros2 run behavior_controller behaviorController --ros-args -p scenario:=dec_tour

System Architecture Integration:
    The behavior controller serves as the central orchestrator in the Pepper DEC Tour system,
    coordinating between:
        - Face detection and person recognition subsystem
        - Speech recognition and text-to-speech subsystem  
        - Robot navigation and localization subsystem
        - Gesture execution and animation subsystem
        - LLM-based conversation management subsystem
        - Knowledge management and cultural adaptation subsystem

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
    std::string softwareVersion = "2.0";
    
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
    RCLCPP_INFO(logger, "  - ASR Enabled: %s", config.isAsrEnabled() ? "Yes" : "No");
    RCLCPP_INFO(logger, "  - Verbose Mode: %s", config.isVerbose() ? "Yes" : "No");
    RCLCPP_INFO(logger, "  - Language: %s", config.getLanguage().c_str());
    RCLCPP_INFO(logger, "  - Scenario: %s", config.getScenarioSpecification().c_str());

    return true;
}

std::string getScenarioSpecification(const rclcpp::Logger& logger) {
    std::string scenario = "lab_tour"; // Default scenario

    try {
        auto& config = ConfigManager::instance();
        scenario = config.getScenarioSpecification();
        
        if (scenario.empty()) {
            RCLCPP_WARN(logger, "Scenario specification is empty, using default: %s", scenario.c_str());
        } else {
            RCLCPP_INFO(logger, "Scenario Specification: %s", scenario.c_str());
        }
        
        return scenario;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Error retrieving scenario (using default '%s'): %s", 
                    scenario.c_str(), e.what());
        return scenario; // Return default
    }
}

void logSystemStatus(const rclcpp::Logger& logger, 
                    BT::NodeStatus status, 
                    int tick_count,
                    const std::chrono::steady_clock::time_point& start_time) {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        current_time - start_time).count();
    
    const char* status_str;
    switch(status) {
        case BT::NodeStatus::RUNNING:
            status_str = "RUNNING";
            break;
        case BT::NodeStatus::SUCCESS:
            status_str = "SUCCESS";
            break;
        case BT::NodeStatus::FAILURE:
            status_str = "FAILURE";
            break;
        default:
            status_str = "IDLE";
            break;
    }
        
    RCLCPP_INFO(logger, 
        "Behavior Tree Status - State: %s | Ticks: %d | Uptime: %ld seconds",
        status_str, tick_count, elapsed);
}

int main(int argc, char** argv) {
    try {
        // Initialize ROS2
        rclcpp::init(argc, argv);
        
        // Create node
        auto node = rclcpp::Node::make_shared("behaviorController");
        auto logger = node->get_logger();
        
        // Display startup information
        displayStartupInfo(logger);
        
        // Initialize system (configuration and knowledge base)
        if (!initializeSystem(logger)) {
            RCLCPP_FATAL(logger, "System initialization failed - cannot continue");
            rclcpp::shutdown();
            return 1;
        }
        
        // Get scenario specification
        std::string scenario = getScenarioSpecification(logger);
        
        // Initialize behavior tree
        RCLCPP_INFO(logger, "Initializing behavior tree for scenario: %s", scenario.c_str());
        BT::Tree tree;
        
        try {
            tree = behavior_controller::initializeTree(scenario, node);
            RCLCPP_INFO(logger, "✓ Behavior tree initialized successfully");
        }
        catch (const std::exception& e) {
            RCLCPP_FATAL(logger, "✗ Failed to initialize behavior tree: %s", e.what());
            rclcpp::shutdown();
            return 1;
        }
        
        // Optional: Enable Groot2 visualization
        try {
            BT::Groot2Publisher publisher(tree);
            RCLCPP_INFO(logger, "✓ Groot2 publisher enabled - connect on port 1667");
        } catch (const std::exception& e) {
            RCLCPP_WARN(logger, "Groot2 publisher initialization failed: %s", e.what());
            RCLCPP_WARN(logger, "Continuing without visualization support");
        }
        
        // Execute the tree
        RCLCPP_INFO(logger, "=== Starting Behavior Tree Execution ===");
        rclcpp::Rate rate(Constants::LOOP_RATE_HZ);  // Default: 10 Hz
        
        int tick_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        BT::NodeStatus status = BT::NodeStatus::IDLE;
        
        while (rclcpp::ok()) {
            // Tick the tree once
            status = tree.tickOnce();
            tick_count++;
            
            // Log periodic status if verbose mode is enabled
            if (ConfigManager::instance().isVerbose() && tick_count % 100 == 0) {
                logSystemStatus(logger, status, tick_count, start_time);
            }
            
            // Check if tree has finished
            if (status != BT::NodeStatus::RUNNING) {
                if (status == BT::NodeStatus::SUCCESS) {
                    RCLCPP_INFO(logger, 
                        "✓ Behavior tree completed successfully after %d ticks", tick_count);
                } else if (status == BT::NodeStatus::FAILURE) {
                    RCLCPP_WARN(logger, 
                        "✗ Behavior tree failed after %d ticks", tick_count);
                } else {
                    RCLCPP_INFO(logger, 
                        "Behavior tree stopped with status IDLE after %d ticks", tick_count);
                }
                
                // Log final execution time
                auto current_time = std::chrono::steady_clock::now();
                auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time).count();
                RCLCPP_INFO(logger, "Total execution time: %.2f seconds", total_time / 1000.0);
                
                break;
            }
            
            // Process ROS callbacks
            rclcpp::spin_some(node);
            
            // Sleep to maintain loop rate
            rate.sleep();
        }
        
        // Cleanup
        RCLCPP_INFO(logger, "=== Shutting Down Behavior Controller ===");
        rclcpp::shutdown();
        
        // Return appropriate exit code
        return (status == BT::NodeStatus::SUCCESS) ? 0 : 1;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Fatal exception in main: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    } catch (...) {
        std::cerr << "✗ Unknown fatal exception in main" << std::endl;
        rclcpp::shutdown();
        return 1;
    }
}