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

Custom Message/Service Libraries:
    - cssr_system/msg/face_detection_data.hpp
    - cssr_system/msg/overt_attention_mode.hpp
    - cssr_system/srv/animate_behavior_set_activation.hpp
    - cssr_system/srv/gesture_execution_perform_gesture.hpp
    - cssr_system/srv/overt_attention_set_mode.hpp
    - cssr_system/srv/robot_localization_reset_pose.hpp
    - cssr_system/srv/robot_navigation_set_goal.hpp
    - cssr_system/srv/speech_event_set_enabled.hpp
    - cssr_system/srv/speech_event_set_language.hpp
    - cssr_system/srv/tablet_event_prompt_and_get_response.hpp
    - cssr_system/srv/text_to_speech_say_text.hpp

Parameters
Launch File Parameters:
    ros2 run cssr_system behaviorController
        No command-line parameters required

Configuration File Parameters:
    Key                             Value                   Description
    scenario_specification          lab_tour                Mission scenario XML file name
    verbose_mode                    true/false              Enable/disable detailed logging
    asr_enabled                     true/false              Enable/disable Automatic Speech Recognition

Behavior Tree Node Types:
    Action Nodes:
        - StartOfTree: Initialize mission and test sequence
        - SayTextRosService: Execute text-to-speech with utility phrases
        - NavigateRosService: Move robot to specified locations
        - SelectExhibit: Choose next tour location
        - RetrieveListOfExhibits: Load tour specification
        - PerformDeicticGestureRosService: Execute pointing gestures
        - PerformIconicGestureRosService: Execute welcome/goodbye gestures
        - DescribeExhibitSpeechRosService: Deliver location-specific content
        - SetSpeechEventRosService: Configure speech recognition
        - SetOvertAttentionModeRosService: Control robot attention system
        - SetAnimateBehaviorRosService: Activate robot animations
        - ResetRobotPoseRosService: Reset robot localization
        - PressYesNoDialogueRosService: Display tablet interface
        - HandleFallBack: Error recovery mechanism

    Condition Nodes:
        - IsVisitorDiscovered: Check face detection status
        - IsMutualGazeDiscovered: Verify eye contact establishment
        - IsVisitorResponseYes: Validate affirmative responses
        - IsListWithExhibit: Check tour completion status

Subscribed Topics and Message Types:
    Topic Name                      Message Type                            Description
    /faceDetection/data             cssr_system::msg::FaceDetectionData     Face detection and visitor presence
    /overtAttention/mode            cssr_system::msg::OvertAttentionMode    Mutual gaze and attention status
    /speechEvent/text               std_msgs::msg::String                   Speech recognition results

Published Topics and Message Types:
    None (Behavior controller acts as orchestrator, not publisher)

Advertised Services:
    None (Behavior controller acts as service client, not server)

Services Invoked:
    Service Name                                    Service Type                                        Description
    /animateBehaviour/setActivation                 cssr_system::srv::AnimateBehaviorSetActivation     Activate/deactivate robot animations
    /gestureExecution/perform_gesture               cssr_system::srv::GestureExecutionPerformGesture   Execute pointing and iconic gestures
    /overtAttention/set_mode                        cssr_system::srv::OvertAttentionSetMode            Configure attention and gaze behavior
    /robotLocalization/reset_pose                   cssr_system::srv::RobotLocalizationResetPose       Reset robot position estimation
    /robotNavigation/set_goal                       cssr_system::srv::RobotNavigationSetGoal           Navigate to target locations
    /speechEvent/set_language                       cssr_system::srv::SpeechEventSetLanguage           Configure speech recognition language
    /speechEvent/set_enabled                        cssr_system::srv::SpeechEventSetEnabled            Enable/disable speech recognition
    /textToSpeech/say_text                          cssr_system::srv::TextToSpeechSayText              Execute text-to-speech synthesis

Input Data Files:
    - lab_tour.xml: Behavior tree definition for tour scenario
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
    ros2 run cssr_system behaviorController

    # With custom configuration
    ros2 run cssr_system behaviorController --ros-args -p config_path:=/path/to/config.yaml

System Architecture Integration:
    The behavior controller serves as the central orchestrator in the CSSR4Africa system architecture,
    coordinating between:
        - Face detection and person recognition subsystem
        - Speech recognition and text-to-speech subsystem  
        - Robot navigation and localization subsystem
        - Gesture execution and animation subsystem
        - Knowledge management and cultural adaptation subsystem

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohanneh@andrew.cmu.edu
Date: July 25, 2025
Version: v2.0 - Updated to use BehaviorTree.ROS2
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
    std::string configPath = packagePath + "/behaviorController/config/behaviorControllerConfiguration.yaml";
    
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
    RCLCPP_INFO(logger, "ASR Enabled: %s", config.isAsrEnabled() ? "Yes" : "No");
    RCLCPP_INFO(logger, "Verbose Mode: %s", config.isVerbose() ? "Yes" : "No");
    RCLCPP_INFO(logger, "Language: %s", config.getLanguage().c_str());

    return true;
}

std::string getScenarioSpecification(const rclcpp::Logger& logger) {
    std::string scenario = "lab_tour"; // Default scenario

    try {
        scenario = getConfigValue("scenario_specification");
        
        if (scenario.empty()) {
            throw std::runtime_error("Scenario specification is empty");
        }
        
        RCLCPP_INFO(logger, "Scenario Specification: %s", scenario.c_str());
        return scenario;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Error retrieving scenario (using default): %s", e.what());
        return scenario; // Return default
    }
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
            RCLCPP_FATAL(logger, "System initialization failed");
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
            RCLCPP_INFO(logger, "Behavior tree initialized successfully");
        }
        catch (const std::exception& e) {
            RCLCPP_FATAL(logger, "Failed to initialize tree: %s", e.what());
            rclcpp::shutdown();
            return 1;
        }
        
        // Optional: Enable Groot2 visualization
        BT::Groot2Publisher publisher(tree);
        RCLCPP_INFO(logger, "Groot2 publisher enabled - connect on port 1667");
        
        // Execute the tree
        RCLCPP_INFO(logger, "Starting behavior tree execution...");
        rclcpp::Rate rate(Constants::LOOP_RATE_HZ);  // Default: 10 Hz
        
        int tick_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        while (rclcpp::ok()) {
            // Tick the tree once
            auto status = tree.tickOnce();
            tick_count++;
            
            // Log periodic status if verbose mode is enabled
            if (ConfigManager::instance().isVerbose() && tick_count % 100 == 0) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    current_time - start_time).count();
                    
                RCLCPP_INFO(logger, 
                    "Behavior tree running - Status: %s, Ticks: %d, Elapsed: %ld seconds",
                    status == BT::NodeStatus::RUNNING ? "RUNNING" :
                    status == BT::NodeStatus::SUCCESS ? "SUCCESS" : "FAILURE",
                    tick_count, elapsed);
            }
            
            // Check if tree has finished
            if (status != BT::NodeStatus::RUNNING) {
                if (status == BT::NodeStatus::SUCCESS) {
                    RCLCPP_INFO(logger, "Behavior tree completed successfully after %d ticks", tick_count);
                } else {
                    RCLCPP_WARN(logger, "Behavior tree failed after %d ticks", tick_count);
                }
                break;
            }
            
            // Process ROS callbacks
            rclcpp::spin_some(node);
            
            // Sleep to maintain loop rate
            rate.sleep();
        }
        
        // Cleanup
        RCLCPP_INFO(logger, "Shutting down behavior controller...");
        rclcpp::shutdown();
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal exception in main: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }
}