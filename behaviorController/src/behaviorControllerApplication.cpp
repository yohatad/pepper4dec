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
        - SayText: Execute text-to-speech with utility phrases
        - Navigate: Move robot to specified locations
        - GetVisitorResponse: Capture and process speech input
        - SelectExhibit: Choose next tour location
        - RetrieveListOfExhibits: Load tour specification
        - PerformDeicticGesture: Execute pointing gestures
        - PerformIconicGesture: Execute welcome/goodbye gestures
        - DescribeExhibitSpeech: Deliver location-specific content
        - SetSpeechEvent: Configure speech recognition
        - SetOvertAttentionMode: Control robot attention system
        - SetAnimateBehavior: Activate robot animations
        - ResetRobotPose: Reset robot localization
        - PressYesNoDialogue: Display tablet interface
        - HandleFallBack: Error recovery mechanism

    Condition Nodes:
        - IsVisitorDiscovered: Check face detection status
        - IsMutualGazeDiscovered: Verify eye contact establishment
        - IsVisitorResponseYes: Validate affirmative responses
        - IsListWithExhibit: Check tour completion status
        - IsASREnabled: Verify speech recognition availability

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
Version: v1.0
*/

#include "behaviorController/behaviorControllerInterface.h"

void displayStartupInfo() {
    std::string softwareVersion = "1.0";
    std::string indent = "\n\t\t";
    
    // RCLCPP_INFO("fix this", "\n"
    //      "**************************************************************************************************\n"
    //      "%s%s\tv%s"
    //      "%sCopyright (C) 2025 CyLab Carnegie Mellon University Africa"
    //      "%sThis program comes with ABSOLUTELY NO WARRANTY."
    //      "\n\n**************************************************************************************************\n\n",
    //      indent.c_str(), softwareVersion.c_str(),
    //      indent.c_str(), indent.c_str()
    // );
}

bool initializeSystem() {
    // Logger logger(node);
    
    // Load configuration
    std::string packagePath = ament_index_cpp::get_package_share_directory("cssr_system");
    std::string configPath = packagePath + "/behaviorController/config/behaviorControllerConfiguration.yaml";
    
    if (!ConfigManager::instance().loadFromFile(configPath)) {
        // logger.error("Failed to load configuration from: " + configPath);
        return false;
    }
    
    // Load knowledge base
    if (!KnowledgeManager::instance().loadFromPackage(packagePath)) {
        // logger.error("Failed to load knowledge base from: " + packagePath);
        return false;
    }
    
    // Log configuration
    // auto& config = ConfigManager::instance();
    // logger.info("ASR Enabled: " + std::string(config.isAsrEnabled() ? "Yes" : "No"));
    // logger.info("Verbose Mode: " + std::string(config.isVerbose() ? "Yes" : "No"));

    return true;
}

// bool checkSystemRequirements() {
//     // Logger logger(node);
//     ServiceManager serviceManager(node);
//     TopicMonitor topicMonitor(node);
    
//     // Required services
//     std::vector<std::string> requiredServices = {
//         "/animateBehaviour/setActivation",
//         "/gestureExecution/perform_gesture",
//         "/overtAttention/set_mode",
//         "/robotLocalization/reset_pose",
//         "/robotNavigation/set_goal",
//         "/speechEvent/set_language",
//         "/speechEvent/set_enabled",
//         "/tabletEvent/prompt_and_get_response",
//         "/textToSpeech/say_text"
//     };
    
//     // Required topics
//     std::vector<std::string> requiredTopics = {
//         "/faceDetection/data",
//         "/overtAttention/mode",
//         "/speechEvent/text"
//     };
    
//     logger.info("Checking Services...");
    
//     if (!serviceManager.checkServicesAvailable(requiredServices)) {
//         logger.error("Not all required services are available");
//         return false;
//     }

//     logger.info("All required services are available");
    
//     logger.info("Checking Topics...");
    
//     if (!topicMonitor.checkTopicsAvailable(requiredTopics)) {
//         logger.error("Not all required topics are available");
//         return false;
//     }

//     logger.info("All required topics are available");
    
//     return true;
// }

int main(int argc, char** argv) {
    
        // logger.info("startup");
        // displayStartupInfo(node);

        std::string scenario = "lab_tour"; // Default scenario

        try {
        scenario = getConfigValue("scenario_specification");
        
        if (scenario.empty()) {
            throw std::runtime_error("Scenario specification is empty");
        }
        } catch (const std::exception& e) {
            // logger.error("Fatal Error retrieving scenario: " + std::string(e.what()));
            return 1; // Exit if scenario retrieval fails
        }

    
        // // Initialize system
        // if (!initializeSystem(node)) {
        //     return 1; // Exit if initialization fails
        // }
        
        // // Check system requirements (skip in standalone mode for testing)
        // if (!checkSystemRequirements(node)) {
        //     return 1; // Exit if requirements are not met
        // }
    
        // Initialize ROS
        rclcpp::init(argc, argv);
        
        // Create node
        auto node = rclcpp::Node::make_shared("behaviorController");

        BT::Tree tree;

        try {
        tree = behavior_controller::initializeTree(scenario, node);
        }
        catch (const std::exception &e) {
            RCLCPP_FATAL(node->get_logger(), "Failed to init tree: %s", e.what());
            return 1;
        }

        // registerRos2Nodes(factory);

        // (Optional) visualize in Groot:
        BT::Groot2Publisher publisher(tree);
                    
        // logger.info("Scenario Specification: " + scenario);
        
        // 7) Execute the tree in a loop until it returns SUCCESS or FAILURE
        rclcpp::Rate rate(Constants::LOOP_RATE_HZ);  // 10 Hz tick rate
        while (rclcpp::ok()) {
            auto status = tree.tickOnce();
            if (status != BT::NodeStatus::RUNNING) break;
            rclcpp::spin_some(node);
            rate.sleep();
        }

        // 8) Shutdown ROS and exit
        rclcpp::shutdown();
        return 0;
        
        // // Periodic status logging
        // if (ConfigManager::instance().isVerbose()) {
        //     static int counter = 0;
        //     if (++counter % 100 == 0) { // Log every 10 seconds at 10Hz
        //         logger.info("running");
        //     }
        // }
}