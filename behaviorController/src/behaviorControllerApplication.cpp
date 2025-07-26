/*
behaviorControllerApplication.cpp ROS2 Node for Robot Mission Behavior Tree Execution and Control.

Copyright (C) 2025 CyLab Carnegie Mellon University Africa

This program comes with ABSOLUTELY NO WARRANTY.
*/

/*
behaviorControllerApplication.cpp   ROS2 node to execute robot missions using Behavior Tree framework.

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
    test_mode                       true/false              Enable/disable test result recording

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
    /tabletEvent/prompt_and_get_response            cssr_system::srv::TabletEventPromptAndGetResponse  Display tablet interface prompts
    /textToSpeech/say_text                          cssr_system::srv::TextToSpeechSayText              Execute text-to-speech synthesis

Input Data Files:
    - lab_tour.xml: Behavior tree definition for tour scenario
    - environmentKnowledgeBase.yaml: Location and tour information
    - cultureKnowledgeBase.yaml: Language-specific phrases and cultural data

Output Data Files:
    None (Results stored as ROS2 parameters when in test mode)

Configuration Files:
    - behaviorControllerConfiguration.yaml: Main configuration file
    - environmentKnowledgeBase.yaml: Spatial and content knowledge
    - cultureKnowledgeBase.yaml: Cultural and linguistic knowledge

Multi-Language Support:
    Supported Languages:
        - English: Primary language with full feature support
        - Kinyarwanda: Local language support for Rwanda deployment
        - IsiZulu: Additional African language support
    
    Language-Specific Features:
        - Utility phrases and greetings
        - Location descriptions and tour content
        - Speech recognition vocabulary
        - Cultural gesture preferences

Tour Management:
    Location Features:
        - Dynamic tour route configuration
        - Multi-language location descriptions
        - Gesture target coordinates
        - Cultural interaction preferences
    
    Interaction Modes:
        - Face-to-face conversation
        - Tablet-based interaction
        - Speech recognition input
        - Gesture-based communication

Error Handling and Recovery:
    - Service availability verification
    - Topic connectivity monitoring
    - Graceful degradation for missing components
    - Comprehensive test result tracking
    - Signal-based shutdown handling

Example Instantiation of the Module:
    # Basic execution
    ros2 run cssr_system behaviorController

    # With custom configuration
    ros2 run cssr_system behaviorController --ros-args -p config_path:=/path/to/config.yaml

    # Launch with full system
    ros2 launch cssr_system behavior_controller_launch.py

    # Test mode execution
    ros2 run cssr_system behaviorController --ros-args -p test_mode:=true

System Architecture Integration:
    The behavior controller serves as the central orchestrator in the CSSR4Africa system architecture,
    coordinating between:
        - Face detection and person recognition subsystem
        - Speech recognition and text-to-speech subsystem  
        - Robot navigation and localization subsystem
        - Gesture execution and animation subsystem
        - Tablet interface and human-computer interaction subsystem
        - Knowledge management and cultural adaptation subsystem

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohanneh@andrew.cmu.edu
Date: July 25, 2025
Version: v1.0
*/

#include "behaviorController/behaviorControllerInterface.h"

namespace {
    std::atomic<bool> shutdownRequested{false};
    std::atomic<bool> cleanupInProgress{false};
    std::shared_ptr<rclcpp::Node> globalNode = nullptr;
    std::unique_ptr<BT::Groot2Publisher> groot2Publisher = nullptr;
    BT::Tree* globalTree = nullptr;
    std::mutex cleanupMutex;
}

void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        bool expected = false;
        if (shutdownRequested.compare_exchange_strong(expected, true)) {
            // Only log the first signal
            if (globalNode) {
                RCLCPP_INFO(globalNode->get_logger(), "Shutdown signal received");
            } else {
                std::cout << "Shutdown signal received" << std::endl;
            }
        }
        // Don't call rclcpp::shutdown() here - let main handle it
    }
}

void cleanupResources() {
    std::lock_guard<std::mutex> lock(cleanupMutex);
    
    // Prevent multiple cleanup attempts
    if (cleanupInProgress.exchange(true)) {
        return;
    }
    
    try {
        // 1. First, halt the behavior tree to stop all ongoing operations
        if (globalTree) {
            try {
                globalTree->haltTree();
                // Give tree time to properly halt
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } catch (const std::exception& e) {
                // Ignore tree halt exceptions during shutdown
            }
            globalTree = nullptr;
        }
        
        // 2. Reset Groot2Publisher before shutting down ROS
        if (groot2Publisher) {
            try {
                groot2Publisher.reset();
            } catch (const std::exception& e) {
                // Ignore publisher reset exceptions
            }
        }
        
        // 3. Clear knowledge manager cache
        try {
            KnowledgeManager::instance().clearCache();
        } catch (const std::exception& e) {
            // Ignore cache clear exceptions
        }
        
        // 4. Give time for any pending operations to complete
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
    } catch (const std::exception& e) {
        // Ignore exceptions during cleanup to prevent cascading errors
        if (globalNode) {
            RCLCPP_WARN(globalNode->get_logger(), 
                       "Exception during cleanup: %s", e.what());
        }
    }
}

void displayStartupInfo(std::shared_ptr<rclcpp::Node> node) {
    std::string softwareVersion = "1.0";
    std::string indent = "\n\t\t";
    
    RCLCPP_INFO(node->get_logger(), "\n"
         "**************************************************************************************************\n"
         "%s%s\tv%s"
         "%sCopyright (C) 2025 CyLab Carnegie Mellon University Africa"
         "%sThis program comes with ABSOLUTELY NO WARRANTY."
         "\n\n**************************************************************************************************\n\n",
         indent.c_str(), ConfigManager::instance().getNodeName().c_str(), softwareVersion.c_str(),
         indent.c_str(), indent.c_str()
    );
}

bool initializeSystem(std::shared_ptr<rclcpp::Node> node) {
    Logger logger(node);
    
    // Load configuration
    std::string packagePath = ament_index_cpp::get_package_share_directory("cssr_system");
    std::string configPath = packagePath + "/behaviorController/config/behaviorControllerConfiguration.yaml";
    
    if (!ConfigManager::instance().loadFromFile(configPath)) {
        logger.error("Failed to load configuration from: " + configPath);
        return false;
    }
    
    // Load knowledge base
    if (!KnowledgeManager::instance().loadFromPackage(packagePath)) {
        logger.error("Failed to load knowledge base from: " + packagePath);
        return false;
    }
    
    // Log configuration
    auto& config = ConfigManager::instance();
    logger.info("Mode: " + std::string(config.isTestMode() ? "Test" : "Normal"));
    logger.info("ASR Enabled: " + std::string(config.isAsrEnabled() ? "Yes" : "No"));
    logger.info("Verbose Mode: " + std::string(config.isVerbose() ? "Yes" : "No"));
    
    return true;
}

bool checkSystemRequirements(std::shared_ptr<rclcpp::Node> node) {
    Logger logger(node);
    ServiceManager serviceManager(node);
    TopicMonitor topicMonitor(node);
    
    // Required services
    std::vector<std::string> requiredServices = {
        "/animateBehaviour/setActivation",
        "/gestureExecution/perform_gesture",
        "/overtAttention/set_mode",
        "/robotLocalization/reset_pose",
        "/robotNavigation/set_goal",
        "/speechEvent/set_language",
        "/speechEvent/set_enabled",
        "/tabletEvent/prompt_and_get_response",
        "/textToSpeech/say_text"
    };
    
    // Required topics
    std::vector<std::string> requiredTopics = {
        "/faceDetection/data",
        "/overtAttention/mode",
        "/speechEvent/text"
    };
    
    logger.info("Checking Services...");
    if (!serviceManager.checkServicesAvailable(requiredServices)) {
        logger.error("Not all required services are available");
        return false;
    }
    logger.info("All required services are available");
    
    logger.info("Checking Topics...");
    if (!topicMonitor.checkTopicsAvailable(requiredTopics)) {
        logger.error("Not all required topics are available");
        return false;
    }
    logger.info("All required topics are available");
    
    return true;
}

int main(int argc, char** argv) {
    int exitCode = 0;
    
    try {
        // Initialize ROS
        rclcpp::init(argc, argv);
        
        // Install signal handlers
        std::signal(SIGINT, signalHandler);
        std::signal(SIGTERM, signalHandler);
        
        // Create node
        auto node = rclcpp::Node::make_shared("behaviorController");
        globalNode = node;
        Logger logger(node);
        
        logger.info("startup");
        displayStartupInfo(node);
        
        // Initialize system
        if (!initializeSystem(node)) {
            exitCode = 1;
            goto cleanup;
        }
        
        // Check system requirements (skip in standalone mode for testing)
        if (!checkSystemRequirements(node)) {
            logger.warn("System requirements not met - running in standalone mode");
        }
        
        // Get scenario specification
        std::string scenario;
        try {
            scenario = getConfigValue("scenario_specification");
            if (scenario.empty()) {
                throw std::runtime_error("Scenario specification is empty");
            }
        } catch (const std::exception& e) {
            logger.error("Fatal Error retrieving scenario: " + std::string(e.what()));
            exitCode = 1;
            goto cleanup;
        }
        
        logger.info("Scenario Specification: " + scenario);
        
        // Initialize behavior tree
        BT::Tree tree;
        try {
            tree = initializeTree(scenario, node);
            globalTree = &tree;
            
            // Set shutdown flag function for behavior tree nodes
            // (You'll need to declare this function in your header)
            extern void setShutdownRequested(bool);
            setShutdownRequested(false);
            
        } catch (const std::exception& e) {
            logger.error("Failed to initialize behavior tree: " + std::string(e.what()));
            exitCode = 1;
            goto cleanup;
        }
        
        // Initialize tree monitoring (optional)
        try {
            groot2Publisher = std::make_unique<BT::Groot2Publisher>(tree);
        } catch (const std::exception& e) {
            logger.warn("Failed to initialize Groot2Publisher: " + std::string(e.what()));
        }
        
        // Main execution loop
        rclcpp::Rate rate(Constants::LOOP_RATE_HZ);
        logger.info("Starting Mission Execution...");
        logger.info("=== START OF TREE ===");
        
        // Track consecutive failures to prevent infinite loops
        int consecutiveFailures = 0;
        const int maxConsecutiveFailures = 5;
        
        while (rclcpp::ok() && !shutdownRequested && consecutiveFailures < maxConsecutiveFailures) {
            try {
                // Check if shutdown was requested
                if (shutdownRequested) {
                    // Signal behavior tree nodes to stop
                    extern void setShutdownRequested(bool);
                    setShutdownRequested(true);
                    
                    logger.info("Shutdown requested, stopping behavior tree execution");
                    break;
                }
                
                // Tick the behavior tree with timeout protection
                BT::NodeStatus status = BT::NodeStatus::RUNNING;
                try {
                    status = tree.tickOnce();
                    consecutiveFailures = 0; // Reset on successful tick
                } catch (const std::exception& e) {
                    logger.error("Exception during tree tick: " + std::string(e.what()));
                    consecutiveFailures++;
                    if (consecutiveFailures >= maxConsecutiveFailures) {
                        logger.error("Too many consecutive failures, shutting down");
                        break;
                    }
                    rate.sleep();
                    continue;
                }
                
                // Handle tree completion
                if (status == BT::NodeStatus::SUCCESS) {
                    logger.info("Mission completed successfully");
                    break;
                } else if (status == BT::NodeStatus::FAILURE) {
                    logger.error("Mission failed");
                    exitCode = 1;
                    break;
                }
                
                // Process ROS callbacks only if ROS is still OK and not shutting down
                if (rclcpp::ok() && !shutdownRequested) {
                    try {
                        rclcpp::spin_some(node);
                    } catch (const std::exception& e) {
                        logger.warn("Exception during spin_some: " + std::string(e.what()));
                        // Continue execution, this is not critical
                    }
                }
                
                // Periodic status logging
                if (ConfigManager::instance().isVerbose()) {
                    static int counter = 0;
                    if (++counter % 100 == 0) { // Log every 10 seconds at 10Hz
                        logger.info("running");
                    }
                }
                
                // Sleep only if not shutting down
                if (!shutdownRequested) {
                    rate.sleep();
                }
                
            } catch (const std::exception& e) {
                logger.error("Exception in main loop: " + std::string(e.what()));
                consecutiveFailures++;
                
                // If we get an exception during shutdown, break the loop
                if (shutdownRequested || !rclcpp::ok()) {
                    break;
                }
                
                // If too many failures, break
                if (consecutiveFailures >= maxConsecutiveFailures) {
                    logger.error("Too many consecutive failures, shutting down");
                    break;
                }
                
                // Continue execution for non-critical exceptions
                if (!shutdownRequested) {
                    rate.sleep();
                }
            }
        }
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("behaviorController"), "Fatal exception: %s", e.what());
        exitCode = 1;
    }
    
    cleanup:
    // Cleanup section - this runs regardless of how we exit
    try {
        if (globalNode && !cleanupInProgress) {
            Logger logger(globalNode);
            logger.info("Shutting down gracefully...");
        }
        
        // Clean up resources in proper order
        cleanupResources();
        
        // Shutdown ROS if it's still running
        if (rclcpp::ok()) {
            rclcpp::shutdown();
        }
        
        // Wait a bit longer for ROS to fully shutdown
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        // Reset global node last
        if (globalNode) {
            try {
                globalNode.reset();
            } catch (const std::exception& e) {
                // Ignore node reset exceptions
            }
        }
        
        // Final status message
        std::cout << "Shutdown complete" << std::endl;
        
    } catch (const std::exception& e) {
        // Final exception handler - don't throw anything from here
        std::cerr << "Exception during final cleanup: " << e.what() << std::endl;
        exitCode = 1;
    } catch (...) {
        // Catch any other type of exception
        std::cerr << "Unknown exception during final cleanup" << std::endl;
        exitCode = 1;
    }
    
    return exitCode;
}