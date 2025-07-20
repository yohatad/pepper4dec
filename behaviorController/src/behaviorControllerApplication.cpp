/**
 * -----------------------------------------------------------------------------
 * @file    behaviorControllerApplication.cpp
 * @brief   Starts the “Robot Mission Interpreter” ROS 2 node—the entry point for
 *          executing the selected robot mission. Note that without the rest of
 *          the CSSR4Africa system, this component will not function.
 * -----------------------------------------------------------------------------
 *
 * @libraries
 *   • Standard: <string>, <fstream>
 *   • ROS 2:    rclcpp/rclcpp.hpp, ament_index_cpp, std_msgs/msg/String
 *   • BehaviorTree.Cpp:
 *       – behaviortree_cpp/bt_factory.h
 *       – behaviortree_cpp/loggers/groot2_publisher.h
 *
 * @parameters
 *   – **Command‑line**: None
 *   – **Configuration File**: behaviorControllerConfiguration.ini
 *     | Key                   | Type  | Description                                                      |
 *     |-----------------------|-------|------------------------------------------------------------------|
 *     | scenarioSpecification | string| Mission scenario to interpret                                     |
 *     | verboseMode           | bool  | Enable/disable diagnostic messages                                |
 *     | asrEnabled            | bool  | Enable/disable ASR; if disabled, tablet is primary input method  |
 *     | testMode              | bool  | Enable/disable test sequence                                      |
 *
 * @subscribed_topics
 *   – `/faceDetection/data`   : FaceDetectionData.msg
 *   – `/overtAttention/mode`  : OvertAttentionMode.msg
 *   – `/speechEvent/text`     : std_msgs::msg::String
 *
 * @published_topics
 *   – None
 *
 * @services_invoked
 *   – /animateBehaviour/setActivation
 *   – /gestureExecution/perform_gesture
 *   – /overtAttention/set_mode
 *   – /robotLocalization/reset_pose
 *   – /robotNavigation/set_goal
 *   – /speechEvent/set_language
 *   – /speechEvent/set_enabled
 *   – /tabletEvent/prompt_and_get_response
 *   – /textToSpeech/say_text
 *
 * @files
 *   – **Input:**  lab_tour.xml
 *   – **Output:** None
 *
 * @example
 *   ros2 run cssr_system behaviorController
 *
 * @author
 *   Yohannes Tadesse Haile
 *   Carnegie Mellon University Africa
 *   yohanneh@andrew.cmu.edu
 *
 * @date    July 18, 2025
 * @version 1.0
 * -----------------------------------------------------------------------------
 */


#include "behaviorController/behaviorControllerInterface.h"

/* Configuration variables */
bool verboseMode;
bool asrEnabled;
std::string missionLanguage;
bool testMode;

/*************************/
std::string nodeName;
std::shared_ptr<rclcpp::Node> node = nullptr;

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    BT::Tree tree;
    node = rclcpp::Node::make_shared("behaviorController");

    std::string scenario;
    nodeName = node->get_name();
    std::string softwareVersion = "1.0";
    std::string leftIndent = "\n\t\t";
    RCLCPP_INFO(node->get_logger(), "[%s]: startup", nodeName.c_str());
    RCLCPP_INFO(node->get_logger(), "\n"
         "**************************************************************************************************\n"
         "%s%s\t%s"
         "%sCopyright (C) 2023 CSSR4Africa Consortium"
         "%sThis project is funded by the African Engineering and Technology Network (Afretec)"
         "%sInclusive Digital Transformation Research Grant Program.\n"
         "%sWebsite: www.cssr4africa.org\n"
         "%sThis program comes with ABSOLUTELY NO WARRANTY."
         "\n\n**************************************************************************************************\n\n",
         leftIndent.c_str(), nodeName.c_str(), softwareVersion.c_str(),
         leftIndent.c_str(), leftIndent.c_str(), leftIndent.c_str(),
         leftIndent.c_str(), leftIndent.c_str()
    );

    /* Retrieve the values from the configuration file       */
    /* Display the error and exit, if the file is unreadable */
    try
    {
        scenario = getValueFromConfig("scenarioSpecification");
        verboseMode = (getValueFromConfig("verboseMode") == "true");
        asrEnabled = (getValueFromConfig("asrEnabled") == "true");
        testMode = (getValueFromConfig("testMode") == "true");
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Fatal Error: %s", e.what());
        rclcpp::shutdown();
        return 0;
    }

    /* Initialize knowledge base objects */
    try {
        environmentKnowledgeBase = std::make_shared<Environment::EnvironmentKnowledgeBase>(node);
        culturalKnowledgeBase = std::make_shared<Culture::CultureKnowledgeBase>(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Failed to initialize knowledge bases: %s", e.what());
        rclcpp::shutdown();
        return 0;
    }

    /* Get the currently set language from the knowledge base*/
    missionLanguage = getMissionLanguage();

    RCLCPP_INFO(node->get_logger(), "Scenario Specification: %s", scenario.c_str());
    RCLCPP_INFO(node->get_logger(), "Mission Language: %s", missionLanguage.c_str());
    RCLCPP_INFO(node->get_logger(), "Mode: %s", (testMode ? "Test" : "Normal"));

    /* List of services to check for life*/
    std::vector<std::string> services = {
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
    
    /* List of topics to check for life*/
    std::vector<std::string> topics = {
        "/faceDetection/data",
        "/overtAttention/mode",
        "/speechEvent/text",
    };

    /* If any of the services from above isn't alive, exit program */
    RCLCPP_INFO(node->get_logger(), "Checking Services...");
    if (checkServices(services)) {
        RCLCPP_INFO(node->get_logger(), "All services available");
    } else {
        rclcpp::shutdown();
        return 0;
    }

    /* If any of the topics from above isn't alive, exit program */
    RCLCPP_INFO(node->get_logger(), "Checking Topics...");
    if (checkTopics(topics)) {
        RCLCPP_INFO(node->get_logger(), "All topics available");
    } else {
        rclcpp::shutdown();
        return 0;
    }

    /* Use the mission specification file to create the tree and initiate the mission*/
    tree = initializeTree(scenario);
    BT::Groot2Publisher Groot2Publisher(tree);
    rclcpp::Rate rate(1);
    RCLCPP_INFO(node->get_logger(), "Starting Mission Execution ....");
    
    while (rclcpp::ok()) {
        tree.tickWhileRunning();
        rclcpp::spin_some(node);
        rate.sleep();
        RCLCPP_INFO(node->get_logger(), "[%s]: running", nodeName.c_str());
    }

    rclcpp::shutdown();
    return 0;
}
