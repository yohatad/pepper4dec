/* behaviorControllerApplication.cpp
* 
* <detailed functional description>
* The component starts the 'Robot Mission Interpreter' ROS2 Node. This node is the starting point for the robot to execute the mission selected.
* Though, this component exists as a standalone and can be started as such, without the set of ROS2 nodes that are part of the CSSR4Afica system architecture(see D3.1 System Architecture), it will not be able function.
...
* Libraries
* Standard libraries
- std::string, std::fstream
* ROS2 libraries
- rclcpp/rclcpp.hpp, ament_index_cpp, std_msgs
* BehaviorTree.Cpp libraries
- behaviortree_cpp/bt_factory.h, behaviortree_cpp/loggers/groot2_publisher.h
...
* Parameters
* *
* Command-line Parameters
* *
* None
 ...
* Configuration File Parameters
* Key | Value
* ----|------
* scenarioSpecification | <the mission scenario to be interpreted>
* verboseMode           | <true/false - enables/disables the display of diagnostic messages>
* asrEnabled            | <true/false> - enables/disables the Automatic Speech Recognition. If diabled, pepper's tablet will be primary input method
* testMode              | <true/false> - enables/disabled the test sequence. If enabled, it's the test sequence that will run
...
* Subscribed Topics and Message Types
**
- /faceDetection/data       FaceDetectionData.msg
- /overtAttention/mode      OvertAttentionMode.msg
- /speechEvent/text         std_msgs::msg::String
...
* Published Topics and Message Types
**
* None
...
* Advertised Services
* 
* None
...
* Services Invoked
* *
* /animateBehaviour/setActivation
* /gestureExecution/perform_gesture
* /overtAttention/set_mode
* /robotLocalization/reset_pose
* /robotNavigation/set_goal
* /speechEvent/set_language
* /speechEvent/set_enabled
* /tabletEvent/prompt_and_get_response
* /textToSpeech/say_text                                    
...
* Input Data Files
*
* lab_tour.xml
...
* Output Data Files
* 
* None
...
* Configuration Files
**
* behaviorControllerConfiguration.ini
...
* Example Instantiation of the Module
* *
* ros2 run cssr_system behaviorController
...
* *
* Author: Tsegazeab Taye Tefferi, Carnegie Mellon University Africa
* Email: ttefferi@andrew.cmu.edu
* Date: April 08, 2025
* Version: v1.0
* *
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
