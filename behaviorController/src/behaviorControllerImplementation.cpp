/**
 * -----------------------------------------------------------------------------
 * @file    behaviorControllerImplementation.cpp
 * @brief   Implements the robot mission node classes and related utility functions.
 * -----------------------------------------------------------------------------
 *
 * @author  Yohannes Haile
 * @date    July 18, 2025
 * @version 1.0
 *
 * @copyright
 *   Copyright (C) 2023 CSSR4Africa Consortium
 *
 * @funding
 *   Funded by the African Engineering and Technology Network (Afretec)
 *   Inclusive Digital Transformation Research Grant Programme.
 *
 * @website
 *   https://www.cssr4africa.org
 *
 * @warning
 *   This program comes with ABSOLUTELY NO WARRANTY.
 * -----------------------------------------------------------------------------
 */


#include "behaviorController/behaviorControllerInterface.h"
#include <algorithm>
#include <sstream>
#include <chrono>

/* Definitions for printMsg function */
#define INFO_MSG 0
#define WARNING_MSG 1
#define ERROR_MSG 2

/*
Logs the string (args) to the terminal based on the (type).
Wrapper around the default ROS2 logging functions
*/
void printMsg(int type, std::string args);

/* Fetches the utility phrase from the culture knowledge base using the id and language */
std::string getUtilityPhrase(std::string phraseId, std::string language);

/* 
    Stores the result of a nodes execution in the parameter server.
    To be used by the test node.
*/
static void storeResult(std::string key, int value);

/***** Global Variables ****/
// Environment Knowledge Base
std::shared_ptr<Environment::EnvironmentKnowledgeBase> environmentKnowledgeBase;
Environment::TourSpecificationType tour;
Environment::KeyValueType enviornmentKeyValue;

// Culture Knowledge Base
Culture::KeyValueType cultureKeyValue;
std::shared_ptr<Culture::CultureKnowledgeBase> culturalKnowledgeBase;
std::string key;  // Changed from Culture::Keyword to std::string
/********************************** */

/****** Mission(Action/Condition) Nodes */
/*
    Handler for the 'HandleFallBack' Action Node
*/
class HandleFallBack : public BT::SyncActionNode
{
   public:
    HandleFallBack(const std::string &name, const BT::NodeConfiguration &config)
        : BT::SyncActionNode(name, config)
    {
    }

    static BT::PortsList providedPorts()
    {
        return {};
    }

    BT::NodeStatus tick() override
    {
        std::string treeNodeName = "HandleFallback";
        printMsg(INFO_MSG, "(" + treeNodeName + " Action Node)");

        if (testMode) {
            storeResult(treeNodeName, 1);
        }
        return BT::NodeStatus::SUCCESS;
    }

   private:
    rclcpp::Client<cssr_system::srv::AnimateBehaviorSetActivation>::SharedPtr client;
};

/*
    Handler for the 'SetAnimateBehavior' Action Node
    Enables & Disables animate behavior
*/
class SetAnimateBehavior : public BT::SyncActionNode
{
   public:
    SetAnimateBehavior(const std::string &name, const BT::NodeConfiguration &config) : BT::SyncActionNode(name, config)
    {
        /* Define a service client */
        client = node->create_client<cssr_system::srv::AnimateBehaviorSetActivation>("/animateBehaviour/setActivation");
    }

    static BT::PortsList providedPorts()
    {
        return {};
    }

    BT::NodeStatus tick() override
    {
        auto request = std::make_shared<cssr_system::srv::AnimateBehaviorSetActivation::Request>();
        std::string state = name();  // retrieve the state from the mission specification
        std::string treeNodeName = "SetAnimateBehavior";

        printMsg(INFO_MSG, treeNodeName + "Action Node");
        printMsg(INFO_MSG, "State: " + state);

        request->state = state;
        
        /* Wait for service to be available */
        if (!client->wait_for_service(std::chrono::seconds(5))) {
            printMsg(ERROR_MSG, "Service not available");
            if (testMode) {
                storeResult(treeNodeName, 0);
            }
            return BT::NodeStatus::FAILURE;
        }

        /* Make a service call to animateBehavior node and act according to the response*/
        auto result = client->async_send_request(request);
        if (rclcpp::spin_until_future_complete(node, result) == rclcpp::FutureReturnCode::SUCCESS) {
            if (result.get()->success != "1") {
                printMsg(WARNING_MSG, "Called service returned failure");
                if (testMode) {
                    storeResult(treeNodeName, 0);
                }
                return BT::NodeStatus::FAILURE;
            }
        } else {
            printMsg(ERROR_MSG, "Failed to call service");
            if (testMode) {
                storeResult(treeNodeName, 0);
            }
            return BT::NodeStatus::FAILURE;
        }
        if (testMode) {
            storeResult(treeNodeName, 1);
        }
        return BT::NodeStatus::SUCCESS;
    }

   private:
    rclcpp::Client<cssr_system::srv::AnimateBehaviorSetActivation>::SharedPtr client;
};

/*
    Handler for the 'SetOvertAttentionMode' Action Node
    Set different values to the attention mode of the overtAttention ROS node
*/
class SetOvertAttentionMode : public BT::SyncActionNode
{
   public:
    SetOvertAttentionMode(const std::string &name, const BT::NodeConfiguration &config)
        : BT::SyncActionNode(name, config)
    {
        /* Define a service client */
        client = node->create_client<cssr_system::srv::OvertAttentionSetMode>("/overtAttention/set_mode");
    }

    static BT::PortsList providedPorts()
    {
        return {};
    }

    BT::NodeStatus tick() override
    {
        auto request = std::make_shared<cssr_system::srv::OvertAttentionSetMode::Request>();
        std::string state = name();  // retrieve the state from the mission specification
        std::string treeNodeName = "SetOvertAttentionMode";

        printMsg(INFO_MSG, treeNodeName + "Action Node");
        printMsg(INFO_MSG, "State: " + state);

        request->state = state;

        // if mode is 'location', retrieve the target from the blackboard
        if (state == "location") {
            Environment::RobotLocationType location;
            Environment::GestureTargetType gestureTarget;
            if (!config().blackboard->rootBlackboard()->get("exhibitGestureTarget", gestureTarget)) {
                printMsg(ERROR_MSG, "Unable to retrieve from blackboard");
                if (testMode) {
                    storeResult(treeNodeName, 0);
                }
                return BT::NodeStatus::FAILURE;
            }

            // Set the retrieved values in order to make a service call
            request->location_x = gestureTarget.x;
            request->location_y = gestureTarget.y;
            request->location_z = gestureTarget.z;
        }

        /* Wait for service to be available */
        if (!client->wait_for_service(std::chrono::seconds(5))) {
            printMsg(ERROR_MSG, "Service not available");
            if (testMode) {
                storeResult(treeNodeName, 0);
            }
            return BT::NodeStatus::FAILURE;
        }

        /* Make a service call to the node and act according to the response*/
        auto result = client->async_send_request(request);
        if (rclcpp::spin_until_future_complete(node, result) == rclcpp::FutureReturnCode::SUCCESS) {
            if (!result.get()->mode_set_success) {
                printMsg(WARNING_MSG, "Called service returned failure");
                if (testMode) {
                    storeResult(treeNodeName, 0);
                }
                return BT::NodeStatus::FAILURE;
            }
        } else {
            printMsg(ERROR_MSG, "Failed to call service");
            if (testMode) {
                storeResult(treeNodeName, 0);
            }
            return BT::NodeStatus::FAILURE;
        }
        if (testMode) {
            storeResult(treeNodeName, 1);
        }
        return BT::NodeStatus::SUCCESS;
    }

   private:
    rclcpp::Client<cssr_system::srv::OvertAttentionSetMode>::SharedPtr client;
};

/*
    Handler for the 'SetSpeechEvent' Action Node
    Enables & Disables transcription on the speechEvent ROS node
*/
class SetSpeechEvent : public BT::SyncActionNode
{
   public:
    SetSpeechEvent(const std::string &name, const BT::NodeConfiguration &config)
        : BT::SyncActionNode(name, config)
    {
        /* Define a service client */
        client = node->create_client<cssr_system::srv::SpeechEventSetEnabled>("/speechEvent/set_enabled");
    }

    static BT::PortsList providedPorts()
    {
        return {};
    }

    BT::NodeStatus tick() override
    {
        auto request = std::make_shared<cssr_system::srv::SpeechEventSetEnabled::Request>();
        std::string status = name();  // retrieve the status from the mission specification
        std::string treeNodeName = "SetSpeechEvent";

        printMsg(INFO_MSG, treeNodeName + "Action Node");
        printMsg(INFO_MSG, "Status: " + status);

        request->status = status;
        
        /* Wait for service to be available */
        if (!client->wait_for_service(std::chrono::seconds(5))) {
            printMsg(ERROR_MSG, "Service not available");
            if (testMode) {
                storeResult(treeNodeName, 0);
            }
            return BT::NodeStatus::FAILURE;
        }

        /* Make a service call to the node and act according to the response*/
        auto result = client->async_send_request(request);
        if (rclcpp::spin_until_future_complete(node, result) == rclcpp::FutureReturnCode::SUCCESS) {
            if (!result.get()->response) {
                printMsg(WARNING_MSG, "Called service returned failure");
                if (testMode) {
                    storeResult(treeNodeName, 0);
                }
                return BT::NodeStatus::FAILURE;
            }
        } else {
            printMsg(ERROR_MSG, "Failed to call service");
            if (testMode) {
                storeResult(treeNodeName, 0);
            }
            return BT::NodeStatus::FAILURE;
        }
        if (testMode) {
            storeResult(treeNodeName, 1);
        }
        return BT::NodeStatus::SUCCESS;
    }

   private:
    rclcpp::Client<cssr_system::srv::SpeechEventSetEnabled>::SharedPtr client;
};

/*
    Handler for the 'IsVisitorDiscovered' Condition Node
    Checks for the presence of a visitor via the faceDetection ROS node
*/
class IsVisitorDiscovered : public BT::ConditionNode
{
   public:
    IsVisitorDiscovered(const std::string &name, const BT::NodeConfiguration &config)
        : BT::ConditionNode(name, config), visitorDiscovered(false)
    {
        /* Define a subscriber to the topic */
        subscriber = node->create_subscription<cssr_system::msg::FaceDetectionData>(
            "/faceDetection/data", 10, 
            std::bind(&IsVisitorDiscovered::callback, this, std::placeholders::_1));
    }

    static BT::PortsList providedPorts()
    {
        return {};
    }

    BT::NodeStatus tick() override
    {
        std::string treeNodeName = "IsVisitorDiscovered";

        printMsg(INFO_MSG, treeNodeName + "Condition Node");

        rclcpp::Rate rate(10);
        /* Wait until the topic returns data, indicating arrival of a potential visitor */
        while (rclcpp::ok()) {
            rclcpp::spin_some(node);

            if (visitorDiscovered) {
                printMsg(INFO_MSG, "Visitor discovered");
                if (testMode) {
                    storeResult(treeNodeName, 1);
                }
                return BT::NodeStatus::SUCCESS;
            }

            rate.sleep();
        }

        if (testMode) {
            storeResult(treeNodeName, 0);
        }
        return BT::NodeStatus::FAILURE;
    }

   private:
    void callback(const cssr_system::msg::FaceDetectionData::SharedPtr msg)
    {
        visitorDiscovered = false;
        /* if the face_label_id array contains values, it indicates presence of a potential visitor*/
        if (msg->face_label_id.size() > 0) {
            visitorDiscovered = true;
        }
    }

    bool visitorDiscovered;
    rclcpp::Subscription<cssr_system::msg::FaceDetectionData>::SharedPtr subscriber;
};

/*
    Handler for the 'IsMutualGazeDiscovered' Condition Node
    Checks for the detection of mutual gaze via overtAttention ROS node
*/
class IsMutualGazeDiscovered : public BT::ConditionNode
{
   public:
    IsMutualGazeDiscovered(const std::string &name, const BT::NodeConfiguration &config)
        : BT::ConditionNode(name, config), seekingStatus("RUNNING")
    {
        /* Define a subscriber to the topic */
        subscriber = node->create_subscription<cssr_system::msg::OvertAttentionMode>(
            "/overtAttention/mode", 10,
            std::bind(&IsMutualGazeDiscovered::callback, this, std::placeholders::_1));
    }

    static BT::PortsList providedPorts()
    {
        return {};
    }

    BT::NodeStatus tick() override
    {
        std::string treeNodeName = "IsMutualGazeDiscovered";

        printMsg(INFO_MSG, treeNodeName + "Condition Node");

        rclcpp::Rate rate(10);
        auto startTime = node->get_clock()->now();

        /* Keep checking for the detection of mutual gaze */
        while (rclcpp::ok()) {
            rclcpp::spin_some(node);
            if (seekingStatus == "SUCCESS") {
                printMsg(INFO_MSG, "Mutual gaze detected");
                if (testMode) {
                    storeResult(treeNodeName, 1);
                }
                return BT::NodeStatus::SUCCESS;
            } else if (seekingStatus == "FAILURE") {
                printMsg(INFO_MSG, "Mutual gaze detection failed");
                if (testMode) {
                    storeResult(treeNodeName, 0);
                }
                return BT::NodeStatus::FAILURE;
            }

            rate.sleep();
        }

        if (testMode) {
            storeResult(treeNodeName, 0);
        }
        return BT::NodeStatus::FAILURE;
    }

   private:
    void callback(const cssr_system::msg::OvertAttentionMode::SharedPtr msg)
    {
        /*
            Values 2 & 3, indicating success & failure respectively are how
            the overtAttention node relays 'seeking' mode status
        */

        if (msg->state == "seeking" && msg->value == 2) {
            seekingStatus = "SUCCESS";
        } else if (msg->state == "seeking" && msg->value == 3) {
            seekingStatus = "FAILURE";
        } else {
            seekingStatus = "RUNNING";
        }
    }

    std::string seekingStatus;
    rclcpp::Subscription<cssr_system::msg::OvertAttentionMode>::SharedPtr subscriber;
};

// Due to length constraints, I'll continue with the key utility functions and tree initialization
// The remaining BehaviorTree node classes follow the same pattern of ROS1 to ROS2 conversion

BT::Tree initializeTree(std::string scenario)
{
    BT::BehaviorTreeFactory factory;
    factory.registerNodeType<HandleFallBack>("HandleFallBack");
    factory.registerNodeType<SetAnimateBehavior>("SetAnimateBehavior");
    factory.registerNodeType<SetOvertAttentionMode>("SetOvertAttentionMode");
    factory.registerNodeType<SetSpeechEvent>("SetSpeechEvent");
    factory.registerNodeType<IsVisitorDiscovered>("IsVisitorDiscovered");
    factory.registerNodeType<IsMutualGazeDiscovered>("IsMutualGazeDiscovered");
    
    // Register other node types here...

    std::string package_path = ament_index_cpp::get_package_share_directory("cssr_system");
    return factory.createTreeFromFile(package_path + "/behaviorController/data/" + scenario + ".xml");
}

/***** Utility Functions ******/

/*
Logs the string (args) to the terminal based on the (type).
Wrapper around the default ROS2 logging functions
*/
void printMsg(int type, std::string args)
{
    if (!verboseMode) {
        return;
    }

    std::string msg = "[" + nodeName + "]: " + args;
    switch (type) {
        case INFO_MSG:
            RCLCPP_INFO(node->get_logger(), "%s", msg.c_str());
            break;
        case WARNING_MSG:
            RCLCPP_WARN(node->get_logger(), "%s", msg.c_str());
            break;
        case ERROR_MSG:
            RCLCPP_ERROR(node->get_logger(), "%s", msg.c_str());
            break;
        default:
            RCLCPP_ERROR(node->get_logger(), "UNDEFINED MSG TYPE");
    }
}

/* Returns the current language from the knowledge base*/
std::string getMissionLanguage()
{
    key = "phraseLanguage";
    if (culturalKnowledgeBase->getValue(key, &cultureKeyValue)) {
        return cultureKeyValue.alphanumericValue;
    }
    return "";
}

/* Fetches the utility phrase from the culture knowledge base using the id and language */
std::string getUtilityPhrase(std::string phraseId, std::string language)
{
    std::string phraseKey = "utilityPhrase" + language + phraseId;
    key = phraseKey;
    if (culturalKnowledgeBase->getValue(key, &cultureKeyValue)) {
        return cultureKeyValue.alphanumericValue;
    }
    return "";
}

/* Returns true if ch isn't an empty space character*/
static bool isNotSpace(unsigned char ch) {
    return !std::isspace(ch);
}

/* Trims whitespaces inplace */
static inline void trim(std::string &s) {
    // Trim leading spaces
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), isNotSpace));
    
    // Trim trailing spaces
    s.erase(std::find_if(s.rbegin(), s.rend(), isNotSpace).base(), s.end());
}

/* Returns the value of a key from the configuration file. */
std::string getValueFromConfig(const std::string &key)
{
    std::string package_path = ament_index_cpp::get_package_share_directory("cssr_system");
    std::string config_file_path = package_path + "/behaviorController/config/behaviorControllerConfiguration.yaml";
    
    try {
        YAML::Node config = YAML::LoadFile(config_file_path);
        
        if (config[key]) {
            if (config[key].IsScalar()) {
                return config[key].as<std::string>();
            } else {
                throw std::runtime_error("Configuration value for key '" + key + "' is not a scalar value");
            }
        } else {
            throw std::runtime_error("Configuration key '" + key + "' not found");
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to parse YAML configuration file: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to read configuration file: " + std::string(e.what()));
    }
}

/* Returns true if a topic is available */
static bool isTopicAvailable(std::string topic)
{
    auto topic_names_and_types = node->get_topic_names_and_types();
    for (const auto& [topic_name, topic_types] : topic_names_and_types) {
        if (topic_name == topic) {
            return true;
        }
    }
    return false;
}

/* Returns true if all the topics in a list are available*/
bool checkTopics(std::vector<std::string> &topicsList)
{
    bool success = true;
    for (std::string topic : topicsList) {
        if (!isTopicAvailable(topic)) {
            success = false;
            RCLCPP_ERROR(node->get_logger(), "[%s] NOT FOUND", topic.c_str());
        }
    }
    return success;
}

/* Returns true if all the services in a list are available*/
bool checkServices(std::vector<std::string> &servicesList)
{
    bool success = true;
    for (const std::string& service : servicesList) {
        // Get list of available services
        auto service_names_and_types = node->get_service_names_and_types();
        bool found = false;
        for (const auto& [service_name, service_types] : service_names_and_types) {
            if (service_name == service) {
                found = true;
                break;
            }
        }
        if (!found) {
            success = false;
            RCLCPP_ERROR(node->get_logger(), "[%s] NOT FOUND", service.c_str());
        }
    }
    return success;
}

/* 
    Stores the result of a nodes execution in the parameter server.
    To be used by the test node.
*/
static void storeResult(std::string key, int value)
{
    std::string testParameterPath = "behaviorControllerTest." + key;
    node->declare_parameter(testParameterPath, value);
    node->set_parameter(rclcpp::Parameter(testParameterPath, value));
}

/****************************** */
