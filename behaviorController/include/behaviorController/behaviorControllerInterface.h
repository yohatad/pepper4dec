/**
 * -----------------------------------------------------------------------------
 * @file    behaviorControllerInterface.h
 * @brief   Public interface declarations for the Robot Mission Interpreter node,
 *          including class definitions and utility function prototypes.
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


#ifndef BEHAVIOR_CONTROLLER_INTERFACE_H
#define BEHAVIOR_CONTROLLER_INTERFACE_H

#include "rclcpp/rclcpp.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

#include "behaviortree_cpp/bt_factory.h"
#include <behaviortree_cpp/loggers/groot2_publisher.h>

#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

#include "utilities/cultureKnowledgeBaseInterface.h"
#include "utilities/environmentKnowledgeBaseInterface.h"

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

/* Configuration variables */
extern bool verboseMode;
extern bool asrEnabled;
extern std::string missionLanguage;
extern bool testMode;

extern std::string nodeName;

//Node Handler
extern std::shared_ptr<rclcpp::Node> node;

/* Returns true if all the topics in a list are available*/
bool checkTopics(std::vector<std::string>& topicsList);

/* Returns true if all the services in a list are available*/
bool checkServices(std::vector<std::string>& servicesList);

/* Returns the value of a key from the configuration file. */
std::string getValueFromConfig(const std::string &key);

/* 
    Returns a behavior tree of type BT::Tree from the scenario specification file
    Scenario specification file must be placed in the data directory
*/
BT::Tree initializeTree(std::string scenario);

/* Returns the current language from the knowledge base*/
std::string getMissionLanguage();
#endif
