/* environmentKnowledgeBaseInterface.h   Interface source code for the environment knowledge base helper class: EnvironmentKnowledgeBase
 *
 * Copyright (C) 2023 CSSR4Africa Consortium
 *
 * This project is funded by the African Engineering and Technology Network (Afretec)
 * Inclusive Digital Transformation Research Grant Programme.
 *
 * Website: www.cssr4africa.org
 *
 * This program comes with ABSOLUTELY NO WARRANTY.
 *
 */
 
#include <stdio.h>
#include <ctype.h>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <utilities/behaviorControllerUtilities.h>

using namespace std;

/***************************************************************************************************************************

   ROS package name

****************************************************************************************************************************/
#ifndef ROS_PACKAGE_NAME
  #define ROS_PACKAGE_NAME "cssr_system"
#endif


/***************************************************************************************************************************

   General purpose definitions 

****************************************************************************************************************************/
#define TRUE                           1
#define FALSE                          0
#define MAX_STRING_LENGTH            200
#define MAX_FILENAME_LENGTH          200
#define STRING_LENGTH                200
#define KEY_LENGTH                   100
#define MAX_CULTURE_KEYS               2


/***************************************************************************************************************************
 
   Class EnvironmentKnowledgeBase
   
   A class to represent the environment knowledge base using std::map instead of binary search tree

****************************************************************************************************************************/
#define NUMBER_OF_VALUE_KEYS          13
#define MAX_NUMBER_OF_TOUR_LOCATIONS  20

#ifndef ENVIRONMENT_KNOWLEDGE_BASE_INTERFACE_H
#define ENVIRONMENT_KNOWLEDGE_BASE_INTERFACE_H

namespace Environment{

typedef struct {
   std::string knowledgeBase;
   bool verboseMode;
} ConfigurationDataType;

typedef struct {
   float x;
   float y;
   float theta;
} RobotLocationType;

typedef struct {
   float x;
   float y;
   float z;
} GestureTargetType;

typedef struct {
   int numberOfLocations;
   int locationIdNumber[MAX_NUMBER_OF_TOUR_LOCATIONS];
} TourSpecificationType;

typedef struct {
    std::string key[MAX_CULTURE_KEYS];
    int numberOfKeys;
} CulturalKnowledgeType;

typedef struct {
    int                   key; // i.e., idNumber
    RobotLocationType     robotLocation;
    GestureTargetType     gestureTarget;
    CulturalKnowledgeType culturalKnowledge;
    std::string           robotLocationDescription;
    std::string           preGestureMessageEnglish;
    std::string           preGestureMessageIsiZulu;
    std::string           preGestureMessageKinyarwanda;
    std::string           postGestureMessageEnglish;
    std::string           postGestureMessageIsiZulu;
    std::string           postGestureMessageKinyarwanda;
} KeyValueType;

class EnvironmentKnowledgeBase {

public:
   EnvironmentKnowledgeBase(std::shared_ptr<rclcpp::Node> node);
   ~EnvironmentKnowledgeBase();
   
   bool                  getValue(int key, KeyValueType *keyValue);                           // return true if the key is in the dictionary; false otherwise
   bool                  getTour(TourSpecificationType  *tour);                               // return true if the tour has one or more locations; false otherwise
   void                  printToScreen();                                                     // print all elements in the knowledge base

private:
   std::shared_ptr<rclcpp::Node> node_;
   std::map<int, KeyValueType> knowledgeMap_;  // Using std::map instead of binary tree
   TourSpecificationType tourSpecification;
   ConfigurationDataType configurationData;
   std::string configuration_filename = "environmentKnowledgeBaseConfiguration.yaml";

   void                  readConfigurationData();                                             // read configuration parameters key-value pairs from the configuration file 
   void                  readKnowledgeBase();                                                 // read knowledge base key-value pairs from the data file
   int                   size();                                                              // returns the size of the knowledge base
};

}

#endif
