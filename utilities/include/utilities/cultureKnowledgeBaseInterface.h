/* cultureKnowledgeBaseInterface.h   Interface source code for the culture knowledge base helper class: CultureKnowledgeBase
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
#include "utilities/behaviorControllerUtilities.h"

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
#define MAX_FILENAME_LENGTH          200
#define STRING_LENGTH                200
#define KEY_LENGTH                   100


/***************************************************************************************************************************
 
   Class CultureKnowledgeBase
   
   A class to represent the culture knowledge base using std::map instead of binary search tree

****************************************************************************************************************************/

#define NUMBER_OF_VALUE_TYPES          4

/* constant definitions for valueType flag to identify which element of the union is to be used */

#define UNDEFINED 0 // value hasn't been initialized
#define NUMBER    1 // value is integer
#define WORD      2 // value is alphanumeric but just one word
#define PHRASE    3 // value is alphanumeric but several words


#ifndef CULTURE_KNOWLEDGE_BASE_INTERFACE_H
#define CULTURE_KNOWLEDGE_BASE_INTERFACE_H
namespace Culture{

typedef struct {
   std::string knowledgeBase;
   std::string valueTypes;
   bool verboseMode;
} ConfigurationDataType;

typedef struct {
    std::string key;
    union {
        int   integerValue;
        char* alphanumericValue;  
    };
    int valueType;
    bool initialized;
} KeyValueType;

class CultureKnowledgeBase {

public:
   CultureKnowledgeBase(std::shared_ptr<rclcpp::Node> node);
   ~CultureKnowledgeBase();
   
   bool                  getValue(const std::string& key, KeyValueType *keyValue);                         // return true if the key is in the dictionary; false otherwise
   void                  printToScreen();                                                                  // print all elements in the knowledge base

private:
   std::shared_ptr<rclcpp::Node> node_;
   std::map<std::string, KeyValueType> knowledgeMap_;  // Using std::map instead of binary tree
   ConfigurationDataType configurationData;
   std::string configuration_filename = "cultureKnowledgeBaseConfiguration.yaml";

   void                  assign_key_attributes(KeyValueType *keyValue, const std::string& key, int valueType, bool operational); // assign a key, value, value type, and operational flag to a key-value pair
   void                  assign_key_value(KeyValueType *keyValue, int integerValue,  bool operational);              // assign an integer value to a key-value pair
   void                  assign_key_value(KeyValueType *keyValue, const std::string& alphanumericValue, bool operational);        // assign a string value to a key-value pair
   bool                  exists(const std::string& key);                                                          // return true if the key is in the dictionary; false otherwise
   int                   getValueType(const std::string& key);                                                    // return the integer code for the value type of the key
   void                  readConfigurationData();                                                                    // read configuration parameters key-value pairs from the configuration file 
   void                  readKnowledgeBase();                                                                        // read knowledge base key-value pairs from the data file
   void                  readKnowledgeBaseValueTypes();                                                              // read the types of the values in the knowledge base key-value pairs from the data file
   int                   size();                                                                                     // returns the size of the knowledge base
};

/***************************************************************************************************************************
 
   Utility function prototypes 
   
****************************************************************************************************************************/

/* convert integer valueType to alphnumeric */
std::string valueType2Alphanumeric(int valueType);

/* convert Boolean operational to alphnumeric */
std::string initialized2Alphanumeric(bool initialized);
}

#endif
