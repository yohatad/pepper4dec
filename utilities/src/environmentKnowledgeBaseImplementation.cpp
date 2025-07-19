/* environmentKnowledgeBaseImplementation.cpp   Source code for the implementation of the environment knowledge base helper class: EnvironmentKnowledgeBase
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

/*
 * This software implements a C++ helper class to read the environment knowledge base file and query the knowledge through access methods.
 * The helper class uses a std::map dictionary to represent the knowledge base (simplified from binary search tree).
 * It provides three public methods to access the knowledge:
 * 
 *    getValue()
 *    getTour()
 *    printToScreen()
 *
 * The knowledge base is built automatically by the constructor when the EnvironmentKnowledgeBase class is instantiated as an object, 
 * reading the key-value pairs from the files specified in the configuration file.
 * The destructor cleans up the dictionary data structure.
 *
 * Libraries
 * ---------
 * stdio, string, stdlib, map, yaml-cpp
 *
 * ROS libraries
 * -------------
 * rclcpp, ament_index_cpp
 *
 * Parameters
 * ----------
 * None.
 *
 * Command-line Parameters
 * ----------------------
 * None
 *
 * Configuration File Parameters 
 * -----------------------------
 *
 * Key              | Value 
 * knowledge_base   | environmentKnowledgeBaseInput.dat
 * verbose_mode     | false
 *
 * Author:   David Vernon, Carnegie Mellon University Africa (Original)
 * Modified for ROS2 and simplified data structures
 * Date:     July 2025
 * Version:  v2.0
 */

#include <utilities/environmentKnowledgeBaseInterface.h>

namespace Environment{

/*
 * EnvironmentKnowledgeBase::EnvironmentKnowledgeBase
 * --------------------------------------------------
 *
 * Class constructor
 *   Read the configuration data from the configuration file
 *   Read the knowledge base and build the map dictionary data structure
 */

EnvironmentKnowledgeBase::EnvironmentKnowledgeBase(std::shared_ptr<rclcpp::Node> node) : node_(node) {
   
   readConfigurationData();

   if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), "EnvironmentKnowledgeBase()");

   knowledgeMap_.clear(); // Initialize empty map
   tourSpecification.numberOfLocations = 0;

   readKnowledgeBase();
}

/*
 * EnvironmentKnowledgeBase::~EnvironmentKnowledgeBase
 * ---------------------------------------------------
 *
 * Class destructor
 *    Clean up the map dictionary data structure
 */

EnvironmentKnowledgeBase::~EnvironmentKnowledgeBase() {

   if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), "~EnvironmentKnowledgeBase()");

   // Clean up allocated memory for cultural knowledge keys
   for (auto& pair : knowledgeMap_) {
       for (int i = 0; i < pair.second.culturalKnowledge.numberOfKeys; i++) {
           // Note: std::string handles its own memory management
       }
   }
   knowledgeMap_.clear();
}

/*
 * bool EnvironmentKnowledgeBase::getTour(TourSpecificationType  *tour)
 * -------------------------------------------------------------------- 
 *
 * return true if the tour has one or more locations; false otherwise
 */

bool EnvironmentKnowledgeBase::getTour(TourSpecificationType  *tour) {
   if (tourSpecification.numberOfLocations > 0) {
       tour->numberOfLocations = tourSpecification.numberOfLocations;

       for (int i = 0; i < tourSpecification.numberOfLocations; i++) {
           tour->locationIdNumber[i] = tourSpecification.locationIdNumber[i];
       }
       return true;
   } else {
       tour->numberOfLocations = 0;
       return false;
   }
}

/*
 * bool EnvironmentKnowledgeBase::getValue(int key, KeyValueType *keyValue)
 * -------------------------------------------------------------------------- 
 *
 * Return true if the key is in the map; false otherwise
 */

bool EnvironmentKnowledgeBase::getValue(int key, KeyValueType *keyValue) {
   auto it = knowledgeMap_.find(key);
   if (it != knowledgeMap_.end()) {
       *keyValue = it->second;
       return true;
   } else {
       // Initialize with default values if key not found
       keyValue->key = 0;
       keyValue->robotLocation.x = 0;
       keyValue->robotLocation.y = 0;
       keyValue->robotLocation.theta = 0;
       keyValue->robotLocationDescription = "";
       keyValue->gestureTarget.x = 0;
       keyValue->gestureTarget.y = 0;
       keyValue->gestureTarget.z = 0;
       keyValue->preGestureMessageEnglish = "";
       keyValue->preGestureMessageIsiZulu = "";
       keyValue->preGestureMessageKinyarwanda = "";
       keyValue->postGestureMessageEnglish = "";
       keyValue->postGestureMessageIsiZulu = "";
       keyValue->postGestureMessageKinyarwanda = "";
       keyValue->culturalKnowledge.numberOfKeys = 0;
       return false;
   }
}

/*
 * void EnvironmentKnowledgeBase::printToScreen() 
 * ----------------------------------------------
 *
 * Print all elements in the knowledge base
 */

void EnvironmentKnowledgeBase::printToScreen() {
   for (const auto& pair : knowledgeMap_) {
       const KeyValueType& kv = pair.second;
       
       printf("Key                               %-4d \n"
              "Location Description              %s \n"
              "Robot Location                    (%.1f, %.1f  %.1f)\n"
              "Gesture Target                    (%.1f, %.1f  %.1f) \n"
              "Pre-Gesture Message English       %s \n"
              "Pre-Gesture Message isiZulu       %s \n"
              "Pre-Gesture Message Kinyarwanda   %s \n"
              "Post-Gesture Message English      %s \n"
              "Post-Gesture Message isiZulu      %s \n"
              "Post-Gesture Message Kinyarwanda  %s \n",
              kv.key,
              kv.robotLocationDescription.c_str(),
              kv.robotLocation.x, kv.robotLocation.y, kv.robotLocation.theta,
              kv.gestureTarget.x, kv.gestureTarget.y, kv.gestureTarget.z,
              kv.preGestureMessageEnglish.c_str(),
              kv.preGestureMessageIsiZulu.c_str(),
              kv.preGestureMessageKinyarwanda.c_str(),
              kv.postGestureMessageEnglish.c_str(),
              kv.postGestureMessageIsiZulu.c_str(),
              kv.postGestureMessageKinyarwanda.c_str());
   
       printf("Cultural Knowledge                ");
       for (int k = 0; k < kv.culturalKnowledge.numberOfKeys; k++) {
           printf("%s ", kv.culturalKnowledge.key[k].c_str());
       }
       printf("\n\n");
   }
   printf("\n");
}

/*
 * void EnvironmentKnowledgeBase::readConfigurationData() 
 * ------------------------------------------------------
 *
 * Read configuration parameters key-value pairs from the YAML configuration file 
 */

void EnvironmentKnowledgeBase::readConfigurationData() {
   bool debug = false;
   
   try {
       // Get package share directory using ROS2 method
       std::string package_share_directory = ament_index_cpp::get_package_share_directory(ROS_PACKAGE_NAME);
       std::string config_path = package_share_directory + "/behaviorController/include/utilities/config/" + configuration_filename;
       
       if (debug) RCLCPP_INFO(node_->get_logger(), "Configuration file: %s", config_path.c_str());
       
       // Load YAML file
       YAML::Node config = YAML::LoadFile(config_path);
       
       // Read configuration parameters
       configurationData.verboseMode = config["verbose_mode"].as<bool>();
       
       std::string knowledge_base_file = config["knowledge_base"].as<std::string>();
       
       // Construct full path
       configurationData.knowledgeBase = package_share_directory + "/behaviorController/include/utilities/data/" + knowledge_base_file;
       
       if (configurationData.verboseMode) {
           RCLCPP_INFO(node_->get_logger(), "readConfigurationData: knowledgeBase %s", configurationData.knowledgeBase.c_str());
           RCLCPP_INFO(node_->get_logger(), "readConfigurationData: verboseMode %d", configurationData.verboseMode);
       }
       
   } catch (const std::exception& e) {
       RCLCPP_ERROR(node_->get_logger(), "Error reading configuration file: %s", e.what());
       throw;
   }
}

/*
 * void EnvironmentKnowledgeBase::readKnowledgeBase()
 * --------------------------------------------------
 *
 * Read knowledge base key-value pairs from the data file 
 */

void EnvironmentKnowledgeBase::readKnowledgeBase() {
   bool debug = false;
   char input_string[MAX_STRING_LENGTH];
   char alphanumericValue[MAX_STRING_LENGTH];
   int idNumber;
   int i, j, k, n;
   int numberOfCultureKnowledgeKeys;
   FILE *fp_in;

   std::vector<std::string> keylist = {
       "robotLocationPose",
       "robotLocationDescription", 
       "gestureTarget",
       "preGestureMessageEnglish",
       "preGestureMessageIsiZulu",
       "preGestureMessageKinyarwanda",
       "postGestureMessageEnglish",
       "postGestureMessageIsiZulu",
       "postGestureMessageKinyarwanda",
       "tourSpecification",
       "culturalKnowledge"
   };
      
   char key[KEY_LENGTH];
   char cultureKey[KEY_LENGTH];

   if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: initializing knowledge base values");
   
   // Open the knowledge base file
   if ((fp_in = fopen(configurationData.knowledgeBase.c_str(),"r")) == 0) {  
       RCLCPP_ERROR(node_->get_logger(), "Error can't open knowledge base file %s", configurationData.knowledgeBase.c_str());
       return;
   }

   // Read the key value pairs
   while (fgets(input_string, STRING_LENGTH, fp_in) != NULL) {
       bool insertFlag = false;
       KeyValueType keyValue;
       
       // Extract the key and idNumber
       if (sscanf(input_string, "%s %d", key, &idNumber) == 2) {
           
           // Get existing keyValue if it exists, otherwise use default
           auto it = knowledgeMap_.find(idNumber);
           if (it != knowledgeMap_.end()) {
               keyValue = it->second;
           } else {
               // Initialize new keyValue
               keyValue.key = idNumber;
               keyValue.robotLocation.x = 0;
               keyValue.robotLocation.y = 0;
               keyValue.robotLocation.theta = 0;
               keyValue.robotLocationDescription = "";
               keyValue.gestureTarget.x = 0;
               keyValue.gestureTarget.y = 0;
               keyValue.gestureTarget.z = 0;
               keyValue.preGestureMessageEnglish = "";
               keyValue.preGestureMessageIsiZulu = "";
               keyValue.preGestureMessageKinyarwanda = "";
               keyValue.postGestureMessageEnglish = "";
               keyValue.postGestureMessageIsiZulu = "";
               keyValue.postGestureMessageKinyarwanda = "";
               keyValue.culturalKnowledge.numberOfKeys = 0;
           }

           std::string keyStr(key);
           
           for (size_t j = 0; j < keylist.size(); j++) {
               if (keyStr == keylist[j]) {
                   switch (j) {
                       case 0: // robotLocationPose
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace
                               
                               if (sscanf(input_string+i, "%f %f %f", &(keyValue.robotLocation.x), &(keyValue.robotLocation.y), &(keyValue.robotLocation.theta)) == 3) {
                                   insertFlag = true;
                                   if (configurationData.verboseMode) 
                                       RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s %d %5.1f %5.1f %5.1f", key, keyValue.key, keyValue.robotLocation.x, keyValue.robotLocation.y, keyValue.robotLocation.theta);
                               }
                           }
                           break;

                       case 1: // robotLocationDescription
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace
                               
                               keyValue.robotLocationDescription = std::string(input_string + i);
                               // Remove newline
                               if (!keyValue.robotLocationDescription.empty() && keyValue.robotLocationDescription.back() == '\n') {
                                   keyValue.robotLocationDescription.pop_back();
                               }
                               insertFlag = true;
                               if (configurationData.verboseMode) 
                                   RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s %d %s", key, keyValue.key, keyValue.robotLocationDescription.c_str());
                           }
                           break;

                       case 2: // gestureTarget
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace
                               
                               if (sscanf(input_string+i, "%f %f %f", &(keyValue.gestureTarget.x), &(keyValue.gestureTarget.y), &(keyValue.gestureTarget.z)) == 3) {
                                   insertFlag = true;
                                   if (configurationData.verboseMode) 
                                       RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s %d %5.1f %5.1f %5.1f", key, keyValue.key, keyValue.gestureTarget.x, keyValue.gestureTarget.y, keyValue.gestureTarget.z);
                               }
                           }
                           break;

                       case 3: // preGestureMessageEnglish
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace
                               
                               keyValue.preGestureMessageEnglish = std::string(input_string + i);
                               if (!keyValue.preGestureMessageEnglish.empty() && keyValue.preGestureMessageEnglish.back() == '\n') {
                                   keyValue.preGestureMessageEnglish.pop_back();
                               }
                               insertFlag = true;
                               if (configurationData.verboseMode) 
                                   RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s %d %s", key, keyValue.key, keyValue.preGestureMessageEnglish.c_str());
                           }
                           break;

                       case 4: // preGestureMessageIsiZulu
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace
                               
                               keyValue.preGestureMessageIsiZulu = std::string(input_string + i);
                               if (!keyValue.preGestureMessageIsiZulu.empty() && keyValue.preGestureMessageIsiZulu.back() == '\n') {
                                   keyValue.preGestureMessageIsiZulu.pop_back();
                               }
                               insertFlag = true;
                               if (configurationData.verboseMode) 
                                   RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s %d %s", key, keyValue.key, keyValue.preGestureMessageIsiZulu.c_str());
                           }
                           break;

                       case 5: // preGestureMessageKinyarwanda
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace
                               
                               keyValue.preGestureMessageKinyarwanda = std::string(input_string + i);
                               if (!keyValue.preGestureMessageKinyarwanda.empty() && keyValue.preGestureMessageKinyarwanda.back() == '\n') {
                                   keyValue.preGestureMessageKinyarwanda.pop_back();
                               }
                               insertFlag = true;
                               if (configurationData.verboseMode) 
                                   RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s %d %s", key, keyValue.key, keyValue.preGestureMessageKinyarwanda.c_str());
                           }
                           break;

                       case 6: // postGestureMessageEnglish
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace
                               
                               keyValue.postGestureMessageEnglish = std::string(input_string + i);
                               if (!keyValue.postGestureMessageEnglish.empty() && keyValue.postGestureMessageEnglish.back() == '\n') {
                                   keyValue.postGestureMessageEnglish.pop_back();
                               }
                               insertFlag = true;
                               if (configurationData.verboseMode) 
                                   RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s %d %s", key, keyValue.key, keyValue.postGestureMessageEnglish.c_str());
                           }
                           break;

                       case 7: // postGestureMessageIsiZulu
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace
                               
                               keyValue.postGestureMessageIsiZulu = std::string(input_string + i);
                               if (!keyValue.postGestureMessageIsiZulu.empty() && keyValue.postGestureMessageIsiZulu.back() == '\n') {
                                   keyValue.postGestureMessageIsiZulu.pop_back();
                               }
                               insertFlag = true;
                               if (configurationData.verboseMode) 
                                   RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s %d %s", key, keyValue.key, keyValue.postGestureMessageIsiZulu.c_str());
                           }
                           break;

                       case 8: // postGestureMessageKinyarwanda
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace
                               
                               keyValue.postGestureMessageKinyarwanda = std::string(input_string + i);
                               if (!keyValue.postGestureMessageKinyarwanda.empty() && keyValue.postGestureMessageKinyarwanda.back() == '\n') {
                                   keyValue.postGestureMessageKinyarwanda.pop_back();
                               }
                               insertFlag = true;
                               if (configurationData.verboseMode) 
                                   RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s %d %s", key, keyValue.key, keyValue.postGestureMessageKinyarwanda.c_str());
                           }
                           break;

                       case 9: // tourSpecification
                           {
                               if (sscanf(input_string, "%s", key)) {
                                   if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %-30s", key);

                                   i = strlen(key);
                                   while(!isalnum(input_string[i])) i++; // skip key
                                   
                                   if (sscanf(input_string+i, "%d", &n) == 1) {
                                       tourSpecification.numberOfLocations = n;
                                       if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), " %d", tourSpecification.numberOfLocations);
                                       
                                       while(isalnum(input_string[i])) i++; // skip numberOfLocations
                                       while(!isalnum(input_string[i])) i++; // skip whitespace

                                       for (k = 0; k < n; k++) {
                                           if (sscanf(input_string+i, "%d", &(tourSpecification.locationIdNumber[k])) == 1) {
                                               if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), " %d", tourSpecification.locationIdNumber[k]);
                                               while(isalnum(input_string[i])) i++; // skip locationIdNumber
                                               while(!isalnum(input_string[i])) i++; // skip whitespace
                                           }
                                       }
                                   }
                               }
                           }
                           break;

                       case 10: // culturalKnowledge
                           {
                               i = strlen(key);
                               while(!isalnum(input_string[i])) i++; // skip key
                               while(isalnum(input_string[i])) i++;  // skip idNumber
                               while(!isalnum(input_string[i])) i++; // skip whitespace

                               numberOfCultureKnowledgeKeys = 0;
                               while (i < strlen(input_string) && numberOfCultureKnowledgeKeys < MAX_CULTURE_KEYS) {
                                   if (sscanf(input_string+i, "%s", cultureKey) == 1) {
                                       keyValue.culturalKnowledge.key[numberOfCultureKnowledgeKeys] = std::string(cultureKey);
                                       while (isalnum(input_string[i])) i++; // skip key
                                       while(!isalnum(input_string[i]) && i < strlen(input_string)) i++; // skip whitespace
                                       numberOfCultureKnowledgeKeys++;
                                   } else {
                                       break;
                                   }
                               }

                               keyValue.culturalKnowledge.numberOfKeys = numberOfCultureKnowledgeKeys;
                               insertFlag = true;

                               if (configurationData.verboseMode) {
                                   std::string msg = "readKnowledgeBase: " + std::string(key) + " " + std::to_string(keyValue.key) + " ";
                                   for (k = 0; k < keyValue.culturalKnowledge.numberOfKeys; k++) {
                                       msg += keyValue.culturalKnowledge.key[k] + " ";
                                   }
                                   RCLCPP_INFO(node_->get_logger(), "%s", msg.c_str());
                               }
                           }
                           break;

                       default:
                           RCLCPP_ERROR(node_->get_logger(), "readKnowledgeBase: invalid key %s", key);
                           break;
                   }
                   break;
               }
           }

           // Add to the knowledge base
           if (insertFlag) {
               knowledgeMap_[idNumber] = keyValue;
           }
       }
   }

   fclose(fp_in);
}

/*
 * int EnvironmentKnowledgeBase::size() 
 * ------------------------------------
 *
 * Return the size of the knowledge base, i.e. the total number of entries
 */

int EnvironmentKnowledgeBase::size() {
   return knowledgeMap_.size();
}

}
