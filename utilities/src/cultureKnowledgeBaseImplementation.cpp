/* cultureKnowledgeBaseImplementation.cpp   Source code for the implementation of the culture knowledge base helper class: CultureKnowledgeBase
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
 * This software implements a C++ helper class to read the culture knowledge base file and query the knowledge through access methods.
 * The helper class uses a std::map dictionary to represent the knowledge base (simplified from binary search tree).
 * It provides two public methods to access the knowledge:
 * 
 *    getValue()
 *    printToScreen()
 *
 * The knowledge base is built automatically by the constructor when the CultureKnowledgeBase class is instantiated as an object, 
 * reading the key-value types and the key-value pairs from the files specified in the configuration file.
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
 * knowledge_base   | cultureKnowledgeBaseInput.dat
 * value_types      | cultureKnowledgeValueTypesInput.dat
 * verbose_mode     | false
 *
 * Author:  David Vernon, Carnegie Mellon University Africa (Original)
 * Modified for ROS2 and simplified data structures
 * Date:    July 2025
 * Version: v2.0
 */

#include <utilities/cultureKnowledgeBaseInterface.h>
namespace Culture {

/*
 * CultureKnowledgeBase::CultureKnowledgeBase
 * ------------------------------------------
 *
 * Class constructor
 *   Read the configuration data from the configuration file
 *   Read the knowledge base value types and build the map dictionary data structure
 *   Read the knowledge base values and update the map dictionary data structure 
 */

CultureKnowledgeBase::CultureKnowledgeBase(std::shared_ptr<rclcpp::Node> node) : node_(node) {

   bool debug = false;
   
   if (debug) RCLCPP_INFO(node_->get_logger(), "CultureKnowledgeBase()");

   knowledgeMap_.clear(); // Initialize empty map

   readConfigurationData();
   readKnowledgeBaseValueTypes();
   readKnowledgeBase();
}

/*
 * CultureKnowledgeBase::~CultureKnowledgeBase
 * -------------------------------------------
 *
 * Class destructor
 *    Clean up the map dictionary data structure
 */

CultureKnowledgeBase::~CultureKnowledgeBase() {

   if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), "~CultureKnowledgeBase()");

   // Clean up allocated memory for alphanumeric values
   for (auto& pair : knowledgeMap_) {
       if (pair.second.valueType != NUMBER && pair.second.alphanumericValue != nullptr) {
           free(pair.second.alphanumericValue);
       }
   }
   knowledgeMap_.clear();
}

/*
 * CultureKnowledgeBase::assign_key_attributes(KeyValueType *keyValue, const std::string& key, int valueType, bool initialized)
 * ----------------------------------------------------------------------------------------------------------------------------
 *
 * Assign a key, value, value type, and initialized flag to a key-value pair 
 * and set the initialized flag to false because the value has not yet been assigned 
 */

void CultureKnowledgeBase::assign_key_attributes(KeyValueType *keyValue, const std::string& key, int valueType, bool initialized) {
   keyValue->key = key;
   keyValue->valueType = valueType;
   keyValue->initialized = false;
}

/*
 * void CultureKnowledgeBase::assign_key_value(KeyValueType *keyValue, int integerValue, bool initialized)
 * --------------------------------------------------------------------------------------------------------
 *
 * Assign an integer value to a key-value pair, and set the initialized value to true 
 */

void CultureKnowledgeBase::assign_key_value(KeyValueType *keyValue, int integerValue, bool initialized) {
   keyValue->integerValue = integerValue;
   keyValue->initialized = true;
}

/*
 * void CultureKnowledgeBase::assign_key_value(KeyValueType *keyValue, const std::string& alphanumericValue, bool initialized)
 * --------------------------------------------------------------------------------------------------------------------------
 *
 * Assign a string value to a key-value pair, and set the initialized value to true
 */

void CultureKnowledgeBase::assign_key_value(KeyValueType *keyValue, const std::string& alphanumericValue, bool initialized) {
   keyValue->alphanumericValue = (char *) malloc(sizeof(char) * (alphanumericValue.length() + 1));
   strcpy(keyValue->alphanumericValue, alphanumericValue.c_str());
   keyValue->initialized = true;
} 

/*
 * bool CultureKnowledgeBase::exists(const std::string& key)
 * ---------------------------------------------------------
 *
 * Return true if the key is in the map; false otherwise
 */

bool CultureKnowledgeBase::exists(const std::string& key) {
   return knowledgeMap_.find(key) != knowledgeMap_.end();
}

/*
 * bool CultureKnowledgeBase::getValue(const std::string& key, KeyValueType *keyValue)
 * -----------------------------------------------------------------------------------
 *
 * Return true if the key is in the map; false otherwise
 */

bool CultureKnowledgeBase::getValue(const std::string& key, KeyValueType *keyValue) {
   auto it = knowledgeMap_.find(key);
   if (it != knowledgeMap_.end()) {
       *keyValue = it->second;
       return true;
   }
   return false;
}

/*
 * void CultureKnowledgeBase::printToScreen() 
 * ------------------------------------------
 *
 * Print all elements in the knowledge base
 */

void CultureKnowledgeBase::printToScreen() {
   for (const auto& pair : knowledgeMap_) {
       const KeyValueType& kv = pair.second;
       if (kv.valueType == NUMBER) {
           printf("%-40s %-50d %-20s %-20s\n", 
                  kv.key.c_str(), 
                  kv.integerValue, 
                  valueType2Alphanumeric(kv.valueType).c_str(), 
                  initialized2Alphanumeric(kv.initialized).c_str());
       } else {
           printf("%-40s %-50s %-20s %-20s\n", 
                  kv.key.c_str(), 
                  kv.alphanumericValue, 
                  valueType2Alphanumeric(kv.valueType).c_str(), 
                  initialized2Alphanumeric(kv.initialized).c_str());
       }
   }
   printf("\n");
}

/*
 * void CultureKnowledgeBase::readConfigurationData() 
 * --------------------------------------------------
 *
 * Read configuration parameters key-value pairs from the YAML configuration file 
 */

void CultureKnowledgeBase::readConfigurationData() {
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
       std::string value_types_file = config["value_types"].as<std::string>();
       
       // Construct full paths
       configurationData.knowledgeBase = package_share_directory + "/behaviorController/include/utilities/data/" + knowledge_base_file;
       configurationData.valueTypes = package_share_directory + "/behaviorController/include/utilities/data/" + value_types_file;
       
       if (configurationData.verboseMode) {
           RCLCPP_INFO(node_->get_logger(), "readConfigurationData: knowledgeBase %s", configurationData.knowledgeBase.c_str());
           RCLCPP_INFO(node_->get_logger(), "readConfigurationData: valueTypes %s", configurationData.valueTypes.c_str());
           RCLCPP_INFO(node_->get_logger(), "readConfigurationData: verboseMode %d", configurationData.verboseMode);
       }
       
   } catch (const std::exception& e) {
       RCLCPP_ERROR(node_->get_logger(), "Error reading configuration file: %s", e.what());
       throw;
   }
}

/*
 * void CultureKnowledgeBase::readKnowledgeBase()
 * ----------------------------------------------
 *
 * Read knowledge base key-value pairs from the data file 
 */

void CultureKnowledgeBase::readKnowledgeBase() {
   bool debug = false;  
   char input_string[MAX_STRING_LENGTH];
   char alphanumericValue[MAX_STRING_LENGTH];
   int integerValue;
   FILE *fp_in;
   char key[KEY_LENGTH];
   char value[STRING_LENGTH];

   if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: initializing knowledge base values");
   
   // Open the knowledge base file
   if ((fp_in = fopen(configurationData.knowledgeBase.c_str(),"r")) == 0) {  
       RCLCPP_ERROR(node_->get_logger(), "Error can't open knowledge base file %s", configurationData.knowledgeBase.c_str());
       return;
   }

   // Read the key value pairs
   while (fgets(input_string, STRING_LENGTH, fp_in) != NULL) {
       if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), "readKnowledgeBase: %s", input_string);

       // Extract the key and value
       sscanf(input_string, " %s %s", key, value);
       
       std::string keyStr(key);
       auto it = knowledgeMap_.find(keyStr);
       
       if (it != knowledgeMap_.end()) {
           KeyValueType keyValue = it->second;
           
           switch (keyValue.valueType) {
               case UNDEFINED:
                   RCLCPP_WARN(node_->get_logger(), "readKnowledgeBase: attempting to assign a value to a key-value pair with an UNDEFINED value type");
                   break;

               case NUMBER:
                   if (sscanf(value, "%d", &integerValue) == 1) {
                       assign_key_value(&keyValue, integerValue, true);
                   } else {
                       RCLCPP_ERROR(node_->get_logger(), "readKnowledgeBase: unsuccessful attempt to assign an integer value to a key-value pair with an NUMERIC value type");
                   }
                   break;

               case WORD:
                   if (sscanf(value, "%s", alphanumericValue) == 1) {
                       assign_key_value(&keyValue, std::string(alphanumericValue), true);
                   } else {
                       RCLCPP_ERROR(node_->get_logger(), "readKnowledgeBase: unsuccessful attempt to assign a string value to a key-value pair with an WORD value type");
                   }
                   break;

               case PHRASE:
                   {
                       // Find start of phrase after key and whitespace
                       size_t i;
                       for (i = strlen(key); i < strlen(input_string) && !isalnum(input_string[i]); i++) {}
                       strcpy(alphanumericValue, input_string + i);
                       alphanumericValue[strlen(alphanumericValue)-1] = '\0'; // remove newline
                       assign_key_value(&keyValue, std::string(alphanumericValue), true);
                   }
                   break;

               default:
                   RCLCPP_ERROR(node_->get_logger(), "readKnowledgeBase: invalid value type %d", keyValue.valueType);
                   break;
           }
           
           // Update the map with the modified keyValue
           knowledgeMap_[keyStr] = keyValue;
       } else {
           RCLCPP_ERROR(node_->get_logger(), "readKnowledgeBase: unsuccessful attempt to get the value of this key %s", key);
       }
   }

   fclose(fp_in);
}

/*
 * void CultureKnowledgeBase::readKnowledgeBaseValueTypes()
 * --------------------------------------------------------
 *
 * Read the types of the values in the knowledge base key-value pairs from the data file 
 */

void CultureKnowledgeBase::readKnowledgeBaseValueTypes() {
   bool debug = false;  
   char input_string[MAX_STRING_LENGTH];
   FILE *fp_in;
   char key[KEY_LENGTH];
   char valueType[KEY_LENGTH];

   std::vector<std::string> valueTypeList = {"UNDEFINED", "NUMBER", "WORD", "PHRASE"};

   if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), "readKnowledgeBaseValueTypes: initializing knowledge base value types");

   // Open the knowledge base value types file
   if ((fp_in = fopen(configurationData.valueTypes.c_str(),"r")) == 0) {  
       RCLCPP_ERROR(node_->get_logger(), "Error can't open knowledge base file %s", configurationData.valueTypes.c_str());
       return;
   }

   // Read the key value type pairs
   while (fgets(input_string, STRING_LENGTH, fp_in) != NULL) {
       if (configurationData.verboseMode) RCLCPP_INFO(node_->get_logger(), "readKnowledgeBaseValueTypes: %s", input_string);
       
       // Extract the key and value type
       sscanf(input_string, " %s %s", key, valueType);

       KeyValueType keyValue;
       std::string valueTypeStr(valueType);
       
       for (size_t j = 0; j < valueTypeList.size(); j++) {
           if (valueTypeStr == valueTypeList[j]) {
               switch (j) {
                   case 0: assign_key_attributes(&keyValue, std::string(key), UNDEFINED, false); break;
                   case 1: assign_key_attributes(&keyValue, std::string(key), NUMBER, false); break;
                   case 2: assign_key_attributes(&keyValue, std::string(key), WORD, false); break;
                   case 3: assign_key_attributes(&keyValue, std::string(key), PHRASE, false); break;
                   default: RCLCPP_ERROR(node_->get_logger(), "readKnowledgeBaseValueTypes: invalid value type"); break;
               }
               break;
           }
       }

       // Insert into map (no duplicates allowed)
       std::string keyStr(key);
       if (knowledgeMap_.find(keyStr) == knowledgeMap_.end()) {
           knowledgeMap_[keyStr] = keyValue;
       }
   }

   fclose(fp_in);
}

/*
 * int CultureKnowledgeBase::size() 
 * --------------------------------
 *
 * Return the size of the knowledge base, i.e. the total number of entries
 */

int CultureKnowledgeBase::size() {
   return knowledgeMap_.size();
}

/*
 * Utility functions 
 * ================
 */

/*
 * std::string initialized2Alphanumeric(bool initialized)
 * ------------------------------------------------------
 * 
 * Convert Boolean initialized to alphanumeric
 */

std::string initialized2Alphanumeric(bool initialized) {
   if (initialized) return "Initialized Value";
   else return "Uninitialized Value";
}

/* 
 * std::string valueType2Alphanumeric(int valueType) 
 * -------------------------------------------------
 *
 * Convert integer valueType to alphanumeric 
 */

std::string valueType2Alphanumeric(int valueType) {
   std::vector<std::string> valueTypeList = {"UNDEFINED", "NUMBER", "WORD", "PHRASE"};

   if (valueType >= 0 && valueType < (int)valueTypeList.size()) {
       return valueTypeList[valueType]; 
   } else {
       printf("valueType2Alphanumeric: invalid value type\n");
       return "INVALID";
   }
}

}
