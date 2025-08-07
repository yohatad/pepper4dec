/* behaviorControllerUtilities.cpp - Utility functions and helper classes
 *
 * Author: Yohannes Tadesse Haile
 * Date: July 25, 2025
 * Version: v1.0
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#include "behaviorController/behaviorControllerInterface.h"

//=============================================================================
// Singleton Implementations
//=============================================================================

class ConfigManager {
public:
    // Singleton accessor
    static ConfigManager& instance() {
    static ConfigManager inst;
    return inst;
  }

  // Call once (e.g. in main()) before spawning threads
  bool loadFromFile(const std::string& configPath) {
    try {
      auto config = YAML::LoadFile(configPath);
      verbose_    = config["verbose_mode"].as<bool>(false);
      asrEnabled_ = config["asr_enabled"].as<bool>(false);
      language_   = config["language"].as<std::string>("");
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }

  // Simple, inline getters
  bool        isVerbose()    const { return verbose_;    }
  bool        isAsrEnabled() const { return asrEnabled_; }
  std::string getLanguage()  const { return language_;   }

private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&)            = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

  // Storage for settings
  bool        verbose_    = false;
  bool        asrEnabled_ = false;
  std::string language_;
  std::string nodeName_;
};

//=============================================================================

KnowledgeManager& KnowledgeManager::instance() {
    static KnowledgeManager instance;
    return instance;
}

bool KnowledgeManager::loadFromPackage(const std::string& packagePath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        // Load culture knowledge base
        std::string cultureFilePath = packagePath + "/behaviorController/data/cultureKnowledgeBase.yaml";
        std::cout << "Loading culture knowledge base from: " << cultureFilePath << std::endl;
        YAML::Node cultureConfig = YAML::LoadFile(cultureFilePath);
        if (cultureConfig["utility_phrases"]) {
            for (const auto& langNode : cultureConfig["utility_phrases"]) {
                std::string lang = langNode.first.as<std::string>();
                for (const auto& phraseNode : langNode.second) {
                    std::string phraseId = phraseNode.first.as<std::string>();
                    std::string phrase = phraseNode.second.as<std::string>();
                    utilityPhrases_[lang + ":" + phraseId] = phrase;
                }
            }
        }

        // Load environment knowledge base
        std::string envFilePath = packagePath + "/behaviorController/data/environmentKnowledgeBase.yaml";
        std::cout << "Loading environment knowledge base from: " << envFilePath << std::endl;
        YAML::Node envConfig = YAML::LoadFile(envFilePath);
        if (envConfig["locations"]) {
            for (const auto& locationNode : envConfig["locations"]) {
                std::string locationId = locationNode.first.as<std::string>();
                LocationInfo info;
                const auto& location = locationNode.second;
                
                info.description = location["robot_location_description"].as<std::string>();
                
                // Robot pose
                info.robotPose.x = location["robot_location_pose"]["x"].as<double>();
                info.robotPose.y = location["robot_location_pose"]["y"].as<double>();
                info.robotPose.theta = location["robot_location_pose"]["theta"].as<double>();
                
                // Gesture target
                info.gestureTarget.x = location["gesture_target"]["x"].as<double>();
                info.gestureTarget.y = location["gesture_target"]["y"].as<double>();
                info.gestureTarget.z = location["gesture_target"]["z"].as<double>();
                
                // Messages by language
                info.preMessages["English"] = location["pre_gesture_message_english"].as<std::string>();
                info.preMessages["Kinyarwanda"] = location["pre_gesture_message_kinyarwanda"].as<std::string>();
                
                info.postMessages["English"] = location["post_gesture_message_english"].as<std::string>();
                info.postMessages["Kinyarwanda"] = location["post_gesture_message_kinyarwanda"].as<std::string>();
                
                
                locations_[locationId] = info;
            }
        }

        // Load tour specification
        if (envConfig["tour_specification"]) {
            TourSpec tour;
            for (const auto& locationId : envConfig["tour_specification"]) {
                tour.locationIds.push_back(locationId.as<std::string>());
            }
            tourSpec_ = tour;
        }

        loaded_ = true;
        std::cout << "Knowledge base loaded successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception loading knowledge base: " << e.what() << std::endl;
        return false;
    }
}

std::string KnowledgeManager::getUtilityPhrase(const std::string& phraseId, const std::string& language) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) throw std::runtime_error("Knowledge base not loaded");
    
    std::string lang = language.empty() ? ConfigManager::instance().getLanguage() : language;
    std::string languageKey = (lang == "English") ? "english" : 
                             (lang == "Kinyarwanda") ? "kinyarwanda" : "english";
    
    std::string key = languageKey + ":" + phraseId;
    auto it = utilityPhrases_.find(key);
    if (it != utilityPhrases_.end()) {
        return it->second;
    }
    throw std::runtime_error("Utility phrase not found: " + phraseId);
}

LocationInfo KnowledgeManager::getLocationInfo(const std::string& locationId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) throw std::runtime_error("Knowledge base not loaded");
    
    auto it = locations_.find(locationId);
    if (it != locations_.end()) {
        return it->second;
    }
    throw std::runtime_error("Location not found: " + locationId);
}

TourSpec KnowledgeManager::getTourSpecification() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) throw std::runtime_error("Knowledge base not loaded");
    
    if (tourSpec_) {
        return *tourSpec_;
    }
    throw std::runtime_error("Tour specification not found");
}

//=============================================================================
// Utility Classes Implementation
//=============================================================================

Logger::Logger(std::shared_ptr<rclcpp::Node> node) : node_(node) {}

std::string Logger::formatMessage(const std::string& msg) {
    return "[" + ConfigManager::instance().getNodeName() + "]: " + msg;
}

void Logger::info(const std::string& msg) {
    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(node_->get_logger(), "%s", formatMessage(msg).c_str());
    }
}

void Logger::warn(const std::string& msg) {
    RCLCPP_WARN(node_->get_logger(), "%s", formatMessage(msg).c_str());
}

void Logger::error(const std::string& msg) {
    RCLCPP_ERROR(node_->get_logger(), "%s", formatMessage(msg).c_str());
}

//=============================================================================

BaseTreeNode::BaseTreeNode(const BT::NodeConfiguration &config)
{
    // 1) grab the shared_ptr<rclcpp::Node> that we stashed under "node"
    //    when we built the tree in initializeTree():
    node_ = config.blackboard->get<std::shared_ptr<rclcpp::Node>>("node");
    if (!node_) {
        throw std::runtime_error("BaseTreeNode: no \"node\" on the blackboard");
    }

    // 2) now we can build our logger and serviceManager
    logger_         = std::make_unique<Logger>(node_);
    serviceManager_ = std::make_unique<ServiceManager>(node_);
}

ServiceManager::ServiceManager(std::shared_ptr<rclcpp::Node> node) : node_(node) {}

bool ServiceManager::checkServicesAvailable(const std::vector<std::string>& services) {
    bool allAvailable = true;
    auto serviceNamesAndTypes = node_->get_service_names_and_types();
    
    for (const auto& serviceName : services) {
        bool found = false;
        for (const auto& serviceInfo : serviceNamesAndTypes) {
            if (serviceInfo.first == serviceName) {
                found = true;
                break;
            }
        }
        if (!found) {
            RCLCPP_ERROR(node_->get_logger(), "Service not found: %s", serviceName.c_str());
            allAvailable = false;
        }
    }
    return allAvailable;
}

//=============================================================================

TopicMonitor::TopicMonitor(std::shared_ptr<rclcpp::Node> node) : node_(node) {}

bool TopicMonitor::isTopicAvailable(const std::string& topicName) {
    auto topicNamesAndTypes = node_->get_topic_names_and_types();
    for (const auto& topicInfo : topicNamesAndTypes) {
        if (topicInfo.first == topicName) {
            return true;
        }
    }
    return false;
}

bool TopicMonitor::checkTopicsAvailable(const std::vector<std::string>& topics) {
    bool allAvailable = true;
    for (const auto& topicName : topics) {
        if (!isTopicAvailable(topicName)) {
            RCLCPP_ERROR(node_->get_logger(), "Topic not found: %s", topicName.c_str());
            allAvailable = false;
        }
    }
    return allAvailable;
}

//=============================================================================

bool TextUtils::containsAnyWord(const std::string& text, const std::vector<std::string>& words) {
    std::string lowerText = toLowerCase(text);
    for (const auto& word : words) {
        std::regex wordRegex("\\b" + toLowerCase(word) + "\\b");
        if (std::regex_search(lowerText, wordRegex)) {
            return true;
        }
    }
    return false;
}

std::string TextUtils::toLowerCase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}


//=============================================================================
// Standalone Utility Functions
//=============================================================================

std::string getConfigValue(const std::string& key) {
    try {
        std::string packagePath = ament_index_cpp::get_package_share_directory("cssr_system");
        YAML::Node config = YAML::LoadFile(packagePath + "/behaviorController/config/behaviorControllerConfiguration.yaml");
        
        if (config[key]) {
            return config[key].as<std::string>();
        }
        
        throw std::runtime_error("Key not found in configuration: " + key);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML parsing error: " + std::string(e.what()));
    }
}

bool validateConfigurationFile(const std::string& configPath) {
    try {
        YAML::Node config = YAML::LoadFile(configPath);
        
        // Check required fields
        std::vector<std::string> requiredFields = {
            "scenario_specification",
            "verbose_mode",
            "asr_enabled",
        };
        
        for (const auto& field : requiredFields) {
            if (!config[field]) {
                RCLCPP_ERROR(rclcpp::get_logger("behaviorController"), 
                           "Missing required configuration field: %s", field.c_str());
                return false;
            }
        }
        
        return true;
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("behaviorController"), 
                   "Configuration file validation failed: %s", e.what());
        return false;
    }
}

void logSystemInfo(std::shared_ptr<rclcpp::Node> node) {
    Logger logger(node);
    
    // Log ROS2 distribution
    const char* rosDistro = std::getenv("ROS_DISTRO");
    if (rosDistro) {
        logger.info("ROS2 Distribution: " + std::string(rosDistro));
    }
    
    // Log available services count
    auto services = node->get_service_names_and_types();
    logger.info("Available services: " + std::to_string(services.size()));
    
    // Log available topics count
    auto topics = node->get_topic_names_and_types();
    logger.info("Available topics: " + std::to_string(topics.size()));
}

bool isValidLanguage(const std::string& language) {
    std::vector<std::string> supportedLanguages = {"English", "Kinyarwanda", "IsiZulu"};
    return std::find(supportedLanguages.begin(), supportedLanguages.end(), language) != supportedLanguages.end();
}
