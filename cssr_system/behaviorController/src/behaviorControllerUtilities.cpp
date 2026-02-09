/* behaviorControllerUtilities.cpp - Utility functions and helper classes
 *
 * Author: Yohannes Tadesse Haile
 * Date: Feb 06, 2026
 * Version: v1.0 - Updated for BehaviorTree.ROS2
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#include "behaviorController/behaviorControllerInterface.h"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <regex>

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
            language_   = config["language"].as<std::string>("English");
            nodeName_   = config["node_name"].as<std::string>("behaviorController");
            cultureKnowledgeBasePath_ = config["culture_knowledge_base"].as<std::string>("cultureKnowledgeBase.yaml");
            environmentKnowledgeBasePath_ = config["environment_knowledge_base"].as<std::string>("labEnvironmentKnowledgeBase.yaml");
            return true;
        } catch (const std::exception& e) {
            std::cerr << "ConfigManager: Failed to load config: " << e.what() << std::endl;
            return false;
        }
    }

    // Simple, inline getters
    bool        isVerbose()    const { return verbose_;    }
    bool        isAsrEnabled() const { return asrEnabled_; }
    std::string getLanguage()  const { return language_;   }
    std::string getNodeName()  const { return nodeName_;   }
    std::string getCultureKnowledgeBasePath() const { return cultureKnowledgeBasePath_; }
    std::string getEnvironmentKnowledgeBasePath() const { return environmentKnowledgeBasePath_; }

private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&)            = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    // Storage for settings
    bool        verbose_    = false;
    bool        asrEnabled_ = false;
    std::string language_   = "English";
    std::string nodeName_   = "behaviorController";
    std::string cultureKnowledgeBasePath_ = "cultureKnowledgeBase.yaml";
    std::string environmentKnowledgeBasePath_ = "labEnvironmentKnowledgeBase.yaml";
};

//=============================================================================

KnowledgeManager& KnowledgeManager::instance() {
    static KnowledgeManager instance;
    return instance;
}

bool KnowledgeManager::loadFromPackage(const std::string& packagePath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        // Load culture knowledge base from configuration
        std::string cultureFilePath = packagePath + "/data/" + ConfigManager::instance().getCultureKnowledgeBasePath();
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

        // Load environment knowledge base from configuration
        std::string envFilePath = packagePath + "/data/" + ConfigManager::instance().getEnvironmentKnowledgeBasePath();
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
        std::cout << "  - Locations loaded: " << locations_.size() << std::endl;
        std::cout << "  - Utility phrases loaded: " << utilityPhrases_.size() << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception loading knowledge base: " << e.what() << std::endl;
        return false;
    }
}

std::string KnowledgeManager::getUtilityPhrase(const std::string& phraseId, const std::string& language) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        throw std::runtime_error("Knowledge base not loaded");
    }
    
    std::string lang = language.empty() ? ConfigManager::instance().getLanguage() : language;
    std::string languageKey = (lang == "English") ? "english" : 
                             (lang == "Kinyarwanda") ? "kinyarwanda" : "english";
    
    std::string key = languageKey + ":" + phraseId;
    auto it = utilityPhrases_.find(key);
    if (it != utilityPhrases_.end()) {
        return it->second;
    }
    
    // Try fallback to English
    std::string fallbackKey = "english:" + phraseId;
    it = utilityPhrases_.find(fallbackKey);
    if (it != utilityPhrases_.end()) {
        std::cerr << "Warning: Using English fallback for phrase: " << phraseId << std::endl;
        return it->second;
    }
    
    throw std::runtime_error("Utility phrase not found: " + phraseId + " (language: " + lang + ")");
}

LocationInfo KnowledgeManager::getLocationInfo(const std::string& locationId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        throw std::runtime_error("Knowledge base not loaded");
    }
    
    auto it = locations_.find(locationId);
    if (it != locations_.end()) {
        return it->second;
    }
    throw std::runtime_error("Location not found: " + locationId);
}

TourSpec KnowledgeManager::getTourSpecification() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        throw std::runtime_error("Knowledge base not loaded");
    }
    
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

void Logger::debug(const std::string& msg) {
    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_DEBUG(node_->get_logger(), "%s", formatMessage(msg).c_str());
    }
}

//=============================================================================

ServiceManager::ServiceManager(std::shared_ptr<rclcpp::Node> node) : node_(node) {}

bool ServiceManager::checkServicesAvailable(const std::vector<std::string>& services) {
    bool allAvailable = true;
    auto serviceNamesAndTypes = node_->get_service_names_and_types();
    
    for (const auto& serviceName : services) {
        bool found = false;
        for (const auto& serviceInfo : serviceNamesAndTypes) {
            if (serviceInfo.first == serviceName) {
                found = true;
                RCLCPP_DEBUG(node_->get_logger(), "Service found: %s", serviceName.c_str());
                break;
            }
        }
        if (!found) {
            RCLCPP_WARN(node_->get_logger(), "Service not found: %s", serviceName.c_str());
            allAvailable = false;
        }
    }
    return allAvailable;
}

bool ServiceManager::waitForService(const std::string& serviceName, 
                                    std::chrono::seconds timeout) {
    auto start = std::chrono::steady_clock::now();
    
    while (rclcpp::ok()) {
        auto serviceNamesAndTypes = node_->get_service_names_and_types();
        for (const auto& serviceInfo : serviceNamesAndTypes) {
            if (serviceInfo.first == serviceName) {
                return true;
            }
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return false;
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
            RCLCPP_WARN(node_->get_logger(), "Topic not found: %s", topicName.c_str());
            allAvailable = false;
        } else {
            RCLCPP_DEBUG(node_->get_logger(), "Topic found: %s", topicName.c_str());
        }
    }
    return allAvailable;
}

bool TopicMonitor::waitForTopic(const std::string& topicName,
                                std::chrono::seconds timeout) {
    auto start = std::chrono::steady_clock::now();
    
    while (rclcpp::ok()) {
        if (isTopicAvailable(topicName)) {
            return true;
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return false;
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

std::vector<std::string> TextUtils::split(const std::string& text, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(text);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

std::string TextUtils::trim(const std::string& text) {
    auto start = text.begin();
    while (start != text.end() && std::isspace(*start)) {
        start++;
    }
    
    auto end = text.end();
    do {
        end--;
    } while (std::distance(start, end) > 0 && std::isspace(*end));
    
    return std::string(start, end + 1);
}

//=============================================================================
// Standalone Utility Functions
//=============================================================================

std::string getConfigValue(const std::string& key) {
    try {
        std::string packagePath = ament_index_cpp::get_package_share_directory("cssr_system");
        std::string configPath = packagePath + "/behaviorController/config/behaviorControllerConfiguration.yaml";
        YAML::Node config = YAML::LoadFile(configPath);
        
        if (config[key]) {
            return config[key].as<std::string>();
        }
        
        throw std::runtime_error("Key not found in configuration: " + key);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML parsing error: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error reading config: " + std::string(e.what()));
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
            "language"
        };
        
        for (const auto& field : requiredFields) {
            if (!config[field]) {
                RCLCPP_ERROR(rclcpp::get_logger("behaviorController"), 
                           "Missing required configuration field: %s", field.c_str());
                return false;
            }
        }
        
        // Validate language
        std::string language = config["language"].as<std::string>();
        if (!isValidLanguage(language)) {
            RCLCPP_ERROR(rclcpp::get_logger("behaviorController"),
                        "Invalid language: %s", language.c_str());
            return false;
        }
        
        return true;
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("behaviorController"), 
                   "Configuration file validation failed: %s", e.what());
        return false;
    }
}

void logSystemInfo(std::shared_ptr<rclcpp::Node> node) {
    auto logger = node->get_logger();
    
    // Log ROS2 distribution
    const char* rosDistro = std::getenv("ROS_DISTRO");
    if (rosDistro) {
        RCLCPP_INFO(logger, "ROS2 Distribution: %s", rosDistro);
    }
    
    // Log node name
    RCLCPP_INFO(logger, "Node name: %s", node->get_name());
    
    // Log available services count
    auto services = node->get_service_names_and_types();
    RCLCPP_INFO(logger, "Available services: %zu", services.size());
    
    // Log available topics count
    auto topics = node->get_topic_names_and_types();
    RCLCPP_INFO(logger, "Available topics: %zu", topics.size());
    
    // Log configuration
    auto& config = ConfigManager::instance();
    RCLCPP_INFO(logger, "Configuration:");
    RCLCPP_INFO(logger, "  - Verbose mode: %s", config.isVerbose() ? "ON" : "OFF");
    RCLCPP_INFO(logger, "  - ASR enabled: %s", config.isAsrEnabled() ? "YES" : "NO");
    RCLCPP_INFO(logger, "  - Language: %s", config.getLanguage().c_str());
}

bool isValidLanguage(const std::string& language) {
    static const std::vector<std::string> supportedLanguages = {
        "English", 
        "Kinyarwanda", 
        "IsiZulu"
    };
    return std::find(supportedLanguages.begin(), supportedLanguages.end(), language) 
           != supportedLanguages.end();
}

std::vector<std::string> getSupportedLanguages() {
    return {"English", "Kinyarwanda", "IsiZulu"};
}

bool fileExists(const std::string& filepath) {
    std::ifstream file(filepath);
    return file.good();
}

std::string getPackageDataPath(const std::string& relativePath) {
    std::string packagePath = ament_index_cpp::get_package_share_directory("cssr_system");
    return packagePath + "/" + relativePath;
}

void printNodeInfo(std::shared_ptr<rclcpp::Node> node) {
    auto logger = node->get_logger();
    
    RCLCPP_INFO(logger, "========================================");
    RCLCPP_INFO(logger, "Node Information");
    RCLCPP_INFO(logger, "========================================");
    RCLCPP_INFO(logger, "Name: %s", node->get_name());
    RCLCPP_INFO(logger, "Namespace: %s", node->get_namespace());
    RCLCPP_INFO(logger, "========================================");
}

// Helper function to safely convert NodeStatus to string
std::string nodeStatusToString(BT::NodeStatus status) {
    switch (status) {
        case BT::NodeStatus::SUCCESS:
            return "SUCCESS";
        case BT::NodeStatus::FAILURE:
            return "FAILURE";
        case BT::NodeStatus::RUNNING:
            return "RUNNING";
        case BT::NodeStatus::IDLE:
            return "IDLE";
        default:
            return "UNKNOWN";
    }
}