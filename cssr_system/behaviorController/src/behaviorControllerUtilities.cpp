/* behaviorControllerUtilities.cpp - Utility functions and helper classes
 *
 * Author: Yohannes Tadesse Haile
 * Date: Feb 09, 2026
 * Version: v2.0 - Updated for BehaviorTree.ROS2 with valid cssr_interfaces
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#include "behaviorController/behaviorControllerInterface.h"
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <regex>
#include <fstream>

//=============================================================================
// ConfigManager - Singleton for configuration management
//=============================================================================

ConfigManager& ConfigManager::instance() {
    static ConfigManager inst;
    return inst;
}

bool ConfigManager::loadFromFile(const std::string& configPath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        auto config = YAML::LoadFile(configPath);
        verbose_    = config["verbose_mode"].as<bool>(false);
        asrEnabled_ = config["asr_enabled"].as<bool>(false);
        testMode_   = config["test_mode"].as<bool>(false);
        language_   = config["language"].as<std::string>("English");
        scenarioSpecification_ = config["scenario_specification"].as<std::string>("lab_tour");
        nodeName_   = config["node_name"].as<std::string>("behaviorController");
        cultureKnowledgeBasePath_ = config["culture_knowledge_base"].as<std::string>("cultureKnowledgeBase.yaml");
        environmentKnowledgeBasePath_ = config["environment_knowledge_base"].as<std::string>("labEnvironmentKnowledgeBase.yaml");
        
        std::cout << "✓ Configuration loaded from: " << configPath << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ ConfigManager: Failed to load config: " << e.what() << std::endl;
        return false;
    }
}

bool ConfigManager::isVerbose() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return verbose_;
}

bool ConfigManager::isAsrEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return asrEnabled_;
}

bool ConfigManager::isTestMode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return testMode_;
}

std::string ConfigManager::getLanguage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return language_;
}

std::string ConfigManager::getNodeName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return nodeName_;
}

std::string ConfigManager::getScenarioSpecification() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return scenarioSpecification_;
}

std::string ConfigManager::getCultureKnowledgeBasePath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cultureKnowledgeBasePath_;
}

std::string ConfigManager::getEnvironmentKnowledgeBasePath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return environmentKnowledgeBasePath_;
}

//=============================================================================
// KnowledgeManager - Singleton for knowledge base management
//=============================================================================

KnowledgeManager& KnowledgeManager::instance() {
    static KnowledgeManager instance;
    return instance;
}

bool KnowledgeManager::loadFromPackage(const std::string& packagePath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        // Load culture knowledge base from configuration
        std::string cultureFilePath = packagePath + "/data/" + 
                                     ConfigManager::instance().getCultureKnowledgeBasePath();
        std::cout << "Loading culture knowledge base from: " << cultureFilePath << std::endl;
        
        if (!fileExists(cultureFilePath)) {
            std::cerr << "✗ Culture knowledge base file not found: " << cultureFilePath << std::endl;
            return false;
        }
        
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
        std::string envFilePath = packagePath + "/data/" + 
                                 ConfigManager::instance().getEnvironmentKnowledgeBasePath();
        std::cout << "Loading environment knowledge base from: " << envFilePath << std::endl;
        
        if (!fileExists(envFilePath)) {
            std::cerr << "✗ Environment knowledge base file not found: " << envFilePath << std::endl;
            return false;
        }
        
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
                if (location["pre_gesture_message_english"]) {
                    info.preMessages["English"] = location["pre_gesture_message_english"].as<std::string>();
                }
                if (location["pre_gesture_message_kinyarwanda"]) {
                    info.preMessages["Kinyarwanda"] = location["pre_gesture_message_kinyarwanda"].as<std::string>();
                }
                
                if (location["post_gesture_message_english"]) {
                    info.postMessages["English"] = location["post_gesture_message_english"].as<std::string>();
                }
                if (location["post_gesture_message_kinyarwanda"]) {
                    info.postMessages["Kinyarwanda"] = location["post_gesture_message_kinyarwanda"].as<std::string>();
                }
                
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
        std::cout << "✓ Knowledge base loaded successfully" << std::endl;
        std::cout << "  - Locations loaded: " << locations_.size() << std::endl;
        std::cout << "  - Tour stops: " << (tourSpec_ ? tourSpec_->getCurrentLocationCount() : 0) << std::endl;
        std::cout << "  - Utility phrases loaded: " << utilityPhrases_.size() << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Exception loading knowledge base: " << e.what() << std::endl;
        return false;
    }
}

std::string KnowledgeManager::getUtilityPhrase(const std::string& phraseId, 
                                               const std::string& language) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        throw std::runtime_error("Knowledge base not loaded");
    }
    
    std::string lang = language.empty() ? ConfigManager::instance().getLanguage() : language;
    
    // Normalize language to lowercase for key lookup
    std::string languageKey = (lang == "English") ? "english" : 
                             (lang == "Kinyarwanda") ? "kinyarwanda" : 
                             (lang == "IsiZulu") ? "isizulu" : "english";
    
    std::string key = languageKey + ":" + phraseId;
    auto it = utilityPhrases_.find(key);
    if (it != utilityPhrases_.end()) {
        return it->second;
    }
    
    // Try fallback to English
    std::string fallbackKey = "english:" + phraseId;
    it = utilityPhrases_.find(fallbackKey);
    if (it != utilityPhrases_.end()) {
        std::cerr << "⚠ Warning: Using English fallback for phrase: " << phraseId 
                  << " (requested language: " << lang << ")" << std::endl;
        return it->second;
    }
    
    throw std::runtime_error("Utility phrase not found: " + phraseId + 
                           " (language: " + lang + ")");
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
// Logger - Logging utility with verbose mode support
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
// ServiceManager - Service availability checking and waiting
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
                RCLCPP_DEBUG(node_->get_logger(), "✓ Service found: %s", serviceName.c_str());
                break;
            }
        }
        if (!found) {
            RCLCPP_WARN(node_->get_logger(), "✗ Service not found: %s", serviceName.c_str());
            allAvailable = false;
        }
    }
    return allAvailable;
}

bool ServiceManager::waitForService(const std::string& serviceName, 
                                    std::chrono::seconds timeout) {
    auto start = std::chrono::steady_clock::now();
    
    RCLCPP_INFO(node_->get_logger(), "Waiting for service: %s", serviceName.c_str());
    
    while (rclcpp::ok()) {
        auto serviceNamesAndTypes = node_->get_service_names_and_types();
        for (const auto& serviceInfo : serviceNamesAndTypes) {
            if (serviceInfo.first == serviceName) {
                RCLCPP_INFO(node_->get_logger(), "✓ Service available: %s", serviceName.c_str());
                return true;
            }
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            RCLCPP_ERROR(node_->get_logger(), 
                        "✗ Timeout waiting for service: %s", serviceName.c_str());
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return false;
}

//=============================================================================
// TopicMonitor - Topic availability checking and monitoring
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
            RCLCPP_WARN(node_->get_logger(), "✗ Topic not found: %s", topicName.c_str());
            allAvailable = false;
        } else {
            RCLCPP_DEBUG(node_->get_logger(), "✓ Topic found: %s", topicName.c_str());
        }
    }
    return allAvailable;
}

bool TopicMonitor::waitForTopic(const std::string& topicName,
                                std::chrono::seconds timeout) {
    auto start = std::chrono::steady_clock::now();
    
    RCLCPP_INFO(node_->get_logger(), "Waiting for topic: %s", topicName.c_str());
    
    while (rclcpp::ok()) {
        if (isTopicAvailable(topicName)) {
            RCLCPP_INFO(node_->get_logger(), "✓ Topic available: %s", topicName.c_str());
            return true;
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            RCLCPP_ERROR(node_->get_logger(), 
                        "✗ Timeout waiting for topic: %s", topicName.c_str());
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return false;
}

//=============================================================================
// TextUtils - Text processing utilities
//=============================================================================

bool TextUtils::containsAnyWord(const std::string& text, 
                               const std::vector<std::string>& words) {
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
        std::string packagePath = ament_index_cpp::get_package_share_directory("behavior_controller");
        std::string configPath = packagePath + "/config/behaviorControllerConfiguration.yaml";
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
        if (!fileExists(configPath)) {
            RCLCPP_ERROR(rclcpp::get_logger("behaviorController"), 
                        "Configuration file not found: %s", configPath.c_str());
            return false;
        }
        
        YAML::Node config = YAML::LoadFile(configPath);
        
        // Check required fields
        std::vector<std::string> requiredFields = {
            "scenario_specification",
            "verbose_mode",
            "asr_enabled",
            "language",
            "node_name",
            "culture_knowledge_base",
            "environment_knowledge_base"
        };
        
        for (const auto& field : requiredFields) {
            if (!config[field]) {
                RCLCPP_ERROR(rclcpp::get_logger("behaviorController"), 
                           "✗ Missing required configuration field: %s", field.c_str());
                return false;
            }
        }
        
        // Validate language
        std::string language = config["language"].as<std::string>();
        if (!isValidLanguage(language)) {
            RCLCPP_ERROR(rclcpp::get_logger("behaviorController"),
                        "✗ Invalid language: %s", language.c_str());
            return false;
        }
        
        RCLCPP_INFO(rclcpp::get_logger("behaviorController"), 
                   "✓ Configuration file validated successfully");
        return true;
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("behaviorController"), 
                   "✗ Configuration file validation failed: %s", e.what());
        return false;
    }
}

void logSystemInfo(std::shared_ptr<rclcpp::Node> node) {
    auto logger = node->get_logger();
    
    RCLCPP_INFO(logger, "=== System Information ===");
    
    // Log available services count
    auto services = node->get_service_names_and_types();
    RCLCPP_INFO(logger, "Available services: %zu", services.size());
    
    // Log available topics count
    auto topics = node->get_topic_names_and_types();
    RCLCPP_INFO(logger, "Available topics: %zu", topics.size());
    
    // Log configuration
    auto& config = ConfigManager::instance();
    RCLCPP_INFO(logger, "Active Configuration:");
    RCLCPP_INFO(logger, "  - Verbose mode: %s", config.isVerbose() ? "ON" : "OFF");
    RCLCPP_INFO(logger, "  - ASR enabled: %s", config.isAsrEnabled() ? "YES" : "NO");
    RCLCPP_INFO(logger, "  - Language: %s", config.getLanguage().c_str());
    RCLCPP_INFO(logger, "  - Scenario: %s", config.getScenarioSpecification().c_str());
    
    RCLCPP_INFO(logger, "=========================");
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
    std::string packagePath = ament_index_cpp::get_package_share_directory("behavior_controller");
    return packagePath + "/data/" + relativePath;
}

void printNodeInfo(std::shared_ptr<rclcpp::Node> node) {
    auto logger = node->get_logger();
    
    RCLCPP_INFO(logger, "========================================");
    RCLCPP_INFO(logger, "Node Information");
    RCLCPP_INFO(logger, "========================================");
    RCLCPP_INFO(logger, "Name: %s", node->get_name());
    RCLCPP_INFO(logger, "Namespace: %s", node->get_namespace());
    RCLCPP_INFO(logger, "Fully Qualified Name: %s", node->get_fully_qualified_name());
    RCLCPP_INFO(logger, "========================================");
}

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

// void logActionServerStatus(std::shared_ptr<rclcpp::Node> node, 
//                            const std::vector<std::string>& actionNames) {
//     auto logger = node->get_logger();
    
//     RCLCPP_INFO(logger, "=== Action Server Status ===");
    
//     // List expected action servers
//     std::vector<std::string> expectedActions = {
//         "/tts",
//         "/navigation",
//         "/gesture",
//         "/speech_recognition",
//         "/animate_behavior"
//     };
    
//     for (const auto& actionName : expectedActions) {
//         RCLCPP_INFO(logger, "  - %s: [checking...]", actionName.c_str());
//     }
    
//     RCLCPP_INFO(logger, "============================");
// }

void logServiceStatus(std::shared_ptr<rclcpp::Node> node) {
    auto logger = node->get_logger();
    
    RCLCPP_INFO(logger, "=== Service Status ===");
    
    // List expected services
    std::vector<std::string> expectedServices = {
        "/overtAttention/set_mode",
        "/animateBehaviour/setActivation",
        "/conversation/prompt"
    };
    
    ServiceManager serviceMgr(node);
    for (const auto& serviceName : expectedServices) {
        auto serviceNamesAndTypes = node->get_service_names_and_types();
        bool found = false;
        for (const auto& serviceInfo : serviceNamesAndTypes) {
            if (serviceInfo.first == serviceName) {
                found = true;
                break;
            }
        }
        RCLCPP_INFO(logger, "  - %s: %s", 
                   serviceName.c_str(), 
                   found ? "✓ Available" : "✗ Not Found");
    }
    
    RCLCPP_INFO(logger, "======================");
}