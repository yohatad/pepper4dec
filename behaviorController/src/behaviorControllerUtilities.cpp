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

ConfigManager& ConfigManager::instance() {
    static ConfigManager instance;
    return instance;
}

bool ConfigManager::loadFromFile(const std::string& configPath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        YAML::Node config = YAML::LoadFile(configPath);
        verbose_ = config["verbose_mode"].as<bool>(false);
        asrEnabled_ = config["asr_enabled"].as<bool>(false);
        testMode_ = config["test_mode"].as<bool>(false);
        return true;
    } catch (const std::exception&) {
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

//=============================================================================

KnowledgeManager& KnowledgeManager::instance() {
    static KnowledgeManager instance;
    return instance;
}

bool KnowledgeManager::loadFromPackage(const std::string& packagePath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        // Load culture knowledge base
        YAML::Node cultureConfig = YAML::LoadFile(packagePath + "/config/behaviorControllerConfiguration.yaml");
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
        YAML::Node envConfig = YAML::LoadFile(packagePath + "/data/environmentKnowledgeBase.yaml");
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
                info.preMessages["IsiZulu"] = location["pre_gesture_message_isizulu"].as<std::string>();
                
                info.postMessages["English"] = location["post_gesture_message_english"].as<std::string>();
                info.postMessages["Kinyarwanda"] = location["post_gesture_message_kinyarwanda"].as<std::string>();
                info.postMessages["IsiZulu"] = location["post_gesture_message_isizulu"].as<std::string>();
                
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
        return true;
    } catch (const std::exception&) {
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

void KnowledgeManager::clearCache() {
    std::lock_guard<std::mutex> lock(mutex_);
    utilityPhrases_.clear();
    locations_.clear();
    tourSpec_.reset();
    loaded_ = false;
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

TestManager::TestManager(std::shared_ptr<rclcpp::Node> node) : node_(node) {}

void TestManager::storeResult(const std::string& key, bool success) {
    if (!ConfigManager::instance().isTestMode()) return;
    
    std::string parameterPath = "behaviorControllerTest." + key;
    int value = success ? 1 : 0;
    
    try {
        if (!node_->has_parameter(parameterPath)) {
            node_->declare_parameter(parameterPath, value);
        } else {
            node_->set_parameter(rclcpp::Parameter(parameterPath, value));
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to store test result %s: %s", key.c_str(), e.what());
    }
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

BaseTreeNode::BaseTreeNode(std::shared_ptr<rclcpp::Node> node) 
    : node_(node)
    , logger_(std::make_unique<Logger>(node))
    , serviceManager_(std::make_unique<ServiceManager>(node))
    , testManager_(std::make_unique<TestManager>(node)) {
}

void BaseTreeNode::storeTestResult(const std::string& nodeName, bool success) {
    testManager_->storeResult(nodeName, success);
}

//=============================================================================
// Standalone Utility Functions
//=============================================================================

std::string getConfigValue(const std::string& key) {
    try {
        std::string packagePath = ament_index_cpp::get_package_share_directory("cssr_system");
        YAML::Node config = YAML::LoadFile(packagePath + "/config/behaviorControllerConfiguration.yaml");
        
        if (config[key]) {
            return config[key].as<std::string>();
        }
        
        throw std::runtime_error("Key not found in configuration: " + key);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("YAML parsing error: " + std::string(e.what()));
    }
}

std::string readValueFromFile(const std::string& filename, const std::string& key) {
    std::string packagePath = ament_index_cpp::get_package_share_directory("cssr_system");
    std::string fullPath = packagePath + "/data/" + filename;
    
    std::ifstream file(fullPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + fullPath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string fileKey;
        std::string value;
        
        if (iss >> fileKey) {
            std::getline(iss, value);
            
            // Trim whitespace
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (fileKey == key) {
                file.close();
                return value;
            }
        }
    }
    
    file.close();
    throw std::runtime_error("Key not found in file " + filename + ": " + key);
}

bool validateConfigurationFile(const std::string& configPath) {
    try {
        YAML::Node config = YAML::LoadFile(configPath);
        
        // Check required fields
        std::vector<std::string> requiredFields = {
            "scenario_specification",
            "verbose_mode",
            "asr_enabled",
            "test_mode"
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

bool waitForRosServices(std::shared_ptr<rclcpp::Node> node, 
                       const std::vector<std::string>& serviceNames,
                       std::chrono::seconds timeout) {
    auto startTime = std::chrono::steady_clock::now();
    
    while (std::chrono::steady_clock::now() - startTime < timeout) {
        bool allAvailable = true;
        auto availableServices = node->get_service_names_and_types();
        
        for (const auto& serviceName : serviceNames) {
            bool found = false;
            for (const auto& [name, types] : availableServices) {
                if (name == serviceName) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                allAvailable = false;
                break;
            }
        }
        
        if (allAvailable) {
            return true;
        }
        
        rclcpp::sleep_for(std::chrono::milliseconds(100));
        rclcpp::spin_some(node);
    }
    
    return false;
}

bool waitForRosTopics(std::shared_ptr<rclcpp::Node> node,
                     const std::vector<std::string>& topicNames,
                     std::chrono::seconds timeout) {
    auto startTime = std::chrono::steady_clock::now();
    
    while (std::chrono::steady_clock::now() - startTime < timeout) {
        bool allAvailable = true;
        auto availableTopics = node->get_topic_names_and_types();
        
        for (const auto& topicName : topicNames) {
            bool found = false;
            for (const auto& [name, types] : availableTopics) {
                if (name == topicName) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                allAvailable = false;
                break;
            }
        }
        
        if (allAvailable) {
            return true;
        }
        
        rclcpp::sleep_for(std::chrono::milliseconds(100));
        rclcpp::spin_some(node);
    }
    
    return false;
}

std::string formatDuration(std::chrono::steady_clock::duration duration) {
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration % std::chrono::seconds(1));
    
    std::ostringstream oss;
    oss << seconds.count() << "." << std::setfill('0') << std::setw(3) << milliseconds.count() << "s";
    return oss.str();
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
