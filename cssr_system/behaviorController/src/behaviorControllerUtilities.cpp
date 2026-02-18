/* behaviorControllerUtilities.cpp - Utility functions and helper classes
 *
 * Author: Yohannes Tadesse Haile
 * Date: Feb 09, 2026
 * Version: v1.0 - Updated for BehaviorTree.ROS2 with valid cssr_interfaces
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#include "behaviorController/behaviorControllerInterface.h"

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <regex>
#include <filesystem>

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

//=============================================================================
// ConfigManager - Singleton for configuration management
//=============================================================================

ConfigManager& ConfigManager::instance() {
    static ConfigManager inst;
    return inst;
}

bool ConfigManager::loadFromFile(const std::string& configPath) {
    try {
        auto config = YAML::LoadFile(configPath);
        scenarioSpecification        = config["scenario_specification"].as<std::string>("lab_tour");
        cultureKnowledgeBasePath     = config["culture_knowledge_base"].as<std::string>("cultureKnowledgeBase.yaml");
        environmentKnowledgeBasePath = config["environment_knowledge_base"].as<std::string>("labEnvironmentKnowledgeBase.yaml");
        language                     = config["language"].as<std::string>("English");
        verbose                      = config["verbose_mode"].as<bool>(false);

        if (verbose) {
            RCLCPP_INFO(rclcpp::get_logger("behavior_controller"),
                        "Configuration loaded from: %s", configPath.c_str());
        }
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("behavior_controller"),
                     "Failed to load config: %s", e.what());
        return false;
    }
}

std::string ConfigManager::getScenarioSpecification()           const { return scenarioSpecification; }
bool ConfigManager::isVerbose()                                 const { return verbose; }
std::string ConfigManager::getLanguage()                        const { return language; }
std::string ConfigManager::getCultureKnowledgeBasePath()        const { return cultureKnowledgeBasePath; }
std::string ConfigManager::getEnvironmentKnowledgeBasePath()    const { return environmentKnowledgeBasePath; }

//=============================================================================
// KnowledgeManager - Singleton for knowledge base management
//=============================================================================

KnowledgeManager& KnowledgeManager::instance() {
    static KnowledgeManager inst;
    return inst;
}

bool KnowledgeManager::loadFromPackage(const std::string& packagePath) {
    std::string cultureFilePath = packagePath + "/data/" +
                                  ConfigManager::instance().getCultureKnowledgeBasePath();
    std::string envFilePath = packagePath + "/data/" +
                              ConfigManager::instance().getEnvironmentKnowledgeBasePath();
    auto logger = rclcpp::get_logger("behavior_controller");
    
    try {
        if (ConfigManager::instance().isVerbose()) {
            RCLCPP_INFO(logger, "Loading culture knowledge base from: %s", cultureFilePath.c_str());
        }
        if (!behavior_controller::fileExists(cultureFilePath)) {
            RCLCPP_ERROR(logger, "Culture knowledge base file not found: %s", cultureFilePath.c_str());
            return false;
        }
        YAML::Node cultureConfig = YAML::LoadFile(cultureFilePath);
        if (cultureConfig["utility_phrases"]) {
            for (const auto& langNode : cultureConfig["utility_phrases"]) {
                std::string lang = langNode.first.as<std::string>();
                for (const auto& phraseNode : langNode.second) {
                    std::string phraseId = phraseNode.first.as<std::string>();
                    std::string phrase = phraseNode.second.as<std::string>();
                    std::string langLower = lang;
                    std::transform(langLower.begin(), langLower.end(), langLower.begin(), ::tolower);
                    utilityPhrases[langLower + ":" + phraseId] = phrase;
                }
            }
        }
        
        if (ConfigManager::instance().isVerbose()) {
            RCLCPP_INFO(logger, "Loading environment knowledge base from: %s", envFilePath.c_str());
        }
        if (!behavior_controller::validateEnvironmentKnowledgeBase(envFilePath)) {
            RCLCPP_ERROR(logger, "Environment knowledge base validation failed: %s", envFilePath.c_str());
            return false;
        }

        YAML::Node envConfig = YAML::LoadFile(envFilePath);

        // locations is required (guaranteed present by validator)
        for (const auto& locationNode : envConfig["locations"]) {
            std::string locationId = locationNode.first.as<std::string>();
            LocationInfo info;
            const auto& location = locationNode.second;

            info.description = location["robot_location_description"].as<std::string>();

            // Robot pose
            info.robotPose.x     = location["robot_location_pose"]["x"].as<double>();
            info.robotPose.y     = location["robot_location_pose"]["y"].as<double>();
            info.robotPose.theta = location["robot_location_pose"]["theta"].as<double>();

            // Gesture target
            info.gestureTarget.x = location["gesture_target"]["x"].as<double>();
            info.gestureTarget.y = location["gesture_target"]["y"].as<double>();
            info.gestureTarget.z = location["gesture_target"]["z"].as<double>();

            // Messages by language (all four required, guaranteed by validator)
            info.preMessages["English"]      = location["pre_gesture_message_english"].as<std::string>();
            info.preMessages["Kinyarwanda"]  = location["pre_gesture_message_kinyarwanda"].as<std::string>();
            info.postMessages["English"]     = location["post_gesture_message_english"].as<std::string>();
            info.postMessages["Kinyarwanda"] = location["post_gesture_message_kinyarwanda"].as<std::string>();

            locations[locationId] = info;
        }

        // tour_specification is required (guaranteed present by validator)
        TourSpec tour;
        for (const auto& locationId : envConfig["tour_specification"]) {
            tour.locationIds.push_back(locationId.as<std::string>());
        }
        tourSpec = tour;

        loaded = true;
        if (ConfigManager::instance().isVerbose()) {
            RCLCPP_INFO(logger, "Knowledge base loaded successfully");
            RCLCPP_INFO(logger, "  - Locations loaded: %zu", locations.size());
            RCLCPP_INFO(logger, "  - Tour stops: %zu", tourSpec ? tourSpec->getLocationCount() : 0u);
            RCLCPP_INFO(logger, "  - Utility phrases loaded: %zu", utilityPhrases.size());
        }
        return true;

    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger, "Exception loading knowledge base: %s", e.what());
        return false;
    }
}

std::string KnowledgeManager::getUtilityPhrase(const std::string& phraseId,
                                               const std::string& language) {
    if (!loaded) {
        throw std::runtime_error("Knowledge base not loaded");
    }
    std::string lang = language.empty() ? ConfigManager::instance().getLanguage() : language;

    // Normalize language to lowercase for key lookup
    std::string languageKey = (lang == "English")     ? "english" :
                              (lang == "Kinyarwanda") ? "kinyarwanda" : "english";
    
    std::string key = languageKey + ":" + phraseId;
    auto it = utilityPhrases.find(key);
    if (it != utilityPhrases.end()) {
        return it->second;
    }
    
    // Try fallback to English
    std::string fallbackKey = "english:" + phraseId;
    it = utilityPhrases.find(fallbackKey);
    if (it != utilityPhrases.end()) {
        RCLCPP_WARN(rclcpp::get_logger("behavior_controller"),
                    "Using English fallback for phrase: %s (requested language: %s)",
                    phraseId.c_str(), lang.c_str());
        return it->second;
    }
    
    throw std::runtime_error("Utility phrase not found: " + phraseId + 
                           " (language: " + lang + ")");
}

LocationInfo KnowledgeManager::getLocationInfo(const std::string& locationId) {
    if (!loaded) {
        throw std::runtime_error("Knowledge base not loaded");
    }
    
    auto it = locations.find(locationId);
    if (it != locations.end()) {
        return it->second;
    }
    throw std::runtime_error("Location not found: " + locationId);
}

TourSpec KnowledgeManager::getTourSpecification() {
    if (!loaded) {
        throw std::runtime_error("Knowledge base not loaded");
    }
    
    if (tourSpec) {
        return *tourSpec;
    }
    throw std::runtime_error("Tour specification not found");
}

//=============================================================================
// Logger - Logging utility with verbose mode support
//=============================================================================

Logger::Logger(std::shared_ptr<rclcpp::Node> node) : node(node) {}

std::string Logger::formatMessage(const std::string& msg) {
    return "[" + std::string(node->get_name()) + "]: " + msg;
}

void Logger::info(const std::string& msg) {
    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(node->get_logger(), "%s", formatMessage(msg).c_str());
    }
}

void Logger::warn(const std::string& msg) {
    RCLCPP_WARN(node->get_logger(), "%s", formatMessage(msg).c_str());
}

void Logger::error(const std::string& msg) {
    RCLCPP_ERROR(node->get_logger(), "%s", formatMessage(msg).c_str());
}

void Logger::debug(const std::string& msg) {
    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_DEBUG(node->get_logger(), "%s", formatMessage(msg).c_str());
    }
}

//=============================================================================
// ServiceManager - Service availability checking and waiting
//=============================================================================

ServiceManager::ServiceManager(std::shared_ptr<rclcpp::Node> node) : node(node) {}

bool ServiceManager::checkServicesAvailable(const std::vector<std::string>& services) {
    bool allAvailable = true;
    auto serviceNamesAndTypes = node->get_service_names_and_types();
    
    for (const auto& serviceName : services) {
        bool found = false;
        for (const auto& serviceInfo : serviceNamesAndTypes) {
            if (serviceInfo.first == serviceName) {
                found = true;
                RCLCPP_DEBUG(node->get_logger(), "✓ Service found: %s", serviceName.c_str());
                break;
            }
        }
        if (!found) {
            RCLCPP_WARN(node->get_logger(), "✗ Service not found: %s", serviceName.c_str());
            allAvailable = false;
        }
    }
    return allAvailable;
}

bool ServiceManager::waitForService(const std::string& serviceName, 
                                    std::chrono::seconds timeout) {
    auto start = std::chrono::steady_clock::now();
    
    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(node->get_logger(), "Waiting for service: %s", serviceName.c_str());
    }

    while (rclcpp::ok()) {
        auto serviceNamesAndTypes = node->get_service_names_and_types();
        for (const auto& serviceInfo : serviceNamesAndTypes) {
            if (serviceInfo.first == serviceName) {
                if (ConfigManager::instance().isVerbose()) {
                    RCLCPP_INFO(node->get_logger(), "✓ Service available: %s", serviceName.c_str());
                }
                return true;
            }
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            RCLCPP_ERROR(node->get_logger(), 
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

TopicMonitor::TopicMonitor(std::shared_ptr<rclcpp::Node> node) : node(node) {}

bool TopicMonitor::isTopicAvailable(const std::string& topicName) {
    auto topicNamesAndTypes = node->get_topic_names_and_types();
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
            RCLCPP_WARN(node->get_logger(), "✗ Topic not found: %s", topicName.c_str());
            allAvailable = false;
        } else {
            RCLCPP_DEBUG(node->get_logger(), "✓ Topic found: %s", topicName.c_str());
        }
    }
    return allAvailable;
}

bool TopicMonitor::waitForTopic(const std::string& topicName,
                                std::chrono::seconds timeout) {
    auto start = std::chrono::steady_clock::now();
    
    if (ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(node->get_logger(), "Waiting for topic: %s", topicName.c_str());
    }

    while (rclcpp::ok()) {
        if (isTopicAvailable(topicName)) {
            if (ConfigManager::instance().isVerbose()) {
                RCLCPP_INFO(node->get_logger(), "✓ Topic available: %s", topicName.c_str());
            }
            return true;
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (elapsed >= timeout) {
            RCLCPP_ERROR(node->get_logger(), 
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
    while (start != text.end() && std::isspace(static_cast<unsigned char>(*start))) {
        ++start;
    }
    if (start == text.end()) return "";

    auto end = text.end();
    do {
        --end;
    } while (end != start && std::isspace(static_cast<unsigned char>(*end)));

    return std::string(start, end + 1);
}

//=============================================================================
// Standalone Utility Functions
//=============================================================================
namespace behavior_controller {

void logSystemInfo(std::shared_ptr<rclcpp::Node> node) {
    auto logger = node->get_logger();
    auto& config = ConfigManager::instance();

    if (!config.isVerbose()) {
        return;
    }

    RCLCPP_INFO(logger, "=== System Information ===");

    // Log available services count
    auto services = node->get_service_names_and_types();
    RCLCPP_INFO(logger, "Available services: %zu", services.size());

    // Log available topics count
    auto topics = node->get_topic_names_and_types();
    RCLCPP_INFO(logger, "Available topics: %zu", topics.size());

    // Log configuration
    RCLCPP_INFO(logger, "Active Configuration:");
    RCLCPP_INFO(logger, "  - Verbose mode: %s", config.isVerbose() ? "ON" : "OFF");
    RCLCPP_INFO(logger, "  - Language: %s", config.getLanguage().c_str());
    RCLCPP_INFO(logger, "  - Scenario: %s", config.getScenarioSpecification().c_str());

    RCLCPP_INFO(logger, "=========================");
}

bool isValidLanguage(const std::string& language) {
    static const std::vector<std::string> supportedLanguages = {
        "English",
        "Kinyarwanda"
    };
    return std::find(supportedLanguages.begin(), supportedLanguages.end(), language)
           != supportedLanguages.end();
}

std::vector<std::string> getSupportedLanguages() {
    return {"English", "Kinyarwanda"};
}

bool fileExists(const std::string& filepath) {
    return std::filesystem::exists(filepath);
}

std::string getPackageDataPath(const std::string& relativePath) {
    std::string packagePath = ament_index_cpp::get_package_share_directory("behavior_controller");
    return packagePath + "/data/" + relativePath;
}

void printNodeInfo(std::shared_ptr<rclcpp::Node> node) {
    if (!ConfigManager::instance().isVerbose()) {
        return;
    }

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

//=============================================================================
// Environment Knowledge Base Validator
//=============================================================================

bool validateEnvironmentKnowledgeBase(const std::string& filePath) {
    auto logger = rclcpp::get_logger("behavior_controller");
    bool valid = true;

    // --- File existence and YAML parse ---
    if (!fileExists(filePath)) {
        RCLCPP_ERROR(logger, "[KB Validation] File not found: %s", filePath.c_str());
        return false;
    }

    YAML::Node root;
    try {
        root = YAML::LoadFile(filePath);
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(logger, "[KB Validation] YAML parse error in '%s': %s",
                     filePath.c_str(), e.what());
        return false;
    }

    // --- tour_specification: required, non-empty sequence ---
    if (!root["tour_specification"] || !root["tour_specification"].IsSequence()) {
        RCLCPP_ERROR(logger, "[KB Validation] Missing or non-sequence 'tour_specification'");
        return false;
    }
    const auto& tourNode = root["tour_specification"];
    if (tourNode.size() == 0) {
        RCLCPP_ERROR(logger, "[KB Validation] 'tour_specification' must not be empty");
        return false;
    }

    // Collect tour IDs as strings (YAML integers are coerced to string)
    std::vector<std::string> tourIds;
    for (std::size_t i = 0; i < tourNode.size(); ++i) {
        try {
            tourIds.push_back(tourNode[i].as<std::string>());
        } catch (const YAML::Exception&) {
            RCLCPP_ERROR(logger,
                         "[KB Validation] 'tour_specification[%zu]' cannot be read as a location ID", i);
            valid = false;
        }
    }

    // --- locations: required, non-empty map ---
    if (!root["locations"] || !root["locations"].IsMap()) {
        RCLCPP_ERROR(logger, "[KB Validation] Missing or non-map 'locations'");
        return false;
    }
    const auto& locsNode = root["locations"];
    if (locsNode.size() == 0) {
        RCLCPP_ERROR(logger, "[KB Validation] 'locations' must not be empty");
        return false;
    }

    // Build set of defined location IDs for cross-reference check
    std::vector<std::string> definedIds;
    for (const auto& locEntry : locsNode) {
        try {
            definedIds.push_back(locEntry.first.as<std::string>());
        } catch (const YAML::Exception&) {
            RCLCPP_ERROR(logger, "[KB Validation] A location key cannot be read as a string");
            valid = false;
        }
    }

    // --- Cross-reference: every tour ID must exist in locations ---
    for (const auto& id : tourIds) {
        bool found = std::find(definedIds.begin(), definedIds.end(), id) != definedIds.end();
        if (!found) {
            RCLCPP_ERROR(logger,
                         "[KB Validation] tour_specification refers to location '%s' which is not defined in 'locations'",
                         id.c_str());
            valid = false;
        }
    }

    // --- Per-location field validation ---
    for (const auto& locEntry : locsNode) {
        std::string locId;
        try {
            locId = locEntry.first.as<std::string>();
        } catch (const YAML::Exception&) {
            locId = "<unknown>";
        }
        const auto& loc = locEntry.second;

        // robot_location_description: non-empty string
        if (!loc["robot_location_description"] ||
            !loc["robot_location_description"].IsScalar() ||
            loc["robot_location_description"].as<std::string>().empty()) {
            RCLCPP_ERROR(logger,
                         "[KB Validation] Location '%s': missing or empty 'robot_location_description'",
                         locId.c_str());
            valid = false;
        }

        // robot_location_pose: map with x, y (numeric), theta in [0, 360]
        if (!loc["robot_location_pose"] || !loc["robot_location_pose"].IsMap()) {
            RCLCPP_ERROR(logger,
                         "[KB Validation] Location '%s': missing or non-map 'robot_location_pose'",
                         locId.c_str());
            valid = false;
        } else {
            const auto& pose = loc["robot_location_pose"];
            for (const char* field : {"x", "y", "theta"}) {
                if (!pose[field]) {
                    RCLCPP_ERROR(logger,
                                 "[KB Validation] Location '%s': robot_location_pose missing field '%s'",
                                 locId.c_str(), field);
                    valid = false;
                } else {
                    try { pose[field].as<double>(); }
                    catch (const YAML::Exception&) {
                        RCLCPP_ERROR(logger,
                                     "[KB Validation] Location '%s': robot_location_pose.%s is not numeric",
                                     locId.c_str(), field);
                        valid = false;
                    }
                }
            }
            // theta range check
            if (pose["theta"]) {
                try {
                    double theta = pose["theta"].as<double>();
                    if (theta < 0.0 || theta > 360.0) {
                        RCLCPP_ERROR(logger,
                                     "[KB Validation] Location '%s': robot_location_pose.theta=%.2f is outside [0, 360]",
                                     locId.c_str(), theta);
                        valid = false;
                    }
                } catch (const YAML::Exception&) { /* already reported above */ }
            }
        }

        // gesture_target: map with x, y, z (numeric), z >= 0
        if (!loc["gesture_target"] || !loc["gesture_target"].IsMap()) {
            RCLCPP_ERROR(logger,
                         "[KB Validation] Location '%s': missing or non-map 'gesture_target'",
                         locId.c_str());
            valid = false;
        } else {
            const auto& gt = loc["gesture_target"];
            for (const char* field : {"x", "y", "z"}) {
                if (!gt[field]) {
                    RCLCPP_ERROR(logger,
                                 "[KB Validation] Location '%s': gesture_target missing field '%s'",
                                 locId.c_str(), field);
                    valid = false;
                } else {
                    try { gt[field].as<double>(); }
                    catch (const YAML::Exception&) {
                        RCLCPP_ERROR(logger,
                                     "[KB Validation] Location '%s': gesture_target.%s is not numeric",
                                     locId.c_str(), field);
                        valid = false;
                    }
                }
            }
            // z must be non-negative (it is a height)
            if (gt["z"]) {
                try {
                    double z = gt["z"].as<double>();
                    if (z < 0.0) {
                        RCLCPP_ERROR(logger,
                                     "[KB Validation] Location '%s': gesture_target.z=%.2f must be >= 0",
                                     locId.c_str(), z);
                        valid = false;
                    }
                } catch (const YAML::Exception&) { /* already reported above */ }
            }
        }

        // Message fields: all four must be non-empty strings
        static const char* msgFields[] = {
            "pre_gesture_message_english",
            "pre_gesture_message_kinyarwanda",
            "post_gesture_message_english",
            "post_gesture_message_kinyarwanda"
        };
        for (const char* field : msgFields) {
            if (!loc[field] || !loc[field].IsScalar()) {
                RCLCPP_ERROR(logger,
                             "[KB Validation] Location '%s': missing field '%s'",
                             locId.c_str(), field);
                valid = false;
            } else {
                try {
                    if (loc[field].as<std::string>().empty()) {
                        RCLCPP_ERROR(logger,
                                     "[KB Validation] Location '%s': '%s' must not be empty",
                                     locId.c_str(), field);
                        valid = false;
                    }
                } catch (const YAML::Exception&) {
                    RCLCPP_ERROR(logger,
                                 "[KB Validation] Location '%s': '%s' cannot be read as a string",
                                 locId.c_str(), field);
                    valid = false;
                }
            }
        }

        // cultural_knowledge: non-empty sequence of non-empty strings
        if (!loc["cultural_knowledge"] || !loc["cultural_knowledge"].IsSequence()) {
            RCLCPP_ERROR(logger,
                         "[KB Validation] Location '%s': missing or non-sequence 'cultural_knowledge'",
                         locId.c_str());
            valid = false;
        } else {
            const auto& ck = loc["cultural_knowledge"];
            if (ck.size() == 0) {
                RCLCPP_ERROR(logger,
                             "[KB Validation] Location '%s': 'cultural_knowledge' must not be empty",
                             locId.c_str());
                valid = false;
            }
            for (std::size_t i = 0; i < ck.size(); ++i) {
                try {
                    if (ck[i].as<std::string>().empty()) {
                        RCLCPP_ERROR(logger,
                                     "[KB Validation] Location '%s': cultural_knowledge[%zu] is an empty string",
                                     locId.c_str(), i);
                        valid = false;
                    }
                } catch (const YAML::Exception&) {
                    RCLCPP_ERROR(logger,
                                 "[KB Validation] Location '%s': cultural_knowledge[%zu] cannot be read as a string",
                                 locId.c_str(), i);
                    valid = false;
                }
            }
        }
    }

    if (valid && ConfigManager::instance().isVerbose()) {
        RCLCPP_INFO(logger,
                    "[KB Validation] '%s' passed all validation checks (%zu locations, %zu tour stops)",
                    filePath.c_str(), locsNode.size(), tourNode.size());
    }
    return valid;
}

} // namespace behavior_controller