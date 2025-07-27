/* behaviorControllerImplementation.cpp

 * Author: Yohannes Tadesse Haile
 * Date: July 25, 2025
 * Version: v1.0
 */

#include "behaviorController/behaviorControllerInterface.h"

// Global node reference for behavior tree nodes
static std::shared_ptr<rclcpp::Node> g_node = nullptr;

void setGlobalNode(std::shared_ptr<rclcpp::Node> node) {
    g_node = node;
}

//=============================================================================
// Behavior Tree Node Implementations
//=============================================================================

class StartOfTree : public BT::SyncActionNode, public BaseTreeNode {
private:
    static std::atomic<bool> testStarted_;

public:
    StartOfTree(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("=== START OF TREE ===");
        
        try {
            auto& config = ConfigManager::instance();
            
            if (config.isTestMode()) {
                if (testStarted_.load()) {
                    testManager_->storeResult("TestEnded", true);
                    logger_->info("Test sequence completed");
                    return BT::NodeStatus::SUCCESS;
                } else {
                    testStarted_.store(true);
                    testManager_->storeResult("TestStarted", true);
                    logger_->info("Test sequence started");
                }
            }

            // Set speech language if ASR enabled (skip in standalone mode)
            if (config.isAsrEnabled()) {
                auto request = std::make_shared<cssr_system::srv::SpeechEventSetLanguage::Request>();
                auto response = std::make_shared<cssr_system::srv::SpeechEventSetLanguage::Response>();
                request->language = config.getLanguage();
                
                if (!callServiceSafely<cssr_system::srv::SpeechEventSetLanguage>(
                    "/speechEvent/set_language", request, response, "StartOfTree")) {
                    logger_->warn("Speech service not available - running in standalone mode");
                    // Don't fail, just continue in standalone mode
                }
            }

            storeTestResult("StartOfTree", true);
            return BT::NodeStatus::SUCCESS;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in StartOfTree: " + std::string(e.what()));
            storeTestResult("StartOfTree", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

std::atomic<bool> StartOfTree::testStarted_{false};

//=============================================================================

class SayText : public BT::SyncActionNode, public BaseTreeNode {
public:
    SayText(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SayText Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::TextToSpeechSayText::Request>();
            auto response = std::make_shared<cssr_system::srv::TextToSpeechSayText::Response>();
            
            request->language = ConfigManager::instance().getLanguage();
            
            std::string phraseId = name(); // Get phrase ID from node name
            request->message = KnowledgeManager::instance().getUtilityPhrase(phraseId);
            
            if (callServiceSafely<cssr_system::srv::TextToSpeechSayText>(
                "/textToSpeech/say_text", request, response, "SayText")) {
                if (response->success) {
                    storeTestResult("SayText", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("SayText", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in SayText: " + std::string(e.what()));
            storeTestResult("SayText", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class Navigate : public BT::SyncActionNode, public BaseTreeNode {
public:
    Navigate(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("Navigate Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::RobotNavigationSetGoal::Request>();
            auto response = std::make_shared<cssr_system::srv::RobotNavigationSetGoal::Response>();
            
            RobotPose location;
            if (!config().blackboard->rootBlackboard()->get("exhibitLocation", location)) {
                logger_->error("Unable to retrieve location from blackboard");
                storeTestResult("Navigate", false);
                return BT::NodeStatus::FAILURE;
            }
            
            request->goal_x = location.x;
            request->goal_y = location.y;
            request->goal_theta = location.theta;
            
            if (callServiceSafely<cssr_system::srv::RobotNavigationSetGoal>(
                "/robotNavigation/set_goal", request, response, "Navigate")) {
                if (response->navigation_goal_success) {
                    storeTestResult("Navigate", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("Navigate", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in Navigate: " + std::string(e.what()));
            storeTestResult("Navigate", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class GetVisitorResponse : public BT::SyncActionNode, public BaseTreeNode {
private:
    std::atomic<bool> responseReceived_{false};
    std::string visitorResponse_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscriber_;
    mutable std::mutex responseMutex_;

public:
    GetVisitorResponse(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {
        
        subscriber_ = node_->create_subscription<std_msgs::msg::String>(
            "/speechEvent/text", 10,
            [this](const std_msgs::msg::String::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(responseMutex_);
                visitorResponse_ = msg->data;
                responseReceived_.store(true);
            });
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("GetVisitorResponse Action Node");
        
        try {
            rclcpp::Rate rate(Constants::LOOP_RATE_HZ);
            auto startTime = node_->get_clock()->now();
            auto timeout = std::chrono::seconds(Constants::VISITOR_RESPONSE_TIMEOUT_SEC);
            
            // Reset state
            responseReceived_.store(false);
            visitorResponse_.clear();
            
            // Define affirmative words for different languages
            std::unordered_map<std::string, std::vector<std::string>> affirmativeWords;
            affirmativeWords["English"] = {"yes", "great", "absolutely", "go", "happy", "good", "love"};
            affirmativeWords["Kinyarwanda"] = {"yego", "ntakibazo", "nibyo"};
            affirmativeWords["IsiZulu"] = {"yebo", "kulungile"};
            
            while (rclcpp::ok()) {
                rclcpp::spin_some(node_);
                
                if (responseReceived_.load()) {
                    std::lock_guard<std::mutex> lock(responseMutex_);
                    
                    std::string currentLanguage = ConfigManager::instance().getLanguage();
                    auto it = affirmativeWords.find(currentLanguage);
                    
                    if (it != affirmativeWords.end()) {
                        if (TextUtils::containsAnyWord(visitorResponse_, it->second)) {
                            config().blackboard->rootBlackboard()->set("visitorResponse", "yes");
                            logger_->info("Visitor Response: " + visitorResponse_);
                            storeTestResult("GetVisitorResponse", true);
                            return BT::NodeStatus::SUCCESS;
                        }
                    }
                }
                
                // Check for timeout
                if ((node_->get_clock()->now() - startTime) > rclcpp::Duration(timeout)) {
                    logger_->warn("No affirmative response received within timeout");
                    config().blackboard->rootBlackboard()->set("visitorResponse", "no");
                    storeTestResult("GetVisitorResponse", false);
                    return BT::NodeStatus::FAILURE;
                }
                
                rate.sleep();
            }
            
            storeTestResult("GetVisitorResponse", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in GetVisitorResponse: " + std::string(e.what()));
            storeTestResult("GetVisitorResponse", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class IsVisitorDiscovered : public BT::ConditionNode, public BaseTreeNode {
private:
    std::atomic<bool> visitorDiscovered_{false};
    rclcpp::Subscription<cssr_system::msg::FaceDetectionData>::SharedPtr subscriber_;

public:
    IsVisitorDiscovered(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ConditionNode(name, config), BaseTreeNode(g_node) {
        
        subscriber_ = node_->create_subscription<cssr_system::msg::FaceDetectionData>(
            "/faceDetection/data", 10,
            [this](const cssr_system::msg::FaceDetectionData::SharedPtr msg) {
                visitorDiscovered_.store(msg->face_label_id.size() > 0);
            });
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("IsVisitorDiscovered Condition Node");
        
        try {
            rclcpp::Rate rate(Constants::LOOP_RATE_HZ);
            auto startTime = node_->get_clock()->now();
            auto timeout = std::chrono::seconds(Constants::RESPONSE_TIMEOUT_SEC);
            
            while (rclcpp::ok()) {
                rclcpp::spin_some(node_);
                
                if (visitorDiscovered_.load()) {
                    logger_->info("Visitor discovered");
                    storeTestResult("IsVisitorDiscovered", true);
                    return BT::NodeStatus::SUCCESS;
                }
                
                if ((node_->get_clock()->now() - startTime) > rclcpp::Duration(timeout)) {
                    logger_->warn("Visitor discovery timeout");
                    break;
                }
                
                rate.sleep();
            }
            
            storeTestResult("IsVisitorDiscovered", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in IsVisitorDiscovered: " + std::string(e.what()));
            storeTestResult("IsVisitorDiscovered", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class SelectExhibit : public BT::SyncActionNode, public BaseTreeNode {
public:
    SelectExhibit(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SelectExhibit Action Node");
        
        try {
            int visits = 0;
            if (!config().blackboard->get("visits", visits)) {
                logger_->warn("Unable to retrieve visits from blackboard");
                storeTestResult("SelectExhibit", false);
                return BT::NodeStatus::FAILURE;
            }
            
            auto& knowledge = KnowledgeManager::instance();
            auto tour = knowledge.getTourSpecification();
            
            if (visits >= tour.getCurrentLocationCount()) {
                logger_->warn("Visit index out of range");
                storeTestResult("SelectExhibit", false);
                return BT::NodeStatus::FAILURE;
            }
            
            // Get location information
            auto locationInfo = knowledge.getLocationInfo(tour.locationIds[visits]);
            std::string currentLanguage = ConfigManager::instance().getLanguage();
            
            // Get messages for current language
            auto preIt = locationInfo.preMessages.find(currentLanguage);
            auto postIt = locationInfo.postMessages.find(currentLanguage);
            
            if (preIt == locationInfo.preMessages.end() || postIt == locationInfo.postMessages.end()) {
                logger_->error("Messages not found for language: " + currentLanguage);
                storeTestResult("SelectExhibit", false);
                return BT::NodeStatus::FAILURE;
            }
            
            if (preIt->second.empty() || postIt->second.empty()) {
                logger_->error("Empty messages for language: " + currentLanguage);
                storeTestResult("SelectExhibit", false);
                return BT::NodeStatus::FAILURE;
            }
            
            // Store values in blackboard
            config().blackboard->set("exhibitPreGestureMessage", preIt->second);
            config().blackboard->set("exhibitPostGestureMessage", postIt->second);
            config().blackboard->rootBlackboard()->set("exhibitLocation", locationInfo.robotPose);
            config().blackboard->rootBlackboard()->set("exhibitGestureTarget", locationInfo.gestureTarget);
            
            logger_->info("Visiting: " + locationInfo.description);
            
            config().blackboard->set("visits", ++visits);
            storeTestResult("SelectExhibit", true);
            return BT::NodeStatus::SUCCESS;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in SelectExhibit: " + std::string(e.what()));
            storeTestResult("SelectExhibit", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class IsListWithExhibit : public BT::ConditionNode, public BaseTreeNode {
public:
    IsListWithExhibit(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ConditionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("IsListWithExhibit Condition Node");
        
        try {
            int visits = 0;
            if (!config().blackboard->get("visits", visits)) {
                logger_->error("Unable to retrieve visits from blackboard");
                storeTestResult("IsListWithExhibit", false);
                return BT::NodeStatus::FAILURE;
            }
            
            auto tour = KnowledgeManager::instance().getTourSpecification();
            
            if (visits < tour.getCurrentLocationCount()) {
                storeTestResult("IsListWithExhibit", true);
                return BT::NodeStatus::SUCCESS;
            } else {
                logger_->info("ALL LOCATIONS VISITED");
                storeTestResult("IsListWithExhibit", false);
                return BT::NodeStatus::FAILURE;
            }
            
        } catch (const std::exception& e) {
            logger_->error("Exception in IsListWithExhibit: " + std::string(e.what()));
            storeTestResult("IsListWithExhibit", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class RetrieveListOfExhibits : public BT::SyncActionNode, public BaseTreeNode {
public:
    RetrieveListOfExhibits(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("RetrieveListOfExhibits Action Node");
        
        try {
            auto tour = KnowledgeManager::instance().getTourSpecification();
            
            if (tour.getCurrentLocationCount() == 0) {
                logger_->error("No exhibits found");
                storeTestResult("RetrieveListOfExhibits", false);
                return BT::NodeStatus::FAILURE;
            }
            
            config().blackboard->set("visits", 0);
            storeTestResult("RetrieveListOfExhibits", true);
            return BT::NodeStatus::SUCCESS;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in RetrieveListOfExhibits: " + std::string(e.what()));
            storeTestResult("RetrieveListOfExhibits", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class PerformDeicticGesture : public BT::SyncActionNode, public BaseTreeNode {
public:
    PerformDeicticGesture(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("PerformDeicticGesture Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::GestureExecutionPerformGesture::Request>();
            auto response = std::make_shared<cssr_system::srv::GestureExecutionPerformGesture::Response>();
            
            Position3D gestureTarget;
            if (!config().blackboard->rootBlackboard()->get("exhibitGestureTarget", gestureTarget)) {
                logger_->error("Unable to retrieve gesture target from blackboard");
                storeTestResult("PerformDeicticGesture", false);
                return BT::NodeStatus::FAILURE;
            }
            
            request->gesture_type = "deictic";
            request->gesture_id = Constants::DEICTIC_GESTURE_ID;
            request->gesture_duration = Constants::GESTURE_DURATION_MS;
            request->bow_nod_angle = 0;
            request->location_x = gestureTarget.x;
            request->location_y = gestureTarget.y;
            request->location_z = gestureTarget.z;
            
            if (callServiceSafely<cssr_system::srv::GestureExecutionPerformGesture>(
                "/gestureExecution/perform_gesture", request, response, "PerformDeicticGesture")) {
                if (response->gesture_success) {
                    storeTestResult("PerformDeicticGesture", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("PerformDeicticGesture", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in PerformDeicticGesture: " + std::string(e.what()));
            storeTestResult("PerformDeicticGesture", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class PerformIconicGesture : public BT::SyncActionNode, public BaseTreeNode {
public:
    PerformIconicGesture(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("PerformIconicGesture Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::GestureExecutionPerformGesture::Request>();
            auto response = std::make_shared<cssr_system::srv::GestureExecutionPerformGesture::Response>();
            
            std::string gestureType = name(); // Get from node name
            logger_->info("Gesture: " + gestureType);
            
            request->gesture_type = "iconic";
            request->gesture_duration = Constants::GESTURE_DURATION_MS;
            request->bow_nod_angle = 0;
            request->location_x = 0.0;
            request->location_y = 0.0;
            request->location_z = 0.0;
            
            if (gestureType == "welcome") {
                request->gesture_id = Constants::WELCOME_GESTURE_ID;
            } else if (gestureType == "goodbye") {
                request->gesture_id = Constants::GOODBYE_GESTURE_ID;
            } else {
                logger_->error("Undefined Iconic Gesture Type: " + gestureType);
                storeTestResult("PerformIconicGesture", false);
                return BT::NodeStatus::FAILURE;
            }
            
            if (callServiceSafely<cssr_system::srv::GestureExecutionPerformGesture>(
                "/gestureExecution/perform_gesture", request, response, "PerformIconicGesture")) {
                if (response->gesture_success) {
                    storeTestResult("PerformIconicGesture", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("PerformIconicGesture", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in PerformIconicGesture: " + std::string(e.what()));
            storeTestResult("PerformIconicGesture", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class DescribeExhibitSpeech : public BT::SyncActionNode, public BaseTreeNode {
public:
    DescribeExhibitSpeech(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("DescribeExhibit Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::TextToSpeechSayText::Request>();
            auto response = std::make_shared<cssr_system::srv::TextToSpeechSayText::Response>();
            
            request->language = ConfigManager::instance().getLanguage();
            
            std::string nodeInstance = name(); // Get instance from node name
            std::string message;
            
            if (nodeInstance == "1") {
                // Pre-Gesture Message
                if (!config().blackboard->get("exhibitPreGestureMessage", message)) {
                    logger_->error("Unable to retrieve pre-gesture message from blackboard");
                    storeTestResult("DescribeExhibit", false);
                    return BT::NodeStatus::FAILURE;
                }
            } else if (nodeInstance == "2") {
                // Post-Gesture Message
                if (!config().blackboard->get("exhibitPostGestureMessage", message)) {
                    logger_->error("Unable to retrieve post-gesture message from blackboard");
                    storeTestResult("DescribeExhibit", false);
                    return BT::NodeStatus::FAILURE;
                }
            } else {
                logger_->warn("Invalid Node Instance: " + nodeInstance);
                storeTestResult("DescribeExhibit", false);
                return BT::NodeStatus::FAILURE;
            }
            
            if (message.empty()) {
                logger_->warn("Empty message for DescribeExhibit");
                storeTestResult("DescribeExhibit", false);
                return BT::NodeStatus::FAILURE;
            }
            
            request->message = message;
            
            if (callServiceSafely<cssr_system::srv::TextToSpeechSayText>(
                "/textToSpeech/say_text", request, response, "DescribeExhibit")) {
                if (response->success) {
                    // Provide buffer after speech
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                    storeTestResult("DescribeExhibit", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("DescribeExhibit", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in DescribeExhibit: " + std::string(e.what()));
            storeTestResult("DescribeExhibit", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class IsMutualGazeDiscovered : public BT::ConditionNode, public BaseTreeNode {
private:
    std::atomic<int> seekingStatus_{0}; // 0=RUNNING, 1=SUCCESS, 2=FAILURE
    rclcpp::Subscription<cssr_system::msg::OvertAttentionMode>::SharedPtr subscriber_;

public:
    IsMutualGazeDiscovered(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ConditionNode(name, config), BaseTreeNode(g_node) {
        
        subscriber_ = node_->create_subscription<cssr_system::msg::OvertAttentionMode>(
            "/overtAttention/mode", 10,
            [this](const cssr_system::msg::OvertAttentionMode::SharedPtr msg) {
                if (msg->state == "seeking") {
                    if (msg->value == 2) {
                        seekingStatus_.store(1); // SUCCESS
                    } else if (msg->value == 3) {
                        seekingStatus_.store(2); // FAILURE
                    }
                }
            });
    }

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("IsMutualGazeDiscovered Condition Node");
        
        try {
            rclcpp::Rate rate(Constants::LOOP_RATE_HZ);
            auto startTime = node_->get_clock()->now();
            auto timeout = std::chrono::seconds(Constants::RESPONSE_TIMEOUT_SEC);
            
            seekingStatus_.store(0);
            
            while (rclcpp::ok()) {
                rclcpp::spin_some(node_);
                
                int status = seekingStatus_.load();
                if (status == 1) { // SUCCESS
                    logger_->info("Mutual gaze detected");
                    storeTestResult("IsMutualGazeDiscovered", true);
                    return BT::NodeStatus::SUCCESS;
                } else if (status == 2) { // FAILURE
                    logger_->info("Mutual gaze detection failed");
                    storeTestResult("IsMutualGazeDiscovered", false);
                    return BT::NodeStatus::FAILURE;
                }
                
                if ((node_->get_clock()->now() - startTime) > rclcpp::Duration(timeout)) {
                    logger_->warn("Mutual gaze detection timeout");
                    break;
                }
                
                rate.sleep();
            }
            
            storeTestResult("IsMutualGazeDiscovered", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in IsMutualGazeDiscovered: " + std::string(e.what()));
            storeTestResult("IsMutualGazeDiscovered", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class IsVisitorResponseYes : public BT::ConditionNode, public BaseTreeNode {
public:
    IsVisitorResponseYes(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ConditionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("IsVisitorResponseYes Condition Node");
        
        try {
            auto blackboard = config().blackboard->rootBlackboard();
            std::string visitorResponse;
            
            if (blackboard->get("visitorResponse", visitorResponse)) {
                if (visitorResponse == "yes") {
                    storeTestResult("IsVisitorResponseYes", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("IsVisitorResponseYes", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in IsVisitorResponseYes: " + std::string(e.what()));
            storeTestResult("IsVisitorResponseYes", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class IsASREnabled : public BT::ConditionNode, public BaseTreeNode {
public:
    IsASREnabled(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ConditionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("IsASREnabled Condition Node");
        
        if (ConfigManager::instance().isAsrEnabled()) {
            return BT::NodeStatus::SUCCESS;
        }
        return BT::NodeStatus::FAILURE;
    }
};

//=============================================================================

class SetSpeechEvent : public BT::SyncActionNode, public BaseTreeNode {
public:
    SetSpeechEvent(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SetSpeechEvent Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::SpeechEventSetEnabled::Request>();
            auto response = std::make_shared<cssr_system::srv::SpeechEventSetEnabled::Response>();
            
            std::string status = name(); // Get status from node name
            logger_->info("Status: " + status);
            request->status = status;
            
            if (callServiceSafely<cssr_system::srv::SpeechEventSetEnabled>(
                "/speechEvent/set_enabled", request, response, "SetSpeechEvent")) {
                if (response->response) {
                    storeTestResult("SetSpeechEvent", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("SetSpeechEvent", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in SetSpeechEvent: " + std::string(e.what()));
            storeTestResult("SetSpeechEvent", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class SetOvertAttentionMode : public BT::SyncActionNode, public BaseTreeNode {
public:
    SetOvertAttentionMode(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SetOvertAttentionMode Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::OvertAttentionSetMode::Request>();
            auto response = std::make_shared<cssr_system::srv::OvertAttentionSetMode::Response>();
            
            std::string state = name(); // Get state from node name
            logger_->info("State: " + state);
            request->state = state;

            // Handle location mode
            if (state == "location") {
                Position3D gestureTarget;
                if (!config().blackboard->rootBlackboard()->get("exhibitGestureTarget", gestureTarget)) {
                    logger_->error("Unable to retrieve gesture target from blackboard");
                    storeTestResult("SetOvertAttentionMode", false);
                    return BT::NodeStatus::FAILURE;
                }
                
                request->location_x = gestureTarget.x;
                request->location_y = gestureTarget.y;
                request->location_z = gestureTarget.z;
            }
            
            if (callServiceSafely<cssr_system::srv::OvertAttentionSetMode>(
                "/overtAttention/set_mode", request, response, "SetOvertAttentionMode")) {
                if (response->mode_set_success) {
                    storeTestResult("SetOvertAttentionMode", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("SetOvertAttentionMode", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in SetOvertAttentionMode: " + std::string(e.what()));
            storeTestResult("SetOvertAttentionMode", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class SetAnimateBehavior : public BT::SyncActionNode, public BaseTreeNode {
public:
    SetAnimateBehavior(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("SetAnimateBehavior Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::AnimateBehaviorSetActivation::Request>();
            auto response = std::make_shared<cssr_system::srv::AnimateBehaviorSetActivation::Response>();
            
            std::string state = name(); // Get state from node name
            logger_->info("State: " + state);
            request->state = state;
            
            if (callServiceSafely<cssr_system::srv::AnimateBehaviorSetActivation>(
                "/animateBehaviour/setActivation", request, response, "SetAnimateBehavior")) {
                if (response->success == "1") {
                    storeTestResult("SetAnimateBehavior", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("SetAnimateBehavior", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in SetAnimateBehavior: " + std::string(e.what()));
            storeTestResult("SetAnimateBehavior", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class ResetRobotPose : public BT::SyncActionNode, public BaseTreeNode {
public:
    ResetRobotPose(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("ResetRobotPose Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::RobotLocalizationResetPose::Request>();
            auto response = std::make_shared<cssr_system::srv::RobotLocalizationResetPose::Response>();
            
            if (callServiceSafely<cssr_system::srv::RobotLocalizationResetPose>(
                "/robotLocalization/reset_pose", request, response, "ResetRobotPose")) {
                if (response->success) {
                    storeTestResult("ResetRobotPose", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("ResetRobotPose", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in ResetRobotPose: " + std::string(e.what()));
            storeTestResult("ResetRobotPose", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class PressYesNoDialogue : public BT::SyncActionNode, public BaseTreeNode {
public:
    PressYesNoDialogue(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("PressYesNoDialogue(Tablet) Action Node");
        
        try {
            auto request = std::make_shared<cssr_system::srv::TabletEventPromptAndGetResponse::Request>();
            auto response = std::make_shared<cssr_system::srv::TabletEventPromptAndGetResponse::Response>();
            
            request->message = "'Yes'|'No'";
            
            if (callServiceSafely<cssr_system::srv::TabletEventPromptAndGetResponse>(
                "/tabletEvent/prompt_and_get_response", request, response, "PressYesNoDialogue")) {
                if (response->success) {
                    storeTestResult("PressYesNoDialogue", true);
                    return BT::NodeStatus::SUCCESS;
                }
            }
            
            storeTestResult("PressYesNoDialogue", false);
            return BT::NodeStatus::FAILURE;
            
        } catch (const std::exception& e) {
            logger_->error("Exception in PressYesNoDialogue: " + std::string(e.what()));
            storeTestResult("PressYesNoDialogue", false);
            return BT::NodeStatus::FAILURE;
        }
    }
};

//=============================================================================

class HandleFallBack : public BT::SyncActionNode, public BaseTreeNode {
public:
    HandleFallBack(const std::string& name, const BT::NodeConfiguration& config)
        : BT::SyncActionNode(name, config), BaseTreeNode(g_node) {}

    static BT::PortsList providedPorts() { return {}; }

    BT::NodeStatus tick() override {
        logger_->info("HandleFallback Action Node");
        storeTestResult("HandleFallback", true);
        return BT::NodeStatus::SUCCESS;
    }
};

//=============================================================================
// Tree Initialization Function
//=============================================================================

BT::Tree initializeTree(const std::string& scenario, std::shared_ptr<rclcpp::Node> node) {
    BT::BehaviorTreeFactory factory;
    
    // Set global node reference for behavior tree nodes
    setGlobalNode(node);
    
    // Register all node types with simplified registration
    factory.registerNodeType<StartOfTree>("StartOfTree");
    factory.registerNodeType<SayText>("SayText");
    factory.registerNodeType<Navigate>("Navigate");
    factory.registerNodeType<GetVisitorResponse>("GetVisitorResponse");
    factory.registerNodeType<IsVisitorDiscovered>("IsVisitorDiscovered");
    factory.registerNodeType<SelectExhibit>("SelectExhibit");
    factory.registerNodeType<IsListWithExhibit>("IsListWithExhibit");
    factory.registerNodeType<RetrieveListOfExhibits>("RetrieveListOfExhibits");
    factory.registerNodeType<PerformDeicticGesture>("PerformDeicticGesture");
    factory.registerNodeType<PerformIconicGesture>("PerformIconicGesture");
    factory.registerNodeType<DescribeExhibitSpeech>("DescribeExhibitSpeech");
    factory.registerNodeType<IsMutualGazeDiscovered>("IsMutualGazeDiscovered");
    factory.registerNodeType<IsVisitorResponseYes>("IsVisitorResponseYes");
    factory.registerNodeType<IsASREnabled>("IsASREnabled");
    factory.registerNodeType<SetSpeechEvent>("SetSpeechEvent");
    factory.registerNodeType<SetOvertAttentionMode>("SetOvertAttentionMode");
    factory.registerNodeType<SetAnimateBehavior>("SetAnimateBehavior");
    factory.registerNodeType<ResetRobotPose>("ResetRobotPose");
    factory.registerNodeType<PressYesNoDialogue>("PressYesNoDialogue");
    factory.registerNodeType<HandleFallBack>("HandleFallBack");
    
    // Load tree from XML file
    std::string packagePath = ament_index_cpp::get_package_share_directory("cssr_system");
    std::string treeFilePath = packagePath + "/data/" + scenario + ".xml";
    
    std::ifstream file(treeFilePath);
    if (!file.good()) {
        throw std::runtime_error("Behavior tree file not found: " + treeFilePath);
    }
    file.close();
    
    return factory.createTreeFromFile(treeFilePath);
}
