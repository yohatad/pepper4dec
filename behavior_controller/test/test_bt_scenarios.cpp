/* test_bt_scenarios.cpp
 *
 * Tests the scenario behavior trees in data/ as a whole:
 *
 *   - every .xml file in data/ builds against the node types that
 *     behavior_controller::registerNodes() registers (unknown node IDs and
 *     ports are rejected), and the shipped configuration names files that
 *     exist;
 *   - each scenario is ticked to completion, or for a few loops of the
 *     pipelines that repeat forever, against a simulated system (see
 *     ScenarioTest): the tests then check what the robot did, i.e. where it
 *     drove, what it pointed at and what it said, in order.
 *
 * The tour simulations load the knowledge bases shipped in the source
 * tree's data/ directory, so they run on the data deployed on the robot.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: October 11, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "bt_test_harness.hpp"

namespace
{

using bt_test::BtRosTest;
using bt_test::Outcome;
using bt_test::Reply;
using bt_test::StatusRecorder;
using BT::NodeStatus;
using std::chrono::milliseconds;

using AnimateBehavior = dec_interfaces::action::AnimateBehavior;
using ConversationManager = dec_interfaces::action::ConversationManager;
using Gesture = dec_interfaces::action::Gesture;
using NavigateToPose = nav2_msgs::action::NavigateToPose;
using SpeechRecognition = dec_interfaces::action::SpeechRecognition;
using SpeechWithFeedback = naoqi_bridge_msgs::action::SpeechWithFeedback;
using TTS = dec_interfaces::action::TTS;
using SetBool = std_srvs::srv::SetBool;
using Trigger = std_srvs::srv::Trigger;
using FaceDetection = dec_interfaces::msg::FaceDetection;
using StringMsg = std_msgs::msg::String;
using Events = std::vector<std::string>;

const char kPackageDir[] = TEST_PACKAGE_DIR;

std::string dataPath(const std::string & file)
{
  return std::string(kPackageDir) + "/data/" + file;
}

YAML::Node shippedParameters()
{
  return YAML::LoadFile(
    std::string(kPackageDir) + "/config/behavior_controller_configuration.yaml")
         ["behavior_controller"]["ros__parameters"];
}

/// Points ConfigManager at the shipped culture KB and @p environment_kb, then
/// loads both from the source tree's data/ directory.
void loadKnowledgeBase(const std::string & environment_kb)
{
  const std::string config = ::testing::TempDir() + "behavior_controller_scenario_config.yaml";
  std::ofstream(config) <<
    "culture_knowledge_base: \"cultureKnowledgeBase.yaml\"\n"
    "environment_knowledge_base: \"" << environment_kb << "\"\n";
  ASSERT_TRUE(ConfigManager::instance().loadFromFile(config));
  ASSERT_TRUE(KnowledgeManager::instance().loadFromPackage(kPackageDir));
}

/// "navigate x=.. y=.. heading=.." with the heading in degrees, in [0, 360).
std::string describeNavigation(double x, double y, double heading_degrees)
{
  double heading = std::fmod(heading_degrees, 360.0);
  if (heading < 0.0) {
    heading += 360.0;
  }
  heading = std::round(heading * 10.0) / 10.0;
  if (heading >= 360.0 || heading == 0.0) {
    heading = 0.0;
  }
  char text[96];
  std::snprintf(text, sizeof(text), "navigate x=%.2f y=%.2f heading=%.1f", x, y, heading);
  return text;
}

std::string describeNavigation(const geometry_msgs::msg::PoseStamped & goal)
{
  const auto & q = goal.pose.orientation;
  const double yaw = 2.0 * std::atan2(q.z, q.w);
  return describeNavigation(goal.pose.position.x, goal.pose.position.y, yaw * 180.0 / M_PI);
}

std::string describePoint(double x, double y, double z)
{
  char text[96];
  std::snprintf(text, sizeof(text), "gesture point at x=%.2f y=%.2f z=%.2f", x, y, z);
  return text;
}

/// The iconic gestures gesture_execution can play: the keys of its
/// data/gesture.yaml (read from the source tree, next to this package).
std::set<std::string> iconicGestures()
{
  std::set<std::string> names;
  const std::string file = std::string(kPackageDir) + "/../gesture_execution/data/gesture.yaml";
  for (const auto & gesture : YAML::LoadFile(file)["gestures"]) {
    names.insert(gesture.first.as<std::string>());
  }
  return names;
}

/// What gesture_execution does with @p goal, as an event: "gesture point at
/// ..." for a deictic gesture, "gesture <name>" for an iconic one, or
/// std::nullopt when it would fail. It points only for gesture_type
/// "deictic", and plays any type other than deictic, bow, nod and symbolic
/// (not implemented) as an iconic gesture named by gesture_name for
/// "iconic", by gesture_type otherwise.
std::optional<std::string> describeGesture(
  const Gesture::Goal & goal, const std::set<std::string> & iconic)
{
  const std::string & type = goal.gesture_type;
  if (type == "deictic") {
    return describePoint(goal.location_x, goal.location_y, goal.location_z);
  }
  if (type == "bow" || type == "nod") {
    return "gesture " + type;
  }
  const std::string name = type == "iconic" ? goal.gesture_name : type;
  if (type == "symbolic" || iconic.count(name) == 0) {
    return std::nullopt;
  }
  return "gesture " + name;
}

/// Where the tour trees send the robot before and after the tour:
/// Navigate goal_x="4.0" goal_y="6.8" goal_theta="1.5708".
std::string standby()
{
  return describeNavigation(4.0, 6.8, 1.5708 * 180.0 / M_PI);
}

/// The phrases the tours read but nothing in them writes: the node provides
/// them from the culture knowledge base (createRootBlackboard()).
const char * const kTourPhraseKeys[] = {
  "welcome_speech", "follow_me_speech", "maybe_another_time_speech",
  "brief_goodbye_speech", "goodbye_speech"};

/// "say <phrase>" for the culture knowledge base phrase @p id.
std::string say(const std::string & id)
{
  return "say " + KnowledgeManager::instance().getUtilityPhrase(id);
}

const char kTourQuestion[] = "say Would you like me to guide you through the Upanzi lab?";

/**
 * @brief The rest of the system, simulated.
 *
 * startWorld() serves every action and service the scenarios call, each
 * answering as a working system would, and puts a visitor in front of the
 * robot who holds its gaze and says @p visitor_says whenever the microphone
 * is on (it starts on; setVisitorSays() changes the answer). The conversation manager reads an utterance starting
 * with "yes" as yes, one starting with "no" as no, and answers anything else
 * with "Answer to: <utterance>".
 *
 * After @p turns utterances the visitor goes quiet, so the pipelines that loop
 * forever come to rest waiting for speech (see runUntil()).
 *
 * The gesture server follows gesture_execution's contract (see
 * describeGesture()) and aborts any other gesture, as the real one does.
 *
 * Every goal and request is logged as an event ("navigate ...", "gesture
 * ...", "say ...", "ask ...", "mic on", ...) in the order the robot made it.
 */
class ScenarioTest : public BtRosTest
{
protected:
  void startWorld(const std::string & visitor_says, int turns = 1000)
  {
    setVisitorSays(visitor_says);
    serveAction<AnimateBehavior>(
      "/animate_behavior", [this](const AnimateBehavior::Goal & goal) {
        log("animate");
        Reply<AnimateBehavior> reply;
        reply.result.success = true;
        if (goal.duration_seconds == 0) {
          reply.outcome = Outcome::kRunUntilCanceled;
        }
        return reply;
      });
    serveService<Trigger>(
      "/animate_behavior/stop", [this](const Trigger::Request &, Trigger::Response & response) {
        log("stop animation");
        response.success = true;
      });
    serveService<SetBool>(
      "/overt_attention/set_enabled",
      [this](const SetBool::Request & request, SetBool::Response & response) {
        log(request.data ? "attention on" : "attention off");
        response.success = true;
      });
    serveService<SetBool>(
      "/speech_event/set_enabled",
      [this](const SetBool::Request & request, SetBool::Response & response) {
        *mic_on_ = request.data;
        if (!request.data) {
          ++*turns_taken_;  // the robot mutes the mic once it has heard a turn
        }
        log(request.data ? "mic on" : "mic off");
        response.success = true;
      });
    serveAction<Gesture>(
      "/gesture_execution", [this, iconic = iconicGestures()](const Gesture::Goal & goal) {
        const auto gesture = describeGesture(goal, iconic);
        Reply<Gesture> reply;
        if (gesture) {
          log(*gesture);
          reply.result.success = true;
        } else {
          // gesture_execution aborts a gesture it cannot perform.
          log("gesture rejected: type=" + goal.gesture_type + " name=" + goal.gesture_name);
          reply.outcome = Outcome::kAbort;
        }
        return reply;
      });
    serveAction<NavigateToPose>(
      "/navigate_to_pose", [this](const NavigateToPose::Goal & goal) {
        log(describeNavigation(goal.pose));
        return Reply<NavigateToPose>();
      });
    serveAction<SpeechWithFeedback>(
      "/naoqi_driver/naoqi_driver/speech_with_feedback",
      [this](const SpeechWithFeedback::Goal & goal) {
        log("say " + goal.say);
        Reply<SpeechWithFeedback> reply;
        reply.result.success = true;
        return reply;
      });
    serveAction<TTS>(
      "/text_to_speech", [this](const TTS::Goal & goal) {
        log("say " + goal.text);
        Reply<TTS> reply;
        reply.result.success = true;
        return reply;
      });
    serveAction<SpeechRecognition>(
      "/speech_recognition", [this, turns](const SpeechRecognition::Goal &) {
        log("recognize");
        Reply<SpeechRecognition> reply;
        reply.result.transcription = visitorSays();
        if (++*turns_taken_ > turns) {
          reply.outcome = Outcome::kRunUntilCanceled;  // still listening
        }
        return reply;
      });
    serveAction<ConversationManager>(
      "/conversation_manager", [this](const ConversationManager::Goal & goal) {
        log("ask " + goal.prompt);
        const std::string lower = TextUtils::toLowerCase(goal.prompt);
        Reply<ConversationManager> reply;
        reply.result.success = true;
        if (lower.rfind("yes", 0) == 0) {
          reply.result.response = "yes";
        } else if (lower.rfind("no", 0) == 0) {
          reply.result.response = "no";
        } else {
          reply.result.response = "Answer to: " + goal.prompt;
        }
        return reply;
      });

    publishEvery<FaceDetection>(
      "/face_detection/data", milliseconds(50), []() {return bt_test::faces({true});});
    auto mic_on = mic_on_;
    auto turns_taken = turns_taken_;
    auto visitor = visitor_;
    publishEvery<StringMsg>(
      "/speech_event/text", milliseconds(100),
      [mic_on, turns_taken, turns, visitor]() -> std::optional<StringMsg> {
        if (!*mic_on || *turns_taken >= turns) {
          return std::nullopt;
        }
        StringMsg msg;
        std::lock_guard<std::mutex> lock(visitor->mutex);
        msg.data = visitor->says;
        return msg;
      });
  }

  void setVisitorSays(const std::string & says)
  {
    std::lock_guard<std::mutex> lock(visitor_->mutex);
    visitor_->says = says;
  }

  /// Logged events starting with @p prefix, in order.
  Events events(const std::string & prefix = "") const
  {
    std::lock_guard<std::mutex> lock(events_mutex_);
    Events matching;
    for (const auto & event : events_) {
      if (event.rfind(prefix, 0) == 0) {
        matching.push_back(event);
      }
    }
    return matching;
  }

  /// All logged events, one per line, for failure messages.
  std::string eventLog() const
  {
    std::string log;
    for (const auto & event : events()) {
      log += "  " + event + "\n";
    }
    return log;
  }

  /// Ticks until @p done() holds, then 300 ms more so the tree comes to rest
  /// in the node it now waits in. Halting an action node whose goal the
  /// server has not answered yet is not safe (see test_bt_nodes.cpp), and the
  /// tree is halted when the test ends.
  static void runUntil(BT::Tree & tree, const std::function<bool()> & done)
  {
    tickUntil(tree, done, milliseconds(20000));
    tickFor(tree, milliseconds(300));
  }

private:
  struct Visitor
  {
    std::mutex mutex;
    std::string says;
  };

  std::string visitorSays() const
  {
    std::lock_guard<std::mutex> lock(visitor_->mutex);
    return visitor_->says;
  }

  void log(const std::string & event)
  {
    std::lock_guard<std::mutex> lock(events_mutex_);
    events_.push_back(event);
  }

  std::shared_ptr<std::atomic<bool>> mic_on_ = std::make_shared<std::atomic<bool>>(true);
  /// Utterances heard: speech recognition results, or mic mutes after
  /// /speech_event/text was heard.
  std::shared_ptr<std::atomic<int>> turns_taken_ = std::make_shared<std::atomic<int>>(0);
  std::shared_ptr<Visitor> visitor_ = std::make_shared<Visitor>();
  Events events_;
  mutable std::mutex events_mutex_;
};

}  // namespace

// The tests sit outside the anonymous namespace: cppcheck 2.7 (Humble's, run
// in CI) reports a syntax error on a TEST_F that follows a definition inside one.

//=============================================================================
// Scenario files resolve against the registered node types
//=============================================================================

TEST_F(ScenarioTest, EveryScenarioXmlBuildsAgainstTheRegisteredNodes) {
  startWorld("yes");  // servers up, so the ROS nodes connect at construction
  std::vector<std::string> files;
  for (const auto & entry : std::filesystem::directory_iterator(dataPath(""))) {
    if (entry.path().extension() == ".xml") {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  ASSERT_FALSE(files.empty());

  for (const auto & file : files) {
    SCOPED_TRACE(file);
    try {
      auto tree = treeFromFile(file);
      EXPECT_NE(tree.rootNode(), nullptr);
    } catch (const std::exception & e) {
      ADD_FAILURE() << e.what();
    }
  }
}

// Shows the check above has teeth: a misspelt node or port does not build.
TEST_F(ScenarioTest, UnknownNodesAndPortsAreRejected) {
  params_.wait_for_server_timeout = milliseconds(10);
  EXPECT_THROW(treeFromText("<NoSuchNode/>"), std::exception);
  EXPECT_THROW(treeFromText(R"(<Gesture gesture_typo="wave"/>)"), std::exception);
  EXPECT_THROW(treeFromText(R"(<SelectExhibit exhibit_goal_heading="{h}"/>)"), std::exception);
}

TEST(ScenarioFiles, ShippedConfigurationNamesExistingFiles) {
  const YAML::Node params = shippedParameters();
  const auto scenario = params["scenario_specification"].as<std::string>();
  const auto culture_kb = params["culture_knowledge_base"].as<std::string>();
  const auto environment_kb = params["environment_knowledge_base"].as<std::string>();

  EXPECT_TRUE(behavior_controller::fileExists(dataPath(scenario + ".xml"))) << scenario;
  EXPECT_TRUE(behavior_controller::fileExists(dataPath(culture_kb))) << culture_kb;
  EXPECT_TRUE(behavior_controller::validateEnvironmentKnowledgeBase(dataPath(environment_kb)))
    << environment_kb;
}

// initializeTree() resolves the scenario in the installed share directory,
// as the node does at configure time.
TEST_F(ScenarioTest, InitializeTreeBuildsTheConfiguredScenario) {
  const YAML::Node params = shippedParameters();
  ASSERT_NO_FATAL_FAILURE(
    loadKnowledgeBase(params["environment_knowledge_base"].as<std::string>()));
  startWorld("yes");
  const auto scenario = params["scenario_specification"].as<std::string>();

  EXPECT_NO_THROW(behavior_controller::initializeTree(scenario, client_));
  EXPECT_THROW(
    behavior_controller::initializeTree("no_such_scenario", client_), std::runtime_error);
}

// The tours read these phrases but nothing in them writes them: the node
// puts the culture knowledge base's utility_phrases on the blackboard.
// Without them both tours fail at their first speech.
TEST_F(ScenarioTest, InitializeTreeProvidesTheTourPhrases) {
  ASSERT_NO_FATAL_FAILURE(loadKnowledgeBase("labEnvironmentKnowledgeBase.yaml"));
  startWorld("yes");
  auto tree = behavior_controller::initializeTree("lab_tour", client_);

  for (const char * key : kTourPhraseKeys) {
    std::string phrase;
    EXPECT_TRUE(tree.rootBlackboard()->get(key, phrase) && !phrase.empty())
      << "{" << key << "} is not set";
  }
}

//=============================================================================
// Tour simulations (lab_tour.xml, dec_Tour.xml)
//=============================================================================

namespace
{

struct Tour
{
  std::string xml;
  std::string environment_kb;
};

void PrintTo(const Tour & tour, std::ostream * os)
{
  *os << tour.xml << " with " << tour.environment_kb;
}

class TourTest : public ScenarioTest, public ::testing::WithParamInterface<Tour>
{
};

}  // namespace

TEST_P(TourTest, VisitorWhoAcceptsIsShownEveryExhibitInOrder) {
  ASSERT_NO_FATAL_FAILURE(loadKnowledgeBase(GetParam().environment_kb));
  startWorld("yes please");
  auto tree = treeFromFile(dataPath(GetParam().xml), behavior_controller::createRootBlackboard());
  StatusRecorder recorder(tree);

  ASSERT_EQ(tickUntilDone(tree, milliseconds(120000)), NodeStatus::SUCCESS)
    << "Leaf nodes that failed:\n" << recorder.failedLeaves()
    << "What the robot did:\n" << eventLog();

  // Drive to the start, to every exhibit in tour order, then to standby;
  // welcome, point at each exhibit, wave goodbye; at each exhibit say
  // "follow me" and then its description, if it has one.
  Events navigation{standby()};
  Events gestures{"gesture welcome"};
  Events speech{say("welcome_speech"), kTourQuestion};
  auto & knowledge = KnowledgeManager::instance();
  for (const auto & id : knowledge.getTourSpecification().locationIds) {
    const LocationInfo info = knowledge.getLocationInfo(id);
    navigation.push_back(
      describeNavigation(info.robotPose.x, info.robotPose.y, info.robotPose.theta));
    gestures.push_back(
      describePoint(info.gestureTarget.x, info.gestureTarget.y, info.gestureTarget.z));
    speech.push_back(say("follow_me_speech"));
    if (!info.gestureMessage.empty()) {
      speech.push_back("say " + info.gestureMessage);
    }
  }
  navigation.push_back(standby());
  gestures.push_back("gesture wave");
  speech.push_back(say("goodbye_speech"));

  EXPECT_EQ(events("navigate"), navigation);
  EXPECT_EQ(events("gesture"), gestures);
  EXPECT_EQ(events("say"), speech);
  EXPECT_TRUE(recorder.reached("FullGoodbye", NodeStatus::SUCCESS));
}

TEST_P(TourTest, VisitorWhoDeclinesGetsTheShortGoodbye) {
  ASSERT_NO_FATAL_FAILURE(loadKnowledgeBase(GetParam().environment_kb));
  startWorld("no thank you");
  auto tree = treeFromFile(dataPath(GetParam().xml), behavior_controller::createRootBlackboard());
  StatusRecorder recorder(tree);

  ASSERT_EQ(tickUntilDone(tree, milliseconds(60000)), NodeStatus::SUCCESS)
    << "Leaf nodes that failed:\n" << recorder.failedLeaves()
    << "What the robot did:\n" << eventLog();

  EXPECT_EQ(events("navigate"), Events{standby()});
  EXPECT_EQ(events("gesture"), (Events{"gesture welcome", "gesture wave"}));
  EXPECT_EQ(
    events("say"),
    (Events{say("welcome_speech"), kTourQuestion, say("maybe_another_time_speech"),
      say("brief_goodbye_speech")}));
  EXPECT_TRUE(recorder.reached("DeclinedGoodbye", NodeStatus::SUCCESS));
}

INSTANTIATE_TEST_SUITE_P(
  ShippedTours, TourTest,
  ::testing::Values(
    Tour{"lab_tour.xml", "labEnvironmentKnowledgeBase.yaml"},
    Tour{"dec_Tour.xml", "decEnvironmentKnowledgeBase.yaml"},
    Tour{"dec_Tour.xml", "decEnvironmentKnowledgeBase_short.yaml"}),
  [](const ::testing::TestParamInfo<Tour> & info) {
    const auto & tour = info.param;
    const std::string xml = tour.xml.substr(0, tour.xml.find('.'));
    return xml + "_" + tour.environment_kb.substr(0, tour.environment_kb.find('.'));
  });

// The node re-ticks the tree once a tour ends, so the next visitor gets a new
// tour on the same blackboard. A visitor who declined must not leave the next
// one, who accepts, with the short goodbye. The second tour also halts
// DetectVisitor's AnimateBehavior on the tick it sends its goal (the visitor
// is already in view): without patches/behaviortree_ros2-reset-goal-handle.patch
// that aborts the process.
TEST_F(ScenarioTest, NextTourStartsAfreshAfterADecline) {
  ASSERT_NO_FATAL_FAILURE(loadKnowledgeBase("labEnvironmentKnowledgeBase.yaml"));
  startWorld("no thank you");
  auto tree = treeFromFile(dataPath("lab_tour.xml"), behavior_controller::createRootBlackboard());
  {
    StatusRecorder first(tree);
    ASSERT_EQ(tickUntilDone(tree, milliseconds(60000)), NodeStatus::SUCCESS)
      << "Leaf nodes that failed:\n" << first.failedLeaves();
    ASSERT_TRUE(first.reached("DeclinedGoodbye", NodeStatus::SUCCESS));
  }

  setVisitorSays("yes please");
  StatusRecorder second(tree);
  ASSERT_EQ(tickUntilDone(tree, milliseconds(120000)), NodeStatus::SUCCESS)
    << "Leaf nodes that failed:\n" << second.failedLeaves()
    << "What the robot did:\n" << eventLog();
  EXPECT_TRUE(second.reached("FullGoodbye", NodeStatus::SUCCESS));
  EXPECT_FALSE(second.reached("DeclinedGoodbye", NodeStatus::SUCCESS));
  const auto tour_stops = KnowledgeManager::instance().getTourSpecification().locationIds.size();
  EXPECT_EQ(events("navigate").size(), 1 + (tour_stops + 2)) << eventLog();
}

//=============================================================================
// Other scenarios
//=============================================================================

TEST_F(ScenarioTest, AsrPipelineSpeaksTheConversationManagersAnswer) {
  startWorld("what is this robot", 2);
  auto tree = treeFromFile(dataPath("asr_cm_tts_pipeline.xml"));

  runUntil(tree, [this]() {return events("recognize").size() >= 3;});
  EXPECT_EQ(
    events(),
    (Events{"recognize", "ask what is this robot", "say Answer to: what is this robot",
      "recognize", "ask what is this robot", "say Answer to: what is this robot",
      "recognize"}));
}

TEST_F(ScenarioTest, AsrPipelinePromptsWhenNothingWasHeard) {
  startWorld("", 1);  // speech recognition returns an empty transcription
  auto tree = treeFromFile(dataPath("asr_cm_tts_pipeline.xml"));

  runUntil(tree, [this]() {return events("recognize").size() >= 2;});
  EXPECT_EQ(events(), (Events{"recognize", "say Any questions?", "recognize"}));
}

TEST_F(ScenarioTest, ListenPipelineMutesTheMicWhileTheRobotSpeaks) {
  startWorld("tell me about the lab", 2);
  auto tree = treeFromFile(dataPath("listen_cm_tts_pipeline.xml"));

  runUntil(tree, [this]() {return events("mic on").size() >= 2;});
  const Events turn{"mic off", "ask tell me about the lab",
    "say Answer to: tell me about the lab", "mic on"};
  Events two_turns = turn;
  two_turns.insert(two_turns.end(), turn.begin(), turn.end());
  EXPECT_EQ(events(), two_turns);
}

TEST_F(ScenarioTest, DeicticTestDrivesToAndPointsAtEveryExhibit) {
  ASSERT_NO_FATAL_FAILURE(loadKnowledgeBase("labEnvironmentKnowledgeBase.yaml"));
  startWorld("yes");
  auto tree = treeFromFile(dataPath("dec_DeicticTest.xml"));
  StatusRecorder recorder(tree);

  ASSERT_EQ(tickUntilDone(tree, milliseconds(60000)), NodeStatus::SUCCESS)
    << "Leaf nodes that failed:\n" << recorder.failedLeaves()
    << "What the robot did:\n" << eventLog();

  Events expected;
  auto & knowledge = KnowledgeManager::instance();
  for (const auto & id : knowledge.getTourSpecification().locationIds) {
    const LocationInfo info = knowledge.getLocationInfo(id);
    expected.push_back(
      describeNavigation(info.robotPose.x, info.robotPose.y, info.robotPose.theta));
    expected.push_back(
      describePoint(info.gestureTarget.x, info.gestureTarget.y, info.gestureTarget.z));
  }
  EXPECT_EQ(events(), expected);
}

TEST_F(ScenarioTest, GestureNavTestRunsItsFixedSequence) {
  startWorld("yes");
  auto tree = treeFromFile(dataPath("dec_GestureNavTest.xml"));
  StatusRecorder recorder(tree);

  ASSERT_EQ(tickUntilDone(tree, milliseconds(30000)), NodeStatus::SUCCESS)
    << "Leaf nodes that failed:\n" << recorder.failedLeaves();
  EXPECT_EQ(
    events(),
    (Events{"gesture welcome", "say I will now navigate to the exhibit.", standby(),
      describePoint(5.0, 6.8, 1.2), "say This is the exhibit.",
      describeNavigation(0.0, 0.0, 0.0), "gesture wave"}));
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
