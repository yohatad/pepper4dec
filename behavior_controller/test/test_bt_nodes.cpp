/* test_bt_nodes.cpp
 *
 * Tests each custom BehaviorTree node class on its own, ticked in a small
 * tree against stub servers (see bt_test_harness.hpp):
 *
 *   - ROS action wrappers (BT::RosActionNode): the goal each builds from its
 *     ports, the outputs it writes from feedback and results, how it maps
 *     success, failure, abort, rejection and a missing server to a node
 *     status, and that halting it cancels its goal;
 *   - ROS service wrappers (BT::RosServiceNode): the request built from the
 *     ports and the response mapped to a status;
 *   - topic-driven stateful nodes: what they wait for on /face_detection/data
 *     and /speech_event/text, their timeouts, and that speech heard before
 *     they start is ignored;
 *   - tour-loop and blackboard nodes: walking the exhibit queue of the
 *     knowledge base in test/fixtures.
 *
 * KnowledgeManager is a process-wide singleton; the tests that need it load
 * the same fixture, so their order does not matter.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: October 11, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "bt_test_harness.hpp"

namespace
{

using bt_test::BtRosTest;
using bt_test::Outcome;
using bt_test::Reply;
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

const char kFixtureDir[] = TEST_FIXTURE_DIR;

/// Reply for the dec_interfaces actions whose result has success + message.
template<typename ActionT>
Reply<ActionT> withMessage(bool success, const std::string & message)
{
  Reply<ActionT> reply;
  reply.result.success = success;
  reply.result.message = message;
  return reply;
}

void loadFixtureKnowledgeBase()
{
  ASSERT_TRUE(
    ConfigManager::instance().loadFromFile(std::string(kFixtureDir) + "/config_good.yaml"));
  ASSERT_TRUE(KnowledgeManager::instance().loadFromPackage(kFixtureDir));
}

}  // namespace

// The tests sit outside the anonymous namespace: cppcheck 2.7 (Humble's, run
// in CI) reports a syntax error on a TEST_F that follows a definition inside one.

//=============================================================================
// ROS action wrappers
//=============================================================================

TEST_F(BtRosTest, AnimateBehaviorSendsPortsAndWritesFeedback) {
  auto server = serveAction<AnimateBehavior>(
    "/animate_behavior", [](const AnimateBehavior::Goal &) {
      auto reply = withMessage<AnimateBehavior>(true, "animation done");
      AnimateBehavior::Feedback feedback;
      feedback.current_limb = "left_arm";
      feedback.gestures_completed = 3;
      feedback.is_running = true;
      reply.feedback = feedback;
      reply.delay = milliseconds(300);  // feedback lands well before the result
      return reply;
    });
  auto tree = treeFromText(
    R"(<AnimateBehavior behavior_type="hands" selected_range="0.8" duration_seconds="5"
         message="{message}" current_limb="{limb}" gestures_completed="{count}"/>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  const auto goals = server->goals();
  ASSERT_EQ(goals.size(), 1u);
  EXPECT_EQ(goals[0].behavior_type, "hands");
  EXPECT_FLOAT_EQ(goals[0].selected_range, 0.8f);
  EXPECT_EQ(goals[0].duration_seconds, 5);
  auto bb = tree.rootBlackboard();
  EXPECT_EQ(bb->get<std::string>("message"), "animation done");
  EXPECT_EQ(bb->get<std::string>("limb"), "left_arm");
  EXPECT_EQ(bb->get<int>("count"), 3);
}

TEST_F(BtRosTest, AnimateBehaviorFailsWhenServerReportsFailure) {
  serveAction<AnimateBehavior>(
    "/animate_behavior", [](const AnimateBehavior::Goal &) {
      return withMessage<AnimateBehavior>(false, "motors stiff");
    });
  auto tree = treeFromText(R"(<AnimateBehavior message="{message}"/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::FAILURE);
  EXPECT_EQ(tree.rootBlackboard()->get<std::string>("message"), "motors stiff");
}

// The tour runs AnimateBehavior (duration 0 = until canceled) beside another
// branch in a Parallel; when that branch finishes, the Parallel halts it.
TEST_F(BtRosTest, HaltingAnimateBehaviorCancelsItsGoal) {
  auto server = serveAction<AnimateBehavior>(
    "/animate_behavior", [](const AnimateBehavior::Goal &) {
      Reply<AnimateBehavior> reply;
      reply.outcome = Outcome::kRunUntilCanceled;
      return reply;
    });
  auto tree = treeFromText(
    R"(<Parallel success_count="1" failure_count="1">
         <AnimateBehavior duration_seconds="0"/>
         <Sleep msec="300"/>
       </Parallel>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  EXPECT_EQ(server->goals().size(), 1u);
  EXPECT_TRUE(waitFor([&]() {return server->canceledCount() == 1;}));
}

// The tours' DetectVisitor runs AnimateBehavior beside IsVisitorDiscovered in
// a Parallel. With a visitor already in view, IsVisitorDiscovered succeeds on
// the first tick and the Parallel halts AnimateBehavior before the server has
// answered its goal. On the node's second run (the next tour, as the node
// re-ticks the tree after it succeeds), BT::RosActionNode still holds the
// goal handle of the first run, and cancelGoal() asks rclcpp_action for that
// finished goal's result, which throws UnknownGoalHandleError out of halt().
// Run as a death test: if halt() throws, only this test fails. Swallowing
// that error is not enough: the second goal must be canceled too, or the
// robot keeps animating.
TEST_F(BtRosTest, HaltingAnActionNodeBeforeItsSecondGoalIsAnsweredIsSafe) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  EXPECT_EXIT(
  {
    auto server = serveAction<AnimateBehavior>(
      "/animate_behavior", [](const AnimateBehavior::Goal &) {
        Reply<AnimateBehavior> reply;
        reply.outcome = Outcome::kRunUntilCanceled;
        return reply;
      });
    auto tree = treeFromText(
      R"(<Parallel success_count="1" failure_count="1">
           <AnimateBehavior duration_seconds="0"/>
           <AlwaysSuccess/>
         </Parallel>)");
    for (int run = 0; run < 2; ++run) {
      if (tickUntilDone(tree) != NodeStatus::SUCCESS) {
        std::_Exit(1);
      }
    }
    const bool both_canceled = waitFor([&]() {return server->canceledCount() == 2;});
    std::_Exit(server->goals().size() == 2u && both_canceled ? 0 : 1);
  }, ::testing::ExitedWithCode(0), "");
}

TEST_F(BtRosTest, GestureSendsEveryPort) {
  auto server = serveAction<Gesture>(
    "/gesture_execution", [](const Gesture::Goal &) {
      return withMessage<Gesture>(true, "pointed");
    });
  auto tree = treeFromText(
    R"(<Gesture gesture_type="deictic" gesture_name="exhibit" gesture_duration="3000"
         bow_nod_angle="15" location_x="1.5" location_y="-2.0" location_z="0.75"
         message="{message}"/>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  const auto goals = server->goals();
  ASSERT_EQ(goals.size(), 1u);
  EXPECT_EQ(goals[0].gesture_type, "deictic");
  EXPECT_EQ(goals[0].gesture_name, "exhibit");
  EXPECT_EQ(goals[0].gesture_duration, 3000);
  EXPECT_EQ(goals[0].bow_nod_angle, 15);
  EXPECT_DOUBLE_EQ(goals[0].location_x, 1.5);
  EXPECT_DOUBLE_EQ(goals[0].location_y, -2.0);
  EXPECT_DOUBLE_EQ(goals[0].location_z, 0.75);
  EXPECT_EQ(tree.rootBlackboard()->get<std::string>("message"), "pointed");
}

TEST_F(BtRosTest, GestureFailsWhenServerReportsFailure) {
  serveAction<Gesture>(
    "/gesture_execution", [](const Gesture::Goal &) {
      return withMessage<Gesture>(false, "unknown gesture");
    });
  auto tree = treeFromText(R"(<Gesture gesture_type="wave"/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::FAILURE);
}

TEST_F(BtRosTest, NavigateBuildsMapPoseFromGoalPorts) {
  auto server = serveAction<NavigateToPose>(
    "/navigate_to_pose", [](const NavigateToPose::Goal &) {
      return Reply<NavigateToPose>();
    });
  auto tree = treeFromText(R"(<Navigate goal_x="1.5" goal_y="-2.0" goal_theta="1.5708"/>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  const auto goals = server->goals();
  ASSERT_EQ(goals.size(), 1u);
  const auto & pose = goals[0].pose;
  EXPECT_EQ(pose.header.frame_id, "map");
  EXPECT_DOUBLE_EQ(pose.pose.position.x, 1.5);
  EXPECT_DOUBLE_EQ(pose.pose.position.y, -2.0);
  EXPECT_DOUBLE_EQ(pose.pose.position.z, 0.0);
  // goal_theta is a heading in radians: yaw-only quaternion.
  EXPECT_DOUBLE_EQ(pose.pose.orientation.x, 0.0);
  EXPECT_DOUBLE_EQ(pose.pose.orientation.y, 0.0);
  EXPECT_NEAR(pose.pose.orientation.z, std::sin(1.5708 / 2.0), 1e-12);
  EXPECT_NEAR(pose.pose.orientation.w, std::cos(1.5708 / 2.0), 1e-12);
  EXPECT_TRUE(goals[0].behavior_tree.empty());
}

TEST_F(BtRosTest, NavigateFailsWhenNav2Aborts) {
  serveAction<NavigateToPose>(
    "/navigate_to_pose", [](const NavigateToPose::Goal &) {
      Reply<NavigateToPose> reply;
      reply.outcome = Outcome::kAbort;
      return reply;
    });
  auto tree = treeFromText(R"(<Navigate goal_x="1.0" goal_y="1.0"/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::FAILURE);
}

TEST_F(BtRosTest, NavigateWithoutGoalSendsNothing) {
  auto server = serveAction<NavigateToPose>(
    "/navigate_to_pose", [](const NavigateToPose::Goal &) {
      return Reply<NavigateToPose>();
    });
  auto tree = treeFromText(R"(<Navigate goal_y="1.0"/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::FAILURE);
  EXPECT_TRUE(server->goals().empty());
}

TEST_F(BtRosTest, SpeechRecognitionWritesTranscription) {
  auto server = serveAction<SpeechRecognition>(
    "/speech_recognition", [](const SpeechRecognition::Goal &) {
      Reply<SpeechRecognition> reply;
      reply.result.transcription = "where is the robot lab";
      return reply;
    });
  auto tree = treeFromText(R"(<SpeechRecognition wait="3.5" transcription="{heard}"/>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  ASSERT_EQ(server->goals().size(), 1u);
  EXPECT_FLOAT_EQ(server->goals()[0].wait, 3.5f);
  EXPECT_EQ(tree.rootBlackboard()->get<std::string>("heard"), "where is the robot lab");
}

TEST_F(BtRosTest, SpeechRecognitionFailsWhenNothingWasHeard) {
  serveAction<SpeechRecognition>(
    "/speech_recognition", [](const SpeechRecognition::Goal &) {
      return Reply<SpeechRecognition>();  // empty transcription
    });
  auto tree = treeFromText(R"(<SpeechRecognition transcription="{heard}"/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::FAILURE);
}

TEST_F(BtRosTest, ConversationManagerWritesResponseIntentAndConfidence) {
  auto server = serveAction<ConversationManager>(
    "/conversation_manager", [](const ConversationManager::Goal &) {
      Reply<ConversationManager> reply;
      reply.result.success = true;
      reply.result.response = "It is a humanoid robot.";
      reply.result.intent = "ASK_EXHIBIT_QUESTION";
      reply.result.confidence = 0.9f;
      return reply;
    });
  auto tree = treeFromText(
    R"(<ConversationManager prompt="what is this" response="{response}" intent="{intent}"
         confidence="{confidence}"/>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  ASSERT_EQ(server->goals().size(), 1u);
  EXPECT_EQ(server->goals()[0].prompt, "what is this");
  auto bb = tree.rootBlackboard();
  EXPECT_EQ(bb->get<std::string>("response"), "It is a humanoid robot.");
  EXPECT_EQ(bb->get<std::string>("intent"), "ASK_EXHIBIT_QUESTION");
  EXPECT_NEAR(bb->get<double>("confidence"), 0.9, 1e-6);
}

TEST_F(BtRosTest, ConversationManagerWithEmptyPromptSendsNothing) {
  auto server = serveAction<ConversationManager>(
    "/conversation_manager", [](const ConversationManager::Goal &) {
      return Reply<ConversationManager>();
    });
  auto tree = treeFromText(R"(<ConversationManager prompt=""/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::FAILURE);
  EXPECT_TRUE(server->goals().empty());
}

TEST_F(BtRosTest, SpeechWithFeedbackSaysTextInContextualModeByDefault) {
  auto server = serveAction<SpeechWithFeedback>(
    "/naoqi_driver/naoqi_driver/speech_with_feedback", [](const SpeechWithFeedback::Goal &) {
      Reply<SpeechWithFeedback> reply;
      reply.result.success = true;
      return reply;
    });
  auto tree = treeFromText(
    R"(<Sequence>
         <SpeechWithFeedback say="Hello there"/>
         <SpeechWithFeedback say="Quietly now" body_language_mode="disabled"/>
       </Sequence>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  const auto goals = server->goals();
  ASSERT_EQ(goals.size(), 2u);
  EXPECT_EQ(goals[0].say, "Hello there");
  EXPECT_EQ(goals[0].body_language_mode, "contextual");
  EXPECT_EQ(goals[1].say, "Quietly now");
  EXPECT_EQ(goals[1].body_language_mode, "disabled");
}

// The environment knowledge base validator accepts an empty
// gesture_message_english, but the node refuses to say nothing.
TEST_F(BtRosTest, SpeechWithFeedbackRefusesEmptyText) {
  auto server = serveAction<SpeechWithFeedback>(
    "/naoqi_driver/naoqi_driver/speech_with_feedback", [](const SpeechWithFeedback::Goal &) {
      Reply<SpeechWithFeedback> reply;
      reply.result.success = true;
      return reply;
    });
  auto tree = treeFromText(R"(<SpeechWithFeedback say=""/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::FAILURE);
  EXPECT_TRUE(server->goals().empty());
}

TEST_F(BtRosTest, SpeechWithFeedbackFailsWhenNaoqiReportsFailure) {
  serveAction<SpeechWithFeedback>(
    "/naoqi_driver/naoqi_driver/speech_with_feedback", [](const SpeechWithFeedback::Goal &) {
      return Reply<SpeechWithFeedback>();  // success = false
    });
  auto tree = treeFromText(R"(<SpeechWithFeedback say="Hello"/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::FAILURE);
}

TEST_F(BtRosTest, TtsSendsTextAndMapsResult) {
  auto server = serveAction<TTS>(
    "/text_to_speech", [](const TTS::Goal & goal) {
      return withMessage<TTS>(goal.text != "fail", "spoken");
    });
  auto tree = treeFromText(
    R"(<Sequence>
         <TTS text="Welcome to the lab" message="{message}"/>
         <Inverter><TTS text="fail"/></Inverter>
       </Sequence>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  ASSERT_EQ(server->goals().size(), 2u);
  EXPECT_EQ(server->goals()[0].text, "Welcome to the lab");
  EXPECT_EQ(tree.rootBlackboard()->get<std::string>("message"), "spoken");
}

TEST_F(BtRosTest, IsVisitorResponseYesFollowsTheClassifiedAnswer) {
  auto server = serveAction<ConversationManager>(
    "/conversation_manager", [](const ConversationManager::Goal & goal) {
      Reply<ConversationManager> reply;
      reply.result.success = true;
      // The conversation manager answers "yes"/"no"; case must not matter.
      reply.result.response = goal.prompt.rfind("yes", 0) == 0 ? "Yes" : "no";
      return reply;
    });

  for (const auto & [utterance, expected] : std::vector<std::pair<std::string, NodeStatus>>{
    {"yes please", NodeStatus::SUCCESS}, {"no thank you", NodeStatus::FAILURE}})
  {
    SCOPED_TRACE(utterance);
    auto tree = treeFromText(R"(<IsVisitorResponseYes visitor_response="{visitor_response}"/>)");
    tree.rootBlackboard()->set("visitor_response", utterance);
    EXPECT_EQ(tickUntilDone(tree), expected);
    EXPECT_EQ(server->goals().back().prompt, utterance);
    // The classified answer, lowercase, for the tours to tell "no" from unclear.
    std::string answer;
    ASSERT_TRUE(tree.rootBlackboard()->get("visitor_answer", answer));
    EXPECT_EQ(answer, expected == NodeStatus::SUCCESS ? "yes" : "no");
  }
}

TEST_F(BtRosTest, ActionNodeFailsWhenItsServerIsMissing) {
  params_.wait_for_server_timeout = milliseconds(100);
  auto tree = treeFromText(R"(<Gesture gesture_type="wave"/>)");

  EXPECT_EQ(tickUntilDone(tree, milliseconds(3000)), NodeStatus::FAILURE);
}

TEST_F(BtRosTest, ActionNodeFailsWhenTheServerRejectsTheGoal) {
  auto server = serveAction<Gesture>(
    "/gesture_execution", [](const Gesture::Goal &) {
      return withMessage<Gesture>(true, "");
    });
  server->rejectGoals();
  auto tree = treeFromText(R"(<Gesture gesture_type="wave"/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::FAILURE);
}

//=============================================================================
// ROS service wrappers
//=============================================================================

TEST_F(BtRosTest, StopAnimateBehaviorMapsTheResponse) {
  auto server = serveService<Trigger>(
    "/animate_behavior/stop", [](const Trigger::Request &, Trigger::Response & response) {
      response.success = true;
      response.message = "stopped";
    });
  auto tree = treeFromText(R"(<StopAnimateBehavior message="{message}"/>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  EXPECT_EQ(server->requests().size(), 1u);
  EXPECT_EQ(tree.rootBlackboard()->get<std::string>("message"), "stopped");

  server->setHandler(
    [](const Trigger::Request &, Trigger::Response & response) {
      response.success = false;
      response.message = "nothing running";
    });
  auto failing = treeFromText(R"(<StopAnimateBehavior/>)");
  EXPECT_EQ(tickUntilDone(failing), NodeStatus::FAILURE);
}

TEST_F(BtRosTest, SetOvertAttentionSendsTheEnabledFlag) {
  auto server = serveService<SetBool>("/overt_attention/set_enabled");
  auto tree = treeFromText(
    R"(<Sequence>
         <SetOvertAttention enabled="false"/>
         <SetOvertAttention enabled="true" message="{message}"/>
       </Sequence>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  const auto requests = server->requests();
  ASSERT_EQ(requests.size(), 2u);
  EXPECT_FALSE(requests[0].data);
  EXPECT_TRUE(requests[1].data);
  EXPECT_EQ(tree.rootBlackboard()->get<std::string>("message"), "ok");
}

TEST_F(BtRosTest, SetSpeechListeningSendsTheFlagAndMapsFailure) {
  auto server = serveService<SetBool>(
    "/speech_event/set_enabled",
    [](const SetBool::Request & request, SetBool::Response & response) {
      response.success = !request.data;  // muting works, unmuting fails
    });
  auto tree = treeFromText(
    R"(<Sequence>
         <SetSpeechListening enabled="false"/>
         <Inverter><SetSpeechListening enabled="true"/></Inverter>
       </Sequence>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  const auto requests = server->requests();
  ASSERT_EQ(requests.size(), 2u);
  EXPECT_FALSE(requests[0].data);
  EXPECT_TRUE(requests[1].data);
}

TEST_F(BtRosTest, ServiceNodeFailsWhenItsServiceIsMissing) {
  params_.wait_for_server_timeout = milliseconds(100);
  auto tree = treeFromText(R"(<SetOvertAttention enabled="true"/>)");

  EXPECT_EQ(tickUntilDone(tree, milliseconds(3000)), NodeStatus::FAILURE);
}

//=============================================================================
// Topic-driven stateful nodes
//=============================================================================

TEST_F(BtRosTest, CheckFaceDetectedWaitsForMutualGaze) {
  auto gaze = std::make_shared<std::atomic<bool>>(false);
  publishEvery<FaceDetection>(
    "/face_detection/data", milliseconds(50),
    [gaze]() {return bt_test::faces({false, gaze->load()});});
  auto tree = treeFromText(
    R"(<CheckFaceDetected require_mutual_gaze="true" face_count="{count}"
         mutual_gaze="{gaze}" face_id="{id}" face_depth="{depth}"/>)");

  EXPECT_EQ(tickFor(tree, milliseconds(400)), NodeStatus::RUNNING);
  *gaze = true;
  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  auto bb = tree.rootBlackboard();
  EXPECT_EQ(bb->get<int>("count"), 2);
  EXPECT_TRUE(bb->get<bool>("gaze"));
  EXPECT_EQ(bb->get<std::string>("id"), "face_0");
  EXPECT_DOUBLE_EQ(bb->get<double>("depth"), 1.5);
}

TEST_F(BtRosTest, CheckFaceDetectedWaitsWhileNoFaceIsVisible) {
  auto visible = std::make_shared<std::atomic<bool>>(false);
  publishEvery<FaceDetection>(
    "/face_detection/data", milliseconds(50),
    [visible]() {
      return visible->load() ? bt_test::faces({false}) : bt_test::faces({});
    });
  auto tree = treeFromText(R"(<CheckFaceDetected require_mutual_gaze="false"/>)");

  EXPECT_EQ(tickFor(tree, milliseconds(400)), NodeStatus::RUNNING);
  *visible = true;
  EXPECT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
}

TEST_F(BtRosTest, IsVisitorDiscoveredSucceedsWhenAFaceAppears) {
  publishEvery<FaceDetection>(
    "/face_detection/data", milliseconds(50), []() {return bt_test::faces({false});});
  auto tree = treeFromText(R"(<IsVisitorDiscovered timeout="5.0"/>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
}

TEST_F(BtRosTest, IsVisitorDiscoveredTimesOutWithoutAFace) {
  publishEvery<FaceDetection>(
    "/face_detection/data", milliseconds(50), []() {return bt_test::faces({});});
  auto tree = treeFromText(R"(<IsVisitorDiscovered timeout="0.3"/>)");

  EXPECT_EQ(tickUntilDone(tree, milliseconds(3000)), NodeStatus::FAILURE);
}

TEST_F(BtRosTest, IsMutualGazeDiscoveredNeedsSustainedGaze) {
  publishEvery<FaceDetection>(
    "/face_detection/data", milliseconds(50), []() {return bt_test::faces({true});});
  auto tree = treeFromText(R"(<IsMutualGazeDiscovered timeout="5.0" min_duration="0.4"/>)");

  const auto start = std::chrono::steady_clock::now();
  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  EXPECT_GE(std::chrono::steady_clock::now() - start, milliseconds(400));
}

TEST_F(BtRosTest, IsMutualGazeDiscoveredTimesOutWithoutGaze) {
  publishEvery<FaceDetection>(
    "/face_detection/data", milliseconds(50), []() {return bt_test::faces({false});});
  auto tree = treeFromText(R"(<IsMutualGazeDiscovered timeout="0.3" min_duration="0.1"/>)");

  EXPECT_EQ(tickUntilDone(tree, milliseconds(3000)), NodeStatus::FAILURE);
}

TEST_F(BtRosTest, GetVisitorResponseIgnoresSpeechFromBeforeItStarted) {
  auto tree = treeFromText(
    R"(<GetVisitorResponse timeout="5.0" visitor_response="{visitor_response}"/>)");
  auto speech = connectedPublisher<StringMsg>("/speech_event/text");
  StringMsg msg;
  msg.data = "stale words";
  speech->publish(msg);
  std::this_thread::sleep_for(milliseconds(300));  // delivered before the node starts

  EXPECT_EQ(tickFor(tree, milliseconds(300)), NodeStatus::RUNNING);
  msg.data = "yes please";
  speech->publish(msg);
  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  EXPECT_EQ(tree.rootBlackboard()->get<std::string>("visitor_response"), "yes please");
}

TEST_F(BtRosTest, GetVisitorResponseTimesOutWithoutSpeech) {
  auto tree = treeFromText(R"(<GetVisitorResponse timeout="0.3"/>)");

  EXPECT_EQ(tickUntilDone(tree, milliseconds(3000)), NodeStatus::FAILURE);
}

TEST_F(BtRosTest, ListenForSpeechIgnoresSpeechFromBeforeItStarted) {
  auto tree = treeFromText(R"(<ListenForSpeech transcription="{heard}"/>)");
  auto speech = connectedPublisher<StringMsg>("/speech_event/text");
  StringMsg msg;
  msg.data = "stale words";
  speech->publish(msg);
  std::this_thread::sleep_for(milliseconds(300));

  EXPECT_EQ(tickFor(tree, milliseconds(300)), NodeStatus::RUNNING);
  msg.data = "tell me about the lab";
  speech->publish(msg);
  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  EXPECT_EQ(tree.rootBlackboard()->get<std::string>("heard"), "tell me about the lab");
}

//=============================================================================
// Tour-loop and blackboard nodes (no ROS traffic; fixture knowledge base:
// tour [1, booth])
//=============================================================================

TEST_F(BtRosTest, TourLoopNodesWalkTheExhibitQueueInOrder) {
  ASSERT_NO_FATAL_FAILURE(loadFixtureKnowledgeBase());
  auto tree = treeFromText(
    R"(<Sequence>
         <RetrieveListOfExhibits exhibit_list="{exhibit_queue}"/>
         <IsListWithExhibit/>
         <SelectExhibit exhibit_speech="{first_speech}" exhibit_goal_x="{first_x}"
           exhibit_goal_y="{first_y}" exhibit_location_x="{first_target_x}"
           exhibit_location_y="{first_target_y}" exhibit_location_z="{first_target_z}"/>
         <PopExhibitFromList remaining_count="{after_first}"/>
         <IsListWithExhibit/>
         <SelectExhibit exhibit_goal_x="{second_x}" exhibit_goal_y="{second_y}"
           exhibit_location_x="{second_target_x}"/>
         <PopExhibitFromList remaining_count="{after_second}"/>
         <Inverter><IsListWithExhibit/></Inverter>
       </Sequence>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  auto bb = tree.rootBlackboard();
  EXPECT_EQ(bb->get<std::string>("first_speech"), "This is the test start.");
  EXPECT_DOUBLE_EQ(bb->get<double>("first_x"), 4.0);
  EXPECT_DOUBLE_EQ(bb->get<double>("first_y"), 0.5);
  EXPECT_DOUBLE_EQ(bb->get<double>("first_target_x"), 1.0);
  EXPECT_DOUBLE_EQ(bb->get<double>("first_target_y"), 2.0);
  EXPECT_DOUBLE_EQ(bb->get<double>("first_target_z"), 0.75);
  EXPECT_EQ(bb->get<int>("after_first"), 1);
  EXPECT_DOUBLE_EQ(bb->get<double>("second_x"), -1.5);
  EXPECT_DOUBLE_EQ(bb->get<double>("second_y"), 8.0);
  EXPECT_DOUBLE_EQ(bb->get<double>("second_target_x"), 6.0);
  EXPECT_EQ(bb->get<int>("after_second"), 0);
  EXPECT_TRUE(bb->get<std::vector<std::string>>("exhibit_queue").empty());
}

// The knowledge base gives robot_location_pose.theta in degrees (the
// validator enforces [0, 360]); exhibit_goal_theta feeds Navigate's
// goal_theta, which is in radians.
TEST_F(BtRosTest, SelectExhibitGivesTheHeadingInRadians) {
  ASSERT_NO_FATAL_FAILURE(loadFixtureKnowledgeBase());
  auto tree = treeFromText(
    R"(<Sequence>
         <RetrieveListOfExhibits exhibit_list="{exhibit_queue}"/>
         <SelectExhibit exhibit_goal_theta="{theta}"/>
       </Sequence>)");

  ASSERT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
  // Fixture location 1: theta 90.0 degrees.
  EXPECT_NEAR(tree.rootBlackboard()->get<double>("theta"), M_PI / 2.0, 1e-9);
}

TEST_F(BtRosTest, BlackboardNodesSetAndCompareStrings) {
  auto tree = treeFromText(
    R"(<Sequence>
         <SetBlackboardValue key="tour_declined" value="true"/>
         <CheckBlackboard key="tour_declined" expected="true"/>
         <Inverter><CheckBlackboard key="tour_declined" expected="false"/></Inverter>
         <Inverter><CheckBlackboard key="never_set" expected="true"/></Inverter>
       </Sequence>)");

  EXPECT_EQ(tickUntilDone(tree), NodeStatus::SUCCESS);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
