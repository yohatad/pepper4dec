// Round-trip every dec_interfaces type through rclcpp's CDR serializer.
//
// Each type is filled with non-default values (negative numbers, extremes,
// non-ASCII text, long and empty arrays), serialized to bytes, deserialized
// into a fresh object, and compared field by field with the original. A
// .msg/.srv/.action change that breaks the generated code, or a field whose
// value does not survive the wire, fails here instead of on the robot.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "dec_interfaces/action/animate_behavior.hpp"
#include "dec_interfaces/action/conversation_manager.hpp"
#include "dec_interfaces/action/gesture.hpp"
#include "dec_interfaces/action/speech_recognition.hpp"
#include "dec_interfaces/action/tts.hpp"
#include "dec_interfaces/msg/face_detection.hpp"
#include "dec_interfaces/msg/person_detection.hpp"
#include "dec_interfaces/srv/get_depth_roi.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "rclcpp/serialization.hpp"
#include "rclcpp/serialized_message.hpp"

namespace
{

// Non-ASCII on purpose: strings are UTF-8 on the wire and must survive as-is.
const char kText[] = "Selam, Pepper! \xE1\x88\xB0\xE1\x88\x8B\xE1\x88\x9D";

template<typename T>
T roundTrip(const T & in)
{
  rclcpp::Serialization<T> serializer;
  rclcpp::SerializedMessage wire;
  serializer.serialize_message(&in, &wire);
  EXPECT_GT(wire.size(), 0u);
  T out;
  serializer.deserialize_message(&wire, &out);
  return out;
}

geometry_msgs::msg::Point point(double x, double y, double z)
{
  geometry_msgs::msg::Point p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}

}  // namespace

// ---------------------------------------------------------------- messages

TEST(Serialization, FaceDetection)
{
  dec_interfaces::msg::FaceDetection m;
  m.face_label_id = {"face_1", "face_2_f2", ""};
  m.centroids = {point(1.5, -2.25, 0.0), point(-0.0, 1e-9, 1e9), point(0, 0, 3.75)};
  m.width = {0.5f, 120.0f, std::numeric_limits<float>::max()};
  m.height = {0.25f, -1.0f, std::numeric_limits<float>::lowest()};
  m.mutual_gaze = {true, false, true};
  EXPECT_EQ(roundTrip(m), m);
}

TEST(Serialization, FaceDetectionEmpty)
{
  dec_interfaces::msg::FaceDetection m;
  EXPECT_EQ(roundTrip(m), m);
}

TEST(Serialization, PersonDetection)
{
  dec_interfaces::msg::PersonDetection m;
  m.person_label_id = {"7", "12"};
  m.class_names = {"person", kText};
  m.class_ids = {0, std::numeric_limits<int32_t>::min()};
  m.confidences = {0.97f, 0.0f};
  m.centroids = {point(0.1, 0.2, 2.4), point(-3.0, 4.0, 0.0)};
  m.width = {64.0f, 1.0f};
  m.height = {128.0f, 2.0f};
  EXPECT_EQ(roundTrip(m), m);
}

TEST(Serialization, PersonDetectionLargeArrays)
{
  dec_interfaces::msg::PersonDetection m;
  for (int i = 0; i < 1000; ++i) {
    m.person_label_id.push_back(std::to_string(i));
    m.class_ids.push_back(i);
    m.confidences.push_back(static_cast<float>(i) / 1000.0f);
  }
  EXPECT_EQ(roundTrip(m), m);
}

// ---------------------------------------------------------------- services

TEST(Serialization, GetDepthROI)
{
  dec_interfaces::srv::GetDepthROI::Request req;
  req.x = -1;
  req.y = std::numeric_limits<int32_t>::max();
  req.width = 640;
  req.height = 480;
  req.points_x = {1, 2, 3};
  req.points_y = {4, 5, 6};
  req.rois_x = {10};
  req.rois_y = {20};
  req.rois_width = {30};
  req.rois_height = {40};
  EXPECT_EQ(roundTrip(req), req);

  dec_interfaces::srv::GetDepthROI::Response res;
  res.success = true;
  res.message = kText;
  res.depth_values = {0.5f, 1.25f, std::numeric_limits<float>::infinity()};
  res.roi_width = 3;
  res.roi_height = 1;
  res.min_depth = 0.5f;
  res.max_depth = 1.25f;
  res.mean_depth = 0.875f;
  res.point_depths = {0.1f, 0.2f, 0.3f};
  res.roi_mean_depths = {0.4f};
  res.roi_min_depths = {0.3f};
  res.roi_max_depths = {0.5f};
  EXPECT_EQ(roundTrip(res), res);
}

// ----------------------------------------------------------------- actions

TEST(Serialization, AnimateBehavior)
{
  using A = dec_interfaces::action::AnimateBehavior;
  A::Goal goal;
  goal.behavior_type = "hands";
  goal.selected_range = 0.75f;
  goal.duration_seconds = 0;
  EXPECT_EQ(roundTrip(goal), goal);

  A::Result result;
  result.success = true;
  result.message = kText;
  result.total_duration = 12.5f;
  EXPECT_EQ(roundTrip(result), result);

  A::Feedback feedback;
  feedback.current_limb = "RArm";
  feedback.gestures_completed = 42;
  feedback.elapsed_time = 3.0f;
  feedback.is_running = true;
  EXPECT_EQ(roundTrip(feedback), feedback);
}

TEST(Serialization, ConversationManager)
{
  using A = dec_interfaces::action::ConversationManager;
  A::Goal goal;
  goal.prompt = kText;
  EXPECT_EQ(roundTrip(goal), goal);

  A::Result result;
  result.success = false;
  result.response = "I do not know.";
  result.intent = "unknown";
  result.confidence = 0.125f;
  EXPECT_EQ(roundTrip(result), result);

  A::Feedback feedback;
  feedback.status = "retrieving";
  EXPECT_EQ(roundTrip(feedback), feedback);
}

TEST(Serialization, Gesture)
{
  using A = dec_interfaces::action::Gesture;
  A::Goal goal;
  goal.gesture_type = "deictic";
  goal.gesture_name = "point_left";
  goal.gesture_duration = std::numeric_limits<int64_t>::max();
  goal.bow_nod_angle = -45;
  goal.location_x = 1.25;
  goal.location_y = -0.5;
  goal.location_z = 1e-12;
  EXPECT_EQ(roundTrip(goal), goal);

  A::Result result;
  result.success = true;
  result.message = "done";
  result.actual_duration_seconds = 2.75f;
  EXPECT_EQ(roundTrip(result), result);

  A::Feedback feedback;
  feedback.elapsed_seconds = 1.5f;
  EXPECT_EQ(roundTrip(feedback), feedback);
}

TEST(Serialization, SpeechRecognition)
{
  using A = dec_interfaces::action::SpeechRecognition;
  A::Goal goal;
  goal.wait = 5.0f;
  EXPECT_EQ(roundTrip(goal), goal);

  A::Result result;
  result.transcription = kText;
  EXPECT_EQ(roundTrip(result), result);

  A::Feedback feedback;
  feedback.status = "listening";
  EXPECT_EQ(roundTrip(feedback), feedback);
}

TEST(Serialization, TTS)
{
  using A = dec_interfaces::action::TTS;
  A::Goal goal;
  goal.text = kText;
  EXPECT_EQ(roundTrip(goal), goal);

  A::Result result;
  result.success = true;
  result.message = "spoken";
  EXPECT_EQ(roundTrip(result), result);

  A::Feedback feedback;
  feedback.status = "speaking";
  EXPECT_EQ(roundTrip(feedback), feedback);
}

// The action wrapper types are what actually crosses the wire during a goal
// request; check one end to end so the generated action plumbing is covered.
TEST(Serialization, GestureSendGoalWrapper)
{
  using A = dec_interfaces::action::Gesture;
  A::Impl::SendGoalService::Request req;
  req.goal_id.uuid.fill(0xAB);
  req.goal.gesture_type = "iconic";
  req.goal.gesture_duration = 1500;
  EXPECT_EQ(roundTrip(req), req);
}
