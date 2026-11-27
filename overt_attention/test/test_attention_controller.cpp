// test_attention_controller.cpp
//
// The attention controller judges saliency peaks in world angles from the
// current head pose. With the head parked outside the saliency limits (as face
// tracking can leave it before a restart) every peak used to be rejected and
// the head never moved again. These tests start the real controller with the
// head where it was found on the robot (yaw 1.109, pitch -0.509 rad) and
// check that it is brought back inside the limits, then follows saliency.
//
// Runs under ament_add_ros_isolated_gtest: its own ROS_DOMAIN_ID, so a robot
// stack on the same machine cannot interfere.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "dec_interfaces/msg/face_detection.hpp"
#include "dec_interfaces/msg/person_detection.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "naoqi_bridge_msgs/msg/joint_angles_with_speed.hpp"
#include "overt_attention/overt_attention_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"

using namespace std::chrono_literals;
using JointAngles = naoqi_bridge_msgs::msg::JointAnglesWithSpeed;

namespace
{

// overt_attention_configuration.yaml values the test relies on
constexpr double kYawLimit = 1.2;
constexpr double kPitchUp = 0.3;
constexpr double kPitchDown = -0.3;

class AttentionControllerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides(
      {
        {"camera_type", "pepper"},
        {"saliency_yaw_lim", kYawLimit},
        {"saliency_pitch_up", kPitchUp},
        {"saliency_pitch_dn", kPitchDown},
        {"saliency_min_score", 0.3},
        {"enable_ior", false},
      });
    controller_ = std::make_shared<OvertAttentionNode>(options);
    io_ = rclcpp::Node::make_shared("saliency_reentry_io");

    caminfo_pub_ = io_->create_publisher<sensor_msgs::msg::CameraInfo>(
      "/pepper/front/camera_info", rclcpp::SensorDataQoS());
    joints_pub_ = io_->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
    peak_pub_ = io_->create_publisher<std_msgs::msg::Float32MultiArray>(
      "/overt_attention/saliency_peak", 10);
    face_pub_ = io_->create_publisher<dec_interfaces::msg::FaceDetection>(
      "/face_detection/data", 10);
    person_pub_ = io_->create_publisher<dec_interfaces::msg::PersonDetection>(
      "/person_detection/data", 10);
    cmd_sub_ = io_->create_subscription<JointAngles>(
      "/joint_angles", 10, [this](const JointAngles::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        commands_.push_back(*msg);
      });

    executor_.add_node(controller_->get_node_base_interface());
    executor_.add_node(io_);
    spinner_ = std::thread([this]() {executor_.spin();});

    ASSERT_EQ(controller_->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
    ASSERT_EQ(controller_->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  }

  void TearDown() override
  {
    executor_.cancel();
    spinner_.join();
  }

  /// Camera info as gscam publishes it for Pepper's front camera.
  void publishCameraInfo()
  {
    sensor_msgs::msg::CameraInfo info;
    info.width = 640;
    info.height = 480;
    info.k = {525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0};
    caminfo_pub_->publish(info);
  }

  void publishHead(double yaw, double pitch)
  {
    sensor_msgs::msg::JointState js;
    js.name = {"HeadYaw", "HeadPitch"};
    js.position = {yaw, pitch};
    joints_pub_->publish(js);
  }

  void publishPeak(float u, float v, float score)
  {
    std_msgs::msg::Float32MultiArray peaks;
    peaks.data = {u, v, score};
    peak_pub_->publish(peaks);
  }

  /// Keep the head at (yaw, pitch) and publish `peak` every 100 ms until a
  /// head command arrives or `timeout` passes.
  std::vector<JointAngles> driveUntilCommand(
    double yaw, double pitch, float u, float v, std::chrono::milliseconds timeout)
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      publishCameraInfo();
      publishHead(yaw, pitch);
      publishPeak(u, v, 1.0f);
      std::this_thread::sleep_for(100ms);
      std::lock_guard<std::mutex> lock(mutex_);
      if (!commands_.empty()) {
        return commands_;
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return commands_;
  }

  void publishPerson(float x, float y, float w, float h)
  {
    dec_interfaces::msg::PersonDetection msg;
    msg.person_label_id = {"1"};
    msg.class_names = {"person"};
    msg.class_ids = {0};
    msg.confidences = {0.9f};
    geometry_msgs::msg::Point c;
    c.x = x;
    c.y = y;
    msg.centroids = {c};
    msg.width = {w};
    msg.height = {h};
    person_pub_->publish(msg);
  }

  void publishFace(float x, float y)
  {
    dec_interfaces::msg::FaceDetection msg;
    msg.face_label_id = {"7"};
    geometry_msgs::msg::Point c;
    c.x = x;
    c.y = y;
    c.z = 1.0;
    msg.centroids = {c};
    msg.width = {80.0f};
    msg.height = {80.0f};
    msg.mutual_gaze = {true};
    face_pub_->publish(msg);
  }

  /// Hold the head at (yaw, pitch), call `publish` every 50 ms, and return the
  /// commands once one arrives or `timeout` passes.
  template<typename PublishFn>
  std::vector<JointAngles> driveWith(
    double yaw, double pitch, PublishFn publish, std::chrono::milliseconds timeout)
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      publishCameraInfo();
      publishHead(yaw, pitch);
      publish();
      std::this_thread::sleep_for(50ms);
      std::lock_guard<std::mutex> lock(mutex_);
      if (!commands_.empty()) {
        return commands_;
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return commands_;
  }

  void clearCommands()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    commands_.clear();
  }

  std::shared_ptr<OvertAttentionNode> controller_;
  rclcpp::Node::SharedPtr io_;
  rclcpp::executors::MultiThreadedExecutor executor_;
  std::thread spinner_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joints_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr peak_pub_;
  rclcpp::Publisher<dec_interfaces::msg::FaceDetection>::SharedPtr face_pub_;
  rclcpp::Publisher<dec_interfaces::msg::PersonDetection>::SharedPtr person_pub_;
  rclcpp::Subscription<JointAngles>::SharedPtr cmd_sub_;
  std::mutex mutex_;
  std::vector<JointAngles> commands_;
};

}  // namespace

TEST_F(AttentionControllerTest, HeadParkedOutsideTheLimitsIsBroughtBackIn) {
  // The pose and the peak recorded on the robot: from there the peak maps to
  // yaw 1.49 rad, past the 1.2 limit, so it was rejected forever.
  const auto commands = driveUntilCommand(1.109, -0.509, 112.0f, 412.0f, 3000ms);

  ASSERT_FALSE(commands.empty()) << "head never commanded from outside the limits";
  const auto & cmd = commands.front();
  ASSERT_EQ(cmd.joint_names.size(), 2u);
  EXPECT_LE(std::abs(cmd.joint_angles[0]), kYawLimit + 1e-6);
  EXPECT_GE(cmd.joint_angles[1], kPitchDown - 1e-6);
  EXPECT_LE(cmd.joint_angles[1], kPitchUp + 1e-6);
  EXPECT_NEAR(cmd.joint_angles[1], kPitchDown, 1e-6);  // nearest edge
}

TEST_F(AttentionControllerTest, AfterReentryTheHeadFollowsSaliencyAgain) {
  ASSERT_FALSE(driveUntilCommand(1.109, -0.509, 112.0f, 412.0f, 3000ms).empty());
  clearCommands();

  // The head has reached the limit; a peak to its right is now acceptable.
  const auto commands = driveUntilCommand(1.0, -0.25, 400.0f, 260.0f, 3000ms);
  ASSERT_FALSE(commands.empty()) << "saliency never moved the head after re-entry";
  EXPECT_LT(commands.front().joint_angles[0], 1.0);   // turned toward the peak
}

TEST_F(AttentionControllerTest, HeadInsideTheLimitsGoesStraightToThePeak) {
  // Peak 119.5 px left of centre: yaw 0.1 + atan(119.5 / 525) = 0.324 rad.
  const auto commands = driveUntilCommand(0.1, -0.1, 200.0f, 239.5f, 3000ms);
  ASSERT_FALSE(commands.empty());
  // The saliency target itself, not an edge clamp.
  EXPECT_NEAR(commands.front().joint_angles[0], 0.324, 0.01);
  EXPECT_NEAR(commands.front().joint_angles[1], -0.1, 0.01);
}

TEST_F(AttentionControllerTest, HeadRecentersWhenEveryPeakIsPastTheLimit) {
  // Recorded on the robot 2026-11-27: head at yaw 0.853 (inside the limits),
  // and the strongest peaks at the left edge (u 28..96), i.e. yaw >= 1.25 --
  // all rejected, so the head sat still for good.
  ASSERT_FALSE(driveUntilCommand(0.1, -0.1, 200.0f, 239.5f, 3000ms).empty());
  clearCommands();
  const auto start = std::chrono::steady_clock::now();

  // No command while waiting out saliency_recenter_timeout (4 s by default)...
  EXPECT_TRUE(driveUntilCommand(0.853, -0.168, 52.0f, 352.0f, 3000ms).empty())
    << "recentered before saliency_recenter_timeout";
  // ...then back to default_yaw/default_pitch (0, -0.2).
  const auto commands = driveUntilCommand(0.853, -0.168, 52.0f, 352.0f, 4000ms);
  ASSERT_FALSE(commands.empty()) << "head stuck: no recentre with every peak out of range";
  const double waited = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - start).count();
  EXPECT_GE(waited, 3.5);
  EXPECT_NEAR(commands.front().joint_angles[0], 0.0, 1e-6);
  EXPECT_NEAR(commands.front().joint_angles[1], -0.2, 1e-6);
}

// ── A person in view without a detected face ────────────────────────────────

TEST_F(AttentionControllerTest, PersonWithoutAFaceAimsAtTheTopOfTheirBox) {
  // Box 150..450 px: head centre at 150 + 0.12 * 300 = 186 px, which is
  // atan((186 - 239.5) / 525) = -0.102 rad (up) from where the head points.
  const auto commands = driveWith(
    0.0, 0.0, [this]() {publishPerson(319.5f, 300.0f, 200.0f, 300.0f);}, 3000ms);
  ASSERT_FALSE(commands.empty()) << "head never moved toward the person";
  EXPECT_NEAR(commands.front().joint_angles[0], 0.0, 0.01);
  EXPECT_NEAR(commands.front().joint_angles[1], -0.102, 0.01);
}

TEST_F(AttentionControllerTest, PersonWhoseHeadIsAboveTheFrameMakesTheHeadLookUp) {
  // Box from the top edge down: the face is above the frame.
  const auto commands = driveWith(
    0.0, -0.1, [this]() {publishPerson(319.5f, 200.0f, 220.0f, 400.0f);}, 3000ms);
  ASSERT_FALSE(commands.empty());
  EXPECT_NEAR(commands.front().joint_angles[1], -0.1 - 0.12, 0.01);  // one step up
}

TEST_F(AttentionControllerTest, AFaceWinsOverThePersonAround) {
  // Face 105 px right of centre; the person box would pull the head up instead.
  const auto commands = driveWith(
    0.0, 0.0, [this]() {
      publishFace(424.5f, 239.5f);
      publishPerson(319.5f, 200.0f, 220.0f, 400.0f);
    }, 3000ms);
  ASSERT_FALSE(commands.empty());
  EXPECT_NEAR(commands.front().joint_angles[0], -0.197, 0.01);   // atan(105 / 525), to the right
  EXPECT_NEAR(commands.front().joint_angles[1], 0.0, 0.01);
}

TEST_F(AttentionControllerTest, SaliencyWaitsWhileAPersonIsInView) {
  // Head inside the saliency limits, a strong peak, and a person: the person
  // wins, so the first command is the person's head, not the peak.
  const auto commands = driveWith(
    0.1, -0.1, [this]() {
      publishPerson(319.5f, 300.0f, 200.0f, 300.0f);
      publishPeak(200.0f, 239.5f, 1.0f);
    }, 3000ms);
  ASSERT_FALSE(commands.empty());
  for (const auto & cmd : commands) {
    EXPECT_NEAR(cmd.joint_angles[0], 0.1, 0.01) << "saliency moved the head off the person";
  }
}

// ── Distance filter (no depth on Pepper's camera: size estimate) ────────────

TEST_F(AttentionControllerTest, AFaceFartherThanTwoMetresIsIgnored) {
  // 525 * 0.31 / 30 px = 5.4 m: a passer-by. Head stays where it is.
  const auto commands = driveWith(
    0.0, 0.0, [this]() {
      dec_interfaces::msg::FaceDetection msg;
      msg.face_label_id = {"9"};
      geometry_msgs::msg::Point c;
      c.x = 424.5;
      c.y = 239.5;
      msg.centroids = {c};
      msg.width = {30.0f};
      msg.height = {30.0f};
      msg.mutual_gaze = {true};
      face_pub_->publish(msg);
    }, 1500ms);
  EXPECT_TRUE(commands.empty()) << "head followed a face 2.8 m away";
}

TEST_F(AttentionControllerTest, APersonFartherThanTwoMetresIsIgnored) {
  // 525 * 0.74 / 80 px = 4.9 m.
  const auto commands = driveWith(
    0.0, 0.0, [this]() {publishPerson(319.5f, 300.0f, 80.0f, 300.0f);}, 1500ms);
  EXPECT_TRUE(commands.empty()) << "head followed a person 3 m away";
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  rclcpp::init(argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
