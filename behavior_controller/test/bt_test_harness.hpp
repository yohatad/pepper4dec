/* bt_test_harness.hpp
 *
 * Shared harness for the behavior tree tests (test_bt_nodes.cpp,
 * test_bt_scenarios.cpp).
 *
 * BtRosTest starts two ROS nodes on a background executor:
 *   bt_test_client  used by the BT nodes for their action/service clients and
 *                   topic subscriptions, the role bt_node_ plays in the real
 *                   behavior_controller;
 *   bt_test_stubs   hosts stub action and service servers standing in for the
 *                   rest of the system, plus any test publishers.
 * Stubs record every goal and request they receive; each test decides how
 * they answer. Trees are built with behavior_controller::registerNodes(), so
 * they resolve against exactly the node types the real node registers.
 *
 * The test binaries run under ament_add_ros_isolated_gtest, which gives each
 * one its own ROS_DOMAIN_ID: a robot stack running on the same machine cannot
 * answer the stubs' goals or publish on their topics.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: October 11, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#pragma once

#include <gtest/gtest.h>
#include <behaviortree_cpp/loggers/abstract_logger.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "behavior_controller/behavior_controller_interface.h"

namespace bt_test
{

/// How a stub action server finishes a goal.
enum class Outcome
{
  kSucceed,          ///< succeed() with the reply's result
  kAbort,            ///< abort() with the reply's result
  kRunUntilCanceled  ///< run until the client cancels, then canceled()
};

/// A stub action server's answer to one goal.
template<typename ActionT>
struct Reply
{
  Outcome outcome = Outcome::kSucceed;
  typename ActionT::Result result;
  /// Published once, shortly after the goal is accepted, when set.
  std::optional<typename ActionT::Feedback> feedback;
  /// Time between accepting the goal and finishing it.
  std::chrono::milliseconds delay{20};
};

/**
 * @brief Action server that records goals and answers them through a handler.
 *
 * Each accepted goal runs on its own thread, so a goal that runs until
 * canceled does not block the executor.
 */
template<typename ActionT>
class StubActionServer
{
public:
  using Goal = typename ActionT::Goal;
  using GoalHandle = rclcpp_action::ServerGoalHandle<ActionT>;
  using Handler = std::function<Reply<ActionT>(const Goal &)>;

  StubActionServer(
    const rclcpp::Node::SharedPtr & node, const std::string & name, Handler handler)
  : handler_(std::move(handler))
  {
    server_ = rclcpp_action::create_server<ActionT>(
      node, name,
      [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const Goal>) {
        return reject_ ? rclcpp_action::GoalResponse::REJECT :
        rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [](std::shared_ptr<GoalHandle>) {return rclcpp_action::CancelResponse::ACCEPT;},
      [this](std::shared_ptr<GoalHandle> handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        goals_.push_back(*handle->get_goal());
        workers_.emplace_back([this, handle]() {execute(handle);});
      });
  }

  ~StubActionServer()
  {
    stopping_ = true;
    std::vector<std::thread> workers;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      workers.swap(workers_);
    }
    for (auto & worker : workers) {
      worker.join();
    }
  }

  void setHandler(Handler handler)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    handler_ = std::move(handler);
  }

  void rejectGoals() {reject_ = true;}

  /// Every goal received so far, in arrival order.
  std::vector<Goal> goals() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return goals_;
  }

  /// Number of goals the client canceled.
  int canceledCount() const {return canceled_;}

private:
  void execute(const std::shared_ptr<GoalHandle> & handle)
  {
    Handler handler;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      handler = handler_;
    }
    const Reply<ActionT> reply = handler(*handle->get_goal());
    auto result = std::make_shared<typename ActionT::Result>(reply.result);

    if (reply.feedback) {
      // Let the client process the goal response first: feedback for a goal
      // it does not know yet is dropped.
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      handle->publish_feedback(
        std::make_shared<typename ActionT::Feedback>(*reply.feedback));
    }

    const auto finish_at = std::chrono::steady_clock::now() + reply.delay;
    while (!stopping_ && !handle->is_canceling() &&
      (reply.outcome == Outcome::kRunUntilCanceled ||
      std::chrono::steady_clock::now() < finish_at))
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (handle->is_canceling()) {
      ++canceled_;
      handle->canceled(result);
    } else if (reply.outcome == Outcome::kSucceed) {
      handle->succeed(result);
    } else {
      handle->abort(result);
    }
  }

  typename rclcpp_action::Server<ActionT>::SharedPtr server_;
  Handler handler_;
  std::vector<Goal> goals_;
  std::vector<std::thread> workers_;
  std::atomic<bool> reject_{false};
  std::atomic<bool> stopping_{false};
  std::atomic<int> canceled_{0};
  mutable std::mutex mutex_;
};

/// Service server that records requests and answers them through a handler.
template<typename ServiceT>
class StubService
{
public:
  using Request = typename ServiceT::Request;
  using Response = typename ServiceT::Response;
  using Handler = std::function<void (const Request &, Response &)>;

  StubService(const rclcpp::Node::SharedPtr & node, const std::string & name, Handler handler)
  : handler_(std::move(handler))
  {
    service_ = node->create_service<ServiceT>(
      name,
      [this](const std::shared_ptr<Request> request, std::shared_ptr<Response> response) {
        Handler handler;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          requests_.push_back(*request);
          handler = handler_;
        }
        handler(*request, *response);
      });
  }

  void setHandler(Handler handler)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    handler_ = std::move(handler);
  }

  /// Every request received so far, in arrival order.
  std::vector<Request> requests() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_;
  }

private:
  typename rclcpp::Service<ServiceT>::SharedPtr service_;
  Handler handler_;
  std::vector<Request> requests_;
  mutable std::mutex mutex_;
};

/// Handler for std_srvs Trigger/SetBool stubs: success with message "ok".
template<typename ServiceT>
void respondSuccess(const typename ServiceT::Request &, typename ServiceT::Response & response)
{
  response.success = true;
  response.message = "ok";
}

/**
 * @brief Records every status change in a tree.
 *
 * Used to assert which branches ran and, when a run fails, to say which
 * leaf nodes failed and in what order.
 */
class StatusRecorder : public BT::StatusChangeLogger
{
public:
  explicit StatusRecorder(const BT::Tree & tree)
  : BT::StatusChangeLogger(tree.rootNode()) {}

  void callback(
    BT::Duration, const BT::TreeNode & node, BT::NodeStatus,
    BT::NodeStatus status) override
  {
    const bool leaf = node.type() == BT::NodeType::ACTION ||
      node.type() == BT::NodeType::CONDITION;
    std::lock_guard<std::mutex> lock(mutex_);
    events_.push_back({node.fullPath(), node.name(), leaf, status});
  }

  void flush() override {}

  /// True when a node whose name (XML name= or node ID) is @p name reached @p status.
  bool reached(const std::string & name, BT::NodeStatus status) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & event : events_) {
      if (event.name == name && event.status == status) {
        return true;
      }
    }
    return false;
  }

  /// The leaf nodes that returned FAILURE, one per line, in order.
  std::string failedLeaves() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    for (const auto & event : events_) {
      if (event.leaf && event.status == BT::NodeStatus::FAILURE) {
        out << "  " << event.path << "\n";
      }
    }
    return out.str();
  }

private:
  struct Event
  {
    std::string path;
    std::string name;
    bool leaf;
    BT::NodeStatus status;
  };
  std::vector<Event> events_;
  mutable std::mutex mutex_;
};

/// Fixture: client and stub nodes spinning on a background executor.
class BtRosTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    client_ = std::make_shared<rclcpp::Node>("bt_test_client");
    stubs_ = std::make_shared<rclcpp::Node>("bt_test_stubs");
    executor_ = std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
      rclcpp::ExecutorOptions(), 4);
    executor_->add_node(client_);
    executor_->add_node(stubs_);
    spinner_ = std::thread([this]() {executor_->spin();});

    params_.nh = client_;
    params_.server_timeout = std::chrono::milliseconds(2000);
    params_.wait_for_server_timeout = std::chrono::milliseconds(3000);
  }

  void TearDown() override
  {
    executor_->cancel();
    spinner_.join();
    keep_alive_.clear();
    executor_.reset();
    stubs_.reset();
    client_.reset();
  }

  template<typename ActionT>
  std::shared_ptr<StubActionServer<ActionT>> serveAction(
    const std::string & name, typename StubActionServer<ActionT>::Handler handler)
  {
    auto stub = std::make_shared<StubActionServer<ActionT>>(stubs_, name, std::move(handler));
    keep_alive_.push_back(stub);
    return stub;
  }

  template<typename ServiceT>
  std::shared_ptr<StubService<ServiceT>> serveService(
    const std::string & name,
    typename StubService<ServiceT>::Handler handler = respondSuccess<ServiceT>)
  {
    auto stub = std::make_shared<StubService<ServiceT>>(stubs_, name, std::move(handler));
    keep_alive_.push_back(stub);
    return stub;
  }

  /// Publisher on @p topic, returned once a subscriber is connected.
  template<typename MsgT>
  typename rclcpp::Publisher<MsgT>::SharedPtr connectedPublisher(const std::string & topic)
  {
    auto publisher = stubs_->create_publisher<MsgT>(topic, 10);
    keep_alive_.push_back(publisher);
    EXPECT_TRUE(waitFor([&]() {return publisher->get_subscription_count() > 0;}))
      << "nothing subscribed to " << topic;
    return publisher;
  }

  /// Every @p period until the test ends, publishes make() on @p topic
  /// (nothing when it returns no message).
  template<typename MsgT>
  void publishEvery(
    const std::string & topic, std::chrono::milliseconds period,
    std::function<std::optional<MsgT>()> make)
  {
    auto publisher = stubs_->create_publisher<MsgT>(topic, 10);
    keep_alive_.push_back(publisher);
    keep_alive_.push_back(
      stubs_->create_wall_timer(
        period, [publisher, make]() {
          if (auto msg = make()) {
            publisher->publish(*msg);
          }
        }));
  }

  /// Tree whose main BehaviorTree is @p body, a fragment of BT XML.
  BT::Tree treeFromText(const std::string & body)
  {
    BT::BehaviorTreeFactory factory;
    behavior_controller::registerNodes(factory, params_);
    return factory.createTreeFromText(
      "<root BTCPP_format=\"4\"><BehaviorTree ID=\"Test\">" + body +
      "</BehaviorTree></root>");
  }

  /// Tree from the XML file at @p path, its root blackboard @p blackboard.
  BT::Tree treeFromFile(
    const std::string & path, BT::Blackboard::Ptr blackboard = BT::Blackboard::create())
  {
    BT::BehaviorTreeFactory factory;
    behavior_controller::registerNodes(factory, params_);
    return factory.createTreeFromFile(path, blackboard);
  }

  /// Ticks @p tree until it finishes, @p done() holds, or @p timeout passes,
  /// and returns its last status. A tree still RUNNING is left running.
  static BT::NodeStatus tickUntil(
    BT::Tree & tree, const std::function<bool()> & done, std::chrono::milliseconds timeout)
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    BT::NodeStatus status = tree.tickOnce();
    while (status == BT::NodeStatus::RUNNING && !done() &&
      std::chrono::steady_clock::now() < deadline)
    {
      tree.sleep(std::chrono::milliseconds(10));
      status = tree.tickOnce();
    }
    return status;
  }

  /// Ticks @p tree until it finishes or @p timeout passes; halts it on timeout.
  static BT::NodeStatus tickUntilDone(
    BT::Tree & tree, std::chrono::milliseconds timeout = std::chrono::milliseconds(10000))
  {
    const BT::NodeStatus status = tickUntil(tree, []() {return false;}, timeout);
    if (status == BT::NodeStatus::RUNNING) {
      tree.haltTree();
    }
    return status;
  }

  /// Ticks @p tree for @p duration, or until it finishes, and returns its last
  /// status (for asserting that a node keeps waiting).
  static BT::NodeStatus tickFor(BT::Tree & tree, std::chrono::milliseconds duration)
  {
    return tickUntil(tree, []() {return false;}, duration);
  }

  /// Polls @p condition until it holds or @p timeout passes.
  static bool waitFor(
    const std::function<bool()> & condition,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(5000))
  {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!condition()) {
      if (std::chrono::steady_clock::now() > deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
  }

  rclcpp::Node::SharedPtr client_;
  rclcpp::Node::SharedPtr stubs_;
  BT::RosNodeParams params_;

private:
  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::thread spinner_;
  std::vector<std::shared_ptr<void>> keep_alive_;
};

/// One FaceDetection frame with a face per entry of @p gaze (its mutual gaze flag).
inline dec_interfaces::msg::FaceDetection faces(const std::vector<bool> & gaze)
{
  dec_interfaces::msg::FaceDetection msg;
  for (size_t i = 0; i < gaze.size(); ++i) {
    geometry_msgs::msg::Point centroid;
    centroid.x = 320.0 + 10.0 * static_cast<double>(i);
    centroid.y = 240.0;
    centroid.z = 1.5;
    msg.face_label_id.push_back("face_" + std::to_string(i));
    msg.centroids.push_back(centroid);
    msg.width.push_back(80.0f);
    msg.height.push_back(100.0f);
    msg.mutual_gaze.push_back(gaze[i]);
  }
  return msg;
}

}  // namespace bt_test
