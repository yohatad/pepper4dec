#include <tf2/LinearMath/Quaternion.h>
#include "behavior_manager/behavior_manager.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace behavior_manager
{
bool LLMPromptServiceNode::setRequest(Request::SharedPtr& request)
{
  auto msg = getInput<std::string>("query");

  if (!msg) {
    RCLCPP_ERROR(node_.lock()->get_logger(), "Missing 'query' input: %s", msg.error().c_str());
    return false;
  }

  request->prompt = msg.value(); 
  return true;
}

NodeStatus LLMPromptServiceNode::onResponseReceived(const Response::SharedPtr& response)
{
  RCLCPP_INFO(node_.lock()->get_logger(), "Received response: %s", response->response.c_str());
  setOutput<std::string>("response", response->response);
  return NodeStatus::SUCCESS;
}

bool ASRActionNode::setGoal(RosActionNode::Goal& goal) {
  double wait_seconds;
  if (!getInput("wait_seconds", wait_seconds)) {
    RCLCPP_ERROR(node_.lock()->get_logger(), "Missing port: wait_seconds");
    return false;
  }
  goal.wait = wait_seconds;
  RCLCPP_INFO(node_.lock()->get_logger(), "Setting ASR wait time to %.2f seconds", wait_seconds);
  return true;
}

NodeStatus ASRActionNode::onResultReceived(const WrappedResult& wr) {
  if (wr.code == rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_INFO(node_.lock()->get_logger(), "ASR Action Succeeded. Recognized Speech: %s", wr.result->transcription.c_str());
    setOutput<std::string>("recognized_text", wr.result->transcription);
    return NodeStatus::SUCCESS;
  }
  RCLCPP_ERROR(node_.lock()->get_logger(), "ASR Action Failed.");
  setOutput<std::string>("recognized_text", "None");
  return NodeStatus::FAILURE;
}

bool NavToPoseAction::setGoal(RosActionNode::Goal& goal) {
  double x, y, theta;
  // Ensure all ports are provided
  if (!getInput("x", x) || !getInput("y", y) || !getInput("theta", theta)) {
    RCLCPP_ERROR(node_.lock()->get_logger(), "Missing ports! Need x, y, and theta.");
    return false;
  }

  goal.pose.header.frame_id = "map";
  goal.pose.header.stamp = node_.lock()->now();
  goal.pose.pose.position.x = x;
  goal.pose.pose.position.y = y;

  // Orientation: Convert Degrees (theta) to Quaternion
  tf2::Quaternion q;
  double radians = theta * (M_PI / 180.0); // Conversion
  q.setRPY(0, 0, radians); // Roll, Pitch, Yaw
  
  goal.pose.pose.orientation = tf2::toMsg(q);

  RCLCPP_INFO(node_.lock()->get_logger(), "Sending Nav2 Goal: x=%.2f, y=%.2f, theta=%.2f deg", x, y, theta);
  return true;
}

NodeStatus NavToPoseAction::onResultReceived(const WrappedResult& wr) {
  if (wr.code == rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_INFO(node_.lock()->get_logger(), "Navigation Successful!");
    return NodeStatus::SUCCESS;
  }
  RCLCPP_ERROR(node_.lock()->get_logger(), "Navigation Failed.");
  return NodeStatus::FAILURE;
}

NodeStatus NavToPoseAction::onFeedback(const std::shared_ptr<const Feedback> feedback) {
  // Use this to log distance remaining if needed
  RCLCPP_INFO(node_.lock()->get_logger(), "Distance Remaining: %.2f meters", feedback->distance_remaining);
  return NodeStatus::RUNNING;
}

} // namespace behavior_manager