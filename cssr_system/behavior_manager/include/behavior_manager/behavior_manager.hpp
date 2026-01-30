#ifndef BEHAVIOR_MANAGER_NODE_HPP_
#define BEHAVIOR_MANAGER_NODE_HPP_

#include "behaviortree_ros2/plugins/ros_action_node.hpp"
#include "behaviortree_ros2/plugins/ros_service_node.hpp"
#include "CSSR_interfaces/srv/LanguageModelPrompt.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"

namespace behavior_manager
{
using namespace BT;
using Prompt = CSSR_interfaces::srv::LanguageModelPrompt;

// Custom Action Node wrapping a ROS 2 Action
class LLMPromptServiceNode : public RosServiceNode<Prompt>
{
public:
  LLMPromptServiceNode(const std::string& name, const NodeConfig& conf, const RosNodeParams& params)
    : LLMPromptServiceNode<Prompt>(name, conf, params) {}

  // Define input/output ports for the BT Blackboard
  static PortsList providedPorts() {
    return { 
      InputPort<int>("query"),
      OutputPort<int>("response") 
    };
  }

  // Populate the request sent to the Service Server
  bool setRequest(Request::SharedPtr& request) override;

  // Handle the response from the Service Server
  NodeStatus onResponseReceived(const Response::SharedPtr& response) override;
};

class NavToPoseAction : public RosActionNode<nav2_msgs::action::NavigateToPose>
{
public:
  NavToPoseAction(const std::string& name, const NodeConfig& conf, const RosNodeParams& params)
    : RosActionNode<nav2_msgs::action::NavigateToPose>(name, conf, params) {}

  // Define ports to receive coordinates from the XML/Blackboard
  static PortsList providedPorts() {
    return { InputPort<double>("x"), InputPort<double>("y"), InputPort<double>("theta") };
  }

  bool setGoal(RosActionNode::Goal& goal) override;
  NodeStatus onResultReceived(const WrappedResult& wr) override;
  
  // Optional: Handle feedback (e.g., distance remaining)
  virtual NodeStatus onFeedback(const std::shared_ptr<const Feedback> feedback) override;
};

} // namespace behavior_manager

#endif