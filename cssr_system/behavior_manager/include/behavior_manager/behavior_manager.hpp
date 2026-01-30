#ifndef BEHAVIOR_MANAGER_NODE_HPP_
#define BEHAVIOR_MANAGER_NODE_HPP_

#include "behaviortree_ros2/plugins/ros_action_node.hpp"
#include "CSSR_interfaces/srv/LanguageModelPrompt.hpp"

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

} // namespace behavior_manager

#endif