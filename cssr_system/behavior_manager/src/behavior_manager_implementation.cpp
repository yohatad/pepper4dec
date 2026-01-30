#include "behavior_manager/behavior_manager.hpp"

namespace behavior_manager
{
// Populate the request sent to the Service Server
bool LLMPromptServiceNode::setRequest(Request::SharedPtr& request)
{
  // Example: Populate the request with data from the Blackboard
  request->prompt = getInput<std::string>("query");
  RCLCPP_INFO(getLogger(), "Sending prompt: %s", request->prompt.c_str());
  return true;
}

// Handle the response from the Service Server
NodeStatus LLMPromptServiceNode::onResponseReceived(const Response::SharedPtr& response)
{
  // Example: Set the response data to the Blackboard
  RCLCPP_INFO(getLogger(), "Received response: %s", response->response.c_str());
  setOutput<std::string>("response", response->response);
  return NodeStatus::SUCCESS;
}

} // namespace behavior_manager