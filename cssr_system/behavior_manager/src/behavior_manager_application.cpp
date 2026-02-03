#include "behavior_manager/behavior_manager.hpp"
#include <behaviortree_cpp/bt_factory.h>


int main(int argc, char** argv)
{
  // Initialize ROS 2
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("behavior_manager_node");

  // Create Behavior Tree factory
  BT::BehaviorTreeFactory factory;
  BT::RosNodeParams llm_prompt_params;
  BT::RosNodeParams nav2_params;

  nav2_params.nh = node;
  llm_prompt_params.nh = node;

  nav2_params.default_port_value = "navigate_to_pose";
  llm_prompt_params.default_port_value = "/conversationManagement/prompt";

  // Register custom nodes from behavior_manager
  factory.registerNodeType<behavior_manager::LLMPromptServiceNode>("LLMPromptServiceNode", llm_prompt_params);
  factory.registerNodeType<behavior_manager::NavToPoseAction>("NavToPose", nav2_params);

  // Load and create the Behavior Tree from XML
  auto tree = factory.createTreeFromFile("behavior_tree.xml");

  // Execute the Behavior Tree
  while (rclcpp::ok() && tree.tickWhileRunning() == BT::NodeStatus::RUNNING) {
    rclcpp::spin_some(node);
  }

  // Shutdown ROS 2
  rclcpp::shutdown();
  return 0;
}