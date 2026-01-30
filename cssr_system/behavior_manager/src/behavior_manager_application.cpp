#include "behavior_manager/behavior_manager.cpp"
#include <behaviortree_cpp/bt_factory.h>

int main(int argc, char** argv)
{
  // Initialize ROS 2
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("behavior_manager_node");

  // Create Behavior Tree factory
  BT::BehaviorTreeFactory factory;
  BT::RosNodeParams params;
  params.nh = node;

  // Register custom nodes from behavior_manager
//   factory.registerFromPlugin("libbehavior_manager_nodes.so");
  factory.registerRosServiceNode<behavior_manager::LLMPromptServiceNode>("LLMPromptServiceNode", params);

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