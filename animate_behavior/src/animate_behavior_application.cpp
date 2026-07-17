/* animate_behavior_application.cpp
 *
 * Entry point for the animate_behavior lifecycle node. Spins the node on a
 * MultiThreadedExecutor so the action server, timers, and lifecycle
 * state-machine callbacks can run concurrently; the AnimateBehaviorNode
 * class itself is implemented in animate_behavior_implementation.cpp.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 05, 2026
 * Version: v1.0
 */

#include "animate_behavior/animate_behavior_interface.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<AnimateBehaviorNode>();

    // MultiThreadedExecutor lets the action server, timers, and
    // lifecycle state-machine callbacks run concurrently.
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions{}, 4u);
    executor.add_node(node->get_node_base_interface());
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
