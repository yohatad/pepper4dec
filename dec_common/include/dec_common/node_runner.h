/* node_runner.h
 *
 * Shared main() helper for dec_system nodes. Replaces the six hand-rolled
 * rclcpp::init -> construct -> spin -> shutdown application files with one
 * template covering their actual variations: an optional startup banner,
 * single- vs multi-threaded executor, optional extra nodes on the executor
 * (behavior_controller's companion BT node), and an optional post-spin hook
 * (face_detection's cleanup()).
 *
 * Header-only and rclcpp-only on purpose — link the light
 * dec_common::dec_common_runner target, not the full dec_common library.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 18, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#ifndef DEC_COMMON_NODE_RUNNER_H
#define DEC_COMMON_NODE_RUNNER_H

#include <rclcpp/rclcpp.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace dec_common {

struct NodeRunOptions {
    // Logged once after rclcpp::init when non-null (one-liner or full banner).
    const char* banner = nullptr;
    // Logger name the banner is logged under.
    const char* logger_name = "dec_system";
    // 1 spins single-threaded; >1 uses a MultiThreadedExecutor with that many
    // threads.
    size_t executor_threads = 1;
};

// Standard node entry point: rclcpp::init -> construct NodeT -> spin ->
// shutdown. `extra_nodes` (optional) runs after construction and returns
// additional node interfaces to spin on the same executor; `after_spin`
// (optional) runs once spin returns, before rclcpp::shutdown.
template <typename NodeT>
int runNode(int argc, char** argv, const NodeRunOptions& options = {},
            std::function<std::vector<rclcpp::node_interfaces::NodeBaseInterface::SharedPtr>(NodeT&)>
                extra_nodes = nullptr,
            std::function<void(NodeT&)> after_spin = nullptr) {
    rclcpp::init(argc, argv);

    if (options.banner != nullptr) {
        RCLCPP_INFO(rclcpp::get_logger(options.logger_name), "%s", options.banner);
    }

    auto node = std::make_shared<NodeT>();

    if (options.executor_threads <= 1 && !extra_nodes) {
        rclcpp::spin(node->get_node_base_interface());
    } else {
        rclcpp::executors::MultiThreadedExecutor executor(
            rclcpp::ExecutorOptions{}, options.executor_threads);
        executor.add_node(node->get_node_base_interface());
        if (extra_nodes) {
            for (auto& extra : extra_nodes(*node)) {
                executor.add_node(extra);
            }
        }
        executor.spin();
    }

    if (after_spin) after_spin(*node);

    rclcpp::shutdown();
    return 0;
}

}  // namespace dec_common

#endif  // DEC_COMMON_NODE_RUNNER_H
