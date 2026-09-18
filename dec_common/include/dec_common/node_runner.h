/* node_runner.h
 *
 * Shared main() helper for dec_system nodes: rclcpp::init -> construct ->
 * spin -> shutdown, with an optional startup banner, a single- or
 * multi-threaded executor, optional extra nodes on that executor, and an
 * optional post-spin hook.
 *
 * Header-only and rclcpp-only on purpose — link the light
 * dec_common::dec_common_runner target, not the full dec_common library.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 18, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#pragma once

#include <rclcpp/rclcpp.hpp>

#include <functional>
#include <memory>
#include <vector>

namespace dec_common {

/** @brief Startup options for runNode(): banner, logger name, and executor threads. */
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

    std::shared_ptr<NodeT> node;
    try {
        node = std::make_shared<NodeT>();
    } catch (const std::exception&) {
        // A shutdown racing construction (SIGINT in the first few ms, as
        // launch_testing does) makes LifecycleNode's built-in service
        // creation throw against an invalidated context. Exit cleanly rather
        // than aborting; genuine failures still throw, since rclcpp::ok()
        // is true for those.
        if (!rclcpp::ok()) {
            return 0;
        }
        throw;
    }

    try {
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
    } catch (const std::exception&) {
        // Same race, one step later: executor setup creates its own guard
        // condition against the context, which SIGINT can invalidate between
        // construction and this call.
        if (!rclcpp::ok()) {
            return 0;
        }
        throw;
    }

    if (after_spin) after_spin(*node);

    rclcpp::shutdown();
    return 0;
}

}  // namespace dec_common

