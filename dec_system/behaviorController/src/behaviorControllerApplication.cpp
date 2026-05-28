/*
 * behaviorControllerApplication.cpp
 *
 * Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
 * Email:  yohatad123@gmail.com
 * Date:   February 09, 2026
 * Version: v3.0 — ROS2 Lifecycle node
 *
 * Lifecycle state machine
 * ───────────────────────
 *   UNCONFIGURED → on_configure  : load YAML config + knowledge base, build BT
 *   INACTIVE     → on_activate   : start 50 Hz tick timer
 *   ACTIVE       → on_deactivate : cancel tick timer (tree stays built)
 *   INACTIVE     → on_cleanup    : halt + destroy BT tree
 *   any          → on_shutdown   : cancel timer, halt tree
 *
 * Architecture note
 * ─────────────────
 *   rclcpp_lifecycle::LifecycleNode does not inherit rclcpp::Node.
 *   BT::RosNodeParams requires std::shared_ptr<rclcpp::Node>, so this node
 *   owns a companion  bt_node_  (plain rclcpp::Node) for all BT machinery.
 *   Both nodes are added to a MultiThreadedExecutor in main():
 *     • lc_node   – lifecycle transitions, 50 Hz tick timer
 *     • bt_node_  – BT action/service clients, topic subscriptions
 */

#include "behaviorController/behaviorControllerInterface.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <behaviortree_cpp/bt_factory.h>
#include <behaviortree_cpp/loggers/groot2_publisher.h>

// Forward declaration (defined in behaviorControllerImplementation.cpp)
namespace behavior_controller {
    BT::Tree initializeTree(const std::string& scenario,
                            std::shared_ptr<rclcpp::Node> node_handle);
}

// ─────────────────────────────────────────────────────────────────────────────
// BehaviorControllerLifecycleNode — constructor
// ─────────────────────────────────────────────────────────────────────────────

BehaviorControllerLifecycleNode::BehaviorControllerLifecycleNode()
    : rclcpp_lifecycle::LifecycleNode("behavior_controller")
{
    // Create the companion BT node immediately so main() can add it to the
    // executor before any lifecycle transition is triggered.
    bt_node_ = rclcpp::Node::make_shared("behavior_controller_bt");

    RCLCPP_INFO(get_logger(), "behavior_controller: created (UNCONFIGURED)");
}

// ─────────────────────────────────────────────────────────────────────────────
// on_configure — load config + knowledge base, build behavior tree
// ─────────────────────────────────────────────────────────────────────────────

BehaviorControllerLifecycleNode::CallbackReturn
BehaviorControllerLifecycleNode::on_configure(const rclcpp_lifecycle::State& /*state*/)
{
    // ── Startup banner ──────────────────────────────────────────────────────
    RCLCPP_INFO(get_logger(),
        "\n"
        "**************************************************\n"
        "\t\tBehavior Controller v2.0 (BehaviorTree.ROS2)\n"
        "\t\tCopyright (C) 2025 CyLab Carnegie Mellon University Africa\n"
        "**************************************************\n");

    // ── Load YAML configuration ─────────────────────────────────────────────
    const std::string packagePath =
        ament_index_cpp::get_package_share_directory("behavior_controller");
    const std::string configPath =
        packagePath + "/config/behaviorControllerConfiguration.yaml";

    if (!ConfigManager::instance().loadFromFile(configPath)) {
        RCLCPP_ERROR(get_logger(),
                     "Failed to load configuration from: %s", configPath.c_str());
        return CallbackReturn::FAILURE;
    }

    // ── Load knowledge base ─────────────────────────────────────────────────
    if (!KnowledgeManager::instance().loadFromPackage(packagePath)) {
        RCLCPP_ERROR(get_logger(),
                     "Failed to load knowledge base from: %s", packagePath.c_str());
        return CallbackReturn::FAILURE;
    }

    const auto& cfg = ConfigManager::instance();
    RCLCPP_INFO(get_logger(), "Configuration loaded successfully:");
    RCLCPP_INFO(get_logger(), "  Verbose mode : %s", cfg.isVerbose() ? "Yes" : "No");
    RCLCPP_INFO(get_logger(), "  Language     : %s", cfg.getLanguage().c_str());
    RCLCPP_INFO(get_logger(), "  Scenario     : %s", cfg.getScenarioSpecification().c_str());

    // ── Build behavior tree ─────────────────────────────────────────────────
    try {
        const std::string scenario = cfg.getScenarioSpecification();
        tree_ = behavior_controller::initializeTree(scenario, bt_node_);
        tree_initialized_ = true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(),
                     "Failed to initialize behavior tree: %s", e.what());
        return CallbackReturn::FAILURE;
    }

    RCLCPP_INFO(get_logger(), "behavior_controller: configured");
    return CallbackReturn::SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// on_activate — start 50 Hz tick timer
// ─────────────────────────────────────────────────────────────────────────────

BehaviorControllerLifecycleNode::CallbackReturn
BehaviorControllerLifecycleNode::on_activate(const rclcpp_lifecycle::State& state)
{
    // Activate any managed publishers (none currently, but good practice).
    LifecycleNode::on_activate(state);

    // 50 Hz = 20 ms period.  The lambda captures this; tick() is thread-safe
    // via the executor's callback group.
    tick_timer_ = create_wall_timer(
        std::chrono::milliseconds(20),
        [this]() {
            if (tree_initialized_) {
                tree_.tickOnce();
            }
        });

    RCLCPP_INFO(get_logger(),
                "behavior_controller: activated — BT ticking at 50 Hz");
    return CallbackReturn::SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// on_deactivate — pause BT ticking (tree + config remain intact)
// ─────────────────────────────────────────────────────────────────────────────

BehaviorControllerLifecycleNode::CallbackReturn
BehaviorControllerLifecycleNode::on_deactivate(const rclcpp_lifecycle::State& state)
{
    if (tick_timer_) {
        tick_timer_->cancel();
        tick_timer_.reset();
    }

    LifecycleNode::on_deactivate(state);
    RCLCPP_INFO(get_logger(),
                "behavior_controller: deactivated — BT ticking paused");
    return CallbackReturn::SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// on_cleanup — halt and discard the BT tree
// ─────────────────────────────────────────────────────────────────────────────

BehaviorControllerLifecycleNode::CallbackReturn
BehaviorControllerLifecycleNode::on_cleanup(const rclcpp_lifecycle::State& /*state*/)
{
    if (tree_initialized_) {
        tree_.haltTree();
        tree_initialized_ = false;
    }
    RCLCPP_INFO(get_logger(), "behavior_controller: cleaned up");
    return CallbackReturn::SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// on_shutdown — cancel timer + halt tree regardless of current state
// ─────────────────────────────────────────────────────────────────────────────

BehaviorControllerLifecycleNode::CallbackReturn
BehaviorControllerLifecycleNode::on_shutdown(const rclcpp_lifecycle::State& /*state*/)
{
    if (tick_timer_) {
        tick_timer_->cancel();
        tick_timer_.reset();
    }
    if (tree_initialized_) {
        tree_.haltTree();
        tree_initialized_ = false;
    }
    RCLCPP_INFO(get_logger(), "behavior_controller: shut down");
    return CallbackReturn::SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);

    auto lc_node = std::make_shared<BehaviorControllerLifecycleNode>();

    // Spin both nodes concurrently:
    //   lc_node   → lifecycle service callbacks, 50 Hz tick timer
    //   bt_node_  → BT action/service/subscription callbacks
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions{}, 4u);
    executor.add_node(lc_node->get_node_base_interface());
    executor.add_node(lc_node->get_bt_node());
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
