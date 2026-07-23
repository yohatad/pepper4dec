/* send_goal.cpp
 *
 * One-shot Nav2 navigation goal sender. Publishes an initial pose at the
 * map origin, waits for the NavigateToPose action server to come up, then
 * sends a single fixed goal pose and prints the remaining distance until
 * the goal completes.
 *
 * nav2_simple_commander (the Python BasicNavigator this replaces) is a
 * Python-only convenience wrapper with no official C++ equivalent, so this
 * reimplements the relevant subset directly: a plain publish to
 * /initialpose (what AMCL listens to for manual pose initialization) and
 * an rclcpp_action client for NavigateToPose.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 05, 2026
 * Version: v1.0 - C++ port of send_goal.py
 */

#include <chrono>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>

using NavigateToPose = nav2_msgs::action::NavigateToPose;
using GoalHandle = rclcpp_action::ClientGoalHandle<NavigateToPose>;

class SendGoal : public rclcpp::Node {
public:
    SendGoal() : Node("send_goal") {
        initial_pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 10);
        action_client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");
    }

    void run() {
        publishInitialPose();

        RCLCPP_INFO(get_logger(), "Waiting for navigate_to_pose action server...");
        if (!action_client_->wait_for_action_server(std::chrono::seconds(30))) {
            RCLCPP_ERROR(get_logger(), "navigate_to_pose action server not available");
            return;
        }

        NavigateToPose::Goal goal;
        goal.pose.header.frame_id = "map";
        goal.pose.header.stamp = now();
        goal.pose.pose.position.x = 2.0;
        goal.pose.pose.position.y = 1.0;
        goal.pose.pose.orientation.w = 1.0;

        auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
        send_goal_options.feedback_callback =
            [this](GoalHandle::SharedPtr, const std::shared_ptr<const NavigateToPose::Feedback> feedback) {
                RCLCPP_INFO(get_logger(), "Distance remaining: %.3f", feedback->distance_remaining);
            };

        auto goal_handle_future = action_client_->async_send_goal(goal, send_goal_options);
        if (rclcpp::spin_until_future_complete(get_node_base_interface(), goal_handle_future) !=
            rclcpp::FutureReturnCode::SUCCESS) {
            RCLCPP_ERROR(get_logger(), "Failed to send goal");
            return;
        }
        auto goal_handle = goal_handle_future.get();
        if (!goal_handle) {
            RCLCPP_ERROR(get_logger(), "Goal was rejected");
            return;
        }

        auto result_future = action_client_->async_get_result(goal_handle);
        if (rclcpp::spin_until_future_complete(get_node_base_interface(), result_future) !=
            rclcpp::FutureReturnCode::SUCCESS) {
            RCLCPP_ERROR(get_logger(), "Failed to get result");
            return;
        }
        RCLCPP_INFO(get_logger(), "Goal reached!");
    }

private:
    void publishInitialPose() {
        geometry_msgs::msg::PoseWithCovarianceStamped initial_pose;
        initial_pose.header.frame_id = "map";
        initial_pose.header.stamp = now();
        initial_pose.pose.pose.position.x = 0.0;
        initial_pose.pose.pose.position.y = 0.0;
        initial_pose.pose.pose.orientation.w = 1.0;
        initial_pose_pub_->publish(initial_pose);
    }

    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_pub_;
    rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SendGoal>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
