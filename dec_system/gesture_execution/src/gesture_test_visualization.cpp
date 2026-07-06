/* gesture_test_visualization.cpp
 *
 * Standalone test node: publishes a fixed set of example markers that
 * simulate a deictic pointing gesture, for verifying gesture visualization
 * in RViz2.
 *
 * Publishes a target sphere, a shoulder-position sphere, a pointing arrow
 * between them, and a text label with the target coordinates to the
 * /gesture_execution/visualization topic. After publishing, run rviz2, add a
 * Marker display, set the topic to /gesture_execution/visualization, and set
 * the fixed frame to base_link to view the markers.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 05, 2026
 * Version: v1.0 - C++ port of gesture_test_visualization.py
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

class VisualizationTestNode : public rclcpp::Node {
public:
    VisualizationTestNode() : rclcpp::Node("visualization_test") {
        marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/gesture_execution/visualization", 10);
        RCLCPP_INFO(get_logger(), "Visualization Test Node started");
        RCLCPP_INFO(get_logger(), "Publishing test markers to /gesture_execution/visualization");
        RCLCPP_INFO(get_logger(), "Run 'ros2 run rviz2 rviz2' to visualize");
    }

    void publishTestMarkers() {
        // Simulate a pointing gesture to (1.0, 0.5, 0.3) meters from base_link
        double target_x = 1000.0;  // mm
        double target_y = 500.0;   // mm
        double target_z = 300.0;   // mm

        // Simulated shoulder position (right arm)
        double shoulder_x = -57.0;     // mm
        double shoulder_y = -149.74;   // mm (right arm is negative)
        double shoulder_z = 86.82;     // mm

        auto stamp = get_clock()->now();

        // 1. Target point marker (sphere) — bright red for better visibility
        visualization_msgs::msg::Marker target_marker;
        target_marker.header.stamp = stamp;
        target_marker.header.frame_id = "base_link";
        target_marker.ns = "deictic_target";
        target_marker.id = 0;
        target_marker.type = visualization_msgs::msg::Marker::SPHERE;
        target_marker.action = visualization_msgs::msg::Marker::ADD;
        target_marker.pose.position.x = target_x / 1000.0;
        target_marker.pose.position.y = target_y / 1000.0;
        target_marker.pose.position.z = target_z / 1000.0;
        target_marker.pose.orientation.w = 1.0;
        target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.1;
        target_marker.color.r = 1.0;
        target_marker.color.g = 0.0;
        target_marker.color.b = 0.0;
        target_marker.color.a = 1.0;
        target_marker.lifetime.sec = 10;
        target_marker.lifetime.nanosec = 0;
        marker_pub_->publish(target_marker);
        RCLCPP_INFO(get_logger(), "Published target marker (bright red sphere)");

        // 2. Shoulder position marker — bright blue for better visibility
        visualization_msgs::msg::Marker shoulder_marker;
        shoulder_marker.header.stamp = stamp;
        shoulder_marker.header.frame_id = "base_link";
        shoulder_marker.ns = "deictic_shoulder";
        shoulder_marker.id = 1;
        shoulder_marker.type = visualization_msgs::msg::Marker::SPHERE;
        shoulder_marker.action = visualization_msgs::msg::Marker::ADD;
        shoulder_marker.pose.position.x = shoulder_x / 1000.0;
        shoulder_marker.pose.position.y = shoulder_y / 1000.0;
        shoulder_marker.pose.position.z = shoulder_z / 1000.0;
        shoulder_marker.pose.orientation.w = 1.0;
        shoulder_marker.scale.x = shoulder_marker.scale.y = shoulder_marker.scale.z = 0.06;
        shoulder_marker.color.r = 0.0;
        shoulder_marker.color.g = 0.0;
        shoulder_marker.color.b = 1.0;
        shoulder_marker.color.a = 1.0;
        shoulder_marker.lifetime.sec = 10;
        shoulder_marker.lifetime.nanosec = 0;
        marker_pub_->publish(shoulder_marker);
        RCLCPP_INFO(get_logger(), "Published shoulder marker (bright blue sphere)");

        // 3. Pointing line from shoulder to target — bright blue arrow
        visualization_msgs::msg::Marker line_marker;
        line_marker.header.stamp = stamp;
        line_marker.header.frame_id = "base_link";
        line_marker.ns = "deictic_line";
        line_marker.id = 2;
        line_marker.type = visualization_msgs::msg::Marker::ARROW;
        line_marker.action = visualization_msgs::msg::Marker::ADD;

        geometry_msgs::msg::Point start_point;
        start_point.x = shoulder_x / 1000.0;
        start_point.y = shoulder_y / 1000.0;
        start_point.z = shoulder_z / 1000.0;
        geometry_msgs::msg::Point end_point;
        end_point.x = target_x / 1000.0;
        end_point.y = target_y / 1000.0;
        end_point.z = target_z / 1000.0;
        line_marker.points.push_back(start_point);
        line_marker.points.push_back(end_point);

        line_marker.color.r = 0.0;
        line_marker.color.g = 0.0;
        line_marker.color.b = 1.0;
        line_marker.color.a = 0.8;
        line_marker.scale.x = 0.03;  // shaft diameter
        line_marker.scale.y = 0.06;  // head diameter
        line_marker.scale.z = 0.12;  // head length
        line_marker.lifetime.sec = 10;
        line_marker.lifetime.nanosec = 0;
        marker_pub_->publish(line_marker);
        RCLCPP_INFO(get_logger(), "Published pointing line (bright blue arrow)");

        // 4. Text label showing coordinates — larger and brighter
        visualization_msgs::msg::Marker text_marker;
        text_marker.header.stamp = stamp;
        text_marker.header.frame_id = "base_link";
        text_marker.ns = "deictic_text";
        text_marker.id = 3;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.pose.position.x = target_x / 1000.0;
        text_marker.pose.position.y = target_y / 1000.0;
        text_marker.pose.position.z = (target_z / 1000.0) + 0.15;
        text_marker.pose.orientation.w = 1.0;
        std::ostringstream text;
        text << "Target: (" << (target_x / 1000.0) << ", " << (target_y / 1000.0) << ", "
             << (target_z / 1000.0) << ") m";
        text_marker.text = text.str();
        text_marker.scale.z = 0.07;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.lifetime.sec = 10;
        text_marker.lifetime.nanosec = 0;
        marker_pub_->publish(text_marker);
        RCLCPP_INFO(get_logger(), "Published text label with coordinates");

        RCLCPP_INFO(get_logger(), "\n===== VISUALIZATION TEST COMPLETE =====");
        RCLCPP_INFO(get_logger(), "To view in RViz2:");
        RCLCPP_INFO(get_logger(), "1. Run: ros2 run rviz2 rviz2");
        RCLCPP_INFO(get_logger(), "2. Add a 'Marker' display");
        RCLCPP_INFO(get_logger(), "3. Set topic to: /gesture_execution/visualization");
        RCLCPP_INFO(get_logger(), "4. Make sure 'Global Options' -> 'Fixed Frame' is set to 'base_link'");
        RCLCPP_INFO(get_logger(), "========================================");
    }

private:
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    auto test_node = std::make_shared<VisualizationTestNode>();

    // Wait a moment for the publisher to be ready
    std::this_thread::sleep_for(std::chrono::seconds(1));

    test_node->publishTestMarkers();

    // Keep node alive for a bit so markers are published
    std::this_thread::sleep_for(std::chrono::seconds(2));

    rclcpp::shutdown();

    std::cout << "\nTest complete! Markers should be visible in RViz2 for 10 seconds.\n"
              << "You can run this test again to refresh the markers.\n";
    return 0;
}
