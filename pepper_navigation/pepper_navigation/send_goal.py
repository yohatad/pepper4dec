"""
Entry point for a one-shot Nav2 navigation goal sender.

send_goal.py

Sets the robot's initial pose, waits for Nav2 to become active, and sends a
single navigation goal using the Nav2 Simple Commander API.

On startup, the script initializes a BasicNavigator, publishes an initial
pose at the map origin, waits until Nav2 is fully active, then sends a fixed
goal pose to the navigation stack via the NavigateToPose action. While the
goal is in progress, it periodically prints the remaining distance to the
goal until the task completes.

Actions:
    /navigate_to_pose (nav2_msgs/action/NavigateToPose)
        Action used (via BasicNavigator.goToPose) to drive the robot to the
        configured goal pose.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: March 08, 2026
Version: v1.0
"""

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator


def main():
    rclpy.init()
    navigator = BasicNavigator()

    # Set initial pose
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = 0.0
    initial_pose.pose.position.y = 0.0
    initial_pose.pose.orientation.w = 1.0
    navigator.setInitialPose(initial_pose)

    # Wait for Nav2 to activate
    navigator.waitUntilNav2Active()

    # Send goal
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    goal_pose.pose.position.x = 2.0
    goal_pose.pose.position.y = 1.0
    goal_pose.pose.orientation.w = 1.0

    navigator.goToPose(goal_pose)

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        print(f'Distance remaining: {feedback.distance_remaining}')

    print('Goal reached!')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
