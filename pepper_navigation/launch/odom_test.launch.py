#!/usr/bin/env python3
"""odom_test.launch.py

Launch RViz2 and the odometry path publisher for closed-loop odometry testing.

Nodes started:
    pepper_navigation/odom_path_publisher (node: odom_path_publisher)
        Accumulates /pepper_odom_filtered poses into a nav_msgs/Path and
        publishes markers for RViz.
    rviz2/rviz2
        Pre-configured with rviz/odometry_test.rviz.

Launch arguments:
    use_sim_time (default: "false")
        Use the /clock topic (Gazebo or bag playback).
    min_distance (default: "0.02")
        Minimum metres between recorded path poses.
    min_angle (default: "0.05")
        Minimum radians between recorded path poses.

Configuration:
    rviz/odometry_test.rviz; the recording thresholds come from the launch
    arguments above rather than a YAML file.

Usage:
    ros2 launch pepper_navigation odom_test.launch.py

    Drive the robot in a closed loop and return to the start position. The
    terminal prints the deviation from the start every 50 recorded poses.
    Call the reset service to start a new run:
        ros2 service call /reset_odom_path std_srvs/srv/Empty

The node's ROS interface is documented in src/tools/odom_path_publisher.cpp.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 8, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('pepper_navigation')
    rviz_config = os.path.join(pkg_dir, 'rviz', 'odometry_test.rviz')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    min_distance = LaunchConfiguration('min_distance', default='0.02')
    min_angle = LaunchConfiguration('min_angle', default='0.05')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use /clock topic (Gazebo/bag playback)'),
        DeclareLaunchArgument('min_distance', default_value='0.02',
                              description='Minimum metres between recorded path poses'),
        DeclareLaunchArgument('min_angle', default_value='0.05',
                              description='Minimum radians between recorded path poses'),

        # Accumulates /pepper_odom_filtered poses into nav_msgs/Path and publishes markers
        Node(
            package='pepper_navigation',
            executable='odom_path_publisher',
            name='odom_path_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'min_distance': min_distance,
                'min_angle': min_angle,
            }],
        ),

        # RViz2 pre-configured for odometry inspection
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
        ),
    ])
