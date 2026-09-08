#!/usr/bin/env python3
"""gesture_execution.launch.py

Launch the gesture_execution node with its package configuration.

Nodes started:
    gesture_execution/gesture_execution (node: gesture_action_server)
        Deictic, iconic, bowing, and nodding gesture action server. The node
        name differs from the package name and is the key the YAML config is
        written under.

Launch arguments:
    (none)

Configuration:
    config/gesture_execution_configuration.yaml
    data/gesture.yaml and data/pepper_topics.yaml load from fixed paths and
    are not exposed as parameters.

Prerequisites:
    naoqi_driver must be publishing /joint_states and accepting
    /joint_angles_trajectory; deictic gestures also need /localization from
    the SLAM stack to aim at a map-frame target.

Usage:
    ros2 launch gesture_execution gesture_execution.launch.py

The node's ROS interface is documented in gesture_execution_application.cpp.

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
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('gesture_execution'),
        'config',
        'gesture_execution_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='gesture_execution',
            executable='gesture_execution',
            name='gesture_action_server',
            parameters=[config],
            output='screen',
        ),
    ])
