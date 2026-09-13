#!/usr/bin/env python3
"""behavior_controller.launch.py

Launch the behavior_controller node with its package configuration.

Nodes started:
    behavior_controller/behavior_controller (node: behavior_controller)
        BehaviorTree.CPP tour-guide mission executor.

Launch arguments:
    (none)

Configuration:
    config/behavior_controller_configuration.yaml — selects the scenario and
    the culture/environment knowledge bases loaded from data/.

Prerequisites:
    The behavior tree ticks action and service clients across the system
    (animate_behavior, gesture_execution, Nav2, speech_event,
    conversation_manager, text_to_speech, overt_attention). Nodes that are
    not running make the corresponding BT nodes fail rather than block, so
    start this last — dec_system.launch.py sequences it for you.

Usage:
    ros2 launch behavior_controller behavior_controller.launch.py

The node's ROS interface is documented in behavior_controller_application.cpp.

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
        get_package_share_directory('behavior_controller'),
        'config',
        'behavior_controller_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='behavior_controller',
            executable='behavior_controller',
            name='behavior_controller',
            parameters=[config],
            output='screen',
        ),
    ])
