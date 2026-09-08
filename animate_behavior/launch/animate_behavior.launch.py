"""animate_behavior.launch.py

Launch the animate_behavior node with its package configuration.

Nodes started:
    animate_behavior/animate_behavior (node: animate_behavior)
        Idle gesture, body rotation, and face-LED animation action server.

Launch arguments:
    (none)

Configuration:
    config/animate_behavior_configuration.yaml

Prerequisites:
    naoqi_driver must be running: the node subscribes to /joint_states,
    publishes /joint_angles and /cmd_vel, and drives the face LEDs through
    the /naoqi_driver/run_led action server.

Usage:
    ros2 launch animate_behavior animate_behavior.launch.py

The node's ROS interface is documented in animate_behavior_application.cpp.

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
        get_package_share_directory('animate_behavior'),
        'config',
        'animate_behavior_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='animate_behavior',
            executable='animate_behavior',
            name='animate_behavior',
            output='screen',
            parameters=[config],
        )
    ])
