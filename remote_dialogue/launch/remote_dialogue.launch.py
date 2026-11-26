#!/usr/bin/env python3
"""remote_dialogue.launch.py

Launch the remote_dialogue node with its package configuration.

Nodes started:
    remote_dialogue/remote_dialogue (node: remote_dialogue)
        Pepper client for the /ws/pepper voice server. A lifecycle node: it
        stays UNCONFIGURED until a lifecycle manager (or `ros2 lifecycle set`)
        configures and activates it.

Launch arguments:
    verbose_mode (default: "false")
        Log every server message.

Configuration:
    config/remote_dialogue_configuration.yaml

Prerequisites:
    export REMOTE_DIALOGUE_TOKEN=<token>
    export REMOTE_DIALOGUE_URL=ws://<laptop-ip>:<port>/ws/pepper  (or set server_url)
    ros2 launch dec_launch naoqi_driver.launch.py nao_ip:=<PEPPER_IP>
        (microphones on /audio, speaker via /naoqi_driver/send_audio_buffer)

Usage:
    ros2 launch remote_dialogue remote_dialogue.launch.py
    ros2 lifecycle set /remote_dialogue configure
    ros2 lifecycle set /remote_dialogue activate

The node's ROS interface is documented in remote_dialogue_application.py.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 26, 2026
Version: v1.0

Copyright (C) 2026 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('remote_dialogue'),
        'config',
        'remote_dialogue_configuration.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'verbose_mode', default_value='false',
            description='Log every server message'),
        Node(
            package='remote_dialogue',
            executable='remote_dialogue',
            name='remote_dialogue',
            parameters=[config, {
                'verbose_mode': ParameterValue(
                    LaunchConfiguration('verbose_mode'), value_type=bool),
            }],
            output='screen',
        ),
    ])
