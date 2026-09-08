#!/usr/bin/env python3
"""text_to_speech.launch.py

Launch the text_to_speech node with its package configuration.

Nodes started:
    text_to_speech/text_to_speech (node: text_to_speech)
        Sentence-streaming speech synthesis and playback action server.

Launch arguments:
    (none)

Configuration:
    config/text_to_speech_configuration.yaml — selects the synthesis backend
    (engine) and its voice, rate, and playback settings.

Prerequisites:
    Depends on the configured engine: the naoqi_ros backend needs
    naoqi_driver; the kokoro_* backends need the local Kokoro-82M model; the
    elevenlabs_* backends need ELEVENLABS_API_KEY and network access. The
    *_pepper variants additionally use naoqi_driver's audio services to play
    on the robot.

Usage:
    ros2 launch text_to_speech text_to_speech.launch.py

The node's ROS interface is documented in text_to_speech_application.py.

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
        get_package_share_directory('text_to_speech'),
        'config',
        'text_to_speech_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='text_to_speech',
            executable='text_to_speech',
            name='text_to_speech',
            parameters=[config],
            output='screen',
        ),
    ])
