"""
remote_dialogue_pipeline.launch.py: launch the remote voice server pipeline.

The /ws/pepper voice server does speech recognition, turn-taking, the
language model and speech synthesis on a GPU server. On the robot side this
launch runs only the client and the behavior tree that drives it, so it
replaces asr_cm_pipeline.launch.py's speech_event + conversation_manager +
SpeechWithFeedback. Do not run speech_event alongside it.

Nodes started:
    remote_dialogue/remote_dialogue (node: remote_dialogue)
        Streams one microphone channel from /audio to the server, plays the
        replies through /naoqi_driver/send_audio_buffer, serves the
        /remote_dialogue action.
    behavior_controller/behavior_controller (node: behavior_controller)
        BehaviorTree.CPP executor running remote_dialogue_pipeline.xml.
    nav2_lifecycle_manager/lifecycle_manager (node:
    lifecycle_manager_remote_dialogue_pipeline)
        Drives configure -> activate for the two lifecycle nodes above.

Launch arguments:
    language (default: "eng")
        Session language: "eng", "kin", or another server-supported code.
    verbose_mode (default: "false")
        Verbose logging on remote_dialogue.

Configuration:
    remote_dialogue/config/remote_dialogue_configuration.yaml and
    behavior_controller/config/behavior_controller_configuration.yaml, with
    scenario_specification pinned to remote_dialogue_pipeline.

Prerequisites:
    export REMOTE_DIALOGUE_TOKEN=<token from the server team>
    export REMOTE_DIALOGUE_URL=ws://<laptop-ip>:<port>/ws/pepper
    ros2 launch dec_launch naoqi_driver.launch.py nao_ip:=<PEPPER_IP>
        (turns the microphones on; serves send_audio_buffer)

Usage:
    ros2 launch dec_launch remote_dialogue_pipeline.launch.py
    ros2 launch dec_launch remote_dialogue_pipeline.launch.py language:=kin

Each node's ROS interface is documented in its application file.

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
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    declare_language = DeclareLaunchArgument(
        'language',
        default_value='eng',
        description='Voice server session language (eng, kin, amh, swh, fra)'
    )

    declare_verbose = DeclareLaunchArgument(
        'verbose_mode',
        default_value='false',
        description='Enable verbose logging on remote_dialogue'
    )

    remote_dialogue_config = os.path.join(
        get_package_share_directory('remote_dialogue'), 'config',
        'remote_dialogue_configuration.yaml')

    behavior_controller_config = os.path.join(
        get_package_share_directory('behavior_controller'), 'config',
        'behavior_controller_configuration.yaml')

    remote_dialogue_node = Node(
        package='remote_dialogue',
        executable='remote_dialogue',
        name='remote_dialogue',
        output='screen',
        parameters=[remote_dialogue_config, {
            'language': LaunchConfiguration('language'),
            'verbose_mode': ParameterValue(
                LaunchConfiguration('verbose_mode'), value_type=bool),
        }],
    )

    behavior_controller_node = Node(
        package='behavior_controller',
        executable='behavior_controller',
        name='behavior_controller',
        output='screen',
        parameters=[behavior_controller_config, {
            'scenario_specification': 'remote_dialogue_pipeline',
        }],
    )

    # Both are lifecycle nodes; without a manager they stay UNCONFIGURED.
    # remote_dialogue comes first so its action server exists when the
    # behavior tree is built. bond_timeout 0: neither implements nav2's bond.
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_remote_dialogue_pipeline',
        output='screen',
        parameters=[{
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': [
                'remote_dialogue',
                'behavior_controller',
            ],
        }],
    )

    return LaunchDescription([
        declare_language,
        declare_verbose,

        LogInfo(msg='[remote_dialogue_pipeline] Starting remote_dialogue...'),
        remote_dialogue_node,

        LogInfo(msg='[remote_dialogue_pipeline] Starting behavior_controller '
                    '(remote_dialogue_pipeline.xml)...'),
        behavior_controller_node,

        LogInfo(msg='[remote_dialogue_pipeline] Starting lifecycle manager...'),
        lifecycle_manager_node,
    ])
