"""
remote_dialogue_pipeline.launch.py: launch the remote voice server pipeline.

The /ws/pepper voice server does speech recognition, turn-taking, the
language model and speech synthesis on a GPU server. On the robot side this
launch runs only the client and the behavior tree that drives it, so it
replaces asr_cm_pipeline.launch.py's speech_event + conversation_manager +
SpeechWithFeedback. Do not run speech_event alongside it.

With attention:=true it also brings up the overt attention stack and runs
remote_dialogue_attention.xml instead: Pepper follows faces, starts a
conversation (with a greeting) when a visitor makes eye contact, and keeps
looking at them while they talk.

Nodes started:
    remote_dialogue/remote_dialogue (node: remote_dialogue)
        Streams one microphone channel from /audio to the server, plays the
        replies through /naoqi_driver/send_audio_buffer, serves the
        /remote_dialogue action.
    behavior_controller/behavior_controller (node: behavior_controller)
        BehaviorTree.CPP executor running remote_dialogue_pipeline.xml
        (remote_dialogue_attention.xml with attention:=true).
    nav2_lifecycle_manager/lifecycle_manager (node:
    lifecycle_manager_remote_dialogue_pipeline)
        Drives configure -> activate for the lifecycle nodes above (and, with
        attention:=true, person_detection, face_detection, overt_attention).

Launch files included (attention:=true only):
    overt_attention/attention_system.launch.py — camera, person and face
        detection, saliency, and the attention controller.

Launch arguments:
    language (default: "eng")
        Session language: "eng", "kin", or another server-supported code.
    verbose_mode (default: "false")
        Verbose logging on remote_dialogue.
    attention (default: "false")
        Add overt attention and start conversations on eye contact.
    enable_viz (default: "false")
        attention:=true only: start the attention visualization node.

Configuration:
    remote_dialogue/config/remote_dialogue_configuration.yaml and
    behavior_controller/config/behavior_controller_configuration.yaml, with
    scenario_specification pinned to remote_dialogue_pipeline.

Prerequisites:
    export REMOTE_DIALOGUE_TOKEN=<token from the server team>
    export REMOTE_DIALOGUE_URL=ws://<laptop-ip>:<port>/ws/pepper
    ros2 launch dec_launch naoqi_driver.launch.py nao_ip:=<PEPPER_IP>
        (turns the microphones on; serves send_audio_buffer). With
        attention:=true add use_camera:=true, after running
        ~/start_camera.sh on the robot.

Usage:
    ros2 launch dec_launch remote_dialogue_pipeline.launch.py
    ros2 launch dec_launch remote_dialogue_pipeline.launch.py language:=kin
    ros2 launch dec_launch remote_dialogue_pipeline.launch.py attention:=true

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
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ATTENTION_NODES = ['person_detection', 'face_detection', 'overt_attention']


def _nodes(context):
    attention = LaunchConfiguration('attention').perform(context).lower() in ('true', '1')
    scenario = 'remote_dialogue_attention' if attention else 'remote_dialogue_pipeline'

    remote_dialogue_config = os.path.join(
        get_package_share_directory('remote_dialogue'), 'config',
        'remote_dialogue_configuration.yaml')

    behavior_controller_config = os.path.join(
        get_package_share_directory('behavior_controller'), 'config',
        'behavior_controller_configuration.yaml')

    actions = []
    if attention:
        actions += [
            LogInfo(msg='[remote_dialogue_pipeline] Starting the overt attention stack...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(
                    get_package_share_directory('overt_attention'), 'launch',
                    'attention_system.launch.py')),
                launch_arguments={'enable_viz': LaunchConfiguration('enable_viz')}.items()),
        ]

    actions += [
        LogInfo(msg='[remote_dialogue_pipeline] Starting remote_dialogue...'),
        Node(
            package='remote_dialogue',
            executable='remote_dialogue',
            name='remote_dialogue',
            output='screen',
            parameters=[remote_dialogue_config, {
                'language': LaunchConfiguration('language'),
                'verbose_mode': ParameterValue(
                    LaunchConfiguration('verbose_mode'), value_type=bool),
            }],
        ),

        LogInfo(msg='[remote_dialogue_pipeline] Starting behavior_controller '
                    f'({scenario}.xml)...'),
        Node(
            package='behavior_controller',
            executable='behavior_controller',
            name='behavior_controller',
            output='screen',
            parameters=[behavior_controller_config, {
                'scenario_specification': scenario,
            }],
        ),

        # All are lifecycle nodes; without a manager they stay UNCONFIGURED.
        # Perception first, then remote_dialogue, so every server the
        # behavior tree uses exists when the tree is built. bond_timeout 0:
        # these nodes do not implement nav2's bond.
        LogInfo(msg='[remote_dialogue_pipeline] Starting lifecycle manager...'),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_remote_dialogue_pipeline',
            output='screen',
            parameters=[{
                'autostart': True,
                'bond_timeout': 0.0,
                'node_names': (ATTENTION_NODES if attention else []) + [
                    'remote_dialogue',
                    'behavior_controller',
                ],
            }],
        ),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'language', default_value='eng',
            description='Voice server session language (eng, kin, amh, swh, fra)'),
        DeclareLaunchArgument(
            'verbose_mode', default_value='false',
            description='Enable verbose logging on remote_dialogue'),
        DeclareLaunchArgument(
            'attention', default_value='false',
            description='Add overt attention; start conversations on eye contact'),
        DeclareLaunchArgument(
            'enable_viz', default_value='false',
            description='attention:=true only: start the attention visualization node'),
        OpaqueFunction(function=_nodes),
    ])
