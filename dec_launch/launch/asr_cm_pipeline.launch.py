"""
asr_cm_pipeline.launch.py: launch the ASR -> ConversationManager -> SpeechWithFeedback pipeline.

Nodes started:
    speech_event/speech_event (node: speech_recognition)
        Microphone capture, VAD, and ASR; publishes
        /speech_event/vad_speech_prob and serves /speech_recognition.
    conversation_manager/conversation_manager (node: conversation_manager)
        RAG + LLM; serves the /conversation_manager action. Gets its package
        YAML, then the collection_name/verbose_mode launch overrides.
    behavior_controller/behavior_controller (node: behavior_controller)
        BehaviorTree.CPP executor running asr_cm_tts_pipeline.xml. Gets its
        package YAML, with scenario_specification forced to
        asr_cm_tts_pipeline.
    nav2_lifecycle_manager/lifecycle_manager (node:
    lifecycle_manager_asr_cm_pipeline)
        Drives configure -> activate for the three lifecycle nodes above, in
        that order. Without it they stay UNCONFIGURED and the pipeline does
        nothing.

Launch arguments:
    collection_name (default: "upanzi_knowledge")
        ChromaDB collection used by conversation_manager.
    verbose_mode (default: "false")
        Verbose logging on conversation_manager. behavior_controller, which
        also declares verbose_mode, reads it from its own config file.

Configuration:
    Each node is given its package config YAML as parameters (the nodes do
    not read the files themselves; without it they run on code defaults,
    e.g. a localhost LLM and the lab_tour scenario).

Prerequisites (start separately before this launch):
    ros2 launch dec_launch naoqi_driver.launch.py nao_ip:=<PEPPER_IP>
    That wrapper turns the microphones on (/audio, which speech_event needs);
    naoqi_driver's own launch leaves audio off. SpeechWithFeedback connects to
    /naoqi_driver/speech_with_feedback, which naoqi_driver serves.

Usage:
    ros2 launch dec_launch asr_cm_pipeline.launch.py
    ros2 launch dec_launch asr_cm_pipeline.launch.py verbose_mode:=true

Each node's ROS interface is documented in its application file.

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
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Launch arguments ──────────────────────────────────────────────────────

    declare_collection_name = DeclareLaunchArgument(
        'collection_name',
        default_value='upanzi_knowledge',
        description='ChromaDB collection name used by conversation_manager'
    )

    declare_verbose = DeclareLaunchArgument(
        'verbose_mode',
        default_value='false',
        description='Enable verbose logging on conversation_manager; '
                    'behavior_controller takes verbose_mode from its own config'
    )

    # ── Package share directories ─────────────────────────────────────────────

    speech_event_share = get_package_share_directory('speech_event')
    conversation_manager_share = get_package_share_directory('conversation_manager')
    behavior_controller_share = get_package_share_directory('behavior_controller')

    # ── Config file paths ─────────────────────────────────────────────────────

    # speech_event uses a ROS2-format params file — pass via parameters=
    speech_event_config = os.path.join(
        speech_event_share, 'config', 'speech_event_configuration.yaml')

    conversation_manager_config = os.path.join(
        conversation_manager_share, 'config',
        'conversation_manager_configuration.yaml')

    behavior_controller_config = os.path.join(
        behavior_controller_share, 'config',
        'behavior_controller_configuration.yaml')

    # ── Node definitions ──────────────────────────────────────────────────────

    # 1. Speech event — microphone, VAD, ASR
    #    The node name MUST be speech_recognition: that is the key
    #    speech_event_configuration.yaml stores its parameters under, and ROS2
    #    matches parameter blocks by node name. Naming it anything else leaves
    #    the node running entirely on its code defaults.
    #    It declares no 'verbose_mode' parameter, so that argument is not
    #    passed here.
    speech_event_node = Node(
        package='speech_event',
        executable='speech_event',
        name='speech_recognition',
        output='screen',
        parameters=[speech_event_config],
    )

    # 2. Conversation manager — package YAML, then the launch overrides
    conversation_manager_node = Node(
        package='conversation_manager',
        executable='conversation_manager',
        name='conversation_manager',
        output='screen',
        parameters=[conversation_manager_config, {
            'collection_name': LaunchConfiguration('collection_name'),
            'verbose_mode':    LaunchConfiguration('verbose_mode'),
        }],
    )

    # 3. Behavior controller — package YAML, scenario pinned to this pipeline
    #    SpeechWithFeedback connects to /naoqi_driver/speech_with_feedback
    #    which is served by naoqi_driver (started separately).
    behavior_controller_node = Node(
        package='behavior_controller',
        executable='behavior_controller',
        name='behavior_controller',
        output='screen',
        parameters=[behavior_controller_config, {
            'scenario_specification': 'asr_cm_tts_pipeline',
        }],
    )

    # ── Launch description ────────────────────────────────────────────────────

    # All three nodes above are lifecycle nodes: they come up UNCONFIGURED and
    # do nothing until something drives them to active. dec_system.launch.py
    # uses nav2_lifecycle_manager for this; without one here the pipeline
    # launched cleanly and then sat idle. bond_timeout is 0.0 because these
    # nodes do not implement the bond protocol nav2's own C++ nodes use.
    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_asr_cm_pipeline',
        output='screen',
        parameters=[{
            'autostart': True,
            'bond_timeout': 0.0,
            'node_names': [
                'speech_recognition',
                'conversation_manager',
                'behavior_controller',
            ],
        }],
    )

    return LaunchDescription([
        declare_collection_name,
        declare_verbose,

        LogInfo(msg='[asr_cm_pipeline] Starting speech_event...'),
        speech_event_node,

        LogInfo(msg='[asr_cm_pipeline] Starting conversation_manager...'),
        conversation_manager_node,

        LogInfo(msg='[asr_cm_pipeline] Starting behavior_controller (asr_cm_tts_pipeline.xml)...'),
        behavior_controller_node,

        LogInfo(msg='[asr_cm_pipeline] Starting lifecycle manager (configure -> activate)...'),
        lifecycle_manager_node,
    ])
