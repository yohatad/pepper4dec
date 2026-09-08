"""
asr_cm_pipeline.launch.py: launch the ASR -> ConversationManager -> SpeechWithFeedback pipeline.

Nodes started:
    speech_event/speech_event (node: speech_recognition)
        Microphone capture, VAD, and ASR; publishes
        /speech_event/vad_speech_prob and serves /speech_recognition.
    conversation_manager/conversation_manager (node: conversation_manager)
        RAG + LLM; serves the /conversation_manager action. Loads its own YAML
        config internally, so only the two overrides below are passed here.
    behavior_controller/behavior_controller (node: behavior_controller)
        BehaviorTree.CPP executor running asr_cm_tts_pipeline.xml. Also loads
        its config internally; no parameters are passed from this launch.
    nav2_lifecycle_manager/lifecycle_manager (node:
    lifecycle_manager_asr_cm_pipeline)
        Drives configure -> activate for the three lifecycle nodes above, in
        that order. Without it they stay UNCONFIGURED and the pipeline does
        nothing.

Launch arguments:
    collection_name (default: "upanzi_knowledge")
        ChromaDB collection used by conversation_manager.
    verbose (default: "false")
        Verbose logging on conversation_manager, the only node in this
        pipeline that declares the parameter.

Configuration:
    speech_event/config/speech_event_configuration.yaml is passed as
    parameters; conversation_manager and behavior_controller read their own
    config files at startup and must not be given them here.

Prerequisites (start separately before this launch):
    ros2 launch naoqi_driver naoqi_driver.launch.py nao_ip:=<PEPPER_IP>
    SpeechWithFeedback connects to /naoqi_driver/speech_with_feedback, which
    naoqi_driver serves.

Usage:
    ros2 launch dec_launch asr_cm_pipeline.launch.py
    ros2 launch dec_launch asr_cm_pipeline.launch.py verbose:=true

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

import os  # needed for speech_event config path
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
        'verbose',
        default_value='false',
        description='Enable verbose logging on conversation_manager '
                    '(the only node in this pipeline declaring it)'
    )

    # ── Package share directories ─────────────────────────────────────────────

    speech_event_share = get_package_share_directory('speech_event')

    # ── Config file paths ─────────────────────────────────────────────────────

    # speech_event uses a ROS2-format params file — pass via parameters=
    speech_event_config = os.path.join(
        speech_event_share, 'config', 'speech_event_configuration.yaml')

    # conversation_manager and behavior_controller load their own config files
    # internally at startup — do NOT pass them via parameters= here.

    # ── Node definitions ──────────────────────────────────────────────────────

    # 1. Speech event — microphone, VAD, ASR
    #    The node name MUST be speech_recognition: that is the key
    #    speech_event_configuration.yaml stores its parameters under, and ROS2
    #    matches parameter blocks by node name. Naming it anything else leaves
    #    the node running entirely on its code defaults.
    #    It declares no 'verbose' parameter, so the verbose argument is not
    #    passed here.
    speech_event_node = Node(
        package='speech_event',
        executable='speech_event',
        name='speech_recognition',
        output='screen',
        parameters=[speech_event_config],
    )

    # 2. Conversation manager — loads its YAML config internally
    conversation_manager_node = Node(
        package='conversation_manager',
        executable='conversation_manager',
        name='conversation_manager',
        output='screen',
        parameters=[{
            'collection_name': LaunchConfiguration('collection_name'),
            'verbose':         LaunchConfiguration('verbose'),
        }],
    )

    # 3. Behavior controller — loads its YAML config internally
    #    SpeechWithFeedback connects to /naoqi_driver/speech_with_feedback
    #    which is served by naoqi_driver (started separately).
    behavior_controller_node = Node(
        package='behavior_controller',
        executable='behavior_controller',
        name='behavior_controller',
        output='screen',
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
