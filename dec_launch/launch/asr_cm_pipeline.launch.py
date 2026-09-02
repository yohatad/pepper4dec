"""
Launch the ASR → ConversationManager → SpeechWithFeedback pipeline.

asr_cm_pipeline.launch.py

Nodes started:
  1. speech_event        – microphone capture, VAD, and ASR
                           publishes /speech_event/vad_speech_prob
                           serves   /speech_recognition_action
  2. conversation_manager – RAG + LLM; serves /conversation_manager action
  3. behavior_controller  – BehaviorTree.CPP executor running
                            asr_cm_tts_pipeline.xml

Prerequisites (start separately before this launch):
  ros2 launch naoqi_driver naoqi_driver.launch.py nao_ip:=<PEPPER_IP>

Usage:
  ros2 launch dec_launch asr_cm_pipeline.launch.py
  ros2 launch dec_launch asr_cm_pipeline.launch.py nao_ip:=10.0.1.230
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
        description='Enable verbose logging on all nodes'
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
    speech_event_node = Node(
        package='speech_event',
        executable='speech_event',
        name='speech_event',
        output='screen',
        parameters=[
            speech_event_config,
            {'verbose': LaunchConfiguration('verbose')},
        ],
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

    return LaunchDescription([
        declare_collection_name,
        declare_verbose,

        LogInfo(msg='[asr_cm_pipeline] Starting speech_event...'),
        speech_event_node,

        LogInfo(msg='[asr_cm_pipeline] Starting conversation_manager...'),
        conversation_manager_node,

        LogInfo(msg='[asr_cm_pipeline] Starting behavior_controller (asr_cm_tts_pipeline.xml)...'),
        behavior_controller_node,
    ])
