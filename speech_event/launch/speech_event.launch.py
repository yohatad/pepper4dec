"""speech_event.launch.py

Launch the speech_event nodes against the shared package configuration.

Nodes started:
    speech_event/speech_event (node: speech_recognition)
        Silero VAD + Whisper transcription; serves the /speech_recognition
        action and the /speech_event/set_enabled service.
    speech_event/speech_event_localization (node: sound_localization)
        SRP-PHAT sound-source azimuth from the 4-microphone array.
    speech_event/speech_event_recorder (node: audio_recorder)
        Only with record_audio:=true. Records the raw microphone stream to
        WAV, about 1.4 GB an hour at 48 kHz x 4 channels, under
        ~/.ros/speech_event/ by default and for at most max_seconds (1 hour
        in the shipped configuration).

Launch arguments:
    record_audio (default: "false")
        Also start audio_recorder.

Configuration:
    config/speech_event_configuration.yaml — one block per node, keyed by
    node name (speech_recognition, sound_localization, audio_recorder).

Prerequisites:
    naoqi_driver must be publishing /audio. The speech_recognition
    node loads Whisper on configure and expects a CUDA device unless the
    device parameter is set to "cpu".

Usage:
    ros2 launch speech_event speech_event.launch.py
    ros2 launch speech_event speech_event.launch.py record_audio:=true

Each node's ROS interface is documented in its application file
(speech_event_application.py, speech_event_localization.py,
speech_event_recorder.py).

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
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('speech_event'),
        'config',
        'speech_event_configuration.yaml'
    )

    return launch.LaunchDescription([
        DeclareLaunchArgument(
            'record_audio',
            default_value='false',
            description='Also start audio_recorder, which writes the raw '
                        'microphone stream to WAV (about 1.4 GB an hour)'
        ),
        launch_ros.actions.Node(
            package='speech_event',
            executable='speech_event',
            name='speech_recognition',
            parameters=[config],
            output='screen',
        ),
        launch_ros.actions.Node(
            package='speech_event',
            executable='speech_event_localization',
            name='sound_localization',
            parameters=[config],
            output='screen',
        ),
        launch_ros.actions.Node(
            package='speech_event',
            executable='speech_event_recorder',
            name='audio_recorder',
            parameters=[config],
            output='screen',
            condition=IfCondition(LaunchConfiguration('record_audio')),
        ),
    ])
