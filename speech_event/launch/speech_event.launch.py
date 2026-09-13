"""speech_event.launch.py

Launch all three speech_event nodes against the shared package configuration.

Nodes started:
    speech_event/speech_event (node: speech_recognition)
        Silero VAD + Whisper transcription; serves the /speech_recognition
        action and the /speech_event/set_enabled service.
    speech_event/speech_event_localization (node: sound_localization)
        SRP-PHAT sound-source azimuth from the 4-microphone array.
    speech_event/speech_event_recorder (node: audio_recorder)
        Records the raw microphone stream to WAV. Note that this launch
        starts the recorder unconditionally, so every run writes audio files
        under the path set by output_base; comment the node out if that is
        not wanted.

Launch arguments:
    (none)

Configuration:
    config/speech_event_configuration.yaml — one block per node, keyed by
    node name (speech_recognition, sound_localization, audio_recorder).

Prerequisites:
    naoqi_driver must be publishing /naoqi_driver/audio. The speech_recognition
    node loads Whisper on configure and expects a CUDA device unless the
    device parameter is set to "cpu".

Usage:
    ros2 launch speech_event speech_event.launch.py

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


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('speech_event'),
        'config',
        'speech_event_configuration.yaml'
    )

    return launch.LaunchDescription([
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
        ),
    ])
