#!/usr/bin/env python3
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
