#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('conversation_manager'),
        'config',
        'converation_manager_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='conversation_manager',
            executable='conversation_manager',
            name='conversation_manager',
            parameters=[config],
            output='screen',
        ),
    ])
