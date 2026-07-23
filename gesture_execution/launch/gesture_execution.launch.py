#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('gesture_execution'),
        'config',
        'gesture_execution_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='gesture_execution',
            executable='gesture_execution',
            name='gesture_action_server',
            parameters=[config],
            output='screen',
        ),
    ])
