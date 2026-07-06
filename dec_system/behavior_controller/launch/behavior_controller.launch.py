#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('behavior_controller'),
        'config',
        'behavior_controller_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='behavior_controller',
            executable='behavior_controller',
            name='behavior_controller',
            parameters=[config],
            output='screen',
        ),
    ])
