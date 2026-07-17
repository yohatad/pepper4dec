"""Launch file for animate_behavior node."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('animate_behavior'),
        'config',
        'animate_behavior_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='animate_behavior',
            executable='animate_behavior',
            name='animate_behavior',
            output='screen',
            parameters=[config],
        )
    ])
