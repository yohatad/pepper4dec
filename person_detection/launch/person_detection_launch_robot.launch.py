#!/usr/bin/env python3

"""
Launch the person_detection node only.

The camera feed is provided separately — a shared camera brought up by the
overt_attention system (``attention_system.launch.py``) or a ROS 2 bag — so
this launch starts nothing but the person_detection node itself. It subscribes
to ``/camera/color/image_raw`` and ``/camera/aligned_depth_to_color/image_raw``.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('person_detection'),
        'config', 'person_detection_configuration.yaml'
    )

    return LaunchDescription([
        Node(
            package='person_detection',
            executable='person_detection',
            name='person_detection',
            output='screen',
            parameters=[config_file],
        )
    ])
