#!/usr/bin/env python3

"""person_detection.launch.py

Launch the person_detection node only.

Nodes started:
    person_detection/person_detection (node: person_detection)
        YOLOv11 detection with ByteTrack tracking over the RGB-D stream.

Launch arguments:
    (none)

Configuration:
    config/person_detection_configuration.yaml — camera type, confidence and
    tracking thresholds, and the target class list.

Prerequisites:
    The camera feed is provided separately — a shared camera brought up by
    the overt_attention system (attention_system.launch.py) or a ROS2 bag —
    so this launch starts nothing but the detection node. It subscribes to
    /camera/color/image_raw and /camera/aligned_depth_to_color/image_raw.

Usage:
    ros2 launch person_detection person_detection.launch.py

The node's ROS interface is documented in person_detection_application.cpp.

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
