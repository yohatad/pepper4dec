#!/usr/bin/env python3

"""age_gender_detection.launch.py

Launch the age_gender_detection node only.

Nodes started:
    face_detection/age_gender_detection (node: age_gender_detection)
        MiVOLO age/gender estimation for tracked persons showing mutual gaze.

Launch arguments:
    (none)

Configuration:
    config/age_gender_detection_configuration.yaml — model path and device,
    the input topics, and the estimation gating and re-estimation intervals.

Prerequisites:
    Assumes /camera/color/image_raw, /face_detection/data, and
    /person_detection/data are already being published (e.g. by
    face_detection.launch.py or a ROS2 bag).

Usage:
    ros2 launch face_detection age_gender_detection.launch.py

The node's ROS interface is documented in age_gender_detection_application.cpp.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 8, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory("face_detection"),
        "config", "age_gender_detection_configuration.yaml"
    )

    return LaunchDescription([
        Node(
            package="face_detection",
            executable="age_gender_detection",
            name="age_gender_detection",
            output="screen",
            parameters=[config_file],
        )
    ])
