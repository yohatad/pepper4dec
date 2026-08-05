#!/usr/bin/env python3

"""
Launch the age_gender_detection node.

Assumes /camera/color/image_raw, /face_detection/data, and
/person_detection/data are already being published (e.g. by
face_detection_launch_robot.launch.py or a ROS2 bag).
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
