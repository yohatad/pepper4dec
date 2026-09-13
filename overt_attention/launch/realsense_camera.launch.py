#!/usr/bin/env python3
"""realsense_camera.launch.py

Launch the Intel RealSense camera driver only.

Nodes started:
    realsense2_camera/realsense2_camera_node
        Aligned RGB-D stream with the profiles and QoS overrides the
        perception nodes expect.

Launch arguments:
    (none)

Configuration:
    Camera settings are set inline as node parameters here, not in a YAML
    config file.

Prerequisites:
    A RealSense device on USB and the realsense2_camera package installed.

Usage:
    ros2 launch overt_attention realsense_camera.launch.py

No other nodes are started; used by attention_system.launch.py to bring up
the shared camera feed for person/face detection and attention.

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


def generate_launch_description():

    return LaunchDescription([

        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            namespace="",
            output="screen",
            parameters=[{
                "camera_name": "",

                # --- Stream profiles ---
                "rgb_camera.color_profile": "640x480x30",
                "depth_module.depth_profile": "640x480x30",
                "align_depth.enable": True,
                "enable_sync": True,

                # --- Sensors ---
                "enable_infra1": False,
                "enable_infra2": False,
                "enable_accel": True,
                "enable_gyro": True,

                # --- TF ---
                "publish_tf": True,

                # --- QoS (BEST_EFFORT for vision pipelines) ---
                "qos_overrides./camera/color/image_raw.publisher.reliability": "best_effort",
                "qos_overrides./camera/aligned_depth_to_color/image_raw.publisher.reliability":
                    "best_effort",
                "qos_overrides./camera/depth/image_rect_raw.publisher.reliability": "best_effort",
            }]
        )
    ])
