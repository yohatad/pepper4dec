#!/usr/bin/env python3
"""
Launch the Intel RealSense camera driver only.

No other nodes are started; used by attention_system.launch.py to bring up
the shared camera feed for person/face detection and attention.
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
