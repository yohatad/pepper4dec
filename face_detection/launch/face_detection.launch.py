#!/usr/bin/env python3

"""face_detection.launch.py

Launch the face_detection node, optionally with its camera driver.

Nodes started:
    face_detection/face_detection (node: face_detection)
        YOLO face detection plus SixDRepNet head pose and mutual gaze.
    naoqi_driver/naoqi_driver_node or realsense2_camera/realsense2_camera_node
        Only when launch_camera is true. Which one starts is read from the
        camera parameter in the YAML config, not from a launch argument.

Launch arguments:
    launch_camera (default: "false")
        Start the camera driver as well. Leave false when the frames come
        from a ROS2 bag or from another launch file.

Configuration:
    config/face_detection_configuration.yaml — read twice: passed to the node
    as parameters, and parsed here to decide which camera driver to start.

Prerequisites:
    With launch_camera false, /camera/color/image_raw and the matching depth
    topic must already be published. The naoqi_driver branch hard-codes the
    robot's qi-url and network interface, so those need editing for a
    different robot or network.

Usage:
    ros2 launch face_detection face_detection.launch.py launch_camera:=true

The node's ROS interface is documented in face_detection_application.cpp.

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
from launch.actions import OpaqueFunction
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import yaml
import os


def launch_setup(context, *args, **kwargs):
    # Get launch argument value
    launch_camera = LaunchConfiguration('launch_camera').perform(context)
    launch_camera_bool = launch_camera.lower() in ['true', '1', 'yes', 'on']

    # Load the camera type from the YAML file
    config_file = os.path.join(
        os.getenv("COLCON_PREFIX_PATH").split(":")[0],  # first install dir
        "face_detection", "share", "face_detection", "config", "face_detection_configuration.yaml"
    )
    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    node_params = params.get("face_detection", {}).get("ros__parameters", {})
    camera_value = node_params.get("camera", "realsense")  # default fallback

    actions = []

    # Only launch camera if launch_camera argument is true
    if launch_camera_bool:
        if camera_value == "pepper":
            actions.append(
                Node(
                    package="naoqi_driver",
                    executable="naoqi_driver_node",
                    namespace="naoqi_driver",
                    arguments=[
                        "--qi-url=tcp://172.29.111.230:9559",
                        "--roscore_ip=127.0.0.1",
                        "--network_interface=wlp0s20f3",
                        "--namespace=naoqi_driver",
                    ],
                    output="screen",
                )
            )
        elif camera_value == "realsense":
            actions.append(
                Node(
                    package="realsense2_camera",
                    executable="realsense2_camera_node",
                    namespace="",
                    parameters=[{
                        "camera_name": "",
                        "rgb_camera.color_profile": "640x480x15",
                        "depth_module.depth_profile": "640x480x15",
                        "align_depth.enable": True,
                        "enable_sync": True,
                        "enable_infra1": False,
                        "enable_infra2": False,
                        "enable_accel": True,
                        "enable_gyro": True,
                        "publish_tf": True,

                        # Set QoS to BEST_EFFORT
                        'qos_overrides./camera.aligned_depth_to_color.image_raw'
                        '.publisher.reliability': 'best_effort',
                        'qos_overrides./camera.color.image_raw.publisher.reliability':
                        'best_effort',
                        'qos_overrides./camera.depth.image_rect_raw.publisher.reliability':
                        'best_effort',
                    }],
                    output="screen",
                )
            )
    else:
        # Log that camera launch is skipped
        print("Camera launch is disabled (launch_camera=false). "
              "Assuming topics are available from ROS2 bag or other source.")

    # Add face detection node
    actions.append(
        Node(
            package="face_detection",
            executable="face_detection",
            name="face_detection",
            output="screen",
            parameters=[config_file],
        )
    )

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_camera',
            default_value='false',
            description='Whether to launch the camera driver (set to false when using ROS2 bags)'
        ),
        OpaqueFunction(function=launch_setup)
    ])
