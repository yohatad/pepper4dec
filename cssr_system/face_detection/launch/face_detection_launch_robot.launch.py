#!/usr/bin/env python3

"""
face_detection_launch_robot.launch.py
ROS2 Launch file for Face Detection Robot
"""

from launch import LaunchDescription
from launch.actions import GroupAction, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import yaml
import os


def launch_setup(context, *args, **kwargs):
    # Load the camera type from the YAML file
    config_file = os.path.join(
        os.getenv("COLCON_PREFIX_PATH").split(":")[0],  # first install dir
        "face_detection", "share", "face_detection", "config", "face_detection_configuration.yaml"
    )
    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    camera_value = params.get("camera", "realsense")  # default fallback

    actions = []

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
                    'qos_overrides./camera.aligned_depth_to_color.image_raw.publisher.reliability': 'best_effort',
                    'qos_overrides./camera.color.image_raw.publisher.reliability': 'best_effort',
                    'qos_overrides./camera.depth.image_rect_raw.publisher.reliability': 'best_effort',
                }],
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])