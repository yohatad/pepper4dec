#!/usr/bin/env python3

"""
face_detection_launch_robot.launch.py
ROS2 Launch file for Face Detection Robot
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.conditions import LaunchConfigurationEquals
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="172.29.111.230",
        description="Pepper robot IP address",
    )

    roscore_ip_arg = DeclareLaunchArgument(
        "roscore_ip",
        default_value="127.0.0.1",
        description="ROS core IP address",
    )

    robot_port_arg = DeclareLaunchArgument(
        "robot_port",
        default_value="9559",
        description="Pepper robot port",
    )

    network_interface_arg = DeclareLaunchArgument(
        "network_interface",
        default_value="wlp0s20f3",
        description="Network interface for Pepper robot connection",
    )

    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="naoqi_driver",
        description="Namespace for naoqi driver",
    )

    camera_arg = DeclareLaunchArgument(
        "camera",
        default_value="realsense",
        description="Set 'pepper' for Pepper, 'realsense' for RealSense",
    )

    # Launch configurations
    robot_ip = LaunchConfiguration("robot_ip")
    roscore_ip = LaunchConfiguration("roscore_ip")
    robot_port = LaunchConfiguration("robot_port")
    network_interface = LaunchConfiguration("network_interface")
    namespace = LaunchConfiguration("namespace")
    camera = LaunchConfiguration("camera")

    # # Face Detection Node (no ROS param injection here)
    # face_detection_node = Node(
    #     package="cssr_system",
    #     executable="face_detection",
    #     name="faceDetection",
    #     output="screen",
    # )

    # Pepper Robot group (only if camera == pepper)
    pepper_group = GroupAction(
        condition=LaunchConfigurationEquals("camera", "pepper"),
        actions=[
            Node(
                package="naoqi_driver",
                executable="naoqi_driver_node",
                namespace=namespace,
                arguments=[
                    TextSubstitution(text="--qi-url=tcp://"),
                    robot_ip,
                    TextSubstitution(text=":"),
                    robot_port,
                    TextSubstitution(text=" --roscore_ip="),
                    roscore_ip,
                    TextSubstitution(text=" --network_interface="),
                    network_interface,
                    TextSubstitution(text=" --namespace="),
                    namespace,
                ],
                output="screen",
            )
        ],
    )

    # RealSense Camera group (only if camera == realsense)
    realsense_group = GroupAction(
        condition=LaunchConfigurationEquals("camera", "realsense"),
        actions=[
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                namespace="",   # this sets the namespace once
                parameters=[{
                    "rgb_camera.width": 640,
                    "rgb_camera.height": 480,
                    "rgb_camera.fps": 15,
                    "depth_module.width": 640,
                    "depth_module.height": 480,
                    "depth_module.fps": 15,
                    "enable_sync": True,
                    "align_depth.enable": True,
                }],
                output="screen",
)
        ],
    )

    return LaunchDescription([
        robot_ip_arg,
        roscore_ip_arg,
        robot_port_arg,
        network_interface_arg,
        namespace_arg,
        camera_arg,
        pepper_group,
        realsense_group,
    ])
