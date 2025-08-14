#!/usr/bin/env python3

"""
face_detection_launch_robot.launch.py ROS2 Launch file for Face Detection Robot

Copyright (C) 2023 CSSR4Africa Consortium

This project is funded by the African Engineering and Technology Network (Afretec)
Inclusive Digital Transformation Research Grant Programme.

Website: www.cssr4africa.org

This program comes with ABSOLUTELY NO WARRANTY.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.conditions import LaunchConfigurationEquals
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Declare launch arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='172.29.111.230',
        description='Pepper robot IP address'
    )

    roscore_ip_arg = DeclareLaunchArgument(
        'roscore_ip',
        default_value='127.0.0.1',
        description='ROS core IP address'
    )

    robot_port_arg = DeclareLaunchArgument(
        'robot_port',
        default_value='9559',
        description='Pepper robot port'
    )

    network_interface_arg = DeclareLaunchArgument(
        'network_interface',
        default_value='wlp0s20f3',
        description='Network interface for Pepper robot connection'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='naoqi_driver',
        description='Namespace for naoqi driver'
    )

    camera_arg = DeclareLaunchArgument(
        'camera',
        default_value='realsense',
        description="Set 'pepper' for Pepper, 'realsense' for RealSense"
    )

    # Get launch configurations
    robot_ip = LaunchConfiguration('robot_ip')
    roscore_ip = LaunchConfiguration('roscore_ip')
    robot_port = LaunchConfiguration('robot_port')
    network_interface = LaunchConfiguration('network_interface')
    namespace = LaunchConfiguration('namespace')
    camera = LaunchConfiguration('camera')

    
    cfg_path = os.path.join(
        FindPackageShare('cssr_system').find('cssr_system'),
        'face_detection', 'config', 'face_detection_configuration.yaml'
    )
    
    face_detection_node = Node(
    package='cssr_system',
    executable='face_detection_application.py',
    name='faceDetection',
    parameters=[
        {'config_path': cfg_path},
        {'camera': camera},
        {'unit_tests': False},
    ],
    output='screen',
    )

    # Pepper Robot group (only if camera == pepper)
    pepper_group = GroupAction(
        condition=LaunchConfigurationEquals('camera', 'pepper'),
        actions=[
            Node(
                package='naoqi_driver',
                executable='naoqi_driver_node',
                name=namespace,
                arguments=[
                    TextSubstitution(text='--qi-url=tcp://'),
                    robot_ip,
                    TextSubstitution(text=':'),
                    robot_port,
                    TextSubstitution(text=' --roscore_ip='),
                    roscore_ip,
                    TextSubstitution(text=' --network_interface='),
                    network_interface,
                    TextSubstitution(text=' --namespace='),
                    namespace
                ],
                output='screen'
            )
        ]
    )

    # RealSense Camera group (only if camera == realsense)
    realsense_group = GroupAction(
        condition=LaunchConfigurationEquals('camera', 'realsense'),
        actions=[
            Node(
                package='realsense2_camera',
                executable='realsense2_camera_node',
                name='camera',
                parameters=[
                    {'color_width': 640},
                    {'color_height': 480},
                    {'color_fps': 15},
                    {'depth_width': 640},
                    {'depth_height': 480},
                    {'depth_fps': 15},
                    {'align_depth.enable': True},
                    {'enable_sync': True}
                ],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        # Declare arguments
        robot_ip_arg,
        roscore_ip_arg,
        robot_port_arg,
        network_interface_arg,
        namespace_arg,
        camera_arg,

        # Launch nodes
        face_detection_node,
        pepper_group,
        realsense_group
    ])
