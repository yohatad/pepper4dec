#!/usr/bin/env python3
"""
dummy_localization.launch.py

Launch file for dummy robot localization with configurable initial position
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'initial_x',
            default_value='0.0',
            description='Initial X position in meters'
        ),
        DeclareLaunchArgument(
            'initial_y',
            default_value='0.0',
            description='Initial Y position in meters'
        ),
        DeclareLaunchArgument(
            'initial_theta',
            default_value='0.0',
            description='Initial theta (yaw) in radians'
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/odom',
            description='Input odometry topic'
        ),
        DeclareLaunchArgument(
            'pose_topic',
            default_value='/robotLocalization/pose',
            description='Output absolute pose topic'
        ),
        DeclareLaunchArgument(
            'publish_rate',
            default_value='10.0',
            description='Publish rate in Hz'
        ),

        # Localization node
        Node(
            package='robot_localization',  # Change to your package name
            executable='robot_localization',
            name='robot_localization',
            output='screen',
            parameters=[{
                'initial_x': LaunchConfiguration('initial_x'),
                'initial_y': LaunchConfiguration('initial_y'),
                'initial_theta': LaunchConfiguration('initial_theta'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'pose_topic': LaunchConfiguration('pose_topic'),
                'publish_rate': LaunchConfiguration('publish_rate'),
            }]
        ),
    ])