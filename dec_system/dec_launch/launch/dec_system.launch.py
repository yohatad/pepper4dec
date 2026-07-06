#!/usr/bin/env python3
"""
dec_system.launch.py
Top-level launch file for the Pepper4DEC system.

Brings up every dec_system package (each with its own launch file, some of
which nest further launch files for their dependencies, e.g.
overt_attention's launch nests person_detection and face_detection) and then
sequences their lifecycle nodes from `unconfigured` to `active` in dependency
order via nav2_lifecycle_manager.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _include(package, launch_file):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)
        )
    )


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_navigation',
            default_value='true',
            description='Whether to bring up pepper_navigation (Nav2 navigation/localization stack)'
        ),

        # Perception: shared camera + person/face detection + overt attention
        _include('overt_attention', 'attention_system.launch.py'),

        # Localization (EKF pose estimate)
        _include('pepper_odom_anchor', 'localization.launch.py'),

        # Actuation
        _include('animate_behavior', 'animate_behavior.launch.py'),
        _include('gesture_execution', 'gesture_execution.launch.py'),

        # Speech / dialogue
        _include('speech_event', 'speech_event.launch.py'),
        _include('text_to_speech', 'text_to_speech_launch_robot.launch.py'),
        _include('conversation_manager', 'conversation_manager.launch.py'),

        # Navigation stack (own internal lifecycle manager)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('pepper_navigation'),
                    'launch', 'pepper_navigation.launch.py'
                )
            ),
            condition=IfCondition(LaunchConfiguration('enable_navigation'))
        ),

        # Top-level behavior orchestration (BT)
        _include('behavior_controller', 'behavior_controller.launch.py'),

        # Sequence configure -> activate for the custom lifecycle nodes above,
        # in dependency order. bond_timeout is disabled because these nodes
        # don't implement the bond protocol used by nav2's C++ lifecycle nodes.
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_dec_system',
            output='screen',
            parameters=[{
                'autostart': True,
                'bond_timeout': 0.0,
                'node_names': [
                    'person_detection',
                    'face_detection',
                    'unified_attention_node',
                    'robot_localization',
                    'animate_behavior',
                    'gesture_action_server',
                    'speech_recognition',
                    'text_to_speech',
                    'conversation_manager',
                    'behavior_controller',
                ],
            }],
        ),
    ])
