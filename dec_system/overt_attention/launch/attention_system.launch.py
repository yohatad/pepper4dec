#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('overt_attention')
    params_file = os.path.join(pkg_dir, 'config', 'overt_attention_configuration.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=params_file,
            description='Path to parameters file'
        ),

        DeclareLaunchArgument(
            'enable_viz',
            default_value='true',
            description='Enable visualization'
        ),

        # Shared camera feed for person/face detection and attention
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'realsense_camera.launch.py')
            )
        ),

        # Person detection (nested dependency)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('person_detection'),
                    'launch', 'person_detection_launch_robot.launch.py'
                )
            ),
            launch_arguments={'launch_camera': 'false'}.items()
        ),

        # Face detection (nested dependency, requires person detection)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('face_detection'),
                    'launch', 'face_detection_launch_robot.launch.py'
                )
            ),
            launch_arguments={'launch_camera': 'false'}.items()
        ),

        # Saliency node (Laptop)
        Node(
            package='overt_attention',
            executable='overt_attention_saliency',
            name='saliency_node',
            parameters=[
                LaunchConfiguration('params_file'),
                {'process_hz': 1.0}
            ],
            output='screen'
        ),
        
        # Unified attention controller (Laptop)
        Node(
            package='overt_attention',
            executable='overt_attention',
            name='unified_attention_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),
        
        # Visualization (Laptop)
        Node(
            package='overt_attention',
            executable='overt_attention_visualization',
            name='attention_visualization',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_viz'))
        ),
    ])
