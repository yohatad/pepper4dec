#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('pepper_attention')
    params_file = os.path.join(pkg_dir, 'config', 'param.yaml')
    
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
        
        # Saliency node (Laptop)
        Node(
            package='pepper_attention',
            executable='saliency_node',
            name='saliency_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),
        
        # Unified attention controller (Laptop)
        Node(
            package='pepper_attention',
            executable='unified_attention_node',
            name='unified_attention_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),
        
        # Visualization (Laptop)
        Node(
            package='pepper_attention',
            executable='visualization_node',
            name='attention_visualization',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_viz'))
        ),
    ])