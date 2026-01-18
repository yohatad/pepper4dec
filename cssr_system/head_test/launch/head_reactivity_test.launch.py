#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the package share directory
    pkg_dir = get_package_share_directory('head_test')
    
    # Create the node action
    head_test_node = Node(
        package='head_test',
        executable='head_reactivity_test',
        name='head_reactivity_test',
        output='screen',
        parameters=[{
            'test_duration': 120.0,          # 2 minutes test
            'point_interval': 3.0,           # 3 seconds between points
            'settle_threshold': 0.05,        # 0.05 rad (~2.9 degrees)
            'max_response_time': 5.0,        # 5 seconds timeout
            'image_width': 640,              # fallback width
            'image_height': 480,             # fallback height
            'joint_state_topic': '/joint_states',
            'camera_info_topic': '/camera/color/camera_info',
            'head_command_topic': '/joint_angles',
            'image_topic': '/camera/color/image_raw_custom',  # Image topic
            'use_image_dimensions': True,    # Use actual image dimensions
            'visualize': False,              # Disable visualization by default
        }]
    )
    
    # Create launch description
    ld = LaunchDescription()
    ld.add_action(head_test_node)
    
    return ld
