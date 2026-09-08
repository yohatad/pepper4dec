"""slam_toolbox.launch.py

Run SLAM Toolbox in online asynchronous mode against /scan.

2D laser SLAM, kept for comparison against the LIO pipelines. It consumes the
flattened /scan rather than the L2's /points, so it sees only one horizontal
slice of the world.

Nodes started:
    slam_toolbox/async_slam_toolbox_node (node: slam_toolbox)

Launch arguments:
    slam_params_file (default: <share>/config/mapper_params_online_async.yaml)
        Full path to the SLAM Toolbox parameters.

Configuration:
    config/mapper_params_online_async.yaml

Prerequisites:
    Something must publish /scan — pointcloud_to_laserscan over the L2 cloud,
    or a 2D lidar driver.

Usage:
    ros2 launch pepper_slam slam_toolbox.launch.py

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
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Path to SLAM Toolbox config
    slam_params_file = LaunchConfiguration('slam_params_file')

    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(
            get_package_share_directory('pepper_slam'),
            'config',
            'mapper_params_online_async.yaml'
        ),
        description='Full path to SLAM Toolbox parameters'
    )

    # SLAM Toolbox Node
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_file],
        remappings=[
            ('/scan', '/scan')  # Your YDLidar topic
        ]
    )

    return LaunchDescription([
        declare_slam_params_file_cmd,
        slam_toolbox_node
    ])
