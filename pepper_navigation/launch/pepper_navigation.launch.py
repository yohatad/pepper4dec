"""pepper_navigation.launch.py

Legacy Nav2 bringup: AMCL on the robot's raw wheel odometry.

This is the `legacy` nav_profile in dec_system.launch.py, kept for reproducing
old runs. It publishes no /localization/pose, so gesture_execution has no
map-frame pose to point with; prefer pepper_nav2_fastloc.launch.py.

Nodes started:
    nav2_map_server/map_server (node: map_server)
        Serves map/pepper_map_lc.yaml.
    nav2_map_server/map_server (node: filter_mask_server)
        Serves map/keepout_zone.yaml on /keepout_filter_mask.
    nav2_map_server/costmap_filter_info_server
        Publishes the keepout filter info from nav2_params.yaml.
    nav2_amcl/amcl
        Particle-filter localization against the static map.
    nav2_controller/controller_server, nav2_planner/planner_server,
    nav2_behaviors/behavior_server, nav2_bt_navigator/bt_navigator
        The standard Nav2 navigation pipeline.
    nav2_lifecycle_manager/lifecycle_manager (node:
    lifecycle_manager_navigation)
        Autostarts all eight lifecycle nodes above with bond_timeout 4.0.

Launch arguments:
    (none) — the map, mask, and parameter paths are fixed to the package
    share directory.

Configuration:
    config/nav2_params.yaml, map/pepper_map_lc.yaml, map/keepout_zone.yaml.

Prerequisites:
    Wheel odometry and the laser scan Nav2's costmaps expect must already be
    published, along with the odom -> base_footprint transform.

Usage:
    ros2 launch pepper_navigation pepper_navigation.launch.py

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 8, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Paths
    pkg_dir = get_package_share_directory('pepper_navigation')
    map_file = os.path.join(pkg_dir, 'map', 'pepper_map_lc.yaml')
    params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    keepout_mask_file = os.path.join(pkg_dir, 'map', 'keepout_zone.yaml')

    return LaunchDescription([
        # Map Server (lifecycle node)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_file}]
        ),

        # Filter Mask Server (lifecycle node - add to lifecycle manager)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='filter_mask_server',
            output='screen',
            parameters=[{
                'yaml_filename': keepout_mask_file,
                'topic_name': '/keepout_filter_mask',  # Absolute path
                'frame_id': 'map'
            }]
        ),

        # Costmap Filter Info Server (lifecycle node)
        Node(
            package='nav2_map_server',
            executable='costmap_filter_info_server',
            name='costmap_filter_info_server',
            output='screen',
            parameters=[params_file]
        ),

        # AMCL (Localization)
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[params_file]
        ),

        # Nav2 Controller
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file]
        ),

        # Nav2 Planner
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file]
        ),

        # Nav2 Behavior Server
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params_file]
        ),

        # Nav2 BT Navigator
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file]
        ),

        # Lifecycle Manager (ONLY lifecycle nodes here)
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': True,
                'bond_timeout': 4.0,
                'node_names': [
                    'map_server',
                    'filter_mask_server',
                    'costmap_filter_info_server',  # Must activate lifecycle node
                    'amcl',
                    'controller_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator'
                ]
            }]
        )
    ])
