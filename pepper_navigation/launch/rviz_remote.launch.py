"""rviz_remote.launch.py

RViz only, for a laptop watching a Nav2 stack running on the robot.

Nav2 and the localizer stay on the Jetson; this starts rviz2 alone with a light
config -- 2D map, costmaps, plans, footprints, safety polygons and the
localizer's pose, but no point clouds, camera or prior map.

Why: measured 2026-09-12, the full stack held the Jetson at ~5.8 load average
on 6 cores, two of the top consumers being RViz rendering a 4.5 M-point prior
map and NoMachine encoding that desktop. Running RViz here removes both; the
Behavior Tree "tick rate exceeded" warnings were a symptom of that saturation.

Reading the light view: a locked pose arrow means localized (it is published
only while locked), and an orange candidate pose means a lock is being
verified. Enable the candidate scan to judge a lock against the map (~1-3k
points, only during verification). /cloud_registered, /prior_map, /points and
the camera are absent by design -- they are what this config exists to avoid.

Requirements -- discovery fails silently if any is wrong:
    1. Same ROS_DOMAIN_ID as the robot (the Jetson uses 5) and the same RMW
       (rmw_cyclonedds_cpp).
    2. The Jetson's CycloneDDS config runs UNICAST discovery with a hardcoded
       <Peers> list: this laptop must appear in it, AND must itself list the
       Jetson (172.29.111.250). Miss either half and `ros2 topic list` is
       silently empty -- check with `ros2 node list`.
    3. pepper_navigation built on this machine, so the config path resolves.

Launch arguments:
    rviz_config (default: <share>/rviz/nav2_fastloc_remote.rviz)
        Pass nav2_fastloc.rviz to get the full view -- but then the point
        clouds stream over WiFi, which is what this file avoids.
    use_sim_time (default: "false")

Usage (on the laptop, robot already running pepper_nav2_fastloc):
    ros2 launch pepper_navigation rviz_remote.launch.py

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 12, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_navigation')

    declare_rviz_config_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_share, 'rviz', 'nav2_fastloc_remote.rviz'),
        description='RViz config. Default is the WiFi-light view; '
                    'nav2_fastloc.rviz gives the full one at the cost of '
                    'streaming every point cloud.')
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='true only when the robot side is replaying a bag with '
                    '--clock.')

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}])

    ld = LaunchDescription()
    ld.add_action(declare_rviz_config_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(rviz)
    return ld
