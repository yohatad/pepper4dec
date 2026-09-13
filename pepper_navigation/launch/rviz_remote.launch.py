"""rviz_remote.launch.py

RViz ONLY, for a laptop watching a Nav2 stack that runs on the robot.

Nav2 and the localizer stay on the Jetson, next to the sensors. This launches
nothing but rviz2, with a deliberately light config that subscribes to no
point cloud stream, no camera, and no prior map -- only the 2D map, the
costmaps, plans, footprints, safety polygons and the localizer's pose.

WHY (MEASURED 2026-09-12): with the full stack up the Jetson sat at a load
average of ~5.8 on 6 cores. Two of the largest consumers were not navigation
at all: RViz rendering a 4.5 M-point prior map, and NoMachine encoding that
desktop to stream it. Both go away when RViz runs here instead. The Behavior
Tree "tick rate exceeded" warnings were the symptom of that saturation.

What you can and cannot see from the light view:
    Locked pose arrow present  -> localized (it is published only while locked)
    Candidate pose (orange)    -> a lock is being verified right now
    Candidate scan (disabled)  -> enable it to judge a lock against the map:
                                  ~1-3k points, only during verification, so
                                  a short burst rather than a stream
    NOT shown: /cloud_registered, /prior_map, /points, the camera. Those are
    the heavy topics this config exists to avoid.

REQUIREMENTS -- discovery is the thing that will silently fail:
    1. Same ROS_DOMAIN_ID as the robot (the Jetson uses 5) and the same RMW
       (rmw_cyclonedds_cpp).
    2. The Jetson's CycloneDDS config (ros2_ws/config/cyclonedds/robot.xml)
       runs UNICAST discovery with multicast OFF and a hardcoded <Peers> list.
       This laptop's WiFi address must be in that list, AND this laptop must
       run a CycloneDDS config that lists the Jetson (172.29.111.250) as a
       peer. Miss either half and `ros2 topic list` here shows nothing, with
       no error. Check with:  ros2 node list   (expect the Jetson's nodes)
    3. pepper_navigation built on this machine, so the config path resolves.
       No robot meshes are needed -- the light config has no RobotModel.

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
