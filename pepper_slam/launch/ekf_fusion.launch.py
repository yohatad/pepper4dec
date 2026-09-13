"""ekf_fusion.launch.py

Fuse leveled LIO odometry with wheel odometry via robot_localization.

Takes x, y, yaw from the LIO estimator and z, roll, pitch from wheel odometry,
as an alternative to lio_odom_bridge.py's flatten_base_frame hard clamp — see
config/ekf_lio_wheel.yaml for why.

ADDITIVE, not a replacement: run this ALONGSIDE fastlio_odometry.launch.py or
pointlio_odometry.launch.py (it needs their odom TF and raw odom topic already
flowing). Publishes /odometry/filtered only — it does not touch the existing TF
tree (publish_tf: false in the EKF config), so it is safe to add without
risking the working default pipeline.

Nodes started:
    pepper_slam/leveled_odometry_publisher.py (node:
    leveled_odometry_publisher)
        Republishes the LIO odometry leveled for fusion.
    pepper_slam/pepper_odom_relabel.py (node: pepper_odom_relabel)
        Relabels the wheel odometry frames to match.
    robot_localization/ekf_node (node: ekf_filter_node)
        The fusion filter itself.

Launch arguments:
    odom_topic (default: "/odom_lio")
        Raw LIO odometry topic to fuse.
    use_sim_time (default: "true")
        Defaults true because this is normally run against a bag.

Configuration:
    config/ekf_lio_wheel.yaml

Usage:
    ros2 launch pepper_slam fastlio_odometry.launch.py flatten_base_frame:=false
    ros2 launch pepper_slam ekf_fusion.launch.py
    ros2 bag play <bag> --clock --topics /points /imu/data /pepper_odom

For Point-LIO instead:
    ros2 launch pepper_slam pointlio_odometry.launch.py flatten_base_frame:=false
    ros2 launch pepper_slam ekf_fusion.launch.py odom_topic:=/odom_lio

flatten_base_frame:=false on the odometry launch is deliberate here: this
fuses FAST-LIO's or Point-LIO's own raw (undoctored) z/roll/pitch with wheel
odometry, so the input needs to still be the real drifting estimate, not
already clamped to zero by the other fix.

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

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_slam')
    ekf_config = os.path.join(pkg_share, 'config', 'ekf_lio_wheel.yaml')

    odom_topic = LaunchConfiguration('odom_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_odom_topic_cmd = DeclareLaunchArgument(
        'odom_topic', default_value='/odom_lio',
        description='Raw LIO odometry topic to fuse. Every mapping launch '
                    'remaps its estimator onto /odom_lio (FAST-LIO natively '
                    '/Odometry, Point-LIO and FAST-LIVO2 /aft_mapped_to_init), '
                    'so the default suits all of them.'
    )
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='true for bag replay (--clock); false on the robot.')

    leveled_odom_node = Node(
        package='pepper_slam',
        executable='leveled_odometry_publisher.py',
        name='leveled_odometry_publisher',
        output='screen',
        parameters=[{
            'odom_topic': odom_topic,
            'output_topic': '/odometry/lio_leveled',
            'use_sim_time': use_sim_time,
        }],
    )

    pepper_odom_relabel_node = Node(
        package='pepper_slam',
        executable='pepper_odom_relabel.py',
        name='pepper_odom_relabel',
        output='screen',
        parameters=[{
            'input_topic': '/pepper_odom',
            'output_topic': '/odometry/pepper_odom_relabeled',
            'use_sim_time': use_sim_time,
        }],
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config, {'use_sim_time': use_sim_time}],
    )

    ld = LaunchDescription()
    ld.add_action(declare_odom_topic_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(leveled_odom_node)
    ld.add_action(pepper_odom_relabel_node)
    ld.add_action(ekf_node)
    return ld
