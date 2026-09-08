"""
dec_robot.launch.py: bring up the whole physical layer in one command.

That is the Unitree L2 LiDAR, the bottom RealSense, and the robot itself.

This replaces the three-command sequence in pepper_navigation/README.md
("Always start the sensors first"). It is drivers only.

Launch files included:
    dec_launch/l2lidar.launch.py — the L2 driver (/points, /imu/data).
    dec_launch/realsense_bottom.launch.py — the bottom RealSense.
    dec_launch/naoqi_driver.launch.py — the NAOqi bridge to Pepper.

Nodes started:
    None directly; everything comes from the three included launch files.

NO STATIC TF HERE — AND WHY
    The transforms that put the L2 and the RealSense in one tree
    (base_footprint -> l2lidar_frame -> {l2lidar_frame_imu,
    camera_camera_link -> ...}) come from pepper_slam's
    pepper_sensor_tf.launch.py, which is NOT included here: every stack that
    consumes these sensors already nests it -- pepper_nav2_fastloc / _amcl /
    _rtabmap_loc (as sensor_tf / sensor_tf_scope), pepper_slam's
    fastlio_odometry and pointlio_odometry, and dec_system.launch.py through
    them. Publishing it from here too would give those latched /tf_static
    edges two publishers, and whichever lands last silently wins.

    So the rule is one owner, and the owner is whatever you launch next.
    Running the drivers ALONE (bag recording, a raw RViz look at /points),
    there is no owner, and nothing relates the two sensors -- start it
    yourself:

        ros2 launch pepper_slam pepper_sensor_tf.launch.py

    scope:=mount is its default and is right on the robot; the RealSense
    driver publishes its own internal extrinsics.

Launch arguments:
    enable_lidar / enable_camera / enable_robot (default: "true")
        Turn each driver off individually.
    l2_ip / l2_port / host_ip / host_port
        Forwarded to l2lidar.launch.py. Factory 192.168.1.0/24 defaults.
    nao_ip / nao_port / user / password / network_interface / qi_listen_url
        Forwarded to naoqi_driver.launch.py. NOTE nao_ip defaults to the
        CMU-Pepper router address; on the PepperNet AP pass
        nao_ip:=10.42.0.204.
    use_camera (default: "false")
        Forwarded to naoqi_driver.launch.py — the gscam2 Pepper FRONT
        camera, which is separate from the bottom RealSense and needs the
        stream started on the robot by hand. Off by default.

    Each included file documents its own arguments; this file only re-declares
    the ones worth setting from here.

Configuration:
    None of its own; each driver's YAML lives in its own package.

Prerequisites:
    The L2 reachable at l2_ip, a RealSense on USB, and the robot awake at
    nao_ip. Any subset works — switch the others off.

Usage:
    ros2 launch dec_launch dec_robot.launch.py
    ros2 launch dec_launch dec_robot.launch.py enable_robot:=false
    ros2 launch dec_launch dec_robot.launch.py nao_ip:=10.42.0.204

    Then bring up an estimator or the full system, which brings the sensor-rig
    TF with it:
        ros2 launch dec_launch dec_system.launch.py

Design notes:
    Every include is wrapped in a scoped GroupAction. IncludeLaunchDescription
    emits its launch_arguments as SetLaunchConfiguration into the CURRENT
    context, so an unscoped include would leak its arguments into the ones
    after it — the same trap dec_system.launch.py and
    pepper_nav2_fastloc.launch.py hit.

    QoS: the L2 publishes /points and /imu/data BEST_EFFORT. A RELIABLE
    subscriber silently receives nothing and the driver logs "requesting
    incompatible QoS ... RELIABILITY_QOS_POLICY". See l2lidar.launch.py.

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
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _include(package, launch_file, **kwargs):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)
        ),
        **kwargs
    )


def generate_launch_description():
    return LaunchDescription([

        # ------------------------------------------------------------
        # Which pieces to start
        # ------------------------------------------------------------
        DeclareLaunchArgument(
            'enable_lidar', default_value='true',
            description='Start the Unitree L2 driver.'),
        DeclareLaunchArgument(
            'enable_camera', default_value='true',
            description='Start the bottom RealSense.'),
        DeclareLaunchArgument(
            'enable_robot', default_value='true',
            description='Start the NAOqi driver (the bridge to Pepper).'),

        # ------------------------------------------------------------
        # L2 network (forwarded to l2lidar.launch.py)
        # ------------------------------------------------------------
        DeclareLaunchArgument('l2_ip', default_value='192.168.1.62',
                              description='L2 device IP address.'),
        DeclareLaunchArgument('l2_port', default_value='6101',
                              description='L2 device UDP port.'),
        DeclareLaunchArgument('host_ip', default_value='192.168.1.2',
                              description='Host IP receiving cloud + IMU. Must '
                                          'be a real local interface.'),
        DeclareLaunchArgument('host_port', default_value='6201',
                              description='Host UDP port.'),

        # ------------------------------------------------------------
        # Robot connection (forwarded to naoqi_driver.launch.py)
        # ------------------------------------------------------------
        DeclareLaunchArgument('nao_ip', default_value='172.29.111.240',
                              description='Robot IP. 172.29.111.240 on the '
                                          'CMU-Pepper router, 10.42.0.204 on '
                                          'the PepperNet AP.'),
        DeclareLaunchArgument('nao_port', default_value='9559',
                              description='NAOqi port.'),
        DeclareLaunchArgument('user', default_value='nao',
                              description='Username for the connection.'),
        DeclareLaunchArgument('password', default_value='no_password',
                              description='Password for the connection.'),
        DeclareLaunchArgument('network_interface', default_value='eth0',
                              description='LOCAL interface handed to NAOqi for '
                                          'the return connection. The node '
                                          'throws if the name does not exist -- '
                                          'check `ip -br addr`.'),
        DeclareLaunchArgument('qi_listen_url', default_value='tcp://0.0.0.0:0',
                              description='Endpoint NAOqi connects back to '
                                          '(audio).'),
        DeclareLaunchArgument('use_camera', default_value='false',
                              description="Also start the gscam2 Pepper FRONT "
                                          "camera. Not the bottom RealSense; "
                                          "needs ~/start_camera.sh on the robot."),

        # ------------------------------------------------------------
        # 1) Unitree L2 -> /points + /imu/data
        # ------------------------------------------------------------
        GroupAction([
            _include('dec_launch', 'l2lidar.launch.py',
                     launch_arguments={
                         'l2_ip': LaunchConfiguration('l2_ip'),
                         'l2_port': LaunchConfiguration('l2_port'),
                         'host_ip': LaunchConfiguration('host_ip'),
                         'host_port': LaunchConfiguration('host_port'),
                     }.items()),
        ], condition=IfCondition(LaunchConfiguration('enable_lidar'))),

        # ------------------------------------------------------------
        # 2) Bottom RealSense. Takes no arguments -- everything it needs is
        #    pinned inside it and in config/realsense_bottom_pointcloud.yaml.
        # ------------------------------------------------------------
        GroupAction([
            _include('dec_launch', 'realsense_bottom.launch.py'),
        ], condition=IfCondition(LaunchConfiguration('enable_camera'))),

        # ------------------------------------------------------------
        # 3) The robot. publish_wheel_odom_tf is NOT passed anywhere in this
        #    chain: it is defined once, in naoqi_driver.launch.py, and must
        #    stay false while FAST-LIO owns odom -> base_footprint.
        # ------------------------------------------------------------
        GroupAction([
            _include('dec_launch', 'naoqi_driver.launch.py',
                     launch_arguments={
                         'nao_ip': LaunchConfiguration('nao_ip'),
                         'nao_port': LaunchConfiguration('nao_port'),
                         'user': LaunchConfiguration('user'),
                         'password': LaunchConfiguration('password'),
                         'network_interface': LaunchConfiguration('network_interface'),
                         'qi_listen_url': LaunchConfiguration('qi_listen_url'),
                         'use_camera': LaunchConfiguration('use_camera'),
                     }.items()),
        ], condition=IfCondition(LaunchConfiguration('enable_robot'))),
    ])
