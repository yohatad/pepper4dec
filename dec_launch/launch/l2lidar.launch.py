"""
l2lidar.launch.py: launch the Unitree L2 4D LiDAR with the settings this rig needs.

Launch files included:
    l2lidar_node/l2lidar.launch.py — the stock driver launch. It starts the
    node with respawn=True / respawn_delay=3.0, which is kept: the L2 link is
    UDP and the driver exits rather than limps if the socket dies.

Nodes started:
    None directly; the node comes from the included driver launch.

Topics published (by the driver):
    /points     sensor_msgs/PointCloud2, ~11 Hz  (aggregateNframes: 19)
    /imu/data   sensor_msgs/Imu
    /tf_static  l2lidar_frame -> l2lidar_frame_imu (the L2's intrinsic IMU
                offset; the driver owns this edge, and pepper_slam's
                sensor_tf.yaml tags it owner:driver so it is not republished)

    BOTH TOPICS ARE PUBLISHED BEST_EFFORT (rclcpp::SensorDataQoS,
    l2lidar_node.cpp:357,360). A subscriber that asks for RELIABLE will not
    match and will receive NOTHING -- DDS refuses a reader that demands more
    than the writer offers. The driver logs this as

        New subscription discovered on topic '/points', requesting
        incompatible QoS. ... Last incompatible policy: RELIABILITY_QOS_POLICY

    which is the only warning you get; the subscribing node stays silent.
    FAST-LIO, Point-LIO, FAST-LIVO2 and kiss-icp all subscribe BEST_EFFORT, the
    rtabmap launches pass qos:=2, and the RViz configs are set to Best Effort.
    `ros2 bag record` defaults to RELIABLE, so pass
    `--qos-profile-overrides-path` (pepper_slam/config/record_qos.yaml) when
    recording these topics.

Launch arguments:
    params_file (default: l2lidar_node/config/l2lidar_node.yaml)
        The driver's parameter file. The shipped one is already tuned for THIS
        unit and must not be swapped casually -- it carries this L2's range
        calibration (calRangeBias -525.0, calRangeScale 0.000984, both measured
        on this device), its 1:1 timestamp scale, cloud_frame pinned to
        l2lidar_frame, and aggregateNframes 19. See that file's comments.
    l2_ip (default: "192.168.1.62")   L2 device address.
    l2_port (default: "6101")         L2 device UDP port.
    host_ip (default: "192.168.1.2")  Address of the host interface receiving
        cloud + IMU. Must be a real local interface or the node fails to start.
    host_port (default: "6201")       Host UDP port.

    The four network arguments are the factory 192.168.1.0/24 defaults. On a
    different subnet, override them rather than editing YAML, e.g.

        ros2 launch dec_launch l2lidar.launch.py l2_ip:=10.42.0.62 host_ip:=10.42.0.2

Configuration:
    l2lidar_node/config/l2lidar_node.yaml (see params_file above). Nothing is
    overridden here: the local tuning lives in that file, next to the comments
    recording how each value was measured, so there is one owner.

Prerequisites:
    The L2 wired to the host and reachable at l2_ip, with host_ip configured on
    the receiving interface. The l2lidar_frame -> base_footprint mount
    transform is NOT published here -- it lives in
    pepper_slam/config/sensor_tf.yaml, the single owner of the rig's static
    TF edges.

Usage:
    ros2 launch dec_launch l2lidar.launch.py

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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    driver_launch = os.path.join(
        get_package_share_directory('l2lidar_node'),
        'launch', 'l2lidar.launch.py')

    default_params = os.path.join(
        get_package_share_directory('l2lidar_node'),
        'config', 'l2lidar_node.yaml')

    return LaunchDescription([

        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='L2 driver parameter file. The default is the shipped '
                        'l2lidar_node.yaml, which carries THIS unit\'s range '
                        'calibration and frame names -- see the file header.'),

        # Network settings are forwarded rather than re-defaulted: the values
        # below are the driver launch's own defaults, repeated here only so
        # they can be set on this command line. Passing an argument the
        # included launch also declares would otherwise be rejected.
        DeclareLaunchArgument(
            'l2_ip', default_value='192.168.1.62',
            description='L2 device IP address.'),
        DeclareLaunchArgument(
            'l2_port', default_value='6101',
            description='L2 device UDP port.'),
        DeclareLaunchArgument(
            'host_ip', default_value='192.168.1.2',
            description='Host IP address (the interface receiving cloud + IMU '
                        'data). Must be a real local interface; the node fails '
                        'to start otherwise.'),
        DeclareLaunchArgument(
            'host_port', default_value='6201',
            description='Host UDP port.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(driver_launch),
            launch_arguments={
                'params_file': LaunchConfiguration('params_file'),
                'l2_ip': LaunchConfiguration('l2_ip'),
                'l2_port': LaunchConfiguration('l2_port'),
                'host_ip': LaunchConfiguration('host_ip'),
                'host_port': LaunchConfiguration('host_port'),
            }.items(),
        ),
    ])
