"""pointlio_localization.launch.py

Standalone pointlio_localization on the Pepper L2 rig, with its required
static TF, for LIVE testing outside the full Nav2 bringup. Twin of
fastlio_localization.launch.py for the Point-LIO backend.

It exists for the same reason as its fastlio twin: point_lio's own
localization_l2.launch.py is not standalone-usable live. It is bag-oriented
(use_sim_time defaults true) and does not publish the rig's static TF, which
pepper_nav2_pointloc.launch.py normally supplies via its own sensor_tf
GroupAction. Run localization_l2 alone on the robot and you get two silent
failures at once: ROS time pinned at 0 (no /clock publisher), and a TF tree
split in two once it does lock, because base_footprint and
camera_imu_optical_frame are disconnected roots.

Launch files included:
    pepper_sensor_tf.launch.py — the rig's static TF.
    point_lio/localization_l2.launch.py — the localizer + its own RViz.

Launch arguments:
    use_sim_time (default: "false")
        false here (LIVE entry point), unlike point_lio's own default of true
        (bag-oriented). Pass true only when replaying a bag WITH --clock.
    publisher (default: "urdf")
    scope (default: "mount")
        Forwarded to pepper_sensor_tf.launch.py. 'mount' is correct live: the
        RealSense driver publishes its own internal extrinsics, so 'all'
        would give those edges two publishers and the last one silently wins.
    rviz (default: "true")
    config_file (default: "l2lidar_rsimu.yaml")
        Point-LIO config (RealSense IMU, matches the prior map);
        l2lidar_node.yaml uses the L2's own IMU.
    body_frame (default: "camera_imu_optical_frame")
        Frame the filter estimates. MUST match config_file:
        camera_imu_optical_frame for l2lidar_rsimu.yaml, l2lidar_frame_imu
        for l2lidar_node.yaml. Forwarded explicitly rather than left to the
        include's default so a config_file override cannot silently pair
        with the wrong frame.
    map_dir, map_pose_file, map_scan_dir
        Prior map location. Defaults match point_lio's own (pepper_navigation's
        shipped map). See localization_l2.launch.py for the full set of
        ScanContext / verification / guard tuning arguments this file does
        not forward.

Usage:
    ros2 launch pepper_slam pointlio_localization.launch.py

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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _echo_resolved(context, *args, **kwargs):
    """Echo the decisions that fail silently if wrong: use_sim_time true with
    no /clock pins time at 0, and a publisher/scope mismatch with what the
    live camera driver already publishes leaves TF split into two trees.
    """
    from launch.actions import LogInfo
    sim = LaunchConfiguration('use_sim_time').perform(context)
    pub = LaunchConfiguration('publisher').perform(context)
    scope = LaunchConfiguration('scope').perform(context)
    return [LogInfo(msg='[pepper_slam] use_sim_time=%s  sensor_tf publisher=%s  scope=%s'
                        % (sim, pub, scope))]


def generate_launch_description():
    point_lio_share = get_package_share_directory('point_lio')
    pkg_share = get_package_share_directory('pepper_slam')

    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='false (default) on the robot; true only for bag replay '
                    'with ros2 bag play --clock.')
    declare_publisher_cmd = DeclareLaunchArgument(
        'publisher', default_value='urdf',
        description="Forwarded to pepper_sensor_tf.launch.py. 'urdf' (default) "
                    "or 'yaml'.")
    declare_scope_cmd = DeclareLaunchArgument(
        'scope', default_value='mount', choices=['mount', 'all'],
        description="Forwarded to pepper_sensor_tf.launch.py. 'mount' (default) "
                    "live, where the RealSense driver publishes its own "
                    "internal extrinsics. 'all' only for a bag recorded "
                    "without /tf_static.")
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2lidar_rsimu.yaml',
        description='Point-LIO config: l2lidar_rsimu.yaml (RealSense IMU, '
                    'matches the prior map) or l2lidar_node.yaml (the L2 s own).')
    declare_body_frame_cmd = DeclareLaunchArgument(
        'body_frame', default_value='camera_imu_optical_frame',
        description='Frame the filter estimates; MUST match config_file. '
                    'camera_imu_optical_frame for l2lidar_rsimu.yaml, '
                    'l2lidar_frame_imu for l2lidar_node.yaml.')
    declare_map_dir_cmd = DeclareLaunchArgument(
        'map_dir',
        default_value=os.path.join(
            get_package_share_directory('pepper_navigation'), 'pcd'),
        description='Directory holding the pose file.')
    declare_map_pose_file_cmd = DeclareLaunchArgument(
        'map_pose_file', default_value='sc_pose_20260823.json',
        description='Pose file within map_dir.')
    declare_map_scan_dir_cmd = DeclareLaunchArgument(
        'map_scan_dir',
        default_value=os.path.join(
            get_package_share_directory('pepper_navigation'),
            'pcd', 'sc_pcd_20260823'),
        description='Directory holding the per-keyframe <N>.pcd clouds.')

    sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'publisher': LaunchConfiguration('publisher'),
            'scope': LaunchConfiguration('scope'),
        }.items())

    pointlio_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(point_lio_share, 'launch', 'localization_l2.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rviz': LaunchConfiguration('rviz'),
            'config_file': LaunchConfiguration('config_file'),
            'body_frame': LaunchConfiguration('body_frame'),
            'map_dir': LaunchConfiguration('map_dir'),
            'map_pose_file': LaunchConfiguration('map_pose_file'),
            'map_scan_dir': LaunchConfiguration('map_scan_dir'),
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_publisher_cmd)
    ld.add_action(declare_scope_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_body_frame_cmd)
    ld.add_action(declare_map_dir_cmd)
    ld.add_action(declare_map_pose_file_cmd)
    ld.add_action(declare_map_scan_dir_cmd)
    # AFTER every DeclareLaunchArgument: the echo reads use_sim_time/publisher/
    # scope, which do not exist in the context until their declares have run.
    ld.add_action(OpaqueFunction(function=_echo_resolved))
    ld.add_action(sensor_tf_launch)
    ld.add_action(pointlio_localization_launch)
    return ld
