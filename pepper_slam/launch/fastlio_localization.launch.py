"""fastlio_localization.launch.py

Standalone fastlio_localization on the Pepper L2 rig, with its required
static TF, for LIVE testing outside the full Nav2 bringup.

It exists because fast_lio's own localization_l2.launch.py is not
standalone-usable live: it is bag-oriented (use_sim_time defaults true) and it
does not publish the rig's static TF, which pepper_nav2_fastloc.launch.py
normally supplies via its own sensor_tf GroupAction. Run localization_l2
alone on the robot and you get two silent failures at once: ROS time pinned
at 0 (no /clock publisher), and "Tf has two or more unconnected trees" /
"NOT broadcasting map -> base_footprint" once it does lock, because
base_footprint and camera_imu_optical_frame are disconnected roots.
MEASURED 2026-09-09: fixed by use_sim_time:=false + this file's sensor_tf
include (publisher:=urdf scope:=mount).

Launch files included:
    pepper_sensor_tf.launch.py — the rig's static TF.
    fast_lio/localization_l2.launch.py — the localizer + its own RViz.

Launch arguments:
    use_sim_time (default: "false")
        false here (LIVE entry point), unlike fast_lio's own default of true
        (bag-oriented). Pass true only when replaying a bag WITH --clock.
    publisher (default: "urdf")
    scope (default: "mount")
        Forwarded to pepper_sensor_tf.launch.py. 'mount' is correct live: the
        RealSense driver publishes its own internal extrinsics, so 'all'
        would give those edges two publishers and the last one silently wins.
    rviz (default: "true")
    config_file (default: "l2_rsimu.yaml")
        FAST-LIO config (RealSense IMU); l2.yaml uses the L2's own.
    map_dir, map_pose_file, map_scan_dir
        Prior map location. Defaults match fast_lio's own (pepper_navigation's
        shipped map). See localization_l2.launch.py for the full set of
        ScanContext/init_* tuning arguments this file does not forward.

Usage:
    ros2 launch pepper_slam fastlio_localization.launch.py

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 9, 2026
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
    fast_lio_share = get_package_share_directory('fast_lio')
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
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config: l2_rsimu.yaml (RealSense IMU, matches '
                    'the prior map) or l2.yaml (the L2 s own).')
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

    fastlio_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_share, 'launch', 'localization_l2.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'rviz': LaunchConfiguration('rviz'),
            'config_file': LaunchConfiguration('config_file'),
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
    ld.add_action(declare_map_dir_cmd)
    ld.add_action(declare_map_pose_file_cmd)
    ld.add_action(declare_map_scan_dir_cmd)
    # AFTER every DeclareLaunchArgument: the echo reads use_sim_time/publisher/
    # scope, which do not exist in the context until their declares have run.
    ld.add_action(OpaqueFunction(function=_echo_resolved))
    ld.add_action(sensor_tf_launch)
    ld.add_action(fastlio_localization_launch)
    return ld
