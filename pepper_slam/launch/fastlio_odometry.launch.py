# Plain FAST-LIO ODOMETRY on the Pepper L2 rig, with its required static TF.
#
# RENAMED 2026-08-10 from fastlio_mapping.launch.py. "mapping" is upstream
# FAST-LIO's word for scan-to-map REGISTRATION -- the ikd-Tree map the estimator
# keeps in order to align each new scan. It is a map used FOR odometry, not a
# SLAM map. There is no loop closure, no pose-graph optimisation and no
# relocalisation here: every scan is stamped into /Laser_map at whatever pose
# odometry believed at that instant and nothing ever revisits it, so revisiting
# a place after N metres of drift lays the same wall down twice, permanently.
# The old name led to exactly that surprise. For a map worth keeping use
# fastlio_lc_pgo fastlio_lc_l2.launch.py (Scan Context + GTSAM) or
# pepper_slam bag_test/rtabmap_fastlio_bag_test.launch.py.
#
# This file is still the right tool for MEASURING odometry quality, precisely
# because nothing here hides the drift.
#
# ros2 launch fast_lio mapping.launch.py by itself is not standalone-usable
# on this robot: lio_map_odom_bridge.py needs the static base_footprint ->
# l2lidar_frame -> l2lidar_frame_imu chain, which only pepper_sensor_tf.launch.py
# provides, and mapping.launch.py doesn't include it (that file is shared
# across every FAST-LIO sensor config in this workspace -- mid360, velodyne,
# avia, etc. -- so it can't bake in Pepper-specific transforms without
# breaking all of those). Forgetting the second launch file is a silent-hang
# footgun; this wraps both together the same way fastlio_lc_l2.launch.py
# (fastlio_lc_pgo) and fastlio_localization_l2.launch.py (lio_localization)
# already do for their own use cases -- this is the missing plain-mapping
# equivalent.
#
# Usage:
#   ros2 launch pepper_slam fastlio_odometry.launch.py
#   ros2 bag play <bag> --clock --topics /points /imu/data
#   (add --topics /tf to that list only if you also want to exclude it --
#    see pepper_sensor_tf.launch.py's header: replaying /tf fights the
#    bridge for base_footprint's parent.)
#
# flatten_base_frame defaults to true HERE (unlike fast_lio's own
# mapping.launch.py, which defaults it false): this file is Pepper-specific
# and Pepper is confirmed to only ever run on flat floor, so the hard
# flat-floor assumption is safe here in a way it isn't for the generic,
# multi-sensor fast_lio package. Pass flatten_base_frame:=false to get
# FAST-LIO's own (drifting) z/roll/pitch back.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    fast_lio_share = get_package_share_directory('fast_lio')
    pkg_share = get_package_share_directory('pepper_slam')

    rviz = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    use_sim_time = LaunchConfiguration('use_sim_time')
    bridge_level_frame = LaunchConfiguration('bridge_level_frame')
    flatten_base_frame = LaunchConfiguration('flatten_base_frame')

    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_rviz_cfg_cmd = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=os.path.join(fast_lio_share, 'rviz', 'fastlio.rviz'))
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='true for bag replay (--clock); false on the robot.')
    declare_bridge_level_frame_cmd = DeclareLaunchArgument(
        'bridge_level_frame', default_value='true',
        description='Have lio_map_odom_bridge publish the static odom -> '
                    'odom leveling frame. Set false when a higher layer owns '
                    'odom (e.g. PGO publishing map -> odom).'
    )
    declare_flatten_base_frame_cmd = DeclareLaunchArgument(
        'flatten_base_frame', default_value='true',
        description='Zero the leveled z/roll/pitch of odom -> base_footprint '
                    'every cycle (keep x, y, yaw). Defaults true here: Pepper '
                    'is confirmed flat-floor-only. Set false to see FAST-LIO\'s '
                    'own (drifting) z/roll/pitch instead.'
    )

    sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_share, 'launch', 'mapping.launch.py')),
        launch_arguments={
            'config_file': 'l2.yaml',
            'rviz': rviz,
            'rviz_cfg': rviz_cfg,
            'use_sim_time': use_sim_time,
            'bridge_level_frame': bridge_level_frame,
            'flatten_base_frame': flatten_base_frame,
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_cfg_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_bridge_level_frame_cmd)
    ld.add_action(declare_flatten_base_frame_cmd)
    ld.add_action(sensor_tf_launch)
    ld.add_action(fast_lio_launch)
    return ld
