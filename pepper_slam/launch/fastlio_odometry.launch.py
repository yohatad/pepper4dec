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
    # 2026-08-12: the RealSense IMU is now the permanent choice for this rig.
    # The L2's own gyro cancels rotation about the gravity axis below ~16 deg/s
    # and cost 139 deg of heading over a 744 s run (utils/L2_IMU/REPORT.md);
    # l2_rsimu.yaml drives the same estimator from /camera/imu instead and
    # measured 3.8% -> 2.4% mean yaw error, 11.2% -> 4.6% worst. l2.yaml is kept
    # selectable for A/B work only.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config under fast_lio/config. l2_rsimu.yaml uses '
                    'the RealSense IMU (default); l2.yaml uses the L2 s own.')
    # Left unset on purpose: mapping.launch.py derives the matching frame from
    # config_file, so the two cannot drift apart. Override only for a config
    # this launch does not know about.
    declare_lidar_imu_frame_cmd = DeclareLaunchArgument(
        'lidar_imu_frame', default_value='',
        description='Override the static frame the estimated body corresponds '
                    'to. Empty (default) lets mapping.launch.py pick it from '
                    'config_file.')
    # Bag replay has no RealSense driver, so the camera TF edges must come from
    # calibration or camera_imu_optical_frame -- which l2_rsimu.yaml names as
    # the body frame -- will not resolve at all. Use 'mount' on the real robot.
    declare_scope_cmd = DeclareLaunchArgument(
        'sensor_tf_scope', default_value='all', choices=['mount', 'all'],
        description="'all' for bag replay (no driver running); 'mount' on the "
                    "robot so the driver's device-read camera values win.")
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
        launch_arguments={'use_sim_time': use_sim_time,
                          'scope': LaunchConfiguration('sensor_tf_scope')}.items())

    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_share, 'launch', 'mapping.launch.py')),
        launch_arguments={
            'config_file': LaunchConfiguration('config_file'),
            'lidar_imu_frame': LaunchConfiguration('lidar_imu_frame'),
            'rviz': rviz,
            'rviz_cfg': rviz_cfg,
            'use_sim_time': use_sim_time,
            'bridge_level_frame': bridge_level_frame,
            'flatten_base_frame': flatten_base_frame,
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_lidar_imu_frame_cmd)
    ld.add_action(declare_scope_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_cfg_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_bridge_level_frame_cmd)
    ld.add_action(declare_flatten_base_frame_cmd)
    ld.add_action(sensor_tf_launch)
    ld.add_action(fast_lio_launch)
    return ld
