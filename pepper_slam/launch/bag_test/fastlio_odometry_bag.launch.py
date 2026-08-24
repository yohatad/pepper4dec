# FAST-LIO odometry only, on a recorded bag.
#
# Thin wrapper over pepper_slam/launch/fastlio_odometry.launch.py -- the live
# entry point -- with use_sim_time forced true. Nothing about the stack is
# duplicated here; see bag_test/README.md for the convention.
#
# Usage (a bag recorded with config/record_qos.yaml, i.e. one that HAS
# /tf_static -- see below for the older ones):
#   ros2 launch pepper_slam fastlio_odometry_bag.launch.py publisher:=none
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 2000 --disable-keyboard-controls \
#     --topics /points /camera/imu /imu/data /tf /tf_static
#
# The QoS overrides are REQUIRED: /imu/data and /camera/imu were recorded
# BEST_EFFORT, and a RELIABLE subscriber matches nothing against them, so
# without the file the estimator waits forever for IMU init and prints nothing.
#
# DISABLING THE LIVE RIG PUBLISHER, for a bag that carries its own /tf_static
# (anything recorded with config/record_qos.yaml):
#
#   ros2 launch pepper_slam fastlio_odometry_bag.launch.py publisher:=none
#
# 'publisher' is pepper_sensor_tf.launch.py's own argument and reaches it by
# inheritance -- a value set on the command line overrides a declared default
# further down the include tree. It carries no choices= restriction and both of
# its nodes are gated on IfCondition(publisher=='urdf'/'yaml'), so any other
# value starts neither and publishes nothing, so 'scope' is then moot.
#
# For a LEGACY bag (slam_recording*, slam_bench_run*) with NO /tf_static,
# pass scope:=all instead -- pepper_sensor_tf must supply the camera edges
# because nothing else will.
#
# Do NOT "helpfully" forward this under an alias from here. An explicit
# launch_arguments entry SHADOWS the command-line value of the inner name:
# publisher:=none would silently become a no-op while still appearing in
# --show-args. Measured 2026-08-24.
#
# --disable-keyboard-controls is needed if you background the player: rosbag2
# reads the terminal for its pause/resume keys, and a background job doing that
# gets SIGTTIN and stops dead.
#
# REPLAYING /tf IS FINE, and you want it. An older version of this header said
# not to, because the bag's wheel odometry would fight lio_map_odom_bridge for
# base_footprint's parent. That stopped being true at commit 8edd1f5, which
# defaulted publish_wheel_odom_tf to false (joint_state.cpp:89) for exactly
# that reason -- so the odom -> base_footprint edge is not recorded at all.
# VERIFIED on slam_20260823_merged: across all 77080 /tf messages, zero
# transforms name base_footprint as a child, 'odom' never appears, and
# base_footprint is the sole root. The bridge attaches above it; the recorded
# tree hangs below it. Dropping /tf only costs you Pepper's body chain --
# including CameraTop_optical_frame, which the head camera images are stamped
# with and which then resolves against nothing.
#
# Pre-8edd1f5 bags may still carry the wheel edge. Check before trusting this:
#   ros2 bag play <bag> --topics /tf & ros2 run tf2_tools view_frames
#
# ARGUMENTS THIS FILE HONOURS (--show-args lists ~10 more that leak up from the
# include tree; they are settable but not all meaningful here):
#   config_file        l2_rsimu.yaml = RealSense IMU (default) | l2.yaml = L2's
#   rviz, rviz_cfg     open RViz, and with which config
#   scope              mount (default) | all -- pepper_sensor_tf's own
#                      argument, reached by inheritance. 'all' only for
#                      legacy bags that carry no /tf_static.
#   publisher          none = publish no rig transforms; for a bag that carries
#                      its own /tf_static
#   flatten_base_frame zero the leveled z/roll/pitch (default true)
#   use_sim_time       FORCED true by this wrapper -- do not pass it

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_launch_dir = os.path.join(
        get_package_share_directory('pepper_slam'), 'launch')

    # Defaults to the RealSense IMU. lidar_imu_frame and the sensor_tf scope are
    # DERIVED (from config_file and use_sim_time respectively) inside the live
    # launch, so switching IMU is genuinely one argument here.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config. l2_rsimu.yaml = RealSense IMU (default); '
                    'l2.yaml = the L2 s own, for A/B only (utils/L2_IMU/REPORT.md).')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')

    fastlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_launch_dir, 'fastlio_odometry.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'config_file': LaunchConfiguration('config_file'),
            'rviz': LaunchConfiguration('rviz'),
        }.items())

    ld = LaunchDescription()
    # This wrapper's whole job is forcing use_sim_time; say so, because
    # a bag replay with sim time NOT set fails silently rather than loudly.
    ld.add_action(LogInfo(msg=['[fastlio_odometry_bag] use_sim_time=true (forced)  '
                              'config_file=', LaunchConfiguration('config_file')]))
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(fastlio)
    return ld
