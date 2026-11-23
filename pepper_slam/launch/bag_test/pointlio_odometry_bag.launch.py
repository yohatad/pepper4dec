r"""pointlio_odometry_bag.launch.py

Point-LIO odometry on a recorded bag.

Thin wrapper over pepper_slam/launch/pointlio_odometry.launch.py -- the live
entry point -- with use_sim_time forced true.

  ros2 launch pepper_slam pointlio_odometry_bag.launch.py
  ros2 bag play <bag> --clock \
    --read-ahead-queue-size 2000 --disable-keyboard-controls \
    --topics /points /camera/imu /imu/data /tf /tf_static

Launch arguments::
  config_file         l2lidar_rsimu.yaml = RealSense IMU (default) |
                      l2lidar_node.yaml = the L2's own
  rviz                open RViz
  publisher           urdf (default) = republish the rig from the URDF |
                      none = rely on the bag's own /tf_static (only when
                      playing from the start, with no --start-offset)
  scope               all (default here) = include the RealSense-internal
                      edges, which a bag replay needs | mount
  flatten_base_frame  zero the leveled z/roll/pitch (default true)
  use_sim_time        FORCED true here; do not pass it

"Bag replay" in the package README covers the shared replay gotchas.

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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_launch_dir = os.path.join(
        get_package_share_directory('pepper_slam'), 'launch')

    # Defaults to the RealSense IMU, and lidar_imu_frame is now hardcoded to the
    # matching camera_imu_optical_frame rather than derived, so an A/B run
    # against the L2's own IMU needs lidar_imu_frame:=l2lidar_frame_imu passed
    # alongside config_file. The sensor_tf scope is not derived either -- pass
    # scope:=all with publisher:=urdf for a legacy bag with an empty /tf_static.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2lidar_rsimu.yaml',
        description='Point-LIO config under point_lio/config. l2lidar_rsimu.yaml '
                    '= RealSense IMU (default); l2lidar_node.yaml = the L2 s own, '
                    'for A/B only (utils/L2_IMU/REPORT.md).')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')

    # 'urdf' with scope 'all' (below): the whole rig is republished even when
    # the bag carries it, because --start-offset skips the bag's copy. Pass
    # 'none' to replay a bag's own /tf_static from the start instead. Keep it
    # DECLARED, not forwarded, or a launch_arguments entry shadows the command
    # line and makes publisher:=none a silent no-op.
    declare_publisher_cmd = DeclareLaunchArgument(
        'publisher', default_value='urdf',
        description="pepper_sensor_tf publisher: 'urdf' (default) publishes the "
                    "rig and gives RViz a RobotModel; 'none' relies on the "
                    "bag's /tf_static.")
    # 'all', not 'mount'. The bag DOES carry these edges -- but all of its
    # /tf_static messages sit at t=0.000 s, so `ros2 bag play --start-offset N`
    # skips them entirely and nothing publishes the rig. base_footprint and
    # camera_imu_optical_frame then come up as separate TF roots and anything
    # needing the extrinsic between them fails, silently.
    #
    # Publishing them here regardless is safe: MEASURED against this bag's own
    # /tf_static, the two agree to 4.8e-7 over all 10 shared edges (float32
    # rounding), because both come from the same rig calibration. Duplicate
    # publishers of IDENTICAL geometry are redundant, not harmful.
    #
    # Live is the opposite case and wants 'mount': there the RealSense driver
    # publishes its internal chain from the device's factory calibration, which
    # is a genuinely different source, and two publishers disagreeing is a
    # silent intermittent wrong answer.
    declare_scope_cmd = DeclareLaunchArgument(
        'scope', default_value='all', choices=['mount', 'all'],
        description="Rig transforms to publish. 'all' includes the RealSense "
                    "internal chain, needed when the bag's /tf_static is not "
                    "replayed. Use 'mount' on the live robot.")

    pointlio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_launch_dir, 'pointlio_odometry.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'config_file': LaunchConfiguration('config_file'),
            'rviz': LaunchConfiguration('rviz'),
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_publisher_cmd)
    ld.add_action(declare_scope_cmd)
    # AFTER the declares: LogInfo resolves config_file immediately, and an
    # undeclared LaunchConfiguration raises at launch time. --show-args does
    # NOT catch this -- it never executes the action.
    ld.add_action(LogInfo(msg=['[pointlio_odometry_bag] use_sim_time=true (forced)  '
                               'config_file=', LaunchConfiguration('config_file')]))
    ld.add_action(pointlio)
    return ld
