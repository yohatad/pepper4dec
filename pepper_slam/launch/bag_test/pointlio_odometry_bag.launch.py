# Point-LIO odometry on a recorded bag.
#
# Thin wrapper over pepper_slam/launch/pointlio_odometry.launch.py -- the live
# entry point -- with use_sim_time forced true.
#
#   ros2 launch pepper_slam pointlio_odometry_bag.launch.py
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 2000 --disable-keyboard-controls \
#     --topics /points /camera/imu /imu/data /tf /tf_static
#
# ARGUMENTS THIS FILE HONOURS:
#   config_file         l2lidar_rsimu.yaml = RealSense IMU (default) |
#                       l2lidar_node.yaml = the L2's own
#   rviz                open RViz
#   publisher           none (DEFAULT HERE) = publish no rig transforms, for
#                       a bag that carries its own /tf_static. Pass urdf
#                       (with scope:=all) for a legacy bag that does not.
#   flatten_base_frame  zero the leveled z/roll/pitch (default true)
#   use_sim_time        FORCED true here; do not pass it
#
# See README.md in this directory for the shared replay gotchas.

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

    # 'none': the bag carries its own /tf_static, and a second latched publisher
    # duplicates the rig edges -- whichever lands last silently wins. Keep it
    # DECLARED, not forwarded, or a launch_arguments entry shadows the command
    # line and makes publisher:=urdf a silent no-op.
    declare_publisher_cmd = DeclareLaunchArgument(
        'publisher', default_value='none',
        description="pepper_sensor_tf publisher: 'none' (default here) starts "
                    "neither, for a bag carrying its own /tf_static; "
                    "'urdf'/'yaml' publish the rig transforms.")

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
    # AFTER the declares: LogInfo resolves config_file immediately, and an
    # undeclared LaunchConfiguration raises at launch time. --show-args does
    # NOT catch this -- it never executes the action.
    ld.add_action(LogInfo(msg=['[pointlio_odometry_bag] use_sim_time=true (forced)  '
                               'config_file=', LaunchConfiguration('config_file')]))
    ld.add_action(pointlio)
    return ld
