# Point-LIO odometry on a recorded bag.
#
# Thin wrapper over pepper_slam/launch/pointlio_odometry.launch.py -- the live
# entry point -- with use_sim_time forced true.
#
#   ros2 launch pepper_slam pointlio_odometry_bag.launch.py publisher:=none
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 2000 --disable-keyboard-controls \
#     --topics /points /camera/imu /imu/data /tf /tf_static
#
# ARGUMENTS THIS FILE HONOURS:
#   config_file         l2lidar_rsimu.yaml = RealSense IMU (default) |
#                       l2lidar_node.yaml = the L2's own
#   rviz                open RViz
#   publisher           none = publish no rig transforms
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

    # Defaults to the RealSense IMU. lidar_imu_frame and the sensor_tf scope are
    # DERIVED (from config_file and use_sim_time respectively) inside the live
    # launch, so switching IMU is genuinely one argument here.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2lidar_rsimu.yaml',
        description='Point-LIO config under point_lio/config. l2lidar_rsimu.yaml '
                    '= RealSense IMU (default); l2lidar_node.yaml = the L2 s own, '
                    'for A/B only (utils/L2_IMU/REPORT.md).')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')

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
    # AFTER the declares: LogInfo resolves config_file immediately, and a
    # LaunchConfiguration that has not been declared yet raises
    # "launch configuration 'config_file' does not exist" at launch time.
    # --show-args does NOT catch this -- it never executes the action.
    ld.add_action(LogInfo(msg=['[pointlio_odometry_bag] use_sim_time=true (forced)  '
                              'config_file=', LaunchConfiguration('config_file')]))
    ld.add_action(pointlio)
    return ld
