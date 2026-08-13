# FAST-LIO odometry only, on a recorded bag.
#
# Thin wrapper over pepper_slam/launch/fastlio_odometry.launch.py -- the live
# entry point -- with use_sim_time forced true. Nothing about the stack is
# duplicated here; see bag_test/README.md for the convention.
#
# Usage:
#   ros2 launch pepper_slam fastlio_odometry_bag.launch.py
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 2000
#
# The QoS overrides are REQUIRED: /imu/data and /camera/imu were recorded
# BEST_EFFORT, and a RELIABLE subscriber matches nothing against them, so
# without the file the estimator waits forever for IMU init and prints nothing.
# Do NOT replay /tf -- the bag's wheel odometry fights the bridge for
# base_footprint's parent.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
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
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(fastlio)
    return ld
