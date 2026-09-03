# FAST-LIO odometry on a recorded bag.
#
# Thin wrapper over pepper_slam/launch/fastlio_odometry.launch.py -- the live
# entry point -- with use_sim_time forced true. Nothing is duplicated here.
#
#   ros2 launch pepper_slam fastlio_odometry_bag.launch.py
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 2000 --disable-keyboard-controls \
#     --topics /points /camera/imu /imu/data /tf /tf_static
#
# ARGUMENTS THIS FILE HONOURS  (--show-args lists ~10 more that leak up from the
# include tree and are not all meaningful here):
#
#   config_file         l2_rsimu.yaml = RealSense IMU (default) | l2.yaml = L2's
#   rviz, rviz_cfg      open RViz, and with which config
#   publisher           none (DEFAULT HERE) = publish no rig transforms, for
#                       a bag that carries its own /tf_static. Pass urdf for
#                       a legacy bag that does not.
#   scope               mount (default) | all -- 'all' only for legacy bags,
#                       and only alongside publisher:=urdf
#   flatten_base_frame  zero the leveled z/roll/pitch (default true)
#   use_sim_time        FORCED true here; do not pass it
#
# See README.md in this directory for the four things that will otherwise waste
# an afternoon: why play_qos.yaml is mandatory, which of publisher/scope your bag
# needs, why replaying /tf is correct, and why backgrounding the player needs
# --disable-keyboard-controls.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_launch_dir = os.path.join(
        get_package_share_directory('pepper_slam'), 'launch')

    # Defaults to the RealSense IMU. lidar_imu_frame IS derived from config_file
    # inside the live launch (its default is empty and mapping.launch.py picks
    # the frame), so switching IMU is genuinely one argument here. The sensor_tf
    # scope is NOT derived from anything -- pass scope:=all explicitly, together
    # with publisher:=urdf, for a legacy bag with an empty /tf_static.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config. l2_rsimu.yaml = RealSense IMU (default); '
                    'l2.yaml = the L2 s own, for A/B only (utils/L2_IMU/REPORT.md).')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')

    # 'none': the bag carries its own /tf_static, and a second latched publisher
    # duplicates the rig edges -- whichever lands last silently wins. Pass
    # publisher:=urdf scope:=all for a legacy bag with an empty /tf_static.
    # Keep this DECLARED, not forwarded: a launch_arguments entry would shadow
    # the command line and make the override above a silent no-op.
    declare_publisher_cmd = DeclareLaunchArgument(
        'publisher', default_value='none',
        description="pepper_sensor_tf publisher: 'none' (default here) starts "
                    "neither, for a bag carrying its own /tf_static; "
                    "'urdf'/'yaml' publish the rig transforms.")

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
    ld.add_action(declare_publisher_cmd)
    # AFTER the declares: LogInfo resolves config_file immediately, and a
    # LaunchConfiguration that has not been declared yet raises
    # "launch configuration 'config_file' does not exist" at launch time.
    # --show-args does NOT catch this -- it never executes the action.
    ld.add_action(LogInfo(msg=['[fastlio_odometry_bag] use_sim_time=true (forced)  '
                               'config_file=', LaunchConfiguration('config_file')]))
    ld.add_action(fastlio)
    return ld
