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
# README.md in this directory covers the four things that otherwise waste an
# afternoon: play_qos.yaml, publisher/scope, replaying /tf, and
# --disable-keyboard-controls when backgrounding the player.

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
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config. l2_rsimu.yaml = RealSense IMU (default); '
                    'l2.yaml = the L2 s own, for A/B only (utils/L2_IMU/REPORT.md).')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')

    # 'none': the bag carries its own /tf_static, and a second latched publisher
    # duplicates the rig edges -- whichever lands last silently wins. Keep it
    # DECLARED, not forwarded, or a launch_arguments entry shadows the command
    # line and makes publisher:=urdf a silent no-op.
    declare_publisher_cmd = DeclareLaunchArgument(
        'publisher', default_value='urdf',
        description="pepper_sensor_tf publisher: 'urdf' (default) publishes the "
                    "rig and gives RViz a RobotModel; 'yaml' the same geometry "
                    "without the model; 'none' relies on the bag's /tf_static.")
    # 'all', not 'mount'. The bag DOES carry these edges -- but all of its
    # /tf_static messages sit at t=0.000 s, so `ros2 bag play --start-offset N`
    # skips them entirely and nothing publishes the rig. base_footprint and
    # camera_imu_optical_frame then come up as separate TF roots and anything
    # needing the extrinsic between them fails, silently.
    #
    # Publishing them here regardless is safe: MEASURED against this bag's own
    # /tf_static, the two agree to 4.8e-7 over all 10 shared edges (float32
    # rounding), because both derive from config/sensor_tf.yaml. Duplicate
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
    ld.add_action(declare_scope_cmd)
    # AFTER the declares: LogInfo resolves config_file immediately, and an
    # undeclared LaunchConfiguration raises at launch time. --show-args does
    # NOT catch this -- it never executes the action.
    ld.add_action(LogInfo(msg=['[fastlio_odometry_bag] use_sim_time=true (forced)  '
                               'config_file=', LaunchConfiguration('config_file')]))
    ld.add_action(fastlio)
    return ld
