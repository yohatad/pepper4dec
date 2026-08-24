# Nav2 + FAST-LIO localization, on a recorded bag.
#
# Thin wrapper over pepper_navigation/launch/pepper_nav2_fastlio_loc.launch.py --
# the live entry point -- with use_sim_time forced true. Nothing about the stack
# is duplicated here; same convention as pepper_slam/launch/bag_test/README.md.
#
# Usage:
#   ros2 launch pepper_navigation pepper_nav2_fastlio_loc_bag.launch.py \
#       map_pcd:=<run>/map_batch.pcd \
#       map:=<run>/grid.yaml \
#       keyframe_poses:=<run>/optimized_poses.txt
#
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 2000 --disable-keyboard-controls \
#     --topics /points /camera/imu /imu/data /tf /tf_static
#
# The map arguments have NO defaults on purpose: they are the output of a
# previous mapping run, not something this launch can produce. A silent default
# pointing at one machine's home directory turned a missing map into a failure
# somewhere inside localization instead of at launch.
#
# --- things that will waste your afternoon if you skip them -----------------
#
# play_qos.yaml is REQUIRED: /imu/data and /camera/imu were recorded BEST_EFFORT
# and the estimator subscribes RELIABLE, which matches nothing -- it then waits
# forever for IMU init and prints nothing at all.
#
# --disable-keyboard-controls is needed if you background the player: rosbag2
# reads the terminal for its pause/resume keys, and a background job doing that
# gets SIGTTIN and stops dead.
#
# REPLAYING /tf IS SAFE and wanted -- it carries Pepper's body chain including
# CameraTop_optical_frame. See pepper_sensor_tf.launch.py's header for why the
# old "do not replay /tf" advice no longer holds.
#
# For a bag recorded with config/record_qos.yaml -- one that carries its own
# /tf_static -- add publisher:=none. That disables the live rig publisher
# entirely so the bag is the sole source of the transforms, instead of both
# publishing the same latched edges. 'publisher' is pepper_sensor_tf's own
# argument and reaches it by inheritance; do not forward it under an alias from
# here, because an explicit launch_arguments entry SHADOWS the command-line
# value of the inner name (measured 2026-08-24).

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    live_launch = os.path.join(
        get_package_share_directory('pepper_navigation'),
        'launch', 'pepper_nav2_fastlio_loc.launch.py')

    map_pcd = LaunchConfiguration('map_pcd')
    grid_map = LaunchConfiguration('map')
    keyframe_poses = LaunchConfiguration('keyframe_poses')
    config_file = LaunchConfiguration('config_file')
    rviz = LaunchConfiguration('rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map_pcd', default_value='',
            description='REQUIRED. map_batch.pcd from the mapping run.'),
        DeclareLaunchArgument(
            'map', default_value='',
            description='REQUIRED. Nav2 occupancy grid .yaml from the same run.'),
        DeclareLaunchArgument(
            'keyframe_poses', default_value='',
            description='REQUIRED. optimized_poses.txt from the same run.'),
        DeclareLaunchArgument(
            'config_file', default_value='l2_rsimu.yaml',
            description='FAST-LIO config. l2_rsimu.yaml = RealSense IMU '
                        '(default); l2.yaml = the L2 s own, for A/B only.'),
        DeclareLaunchArgument('rviz', default_value='true'),

        # Echo the decisions this wrapper made. Every failure mode above is
        # silent, so one line naming the resolved values is worth more than the
        # comments explaining them.
        LogInfo(msg=['[nav2_fastlio_loc_bag] use_sim_time=true (forced)  ',
                     'config_file=', config_file, '  map_pcd=', map_pcd]),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(live_launch),
            launch_arguments={
                'use_sim_time': 'true',
                'map_pcd': map_pcd,
                'map': grid_map,
                'keyframe_poses': keyframe_poses,
                'config_file': config_file,
                'rviz': rviz,
            }.items()),
    ])
