# FAST-LIO + prior-map ICP LOCALIZATION, on a recorded bag.
#
# Thin wrapper over lio_localization/launch/fastlio_localization_l2.launch.py -- the live entry
# point -- with use_sim_time forced true. See bag_test/README.md.
#
# Usage:
#   ros2 launch pepper_slam fastlio_localization_bag.launch.py
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 2000
#
# The QoS overrides are REQUIRED: /imu/data and /camera/imu were recorded
# BEST_EFFORT and a RELIABLE subscriber matches nothing against them, so without
# the file the estimator waits forever for IMU init and prints nothing.
# Do NOT replay /tf -- the bag's wheel odometry fights the bridge for
# base_footprint's parent.
#
# map_pcd and keyframe_poses MUST come from the same mapping run. Build them
# with bag_test/fastlio_lc_bag.launch.py, then /pgo_batch_optimize.
#
# With no seed the node waits. Either publish /initialpose, or call
#   ros2 service call /relocalize std_srvs/srv/Trigger
# to search the keyframe poses, or pass auto_initialize:=true.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    launch_dir = os.path.join(
        get_package_share_directory('lio_localization'), 'launch')

        # No default: this is the output of a PREVIOUS mapping run, not something
        # this launch can produce. It used to default to a path under /home/yoha
        # that exists on one machine only; a wrong or missing map then failed
        # somewhere inside PGO instead of at launch. Pass it explicitly.
    declare_map_cmd = DeclareLaunchArgument(
        'map_pcd',
        default_value='',
        description='Prior map to localize against.')
    declare_kf_cmd = DeclareLaunchArgument(
        'keyframe_poses',
        default_value='',
        description='Candidates for /relocalize and auto_initialize. MUST be '
                    'from the same run as map_pcd.')
    declare_auto_cmd = DeclareLaunchArgument(
        'auto_initialize', default_value='false',
        description='Search automatically at startup instead of waiting for a '
                    'seed. Off by default: a silent wrong lock is worse.')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')

    inner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'fastlio_localization_l2.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'map_pcd': LaunchConfiguration('map_pcd'),
            'keyframe_poses': LaunchConfiguration('keyframe_poses'),
            'auto_initialize': LaunchConfiguration('auto_initialize'),
            'rviz': LaunchConfiguration('rviz'),
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_map_cmd)
    ld.add_action(declare_kf_cmd)
    ld.add_action(declare_auto_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(inner)
    return ld
