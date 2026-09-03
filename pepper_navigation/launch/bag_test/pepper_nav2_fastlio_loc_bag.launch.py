# Nav2 + FAST-LIO localization on a recorded bag.
#
# Thin wrapper over pepper_navigation/launch/pepper_nav2_fastlio_loc.launch.py --
# the live entry point -- with use_sim_time forced true.
#
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
# THE MAP ARGUMENTS HAVE NO DEFAULTS on purpose: they are the output of a
# previous mapping run, not something this launch can produce. A silent default
# pointing into one machine's home directory turns a missing map into a failure
# somewhere inside localization instead of at launch.
#
# For the shared replay gotchas -- why play_qos.yaml is mandatory, which of
# publisher/scope your bag needs, why replaying /tf is correct, and why
# backgrounding the player needs --disable-keyboard-controls -- see
# pepper_slam/launch/bag_test/README.md.

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

        # 'none': the bag carries its own /tf_static, and a second latched
        # publisher duplicates the rig edges -- whichever lands last silently
        # wins. Pass publisher:=urdf scope:=all for a legacy bag without one.
        # Keep this DECLARED, not forwarded: a launch_arguments entry would
        # shadow the command line and make that override a silent no-op.
        DeclareLaunchArgument(
            'publisher', default_value='none',
            description="pepper_sensor_tf publisher: 'none' (default here) "
                        "starts neither, for a bag carrying its own "
                        "/tf_static; 'urdf'/'yaml' publish the rig transforms."),

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
