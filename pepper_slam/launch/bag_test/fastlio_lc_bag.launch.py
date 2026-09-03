# FAST-LIO + pose-graph loop closure (MAPPING), on a recorded bag.
#
# This is how you BUILD map_batch.pcd -- the prior every localization run ICPs
# against. After the bag finishes, call:
#   ros2 service call /pgo_batch_optimize std_srvs/srv/Trigger
# Without that call nothing is written but optimized_poses.txt and Scans/.
#
# Thin wrapper over fastlio_lc_pgo/launch/fastlio_lc_l2.launch.py -- the live entry
# point -- with use_sim_time forced true. See bag_test/README.md.
#
# Usage:
#   ros2 launch pepper_slam fastlio_lc_bag.launch.py
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 2000
#
# The QoS overrides are REQUIRED: /imu/data and /camera/imu were recorded
# BEST_EFFORT and a RELIABLE subscriber matches nothing against them, so without
# the file the estimator waits forever for IMU init and prints nothing.
# Replaying /tf is SAFE and wanted -- see pepper_sensor_tf.launch.py's
# header for why the old "do not replay /tf" advice no longer holds. Bags
# recorded before commit 8edd1f5 may still carry a wheel-odometry edge that
# claims base_footprint as a child; check with
# `ros2 bag play <bag> --topics /tf & ros2 run tf2_tools view_frames` before
# replaying /tf from one of those.
#
# keyframe_filter_size matters here: it is applied BEFORE keyframes are stored,
# so map_save_filter_size can never recover resolution it discarded. 0.25
# matches FAST-LIO's own filter_size_surf, the real floor.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    launch_dir = os.path.join(
        get_package_share_directory('fastlio_lc_pgo'), 'launch')

    # No default: this is the output of a PREVIOUS mapping run, not something
    # this launch can produce. It used to default to a path under /home/yoha
    # that exists on one machine only; a wrong or missing map then failed
    # somewhere inside PGO instead of at launch. Pass it explicitly.
    declare_save_dir_cmd = DeclareLaunchArgument(
        'save_directory',
        default_value='',
        description='Where PGO writes optimized_poses.txt, Scans/ and (on '
                    '/pgo_batch_optimize) map_batch.pcd.')
    declare_kf_filter_cmd = DeclareLaunchArgument(
        'keyframe_filter_size', default_value='0.25',
        description='Voxel leaf applied to each keyframe BEFORE storage, so it '
                    'bounds the density of every downstream product.')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='false')

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

    inner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'fastlio_lc_l2.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'save_directory': LaunchConfiguration('save_directory'),
            'keyframe_filter_size': LaunchConfiguration('keyframe_filter_size'),
            'rviz': LaunchConfiguration('rviz'),
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_save_dir_cmd)
    ld.add_action(declare_kf_filter_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_publisher_cmd)
    ld.add_action(inner)
    return ld
