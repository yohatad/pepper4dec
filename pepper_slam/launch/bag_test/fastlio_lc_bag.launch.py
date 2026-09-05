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
# BEST_EFFORT, so without them the estimator waits forever for IMU init and
# prints nothing. Replaying /tf is safe and wanted. README.md in this directory
# has both in full, plus the pre-8edd1f5 bags that need a check first.
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
    # this launch can produce. Pass it explicitly.
    declare_save_dir_cmd = DeclareLaunchArgument(
        'save_directory',
        default_value='',
        description='Where PGO writes optimized_poses.txt, Scans/ and (on '
                    '/pgo_batch_optimize) map_batch.pcd.')
    declare_kf_filter_cmd = DeclareLaunchArgument(
        'keyframe_filter_size', default_value='0.25',
        description='Voxel leaf applied to each keyframe BEFORE storage, so it '
                    'bounds the density of every downstream product.')
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
    ld.add_action(declare_scope_cmd)
    ld.add_action(inner)
    return ld
