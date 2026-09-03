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
# BEST_EFFORT, so without them the estimator waits forever for IMU init and
# prints nothing. Replaying /tf is safe and wanted. README.md in this directory
# has both in full, plus the pre-8edd1f5 bags that need a check first.
#
# map_pcd and keyframe_poses MUST come from the same mapping run. Build them
# with bag_test/fastlio_lc_bag.launch.py, then /pgo_batch_optimize.
#
# SEEDING. With no seed the node waits, and rightly so -- auto_initialize picks
# a wrong hypothesis often enough in a corridor that a silent bad lock is the
# likelier outcome than a good one (the global search scores candidates by ICP
# fitness, which in a corridor cannot separate the true pose from several wrong
# ones). seed_from_map_start:=true (the default) instead publishes /initialpose
# at the FIRST KEYFRAME of keyframe_poses -- where the mapping run began, which
# is where a replay of that same run begins too. It is read from the poses file
# rather than hardcoded, so it follows whatever map is being localized against.
#
# Turn it off (seed_from_map_start:=false) when replaying a DIFFERENT bag than
# the map was built from: the start pose would then be a fabrication, and a
# confidently wrong seed is worse than none. Use /relocalize or /initialpose.

import os

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def _seed_from_first_keyframe(context, *args, **kwargs):
    """
    Publish /initialpose at the pose the mapping run started from.

    The poses file holds 3x4 row-major map <- body matrices. The seed must be
    map -> base_footprint (initial_pose_is_base), so only x, y and yaw carry
    over: base_footprint is on the floor, so z is 0, and roll/pitch are not the
    robot's to have.
    """
    import math

    # keyframe_poses is deliberately undeclared here (see the note in
    # generate_launch_description), and LaunchConfiguration.perform() RAISES on an
    # undeclared name rather than returning '' -- so this launch died outright with
    # "launch configuration 'keyframe_poses' does not exist" whenever it was not
    # passed on the command line, i.e. the normal case, and the fallback below was
    # unreachable. Read the context dict instead: it still holds the value when
    # keyframe_poses:=<path> is passed, and is simply absent otherwise.
    path = context.launch_configurations.get('keyframe_poses', '')
    if not path:
        from ament_index_python.packages import get_package_share_directory as g
        path = os.path.join(g('pepper_navigation'), 'pcd',
                            'pepper_map_lc_poses.txt')
    try:
        with open(path) as f:
            v = [float(x) for x in f.readline().split()]
        if len(v) != 12:
            raise ValueError(f'expected 12 values per row, got {len(v)}')
    except Exception as exc:
        from launch.actions import LogInfo
        return [LogInfo(msg=f'[seed] cannot read {path} ({exc}); '
                            f'not seeding -- use /initialpose or /relocalize.')]

    x, y = v[3], v[7]
    yaw = math.atan2(v[4], v[0])          # atan2(R10, R00)
    qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    msg = ('{header: {frame_id: "map"}, pose: {pose: '
           f'{{position: {{x: {x:.6f}, y: {y:.6f}, z: 0.0}}, '
           f'orientation: {{z: {qz:.9f}, w: {qw:.9f}}}}}}}}}')

    from launch.actions import LogInfo
    return [
        LogInfo(msg=f'[seed] map start x={x:.3f} y={y:.3f} '
                    f'yaw={math.degrees(yaw):+.2f} deg (from {os.path.basename(path)})'),
        # Delayed: FAST-LIO must finish IMU init and publish odom before a seed
        # means anything, and `ros2 topic pub --once` exits immediately, so it
        # has to fire after the subscriber exists rather than before.
        TimerAction(period=LaunchConfiguration('seed_delay'), actions=[
            ExecuteProcess(
                cmd=['ros2', 'topic', 'pub', '--times', '3', '-w', '1',
                     '/initialpose',
                     'geometry_msgs/msg/PoseWithCovarianceStamped', msg],
                output='screen')]),
    ]


def generate_launch_description():
    launch_dir = os.path.join(
        get_package_share_directory('lio_localization'), 'launch')

    # map_pcd and keyframe_poses are deliberately NOT declared here. The inner
    # launch already defaults them to the copies installed in pepper_navigation's
    # share, which is package-relative and so resolves on any machine -- the
    # concern that originally motivated an empty default here. Redeclaring them
    # empty and forwarding that empty SHADOWED those defaults, and this wrapper
    # died on "map_pcd '' does not exist" while the map sat where the inner
    # launch was already looking. Undeclared arguments still reach the inner
    # launch from the command line, so map_pcd:=<path> works as before.
    declare_auto_cmd = DeclareLaunchArgument(
        'auto_initialize', default_value='false',
        description='Search automatically at startup instead of waiting for a '
                    'seed. Off by default: a silent wrong lock is worse.')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_seed_cmd = DeclareLaunchArgument(
        'seed_from_map_start', default_value='true',
        description='Publish /initialpose at the first keyframe of '
                    'keyframe_poses. Turn off when replaying a bag the map was '
                    'NOT built from.')
    declare_seed_delay_cmd = DeclareLaunchArgument(
        'seed_delay', default_value='25.0',
        description='Seconds to wait before seeding, so FAST-LIO has finished '
                    'IMU init and is publishing odometry.')

    # 'none': the bag carries its own /tf_static, and a second latched publisher
    # duplicates the rig edges -- whichever lands last silently wins. Keep it
    # DECLARED, not forwarded, or a launch_arguments entry shadows the command
    # line and makes publisher:=urdf a silent no-op.
    declare_publisher_cmd = DeclareLaunchArgument(
        'publisher', default_value='none',
        description="pepper_sensor_tf publisher: 'none' (default here) starts "
                    "neither, for a bag carrying its own /tf_static; "
                    "'urdf'/'yaml' publish the rig transforms.")

    inner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'fastlio_localization_l2.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'auto_initialize': LaunchConfiguration('auto_initialize'),
            'rviz': LaunchConfiguration('rviz'),
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_auto_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_seed_cmd)
    ld.add_action(declare_seed_delay_cmd)
    ld.add_action(declare_publisher_cmd)
    ld.add_action(OpaqueFunction(
        function=_seed_from_first_keyframe,
        condition=IfCondition(LaunchConfiguration('seed_from_map_start'))))
    ld.add_action(inner)
    return ld
