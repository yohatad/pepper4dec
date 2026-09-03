# lio_odom_bridge, in one place, for every estimator.
#
# The bridge turns a LIO estimator's /odom_lio (lio_init -> <body frame>) into
# odom -> base_footprint, closing the tree per REP-105. FAST-LIO and Point-LIO
# need it identically, so include this file rather than starting the node
# yourself -- pass use_sim_time, config_path and config_file.
#
# THE BODY FRAME MUST EQUAL WHAT THE ESTIMATOR STAMPS, or the bridge composes
# odom -> base_footprint through the wrong rigid offset and yields a pose that
# looks plausible and is wrong. lidar_imu_frame now names it explicitly;
# passing '' falls back to reading publish.body_frame from the config, which
# makes the two impossible to desync.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_body_frame(context, *args, **kwargs):
    """Read publish.body_frame out of the estimator's config; explicit wins."""
    from launch.actions import LogInfo, SetLaunchConfiguration

    explicit = LaunchConfiguration('lidar_imu_frame').perform(context)
    if explicit:
        return [SetLaunchConfiguration('resolved_body_frame', explicit),
                LogInfo(msg='[lio_odom_bridge] body frame=%s (explicit)' % explicit)]

    import yaml
    path = os.path.join(
        LaunchConfiguration('config_path').perform(context),
        os.path.basename(LaunchConfiguration('config_file').perform(context)))
    try:
        with open(path) as fh:
            doc = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise RuntimeError(
            f"cannot read '{path}' to determine the estimator's body frame "
            f"({exc}). Pass lidar_imu_frame:=<frame> explicitly.")

    frame = None
    for top in doc.values():
        if isinstance(top, dict):
            params = top.get('ros__parameters', top)
            if isinstance(params, dict):
                pub = params.get('publish')
                if isinstance(pub, dict) and pub.get('body_frame'):
                    frame = pub['body_frame']
                    break

    # 'body' matches both estimators' own declare_parameter default, so a config
    # that omits publish.body_frame still lines up with what the node stamps.
    frame = frame or 'body'
    return [SetLaunchConfiguration('resolved_body_frame', frame),
            LogInfo(msg='[lio_odom_bridge] body frame=%s (from %s)'
                        % (frame, os.path.basename(path)))]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'config_path',
            default_value=os.path.join(
                get_package_share_directory('fast_lio'), 'config'),
            description="Directory holding the estimator config. Pass point_lio's "
                        'when bridging Point-LIO.'),
        DeclareLaunchArgument(
            'config_file', default_value='l2_rsimu.yaml',
            description='Config whose publish.body_frame the bridge must match.'),
        # Hardcoded to match every caller: the RealSense IMU is the only
        # configuration in use, and it is what l2_rsimu.yaml and
        # l2lidar_rsimu.yaml both name as publish.body_frame. Bridging an
        # L2-IMU config (l2.yaml, l2lidar_node.yaml -> l2lidar_frame_imu) now
        # needs the frame passed explicitly; an empty string restores the
        # read-it-from-the-config behaviour _resolve_body_frame implements.
        DeclareLaunchArgument(
            'lidar_imu_frame', default_value='camera_imu_optical_frame',
            description='Body frame the bridge stamps. Must match the '
                        'estimator config\'s publish.body_frame. Empty reads '
                        'it from that config instead.'),
        DeclareLaunchArgument(
            'bridge_level_frame', default_value='true',
            description='Publish the static odom -> odom leveling frame. False '
                        'when a higher layer owns odom (e.g. PGO map -> odom).'),
        DeclareLaunchArgument('level_frame_as_child', default_value='false'),
        DeclareLaunchArgument(
            'flatten_base_frame', default_value='true',
            description='Zero the leveled z/roll/pitch every cycle (keep x, y, '
                        'yaw). Safe here: Pepper is flat-floor only.'),
        DeclareLaunchArgument('odom_topic', default_value='/odom_lio'),

        # AFTER the declares: the resolver reads config_path/config_file.
        OpaqueFunction(function=_resolve_body_frame),

        Node(
            package='pepper_slam',
            executable='lio_odom_bridge.py',
            name='lio_odom_bridge',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'odom_topic': LaunchConfiguration('odom_topic'),
                'lidar_imu_frame': LaunchConfiguration('resolved_body_frame'),
                'flatten_base_frame': LaunchConfiguration('flatten_base_frame'),
                'publish_level_frame': LaunchConfiguration('bridge_level_frame'),
                'level_frame_as_child': LaunchConfiguration('level_frame_as_child'),
            }],
        ),
    ])
