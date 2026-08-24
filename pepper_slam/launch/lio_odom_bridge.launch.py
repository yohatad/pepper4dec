# lio_map_odom_bridge, in one place, for every estimator.
#
# The bridge turns a LIO estimator's /odom_lio (lio_init -> <body frame>) into
# odom -> base_footprint, closing the tree per REP-105. FAST-LIO and Point-LIO
# need it identically -- same six parameters, same topic -- so include this
# rather than starting the node yourself:
#
#     IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(os.path.join(
#             get_package_share_directory('pepper_slam'),
#             'launch', 'lio_odom_bridge.launch.py')),
#         launch_arguments={
#             'use_sim_time': use_sim_time,
#             'config_path': <the estimator's config dir>,
#             'config_file': <the config being used>,
#         }.items())
#
# THE BODY FRAME IS READ FROM THE CONFIG (publish.body_frame), not from a table.
# It must equal what the estimator stamps, or the bridge composes
# odom -> base_footprint through the wrong rigid offset and yields a pose that
# looks plausible and is wrong. Reading the yaml makes the two impossible to
# desync, and a new config needs no edit here.
#
# This used to be duplicated in FAST_LIO/launch/mapping.launch.py and
# point_lio/launch/mapping_l2lidar_node.launch.py, each with its own copy of the
# resolver and a hardcoded {config_file: frame} table.

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
        DeclareLaunchArgument(
            'lidar_imu_frame', default_value='',
            description='Override the body frame. Empty (default) reads it from '
                        'the config, which is what you want.'),
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
            executable='lio_map_odom_bridge.py',
            name='lio_map_odom_bridge',
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
