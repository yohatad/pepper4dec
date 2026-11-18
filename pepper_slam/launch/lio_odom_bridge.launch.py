"""lio_odom_bridge.launch.py

lio_odom_bridge, in one place, for every estimator.

The bridge turns a LIO estimator's /odom_lio (lio_init -> <body frame>) into
odom -> base_footprint, closing the tree per REP-105. FAST-LIO and Point-LIO
need it identically, so include this file rather than starting the node
yourself — pass use_sim_time, config_path and config_file.

Nodes started:
    pepper_slam/lio_odom_bridge (node: lio_odom_bridge), parameters from
    config/pepper_slam_configuration.yaml

Launch arguments:
    use_sim_time (default: "false")
    config_path (default: <fast_lio share>/config)
        Directory holding config_file.
    config_file (default: "l2_rsimu.yaml")
        Estimator config the body frame is read from when lidar_imu_frame is
        empty.
    lidar_imu_frame (default: "camera_imu_optical_frame")
        The body frame the estimator stamps. Passing '' falls back to reading
        publish.body_frame from the config.
    bridge_level_frame, level_frame_as_child, flatten_base_frame, odom_topic,
    guard_enable, wheel_odom_topic, max_linear_speed, max_angular_speed,
    max_hold_duration (default: "" for all)
        Overrides. The values live in config/pepper_slam_configuration.yaml;
        a non-empty argument replaces the file's value for this launch only.
        bridge_level_frame sets the node's publish_level_frame.

THE BODY FRAME MUST EQUAL WHAT THE ESTIMATOR STAMPS, or the bridge composes
odom -> base_footprint through the wrong rigid offset and yields a pose that
looks plausible and is wrong. lidar_imu_frame names it explicitly, which makes
the two impossible to desync.

Usage:
    Included by fastlio_odometry.launch.py, pointlio_odometry.launch.py, and
    the Nav2 profiles rather than launched directly.

The node's behavior is documented in src/lio_odom_bridge.cpp.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 8, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

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


# Launch arguments that override config/pepper_slam_configuration.yaml, and
# the node parameter each one sets.
_OVERRIDES = {
    'bridge_level_frame': 'Publish the static odom leveling frame (node '
                          'parameter publish_level_frame). False when a higher '
                          'layer owns odom, e.g. PGO map -> odom.',
    'level_frame_as_child': 'Publish the leveling frame as the child.',
    'flatten_base_frame': 'Zero the leveled z/roll/pitch every cycle.',
    'odom_topic': "The estimator's odometry topic.",
    'guard_enable': 'Reject LIO poses that break a physical speed bound and '
                    'dead reckon on wheel odometry instead.',
    'wheel_odom_topic': 'Wheel odometry the guard dead reckons on.',
    'max_linear_speed': 'm/s the base cannot exceed.',
    'max_angular_speed': 'rad/s the base cannot exceed.',
    'max_hold_duration': 'Seconds the pose may coast on wheels before FAULT.',
}
_PARAM_NAME = {'bridge_level_frame': 'publish_level_frame'}


def _bridge_node(context, *args, **kwargs):
    """Start the bridge from its configuration file plus the passed overrides."""
    import yaml
    overrides = {}
    for name in _OVERRIDES:
        value = LaunchConfiguration(name).perform(context)
        if value != '':
            overrides[_PARAM_NAME.get(name, name)] = yaml.safe_load(value)
    return [Node(
        package='pepper_slam',
        executable='lio_odom_bridge',
        name='lio_odom_bridge',
        output='screen',
        parameters=[
            os.path.join(get_package_share_directory('pepper_slam'), 'config',
                         'pepper_slam_configuration.yaml'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'lidar_imu_frame': LaunchConfiguration('resolved_body_frame'),
                **overrides,
            },
        ],
    )]


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
        # l2lidar_rsimu.yaml both name as publish.body_frame. Bridging
        # Point-LIO's L2-IMU config (l2lidar_node.yaml -> l2lidar_frame_imu)
        # needs the frame passed explicitly; an empty string restores the
        # read-it-from-the-config behaviour _resolve_body_frame implements.
        DeclareLaunchArgument(
            'lidar_imu_frame', default_value='camera_imu_optical_frame',
            description='Body frame the bridge stamps. Must match the '
                        'estimator config\'s publish.body_frame. Empty reads '
                        'it from that config instead.'),
        # Overrides of config/pepper_slam_configuration.yaml. Empty (the
        # default) keeps the file's value; anything else replaces it.
        *[DeclareLaunchArgument(name, default_value='', description=desc)
          for name, desc in _OVERRIDES.items()],
        # AFTER the declares: the resolver reads config_path/config_file.
        OpaqueFunction(function=_resolve_body_frame),
        OpaqueFunction(function=_bridge_node),
    ])
