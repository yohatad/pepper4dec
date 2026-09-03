# Plain FAST-LIO ODOMETRY on the Pepper L2 rig, with its required static TF.
#
# ODOMETRY, not SLAM. "mapping" is upstream's word for the ikd-Tree the
# estimator aligns each scan against -- there is no loop closure and nothing
# ever revisits a pose, so returning to a place after N metres of drift lays
# the same wall down twice, permanently. For a map worth keeping use
# fastlio_lc_pgo's fastlio_lc_l2.launch.py or bag_test/rtabmap_fastlio_bag.
# This file is the right tool for MEASURING odometry, precisely because
# nothing here hides the drift.
#
# It exists because fast_lio's own mapping.launch.py is not standalone-usable
# here: the bridge needs the static base_footprint -> l2lidar_frame_imu chain
# that only pepper_sensor_tf.launch.py provides, and mapping.launch.py is
# shared with every other FAST-LIO sensor config so it cannot bake that in.
# Forgetting the second launch file is a silent hang.
#
#   ros2 launch pepper_slam fastlio_odometry.launch.py
#   ros2 bag play <bag> --clock --topics /points /imu/data /tf /tf_static

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _echo_resolved(context, *args, **kwargs):
    """Echo the decisions that fail silently if wrong: use_sim_time true with
    no /clock pins time at 0, and publisher:=none against a bag with no
    /tf_static leaves the bridge waiting. Neither prints an error.
    """
    from launch.actions import LogInfo
    sim = LaunchConfiguration('use_sim_time').perform(context)
    pub = LaunchConfiguration('publisher', default='urdf').perform(context)
    scope = LaunchConfiguration('scope', default='mount').perform(context)
    return [LogInfo(msg='[pepper_slam] use_sim_time=%s  sensor_tf publisher=%s  scope=%s'
                        % (sim, pub, scope))]


def generate_launch_description():
    fast_lio_share = get_package_share_directory('fast_lio')
    pkg_share = get_package_share_directory('pepper_slam')

    rviz = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    use_sim_time = LaunchConfiguration('use_sim_time')
    bridge_level_frame = LaunchConfiguration('bridge_level_frame')
    flatten_base_frame = LaunchConfiguration('flatten_base_frame')

    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_publish_map_identity_cmd = DeclareLaunchArgument(
        'publish_map_identity', default_value='true',
        description='Publish a static identity map -> odom so "map" can be used '
                    'as a fixed frame in odometry-only runs. Set false when PGO '
                    'or a localizer owns that edge.')
    declare_rviz_cfg_cmd = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=os.path.join(fast_lio_share, 'rviz', 'fastlio.rviz'))
    # false, NOT true: this is the LIVE entry point, and 'true' on the robot
    # pins sim time at 0, so tf never resolves and nothing fuses, silently.
    # pepper_sensor_tf's publisher/scope are NOT derived from this -- on a bag
    # pass publisher:=none if it carries its own /tf_static, publisher:=urdf
    # scope:=all if it does not.
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='false (default) on the robot; true for bag replay with '
                    'ros2 bag play --clock. The bag_test wrappers set this for you.')
    declare_bridge_level_frame_cmd = DeclareLaunchArgument(
        'bridge_level_frame', default_value='true',
        description='Have lio_odom_bridge publish the static odom -> '
                    'odom leveling frame. Set false when a higher layer owns '
                    'odom (e.g. PGO publishing map -> odom).'
    )
    # The L2's own gyro cancels rotation about the gravity axis below ~16 deg/s
    # and cost 139 deg of heading over a 744 s run (utils/L2_IMU/REPORT.md).
    # The RealSense measured 3.8% -> 2.4% mean yaw error, 11.2% -> 4.6% worst,
    # so it is the permanent choice; l2.yaml is kept for A/B only.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config under fast_lio/config. l2_rsimu.yaml uses '
                    'the RealSense IMU (default); l2.yaml uses the L2 s own.')
    # Hardcoded, matching every other entry point, and NOT derived from
    # config_file: an A/B run against l2.yaml must pass
    # lidar_imu_frame:=l2lidar_frame_imu too, or the bridge stamps a frame the
    # estimator never publishes and odom -> base_footprint never closes. An
    # empty string restores the config-derived behaviour.
    declare_lidar_imu_frame_cmd = DeclareLaunchArgument(
        'lidar_imu_frame', default_value='camera_imu_optical_frame',
        description='Static frame the estimated body corresponds to. '
                    'camera_imu_optical_frame (default) for l2_rsimu.yaml; '
                    'l2lidar_frame_imu for l2.yaml. Empty derives it from '
                    'config_file instead.')
    declare_flatten_base_frame_cmd = DeclareLaunchArgument(
        'flatten_base_frame', default_value='true',
        description='Zero the leveled z/roll/pitch of odom -> base_footprint '
                    'every cycle (keep x, y, yaw). Defaults true here: Pepper '
                    'is confirmed flat-floor-only. Set false to see FAST-LIO\'s '
                    'own (drifting) z/roll/pitch instead.'
    )

    sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    fast_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_share, 'launch', 'mapping.launch.py')),
        launch_arguments={
            'config_file': LaunchConfiguration('config_file'),
            'lidar_imu_frame': LaunchConfiguration('lidar_imu_frame'),
            'rviz': rviz,
            'rviz_cfg': rviz_cfg,
            'use_sim_time': use_sim_time,
            'bridge_level_frame': bridge_level_frame,
            'flatten_base_frame': flatten_base_frame,
        }.items())

    # Odometry only, so the REP-105 map -> odom correction is identity by
    # definition. Publishing it makes 'map' a usable RViz fixed frame, so one
    # rviz config works with or without PGO/AMCL. MUST be false when something
    # else owns that edge (pgo_map_odom_bridge, AMCL, transform_fusion) --
    # two publishers give odom two parents and split the tree.
    map_identity = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_identity',
        arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('publish_map_identity')),
    )

    # The odom -> base_footprint bridge, which fast_lio's mapping.launch.py no
    # longer starts (it was Pepper glue in a file shared by every sensor config).
    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'lio_odom_bridge.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            # forwarded, not hardcoded: config_path selects where config_file
            # lives, and the bridge reads publish.body_frame from that same file.
            'config_path': LaunchConfiguration('config_path'),
            'config_file': LaunchConfiguration('config_file'),
            'lidar_imu_frame': LaunchConfiguration('lidar_imu_frame'),
            'bridge_level_frame': bridge_level_frame,
            'flatten_base_frame': flatten_base_frame,
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_lidar_imu_frame_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_cfg_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_bridge_level_frame_cmd)
    ld.add_action(declare_flatten_base_frame_cmd)
    # AFTER every DeclareLaunchArgument: the echo reads use_sim_time,
    # which does not exist in the context until its declare has run.
    ld.add_action(OpaqueFunction(function=_echo_resolved))
    ld.add_action(declare_publish_map_identity_cmd)
    ld.add_action(sensor_tf_launch)
    ld.add_action(map_identity)
    ld.add_action(fast_lio_launch)
    ld.add_action(bridge_launch)
    return ld
