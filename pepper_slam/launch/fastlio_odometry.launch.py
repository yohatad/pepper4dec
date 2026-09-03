# Plain FAST-LIO ODOMETRY on the Pepper L2 rig, with its required static TF.
#
# RENAMED 2026-08-10 from fastlio_mapping.launch.py. "mapping" is upstream
# FAST-LIO's word for scan-to-map REGISTRATION -- the ikd-Tree map the estimator
# keeps in order to align each new scan. It is a map used FOR odometry, not a
# SLAM map. There is no loop closure, no pose-graph optimisation and no
# relocalisation here: every scan is stamped into /Laser_map at whatever pose
# odometry believed at that instant and nothing ever revisits it, so revisiting
# a place after N metres of drift lays the same wall down twice, permanently.
# The old name led to exactly that surprise. For a map worth keeping use
# fastlio_lc_pgo fastlio_lc_l2.launch.py (Scan Context + GTSAM) or
# pepper_slam bag_test/rtabmap_fastlio_bag.launch.py.
#
# This file is still the right tool for MEASURING odometry quality, precisely
# because nothing here hides the drift.
#
# ros2 launch fast_lio mapping.launch.py by itself is not standalone-usable
# on this robot: lio_odom_bridge.py needs the static base_footprint ->
# l2lidar_frame -> l2lidar_frame_imu chain, which only pepper_sensor_tf.launch.py
# provides, and mapping.launch.py doesn't include it (that file is shared
# across every FAST-LIO sensor config in this workspace -- mid360, velodyne,
# avia, etc. -- so it can't bake in Pepper-specific transforms without
# breaking all of those). Forgetting the second launch file is a silent-hang
# footgun; this wraps both together the same way fastlio_lc_l2.launch.py
# (fastlio_lc_pgo) and fastlio_localization_l2.launch.py (lio_localization)
# already do for their own use cases -- this is the missing plain-mapping
# equivalent.
#
# Usage:
#   ros2 launch pepper_slam fastlio_odometry.launch.py
#   ros2 bag play <bag> --clock --topics /points /imu/data /tf /tf_static
#   (replaying /tf is SAFE and wanted -- see pepper_sensor_tf.launch.py's
#    header for why the old "do not replay /tf" advice no longer holds.)
#
# flatten_base_frame defaults to true HERE (unlike fast_lio's own
# mapping.launch.py, which defaults it false): this file is Pepper-specific
# and Pepper is confirmed to only ever run on flat floor, so the hard
# flat-floor assumption is safe here in a way it isn't for the generic,
# multi-sensor fast_lio package. Pass flatten_base_frame:=false to get
# FAST-LIO's own (drifting) z/roll/pitch back.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _echo_resolved(context, *args, **kwargs):
    """Print the decisions that fail silently if they are wrong.

    use_sim_time true with no /clock pins time at 0 and nothing renders or
    fuses; publisher:=none against a bag with NO /tf_static leaves the rig
    transforms missing and lio_odom_bridge simply waits. Neither prints an
    error. One line here beats bisecting either.
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
    # false, NOT true: this is the LIVE entry point. Every wrapper in
    # pepper_slam/launch/bag_test sets use_sim_time:='true' explicitly, so this
    # default only ever applies on the robot -- where 'true' pins sim time at 0,
    # so tf never resolves and nothing fuses, silently and with no error.
    # pepper_sensor_tf's 'publisher'/'scope' are NOT derived from this -- only
    # use_sim_time is forwarded. On a bag, pass them yourself: publisher:=none
    # if it carries its own /tf_static, publisher:=urdf scope:=all if it does
    # not. The bag_test wrappers already default publisher to none.
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
    # 2026-08-12: the RealSense IMU is now the permanent choice for this rig.
    # The L2's own gyro cancels rotation about the gravity axis below ~16 deg/s
    # and cost 139 deg of heading over a 744 s run (utils/L2_IMU/REPORT.md);
    # l2_rsimu.yaml drives the same estimator from /camera/imu instead and
    # measured 3.8% -> 2.4% mean yaw error, 11.2% -> 4.6% worst. l2.yaml is kept
    # selectable for A/B work only.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config under fast_lio/config. l2_rsimu.yaml uses '
                    'the RealSense IMU (default); l2.yaml uses the L2 s own.')
    # Left unset on purpose: mapping.launch.py derives the matching frame from
    # config_file, so the two cannot drift apart. Override only for a config
    # this launch does not know about.
    declare_lidar_imu_frame_cmd = DeclareLaunchArgument(
        'lidar_imu_frame', default_value='',
        description='Override the static frame the estimated body corresponds '
                    'to. Empty (default) lets mapping.launch.py pick it from '
                    'config_file.')
    # (A block here used to describe a 'scope' argument DERIVED from
    # use_sim_time. The declaration was removed; the derivation never existed.
    # scope and publisher are pepper_sensor_tf's own arguments, reached from the
    # command line by inheritance -- see its header and bag_test/README.md.)
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

    # REP-105 says map -> odom is the loop-closure / localization correction.
    # This launch is odometry ONLY -- nothing corrects anything -- so that edge
    # is identity by definition. Publishing it costs nothing and makes 'map' a
    # usable RViz fixed frame here, so the same rviz config works whether or not
    # PGO/AMCL is running.
    #
    # MUST be false when something else owns map -> odom (pgo_map_odom_bridge,
    # AMCL, or lio_localization's transform_fusion) -- two publishers would give
    # odom two parents and split the tree.
    map_identity = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_identity',
        arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('publish_map_identity')),
    )

    # The odom -> base_footprint bridge. It used to be started inside
    # FAST_LIO/launch/mapping.launch.py, which meant Pepper glue lived in a
    # launch file shared with every other FAST-LIO sensor config, duplicated
    # point_lio's identical copy, and pinned the script inside fast_lio.
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
