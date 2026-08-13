# Plain Point-LIO ODOMETRY on the Pepper L2 rig, with its required static TF.
#
# RENAMED 2026-08-10 from pointlio_mapping.launch.py -- see the header of
# fastlio_odometry.launch.py for why. Short version: "mapping" upstream means
# scan-to-map REGISTRATION, not SLAM. No loop closure, no pose graph, so the
# accumulated cloud drifts and duplicates on revisit. Use
# fastlio_lc_pgo pointlio_lc_l2.launch.py for a loop-corrected map.
#
# Same defect as plain FAST-LIO's mapping.launch.py (see fastlio_odometry.launch.py
# in this directory): point_lio's mapping_l2lidar_node.launch.py runs
# lio_map_odom_bridge.py itself, which needs the static base_footprint ->
# l2lidar_frame -> l2lidar_frame_imu chain -- but that launch file never includes
# pepper_sensor_tf.launch.py, so launched alone it silently hangs waiting for
# a transform that will never appear. This wraps both together.
#
# STALE UNTIL 2026-08-12: mapping_l2lidar_node.launch.py DOES take a
# config_file argument now, and this file forwards it. It used to be
# config/l2lidar_node.yaml. use_sim_time, bridge_level_frame and
# flatten_base_frame ARE forwarded (see that file).
#
# flatten_base_frame defaults to true HERE (unlike mapping_l2lidar_node.launch.py's
# own default of false): this file is Pepper-specific and Pepper is confirmed
# to only ever run on flat floor. Pass flatten_base_frame:=false to see
# Point-LIO's own (drifting) z/roll/pitch instead.
#
# Usage:
#   ros2 launch pepper_slam pointlio_odometry.launch.py
#   ros2 bag play <bag> --clock --topics /points /imu/data
#   (do NOT also replay /tf -- see pepper_sensor_tf.launch.py's header)

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _resolve_scope(context, *args, **kwargs):
    """sensor_tf scope follows use_sim_time unless set explicitly."""
    from launch.actions import SetLaunchConfiguration
    explicit = LaunchConfiguration('sensor_tf_scope').perform(context)
    if explicit:
        return [SetLaunchConfiguration('resolved_scope', explicit)]
    sim = LaunchConfiguration('use_sim_time').perform(context).lower()
    return [SetLaunchConfiguration(
        'resolved_scope', 'all' if sim in ('true', '1', 'yes') else 'mount')]


def generate_launch_description():
    point_lio_share = get_package_share_directory('point_lio')
    pkg_share = get_package_share_directory('pepper_slam')

    rviz = LaunchConfiguration('rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    flatten_base_frame = LaunchConfiguration('flatten_base_frame')
    bridge_level_frame = LaunchConfiguration('bridge_level_frame')

    # 2026-08-12: the RealSense IMU is the permanent choice for this rig, so
    # l2lidar_rsimu.yaml is the default. The matching lidar_imu_frame is DERIVED
    # inside mapping_l2lidar_node.launch.py, so switching IMU is one argument.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2lidar_rsimu.yaml',
        description='Point-LIO config under point_lio/config. l2lidar_rsimu.yaml '
                    'uses the RealSense IMU (default); l2lidar_node.yaml uses '
                    'the L2 s own -- see utils/L2_IMU/REPORT.md.')
    # Scope follows use_sim_time: replay has no RealSense driver, so the camera
    # TF edges must come from calibration; on the robot the driver owns them and
    # sensor_tf.yaml warns the two sources CAN DIFFER.
    declare_scope_cmd = DeclareLaunchArgument(
        'sensor_tf_scope', default_value='', choices=['', 'mount', 'all'],
        description="Empty (default) derives it from use_sim_time: 'all' for "
                    "bag replay, 'mount' on the robot.")
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='true for bag replay (--clock); false on the robot. '
                    'Reaches pepper_sensor_tf, Point-LIO and the odom bridge.')
    declare_flatten_base_frame_cmd = DeclareLaunchArgument(
        'flatten_base_frame', default_value='true',
        description='Zero the leveled z/roll/pitch of odom -> base_footprint '
                    'every cycle (keep x, y, yaw). Defaults true here: Pepper '
                    'is confirmed flat-floor-only.'
    )
    declare_bridge_level_frame_cmd = DeclareLaunchArgument(
        'bridge_level_frame', default_value='true',
        description='Have lio_map_odom_bridge publish the static odom -> '
                    'odom leveling frame. Set false when a higher layer owns '
                    'odom (e.g. PGO publishing map -> odom).'
    )

    sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time,
                          'scope': LaunchConfiguration('resolved_scope')}.items())

    point_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(point_lio_share, 'launch', 'mapping_l2lidar_node.launch.py')),
        launch_arguments={
            'rviz': rviz,
            'config_file': LaunchConfiguration('config_file'),
            'use_sim_time': use_sim_time,
            'flatten_base_frame': flatten_base_frame,
            'bridge_level_frame': bridge_level_frame,
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_scope_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_flatten_base_frame_cmd)
    ld.add_action(declare_bridge_level_frame_cmd)
    # AFTER every declare: the resolver reads sensor_tf_scope and use_sim_time.
    ld.add_action(OpaqueFunction(function=_resolve_scope))
    ld.add_action(sensor_tf_launch)
    ld.add_action(point_lio_launch)
    return ld
