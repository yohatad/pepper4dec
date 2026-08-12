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
# Unlike fast_lio's mapping.launch.py, mapping_l2lidar_node.launch.py has no
# config_file argument to forward -- its config is hardcoded to
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
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    point_lio_share = get_package_share_directory('point_lio')
    pkg_share = get_package_share_directory('pepper_slam')

    rviz = LaunchConfiguration('rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    flatten_base_frame = LaunchConfiguration('flatten_base_frame')
    bridge_level_frame = LaunchConfiguration('bridge_level_frame')

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
        launch_arguments={'use_sim_time': use_sim_time}.items())

    point_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(point_lio_share, 'launch', 'mapping_l2lidar_node.launch.py')),
        launch_arguments={
            'rviz': rviz,
            'use_sim_time': use_sim_time,
            'flatten_base_frame': flatten_base_frame,
            'bridge_level_frame': bridge_level_frame,
        }.items())

    ld = LaunchDescription()
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_flatten_base_frame_cmd)
    ld.add_action(declare_bridge_level_frame_cmd)
    ld.add_action(sensor_tf_launch)
    ld.add_action(point_lio_launch)
    return ld
