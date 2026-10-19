"""collision_monitor_bag.launch.py

Test 1b of the safety-chain plan: the real nav2 collision monitor on a
recorded bag, scored scan by scan against plain polygon geometry.

Nodes started:
    pepper_slam/cloud_range_filter (node: points_safety_filter) -- /points ->
        /points_safety, with the same settings as pepper_nav2_fastloc.launch.py.
    nav2_collision_monitor/collision_monitor -- the node under test.
    nav2_lifecycle_manager/lifecycle_manager (node: lifecycle_manager_cm).
    pepper_navigation/cm_bag_judge.py -- drives the monitor and scores it.
    pepper_navigation/tf_nav_relay -- only when localize:=true.

Launch files included:
    pepper_slam/pepper_sensor_tf.launch.py -- the sensor rig TF.
    fast_lio/localization_l2.launch.py -- only when localize:=true.

SAFETY: the monitor's velocity topics are renamed to /cm_bag/cmd_vel_in and
/cm_bag/cmd_vel_out, so nothing here ever publishes /cmd_vel -- a live
naoqi_driver on the same domain cannot be driven by the test.

Usage:
    ros2 launch pepper_navigation collision_monitor_bag.launch.py \
        out_prefix:=/tmp/cm_bsc_false
    ros2 bag play <bag> --clock --topics /points /tf_static

    base_shift_correction:=true also needs map -> base_footprint: add
    localize:=true for a bag without a pose (FAST-LIO locks after ~25 s; the
    monitor ignores every cloud until then), or play /tf_nav for one with it.

    Ctrl+C after the bag ends: the judge prints the summary and writes
    <out_prefix>_scans.csv and <out_prefix>_summary.json.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    nav = get_package_share_directory('pepper_navigation')
    slam = get_package_share_directory('pepper_slam')
    lio = get_package_share_directory('fast_lio')
    sim = {'use_sim_time': True}
    params = LaunchConfiguration('params_file')
    localize = LaunchConfiguration('localize')
    tf_nav = [('/tf', '/tf_nav')]
    cmd_in, cmd_out = '/cm_bag/cmd_vel_in', '/cm_bag/cmd_vel_out'

    declare = [
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(nav, 'config', 'nav2_params_fastloc.yaml'),
            description='Nav2 params; the monitor and the judge both read its '
                        'collision_monitor section.'),
        DeclareLaunchArgument('base_shift_correction', default_value='false'),
        DeclareLaunchArgument(
            'localize', default_value='false',
            description='Run FAST-LIO for map -> base_footprint. Only '
                        'base_shift_correction:=true needs a pose.'),
        DeclareLaunchArgument(
            'sensor_tf', default_value='urdf', choices=['urdf', 'yaml', 'none'],
            description="'none' for a bag whose /tf_static already holds the rig."),
        DeclareLaunchArgument('out_prefix', default_value='/tmp/cm_bag'),
        DeclareLaunchArgument('map_dir', default_value=os.path.join(nav, 'pcd')),
        DeclareLaunchArgument('map_pose_file', default_value='sc_pose_20260823.json'),
        DeclareLaunchArgument(
            'map_scan_dir', default_value=os.path.join(nav, 'pcd', 'sc_pcd_20260823')),
        DeclareLaunchArgument('config_file', default_value='l2_rsimu.yaml'),
    ]

    sensor_tf = GroupAction([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': 'true',
                          'publisher': LaunchConfiguration('sensor_tf'),
                          'scope': 'all'}.items())])

    fastloc = GroupAction([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lio, 'launch', 'localization_l2.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'config_file': LaunchConfiguration('config_file'),
            'map_dir': LaunchConfiguration('map_dir'),
            'map_pose_file': LaunchConfiguration('map_pose_file'),
            'map_scan_dir': LaunchConfiguration('map_scan_dir'),
            'rviz': 'false'}.items())], condition=IfCondition(localize))

    nodes = [
        Node(package='pepper_navigation', executable='tf_nav_relay',
             name='tf_nav_relay', output='screen', parameters=[sim],
             condition=IfCondition(localize)),
        # Keep in step with points_safety_filter in pepper_nav2_fastloc.launch.py.
        Node(package='pepper_slam', executable='cloud_range_filter',
             name='points_safety_filter', output='screen',
             parameters=[{**sim,
                          'input_topic': '/points',
                          'output_topic': '/points_safety',
                          'min_range': 0.22,
                          'self_cut_range': 0.40,
                          'self_cut_angle': 25.0,
                          'ror_min_neighbors': 0}]),
        Node(package='nav2_collision_monitor', executable='collision_monitor',
             name='collision_monitor', output='screen',
             parameters=[params, {
                 **sim,
                 'base_shift_correction': ParameterValue(
                     LaunchConfiguration('base_shift_correction'), value_type=bool),
                 'cmd_vel_in_topic': cmd_in,
                 'cmd_vel_out_topic': cmd_out,
                 # Never go quiet during a long STOP: the judge reads every
                 # verdict off the output, and silence would be ambiguous.
                 'stop_pub_timeout': 1.0e6}],
             remappings=tf_nav),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_cm', output='screen',
             parameters=[{**sim, 'autostart': True, 'bond_timeout': 4.0,
                          'node_names': ['collision_monitor']}]),
        Node(package='pepper_navigation', executable='cm_bag_judge.py',
             name='cm_bag_judge', output='screen',
             parameters=[{**sim, 'params_file': params,
                          'cmd_in_topic': cmd_in, 'cmd_out_topic': cmd_out,
                          'out_prefix': LaunchConfiguration('out_prefix')}],
             remappings=tf_nav),
    ]

    return LaunchDescription(declare + [sensor_tf, fastloc] + nodes)
