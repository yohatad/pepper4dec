"""pepper_nav2_kissicp.launch.py

Nav2 bringup for Pepper: AMCL + map_server, over KISS-ICP odometry.

A twin of pepper_nav2_amcl.launch.py with only the odometry source swapped
(KISS-ICP instead of FAST-LIO + lio_odom_bridge), so comparing the two isolates
the odometry. KISS-ICP is odometry only -- it has no prior-map localization --
so AMCL provides map -> odom, exactly as in the FAST-LIO baseline.

kiss_icp_node publishes odom -> base_footprint itself. With base_frame set to
base_footprint it registers in its own world frame and reports the BASE pose,
anchored at the base's start pose: odom is level and floor-referenced from the
first scan, so there is no tilted lio_init and no bridge. Its motion prior
uses /pepper_odom (wheel odometry) and falls back to constant velocity when
that is absent. Unlike the LIO stacks there is no IMU and no lio_odom_guard
gating the pose.

Launch files included:
    pepper_slam/pepper_sensor_tf.launch.py — the sensor rig TF.

Nodes started:
    kiss_icp/kiss_icp_node — odometry, odom -> base_footprint.
    pointcloud_to_laserscan/pointcloud_to_laserscan_node — /points -> /scan.
    nav2_map_server/map_server, nav2_amcl/amcl,
    nav2_controller/controller_server, nav2_planner/planner_server,
    nav2_behaviors/behavior_server, nav2_bt_navigator/bt_navigator.
    pepper_slam/cloud_range_filter x2 (points_safety_filter,
    points_costmap_filter).
    nav2_collision_monitor/collision_monitor.
    pepper_navigation/sensor_guard — the last gate before cmd_vel: stops
        translation while /points_safety is silent.
    nav2_costmap_2d/nav2_costmap_2d_markers x2 — voxel visualization.
    pepper_navigation/localization_recovery.
    pepper_navigation/tf_nav_relay.
    rviz2/rviz2 — only when rviz is true.
    nav2_lifecycle_manager/lifecycle_manager (node:
    lifecycle_manager_navigation).

Launch arguments:
    use_sim_time (default: "false")
    map (default: <share>/map/pepper_map_lc.yaml)
        2D occupancy grid served as /map. MUST exist.
    kiss_config (default: <kiss_icp share>/config/pepper_l2.yaml)
        KISS-ICP parameters. Its frame settings are overridden below so the
        TF contract cannot drift with the config.
    scan_min_height / scan_max_height (defaults: "0.20" / "1.50")
        The /points slice flattened into /scan for amcl.
    rviz_config (default: <share>/rviz/nav2_amcl.rviz)
    rviz (default: "true")

Configuration:
    config/nav2_params_kissicp.yaml, config/pepper_navigation_configuration.yaml,
    pepper_slam's pepper_slam_configuration.yaml, map/pepper_map_lc.yaml, and
    kiss_icp's pepper_l2.yaml. The nodes shared by every profile are defined in
    pepper_navigation/nav2_nodes.py.

Prerequisites:
    sudo apt install ros-humble-pointcloud-to-laserscan

Usage (real robot):
    ros2 launch l2lidar_node l2lidar.launch.py
    ros2 launch pepper_navigation pepper_nav2_kissicp.launch.py map:=<path>

    Set the initial pose in RViz (2D Pose Estimate) — amcl starts
    unlocalized.

Usage (bag replay):
    ros2 launch pepper_navigation pepper_nav2_kissicp.launch.py use_sim_time:=true
    ros2 bag play <bag> --clock --topics /points /pepper_odom

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 29, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml
from pepper_navigation import nav2_nodes


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_navigation')
    sensor_tf_launch_dir = os.path.join(
        get_package_share_directory('pepper_slam'), 'launch')

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml = LaunchConfiguration('map')
    scan_min_height = LaunchConfiguration('scan_min_height')
    scan_max_height = LaunchConfiguration('scan_max_height')
    rviz = LaunchConfiguration('rviz')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use bag/simulation clock instead of wall time.')
    declare_map_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(pkg_share, 'map', 'pepper_map_lc.yaml'),
        description='2D occupancy grid served as /map, for amcl and the global '
                    'costmap static layer. MUST exist: map_server fails to '
                    'configure otherwise and the lifecycle manager aborts the '
                    'whole nav2 bringup.')
    # Reach for these first if amcl won't converge (band is in base_footprint,
    # floor at z=0): too low and floor returns swamp wall hits, too high and
    # unmapped furniture/people mismatch every beam.
    declare_scan_min_height_cmd = DeclareLaunchArgument(
        'scan_min_height', default_value='0.20',
        description='Bottom of the /points slice flattened into /scan.')
    declare_scan_max_height_cmd = DeclareLaunchArgument(
        'scan_max_height', default_value='1.50',
        description='Top of the /points slice flattened into /scan.')
    declare_rviz_config_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_share, 'rviz', 'nav2_amcl.rviz'),
        description='RViz config. nav2_amcl_voxel.rviz gives the 3D voxel view '
                    '(needs z_voxels <= 16 in the nav2 params).')
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz2 pre-configured for this stack.')
    declare_kiss_config_cmd = DeclareLaunchArgument(
        'kiss_config',
        default_value=os.path.join(
            get_package_share_directory('kiss_icp'), 'config', 'pepper_l2.yaml'),
        description='KISS-ICP parameter file. Frame settings are overridden '
                    'in this launch file regardless.')

    # GroupAction scopes each include's launch_arguments to itself -- plain
    # IncludeLaunchDescription would leak them into this file's own context
    # (e.g. rviz:=false below would silently suppress rviz_node).

    # base_footprint -> l2lidar_frame (+ cams). Not included by
    # mapping.launch.py; pointcloud_to_laserscan needs it.
    sensor_tf = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensor_tf_launch_dir, 'pepper_sensor_tf.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),
    ])

    # KISS-ICP odometry: odom -> base_footprint. The frame keys come AFTER
    # the config file so they win over it: the TF contract (odom level,
    # base_footprint the child, no inversion) is this stack's, not the
    # tuning file's.
    kiss_icp = Node(
        package='kiss_icp',
        executable='kiss_icp_node',
        name='kiss_icp_node',
        output='screen',
        remappings=[('pointcloud_topic', '/points'),
                    # nav2_params_kissicp.yaml's odom_topic.
                    ('kiss/odometry', '/odom_lio')],
        parameters=[
            LaunchConfiguration('kiss_config'),
            {
                'use_sim_time': use_sim_time,
                'base_frame': 'base_footprint',
                'lidar_odom_frame': 'odom',
                'publish_odom_tf': True,
                'invert_odom_tf': False,
                'publish_debug_clouds': False,
            },
        ],
    )

    # /points (3D, 360 deg) -> /scan (2D LaserScan) for amcl only.
    pointcloud_to_laserscan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        remappings=[('cloud_in', '/points'), ('scan', '/scan')],
        parameters=[{
            'use_sim_time': use_sim_time,
            'target_frame': 'base_footprint',
            'transform_tolerance': 0.05,
            # value_type=float: a bare LaunchConfiguration arrives as a string.
            'min_height': ParameterValue(scan_min_height, value_type=float),
            'max_height': ParameterValue(scan_max_height, value_type=float),
            'angle_min': -3.141592653589793,   # full 360 deg, like the L2
            'angle_max': 3.141592653589793,
            'angle_increment': 0.008726646259971648,   # 0.5 deg -> 720 beams
            'scan_time': 0.1,
            # Matches amcl's laser_min_range / costmaps' obstacle_min_range --
            # below it the low-mounted L2 sees Pepper itself.
            'range_min': 0.8,
            'range_max': 20.0,
            'use_inf': True,
            'inf_epsilon': 1.0,
            'concurrency_level': 2,
        }],
    )

    nav2_params_file = os.path.join(
        pkg_share, 'config', 'nav2_params_kissicp.yaml')
    configured_params = RewrittenYaml(
        source_file=nav2_params_file,
        root_key='',
        param_rewrites={'use_sim_time': use_sim_time},
        convert_types=True)

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[configured_params],
    )

    rviz_config = LaunchConfiguration('rviz_config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'voxel_markers', default_value='false',
            description='Start the voxel-marker converters for the 3D costmap view '
                        '(nav2_*_voxel.rviz, nav2_amcl.rviz). Off by default: '
                        'nothing in navigation uses them and they cost CPU.'),
        declare_use_sim_time_cmd,
        declare_map_cmd,
        declare_kiss_config_cmd,
        declare_scan_min_height_cmd,
        declare_scan_max_height_cmd,
        declare_rviz_cmd,
        declare_rviz_config_cmd,
        sensor_tf,
        kiss_icp,
        pointcloud_to_laserscan,
        nav2_nodes.map_server(use_sim_time, map_yaml),
        amcl,
        *nav2_nodes.nav2_servers(configured_params),
        nav2_nodes.points_safety_filter(use_sim_time),
        nav2_nodes.points_costmap_filter(use_sim_time),
        nav2_nodes.collision_monitor(configured_params),
        nav2_nodes.sensor_guard(use_sim_time),
        *nav2_nodes.voxel_markers(
            use_sim_time, 'warn', IfCondition(LaunchConfiguration('voxel_markers'))),
        nav2_nodes.rviz(use_sim_time, rviz_config, IfCondition(rviz)),
        # map_server first so /map is up before amcl and the global costmap
        # activate; amcl before the costmaps so map -> odom exists.
        nav2_nodes.lifecycle_manager(use_sim_time, [
            'map_server',
            'amcl',
            'controller_server',
            'planner_server',
            'behavior_server',
            'bt_navigator',
            'collision_monitor',
        ]),
        # Here it forwards to amcl's /reinitialize_global_localization.
        nav2_nodes.localization_recovery(use_sim_time, 'amcl'),
        nav2_nodes.tf_nav_relay(use_sim_time),
    ])
