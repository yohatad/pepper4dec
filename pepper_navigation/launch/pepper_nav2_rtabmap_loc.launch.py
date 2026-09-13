r"""pepper_nav2_rtabmap_loc.launch.py

Nav2 bringup for Pepper on FAST-LIO + RTAB-Map (localization mode).

Localizes against the saved rtabmap_fastlio_refined.db instead of AMCL and a
static map_server: RTAB-Map runs with Mem/IncrementalMemory=false, reusing the
odometry/appearance/ICP pipeline tuned for mapping, and publishes /map itself.
No pointcloud_to_laserscan — the costmaps take /points directly.

Launch files included:
    fast_lio/mapping.launch.py — FAST-LIO odometry.
    pepper_slam/lio_odom_bridge.launch.py — the gravity-leveled odom frame.
    pepper_slam/rtabmap_base.launch.py — RTAB-Map in localization mode.

Nodes started:
    nav2_controller/controller_server, nav2_planner/planner_server,
    nav2_behaviors/behavior_server, nav2_bt_navigator/bt_navigator.
    pepper_slam/cloud_range_filter.py (node: points_safety_filter).
    nav2_collision_monitor/collision_monitor.
    pepper_navigation/localization_recovery.py.
    nav2_lifecycle_manager/lifecycle_manager x2 — one for navigation
    (lifecycle_manager_navigation) and a separate one for the collision
    monitor (lifecycle_manager_collision_monitor).

Launch arguments:
    use_sim_time (default: "false")
        Use bag/simulation clock instead of wall time.
    database_path (default: "~/.ros/rtabmap_fastlio_refined.db")
        Map database to localize against.

Configuration:
    config/nav2_params_rtabmap_loc.yaml and FAST-LIO's l2.yaml.

Frames:
    FAST-LIO odom -> lio_odom_bridge's gravity-leveled odom -> RTAB-Map map.
    See nav2_params_rtabmap_loc.yaml for why local_costmap uses odom (not
    pepper_odom) as its global_frame.

Usage (real robot):
    ros2 launch pepper_navigation pepper_nav2_rtabmap_loc.launch.py

Usage (bag replay, to sanity-check localization/costmaps without driving):
    ros2 launch pepper_navigation pepper_nav2_rtabmap_loc.launch.py use_sim_time:=true
    ros2 bag play <bag> --clock --topics /points /imu/data /tf_static \
        /camera/color/image_raw /camera/color/camera_info

    Nav2 will localize and build costmaps, but a bag can't react to cmd_vel —
    driving to a goal needs the real robot or a simulator.

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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_navigation')
    slam_launch_dir = os.path.join(
        get_package_share_directory('pepper_slam'), 'launch')
    fast_lio_launch_dir = os.path.join(
        get_package_share_directory('fast_lio'), 'launch')

    use_sim_time = LaunchConfiguration('use_sim_time')
    database_path = LaunchConfiguration('database_path')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use bag/simulation clock instead of wall time.')
    declare_database_path_cmd = DeclareLaunchArgument(
        'database_path', default_value='~/.ros/rtabmap_fastlio_refined.db',
        description="Map database to localize against (today's best "
                    "validated run -- see project_l2_slam_stack memory).")

    fast_lio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_launch_dir, 'mapping.launch.py')),
        launch_arguments={
            'config_file': 'l2.yaml',
            'rviz': 'false',
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # odom -> base_footprint (FAST_LIO's own launch file no longer starts this).
    lio_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('pepper_slam'),
                         'launch', 'lio_odom_bridge.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'config_file': 'l2.yaml',
        }.items(),
    )

    rtabmap_localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_launch_dir, 'rtabmap_base.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'frame_id': 'base_footprint',
            'odom_frame_id': 'odom',

            'visual_odometry': 'false',
            'icp_odometry': 'false',

            'depth': 'false',
            'subscribe_rgb': 'true',
            'rgb_topic': '/camera/color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'rgbd_sync': 'false',
            'subscribe_scan': 'false',
            'subscribe_scan_cloud': 'true',
            'scan_cloud_topic': '/points',

            'approx_sync': 'true',
            'qos': '2',

            'localization': 'true',
            'database_path': database_path,
            # Same ICP/grid tuning validated for mapping (see
            # rtabmap_fastlio_bag.launch.py); no --delete_db_on_start
            # (would erase the map) or NeighborLinkRefining/Proximity params
            # (govern new loop-closure links, moot with IncrementalMemory=false).
            'rtabmap_args': '--Reg/Strategy 1 '
                            '--Icp/VoxelSize 0.15 --Icp/PointToPlaneK 20 '
                            '--Icp/MaxCorrespondenceDistance 0.5 '
                            '--Icp/CorrespondenceRatio 0.2 '
                            '--Grid/Sensor 0 --Grid/CellSize 0.05 '
                            '--Grid/RangeMax 8.0 '
                            '--Grid/MaxGroundHeight 0.10 '
                            '--Grid/MaxObstacleHeight 1.7 '
                            '--Grid/RayTracing true '
                            '--Grid/NoiseFilteringRadius 0.15 '
                            '--Grid/NoiseFilteringMinNeighbors 3 '
                            '--Grid/3D true',
            'rtabmap_viz': 'false',
            'rviz': 'false',
        }.items(),
    )

    nav2_params_file = os.path.join(
        pkg_share, 'config', 'nav2_params_rtabmap_loc.yaml')
    configured_params = RewrittenYaml(
        source_file=nav2_params_file,
        root_key='',
        param_rewrites={'use_sim_time': use_sim_time},
        convert_types=True)

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[configured_params],
        # Route velocity through the collision monitor: controller -> cmd_vel_raw
        # -> collision_monitor -> cmd_vel (what Pepper drives on).
        remappings=[('cmd_vel', 'cmd_vel_raw')],
    )
    planner_server = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[configured_params],
    )
    behavior_server = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[configured_params],
        # Recovery motions (spin/backup) also go through the collision monitor.
        remappings=[('cmd_vel', 'cmd_vel_raw')],
    )

    # Strips Pepper's own body (< 0.8 m) so the collision monitor doesn't
    # freeze on self-hits.
    points_safety_filter = Node(
        package='pepper_slam',
        executable='cloud_range_filter.py',
        name='points_safety_filter',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_topic': '/points',
            'output_topic': '/points_safety',
            'min_range': 0.8,
            'ror_min_neighbors': 0,   # ROR off (see cloud_range_filter notes)
        }],
    )

    collision_monitor = Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        parameters=[configured_params],
    )
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[configured_params],
    )
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'bond_timeout': 4.0,
            'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'collision_monitor',
            ],
        }],
    )
    # Separate manager so a planner failure and a collision monitor failure
    # don't take each other's bond down.
    lifecycle_manager_collision_monitor = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_collision_monitor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'bond_timeout': 4.0,
            'node_names': ['collision_monitor'],
        }],
    )

    # One /localization_recover entry point, identical across all three nav
    # profiles. rtabmap has no forced re-search to forward to (it relocalizes
    # from loop closure on its own), so here the service exists only to say so
    # and point at /initialpose -- better than the operator discovering that
    # by trying whatever worked on the other two profiles.
    localization_recovery = Node(
        package='pepper_navigation',
        executable='localization_recovery.py',
        name='localization_recovery',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'backend': 'rtabmap'}],
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_database_path_cmd,
        fast_lio,
        lio_bridge,
        rtabmap_localization,
        controller_server,
        planner_server,
        behavior_server,
        bt_navigator,
        points_safety_filter,
        collision_monitor,
        lifecycle_manager,
        lifecycle_manager_collision_monitor,
        localization_recovery,
    ])
