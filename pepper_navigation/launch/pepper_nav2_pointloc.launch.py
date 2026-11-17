r"""pepper_nav2_pointloc.launch.py

Nav2 bringup for Pepper on pointlio_localization.

Point-LIO with the prior map loaded into the filter, so the map constrains the
estimate at scan rate from inside it. Owns map -> base_footprint;
nav2_map_server serves the matching 2D grid as /map for the global costmap.

The Point-LIO twin of pepper_nav2_fastloc.launch.py. Everything downstream of
localization — costmaps, MPPI, collision monitor, safety chain — is identical
(nav2_params_pointloc.yaml is byte-identical to the fastloc one apart from its
header), so a behavioural difference between the two profiles is a BACKEND
difference and nothing else.

NOT YET VALIDATED against a bag: pointlio_localization is newer than the
fastloc path. Prefer pepper_nav2_fastloc.launch.py until this one has been run
on real data.

Launch files included:
    pepper_slam/pepper_sensor_tf.launch.py — the sensor rig TF.
    point_lio/localization_l2.launch.py — the localizer itself.

Nodes started:
    Same set as pepper_nav2_fastloc.launch.py: the Nav2 pipeline
    (map_server, controller_server, planner_server, behavior_server,
    bt_navigator), the points_safety_filter and collision monitor, the two
    voxel marker converters, wait_for_map_then_start, localization_recovery,
    localization_watchdog, RViz when enabled, and
    lifecycle_manager_navigation.

Launch arguments:
    use_sim_time (default: "false")
        Use bag/simulation clock instead of wall time.
    sensor_tf (default: "urdf")
        Publish the sensor rig; 'none' if the bag already provides
        /tf_static.
    sensor_tf_scope (default: "all")
        'all' publishes the RealSense internal extrinsics too; use 'mount'
        live.
    map_dir (default: <share>/pcd)
        Directory holding the ScanContext pose file.
    map_pose_file (default: "sc_pose_20260823.json")
        Per-keyframe poses. MUST come from the same mapping run as map.
    map_scan_dir (default: <share>/pcd/sc_pcd_20260823)
        Per-keyframe clouds indexed BY NUMBER from map_pose_file.
    map (default: <share>/map/pepper_map_lc.yaml)
        2D occupancy grid served as /map for the global costmap static layer.
    config_file (default: "l2lidar_rsimu.yaml")
        Point-LIO config.
    body_frame (default: "camera_imu_optical_frame")
        Body frame matching config_file.
    rviz_config (default: <share>/rviz/nav2_fastloc.rviz)
    rviz (default: "true")
        Open RViz2 pre-configured for this nav stack.
    watchdog (default: "true")
        Cancel navigation goals while the localizer reports itself lost.

Configuration:
    config/nav2_params_pointloc.yaml, config/pepper_navigation_configuration.yaml,
    pepper_slam's pepper_slam_configuration.yaml, map/pepper_map_lc.yaml, and
    the Point-LIO config selected by config_file. The nodes shared by every
    profile are defined in pepper_navigation/nav2_nodes.py.

Frames:
    map --(pointlio_localization)--> base_footprint
        --(pepper_sensor_tf / bag tf_static)--> l2lidar_frame_imu, cams

    No odom frame: after handover the filter state IS the map pose, so the
    local costmap rolls in 'map' (see local_costmap global_frame in
    config/nav2_params_pointloc.yaml). Don't add lio_odom_bridge here —
    base_footprint would get two parents.

Usage (real robot):
    ros2 launch pepper_navigation pepper_nav2_pointloc.launch.py

    No initial pose needed: ScanContext finds it. Call /relocalize if lost.

Usage (bag replay):
    ros2 launch pepper_navigation pepper_nav2_pointloc.launch.py use_sim_time:=true
    ros2 bag play <bag> --clock \
        --read-ahead-queue-size 2000

To use a DIFFERENT mapping run, change map, map_pose_file and map_scan_dir
together, or the localizer and the costmap disagree about where the world is.

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
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml
from pepper_navigation import nav2_nodes


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_navigation')

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml = LaunchConfiguration('map')
    rviz = LaunchConfiguration('rviz')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use bag/simulation clock instead of wall time.')
    # Publishing the rig here makes playback position irrelevant (the bag's own
    # /tf_static is skipped by --start-offset). Set sensor_tf:=none if playing
    # from the START, to avoid both publishing the same edges.
    declare_sensor_tf_cmd = DeclareLaunchArgument(
        'sensor_tf', default_value='urdf', choices=['urdf', 'yaml', 'none'],
        description="Publish the sensor rig. 'urdf' also gives RViz a "
                    "RobotModel; 'none' if the bag already provides /tf_static.")
    declare_sensor_tf_scope_cmd = DeclareLaunchArgument(
        'sensor_tf_scope', default_value='all', choices=['mount', 'all'],
        description="'all' publishes the RealSense internal extrinsics too, "
                    "needed when the bag's /tf_static is unavailable. Use "
                    "'mount' live, where the RealSense driver publishes its own.")

    declare_map_dir_cmd = DeclareLaunchArgument(
        'map_dir', default_value=os.path.join(pkg_share, 'pcd'),
        description='Directory holding the ScanContext pose file.')
    declare_map_pose_file_cmd = DeclareLaunchArgument(
        'map_pose_file', default_value='sc_pose_20260823.json',
        description='Per-keyframe poses for pointlio_localization. MUST come '
                    'from the same mapping run as map.')
    declare_map_scan_dir_cmd = DeclareLaunchArgument(
        'map_scan_dir', default_value=os.path.join(pkg_share, 'pcd', 'sc_pcd_20260823'),
        description='Per-keyframe clouds indexed BY NUMBER from map_pose_file.')
    declare_map_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(pkg_share, 'map', 'pepper_map_lc.yaml'),
        description='2D occupancy grid served as /map for the global costmap '
                    'static layer. MUST be from the SAME mapping run as '
                    'map_pose_file and map_scan_dir, or the localizer and the '
                    'costmap disagree about where the world is. '
                    'map/pepper_map_lc.pgm, pcd/sc_pose_20260823.json and '
                    'pcd/sc_pcd_20260823/ are ONE set. MUST exist, or the '
                    'lifecycle manager aborts the whole bringup.')

    # The map was built with the RealSense IMU, so localize with it too: the
    # lidar/IMU extrinsic enters the initial pose composition, and body_frame
    # must match whichever config is used or the map -> base edge is composed
    # through the wrong mount.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2lidar_rsimu.yaml',
        description='Point-LIO config: l2lidar_rsimu.yaml (RealSense IMU, '
                    'matches the prior map) or l2lidar_node.yaml (the L2 s own).')
    declare_body_frame_cmd = DeclareLaunchArgument(
        'body_frame', default_value='camera_imu_optical_frame',
        description='Frame the filter estimates, matching config_file. '
                    'camera_imu_optical_frame for l2lidar_rsimu.yaml, '
                    'l2lidar_frame_imu for l2lidar_node.yaml.')
    declare_rviz_config_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_share, 'rviz', 'nav2_fastloc.rviz'),
        description='RViz config. Shared with the fastloc profile on purpose: '
                    'the topics it shows (/prior_map, /cloud_registered, '
                    'costmaps, plans) are the same for both backends, so a '
                    'second near-identical copy would only be one more file to '
                    'keep in sync. nav2_fastloc_voxel.rviz for the 3D voxel view.')
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz2 pre-configured for this nav stack (map, costmaps, '
                    'plans, safety zones, 2D Pose Estimate / Nav2 Goal tools).')
    declare_watchdog_cmd = DeclareLaunchArgument(
        'watchdog', default_value='true',
        description='Cancel navigation goals while pointlio_localization reports '
                    'itself lost. Set false to monitor without ever holding nav.')
    declare_odom_frame_cmd = DeclareLaunchArgument(
        'odom_frame', default_value='',
        description='Empty (default): pointlio_localization broadcasts map -> '
                    'base_footprint and the local costmap rolls in map. '
                    '"odom": REP-105 map -> odom -> base_footprint instead, '
                    'with odom -> base_footprint from wheel odometry '
                    '(pepper_slam wheel_odom_tf) and the local costmap in '
                    'odom, so relocalization jumps never reach the controller. '
                    'See pepper_slam/FRAMES.md.')
    odom_frame = LaunchConfiguration('odom_frame')
    odom_on = IfCondition(PythonExpression(["'", odom_frame, "' != ''"]))

    # GroupAction (scoped by default) is REQUIRED here: IncludeLaunchDescription
    # emits its launch_arguments as SetLaunchConfiguration into the CURRENT
    # context, so 'rviz': 'false' would otherwise overwrite this file's own
    # 'rviz' argument and silently suppress rviz_node below.
    sensor_tf = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('pepper_slam'),
                             'launch', 'pepper_sensor_tf.launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'publisher': LaunchConfiguration('sensor_tf'),
                'scope': LaunchConfiguration('sensor_tf_scope'),
            }.items(),
        ),
    ], condition=IfCondition(
        PythonExpression(["'", LaunchConfiguration('sensor_tf'), "' != 'none'"])))

    # Point-LIO with the prior map inside the filter. It owns
    # map -> base_footprint directly (publish.tf_child_frame), so there is no
    # odom edge and no separate correction node.
    pointloc = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('point_lio'),
                    'launch', 'localization_l2.launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'config_file': LaunchConfiguration('config_file'),
                'body_frame': LaunchConfiguration('body_frame'),
                'map_dir': LaunchConfiguration('map_dir'),
                'map_pose_file': LaunchConfiguration('map_pose_file'),
                'map_scan_dir': LaunchConfiguration('map_scan_dir'),
                'rviz': 'false',
                'odom_frame': odom_frame,
            }.items(),
        ),
    ])

    nav2_params_file = os.path.join(
        pkg_share, 'config', 'nav2_params_pointloc.yaml')
    configured_params = RewrittenYaml(
        source_file=nav2_params_file,
        root_key='',
        param_rewrites={
            'use_sim_time': use_sim_time,
            # Full path, so the global costmap stays in map.
            'local_costmap.local_costmap.ros__parameters.global_frame':
                PythonExpression(["'", odom_frame, "' or 'map'"]),
            # base_shift_correction needs the CONTINUOUS frame, which is
            # map only while the localizer owns base_footprint; in odom mode
            # map is the one that jumps on relocalization.
            'collision_monitor.ros__parameters.odom_frame_id':
                PythonExpression(["'", odom_frame, "' or 'map'"]),
        },
        convert_types=True)

    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_sensor_tf_cmd,
        declare_sensor_tf_scope_cmd,
        declare_map_dir_cmd,
        declare_map_pose_file_cmd,
        declare_map_scan_dir_cmd,
        declare_map_cmd,
        declare_config_file_cmd,
        declare_body_frame_cmd,
        declare_rviz_cmd,
        declare_rviz_config_cmd,
        declare_watchdog_cmd,
        declare_odom_frame_cmd,
        sensor_tf,
        pointloc,
        nav2_nodes.map_server(use_sim_time, map_yaml),
        *nav2_nodes.nav2_servers(configured_params),
        nav2_nodes.points_safety_filter(use_sim_time),
        nav2_nodes.collision_monitor(configured_params),
        *nav2_nodes.voxel_markers(use_sim_time),
        nav2_nodes.rviz(use_sim_time, LaunchConfiguration('rviz_config'), IfCondition(rviz)),
        # autostart OFF, as in the fastloc profile: local_costmap blocks
        # configuring until map exists, which waits on ScanContext's lock, so
        # wait_for_map_then_start calls STARTUP the moment the frame is up.
        # map_server first so /map is up before the global costmap activates.
        nav2_nodes.lifecycle_manager(
            use_sim_time,
            ['map_server', 'controller_server', 'planner_server', 'behavior_server',
             'bt_navigator', 'collision_monitor'],
            autostart=False),
        nav2_nodes.wait_for_map_then_start(use_sim_time),
        # Here it forwards to pointlio_localization's /relocalize.
        nav2_nodes.localization_recovery(use_sim_time, 'pointloc'),
        # Point-LIO's own statuses in place of the configuration file's
        # fastloc names. No bumper_guard in this profile, so no liveness.
        nav2_nodes.localization_watchdog(
            use_sim_time, IfCondition(LaunchConfiguration('watchdog')),
            status_names=['point_lio_localization: pose lock',
                          'point_lio_localization: map -> odom gate']),
        nav2_nodes.wheel_odom_tf(use_sim_time, odom_frame, odom_on),
        nav2_nodes.tf_nav_relay(use_sim_time),
    ])
