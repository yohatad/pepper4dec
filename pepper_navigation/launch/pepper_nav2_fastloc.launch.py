r"""pepper_nav2_fastloc.launch.py

Nav2 bringup for Pepper on fastlio_localization (the default nav_profile).

FAST-LIO with the prior map loaded into the iEKF, so the map constrains the
estimate at scan rate from inside the filter. Owns map -> base_footprint;
nav2_map_server serves the matching 2D grid as /map for the global costmap.
Replaced lio_localization (github.com/yohatad/lio_localization), which applied
the map constraint as a discrete, outside-the-filter map->odom step and
produced jumps; this stack has no such step to jump.

Launch files included:
    pepper_slam/pepper_sensor_tf.launch.py — the sensor rig TF.
    fast_lio/localization_l2.launch.py — the localizer itself.

Nodes started:
    nav2_map_server/map_server, nav2_controller/controller_server,
    nav2_planner/planner_server, nav2_behaviors/behavior_server,
    nav2_bt_navigator/bt_navigator — the Nav2 pipeline.
    pepper_slam/cloud_range_filter (node: points_safety_filter)
        Range-limited cloud feeding the collision monitor.
    nav2_collision_monitor/collision_monitor — the safety chain.
    nav2_costmap_2d/nav2_costmap_2d_markers x2 (local_voxel_markers,
    global_voxel_markers) — voxel visualization for the 3D RViz view. The
        LOCAL grid covers the RealSense forward cone only: the local costmap
        is a hybrid whose 3D layer is fed by the camera alone, with the L2 on
        a separate 2D layer that publishes no voxels (see
        nav2_params_fastloc.yaml). Sides and rear will look empty in the
        local voxel view even when the L2 has marked them.
    pepper_navigation/sensor_guard — the last gate before cmd_vel: stops
        translation while /points_safety is silent.
    pepper_navigation/wait_for_map_then_start — holds the lifecycle
        manager until /map is available.
    pepper_navigation/localization_recovery — /relocalize recovery.
    pepper_navigation/localization_watchdog — cancels goals while the
        localizer reports itself lost (see the watchdog argument).
    pepper_navigation/tf_nav_relay — /tf without the joint tree, for Nav2.
    pepper_slam/wheel_odom_tf — only with odom_frame set.
    rviz2/rviz2 — only when rviz is true.
    nav2_lifecycle_manager/lifecycle_manager (node:
    lifecycle_manager_navigation).

Launch arguments:
    use_sim_time (default: "false")
        Use bag/simulation clock instead of wall time.
    sensor_tf (default: "urdf")
        Publish the sensor rig. 'urdf' also gives RViz a RobotModel; 'none'
        if the bag already provides /tf_static.
    sensor_tf_scope (default: "all")
        'all' publishes the RealSense internal extrinsics too, needed when
        the bag's /tf_static is unavailable. Use 'mount' live, where the
        RealSense driver publishes its own.
    map_dir (default: <share>/pcd)
        Directory holding the ScanContext pose file.
    map_pose_file (default: "sc_pose_20260823.json")
        Per-keyframe poses for fastlio_localization. MUST come from the same
        mapping run as map.
    map_scan_dir (default: <share>/pcd/sc_pcd_20260823)
        Per-keyframe clouds indexed BY NUMBER from map_pose_file.
    map (default: <share>/map/pepper_map_lc.yaml)
        2D occupancy grid served as /map for the global costmap static layer.
        MUST be from the SAME mapping run as map_pose_file and map_scan_dir.
    config_file (default: "l2_rsimu.yaml")
        FAST-LIO config: l2_rsimu.yaml (RealSense IMU, matches the prior map)
        (the RealSense IMU; the only L2 config, since l2.yaml was removed
        for losing heading in slow motion).
    lidar_imu_frame (default: "camera_imu_optical_frame")
        Body frame matching config_file (camera_imu_optical_frame).
    rviz_config (default: <share>/rviz/nav2_fastloc.rviz)
        Pass nav2_fastloc_voxel.rviz for the 3D voxel-map view (needs the
        voxel marker converters this file launches, and z_voxels <= 16 in the
        nav2 params).
    rviz (default: "true")
        Open RViz2 pre-configured for this nav stack.
    watchdog (default: "true")
        Cancel navigation goals while fastlio_localization reports itself
        lost. Set false to monitor without ever holding nav.
    collision_monitor (default: "true")
        TESTING ONLY. false leaves the collision monitor out entirely --
        not started, not in the lifecycle list (a deactivated monitor drops
        its bond and the manager treats it as a crash) -- and feeds the
        controller and behaviours straight into sensor_guard. Nothing then
        stops the robot BEFORE contact but naoqi_driver's 0.35 m proximity
        stop: keep the e-stop in hand.

Configuration:
    config/nav2_params_fastloc.yaml, config/pepper_navigation_configuration.yaml,
    pepper_slam's pepper_slam_configuration.yaml, map/pepper_map_lc.yaml, and
    the FAST-LIO config selected by config_file. The nodes shared by every
    profile are defined in pepper_navigation/nav2_nodes.py.

Frames:
    map --(fastlio_localization)--> base_footprint
        --(pepper_sensor_tf / bag tf_static)--> l2lidar_frame_imu, cams

    No odom frame: after handover the filter state IS the map pose, so the
    local costmap rolls in 'map' (see local_costmap global_frame in
    config/nav2_params_fastloc.yaml). Don't add transform_fusion or
    lio_odom_bridge here — base_footprint would get two parents.

Usage (real robot):
    ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py

    No initial pose needed: ScanContext finds it. Call /relocalize if lost.

Usage (bag replay):
    sensor_tf_scope now defaults to 'mount' (live is the zero-argument case
    this file optimizes for) -- a bag without the RealSense's own internal
    /tf_static (no live camera driver to publish it) needs 'all' explicitly:
    ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py \
        use_sim_time:=true sensor_tf_scope:=all
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
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            LogInfo)
from launch.conditions import IfCondition, UnlessCondition
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
        'sensor_tf_scope', default_value='mount', choices=['mount', 'all'],
        description="'mount' (default) is correct live: the RealSense driver "
                    "publishes its own internal extrinsics, and a second copy "
                    "here would give those edges two publishers with whichever "
                    "lands last silently in force. Pass 'all' only for a bag "
                    "recorded without /tf_static.")

    declare_map_dir_cmd = DeclareLaunchArgument(
        'map_dir', default_value=os.path.join(pkg_share, 'pcd'),
        description='Directory holding the ScanContext pose file.')
    declare_map_pose_file_cmd = DeclareLaunchArgument(
        'map_pose_file', default_value='sc_pose_20260823.json',
        description='Per-keyframe poses for fastlio_localization. MUST come '
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
                    'pcd/sc_pcd_20260823/ are ONE set: same bag '
                    '(slam_20260823_aligned), same PGO run, rotated together by '
                    'utils/align_map.py, so they share a frame by construction '
                    'rather than by coincidence. Every other grid here belongs '
                    'to an older run. Checking the pairing is cheap: every '
                    'keyframe pose should have prior-map points around it -- '
                    'measured at 100%% for this set and 86.6%% for a mismatched '
                    'pair, which is how the wrong-map bug was found. MUST '
                    'exist, or the lifecycle manager aborts the whole bringup.')

    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config: l2_rsimu.yaml (RealSense IMU, matches the '
                    'prior map).')
    declare_lidar_imu_frame_cmd = DeclareLaunchArgument(
        'lidar_imu_frame', default_value='camera_imu_optical_frame',
        description='Body frame matching config_file. camera_imu_optical_frame '
                    'for l2_rsimu.yaml.')
    declare_rviz_config_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_share, 'rviz', 'nav2_fastloc.rviz'),
        description='RViz config. Default is the standard view; pass '
                    'nav2_fastloc_voxel.rviz for the 3D voxel-map view '
                    '(needs the voxel marker converters this file launches, '
                    'and z_voxels <= 16 in the nav2 params).')
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz2 pre-configured for this nav stack (map, costmaps, '
                    'plans, safety zones, 2D Pose Estimate / Nav2 Goal tools).')
    declare_watchdog_cmd = DeclareLaunchArgument(
        'watchdog', default_value='true',
        description='Cancel navigation goals while fastlio_localization reports '
                    'itself lost. Set false to monitor without ever holding nav.')
    declare_collision_monitor_cmd = DeclareLaunchArgument(
        'collision_monitor', default_value='true',
        description='TESTING ONLY: false runs without the collision monitor '
                    '(controller -> sensor_guard -> cmd_vel). Nothing stops '
                    'the robot before contact but the driver\'s proximity stop.')
    cm_on = LaunchConfiguration('collision_monitor')
    is_true = "'.lower() in ('true', '1')"
    declare_odom_frame_cmd = DeclareLaunchArgument(
        'odom_frame', default_value='',
        description='Empty (default): fastlio_localization broadcasts map -> '
                    'base_footprint and the local costmap rolls in map. '
                    '"odom": REP-105 map -> odom -> base_footprint instead, '
                    'with odom -> base_footprint from wheel odometry '
                    '(pepper_slam wheel_odom_tf) and the local costmap in '
                    'odom, so relocalization jumps never reach the controller. '
                    'See pepper_slam/FRAMES.md.')
    odom_frame = LaunchConfiguration('odom_frame')
    odom_on = IfCondition(PythonExpression(["'", odom_frame, "' != ''"]))
    declare_log_level_cmd = DeclareLaunchArgument(
        'log_level', default_value='warn',
        description='rclcpp logger severity for every node this file launches '
                    'directly (not sensor_tf or fastloc, which keep their own '
                    'includes'"'"' defaults). warn by default so bond/costmap/'
                    'lifecycle chatter does not bury real errors; pass '
                    'log_level:=info to see it again.')
    log_level = LaunchConfiguration('log_level')

    # Sensor TF + fastlio_localization (FAST_LIO), which loads the prior map
    # INTO the ikd-Tree the iEKF registers against, so the map constrains the
    # estimate inside the filter at scan rate rather than as a correction
    # applied beside it. See the WHY note in the header.
    #
    # GroupAction (scoped by default) is REQUIRED here: IncludeLaunchDescription
    # emits its launch_arguments as SetLaunchConfiguration into the CURRENT
    # context, so 'rviz': 'false' would otherwise overwrite this file's own
    # 'rviz' argument and silently suppress rviz_node below.
    #
    # It owns map -> base_footprint directly (publish.tf_child_frame), so
    # there is no odom edge and no separate correction node. See the frames
    # note in the header.
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

    fastloc = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('fast_lio'),
                    'launch', 'localization_l2.launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'config_file': LaunchConfiguration('config_file'),
                'map_dir': LaunchConfiguration('map_dir'),
                'map_pose_file': LaunchConfiguration('map_pose_file'),
                'map_scan_dir': LaunchConfiguration('map_scan_dir'),
                'rviz': 'false',
                'odom_frame': odom_frame,
            }.items(),
        ),
    ])

    nav2_params_file = os.path.join(
        pkg_share, 'config', 'nav2_params_fastloc.yaml')
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

    collision_monitor_off_warning = LogInfo(
        msg='WARNING: collision_monitor:=false -- TESTING ONLY. No stop or '
            'slowdown zone is active; only the driver\'s proximity and bumper '
            'stops stand between the controller and the wheels. Keep the e-stop '
            'in hand.',
        condition=UnlessCondition(cm_on))

    def make_lifecycle_manager(with_monitor):
        # OFF deliberately. local_costmap blocks configuring until a transform
        # to its global_frame (map) exists, and with this localizer that frame
        # appears only after ScanContext locks. That lock waits on the world,
        # not the clock -- enough scan overlap with the prior map, and
        # init_agree_count estimates agreeing on where it is -- so the wait is
        # unbounded and autostart would stall the whole bringup.
        # wait_for_map_then_start calls STARTUP the moment the frame is up.
        # map_server first so /map is up before the global costmap activates.
        # Exactly one of the two starts, so the name stays unique.
        return nav2_nodes.lifecycle_manager(
            use_sim_time,
            ['map_server', 'controller_server', 'planner_server', 'behavior_server',
             'bt_navigator'] + (['collision_monitor'] if with_monitor else []),
            autostart=False, log_level=log_level,
            condition=IfCondition(cm_on) if with_monitor else UnlessCondition(cm_on))

    return LaunchDescription([
        DeclareLaunchArgument(
            'voxel_markers', default_value='false',
            description='Start the voxel-marker converters for the 3D costmap view '
                        '(nav2_*_voxel.rviz, nav2_amcl.rviz). Off by default: '
                        'nothing in navigation uses them and they cost CPU.'),
        declare_use_sim_time_cmd,
        declare_sensor_tf_cmd,
        declare_sensor_tf_scope_cmd,
        declare_map_dir_cmd,
        declare_map_pose_file_cmd,
        declare_map_scan_dir_cmd,
        declare_map_cmd,
        declare_config_file_cmd,
        declare_lidar_imu_frame_cmd,
        declare_rviz_cmd,
        declare_rviz_config_cmd,
        declare_watchdog_cmd,
        declare_collision_monitor_cmd,
        collision_monitor_off_warning,
        declare_odom_frame_cmd,
        declare_log_level_cmd,
        sensor_tf,
        fastloc,
        nav2_nodes.map_server(use_sim_time, map_yaml, log_level),
        *nav2_nodes.nav2_servers(configured_params, log_level=log_level),
        nav2_nodes.points_safety_filter(use_sim_time, log_level),
        nav2_nodes.collision_monitor(configured_params, log_level, IfCondition(cm_on)),
        # Without the monitor, sensor_guard takes the controller's output.
        nav2_nodes.sensor_guard(
            use_sim_time,
            PythonExpression(["'cmd_vel_safe' if '", cm_on, is_true, " else 'cmd_vel_raw'"]),
            log_level),
        # The 3D view's LOCAL grid covers the RealSense cone only: the L2 is
        # on a 2D ObstacleLayer beside it, which publishes no voxels.
        *nav2_nodes.voxel_markers(
            use_sim_time, log_level, IfCondition(LaunchConfiguration('voxel_markers'))),
        nav2_nodes.rviz(use_sim_time, LaunchConfiguration('rviz_config'),
                        IfCondition(rviz), log_level),
        make_lifecycle_manager(True),
        make_lifecycle_manager(False),
        nav2_nodes.wait_for_map_then_start(use_sim_time, log_level),
        # Here it forwards to fastlio_localization's /relocalize.
        nav2_nodes.localization_recovery(use_sim_time, 'fastloc', log_level),
        # The configuration file's status names are fastloc's: the pose lock,
        # the map -> odom gate, and sensor_guard's sensor liveness.
        nav2_nodes.localization_watchdog(
            use_sim_time, IfCondition(LaunchConfiguration('watchdog')),
            log_level=log_level),
        nav2_nodes.wheel_odom_tf(use_sim_time, odom_frame, odom_on, log_level),
        nav2_nodes.tf_nav_relay(use_sim_time),
    ])
