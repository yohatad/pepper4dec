# Nav2 bringup for Pepper on fastlio_localization (FAST-LIO + prior map loaded
# into the iEKF, so the map constrains the estimate at scan rate from inside
# the filter). Owns map -> base_footprint; nav2_map_server serves the matching
# 2D grid as /map for the global costmap.
#
# Replaced lio_localization (github.com/yohatad/lio_localization), which
# applied the map constraint as a discrete, outside-the-filter map->odom step
# and produced jumps; this stack has no such step to jump.
#
# FRAMES.  map --(fastlio_localization)--> base_footprint
#              --(pepper_sensor_tf / bag tf_static)--> l2lidar_frame_imu, cams
#
# No odom frame: after handover the filter state IS the map pose, so the local
# costmap rolls in 'map' (see local_costmap global_frame in
# config/nav2_params_fastloc.yaml). Don't add transform_fusion/lio_odom_bridge
# here -- base_footprint would get two parents.
#
# Usage (real robot):
#   ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py
#   No initial pose needed: ScanContext finds it. Call /relocalize if lost.
#   A lock is accepted standing still; pass init_require_motion:=true to
#   require ~0.5 m of motion first (see that argument).
#
# Usage (bag replay):
#   ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py use_sim_time:=true
#   ros2 bag play <bag> --clock \
#       --qos-profile-overrides-path config/play_qos.yaml \
#       --read-ahead-queue-size 2000
#
# To use a DIFFERENT mapping run, change map, map_pose_file and map_scan_dir
# together, or the localizer and the costmap disagree about where the world is.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml


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
                    'prior map) or l2.yaml (the L2 s own).')
    declare_lidar_imu_frame_cmd = DeclareLaunchArgument(
        'lidar_imu_frame', default_value='camera_imu_optical_frame',
        description='Body frame matching config_file. camera_imu_optical_frame '
                    'for l2_rsimu.yaml, l2lidar_frame_imu for l2.yaml.')
    # Declared here even though it matches localization_l2.launch.py's own
    # default, so the knob is visible in --show-args at the nav level: it was
    # previously set only inside the include, where this file's header promised
    # motion was required and nothing was enforcing it.
    #
    # Left OFF deliberately. Turning it on trades one failure for another rather
    # than removing one. Off, agreement between the two ScanContext estimates can
    # be vacuous -- two matches can agree on the SAME wrong place without ever
    # being forced apart in space (MEASURED: 41 m off in a corridor). On, there is
    # no pose at all until the robot drives ~0.5 m, which stalls bringup entirely
    # (wait_for_map_then_start never fires) on a stationary start or a bag that
    # begins parked. The post-lock health check and localization_watchdog already
    # cover the wrong-lock case downstream; nothing covers never starting.
    declare_init_require_motion_cmd = DeclareLaunchArgument(
        'init_require_motion', default_value='false',
        description='Require init_motion_min (0.5 m) of motion between the '
                    'agreeing initial estimates before accepting a pose lock. '
                    'Off by default: a lock is available standing still. Set '
                    'true to harden startup against a wrong lock, at the cost '
                    'of no pose at all until the robot has driven ~0.5 m.')
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
                'init_require_motion': LaunchConfiguration('init_require_motion'),
                'rviz': 'false',
            }.items(),
        ),
    ])

    nav2_params_file = os.path.join(
        pkg_share, 'config', 'nav2_params_fastloc.yaml')
    configured_params = RewrittenYaml(
        source_file=nav2_params_file,
        root_key='',
        param_rewrites={'use_sim_time': use_sim_time},
        convert_types=True)

    # Serves the 2D grid as /map for the global costmap static layer.
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'yaml_filename': map_yaml,
            'frame_id': 'map',
        }],
    )

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
        remappings=[('cmd_vel', 'cmd_vel_raw')],
    )

    # Self-hit filter feeding the safety layer: strip Pepper's own body (< 0.8 m)
    # from the raw L2 /points so the collision monitor doesn't freeze on it.
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

    # VoxelLayer publishes nav2_msgs/VoxelGrid, which RViz cannot draw --
    # these converters turn it into a MarkerArray it can. See the equivalent
    # block in pepper_nav2_amcl.launch.py.
    local_voxel_markers = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d_markers',
        name='local_voxel_markers',
        output='log',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('voxel_grid', '/local_costmap/voxel_grid'),
                    ('visualization_marker', '/local_costmap/voxel_markers')],
    )
    global_voxel_markers = Node(
        package='nav2_costmap_2d',
        executable='nav2_costmap_2d_markers',
        name='global_voxel_markers',
        output='log',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('voxel_grid', '/global_costmap/voxel_grid'),
                    ('visualization_marker', '/global_costmap/voxel_markers')],
    )

    rviz_config = LaunchConfiguration('rviz_config')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(rviz),
    )

    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[configured_params],
    )
    nav2_starter = Node(
        package='pepper_navigation',
        executable='wait_for_map_then_start.py',
        name='wait_for_map_then_start',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time,
                     'target_frame': 'map',
                     'source_frame': 'base_footprint'}],
    )

    # One /localization_recover entry point, identical across all three nav
    # profiles, so recovering does not depend on remembering which backend is
    # up. Here it forwards to fastlio_localization's /relocalize.
    localization_recovery = Node(
        package='pepper_navigation',
        executable='localization_recovery.py',
        name='localization_recovery',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'backend': 'fastloc'}],
    )

    # Consumes fastlio_localization's own health status and holds navigation
    # while it says it is lost. Only wired into this profile: amcl and rtabmap
    # publish no comparable signal, so there is nothing for it to watch there.
    #
    # call_recovery stays FALSE: fastlio_localization already re-arms its own
    # search (auto_relocalize), so calling /localization_recover on top would
    # restart a search that is already running.
    localization_watchdog = Node(
        package='pepper_navigation',
        executable='localization_watchdog.py',
        name='localization_watchdog',
        output='screen',
        condition=IfCondition(LaunchConfiguration('watchdog')),
        parameters=[{
            'use_sim_time': use_sim_time,
            'status_name': 'fastlio_localization: pose lock',
            'lost_duration': 5.0,
            'cancel_goals': True,
            'call_recovery': False,
        }],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            # OFF deliberately. local_costmap blocks configuring until a
            # transform to its global_frame (map) exists, and with this
            # localizer that frame appears only after ScanContext locks. That
            # lock waits on the world, not the clock -- enough scan overlap
            # with the prior map, and init_agree_count estimates agreeing on
            # where it is -- so the wait is unbounded (longer still with
            # init_require_motion:=true, which also needs ~0.5 m of driving)
            # and autostart stalls the whole bringup. wait_for_map_then_start
            # calls STARTUP the moment the frame is up.
            'autostart': False,
            'bond_timeout': 4.0,
            # map_server first so /map is up before the global costmap activates.
            'node_names': [
                'map_server',
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'collision_monitor',
            ],
        }],
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_sensor_tf_cmd,
        declare_sensor_tf_scope_cmd,
        declare_map_dir_cmd,
        declare_map_pose_file_cmd,
        declare_map_scan_dir_cmd,
        declare_map_cmd,
        declare_config_file_cmd,
        declare_lidar_imu_frame_cmd,
        declare_init_require_motion_cmd,
        declare_rviz_cmd,
        declare_rviz_config_cmd,
        declare_watchdog_cmd,
        sensor_tf,
        fastloc,
        map_server,
        controller_server,
        planner_server,
        behavior_server,
        bt_navigator,
        points_safety_filter,
        collision_monitor,
        local_voxel_markers,
        global_voxel_markers,
        rviz_node,
        lifecycle_manager,
        nav2_starter,
        localization_recovery,
        localization_watchdog,
    ])
