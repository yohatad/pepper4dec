# Nav2 bringup for Pepper with SOCIALLY-AWARE navigation.
#
# Derived from pepper_navigation/launch/pepper_nav2_fastlio_loc.launch.py --
# same localization (lio_localization ICP against a prior .pcd), same maps,
# same frames, same velocity chain. Three things differ, and config/
# nav2_params_social.yaml explains each: MPPI instead of DWB, a holonomic
# lateral channel, and the two social components (ProxemicLayer + SocialForceCritic).
#
# WHAT THIS LAUNCH DOES *NOT* START. No camera driver, no lidar, no person
# detector. Perception is expected to run OFFBOARD on a laptop -- see
# perception_offboard.launch.py -- because YOLO on two camera streams does not
# fit the Jetson budget alongside FAST-LIO and MPPI. What crosses back over
# WiFi is only social_nav_msgs/Pedestrians, which is a few hundred bytes.
#
# Set run_tracker:=true to run the people tracker HERE instead (then the
# laptop only runs detection, and detections cross the link rather than tracks).
#
# WHAT STAYS ON THE ROBOT, ALWAYS: the controller, the costmaps and the
# collision monitor. A WiFi dropout must never sit between a lidar return and
# the brakes. The social layers all degrade to "no social cost" on a stale
# link (message_timeout), leaving the lidar-driven collision monitor intact.
#
# Usage (robot):
#   ros2 launch pepper_social_nav pepper_nav2_social.launch.py
# Usage (laptop, separately):
#   ros2 launch pepper_social_nav perception_offboard.launch.py
#
# VALIDATE IN STAGES. Before trusting the social terms, run this stack with
# SocialForceCritic.cost_weight 0.0 and proxemic_layer.enabled false and
# confirm the robot drives its tour as well as the DWB stack did. Switching
# controller and adding social costs at once makes a regression unattributable.

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_navigation')
    social_share = get_package_share_directory('pepper_social_nav')
    loc_launch_dir = os.path.join(
        get_package_share_directory('lio_localization'), 'launch')

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_pcd = LaunchConfiguration('map_pcd')
    map_yaml = LaunchConfiguration('map')
    rviz = LaunchConfiguration('rviz')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use bag/simulation clock instead of wall time.')
    declare_map_pcd_cmd = DeclareLaunchArgument(
        'map_pcd',
        default_value=os.path.join(pkg_share, 'map', 'pepper_map_lc.pcd'),
        description='Prior 3D .pcd map that lio_localization registers against. '
                    'Now shipped in this package (map/) alongside the 2D grid, so '
                    'the pair cannot drift apart. /pgo_batch_optimize writes it '
                    'there directly (fastlio_lc_pgo map_pcd_path). MUST come from '
                    'the same mapping run as map and keyframe_poses.')
    declare_map_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(pkg_share, 'map', 'pepper_map_lc_aug22_clean.yaml'),
        description='2D occupancy grid (the projection of the same environment as '
                    'map_pcd) served as /map for the global costmap static layer. '
                    'Defaults to the copy shipped in this package (map/). MUST '
                    'exist, or the lifecycle manager aborts the whole bringup.')
    # localization_th was declared here and forwarded to
    # fastlio_localization_l2.launch.py, which declared it too and never passed
    # it to any node -- so the value did nothing while appearing to work, and
    # its 0.90 default silently disagreed with the 0.85 in
    # lio_localization/config/localization.yaml. That YAML is the only place
    # the ICP acceptance gate can be set; it is coupled to max_corr_dist and
    # the two must be changed together.
    # Seedless localization / /relocalize candidates. This was NOT forwarded
    # before, so overriding map_pcd alone still left the global search loading
    # whatever lio_localization defaulted to -- candidate places expressed in a
    # DIFFERENT run's frame, with nothing in the logs saying so. Declaring it
    # here keeps the three artifacts (map_pcd, map, keyframe_poses) switchable
    # as one set.
    # These are LEVELLED poses. PGO writes optimized_poses*.txt in the RAW
    # pgo_init frame while the .pcd it saves is gravity-levelled, so the two
    # are ~90 deg apart as written -- feeding the raw file here makes the global
    # search test candidate places that do not exist in the map's frame. The
    # shipped copy has the level transform already applied (verified: post-
    # levelling the pose z std is 0.22 m, i.e. a robot on a flat floor, and the
    # track lies inside the grid footprint).
    declare_keyframe_poses_cmd = DeclareLaunchArgument(
        'keyframe_poses',
        default_value=os.path.join(pkg_share, 'map', 'pepper_map_lc_poses.txt'),
        description='KITTI-format keyframe poses from the SAME mapping run as '
                    'map_pcd, used as candidates for global localization and '
                    '/relocalize. Empty disables both (manual /initialpose only).')
    # Which IMU drives FAST-LIO. Forwarded so the whole stack switches from one
    # place; lio_localization derives nothing on its own.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2_rsimu.yaml',
        description='FAST-LIO config: l2_rsimu.yaml (RealSense IMU, matches the '
                    'prior map) or l2.yaml (the L2 s own).')
    declare_lidar_imu_frame_cmd = DeclareLaunchArgument(
        'lidar_imu_frame', default_value='camera_imu_optical_frame',
        description='Body frame matching config_file. camera_imu_optical_frame '
                    'for l2_rsimu.yaml, l2lidar_frame_imu for l2.yaml.')
    declare_rviz_config_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(social_share, 'rviz', 'social_nav.rviz'),
        description='RViz config. Default shows both costmaps, the proxemic '
                    'cost field, tracked pedestrians with velocity arrows, '
                    'MPPI candidate trajectories and the safety polygons.')
    declare_run_tracker_cmd = DeclareLaunchArgument(
        'run_tracker', default_value='false',
        description='Run the people tracker on the robot instead of on the '
                    'laptop. Default false: perception_offboard.launch.py runs '
                    'detection AND tracking, so only Pedestrians crosses WiFi.')
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz2 pre-configured for this nav stack (map, costmaps, '
                    'plans, safety zones, 2D Pose Estimate / Nav2 Goal tools).')

    # FAST-LIO (odometry, no PGO) + sensor TF + global_localization +
    # transform_fusion. This owns odom -> base_footprint and map -> odom.
    # GroupAction (scoped by default) is REQUIRED here: IncludeLaunchDescription
    # emits its launch_arguments as SetLaunchConfiguration into the CURRENT
    # context, so 'rviz': 'false' would otherwise overwrite this file's own
    # 'rviz' argument and silently suppress rviz_node below.
    lio_localization = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(loc_launch_dir, 'fastlio_localization_l2.launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'map_pcd': map_pcd,
                'keyframe_poses': LaunchConfiguration('keyframe_poses'),
                'config_file': LaunchConfiguration('config_file'),
                'lidar_imu_frame': LaunchConfiguration('lidar_imu_frame'),
                'rviz': 'false',
            }.items(),
        ),
    ])

    nav2_params_file = os.path.join(
        social_share, 'config', 'nav2_params_social.yaml')
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
            # 0.30, NOT the 0.8 the other stacks use. At 0.8 the closest point
            # that can survive is ~0.78 m out horizontally, so the collision
            # monitor's 0.40 m stop polygon could never fire -- a dead safety
            # zone in exactly the band a social robot works in. Self-hits are
            # removed by GEOMETRY below instead of by radius.
            'min_range': 0.30,
            # Pepper's own body, as a box in base_footprint:
            # [xmin, xmax, ymin, ymax, zmin, zmax].
            #
            # !! STARTING POINT FROM NOMINAL DIMENSIONS, NOT A MEASUREMENT !!
            # Pepper's base is ~0.48 m across, but the arms and the tablet
            # extend further and the measured self-hit shell reached ~0.6 m from
            # the lidar. Verify before driving: watch /points_safety in RViz and
            # confirm no returns sit on the robot. If it freezes in place, the
            # box is leaking self-hits into the stop polygon -- widen it, or set
            # min_range back to 0.8 and accept the dead zone until it is tuned.
            'self_filter_box': [-0.38, 0.38, -0.32, 0.32, 0.0, 1.40],
            'self_filter_frame': 'base_footprint',
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

    # Optional on-robot tracker. Same node and config as the offboard one --
    # only where it runs differs.
    people_tracker = Node(
        package='pepper_social_nav',
        executable='people_tracker',
        name='people_tracker',
        output='screen',
        parameters=[
            os.path.join(social_share, 'config', 'people_tracker.yaml'),
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration('run_tracker')),
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
        declare_map_pcd_cmd,
        declare_map_cmd,
        declare_keyframe_poses_cmd,
        declare_config_file_cmd,
        declare_lidar_imu_frame_cmd,
        declare_run_tracker_cmd,
        declare_rviz_cmd,
        declare_rviz_config_cmd,
        lio_localization,
        map_server,
        controller_server,
        planner_server,
        behavior_server,
        bt_navigator,
        people_tracker,
        points_safety_filter,
        collision_monitor,
        local_voxel_markers,
        global_voxel_markers,
        rviz_node,
        lifecycle_manager,
    ])
