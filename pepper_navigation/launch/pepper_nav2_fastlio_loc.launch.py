# Nav2 bringup for Pepper on FAST-LIO + lio_localization (prior-map ICP).
#
# This is the RTAB-Map-free localization path: instead of RTAB-Map providing
# both map->odom and /map, we use
#   * lio_localization (C++/PCL): registers FAST-LIO's /cloud_registered
#     against a saved .pcd (map_pcd) and owns map -> odom (via transform_fusion);
#   * nav2_map_server: serves the matching 2D occupancy grid (map) as /map for
#     the global costmap's static layer.
# No Open3D, no RTAB-Map, no PGO at runtime -- the lightest localization stack
# (see the Jetson CPU-budget discussion).
#
# Frames:  map --(transform_fusion)--> odom --(lio_map_odom_bridge)-->
#          base_footprint --(pepper_sensor_tf, static)--> l2lidar_frame_imu / cams
# The local costmap rolls in 'odom'; the global costmap and map_server in 'map'.
#
# Usage (real robot):
#   ros2 launch pepper_navigation pepper_nav2_fastlio_loc.launch.py \
#       map_pcd:=/home/yoha/Lidar/run_l2_lc/pgo_output/map_batch.pcd \
#       map:=/home/yoha/maps/pepper_clean.yaml
#   Then set the initial pose in RViz (2D Pose Estimate) so ICP locks on.
#
# Usage (bag replay sanity-check):
#   ros2 launch pepper_navigation pepper_nav2_fastlio_loc.launch.py use_sim_time:=true ...
#   ros2 bag play <bag> --clock
#
# RViz2 (rviz/nav2_fastlio_loc.rviz) opens by default -- map, both costmaps,
# global/local plans, the L2 pointcloud, and the collision monitor's stop/slow
# zones, plus the 2D Pose Estimate and Nav2 Goal tools. Pass rviz:=false to
# skip it (e.g. running headless on the robot with RViz on a remote machine).

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
    loc_launch_dir = os.path.join(
        get_package_share_directory('lio_localization'), 'launch')

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_pcd = LaunchConfiguration('map_pcd')
    map_yaml = LaunchConfiguration('map')
    localization_th = LaunchConfiguration('localization_th')
    rviz = LaunchConfiguration('rviz')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use bag/simulation clock instead of wall time.')
    declare_map_pcd_cmd = DeclareLaunchArgument(
        'map_pcd',
        default_value='/home/yoha/Lidar/run_l2_lc/pgo_output/map_batch.pcd',
        description='Prior 3D .pcd map that lio_localization registers against. '
                    'Matches fastlio_localization_l2.launch.py\'s own default: the '
                    'loop-closed map PGO writes when /pgo_batch_optimize is called '
                    'after a fastlio_lc_l2.launch.py run. That launch file checks the '
                    'path exists and fails with instructions if not.')
    declare_map_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(pkg_share, 'map', 'pepper_map_lc_clean.yaml'),
        description='2D occupancy grid (the projection of the same environment as '
                    'map_pcd) served as /map for the global costmap static layer. '
                    'Defaults to the copy shipped in this package (map/). MUST '
                    'exist, or the lifecycle manager aborts the whole bringup.')
    declare_localization_th_cmd = DeclareLaunchArgument(
        'localization_th', default_value='0.90',
        description='Min ICP inlier-ratio fitness to accept (lower for partial '
                    'L2 overlap; 0.6-0.9 typical).')
    declare_rviz_config_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_share, 'rviz', 'nav2_fastlio_loc.rviz'),
        description='RViz config. Default is the standard view; pass '
                    'nav2_fastlio_loc_voxel.rviz for the 3D voxel-map view '
                    '(needs the voxel marker converters this file launches, '
                    'and z_voxels <= 16 in the nav2 params).')
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
                'localization_th': localization_th,
                'rviz': 'false',
            }.items(),
        ),
    ])

    nav2_params_file = os.path.join(
        pkg_share, 'config', 'nav2_params_fastlio_loc.yaml')
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
        package='fast_lio',
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
        declare_localization_th_cmd,
        declare_rviz_cmd,
        declare_rviz_config_cmd,
        lio_localization,
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
    ])
