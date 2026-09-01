# Nav2 bringup for Pepper on AMCL + map_server, over FAST-LIO odometry.
#
# The localization BASELINE, to compare against the two LIO paths:
#   * this file                         -- amcl (particle filter vs a 2D grid)
#   * pepper_nav2_fastlio_loc.launch.py -- lio_localization (3D ICP vs .pcd)
#   * pepper_nav2_rtabmap_loc.launch.py -- RTAB-Map localization mode (.db)
# Everything downstream of localization (costmaps, DWB, collision monitor) is
# identical across the three, so a behavioural difference is a localization
# difference.
#
# AMCL needs two things Pepper's sensor rig doesn't provide directly:
#   1. LEVEL odom -> base_footprint dead reckoning. FAST-LIO supplies it (NOT
#      naoqi's wheel odom, which lives on the separate 'pepper_odom' frame --
#      see config/README.md), but its raw 'odom' is the initial-IMU frame,
#      tilted ~90deg on Pepper's mount. amcl needs a level frame, so
#      mapping.launch.py runs with bridge_level_frame:=true and amcl corrects
#      'odom'. Getting this wrong makes the pose jump on every update.
#   2. a sensor_msgs/LaserScan. The L2 is a 360 deg PointCloud2, so
#      pointcloud_to_laserscan flattens /points -> /scan in a height band. The
#      costmaps still take the full 3D /points; the scan is localization-only.
#
# Frames:  map --(amcl)--> odom --(bridge, static)--> odom
#          --(lio_odom_bridge)--> base_footprint
#          --(pepper_sensor_tf, static)--> l2lidar_frame / cams
#
# Usage (real robot):
#   ros2 launch l2lidar_node l2lidar.launch.py            # /points + /imu/data
#   ros2 launch pepper_navigation pepper_nav2_amcl.launch.py \
#       map:=<path to a .yaml>     # defaults to this package's map/
#   Then set the initial pose in RViz (2D Pose Estimate) -- amcl starts
#   UNLOCALIZED (set_initial_pose: false) and the particle cloud will not
#   converge until you do.
#
# Usage (bag replay sanity-check):
#   ros2 launch pepper_navigation pepper_nav2_amcl.launch.py use_sim_time:=true
#   ros2 bag play <bag> --clock --topics /points /imu/data
#
# Requires ros-humble-pointcloud-to-laserscan (not a default Nav2 dependency):
#   sudo apt install ros-humble-pointcloud-to-laserscan
#
# RViz2 (rviz/nav2_amcl.rviz) opens by default -- as the ICP stack's config,
# plus the flattened /scan and amcl's /particle_cloud so you can watch the
# filter converge. Pass rviz:=false to run headless.

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


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_navigation')
    fast_lio_launch_dir = os.path.join(
        get_package_share_directory('fast_lio'), 'launch')
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
        default_value=os.path.join(pkg_share, 'map', 'pepper_map_lc_clean.yaml'),
        description='2D occupancy grid served as /map -- both what amcl matches '
                    'the flattened scan against and the global costmap static layer. '
                    'Defaults to the copy shipped in this package (map/), so the '
                    'stack comes up on a fresh checkout with no absolute paths. '
                    'MUST exist: map_server fails to configure otherwise, and the '
                    'lifecycle manager then aborts the WHOLE nav2 bringup.')
    # The two knobs to reach for first if amcl will not converge. The band is in
    # base_footprint (floor at z=0). Too low and the flattened floor returns
    # swamp the wall hits; too high and tables/desks/people appear in the scan
    # but not in the grid, so every beam mismatches.
    declare_scan_min_height_cmd = DeclareLaunchArgument(
        'scan_min_height', default_value='0.20',
        description='Bottom of the /points slice flattened into /scan, in '
                    'base_footprint (floor at z=0). Raise it if floor returns '
                    'leak into the scan.')
    declare_scan_max_height_cmd = DeclareLaunchArgument(
        'scan_max_height', default_value='1.50',
        description='Top of the /points slice flattened into /scan. Lower it if '
                    'furniture absent from the 2D grid is confusing amcl.')
    declare_rviz_config_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(pkg_share, 'rviz', 'nav2_amcl.rviz'),
        description='RViz config. Default is the standard view; pass '
                    'nav2_amcl_voxel.rviz for the 3D voxel-map view '
                    '(needs the voxel marker converters this file launches, '
                    'and z_voxels <= 16 in the nav2 params).')
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz2 pre-configured for this stack (map, particle '
                    'cloud, flattened scan, costmaps, plans, safety zones).')

    # NOTE ON GroupAction: IncludeLaunchDescription does NOT scope its
    # launch_arguments -- it emits plain SetLaunchConfiguration actions into the
    # CURRENT context, so 'rviz': 'false' below would otherwise overwrite THIS
    # file's own 'rviz' configuration and silently suppress rviz_node (whose
    # IfCondition is evaluated later). GroupAction is scoped by default, which
    # keeps each include's arguments to itself.

    # base_footprint -> l2lidar_frame (+ cams). NOT included by
    # mapping.launch.py, and pointcloud_to_laserscan needs it to reproject the
    # cloud into base_footprint, so it must be launched here.
    sensor_tf = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensor_tf_launch_dir, 'pepper_sensor_tf.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),
    ])

    # FAST-LIO odometry. bridge_level_frame:=TRUE is REQUIRED here, and amcl
    # corrects odom (not odom) -- see nav2_params_amcl.yaml's header.
    # FAST-LIO's raw 'odom' is the initial-IMU frame, which on Pepper's mount is
    # tilted ~90deg (its Z axis runs along base_footprint's +X). amcl's motion
    # model reads (x, y, yaw) out of odom_frame -> base_frame and assumes that
    # frame is level: in the raw frame forward travel barely registers in x-y and
    # the yaw is about a horizontal axis, so the filter cannot track and the pose
    # jumps on every scan update. The bridge's odom IS gravity-aligned.
    #
    # This still leaves every frame with exactly one parent:
    #   map --(amcl)--> odom --(bridge, static)--> odom --(bridge)--> base_footprint
    fast_lio = GroupAction([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(fast_lio_launch_dir, 'mapping.launch.py')),
            launch_arguments={
                'config_file': 'l2.yaml',
                'rviz': 'false',
                'use_sim_time': use_sim_time,
            }.items(),
        ),
        # odom -> base_footprint. FAST_LIO's mapping.launch.py no longer starts this
        # (see FAST_LIO d8b274c): it was Pepper glue in a launch file shared with
        # every other FAST-LIO sensor config.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('pepper_slam'),
                             'launch', 'lio_odom_bridge.launch.py')),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'config_file': 'l2.yaml',
                'bridge_level_frame': 'true',
            }.items(),
        ),
    ])

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
            # value_type=float: a bare LaunchConfiguration would arrive as a
            # string and the node would reject the parameter type.
            'min_height': ParameterValue(scan_min_height, value_type=float),
            'max_height': ParameterValue(scan_max_height, value_type=float),
            'angle_min': -3.141592653589793,   # full 360 deg, like the L2
            'angle_max': 3.141592653589793,
            'angle_increment': 0.008726646259971648,   # 0.5 deg -> 720 beams
            'scan_time': 0.1,
            # 0.8 m matches amcl's laser_min_range and the costmaps'
            # obstacle_min_range: below it the low-mounted L2 sees Pepper itself.
            'range_min': 0.8,
            'range_max': 20.0,
            'use_inf': True,
            'inf_epsilon': 1.0,
            'concurrency_level': 2,
        }],
    )

    nav2_params_file = os.path.join(
        pkg_share, 'config', 'nav2_params_amcl.yaml')
    configured_params = RewrittenYaml(
        source_file=nav2_params_file,
        root_key='',
        param_rewrites={'use_sim_time': use_sim_time},
        convert_types=True)

    # Serves the 2D grid both to amcl (match target) and to the global
    # costmap's static layer.
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

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[configured_params],
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
    bt_navigator = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[configured_params],
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

    # Self-hit + adaptive ground-plane filter feeding the COSTMAPS specifically
    # (separate instance from points_safety_filter above -- the collision
    # monitor's safety-critical stop zone stays on the simpler, cheaper filter
    # untouched by this). The costmaps' voxel_layer marks obstacles with a
    # FIXED height band evaluated in their global_frame (odom/map),
    # whose leveling is a one-time snapshot that can drift enough over a long
    # run to mis-mark the real floor as an obstacle. RANSAC-fitting the ground
    # plane per scan in base_footprint (continuously re-anchored to FAST-LIO's
    # live, gravity-referenced attitude) instead -- the same idea used to build
    # this map via octomap_server's filter_ground_plane -- removes that failure
    # mode instead of just giving it more margin to drift into.
    points_costmap_filter = Node(
        package='pepper_slam',
        executable='cloud_range_filter.py',
        name='points_costmap_filter',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_topic': '/points',
            'output_topic': '/points_costmap',
            'min_range': 0.8,
            'ror_min_neighbors': 0,
            'remove_ground_plane': True,
            'ground_frame': 'base_footprint',
            'ground_distance_thresh': 0.05,
            'ground_angle_thresh': 0.15,
            # MEASURED: with the defaults (z_thresh 0.12, 60 iterations) roughly
            # HALF the scans logged "0 dropped as floor" -- RANSAC found no
            # acceptable plane and fell back to keep-everything, so ground
            # removal was silently only working half the time and the costmap
            # height band was quietly doing the job instead.
            #   * z_thresh 0.12 sat right at the measured base_footprint z drift
            #     (-0.04 .. +0.13 m): past 0.12 the real floor looks too far from
            #     z=0 and EVERY candidate plane is rejected. 0.20 clears it.
            #   * 60 iterations is thin when the floor is only ~15% of a ~4k-point
            #     scan -- the chance of drawing 3 floor points in one sample is
            #     ~0.3%, so 60 tries misses more often than not. It is pure NumPy
            #     over the scan, so 300 is still cheap.
            'ground_z_thresh': 0.20,
            'ground_ransac_iterations': 300,
        }],
    )

    collision_monitor = Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        parameters=[configured_params],
    )

    # The costmaps' VoxelLayer publishes nav2_msgs/VoxelGrid on
    # <costmap>/voxel_grid, which RViz has NO display for -- so the 3D voxel
    # map was invisible no matter what you added to the RViz config. These
    # converters turn it into a MarkerArray RViz can draw. Without them the
    # rviz config's "Local/Global Voxel Grid" displays subscribe to a topic
    # nobody publishes. (Also requires z_voxels <= 16 in nav2_params_amcl.yaml
    # -- above that the layer refuses to build its grid at all.)
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

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'bond_timeout': 4.0,
            # map_server first so /map is up before amcl and the global costmap
            # activate; amcl before the costmaps so map -> odom exists.
            'node_names': [
                'map_server',
                'amcl',
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
        declare_map_cmd,
        declare_scan_min_height_cmd,
        declare_scan_max_height_cmd,
        declare_rviz_cmd,
        declare_rviz_config_cmd,
        sensor_tf,
        fast_lio,
        pointcloud_to_laserscan,
        map_server,
        amcl,
        controller_server,
        planner_server,
        behavior_server,
        bt_navigator,
        points_safety_filter,
        points_costmap_filter,
        collision_monitor,
        local_voxel_markers,
        global_voxel_markers,
        rviz_node,
        lifecycle_manager,
    ])
