# Nav2 bringup for Pepper on FAST-LIO + RTAB-Map (localization mode).
#
# Localizes against the saved rtabmap_fastlio_refined.db (today's best
# validated map -- see project_l2_slam_stack memory) instead of AMCL + a
# static map_server: RTAB-Map runs with Mem/IncrementalMemory=false, reusing
# the exact odometry/appearance/ICP pipeline already tuned for mapping, and
# publishes /map itself. No pointcloud_to_laserscan conversion needed --
# Nav2's costmaps take /points (PointCloud2) directly.
#
# Frames: FAST-LIO's odom -> lio_map_odom_bridge's gravity-leveled
# odom_level -> RTAB-Map's map. See nav2_params_rtabmap_loc.yaml for why
# local_costmap uses odom_level (not pepper_odom) as its global_frame.
#
# Usage (real robot):
#   ros2 launch pepper_navigation pepper_nav2_bringup.launch.py
#
# Usage (bag replay, to sanity-check localization/costmaps without driving):
#   ros2 launch pepper_navigation pepper_nav2_bringup.launch.py use_sim_time:=true
#   ros2 bag play <bag> --clock --topics /points /imu/data /tf_static \
#       /camera/color/image_raw /camera/color/camera_info
#   (Nav2 will localize and build costmaps, but a bag can't react to cmd_vel
#   -- driving to a goal needs the real robot or a simulator.)

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

    rtabmap_localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'rtabmap_realsense.launch.py')),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'frame_id': 'base_footprint',
            'odom_frame_id': 'odom_level',

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
            # rtabmap_fastlio_bag_test.launch.py); dropped --delete_db_on_start
            # (would erase the map!) and the NeighborLinkRefining/Proximity
            # params (those govern adding NEW loop-closure links, moot with
            # Mem/IncrementalMemory=false).
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
            ],
        }],
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        declare_database_path_cmd,
        fast_lio,
        rtabmap_localization,
        controller_server,
        planner_server,
        behavior_server,
        bt_navigator,
        lifecycle_manager,
    ])
