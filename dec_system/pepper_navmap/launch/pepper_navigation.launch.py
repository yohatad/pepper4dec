import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paths
    pkg_dir = get_package_share_directory('pepper_navigation')
    map_file = os.path.join(pkg_dir, 'map', 'rtabmap_feb_26.yaml')
    params_file = os.path.join(pkg_dir, 'config', 'nav2_params.yaml')
    keepout_mask_file = os.path.join(pkg_dir, 'map', 'keepout_zones.yaml')
    
    return LaunchDescription([
        # Map Server (lifecycle node)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_file}]
        ),
        
        # Filter Mask Server (lifecycle node - add to lifecycle manager)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='filter_mask_server',
            output='screen',
            parameters=[{
                'yaml_filename': keepout_mask_file,
                'topic_name': '/keepout_filter_mask',  # Absolute path
                'frame_id': 'map'
            }]
        ),

        # Costmap Filter Info Server (lifecycle node)
        Node(
            package='nav2_map_server',
            executable='costmap_filter_info_server',
            name='costmap_filter_info_server',
            output='screen',
            parameters=[params_file]
        ),
        
        # AMCL (Localization)
        # Node(
        #     package='nav2_amcl',
        #     executable='amcl',
        #     name='amcl',
        #     output='screen',
        #     parameters=[params_file]
        # ),
        
        # Nav2 Controller
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file]
        ),
        
        # Nav2 Planner
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file]
        ),
        
        # Nav2 Behavior Server
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params_file]
        ),
        
        # Nav2 BT Navigator
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_map_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
        ),

        # Lifecycle Manager (ONLY lifecycle nodes here)
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': True,
                'bond_timeout': 4.0,
                'node_names': [
                    'map_server',
                    'filter_mask_server',
                    'costmap_filter_info_server',  # Must activate lifecycle node
                    # 'amcl',
                    'controller_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator'
                ]
            }]
        )
    ])