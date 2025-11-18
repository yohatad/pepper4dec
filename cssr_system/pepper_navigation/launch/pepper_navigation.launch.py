import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paths
    map_file = '/home/lab/.ros/map.yaml'
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    
    return LaunchDescription([
        # Map Server
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_file}]
        ),
        
        # AMCL (Localization)
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'alpha1': 0.2,
                'alpha2': 0.2,
                'alpha3': 0.2,
                'alpha4': 0.2,
                'base_frame_id': 'base_footprint',
                'global_frame_id': 'map',
                'odom_frame_id': 'odom',
                'scan_topic': '/scan',  # Your lidar topic
                'transform_tolerance': 1.0,
                'robot_model_type': 'nav2_amcl::OmniMotionModel',
                'set_initial_pose': True,
                'initial_pose.x': 0.0,
                'initial_pose.y': 0.0,
                'initial_pose.yaw': 0.0
            }]
        ),
        
        # Nav2 Controller
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'controller_frequency': 20.0,
                'min_x_velocity_threshold': 0.001,
                'min_y_velocity_threshold': 0.5,
                'min_theta_velocity_threshold': 0.001,
                'progress_checker_plugin': 'progress_checker',
                'goal_checker_plugins': ['general_goal_checker'],
                'controller_plugins': ['FollowPath'],
                
                'progress_checker': {
                    'plugin': 'nav2_controller::SimpleProgressChecker',
                    'required_movement_radius': 0.5,
                    'movement_time_allowance': 10.0
                },
                'general_goal_checker': {
                    'plugin': 'nav2_controller::SimpleGoalChecker',
                    'xy_goal_tolerance': 0.25,
                    'yaw_goal_tolerance': 0.25
                },
                'FollowPath': {
                    'plugin': 'dwb_core::DWBLocalPlanner',
                    'debug_trajectory_details': True,
                    'min_vel_x': 0.0,
                    'min_vel_y': 0.0,
                    'max_vel_x': 0.26,
                    'max_vel_y': 0.0,
                    'max_vel_theta': 1.0,
                    'min_speed_xy': 0.0,
                    'max_speed_xy': 0.26,
                    'min_speed_theta': 0.0,
                    'acc_lim_x': 2.5,
                    'acc_lim_y': 0.0,
                    'acc_lim_theta': 3.2,
                    'decel_lim_x': -2.5,
                    'decel_lim_y': 0.0,
                    'decel_lim_theta': -3.2,
                    'vx_samples': 20,
                    'vy_samples': 5,
                    'vtheta_samples': 20,
                    'sim_time': 1.7,
                    'linear_granularity': 0.05,
                    'angular_granularity': 0.025
                }
            }]
        ),
        
        # Nav2 Planner
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'planner_plugins': ['GridBased'],
                'GridBased': {
                    'plugin': 'nav2_navfn_planner/NavfnPlanner',
                    'tolerance': 0.5,
                    'use_astar': False,
                    'allow_unknown': True
                }
            }]
        ),
        
        # Nav2 Behavior Server
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'local_costmap_topic': 'local_costmap/costmap_raw',
                'global_costmap_topic': 'global_costmap/costmap_raw',
                'local_footprint_topic': 'local_costmap/published_footprint',
                'global_footprint_topic': 'global_costmap/published_footprint',
                'cycle_frequency': 10.0,
                'behavior_plugins': ['spin', 'backup', 'wait'],
                'spin': {'plugin': 'nav2_behaviors/Spin'},
                'backup': {'plugin': 'nav2_behaviors/BackUp'},
                'wait': {'plugin': 'nav2_behaviors/Wait'}
            }]
        ),
        
        # Nav2 BT Navigator
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'global_frame': 'map',
                'robot_base_frame': 'base_footprint',
                'odom_topic': '/odom',
                'default_nav_to_pose_bt_xml': os.path.join(
                    nav2_bringup_dir, 'behavior_trees', 'navigate_to_pose_w_replanning_and_recovery.xml'
                ),
                'transform_tolerance': 0.1
            }]
        ),
        
        # Lifecycle Manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': [
                    'map_server',
                    'amcl',
                    'controller_server',
                    'planner_server',
                    'behavior_server',
                    'bt_navigator'
                ]
            }]
        )
    ])