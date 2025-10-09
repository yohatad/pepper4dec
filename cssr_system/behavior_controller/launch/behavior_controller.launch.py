#!/usr/bin/env python3
"""
Launch file for Tour Guide BehaviorTree (Python)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for tour guide"""
    
    # Declare launch arguments
    bt_loop_rate_arg = DeclareLaunchArgument(
        'bt_loop_rate',
        default_value='10',
        description='BehaviorTree tick rate in Hz'
    )
    
    asr_enabled_arg = DeclareLaunchArgument(
        'asr_enabled',
        default_value='false',
        description='Enable Automatic Speech Recognition'
    )
    
    display_tree_arg = DeclareLaunchArgument(
        'display_tree_on_start',
        default_value='true',
        description='Display tree structure on startup'
    )
    
    show_live_status_arg = DeclareLaunchArgument(
        'show_live_status',
        default_value='false',
        description='Show live tree status updates (refreshes terminal)'
    )
    
    # Tour Guide BehaviorTree Node
    tour_guide_node = Node(
        package='behavior_controller',
        executable='tour_guide_node',
        name='behavior_controller_node',
        output='screen',
        parameters=[{
            'bt_loop_rate': LaunchConfiguration('bt_loop_rate'),
            'asr_enabled': LaunchConfiguration('asr_enabled'),
            'display_tree_on_start': LaunchConfiguration('display_tree_on_start'),
            'show_live_status': LaunchConfiguration('show_live_status'),
        }],
        emulate_tty=True,
    )
    
    return LaunchDescription([
        bt_loop_rate_arg,
        asr_enabled_arg,
        display_tree_arg,
        show_live_status_arg,
        tour_guide_node,
    ])