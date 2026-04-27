"""Launch file for animate_behavior node with configurable parameters."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='animate_behavior',
            executable='animate_behavior',
            name='animate_behavior',
            output='screen',
            parameters=[{
                # General
                'verbose_mode': True,
                
                # LED Animation
                'led_enabled': True,
                'led_white_step': 0.06,
                'led_dark_step': 0.04,
                'led_fade_duration': 0.10,
                'led_white_hold': 2.0,
                'led_dark_pause': 0.2,
                
                # Gesture Animation
                'gesture_update_rate': 30.0,
                'gesture_smoothing_factor': 0.15,
                'gesture_motion_speed': 0.08,
                'gesture_interval_min': 2.5,
                'gesture_interval_max': 4.5,
                'gesture_rotation_interval': 5.0,
            }]
        )
    ])