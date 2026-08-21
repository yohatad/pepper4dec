#!/usr/bin/env python3
"""
Top-level launch file for the Pepper4DEC system.

dec_system.launch.py

Brings up every dec_system package (each with its own launch file, some of
which nest further launch files for their dependencies, e.g.
overt_attention's launch nests person_detection and face_detection) and then
sequences their lifecycle nodes from `unconfigured` to `active` in dependency
order via nav2_lifecycle_manager.

Localization: the absolute `map -> base_footprint` pose (`/localization/pose`,
consumed by gesture_execution for pointing IK) comes from lio_localization's
transform_fusion node. Each of the `nav_profile` Nav2 bringups except `legacy`
already nests its own localization, so lio_localization is launched standalone
here only when navigation is off -- launching it twice would fight over the
`map -> odom` transform.
"""
import os
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Nav2 bringups selectable via `nav_profile`. See pepper_navigation/README.md
# for the trade-offs; `legacy` is AMCL on naoqi's raw wheel odometry and is
# kept only for reproducing old runs -- it publishes no /localization/pose.
NAV_PROFILES = {
    'fastlio_loc': 'pepper_nav2_fastlio_loc.launch.py',
    'rtabmap_loc': 'pepper_nav2_rtabmap_loc.launch.py',
    'amcl': 'pepper_nav2_amcl.launch.py',
    'legacy': 'pepper_navigation.launch.py',
}


def _include(package, launch_file, **kwargs):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package), 'launch', launch_file)
        ),
        **kwargs
    )


def _nav_stack(context, *args, **kwargs):
    """
    Resolve `nav_profile` to a launch file at runtime.

    No membership check needed: the argument's `choices` are generated from
    NAV_PROFILES below, so the two cannot drift apart and launch rejects an
    unknown profile before this ever runs.
    """
    profile = LaunchConfiguration('nav_profile').perform(context)
    return [
        _include('pepper_navigation', NAV_PROFILES[profile],
                 condition=IfCondition(LaunchConfiguration('enable_navigation')))
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_navigation',
            default_value='true',
            description='Whether to bring up pepper_navigation '
                        '(Nav2 navigation/localization stack)'
        ),
        DeclareLaunchArgument(
            'nav_profile',
            default_value='fastlio_loc',
            choices=sorted(NAV_PROFILES),
            description='Which Nav2 bringup to use when enable_navigation is '
                        'true. fastlio_loc = FAST-LIO + prior-map ICP '
                        '(lio_localization); rtabmap_loc = RTAB-Map '
                        'localization mode; amcl = AMCL on FAST-LIO odom; '
                        'legacy = AMCL on wheel odom (no /localization/pose)'
        ),

        # Perception: shared camera + person/face detection + overt attention
        _include('overt_attention', 'attention_system.launch.py'),

        # Localization, only when no nav profile is nesting it already.
        # Owns map -> odom and publishes /localization/pose.
        # Scoped GroupAction: IncludeLaunchDescription emits its
        # launch_arguments as SetLaunchConfiguration into the CURRENT context,
        # so an unscoped 'rviz': 'false' would leak into the nav profile's own
        # 'rviz' argument -- same trap pepper_nav2_fastlio_loc.launch.py hit.
        GroupAction([
            _include('lio_localization', 'fastlio_localization_l2.launch.py',
                     launch_arguments={'rviz': 'false'}.items()),
        ], condition=UnlessCondition(LaunchConfiguration('enable_navigation'))),

        # Actuation
        _include('animate_behavior', 'animate_behavior.launch.py'),
        _include('gesture_execution', 'gesture_execution.launch.py'),

        # Speech / dialogue
        _include('speech_event', 'speech_event.launch.py'),
        _include('text_to_speech', 'text_to_speech.launch.py'),
        _include('conversation_manager', 'conversation_manager.launch.py'),

        # Navigation stack (own internal lifecycle manager)
        OpaqueFunction(function=_nav_stack),

        # Top-level behavior orchestration (BT)
        _include('behavior_controller', 'behavior_controller.launch.py'),

        # Sequence configure -> activate for the custom lifecycle nodes above,
        # in dependency order. bond_timeout is disabled because these nodes
        # don't implement the bond protocol used by nav2's C++ lifecycle nodes.
        # lio_localization's nodes are plain rclcpp nodes, not lifecycle ones,
        # so they are deliberately absent from this list.
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_dec_system',
            output='screen',
            parameters=[{
                'autostart': True,
                'bond_timeout': 0.0,
                'node_names': [
                    'person_detection',
                    'face_detection',
                    'overt_attention',
                    'animate_behavior',
                    'gesture_action_server',
                    'speech_recognition',
                    'text_to_speech',
                    'conversation_manager',
                    'behavior_controller',
                ],
            }],
        ),
    ])
