#!/usr/bin/env python3
"""dec_system.launch.py

Top-level launch file for the Pepper4DEC system.

Brings up every dec_system package (each with its own launch file, some of
which nest further launch files for their dependencies, e.g. overt_attention's
launch nests person_detection and face_detection) and then sequences their
lifecycle nodes from `unconfigured` to `active` in dependency order via
nav2_lifecycle_manager.

Launch files included:
    overt_attention/attention_system.launch.py — camera, person and face
        detection, saliency, and the attention controller.
    fast_lio/localization_l2.launch.py — only when enable_navigation is false
        (see Notes); rviz:=false, sensor_tf_scope:=mount.
    animate_behavior, gesture_execution, speech_event, text_to_speech,
        conversation_manager, behavior_controller — each package's own launch
        file, unmodified.
    pepper_navigation/<nav_profile> — the selected Nav2 bringup, only when
        enable_navigation is true.

Nodes started:
    nav2_lifecycle_manager/lifecycle_manager (node: lifecycle_manager_dec_system)
        Drives configure -> activate for person_detection, face_detection,
        overt_attention, animate_behavior, gesture_action_server,
        speech_recognition, text_to_speech, conversation_manager, and
        behavior_controller. bond_timeout is 0.0 because these nodes do not
        implement the bond protocol nav2's own C++ lifecycle nodes use, and
        the localization nodes are plain rclcpp nodes, so they are
        deliberately absent from the managed list.

Launch arguments:
    enable_navigation (default: "true")
        Bring up pepper_navigation (the Nav2 navigation/localization stack).
    nav_profile (default: "fastloc"; choices: amcl, fastloc, legacy,
                 rtabmap_loc)
        Which Nav2 bringup to use when enable_navigation is true.
        fastloc = fastlio_localization, FAST-LIO with the prior map inside the
        iEKF; rtabmap_loc = RTAB-Map localization mode; amcl = AMCL on FAST-LIO
        odom; legacy = AMCL on raw wheel odom, kept only for reproducing old
        runs and publishing no /localization/pose.

Configuration:
    None of its own; each included launch file loads its package's YAML.

Usage:
    ros2 launch dec_launch dec_system.launch.py
    ros2 launch dec_launch dec_system.launch.py nav_profile:=rtabmap_loc
    ros2 launch dec_launch dec_system.launch.py enable_navigation:=false

Notes:
    Localization: the absolute `map -> base_footprint` pose
    (`/localization/pose`, consumed by gesture_execution for pointing IK)
    comes from fast_lio's fastlio_localization node, which publishes that
    topic with the pose AND twist composed into base_footprint.

    It replaces lio_localization, which has been removed (it lives on at
    github.com/yohatad/lio_localization). The difference is where the map
    constraint is applied: inside the iEKF at scan rate, rather than as a
    discrete map->odom correction computed by a node beside the filter. On
    slam_20260823_aligned this path had 0 correction steps over 0.30 m, 4.5 cm
    maximum.

    NOT YET RUN ON THE ROBOT -- everything measured is bag replay. Each of the
    `nav_profile` Nav2 bringups except `legacy` already nests its own
    localization, so it is launched standalone here only when navigation is
    off -- launching it twice would fight over the `map -> odom` transform.

    The localization include is wrapped in a scoped GroupAction because
    IncludeLaunchDescription emits its launch_arguments as
    SetLaunchConfiguration into the current context, so an unscoped
    'rviz': 'false' would leak into the nav profile's own 'rviz' argument.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 8, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
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
    # Default. fastlio_localization: the prior map IS the ikd-Tree the iEKF
    # registers against, so the map constrains the estimate at scan rate from
    # inside the filter, with no map->odom correction step to jump.
    'fastloc': 'pepper_nav2_fastloc.launch.py',
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
            default_value='fastloc',
            choices=sorted(NAV_PROFILES),
            description='Which Nav2 bringup to use when enable_navigation is '
                        'true. fastloc = fastlio_localization, FAST-LIO with '
                        'the prior map inside the iEKF (the default); '
                        'rtabmap_loc = RTAB-Map '
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
        # 'rviz' argument -- same trap pepper_nav2_fastloc.launch.py hit.
        GroupAction([
            _include('fast_lio', 'localization_l2.launch.py',
                     launch_arguments={'rviz': 'false',
                                       'sensor_tf_scope': 'mount'}.items()),
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
        # the localization nodes are plain rclcpp nodes, not lifecycle ones,
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
