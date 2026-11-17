"""The nodes every Pepper Nav2 profile starts, defined once.

The five pepper_nav2_*.launch.py profiles differ only in localization; the
Nav2 servers, the safety chain, the visualization and pepper_navigation's own
nodes are the same in each. Each function here returns those Node actions, so
a fix to one block reaches every profile instead of one copy of it.

Parameters come from the configuration files, not from here:
config/pepper_navigation_configuration.yaml for this package's nodes,
pepper_slam_configuration.yaml for the cloud filters and wheel_odom_tf, and
the profile's nav2_params_*.yaml (as a RewrittenYaml) for Nav2's servers. A
function takes only what a profile overrides, plus use_sim_time.

log_level: None starts the node with its default severity; a substitution
(the fastloc profile's log_level argument) passes --log-level.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 17, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

NAV_CONFIG = os.path.join(
    get_package_share_directory('pepper_navigation'), 'config',
    'pepper_navigation_configuration.yaml')

# The cloud_range_filter instances (points_safety_filter,
# points_costmap_filter) and wheel_odom_tf read their sections of it.
PEPPER_SLAM_CONFIG = os.path.join(
    get_package_share_directory('pepper_slam'), 'config',
    'pepper_slam_configuration.yaml')

# Nav2 reads /tf_nav, a copy of /tf holding only the navigation edges:
# naoqi_driver's joint tree (~4150 transforms/s) saturated the costmap's TF
# thread until it stopped updating. See src/tf_nav_relay.cpp. Only nodes that
# LISTEN to TF are remapped -- never one that broadcasts it.
TF_NAV = ('/tf', '/tf_nav')


def _log(log_level, arguments=()):
    """Node arguments, with --log-level when the profile sets one."""
    args = list(arguments)
    if log_level is not None:
        args += ['--ros-args', '--log-level', log_level]
    return args


def nav2_servers(params, cmd_vel_out='cmd_vel_raw', log_level=None):
    """controller_server, planner_server, behavior_server and bt_navigator.

    The controller's and behaviours' velocity goes to cmd_vel_out: cmd_vel_raw
    is the collision monitor's input, so recovery motions (spin, backup) go
    through it too. The monitor publishes cmd_vel_safe (nav2_params_*.yaml),
    and sensor_guard passes it on as cmd_vel.
    """
    def server(package, executable, remappings):
        return Node(
            package=package,
            executable=executable,
            name=executable,
            output='screen',
            arguments=_log(log_level),
            parameters=[params],
            remappings=remappings,
        )
    return [
        server('nav2_controller', 'controller_server', [('cmd_vel', cmd_vel_out), TF_NAV]),
        server('nav2_planner', 'planner_server', [TF_NAV]),
        server('nav2_behaviors', 'behavior_server', [('cmd_vel', cmd_vel_out), TF_NAV]),
        server('nav2_bt_navigator', 'bt_navigator', [TF_NAV]),
    ]


def map_server(use_sim_time, map_yaml, log_level=None):
    """Serves the 2D grid as /map (amcl's match target, the static layer)."""
    return Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        arguments=_log(log_level),
        parameters=[{
            'use_sim_time': use_sim_time,
            'yaml_filename': map_yaml,
            'frame_id': 'map',
        }],
    )


def points_safety_filter(use_sim_time, log_level=None):
    """The collision monitor's input: /points without the L2's own housing.

    MEASURED: the old 0.8 m cut also hid everything below ~0.7 m inside both
    monitor zones, so legs never triggered a stop.
    """
    return Node(
        package='pepper_slam',
        executable='cloud_range_filter',
        name='points_safety_filter',
        output='screen',
        arguments=_log(log_level),
        parameters=[PEPPER_SLAM_CONFIG, {'use_sim_time': use_sim_time}],
    )


def points_costmap_filter(use_sim_time):
    """Self-hit and ground-plane filter feeding the costmaps (amcl profiles).

    A separate instance from points_safety_filter, so the safety-critical stop
    zone stays on the simpler filter. It RANSAC-fits the ground per scan
    instead of the voxel layer's fixed height band, which can drift into
    marking the real floor as an obstacle over a long run.
    """
    return Node(
        package='pepper_slam',
        executable='cloud_range_filter',
        name='points_costmap_filter',
        output='screen',
        parameters=[PEPPER_SLAM_CONFIG, {'use_sim_time': use_sim_time}],
        # It only listens, and only for base_footprint <- l2lidar_frame, which
        # is all /tf_static: the joint tree on /tf is dead weight here.
        remappings=[TF_NAV],
    )


def collision_monitor(params, log_level=None, condition=None):
    """Nav2's collision monitor: cmd_vel_raw -> stop/slowdown zones -> out."""
    return Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        arguments=_log(log_level),
        parameters=[params],
        remappings=[TF_NAV],
        condition=condition,
    )


def sensor_guard(use_sim_time, cmd_vel_in=None, log_level=None):
    """The last gate, between the collision monitor and cmd_vel, in every profile.

    Humble's collision monitor holds its last verdict when its sensor goes
    silent; this stops translation while /points_safety is lost (rotation
    stays free). Bumper contacts are naoqi_driver's: it latches any press into
    a full stop. cmd_vel_in: None keeps the configuration file's cmd_vel_safe.
    """
    overrides = {'use_sim_time': use_sim_time}
    if cmd_vel_in is not None:
        overrides['cmd_vel_in_topic'] = cmd_vel_in
    return Node(
        package='pepper_navigation',
        executable='sensor_guard',
        name='sensor_guard',
        output='screen',
        arguments=_log(log_level),
        parameters=[NAV_CONFIG, overrides],
    )


def voxel_markers(use_sim_time, log_level='warn'):
    """local_voxel_markers and global_voxel_markers, for the 3D RViz view.

    VoxelLayer publishes nav2_msgs/VoxelGrid, which RViz cannot draw; these
    turn it into a MarkerArray it can. Needs z_voxels <= 16 in the nav2 params.

    warn by default, in every profile: at info each one logs "Published N
    markers" on every grid update (~30 lines/s for the pair), and output='log'
    does not hide it -- launch still echoes stderr, where rclcpp logs, to the
    screen.
    """
    def markers(costmap):
        return Node(
            package='nav2_costmap_2d',
            executable='nav2_costmap_2d_markers',
            name=f'{costmap}_voxel_markers',
            output='log',
            arguments=_log(log_level),
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[('voxel_grid', f'/{costmap}_costmap/voxel_grid'),
                        ('visualization_marker', f'/{costmap}_costmap/voxel_markers')],
        )
    return [markers('local'), markers('global')]


def rviz(use_sim_time, rviz_config, condition, log_level=None):
    return Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=_log(log_level, ['-d', rviz_config]),
        parameters=[{'use_sim_time': use_sim_time}],
        condition=condition,
    )


def lifecycle_manager(use_sim_time, node_names, autostart=True,
                      name='lifecycle_manager_navigation', log_level=None, condition=None):
    return Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name=name,
        output='screen',
        arguments=_log(log_level),
        condition=condition,
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': autostart,
            'bond_timeout': 4.0,
            'node_names': node_names,
        }],
    )


def wait_for_map_then_start(use_sim_time, log_level=None):
    """Starts the lifecycle manager once map -> base_footprint exists.

    For the localizers whose lock waits on the world (ScanContext), where
    autostart would stall the bringup; see src/wait_for_map_then_start.cpp.
    """
    return Node(
        package='pepper_navigation',
        executable='wait_for_map_then_start',
        name='wait_for_map_then_start',
        output='screen',
        arguments=_log(log_level),
        parameters=[NAV_CONFIG, {'use_sim_time': use_sim_time}],
    )


def localization_recovery(use_sim_time, backend, log_level=None):
    """One /localization_recover entry point, the same in every profile.

    Recovering does not depend on remembering which backend is up: it forwards
    to the backend's own re-search, or (rtabmap) says there is none.
    """
    return Node(
        package='pepper_navigation',
        executable='localization_recovery',
        name='localization_recovery',
        output='screen',
        arguments=_log(log_level),
        parameters=[NAV_CONFIG, {'use_sim_time': use_sim_time, 'backend': backend}],
    )


def localization_watchdog(use_sim_time, condition, status_names=None, log_level=None):
    """Cancels navigation goals while the localizer reports itself lost.

    Only in the fastloc and pointloc profiles: amcl and rtabmap publish no
    comparable signal. It also asks for one re-search per LOST episode
    (call_recovery), replacing the localizers' removed auto_relocalize, which
    re-armed while the robot kept driving so attempts compounded. Recovery is
    still "stop and try once": in self-similar geometry the same ScanContext
    search can re-acquire the same wrong place.

    status_names: None keeps the configuration file's (fastloc's).
    """
    overrides = {'use_sim_time': use_sim_time}
    if status_names is not None:
        overrides['status_names'] = status_names
    return Node(
        package='pepper_navigation',
        executable='localization_watchdog',
        name='localization_watchdog',
        output='screen',
        arguments=_log(log_level),
        condition=condition,
        parameters=[NAV_CONFIG, overrides],
    )


def wheel_odom_tf(use_sim_time, odom_frame, condition, log_level=None):
    """Wheel odometry as odom -> base_footprint, for the odom_frame mode only.

    In the default mode the localizer owns base_footprint's parent, and a
    second one would split the tree.
    """
    return Node(
        package='pepper_slam',
        executable='wheel_odom_tf',
        name='wheel_odom_tf',
        output='screen',
        arguments=_log(log_level),
        parameters=[PEPPER_SLAM_CONFIG,
                    {'use_sim_time': use_sim_time, 'odom_frame': odom_frame}],
        condition=condition,
    )


def tf_nav_relay(use_sim_time, condition=None):
    """/tf without Pepper's joint tree, on /tf_nav (see TF_NAV)."""
    return Node(
        package='pepper_navigation',
        executable='tf_nav_relay',
        name='tf_nav_relay',
        output='screen',
        parameters=[NAV_CONFIG, {'use_sim_time': use_sim_time}],
        condition=condition,
    )
