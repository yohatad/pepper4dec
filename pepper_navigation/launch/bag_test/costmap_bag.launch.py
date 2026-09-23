"""costmap_bag.launch.py

The two Nav2 costmaps on a recorded bag, for tuning them against real sensor
data: which points get marked, and which of those are floor reflections rather
than obstacles. No controller, no planner goal, no robot -- the costmaps and
the tools that explain them.

Nodes started:
    nav2_map_server/map_server -- the 2D grid for the global costmap.
    nav2_controller/controller_server -- carries the LOCAL costmap.
    nav2_planner/planner_server -- carries the GLOBAL costmap.
    nav2_lifecycle_manager/lifecycle_manager -- brings those three up.
    nav2_costmap_2d/nav2_costmap_2d_markers x2 -- the voxel grids.
    pepper_navigation/tf_nav_relay.py -- /tf without Pepper's joint tree.
    pepper_navigation/cloud_delay.py -- see 'Timing' below.
    pepper_navigation/costmap_explain.py -- colours each point by what the
        costmap does with it (marked / ignored / below the floor).
    pepper_navigation/depth_to_cloud.py -- older bags only (depth_images:=true):
        rebuilds the RealSense cloud from the recorded depth+colour images.
    pepper_navigation/points_to_image.py -- the camera picture, for bags that
        have the cloud but no image topic.
    rviz2 -- rviz/costmap_bag_test.rviz.

Launch files included:
    pepper_slam/pepper_sensor_tf.launch.py -- the sensor rig TF.
    fast_lio/localization_l2.launch.py -- only when localize:=true.

Usage (bag that already carries map -> base_footprint, e.g. the
costmap_empty_floor_* recordings):
    ros2 launch pepper_navigation costmap_bag.launch.py \
        localize:=false depth_images:=false
    ros2 bag play <bag> --clock \
        --topics /points /camera/depth/color/points /tf /tf_nav /tf_static

    No --remap: cloud_delay.py reads the bag's own topics and publishes the
    delayed copies beside them.

Usage (bag without a pose, e.g. slam_20260823_aligned -- the defaults):
    ros2 launch pepper_navigation costmap_bag.launch.py
    ros2 bag play <bag> --clock \
        --qos-profile-overrides-path <ws>/config/play_qos.yaml \
        --read-ahead-queue-size 2000

    Nothing is drawn until FAST-LIO reports LOCKED (~25 s of replay). The
    costmaps are held inactive until then by wait_for_map_then_start, because
    local_costmap blocks in configure() while the map frame has nothing linking
    it to base_footprint.

    Play from the START to see the robot: Pepper's own model is a single
    /robot_description message at t=0, which --start-offset skips, and
    play_qos.yaml latches it so RViz still gets it if it connects later. The
    sensor rig has its own model on /sensor_rig/robot_description (a separate
    topic, so the two never compete), drawn as a second RobotModel.

Timing (the reason this file exists rather than a bag + the normal stack):
    A costmap that receives a cloud BEFORE the pose for that cloud's stamp
    goes into tf2_ros' wait-for-transform path, and Humble's tf2_ros 0.25.23
    can deadlock there (Buffer::waitForTransform vs the TF listener take two
    mutexes in opposite order). The costmap then ignores every later cloud and
    the map freezes -- silently, while everything else keeps running.

    So the costmaps here read <topic>_delayed, republished 0.3 s late by
    cloud_delay.py, while the localizer keeps reading the bag's own topics and
    publishes its pose long before that copy arrives. The delayed topics are
    substituted into a rewritten copy of nav2_params_fastloc.yaml; the file in
    config/ is untouched, so what you tune is what the robot runs.

    Do NOT play with --loop: the clock jumps back, every TF buffer is cleared
    and the costmaps lose the map frame. Relaunch instead.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 20, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

SUFFIX = '_delayed'


def delayed_params(src):
    """Copy the nav2 params with every costmap observation source delayed.

    Only the two costmaps are touched: the collision monitor keeps its own
    live topic, since it must not react 0.3 s late even on a bag.
    """
    src = os.path.expanduser(src)       # '~' is not expanded inside key:=value
    with open(src, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == 'topic' and isinstance(value, str) and not value.endswith(SUFFIX):
                    node[key] = value + SUFFIX
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for costmap in ('local_costmap', 'global_costmap'):
        walk(cfg.get(costmap, {}))
        cfg[costmap][costmap]['ros__parameters']['use_sim_time'] = True
    for server in ('controller_server', 'planner_server'):
        cfg[server]['ros__parameters']['use_sim_time'] = True
    # A unique file per launch, so two launches never overwrite each other's.
    fd, out = tempfile.mkstemp(prefix='pepper_costmap_bag_', suffix='.yaml')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f)
    return out


def servers(params, tf_nav):
    """controller_server (local costmap) and planner_server (global costmap)."""
    return [
        Node(package='nav2_controller', executable='controller_server',
             name='controller_server', output='screen', parameters=[params],
             remappings=[('cmd_vel', 'cmd_vel_raw')] + tf_nav),
        Node(package='nav2_planner', executable='planner_server',
             name='planner_server', output='screen', parameters=[params],
             remappings=tf_nav),
    ]


def generate_launch_description():
    nav = get_package_share_directory('pepper_navigation')
    slam = get_package_share_directory('pepper_slam')
    lio = get_package_share_directory('fast_lio')
    sim = {'use_sim_time': True}
    localize = LaunchConfiguration('localize')
    tf_nav = [('/tf', '/tf_nav')]

    declare = [
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(nav, 'config', 'nav2_params_fastloc.yaml'),
            description='Nav2 params to test. The costmap sections are copied with '
                        'their observation topics switched to *_delayed; the file '
                        'itself is never modified. Point it at a copy for an A/B run.'),
        DeclareLaunchArgument(
            'localize', default_value='true',
            description='Run FAST-LIO localization. false for a bag that already '
                        'carries map -> base_footprint (then also play /tf_nav).'),
        DeclareLaunchArgument(
            'depth_images', default_value='true',
            description='Build the RealSense cloud from the recorded depth+colour '
                        'images. false for a bag that records the cloud itself.'),
        DeclareLaunchArgument(
            'sensor_tf', default_value='urdf', choices=['urdf', 'yaml', 'none'],
            description="Sensor rig TF. 'urdf' also publishes the rig's model on "
                        '/sensor_rig/robot_description, which RViz draws at the '
                        "lidar; 'yaml' publishes the same transforms without a "
                        "model; 'none' for a bag whose /tf_static already holds "
                        'the rig.'),
        DeclareLaunchArgument(
            'map', default_value=os.path.join(nav, 'map', 'pepper_map_lc.yaml')),
        DeclareLaunchArgument('map_dir', default_value=os.path.join(nav, 'pcd')),
        DeclareLaunchArgument('map_pose_file', default_value='sc_pose_20260823.json'),
        DeclareLaunchArgument(
            'map_scan_dir', default_value=os.path.join(nav, 'pcd', 'sc_pcd_20260823')),
        DeclareLaunchArgument('config_file', default_value='l2_rsimu.yaml'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(nav, 'rviz', 'costmap_bag_test.rviz')),
        DeclareLaunchArgument('rviz', default_value='true'),
    ]

    # GroupAction, not a bare include: an include is evaluated in THIS file's
    # context, so the 'rviz': 'false' below would overwrite this file's own
    # 'rviz' argument and silently suppress rviz_node (as it did once).
    sensor_tf = GroupAction([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': 'true',
                          'publisher': LaunchConfiguration('sensor_tf'),
                          # 'all': a bag has no live camera driver to publish
                          # the RealSense's own internal extrinsics.
                          'scope': 'all'}.items())])

    fastloc = GroupAction([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lio, 'launch', 'localization_l2.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'config_file': LaunchConfiguration('config_file'),
            'map_dir': LaunchConfiguration('map_dir'),
            'map_pose_file': LaunchConfiguration('map_pose_file'),
            'map_scan_dir': LaunchConfiguration('map_scan_dir'),
            'rviz': 'false'}.items())], condition=IfCondition(localize))

    helpers = [
        Node(package='pepper_navigation', executable='tf_nav_relay.py',
             name='tf_nav_relay', output='screen', parameters=[sim],
             condition=IfCondition(localize)),
        Node(package='pepper_navigation', executable='depth_to_cloud.py',
             name='depth_to_cloud', output='screen',
             parameters=[{**sim, 'output': '/camera/depth/color/points'}],
             condition=IfCondition(LaunchConfiguration('depth_images'))),
        # in_prefix '': take the bag's own topics and publish delayed copies
        # beside them, so the localizer still gets the clouds immediately.
        Node(package='pepper_navigation', executable='cloud_delay.py',
             name='cloud_delay', output='screen',
             parameters=[{**sim, 'in_prefix': '', 'out_suffix': SUFFIX}]),
        Node(package='pepper_navigation', executable='costmap_explain.py',
             name='costmap_explain', output='screen',
             parameters=[{**sim, 'l2_topic': '/points' + SUFFIX,
                          'rs_topic': '/camera/depth/color/points' + SUFFIX}]),
        Node(package='pepper_navigation', executable='points_to_image.py',
             name='points_to_image', output='screen', parameters=[sim]),
    ]

    costmaps = [
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             output='screen',
             parameters=[{**sim, 'yaml_filename': LaunchConfiguration('map'),
                          'frame_id': 'map'}]),
        # The params file is a launch argument, so it is only known at launch
        # time: resolve it there and rewrite the copy then.
        OpaqueFunction(function=lambda context: servers(
            delayed_params(LaunchConfiguration('params_file').perform(context)), tf_nav)),
        # autostart only when the bag carries the pose. With FAST-LIO the map
        # frame appears only after ScanContext locks, and local_costmap BLOCKS
        # in configure() until then: under autostart it floods the log with
        # "Timed out waiting for transform ... two or more unconnected trees"
        # (map -> lio_init is published at startup, so the frame exists while
        # nothing links it to base_footprint yet) and a Ctrl+C in that state
        # leaves the servers in errorprocessing. wait_for_map_then_start.py
        # calls STARTUP the moment the transform is real -- same pattern as
        # pepper_nav2_fastloc.launch.py.
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_costmaps', output='screen',
             parameters=[{**sim, 'bond_timeout': 4.0,
                          'autostart': PythonExpression(
                              ["'", localize, "'.lower() not in ('true', '1')"]),
                          'node_names': ['map_server', 'controller_server',
                                         'planner_server']}]),
        Node(package='pepper_navigation', executable='wait_for_map_then_start.py',
             name='wait_for_map_then_start', output='screen',
             parameters=[{**sim, 'target_frame': 'map', 'source_frame': 'base_footprint',
                          'manager': '/lifecycle_manager_costmaps/manage_nodes'}],
             condition=IfCondition(localize)),
        Node(package='nav2_costmap_2d', executable='nav2_costmap_2d_markers',
             name='local_voxel_markers', output='log', parameters=[sim],
             remappings=[('voxel_grid', '/local_costmap/voxel_grid'),
                         ('visualization_marker', '/local_costmap/voxel_markers')]),
        Node(package='nav2_costmap_2d', executable='nav2_costmap_2d_markers',
             name='global_voxel_markers', output='log', parameters=[sim],
             remappings=[('voxel_grid', '/global_costmap/voxel_grid'),
                         ('visualization_marker', '/global_costmap/voxel_markers')]),
    ]

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')], parameters=[sim],
        condition=IfCondition(LaunchConfiguration('rviz')))

    return LaunchDescription(declare + [sensor_tf, fastloc] + helpers + costmaps
                             + [rviz_node])
