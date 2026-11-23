"""pepper_sensor_tf.launch.py

Static TF tree for the Pepper sensor rig: Unitree L2 + RealSense.

Publishes up to 10 transforms rooted at base_footprint:
    base_footprint -> l2lidar_frame -> {l2lidar_frame_imu,
                                        camera_camera_link -> ...}

Nodes started:
    robot_state_publisher/robot_state_publisher (node: pepper_sensor_tf)
        When publisher is 'urdf'.

Launch arguments:
    use_sim_time (default: "false")
    publisher (default: "urdf")
        'urdf': robot_state_publisher over urdf/pepper_sensor_rig.urdf.xacro.
        It also gives RViz a RobotModel, so a wrong mount shows as geometry
        rather than bare axes. 'none': publishes nothing, correct for a bag
        that carries its own /tf_static.
    scope (default: "mount")
        'mount': only the rig edges. 'all' adds the seven RealSense-internal
        edges, for a bag recorded without them. On the robot the camera driver
        publishes its own, and a second copy leaves whichever /tf_static lands
        last silently in force.
    urdf_file (default: <share>/urdf/pepper_sensor_rig.urdf.xacro)

Configuration:
    The geometry is not edited here. urdf/sensor_rig.xacro holds it and is
    the calibration source of truth; its header carries the provenance,
    including which DOF remain unverified, and lists the files that hold
    derived copies (test/test_rig_consistency.py checks them).

    The rig is its own description rather than an extension of naoqi's
    pepper.urdf, which roots at base_link and would give base_footprint two
    parents. See the URDF header.

Usage:
    ros2 launch pepper_slam pepper_sensor_tf.launch.py

Replaying /tf from a bag is safe and wanted. See "Bag replay" in the package
README, which also covers which of publisher/scope your bag needs.

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

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_slam')

    use_sim_time = LaunchConfiguration('use_sim_time')
    scope = LaunchConfiguration('scope')
    publisher = LaunchConfiguration('publisher')
    urdf_file = LaunchConfiguration('urdf_file')

    use_urdf = PythonExpression(["'", publisher, "' == 'urdf'"])

    return LaunchDescription([
        # false, not true: this is the LIVE entry point. Every bag wrapper in
        # pepper_slam/launch/bag_test sets use_sim_time:='true' explicitly, so
        # this default only ever applies on the robot -- where 'true' leaves sim
        # time pinned at 0, so tf never resolves and nothing fuses, silently.
        # It does NOT drive 'publisher' or 'scope' -- no caller derives them.
        # Set those yourself: publisher:=none for a bag carrying its own
        # /tf_static, publisher:=urdf scope:=all for one without.
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        # choices, so a stale publisher:=yaml (the removed static_tf_publisher
        # path) fails the launch instead of silently publishing no rig at all.
        DeclareLaunchArgument(
            'publisher', default_value='urdf', choices=['urdf', 'none'],
            description="'urdf' (robot_state_publisher, gives an RViz "
                        "RobotModel) or 'none' (the bag carries the rig)."),
        DeclareLaunchArgument(
            'scope', default_value='mount', choices=['mount', 'all'],
            description="'mount' (default) publishes only the rig transforms "
                        "nothing else provides. The RealSense driver publishes "
                        "its own internal extrinsics; duplicating them here "
                        "gives those edges two publishers and the last one "
                        "silently wins. Use 'all' only for bags recorded "
                        "without /tf_static (slam_recording*, slam_bench_run*)."),
        DeclareLaunchArgument(
            'urdf_file',
            default_value=os.path.join(pkg_share, 'urdf', 'pepper_sensor_rig.urdf.xacro'),
            description='Sensor-rig xacro, expanded with `xacro` at launch. '
                        'Rooted at base_footprint so it never competes for '
                        'that frame.'),

        # robot_description is namespaced so it cannot clash with a Pepper body
        # description published by the naoqi driver.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='pepper_sensor_tf',
            namespace='sensor_rig',
            output='screen',
            condition=IfCondition(use_urdf),
            parameters=[{
                'use_sim_time': use_sim_time,
                # scope reaches the description through xacro, not as a node
                # parameter -- robot_state_publisher has no such parameter.
                'robot_description': ParameterValue(
                    Command(['xacro ', urdf_file, ' scope:=', scope]), value_type=str),
                # Every joint is fixed, so nothing needs /joint_states and
                # nothing is published on /tf -- only /tf_static.
                'publish_frequency': 0.0,
            }],
        ),
    ])
