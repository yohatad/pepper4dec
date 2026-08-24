# Static TF tree for the Pepper sensor rig: Unitree L2 + RealSense (colour,
# depth, IMU).
#
# WHY THIS EXISTS: bags/slam_recording, slam_recording2 and slam_bench_run* were
# all recorded with /tf_static containing ZERO messages, so the RealSense driver's
# internal extrinsics (camera_link -> color/depth/gyro/accel) and the rig mount
# transforms were never captured. Without them RTAB-Map (or any multi-sensor SLAM)
# cannot relate the camera to the lidar and refuses to start.
#
# The values below were recovered from bags/lidar_cam_calib2, which is one of the
# few bags whose /tf_static was captured (4 messages).
#
# ONE DELIBERATE DEVIATION -- l2lidar_frame -> camera_camera_link:
# the recorded value is the CAD-based initial guess. The refined result from
# direct_visual_lidar_calibration (koide3, NID-based, recorded in the header of
# FAST-LIVO2/config/unitree_l2_pepper.yaml) differs from it by 6.16 cm in
# translation -- mostly X/Z -- and 1.34 deg in rotation. The refined value is used
# here, back-composed through camera_camera_link -> camera_color_frame ->
# camera_color_optical_frame so the rest of the recorded chain still applies.
#
# TWO WAYS TO PUBLISH THE SAME 10 TRANSFORMS, pick with publisher:=
#   urdf (default) -- robot_state_publisher over urdf/pepper_sensor_rig.urdf.xacro.
#                     Conventional, and gives an RViz RobotModel so a wrong
#                     mount is visible as geometry rather than bare axes.
#   yaml           -- static_tf_publisher over config/sensor_tf.yaml.
# The URDF is GENERATED from the YAML (quaternion to rpy, verified lossless),
# so both publish identical geometry. The YAML remains the calibration source
# of truth; regenerate the URDF after editing it.
#
# The URDF is the sensor rig ONLY, rooted at base_footprint. It deliberately
# does not extend naoqi's pepper.urdf: that description roots at base_link with
# base_footprint hanging off the leg chain, so publishing it here would give
# base_footprint two parents and split the tree. See the URDF header.
#
# ONE NODE, NOT TEN: the geometry now lives in config/sensor_tf.yaml and is
# published by a single static_tf_publisher (pepper_slam). This used to spawn
# 10 tf2_ros/static_transform_publisher processes -- one per edge, each a full
# ROS node with its own executor, DDS participant and discovery traffic, to
# publish seven constants. tf2 concatenates all /tf_static publishers anyway,
# so one message carrying the whole chain is equivalent at a tenth the cost.
# To change the rig geometry, edit config/sensor_tf.yaml -- not this file.
#
# Usage (include from another launch file, or run standalone):
#   ros2 launch pepper_slam pepper_sensor_tf.launch.py
#
# REPLAYING /tf IS SAFE, and you usually want it: it carries Pepper's whole body
# chain, including CameraTop_optical_frame, which the head-camera images are
# stamped with and which resolves against nothing without it.
#
# This header, and four other launch files, used to say the opposite -- that a
# recorded wheel-odometry edge (pepper_odom -> base_footprint) would fight
# lio_map_odom_bridge for base_footprint's parent. That stopped being true at
# commit 8edd1f5, which defaulted publish_wheel_odom_tf to false
# (naoqi_driver2/src/converters/joint_state.cpp:89) for precisely that reason,
# so the edge is never recorded at all. Wheel odometry exists only as the
# /pepper_odom topic.
#
# VERIFIED 2026-08-24 on bags/slam_20260823_merged: across ALL 77080 /tf
# messages, zero transforms name base_footprint as a child, 'odom' never
# appears, and base_footprint is the sole root. The bridge attaches above it;
# the recorded tree hangs below it. Different ends; no contention.
#
# Bags recorded BEFORE 8edd1f5 may still carry the edge. Check rather than
# assume:  ros2 bag play <bag> --topics /tf & ros2 run tf2_tools view_frames

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
    transforms_file = LaunchConfiguration('transforms_file')
    urdf_file = LaunchConfiguration('urdf_file')

    use_urdf = PythonExpression(["'", publisher, "' == 'urdf'"])
    use_yaml = PythonExpression(["'", publisher, "' == 'yaml'"])

    return LaunchDescription([
        # false, not true: this is the LIVE entry point. Every bag wrapper in
        # pepper_slam/launch/bag_test sets use_sim_time:='true' explicitly, so
        # this default only ever applies on the robot -- where 'true' leaves sim
        # time pinned at 0, so tf never resolves and nothing fuses, silently.
        # It also feeds the sensor_tf scope derivation: false -> 'mount', which
        # is correct live because the RealSense driver publishes its own camera
        # edges (publishing them here too is the nondeterministic-latch problem
        # sensor_tf.yaml warns about).
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'publisher', default_value='urdf',
            description="'urdf' (robot_state_publisher, gives an RViz "
                        "RobotModel) or 'yaml' (static_tf_publisher). Both "
                        "publish identical geometry."),
        DeclareLaunchArgument(
            'transforms_file',
            default_value=os.path.join(pkg_share, 'config', 'sensor_tf.yaml'),
            description='YAML transform list, used when publisher:=yaml. This '
                        'is the calibration source of truth.'),
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
            description='Sensor-rig xacro, used when publisher:=urdf. Expanded with '
                        '`xacro` at launch. Rooted at '
                        'base_footprint so it never competes for that frame.'),

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

        Node(
            package='pepper_slam',
            executable='static_tf_publisher.py',
            name='pepper_sensor_tf',
            output='screen',
            condition=IfCondition(use_yaml),
            parameters=[{
                'use_sim_time': use_sim_time,
                'transforms_file': transforms_file,
                'scope': scope,
            }],
        ),
    ])
