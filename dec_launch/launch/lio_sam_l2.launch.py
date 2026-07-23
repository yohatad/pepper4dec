"""
LIO-SAM with Unitree L2 LiDAR — launch file.

Starts:
  1. l2lidar_node        — L2 driver  (/points, /imu/data, TFs)
  2. LIO-SAM stack       — imuPreintegration, imageProjection,
                           featureExtraction, mapOptimization
  3. RViz2               — optional (set visualize:=false to skip)

LIO-SAM consumes the raw L2 IMU on /imu/data.

Run:
    ros2 launch dec_launch lio_sam_l2.launch.py
    ros2 launch dec_launch lio_sam_l2.launch.py visualize:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    visualize = LaunchConfiguration("visualize", default="true")

    lio_sam_share = get_package_share_directory("lio_sam")
    l2lidar_share = get_package_share_directory("l2lidar_node")

    lio_sam_params = os.path.join(lio_sam_share, "config", "params.yaml")
    rviz_config = os.path.join(lio_sam_share, "config", "rviz2.rviz")

    return LaunchDescription([

        DeclareLaunchArgument(
            "visualize", default_value="true",
            description="Launch RViz2 alongside LIO-SAM"
        ),

        # ── 1. Unitree L2 driver (points + raw IMU + TFs) ────────────────────
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(l2lidar_share, "launch", "l2lidar.launch.py")
            )
        ),

        # ── 2. LIO-SAM nodes ──────────────────────────────────────────────────
        Node(
            package="lio_sam",
            executable="lio_sam_imuPreintegration",
            name="lio_sam_imuPreintegration",
            parameters=[lio_sam_params],
            output="screen"
        ),
        Node(
            package="lio_sam",
            executable="lio_sam_imageProjection",
            name="lio_sam_imageProjection",
            parameters=[lio_sam_params],
            output="screen"
        ),
        Node(
            package="lio_sam",
            executable="lio_sam_featureExtraction",
            name="lio_sam_featureExtraction",
            parameters=[lio_sam_params],
            output="screen"
        ),
        Node(
            package="lio_sam",
            executable="lio_sam_mapOptimization",
            name="lio_sam_mapOptimization",
            parameters=[lio_sam_params],
            output="screen"
        ),

        # ── 3. RViz2 (optional) ───────────────────────────────────────────────
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output="screen",
            condition=IfCondition(visualize)
        ),
    ])
