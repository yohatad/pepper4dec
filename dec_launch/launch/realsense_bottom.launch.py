"""
realsense_bottom.launch.py: launch the bottom-mounted RealSense with its point cloud.

The point cloud is the one this build needs; see Configuration below.

Launch files included:
    realsense2_camera/rs_launch.py — the stock driver launch, with the
    profiles, IMU settings, and filter arguments set here.

Nodes started:
    None directly; everything comes from the included driver launch.

Launch arguments:
    (none)

Configuration:
    dec_launch/config/realsense_bottom_pointcloud.yaml, passed as the driver's
    config_file and merged with HIGHER priority than the launch_arguments
    below it. This is where the point cloud is actually enabled: this
    librealsense build exposes the filter as `pointcloud__neon_.*`, so the
    generic `pointcloud.enable` argument here is a name mismatch that does
    nothing. Decimation is on at magnitude 2 (~1/4 the points), which keeps
    the Nav2 VoxelLayer and collision monitor cheap; raise it to 3-4 for a
    lighter cloud.

Prerequisites:
    A RealSense device on USB. The l2lidar_frame -> camera_camera_link static
    transform is NOT published here — it lives in pepper_slam/config/
    sensor_tf.yaml so there is exactly one owner, since two publishers of a
    latched /tf_static edge means whichever lands last silently wins.

Usage:
    ros2 launch dec_launch realsense_bottom.launch.py

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: September 8, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


# The l2lidar_frame -> camera_camera_link static transform used to be published
# here. It now lives in pepper_slam/config/sensor_tf.yaml, so there is exactly
# one owner; two publishers of a latched /tf_static edge means whichever lands
# last silently wins.
#
# CORRECTING WHAT THAT BLOCK CLAIMED: it described camera_camera_link as the
# "back-center of D435i housing". It is not. Intel puts camera_link at
# mid-depth, mid-height, and 17.5 mm off the width-centre -- a point INSIDE the
# body, 21.5 mm from the back face (realsense2_description
# _d435.urdf.xacro:54-56).
#
# The measurement it recorded was right, and is still in use:
#     L2 mounting-plate centre -> camera BACK-face centre, base_footprint axes
#     X +0.62 mm   Y 0.00 mm   Z +50.85 mm
# so is its rotation into l2lidar_frame ([0.04644, -0.02069, 0.00040]) -- the
# value now in sensor_tf.yaml reproduces that to 0.01 mm. What was wrong was
# publishing that as camera_camera_link directly, without Intel's 21.5 mm
# internal offset; the numbers it actually published were a CAD value that did
# not match its own comment either.


def generate_launch_description():

    return LaunchDescription([

        # ------------------------------------------------------------
        # 1) Launch RealSense
        # ------------------------------------------------------------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('realsense2_camera'),
                    'launch',
                    'rs_launch.py'
                ])
            ]),
            launch_arguments={
                'camera_namespace': '',
                'camera_name': 'camera',
                'base_frame_id': 'camera_link',

                'align_depth.enable': 'true',
                'enable_sync': 'true',
                'accelerate_gpu_with_glsl': 'true',

                'publish_tf': 'true',
                'tf_publish_rate': '0.0',

                'enable_color': 'true',
                'enable_depth': 'true',
                'enable_infra1': 'false',
                'enable_infra2': 'false',            # <-- disabled, redundant for VIO
                'enable_accel': 'true',
                'enable_gyro': 'true',
                'unite_imu_method': '2',

                'rgb_camera.color_profile': '640x480x30',
                # Default (BGR8) never matches the pointcloud filter's texture
                # format check (RGB8/Y8 only, see config_file below) -- the
                # cloud published with no color regardless of stream_filter.
                # NOTE: an rgb8-sourced image crashes this build's compressed
                # image transport (OpenCV(4.8.0) alloc.cpp OutOfMemoryError,
                # MEASURED 2026-09-08) -- if viewing /camera/color/image_raw
                # in RViz, set that Image display's Image Transport Hint to
                # 'raw', not 'compressed'.
                'rgb_camera.color_format': 'RGB8',
                'depth_module.depth_profile': '640x480x30',
                'depth_module.infra_profile': '640x480x30',

                # This is a NAME MISMATCH, not a toggle: this node's librealsense
                # build exposes the point-cloud filter as `pointcloud__neon_.*`,
                # not the generic `pointcloud.*` this launch arg sets, so setting
                # it here does nothing (see config_file below, which is what
                # actually turns the point cloud on/off - flip .enable there).
                'pointcloud.enable': 'true',

                # Real point-cloud enable/params live here (pointcloud__neon_.*
                # on this build) - config_file is merged with HIGHER priority
                # than the launch_arguments dict above.
                'config_file': PathJoinSubstitution([
                    FindPackageShare('dec_launch'),
                    'config',
                    'realsense_bottom_pointcloud.yaml'
                ]),

                # Decimation post-processing: downsamples the depth image (and
                # therefore /camera/depth/color/points) before it is published.
                # magnitude N -> ~1/N^2 the points; 2 gives 1/4, which is plenty
                # for the Nav2 VoxelLayer + collision monitor and keeps their CPU
                # sane. Both args are declared in rs_launch.py. Raise magnitude to
                # 3-4 if you want the cloud even lighter (coarser obstacles).
                'decimation_filter.enable': 'true',
                'decimation_filter.filter_magnitude': '2',

                # NOTE: hold_back_imu_for_frames and all *_qos/*_info_qos
                # launch arguments were removed here -- neither is a
                # declared argument in the installed rs_launch.py (checked
                # against realsense2_camera's configurable_parameters list),
                # so they were silently no-ops. If real per-topic QoS
                # control is needed (e.g. BEST_EFFORT on the image topics
                # under load), use ROS2's generic qos_overrides parameter
                # file mechanism instead -- it works regardless of what this
                # launch file exposes:
                #
                #   /**:
                #     ros__parameters:
                #       qos_overrides:
                #         /camera/color/image_raw:
                #           publisher:
                #             reliability: best_effort
            }.items(),
        ),
    ])
