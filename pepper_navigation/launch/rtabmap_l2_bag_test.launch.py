# RTAB-Map lidar variant on a recorded bag: Unitree L2 (/points) + its IMU
# (/imu/data), no camera. rtabmap's icp_odometry tracks the pose from the
# point cloud (IMU used for gravity/initialization) and the SLAM node uses
# the scan cloud for proximity loop closures and map assembly.
#
# Wraps rtabmap_realsense.launch.py (unchanged), like rtabmap_bag_test.launch.py.
#
# Usage:
#   ros2 launch pepper_navigation rtabmap_l2_bag_test.launch.py
#   ros2 bag play <bag> --clock --topics /points /imu/data /tf /tf_static

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_launch_dir = os.path.join(
        get_package_share_directory('pepper_navigation'), 'launch')

    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_launch_dir, 'rtabmap_realsense.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'frame_id': 'base_footprint',   # bag static TF: base_footprint -> l2lidar_frame

            # lidar odometry instead of wheel TF / visual odometry
            'visual_odometry': 'false',
            'icp_odometry': 'true',
            'imu_topic': '/imu/data',
            'wait_imu_to_init': 'true',

            # no camera pipeline at all
            'depth': 'false',
            'subscribe_rgb': 'false',
            'rgbd_sync': 'false',
            'subscribe_scan': 'false',
            'subscribe_scan_cloud': 'true',
            'scan_cloud_topic': '/points',

            # ICP tuning for the L2's aggregated 11 Hz scans: voxel matched to
            # the FAST-LIO map resolution, point-to-plane on local normals
            'odom_args': '--Icp/VoxelSize 0.25 --Icp/PointToPlaneK 20 '
                         '--Icp/Epsilon 0.001 --Icp/MaxTranslation 1.0 '
                         '--Odom/ScanKeyFrameThr 0.6',

            'approx_sync': 'true',
            'qos': '2',
            'database_path': '~/.ros/rtabmap_l2_bag_test.db',
            'rtabmap_args': '--delete_db_on_start --Reg/Strategy 1 '
                            '--Icp/VoxelSize 0.25 --Icp/PointToPlaneK 20',
            'rtabmap_viz': 'true',
            'rviz': 'false',
        }.items(),
    )

    return LaunchDescription([rtabmap])
