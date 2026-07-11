# FAST-LIO + RTAB-Map hybrid on a recorded bag: FAST-LIO supplies the odometry
# (best measured on the L2: 0.19 m closure), RTAB-Map builds the maps on top --
# 2D occupancy grid (/map) for Nav2 plus the 3D cloud/occupancy (/cloud_map,
# Grid/3D) -- with ICP proximity loop closures on /points.
#
# Frames: FAST-LIO's own odom frame is IMU-aligned (tilted ~90 deg on Pepper's
# mount). lio_map_odom_bridge publishes odom -> base_footprint plus a one-time
# gravity-leveled odom_level -> odom. RTAB-Map anchors on odom_level so its map
# frame is Z-up, which the 2D occupancy projection requires.
#
# Usage:
#   ros2 launch pepper_navigation rtabmap_fastlio_bag_test.launch.py
#   ros2 bag play <bag> --clock --topics /points /imu/data /tf_static
#
# NOTE: do NOT replay /tf -- the bag's wheel-odometry TF (pepper_odom ->
# base_footprint) would fight the bridge for base_footprint's parent.

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_launch_dir = os.path.join(
        get_package_share_directory('pepper_navigation'), 'launch')
    fast_lio_launch_dir = os.path.join(
        get_package_share_directory('fast_lio'), 'launch')

    fast_lio = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(fast_lio_launch_dir, 'mapping.launch.py')),
        launch_arguments={
            'config_file': 'l2.yaml',
            'rviz': 'false',
            'use_sim_time': 'true',
        }.items(),
    )

    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_launch_dir, 'rtabmap_realsense.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'frame_id': 'base_footprint',
            'odom_frame_id': 'odom_level',   # gravity-leveled parent from the bridge

            # odometry comes from FAST-LIO via TF; no rtabmap odometry nodes
            'visual_odometry': 'false',
            'icp_odometry': 'false',

            # lidar geometry + RGB appearance: the color image feeds the
            # bag-of-words vocabulary for GLOBAL loop-closure detection, while
            # the loop constraint itself is computed by ICP on the L2 scans
            # (Reg/Strategy 1) -- so no depth image is required, sidestepping
            # the missing aligned-depth topic and the IR dot-pattern problem.
            'depth': 'false',
            'subscribe_rgb': 'true',
            'rgb_topic': '/camera/color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'rgbd_sync': 'false',
            'subscribe_scan': 'false',
            'subscribe_scan_cloud': 'true',
            'scan_cloud_topic': '/points',

            'approx_sync': 'true',
            'qos': '2',
            'database_path': '~/.ros/rtabmap_fastlio_bag_test.db',
            # Reg/Strategy 1: ICP refinement + proximity loop closures on scans.
            # Grid/Sensor 0: occupancy from the scan cloud (not a camera).
            # Ground/obstacle split assumes Z-up map frame (hence odom_level).
            # NeighborLinkRefining + dense proximity closures: without them the
            # residual odometry drift between passes printed walls twice in the
            # 2D grid (validated via rtabmap-reprocess on the first run's db:
            # 6 -> 49 loop closures, wall duplication gone).
            'rtabmap_args': '--delete_db_on_start '
                            '--Reg/Strategy 1 '
                            '--RGBD/NeighborLinkRefining true '
                            '--RGBD/ProximityBySpace true '
                            '--RGBD/ProximityPathMaxNeighbors 10 '
                            '--Icp/VoxelSize 0.15 --Icp/PointToPlaneK 20 '
                            '--Icp/MaxCorrespondenceDistance 0.5 '
                            '--Icp/CorrespondenceRatio 0.2 '
                            '--Grid/Sensor 0 --Grid/CellSize 0.05 '
                            '--Grid/RangeMax 8.0 '
                            '--Grid/MaxGroundHeight 0.10 '
                            '--Grid/MaxObstacleHeight 1.7 '
                            '--Grid/RayTracing true '
                            '--Grid/NoiseFilteringRadius 0.15 '
                            '--Grid/NoiseFilteringMinNeighbors 3 '
                            '--Grid/3D true',
            'rtabmap_viz': 'true',
            'rviz': 'true',
            'rviz_cfg': os.path.join(
                get_package_share_directory('pepper_navigation'),
                'rviz', 'rtabmap_fastlio_mapping.rviz'),
        }.items(),
    )

    # 5 Hz display copy of the camera stream so the RViz image panel doesn't
    # compete with the SLAM subscribers for the full 30 fps raw stream
    image_throttle = Node(
        package='pepper_navigation',
        executable='image_throttle.py',
        name='image_throttle',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([fast_lio, rtabmap, image_throttle])
