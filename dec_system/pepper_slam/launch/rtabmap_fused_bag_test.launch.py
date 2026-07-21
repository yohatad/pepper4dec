# RTAB-Map on a recorded bag using ALL THREE sensors: RealSense RGB + aligned
# depth, the RealSense IMU, and the Unitree L2 lidar.
#
# Unlike rtabmap_l2_bag_test.launch.py (lidar only) and
# rtabmap_fastlio_bag_test.launch.py (lidar odometry via FAST-LIO, RGB for
# appearance only), this variant subscribes to the aligned depth image as well.
# bags/slam_recording does publish /camera/aligned_depth_to_color/image_raw
# (6297 msgs, 16UC1, already in camera_color_optical_frame), so the depth topic
# that earlier bags lacked is available here.
#
# ODOMETRY (odom_source arg)
#   wheel (default) -- the bag's own /pepper_odom. Measured against the other
#       stacks on slam_recording it is by far the smoothest: median step 0.35 cm
#       vs 5.74 cm for rtabmap icp_odometry, 2.99 cm for Point-LIO. Wheel odom
#       drifts over distance, which is exactly what loop closure corrects.
#   icp   -- rtabmap's icp_odometry on the L2. Measured median step 5.74 cm,
#       p99 12.52 cm, with two 44-47 cm outliers at t=168 s (5.09 m/s implied,
#       against Pepper's 0.35 m/s ceiling). Icp/MaxTranslation is capped at 0.3 m
#       below to reject those; real motion is ~3 cm/frame at ~11 Hz.
#
# Only ONE odometry source may publish base_footprint's TF parent. With
# odom_source:=wheel the bag's /tf already supplies pepper_odom -> base_footprint,
# so icp_odometry is not started.
#
# IMU: /camera/imu is raw accel+gyro with orientation_covariance[0] = -1, i.e. no
# orientation, so RTAB-Map cannot apply a gravity constraint from it directly.
# imu_filter_madgwick fuses it into /imu/filtered first. The RealSense IMU is used
# rather than the L2's /imu/data because the latter's timestamps carry a ~17 ms
# sawtooth (std 5.139 ms vs 0.001 ms) from l2_sync_rate_ms:30.
#
# Usage:
#   ros2 launch pepper_slam rtabmap_fused_bag_test.launch.py
#   ros2 launch pepper_slam rtabmap_fused_bag_test.launch.py odom_source:=icp
#
#   ros2 bag play <bag> --clock --read-ahead-queue-size 2000 \
#     --topics /points /camera/imu /tf \
#              /camera/color/image_raw /camera/color/camera_info \
#              /camera/aligned_depth_to_color/image_raw \
#              /camera/aligned_depth_to_color/camera_info \
#              /pepper_odom
#   (/tf and /pepper_odom are required for odom_source:=wheel; harmless otherwise.)

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_slam')
    pkg_launch_dir = os.path.join(pkg_share, 'launch')

    odom_source = LaunchConfiguration('odom_source')
    is_icp = ["'", odom_source, "' == 'icp'"]

    declare_args = [
        DeclareLaunchArgument(
            'odom_source', default_value='wheel', choices=['wheel', 'icp'],
            description='wheel = bag /pepper_odom (smoothest); icp = rtabmap icp_odometry on the L2',
        ),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rtabmap_viz', default_value='true'),
    ]

    # Sensor extrinsics: the bags recorded /tf_static empty, so republish them.
    sensor_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_launch_dir, 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items(),
    )

    # RealSense publishes no orientation; fuse accel+gyro so rtabmap gets gravity.
    imu_filter = Node(
        package='imu_filter_madgwick', executable='imu_filter_madgwick_node',
        name='imu_filter', output='screen',
        parameters=[{
            'use_sim_time': True,
            'use_mag': False,
            'world_frame': 'enu',
            'publish_tf': False,
        }],
        remappings=[('imu/data_raw', '/camera/imu'), ('imu/data', '/imu/filtered')],
    )

    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_launch_dir, 'rtabmap_realsense.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'frame_id': 'base_footprint',
            'odom_frame_id': PythonExpression(
                ["'odom' if ", *is_icp, " else 'pepper_odom'"]),

            # exactly one odometry source (see header)
            'visual_odometry': 'false',
            'icp_odometry': PythonExpression(["'true' if ", *is_icp, " else 'false'"]),
            'odom_topic': PythonExpression(
                ["'/odom' if ", *is_icp, " else '/pepper_odom'"]),
            'subscribe_odom_info': PythonExpression(
                ["'true' if ", *is_icp, " else 'false'"]),

            # Because odom_frame_id is set, rtabmap takes odometry from TF, so
            # these variances -- NOT /pepper_odom's message covariance -- are what
            # the pose graph uses. The defaults (0.001 / 0.01) imply a per-node
            # stddev of ~3 cm, which is wildly over-confident for a slipping
            # 3-wheel omni base: measured wheel-odometry extent on slam_recording
            # is 3.62 x 3.86 m against the LIO stacks' 3.4 x 1.7 m, i.e. ~2x yaw
            # drift on the y axis. That over-confidence made genuine loop closures
            # look like 19-24 sigma errors and got them rejected.
            'odom_tf_linear_variance': '0.01',
            'odom_tf_angular_variance': '0.05',

            # RGB + aligned depth
            'depth': 'true',
            'subscribe_rgb': 'true',
            'rgb_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'rgbd_sync': 'false',

            # lidar
            'subscribe_scan': 'false',
            'subscribe_scan_cloud': 'true',
            'scan_cloud_topic': '/points',

            # imu (gravity constraint)
            'subscribe_imu': 'true',
            'imu_topic': '/imu/filtered',
            'wait_imu_to_init': 'false',

            'approx_sync': 'true',
            'qos': '2',
            'database_path': '~/.ros/rtabmap_fused_bag_test.db',

            # Reg/Strategy 2 = visual bag-of-words proposes the loop closure,
            #   lidar ICP refines the constraint. This is the point of running
            #   all three sensors: the L2 alone cannot do appearance-based
            #   global loop closure, and the camera alone registers poorly here
            #   (only ~1689 lidar points land in the 640x480 image, 30.7% of a
            #   scan -- see LOG.md section 4.5).
            # Icp/CorrespondenceRatio 0.2, not 0.3: at 0.3 valid closures were
            #   rejected at corrRatio 0.159-0.279 -- the L2's sparse aggregated
            #   clouds barely clear that bar.
            # Icp/MaxTranslation 0.3: caps the icp_odometry outliers described
            #   in the header. Ignored when odom_source:=wheel.
            'rtabmap_args': '--delete_db_on_start '
                            '--Reg/Strategy 2 '
                            '--Reg/Force3DoF true '
                            '--Vis/MinInliers 15 '
                            '--Vis/InlierDistance 0.1 '
                            '--Kp/MaxFeatures 500 '
                            '--RGBD/LoopClosureReextractFeatures true '
                            '--RGBD/NeighborLinkRefining true '
                            '--RGBD/ProximityBySpace true '
                            '--RGBD/ProximityPathMaxNeighbors 10 '
                            '--RGBD/AngularUpdate 0.05 '
                            '--RGBD/LinearUpdate 0.05 '
                            '--Icp/VoxelSize 0.15 --Icp/PointToPlaneK 20 '
                            '--Icp/PointToPlane true '
                            '--Icp/MaxCorrespondenceDistance 0.5 '
                            '--Icp/CorrespondenceRatio 0.2 '
                            '--Icp/MaxTranslation 0.3 '
                            # OptimizeMaxError 30, not the default 3: validated
                            # with rtabmap-reprocess on a real run's database --
                            # 3 -> 1 global closure, 8 -> 1, 15 -> 3, 30 -> 9.
                            # At the default, wheel odometry's yaw drift makes
                            # true closures (systematic revisit: 108<->7, 109<->8)
                            # look like 19-24 sigma errors and all get rejected.
                            '--RGBD/OptimizeMaxError 30 '
                            # More loop-closure candidates tested per node. 39% of
                            # genuine revisits closed no loop at the defaults
                            # (MaxRetrieved 2, ProximityMaxPaths 3). Validated with
                            # rtabmap-reprocess on a real run: global 27->49,
                            # localspace 81->160. Low compute cost -- these widen
                            # the candidate search, they do NOT loosen the accept
                            # threshold (that is still OptimizeMaxError above).
                            '--Rtabmap/MaxRetrieved 4 '
                            '--RGBD/ProximityMaxPaths 5 '
                            # DetectionRate 1.5 (default 1.0): ~50% more keyframes
                            # -> finer revisit granularity. COMPUTE TRADEOFF: the
                            # earlier run was already ~0.30 s/iter with delay 0.42 s,
                            # so if playback starts dropping frames, drop this back
                            # to 1.0. Unlike the levers above this changes node
                            # creation, so it could not be pre-validated by reprocess.
                            '--Rtabmap/DetectionRate 1.5 '
                            '--Optimizer/Strategy 1 --Optimizer/Robust true '
                            # RangeMax 8 (was 15) and CellSize 0.08 (was 0.06):
                            # the mapped room is only ~4 m across, so 15 m range
                            # at 6 cm cells just inflates /cloud_map -- which is
                            # what pushed it past the DDS sample limit and killed
                            # the node mid-run.
                            '--Grid/Sensor 2 --Grid/3D true '
                            '--Grid/CellSize 0.08 --Grid/RangeMax 8.0 '
                            '--Grid/RayTracing true '
                            '--Grid/NoiseFilteringRadius 0.15 '
                            '--Grid/NoiseFilteringMinNeighbors 3',
            'rtabmap_viz': LaunchConfiguration('rtabmap_viz'),
            'rviz': LaunchConfiguration('rviz'),
            'rviz_cfg': os.path.join(pkg_share, 'rviz', 'rtabmap_fused_mapping.rviz'),
        }.items(),
    )

    return LaunchDescription(declare_args + [sensor_tf, imu_filter, rtabmap])
