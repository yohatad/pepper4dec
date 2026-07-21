# RTAB-Map on a recorded bag (slam_session_*): RGB-D SLAM on Pepper's wheel
# odometry, everything read from the bag instead of the live robot.
#
# Wraps rtabmap_realsense.launch.py (unchanged) with bag-specific overrides:
#   - bag topic names (the robot launch defaults to the *_custom republished ones)
#   - odometry from the bag's TF tree (pepper_odom -> base_footprint), so no
#     visual/icp odometry node is started
#   - sim time driven by `ros2 bag play --clock`
#   - a throwaway database so recorded maps (rtabmap_march_28.db, ...) are safe
#
# Usage:
#   ros2 launch pepper_slam rtabmap_bag_test.launch.py
#   ros2 bag play <bag> --clock --topics /camera/color/image_raw \
#       /camera/aligned_depth_to_color/image_raw /camera/color/camera_info \
#       /tf /tf_static

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_launch_dir = os.path.join(
        get_package_share_directory('pepper_slam'), 'launch')

    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_launch_dir, 'rtabmap_realsense.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'frame_id': 'base_footprint',
            'odom_frame_id': 'pepper_odom',
            'visual_odometry': 'false',
            'icp_odometry': 'false',
            # The bag has no depth aligned to color (only /camera/depth/image_rect_raw).
            # D435i depth is natively registered to the left IR imager, so pair it
            # with infra1 (grayscale) instead of color -- geometrically exact with
            # no re-registration node needed.
            'rgb_topic': '/camera/infra1/image_rect_raw',
            'depth_topic': '/camera/depth/image_rect_raw',
            'camera_info_topic': '/camera/infra1/camera_info',
            # the robot launch defaults subscribe_scan to true (Pepper's laser);
            # the bag has no /scan, so RGB-D only here
            'subscribe_scan': 'false',
            'approx_sync': 'true',
            'qos': '2',
            'database_path': '~/.ros/rtabmap_bag_test.db',
            'rtabmap_args': '--delete_db_on_start',
            'rtabmap_viz': 'true',
            'rviz': 'true',
        }.items(),
    )

    return LaunchDescription([rtabmap])
