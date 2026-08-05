from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


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
                'depth_module.depth_profile': '640x480x30',
                'depth_module.infra_profile': '640x480x30',

                'pointcloud.enable': 'true',        # <-- disable during recording

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
