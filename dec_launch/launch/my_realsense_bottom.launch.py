from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


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

                'pointcloud.enable': 'false',        # <-- disable during recording

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

        # ------------------------------------------------------------
        # 2) L2 LiDAR → RealSense D435i mount transform  (CAD-measured)
        #
        #  Parent : l2lidar_frame  (Z = forward, X ≈ up-right, Y ≈ right-down;
        #                           established by l2lidar.launch.py)
        #  Child  : camera_camera_link  (back-center of D435i housing)
        #
        #  Translation from L2 LiDAR bottom-center to camera back-center.
        #  CAD measurement is in base_footprint axes (X=fwd, Y=left, Z=up):
        #      X =  0.00062 m  (0.62 mm forward)
        #      Y =  0.00000 m  (0 mm lateral)
        #      Z =  0.05084 m  (50.84 mm up — camera sits on top of LiDAR)
        #
        #  Because l2lidar_frame is tilted ~24° off vertical, those values
        #  are rotated into l2lidar_frame coords (R_BL^T * t_base):
        #      --x =  0.04644 m   --y = -0.02069 m   --z = 0.00040 m
        #
        #  Rotation: inverse of base_footprint→l2lidar_frame rotation (from
        #  l2lidar.launch.py), so that camera_camera_link is axis-aligned
        #  with base_footprint (X=forward, Y=left, Z=up).
        #  Replace translation + rotation with extrinsic calibration result.
        # ------------------------------------------------------------
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='l2lidar_to_realsense_tf',
            arguments=[
                '--x',  '0.066236',   # (0.069807 + 0.062665) / 2
                '--y', '-0.023465',   # (-0.020211 + -0.026718) / 2
                '--z',  '-0.026186',  # (-0.025706 + -0.026665) / 2
                '--qx', '0.680662',
                '--qy', '-0.150057',
                '--qz', '0.701193',
                '--qw', '0.149946',
                '--frame-id',       'l2lidar_frame',
                '--child-frame-id', 'camera_camera_link',
            ]
        ),

        # ------------------------------------------------------------
        # 3) RealSense Internal TF Tree -- NOT needed, kept only as a
        #    reference of the measured values. publish_tf:=true above
        #    already makes realsense2_camera broadcast this exact chain
        #    (camera_camera_link -> color/depth/imu frames -> optical
        #    frames) internally from the device's own factory calibration.
        #    Uncommenting this block would double-publish the same frames
        #    and fight the internal broadcaster. If you ever do need it
        #    standalone, note it uses the OLD positional argument form
        #    ('x','y','z',...,'frame_id','child_frame_id') which this
        #    Humble static_transform_publisher does NOT accept -- it would
        #    need converting to --x/--y/--z/.../--frame-id/--child-frame-id
        #    first (see the tf published in section 2 for the syntax).
        # ------------------------------------------------------------

        # # CAMERA COLOR FRAME
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='camera_link_to_color_frame',
        #     arguments=[
        #         '-0.0002168898', '0.0149532212', '0.0000253627',
        #         '0.0008947', '-0.0011830', '0.0035573', '0.9999925',
        #         'camera_camera_link',
        #         'camera_color_frame'
        #     ]
        # ),

        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='color_frame_to_color_optical',
        #     arguments=[
        #         '0', '0', '0',
        #         '-0.5', '0.5', '-0.5', '0.5',
        #         'camera_color_frame',
        #         'camera_color_optical_frame'
        #     ]
        # ),

        # # CAMERA DEPTH FRAME
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='camera_link_to_depth_frame',
        #     arguments=[
        #         '0', '0', '0',
        #         '0', '0', '0', '1',
        #         'camera_camera_link',
        #         'camera_depth_frame'
        #     ]
        # ),

        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='depth_frame_to_depth_optical',
        #     arguments=[
        #         '0', '0', '0',
        #         '-0.5', '0.5', '-0.5', '0.5',
        #         'camera_depth_frame',
        #         'camera_depth_optical_frame'
        #     ]
        # ),

        # # IMU: ACCEL
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='camera_link_to_accel_frame',
        #     arguments=[
        #         '-0.01174', '-0.00552', '0.00510',
        #         '0', '0', '0', '1',
        #         'camera_camera_link',
        #         'camera_accel_frame'
        #     ]
        # ),

        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='accel_frame_to_optical',
        #     arguments=[
        #         '0', '0', '0',
        #         '-0.5', '0.5', '-0.5', '0.5',
        #         'camera_accel_frame',
        #         'camera_accel_optical_frame'
        #     ]
        # ),

        # # IMU: GYRO
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='camera_link_to_gyro_frame',
        #     arguments=[
        #         '-0.01174', '-0.00552', '0.00510',
        #         '0', '0', '0', '1',
        #         'camera_camera_link',
        #         'camera_gyro_frame'
        #     ]
        # ),

        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='gyro_frame_to_gyro_optical',
        #     arguments=[
        #         '0', '0', '0',
        #         '-0.5', '0.5', '-0.5', '0.5',
        #         'camera_gyro_frame',
        #         'camera_gyro_optical_frame'
        #     ]
        # ),

        # # IMU FRAME (gyro → imu_frame)
        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='gyro_to_imu_frame',
        #     arguments=[
        #         '0', '0', '0',
        #         '0', '0', '0', '1',
        #         'camera_gyro_frame',
        #         'camera_imu_frame'
        #     ]
        # ),

        # Node(
        #     package='tf2_ros',
        #     executable='static_transform_publisher',
        #     name='imu_frame_to_imu_optical',
        #     arguments=[
        #         '0', '0', '0',
        #         '-0.5', '0.5', '-0.5', '0.5',
        #         'camera_imu_frame',
        #         'camera_imu_optical_frame'
        #     ]
        # ),

    ])
