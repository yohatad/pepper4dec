from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([

        # ------------------------------------------------------------
        # 1) Launch RealSense (WITH publish_tf := false)
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
                'base_frame_id': 'camera_camera_link',

                # Alignment and sync
                'align_depth.enable': 'true',
                'enable_sync': 'true',
                'accelerate_gpu_with_glsl': 'false',

                # IMPORTANT: We now publish ALL TFs manually
                'publish_tf': 'false',
                'tf_publish_rate': '0.0',

                # Streams
                'enable_color': 'true',
                'enable_depth': 'true',
                'enable_accel': 'true',
                'enable_gyro': 'true',
                'unite_imu_method': '2',
                'hold_back_imu_for_frames': 'true',

                'rgb_camera.color_profile': '640x480x15',
                'depth_module.depth_profile': '640x480x15',

                # ---------------- QoS (Wi-Fi SAFE) ----------------
                'color_qos': 'SENSOR_DATA',
                'color_info_qos': 'SENSOR_DATA',

                'depth_qos': 'SENSOR_DATA',
                'depth_info_qos': 'SENSOR_DATA',

                'infra1_qos': 'SENSOR_DATA',
                'infra1_info_qos': 'SENSOR_DATA',

                'infra2_qos': 'SENSOR_DATA',
                'infra2_info_qos': 'SENSOR_DATA',

                'accel_qos': 'SENSOR_DATA',
                'accel_info_qos': 'SENSOR_DATA',

                'gyro_qos': 'SENSOR_DATA',
                'gyro_info_qos': 'SENSOR_DATA',
            }.items(),
        ),

        # ------------------------------------------------------------
        # 2) Pepper → RealSense mount transform
        # ------------------------------------------------------------
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='pepper_to_realsense_tf',
            arguments=[
                '0', '0', '0.04',           # translation
                '0', '0', '0',              # rotation (RPY -> quaternion not needed here)
                'CameraTop_frame',
                'camera_camera_link'
            ]
        ),

        # ------------------------------------------------------------
        # 3) RealSense Internal TF Tree (from measured values)
        # ------------------------------------------------------------

        # CAMERA COLOR FRAME
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_camera_link_to_color_frame',
            arguments=[
                '-0.0002168898', '0.0149532212', '0.0000253627',
                '0.0008947', '-0.0011830', '0.0035573', '0.9999925',
                'camera_camera_link',
                'camera_color_frame'
            ]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='color_frame_to_color_optical',
            arguments=[
                '0', '0', '0',
                '-0.5', '0.5', '-0.5', '0.5',
                'camera_color_frame',
                'camera_color_optical_frame'
            ]
        ),

        # CAMERA DEPTH FRAME
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_camera_link_to_depth_frame',
            arguments=[
                '0', '0', '0',
                '0', '0', '0', '1',
                'camera_camera_link',
                'camera_depth_frame'
            ]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='depth_frame_to_depth_optical',
            arguments=[
                '0', '0', '0',
                '-0.5', '0.5', '-0.5', '0.5',
                'camera_depth_frame',
                'camera_depth_optical_frame'
            ]
        ),

        # IMU: ACCEL
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_camera_link_to_accel_frame',
            arguments=[
                '-0.01174', '-0.00552', '0.00510',
                '0', '0', '0', '1',
                'camera_camera_link',
                'camera_accel_frame'
            ]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='accel_frame_to_optical',
            arguments=[
                '0', '0', '0',
                '-0.5', '0.5', '-0.5', '0.5',
                'camera_accel_frame',
                'camera_accel_optical_frame'
            ]
        ),

        # IMU: GYRO
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_camera_link_to_gyro_frame',
            arguments=[
                '-0.01174', '-0.00552', '0.00510',
                '0', '0', '0', '1',
                'camera_camera_link',
                'camera_gyro_frame'
            ]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='gyro_frame_to_gyro_optical',
            arguments=[
                '0', '0', '0',
                '-0.5', '0.5', '-0.5', '0.5',
                'camera_gyro_frame',
                'camera_gyro_optical_frame'
            ]
        ),

        # IMU FRAME (gyro → imu_frame)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='gyro_to_imu_frame',
            arguments=[
                '0', '0', '0',
                '0', '0', '0', '1',
                'camera_gyro_frame',
                'camera_imu_frame'
            ]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='imu_frame_to_imu_optical',
            arguments=[
                '0', '0', '0',
                '-0.5', '0.5', '-0.5', '0.5',
                'camera_imu_frame',
                'camera_imu_optical_frame'
            ]
        ),

    ])
