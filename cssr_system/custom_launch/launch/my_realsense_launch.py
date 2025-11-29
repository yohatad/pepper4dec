from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch RealSense camera
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
                'align_depth.enable': 'false',
                'enable_sync': 'true',
                'accelerate_gpu_with_glsl': 'false',
                'publish_tf': 'true',
                'tf_publish_rate': '15.0',
                'diagnostics_period': '0.0',
                'rgb_camera.color_profile': '640x480x15', 
                'depth_module.depth_profile': '640x480x15',
                'enable_accel': 'false',  
                'enable_gyro': 'false',
                'unite_imu_method': '0',
                'enable_depth': 'true',  # Disable continuous depth stream
            }.items(),
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='realsense_to_pepper_tf',
            arguments=[
                '0', '0', '0.04',
                '0', '0', '0',
                'CameraTop_frame', 'camera_link'
            ]
        ),

        Node(
            package='custom_launch',  # Change to your package name
            executable='color_compressor.py',
            name='color_compressor',
            parameters=[{
                'jpeg_quality': 80,
            }],
        ),
    ])