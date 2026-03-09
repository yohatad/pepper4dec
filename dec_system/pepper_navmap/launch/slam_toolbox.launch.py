from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Path to SLAM Toolbox config
    slam_params_file = LaunchConfiguration('slam_params_file')
    
    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(
            get_package_share_directory('dec_launch'),
            'config',
            'mapper_params_online_async.yaml'
        ),
        description='Full path to SLAM Toolbox parameters'
    )
    
    # SLAM Toolbox Node
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_file],
        remappings=[
            ('/scan', '/scan')  # Your YDLidar topic
        ]
    )
    
    return LaunchDescription([
        declare_slam_params_file_cmd,
        slam_toolbox_node
    ])