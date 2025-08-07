from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    robot_ip = LaunchConfiguration('robot_ip', default='172.29.111.230')
    robot_port = LaunchConfiguration('robot_port', default='9559')
    roscore_ip = LaunchConfiguration('roscore_ip', default='127.0.0.1')
    network_interface = LaunchConfiguration('network_interface', default='eth0')
    namespace = LaunchConfiguration('namespace', default='naoqi_driver')

    # Pepper description
    pepper_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('pepper_description'),
                'launch/pepper_upload.launch.py'
            )
        ])
    )

    # Pepper control
    pepper_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('pepper_control'),
                'launch/pepper_control_trajectory.launch.py'
            )
        ])
    )

    # Naoqi DCM driver
    naoqi_dcm_driver = Node(
        package='naoqi_dcm_driver',
        executable='naoqi_dcm_driver',
        name='naoqi_dcm_driver',
        output='screen',
        parameters=[
            os.path.join(
                get_package_share_directory('pepper_dcm_bringup'),
                'config/pepper_dcm.yaml'
            ),
            os.path.join(
                get_package_share_directory('pepper_control'),
                'config/pepper_trajectory_control.yaml'
            ),
            {
                'RobotIP': robot_ip,
                'RobotPort': robot_port,
                'DriverBrokerIP': roscore_ip,
                'network_interface': network_interface,
                'Prefix': 'pepper_dcm',
                'motor_groups': 'Body',
                'use_dcm': False,
                'max_stiffness': 0.9
            }
        ]
    )

    # Naoqi driver node
    naoqi_driver = Node(
        package='naoqi_driver',
        executable='naoqi_driver_node',
        name=namespace,
        output='screen',
        arguments=[
            '--qi-url', f'tcp://{robot_ip}:{robot_port}',
            '--roscore_ip', roscore_ip,
            '--network_interface', network_interface,
            '--namespace', namespace
        ]
    )

    return LaunchDescription([
        pepper_description_launch,
        pepper_control_launch,
        naoqi_dcm_driver,
        naoqi_driver
    ])
