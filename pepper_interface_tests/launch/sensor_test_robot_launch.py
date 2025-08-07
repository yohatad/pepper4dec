from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    robot_ip = LaunchConfiguration('robot_ip', default='172.29.111.230')
    robot_port = LaunchConfiguration('robot_port', default='9559')
    roscore_ip = LaunchConfiguration('roscore_ip', default='127.0.0.1')
    network_interface = LaunchConfiguration('network_interface', default='eth0')
    namespace = LaunchConfiguration('namespace', default='naoqi_driver')
    launch_audio_nodes = LaunchConfiguration('launch_audio_nodes', default='true')
    camera = LaunchConfiguration('camera', default='both')

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

    # Audio publisher nodes
    naoqi_audio_publisher = Node(
        package='naoqi_driver',
        executable='naoqiAudioPublisher.py',
        name='naoqiAudioPublisher',
        output='screen',
        condition=IfCondition(launch_audio_nodes)
    )

    naoqi_audio = Node(
        package='naoqi_driver',
        executable='run_naoqiAudio.sh',
        name='naoqiAudio',
        arguments=[f'--ip={robot_ip}', f'--port={robot_port}'],
        condition=IfCondition(launch_audio_nodes)
    )

    # RealSense camera
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch/rs_launch.py'
            )
        ]),
        condition=IfCondition(f"'{camera}' == 'realsense' or '{camera}' == 'both'"),
        launch_arguments={
            'color_width': '640',
            'color_height': '480',
            'color_fps': '15',
            'depth_width': '640',
            'depth_height': '480',
            'depth_fps': '15',
            'align_depth': 'true',
            'enable_sync': 'true'
        }
    )

    return LaunchDescription([
        naoqi_driver,
        naoqi_audio_publisher,
        naoqi_audio,
        realsense_launch
    ])
