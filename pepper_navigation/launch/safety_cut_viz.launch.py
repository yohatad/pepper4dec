"""safety_cut_viz.launch.py

See which L2 points the safety chain cuts, and which reach the collision
monitor, coloured in RViz. Read-only: subscribes to /points, publishes only
/safety_cut_viz/* -- never a velocity -- so it is safe with naoqi_driver up.

Nodes started:
    pepper_navigation/safety_cut_viz.py -- colours each point by the first
        stage that removes it (see the script header for the colours).
    rviz2 -- rviz/safety_cut_viz.rviz, fixed frame base_footprint.

Launch files included:
    pepper_slam/pepper_sensor_tf.launch.py -- only when sensor_tf:=true, for
        a robot where nothing else publishes base_footprint <- l2lidar_frame.
        Leave it false when a Nav2 launch is already running (it starts its
        own), or the rig gets two publishers.

Launch arguments:
    sensor_tf (default: "false")
    rviz (default: "true") -- false to run the node alone, e.g. with RViz on
        the laptop: rviz2 -d <share>/rviz/safety_cut_viz.rviz
    min_range, self_cut_range, self_cut_angle -- the points_safety_filter
        values to show (defaults: pepper_nav2_fastloc.launch.py). Live too:
        ros2 param set /safety_cut_viz min_range 0.12
    body_boxes -- set live only (default: the launch files' box):
        ros2 param set /safety_cut_viz body_boxes "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"

Usage (on the Jetson):
    ros2 launch pepper_navigation safety_cut_viz.launch.py sensor_tf:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    nav = get_package_share_directory('pepper_navigation')
    slam = get_package_share_directory('pepper_slam')

    def num(name):
        return ParameterValue(LaunchConfiguration(name), value_type=float)

    return LaunchDescription([
        DeclareLaunchArgument('sensor_tf', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('min_range', default_value='0.12'),
        DeclareLaunchArgument('self_cut_range', default_value='0.30'),
        DeclareLaunchArgument('self_cut_angle', default_value='25.0'),

        GroupAction([IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam, 'launch', 'pepper_sensor_tf.launch.py')),
            launch_arguments={'use_sim_time': 'false'}.items())],
            condition=IfCondition(LaunchConfiguration('sensor_tf'))),

        Node(package='pepper_navigation', executable='safety_cut_viz.py',
             name='safety_cut_viz', output='screen',
             parameters=[{'min_range': num('min_range'),
                          'self_cut_range': num('self_cut_range'),
                          'self_cut_angle': num('self_cut_angle')}]),

        Node(package='rviz2', executable='rviz2', name='safety_cut_rviz',
             arguments=['-d', os.path.join(nav, 'rviz', 'safety_cut_viz.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
