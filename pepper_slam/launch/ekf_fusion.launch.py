# Fuses leveled LIO odometry (x, y, yaw) with wheel odometry's z/roll/pitch
# via robot_localization, as an alternative to lio_map_odom_bridge.py's
# flatten_base_frame hard clamp -- see ekf_lio_wheel.yaml for why.
#
# ADDITIVE, not a replacement: run this ALONGSIDE fastlio_mapping.launch.py
# or pointlio_mapping.launch.py (needs their odom<-odom TF and raw
# odom topic already flowing). Publishes /odometry/filtered only -- does not
# touch the existing TF tree (publish_tf: false in the EKF config), so it's
# safe to add without risking the working default pipeline.
#
# Usage:
#   ros2 launch pepper_slam fastlio_mapping.launch.py flatten_base_frame:=false
#   ros2 launch pepper_slam ekf_fusion.launch.py
#   ros2 bag play <bag> --clock --topics /points /imu/data /pepper_odom
#
# For Point-LIO instead:
#   ros2 launch pepper_slam pointlio_mapping.launch.py flatten_base_frame:=false
#   ros2 launch pepper_slam ekf_fusion.launch.py odom_topic:=/aft_mapped_to_init
#
# flatten_base_frame:=false on the mapping launch is deliberate here: this
# fuses FAST-LIO's/Point-LIO's own raw (undoctored) z/roll/pitch with wheel
# odometry, so the input needs to still be the real drifting estimate, not
# already clamped to zero by the other fix.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_slam')
    ekf_config = os.path.join(pkg_share, 'config', 'ekf_lio_wheel.yaml')

    odom_topic = LaunchConfiguration('odom_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_odom_topic_cmd = DeclareLaunchArgument(
        'odom_topic', default_value='/Odometry',
        description='Raw LIO odometry topic to fuse. FAST-LIO: /Odometry '
                    '(default). Point-LIO: /aft_mapped_to_init.'
    )
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='true for bag replay (--clock); false on the robot.')

    leveled_odom_node = Node(
        package='pepper_slam',
        executable='leveled_odometry_publisher.py',
        name='leveled_odometry_publisher',
        output='screen',
        parameters=[{
            'odom_topic': odom_topic,
            'output_topic': '/odometry/lio_leveled',
            'use_sim_time': use_sim_time,
        }],
    )

    pepper_odom_relabel_node = Node(
        package='pepper_slam',
        executable='pepper_odom_relabel.py',
        name='pepper_odom_relabel',
        output='screen',
        parameters=[{
            'input_topic': '/pepper_odom',
            'output_topic': '/odometry/pepper_odom_relabeled',
            'use_sim_time': use_sim_time,
        }],
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config, {'use_sim_time': use_sim_time}],
    )

    ld = LaunchDescription()
    ld.add_action(declare_odom_topic_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(leveled_odom_node)
    ld.add_action(pepper_odom_relabel_node)
    ld.add_action(ekf_node)
    return ld
