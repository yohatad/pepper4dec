# Global localization driven by WHEEL odometry instead of FAST-LIO.
#
# WHY: FAST-LIO odometry is good enough that the ICP correction is rarely asked
# to do real work, so a clean run says little about the localizer itself.
# /pepper_odom drifts hard, which is exactly what puts the part under test under
# load.
#
# This launch deliberately runs NEITHER FAST-LIO NOR lio_map_odom_bridge.
# /pepper_odom sits on its own tree (pepper_odom -> base_footprint) which is
# disconnected from odom -> lio_init -> base_footprint on purpose; running the
# bridge too would give base_footprint two live parents and break the tree.
# See pepper_odom_relabel.py for the full rationale.
#
# The bag's /tf IS required here (it carries pepper_odom -> base_footprint), so
# do NOT remap it away as the FAST-LIO bag launches instruct:
#
#   ros2 launch pepper_slam pepper_odom_localization_bag.launch.py
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 1000 --disable-keyboard-controls --rate 2
#
# Then seed it -- auto_initialize picks a wrong hypothesis often enough in a
# corridor that a known seed is the only way to read the result:
#   ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
#     '{header: {frame_id: "map"}, pose: {pose: {position: {x: -0.012, y: 0.022,
#       z: 0.0}, orientation: {z: -0.638350, w: 0.769746}}}}'
#
# CAVEAT: the scan is the RAW /points, which FAST-LIO would otherwise deskew
# using the IMU. Motion distortion is left in, worst while turning, so results
# here are a LOWER BOUND on the localizer's accuracy.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    slam_share = get_package_share_directory('pepper_slam')
    loc_share = get_package_share_directory('lio_localization')
    nav_share = get_package_share_directory('pepper_navigation')

    declare_map_pcd_cmd = DeclareLaunchArgument(
        'map_pcd', default_value=os.path.join(nav_share, 'pcd',
                                              'pepper_map_lc.pcd'))
    declare_kf_cmd = DeclareLaunchArgument(
        'keyframe_poses', default_value=os.path.join(
            nav_share, 'pcd', 'pepper_map_lc_poses.txt'))
    declare_params_cmd = DeclareLaunchArgument(
        'params_file', default_value=os.path.join(
            loc_share, 'config', 'localization.yaml'))
    declare_auto_cmd = DeclareLaunchArgument(
        'auto_initialize', default_value='false',
        description='Leave off and seed with /initialpose: in a corridor the '
                    'global search cannot tell the true pose from several '
                    'wrong ones that score just as well.')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_rviz_cfg_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value=os.path.join(
            loc_share, 'rviz', 'localization_debug.rviz'))

    # Rig geometry: base_footprint -> l2lidar_frame, needed to carry the scan.
    sensor_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': 'true'}.items())

    to_odom = Node(
        package='pepper_slam',
        executable='cloud_to_odom_frame.py',
        name='cloud_to_odom_frame',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'input_topic': '/points',
            'output_topic': '/cloud_in_odom',
            'target_frame': 'pepper_odom',
        }],
    )

    localization = Node(
        package='lio_localization',
        executable='global_localization',
        name='fast_lio_localization',
        output='screen',
        parameters=[LaunchConfiguration('params_file'), {
            'use_sim_time': True,
            'map_pcd': LaunchConfiguration('map_pcd'),
            'keyframe_poses': LaunchConfiguration('keyframe_poses'),
            'auto_initialize': ParameterValue(
                LaunchConfiguration('auto_initialize'), value_type=bool),
            'map_frame': 'map',
            'odom_frame': 'pepper_odom',
            'cloud_topic': '/cloud_in_odom',
            'odom_topic': '/pepper_odom',
            'scan_voxel_size': 0.1,
            'fov': 6.28,
        }],
    )

    fusion = Node(
        package='lio_localization',
        executable='transform_fusion',
        name='transform_fusion',
        output='screen',
        parameters=[LaunchConfiguration('params_file'), {
            'use_sim_time': True,
            'map_frame': 'map',
            'odom_frame': 'pepper_odom',
            'body_frame': 'base_footprint',
            'fusion_rate': 50.0,
        }],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_cfg')],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='log',
    )

    ld = LaunchDescription()
    for a in (declare_map_pcd_cmd, declare_kf_cmd, declare_params_cmd,
              declare_auto_cmd, declare_rviz_cmd, declare_rviz_cfg_cmd):
        ld.add_action(a)
    for n in (sensor_tf, to_odom, localization, fusion, rviz):
        ld.add_action(n)
    return ld
