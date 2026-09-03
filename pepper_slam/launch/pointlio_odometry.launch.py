# Plain Point-LIO ODOMETRY on the Pepper L2 rig, with its required static TF.
# The Point-LIO twin of fastlio_odometry.launch.py; see that header for why
# this is odometry and not SLAM, and use fastlio_lc_pgo's pointlio_lc_l2 for a
# loop-corrected map.
#
# It exists for the same reason: point_lio's mapping_l2lidar_node.launch.py
# runs lio_odom_bridge itself but never includes pepper_sensor_tf.launch.py,
# so launched alone it hangs waiting for a transform that never appears.
#
# flatten_base_frame defaults true here (false upstream): Pepper is confirmed
# flat-floor-only. Pass false to see Point-LIO's own drifting z/roll/pitch.
#
#   ros2 launch pepper_slam pointlio_odometry.launch.py
#   ros2 bag play <bag> --clock --topics /points /imu/data /tf /tf_static

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _echo_resolved(context, *args, **kwargs):
    """Print the decisions that fail silently if they are wrong.

    use_sim_time true with no /clock pins time at 0 and nothing renders or
    fuses; publisher:=none against a bag with NO /tf_static leaves the rig
    transforms missing and lio_odom_bridge simply waits. Neither prints an
    error. One line here beats bisecting either.
    """
    from launch.actions import LogInfo
    sim = LaunchConfiguration('use_sim_time').perform(context)
    pub = LaunchConfiguration('publisher', default='urdf').perform(context)
    scope = LaunchConfiguration('scope', default='mount').perform(context)
    return [LogInfo(msg='[pepper_slam] use_sim_time=%s  sensor_tf publisher=%s  scope=%s'
                        % (sim, pub, scope))]


def generate_launch_description():
    point_lio_share = get_package_share_directory('point_lio')
    pkg_share = get_package_share_directory('pepper_slam')

    rviz = LaunchConfiguration('rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    flatten_base_frame = LaunchConfiguration('flatten_base_frame')
    bridge_level_frame = LaunchConfiguration('bridge_level_frame')

    # The RealSense IMU is the permanent choice, and lio_odom_bridge hardcodes
    # the matching camera_imu_optical_frame rather than deriving it, so
    # l2lidar_node.yaml needs lidar_imu_frame:=l2lidar_frame_imu passed too.
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='l2lidar_rsimu.yaml',
        description='Point-LIO config under point_lio/config. l2lidar_rsimu.yaml '
                    'uses the RealSense IMU (default); l2lidar_node.yaml uses '
                    'the L2 s own -- see utils/L2_IMU/REPORT.md.')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_publish_map_identity_cmd = DeclareLaunchArgument(
        'publish_map_identity', default_value='true',
        description='Publish a static identity map -> odom so "map" can be used '
                    'as a fixed frame in odometry-only runs. Set false when PGO '
                    'or a localizer owns that edge.')
    # false, NOT true: this is the LIVE entry point, and 'true' on the robot
    # pins sim time at 0, so tf never resolves and nothing fuses, silently.
    # pepper_sensor_tf's publisher/scope are NOT derived from this -- on a bag
    # pass publisher:=none if it carries its own /tf_static, publisher:=urdf
    # scope:=all if it does not.
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='true for bag replay (--clock); false on the robot. '
                    'Reaches pepper_sensor_tf, Point-LIO and the odom bridge.')
    declare_flatten_base_frame_cmd = DeclareLaunchArgument(
        'flatten_base_frame', default_value='true',
        description='Zero the leveled z/roll/pitch of odom -> base_footprint '
                    'every cycle (keep x, y, yaw). Defaults true here: Pepper '
                    'is confirmed flat-floor-only.'
    )
    declare_bridge_level_frame_cmd = DeclareLaunchArgument(
        'bridge_level_frame', default_value='true',
        description='Have lio_odom_bridge publish the static odom -> '
                    'odom leveling frame. Set false when a higher layer owns '
                    'odom (e.g. PGO publishing map -> odom).'
    )

    sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'pepper_sensor_tf.launch.py')),
        launch_arguments={'use_sim_time': use_sim_time}.items())

    point_lio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(point_lio_share, 'launch', 'mapping_l2lidar_node.launch.py')),
        launch_arguments={
            'rviz': rviz,
            'config_file': LaunchConfiguration('config_file'),
            'use_sim_time': use_sim_time,
            'flatten_base_frame': flatten_base_frame,
            'bridge_level_frame': bridge_level_frame,
        }.items())

    # REP-105 says map -> odom is the loop-closure / localization correction.
    # This launch is odometry ONLY -- nothing corrects anything -- so that edge
    # is identity by definition. Publishing it costs nothing and makes 'map' a
    # usable RViz fixed frame here, so the same rviz config works whether or not
    # PGO/AMCL is running.
    #
    # MUST be false when something else owns map -> odom (pgo_map_odom_bridge,
    # AMCL, or lio_localization's transform_fusion) -- two publishers would give
    # odom two parents and split the tree.
    map_identity = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_identity',
        arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(LaunchConfiguration('publish_map_identity')),
    )

    ld = LaunchDescription()
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_flatten_base_frame_cmd)
    ld.add_action(declare_bridge_level_frame_cmd)
    # AFTER every declare: the echo reads use_sim_time.
    ld.add_action(OpaqueFunction(function=_echo_resolved))
    ld.add_action(declare_publish_map_identity_cmd)
    ld.add_action(sensor_tf_launch)
    ld.add_action(map_identity)
    ld.add_action(point_lio_launch)
    return ld
