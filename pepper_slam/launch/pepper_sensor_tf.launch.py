# Static TF tree for the Pepper sensor rig: Unitree L2 + RealSense.
#
# Publishes 10 transforms rooted at base_footprint:
#   base_footprint -> l2lidar_frame -> {l2lidar_frame_imu, camera_camera_link -> ...}
#
#   ros2 launch pepper_slam pepper_sensor_tf.launch.py
#
# ARGUMENTS
#   publisher   urdf (default) -- robot_state_publisher over
#               urdf/pepper_sensor_rig.urdf.xacro; also gives an RViz
#               RobotModel, so a wrong mount shows as geometry not bare axes.
#               yaml -- static_tf_publisher over config/sensor_tf.yaml.
#               ANY OTHER VALUE (e.g. none) starts neither and publishes
#               nothing -- correct for a bag that carries its own /tf_static.
#   scope       mount (default) -- only the rig edges. 'all' adds the seven
#               RealSense-internal edges, for a bag recorded without them.
#               On the robot the camera driver publishes its own, and a second
#               copy leaves whichever /tf_static lands last silently in force.
#
# The URDF is GENERATED from the YAML (quaternion to rpy, verified lossless), so
# both publishers emit identical geometry.
#
# THE GEOMETRY IS NOT EDITED HERE. config/sensor_tf.yaml is the calibration
# source of truth and carries the full provenance -- including why the
# direct_visual_lidar_calibration result was rejected as unreproducible and
# replaced with a tape measurement, and which DOF remain unverified. Read that
# before trusting any number in the rig. Regenerate the URDF after editing it.
#
# WHY THE RIG IS ITS OWN DESCRIPTION, not an extension of naoqi's pepper.urdf:
# that one roots at base_link with base_footprint hanging off the leg chain, so
# including it here would give base_footprint two parents and split the tree.
# See the URDF header.
#
# Replaying /tf from a bag is safe and usually wanted -- see
# launch/bag_test/README.md, which also covers which of publisher/scope your
# bag needs.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('pepper_slam')

    use_sim_time = LaunchConfiguration('use_sim_time')
    scope = LaunchConfiguration('scope')
    publisher = LaunchConfiguration('publisher')
    transforms_file = LaunchConfiguration('transforms_file')
    urdf_file = LaunchConfiguration('urdf_file')

    use_urdf = PythonExpression(["'", publisher, "' == 'urdf'"])
    use_yaml = PythonExpression(["'", publisher, "' == 'yaml'"])

    return LaunchDescription([
        # false, not true: this is the LIVE entry point. Every bag wrapper in
        # pepper_slam/launch/bag_test sets use_sim_time:='true' explicitly, so
        # this default only ever applies on the robot -- where 'true' leaves sim
        # time pinned at 0, so tf never resolves and nothing fuses, silently.
        # It does NOT drive 'publisher' or 'scope' -- no caller derives them.
        # Set those yourself: publisher:=none for a bag carrying its own
        # /tf_static, publisher:=urdf scope:=all for one without.
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'publisher', default_value='urdf',
            description="'urdf' (robot_state_publisher, gives an RViz "
                        "RobotModel) or 'yaml' (static_tf_publisher). Both "
                        "publish identical geometry."),
        DeclareLaunchArgument(
            'transforms_file',
            default_value=os.path.join(pkg_share, 'config', 'sensor_tf.yaml'),
            description='YAML transform list, used when publisher:=yaml. This '
                        'is the calibration source of truth.'),
        DeclareLaunchArgument(
            'scope', default_value='mount', choices=['mount', 'all'],
            description="'mount' (default) publishes only the rig transforms "
                        "nothing else provides. The RealSense driver publishes "
                        "its own internal extrinsics; duplicating them here "
                        "gives those edges two publishers and the last one "
                        "silently wins. Use 'all' only for bags recorded "
                        "without /tf_static (slam_recording*, slam_bench_run*)."),
        DeclareLaunchArgument(
            'urdf_file',
            default_value=os.path.join(pkg_share, 'urdf', 'pepper_sensor_rig.urdf.xacro'),
            description='Sensor-rig xacro, used when publisher:=urdf. Expanded with '
                        '`xacro` at launch. Rooted at '
                        'base_footprint so it never competes for that frame.'),

        # robot_description is namespaced so it cannot clash with a Pepper body
        # description published by the naoqi driver.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='pepper_sensor_tf',
            namespace='sensor_rig',
            output='screen',
            condition=IfCondition(use_urdf),
            parameters=[{
                'use_sim_time': use_sim_time,
                # scope reaches the description through xacro, not as a node
                # parameter -- robot_state_publisher has no such parameter.
                'robot_description': ParameterValue(
                    Command(['xacro ', urdf_file, ' scope:=', scope]), value_type=str),
                # Every joint is fixed, so nothing needs /joint_states and
                # nothing is published on /tf -- only /tf_static.
                'publish_frequency': 0.0,
            }],
        ),

        Node(
            package='pepper_slam',
            executable='static_tf_publisher.py',
            name='pepper_sensor_tf',
            output='screen',
            condition=IfCondition(use_yaml),
            parameters=[{
                'use_sim_time': use_sim_time,
                'transforms_file': transforms_file,
                'scope': scope,
            }],
        ),
    ])
