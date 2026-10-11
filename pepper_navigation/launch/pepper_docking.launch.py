r"""pepper_docking.launch.py

Autonomous docking onto Pepper's official charging station with opennav_docking.

Run it next to a Nav2 profile (pepper_nav2_*.launch.py): the docking server
asks Nav2 to drive to the staging pose, then takes over. See
config/docking.yaml for the sequence and the geometry.

Nodes started:
    opennav_docking/opennav_docking (node: docking_server)
        The DockRobot / UndockRobot action servers. Velocity goes to
        cmd_vel_raw, through the profile's collision monitor like Nav2's own,
        and it reads TF from /tf_nav like the other Nav2 nodes.
    nav2_lifecycle_manager/lifecycle_manager (node: lifecycle_manager_docking)
        Configures and activates docking_server.
    pepper_navigation/dock_detector.py (node: dock_detector)
        Measures the station's real pose (L2 lidar + bottom camera) and
        publishes it on detected_dock_pose.
    pepper_navigation/naoqi_battery_state.py (node: naoqi_battery_state)
        /diagnostics -> battery_state and docking/battery_state (plugged in).

Launch arguments:
    params_file (default: config/docking.yaml)
    dock_database (default: config/docks.yaml)
        The station's docked pose in the map. Starts as a NaN placeholder;
        the launch refuses to run until it is measured.
    odom_topic (default: /localization/pose)
        The Nav2 profile's odometry: /localization/pose for fastloc and
        pointloc, /odom_lio for amcl, kissicp and rtabmap_loc.
    use_camera (default: true)
        Use the bottom camera as well as the lidar. Needs naoqi_driver started
        with bottom_camera:=true (dec_launch naoqi_driver.launch.py).

Prerequisites:
    opennav_docking built from its humble branch, which has rotate_to_dock
    (the apt package 0.0.2 does not): the launch checks for it.

Usage:
    ros2 launch pepper_navigation pepper_docking.launch.py
    ros2 action send_goal /dock_robot opennav_docking_msgs/action/DockRobot \
        "{use_dock_id: true, dock_id: home_dock}" --feedback
    ros2 action send_goal /undock_robot opennav_docking_msgs/action/UndockRobot \
        "{dock_type: pepper_station}"
"""

import glob
import math
import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml


def check_rotate_to_dock():
    """Raise unless the installed opennav_docking supports rotate_to_dock."""
    try:
        prefix = get_package_prefix('opennav_docking')
    except Exception:
        raise RuntimeError(
            'opennav_docking is not installed. Clone its humble branch into the workspace '
            '(see README.md, "Docking") and build it.')
    libraries = glob.glob(os.path.join(prefix, 'lib', 'libopennav_docking_core.so*'))
    for library in libraries:
        with open(library, 'rb') as f:
            if b'rotate_to_dock' in f.read():
                return
    raise RuntimeError(
        f'The opennav_docking in {prefix} has no rotate_to_dock: it is the apt release '
        '(0.0.2), which would back towards the station without first turning around. '
        'Build its humble branch from source (see README.md, "Docking").')


def check_dock(path, dock_id):
    """Raise unless dock_id has a measured (finite) pose in the dock database."""
    with open(path) as f:
        pose = yaml.safe_load(f)['docks'][dock_id]['pose']
    if len(pose) != 3 or not all(math.isfinite(float(v)) for v in pose):
        raise RuntimeError(
            f'{dock_id} has no pose in {path}. Measure the docked pose first '
            '(TEST_PLAN_docking.md, test 3).')


def launch_setup(context):
    params_file = LaunchConfiguration('params_file').perform(context)
    dock_database = LaunchConfiguration('dock_database').perform(context)
    with open(params_file) as f:
        params = yaml.safe_load(f)
    detector = params['dock_detector']['ros__parameters']
    dock = params['docking_server']['ros__parameters']['pepper_station']

    check_rotate_to_dock()
    check_dock(dock_database, detector['dock_id'])

    return [
        Node(
            package='opennav_docking',
            executable='opennav_docking',
            name='docking_server',
            output='screen',
            parameters=[params_file, {
                'dock_database': dock_database,
                'odom_topic': LaunchConfiguration('odom_topic'),
            }],
            remappings=[('cmd_vel', 'cmd_vel_raw'), ('/tf', '/tf_nav'),
                        ('battery_state', 'docking/battery_state')],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_docking',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='pepper_navigation',
            executable='dock_detector.py',
            name='dock_detector',
            output='screen',
            parameters=[params_file, {
                'dock_database': dock_database,
                # One number in docking.yaml: the dock plugin's translation
                # from the station's centre back to the docked base.
                'station_offset': -float(dock['external_detection_translation_x']),
                'camera.enabled': ParameterValue(LaunchConfiguration('use_camera'),
                                                 value_type=bool),
            }],
        ),
        Node(
            package='pepper_navigation',
            executable='naoqi_battery_state.py',
            name='naoqi_battery_state',
            output='screen',
            parameters=[params_file],
        ),
    ]


def generate_launch_description():
    config = os.path.join(get_package_share_directory('pepper_navigation'), 'config')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(config, 'docking.yaml'),
                              description='Docking server, detector and battery parameters.'),
        DeclareLaunchArgument('dock_database', default_value=os.path.join(config, 'docks.yaml'),
                              description="The station's docked pose in the map."),
        DeclareLaunchArgument('odom_topic', default_value='/localization/pose',
                              description="The Nav2 profile's odometry topic."),
        DeclareLaunchArgument('use_camera', default_value='true',
                              description='Use the bottom camera as well as the lidar.'),
        OpaqueFunction(function=launch_setup),
    ])
