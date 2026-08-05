# Look at the sensor rig in RViz. Nothing else: no LIO, no bag, no robot.
#
#   ros2 launch pepper_slam view_rig.launch.py              # rig alone
#   ros2 launch pepper_slam view_rig.launch.py model:=full  # rig + Pepper body
#
# Use this to check a mesh or a mount before committing to a mapping run --
# a wrong scale or a mis-seated camera is instantly obvious here and very hard
# to spot underneath a point cloud.
#
# WHY IT DOES NOT REUSE pepper_sensor_tf.launch.py
# That launch publishes robot_description under the /sensor_rig namespace, so
# it can never collide with a Pepper body description from the naoqi driver.
# That is right on the robot and wrong here: this is a standalone viewer, the
# collision it guards against cannot happen, and the namespace would just force
# a non-default topic into the RViz config. So this file runs its own
# robot_state_publisher on the plain /robot_description.
#
# JOINT STATES: model:=full pulls in Pepper's body, which has 48 revolute and
# continuous joints. robot_state_publisher will not publish TF for a movable
# joint it has no state for, so the body would appear collapsed at the root.
# joint_state_publisher_gui supplies them and gives you sliders. The rig alone
# is all-fixed and needs none, so it is not started in that case.

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

    model = LaunchConfiguration('model')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    gui = LaunchConfiguration('gui')
    rviz = LaunchConfiguration('rviz')

    # Both halves must be quoted into the expression as STRINGS. A bare
    # LaunchConfiguration substitutes its literal text, so 'true'/'false' would
    # be evaluated as Python identifiers and raise NameError before the launch
    # even starts.
    want_gui = PythonExpression([
        "'", model, "' == 'full' and '", gui, "'.lower() in ('true', '1')",
    ])

    urdf_file = PythonExpression([
        "'", os.path.join(pkg_share, 'urdf', ''), "' + ",
        "('pepper_display_with_rig.urdf.xacro' if '", model,
        "' == 'full' else 'pepper_sensor_rig.urdf.xacro')",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'model', default_value='rig', choices=['rig', 'full'],
            description="'rig' = sensors only (all-fixed, no GUI needed). "
                        "'full' = rig on Pepper's body -- DISPLAY ONLY, it fakes "
                        "a fixed base_footprint to base_link offset that is valid "
                        "at zero leg angles."),
        DeclareLaunchArgument(
            'rviz_cfg',
            default_value=os.path.join(pkg_share, 'rviz', 'view_rig.rviz'),
            description='RViz config. Fixed frame base_footprint, 0.1 m grid.'),
        DeclareLaunchArgument(
            'gui', default_value='true',
            description='Joint sliders for model:=full. Ignored for the rig, '
                        'which has no movable joints.'),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Set false to publish the description without opening '
                        'RViz (headless checks, or inspecting /robot_description '
                        'over SSH).'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': ParameterValue(
                    Command(['xacro ', urdf_file]), value_type=str),
                # Static viewer: no /clock, so sim time would freeze every stamp.
                'use_sim_time': False,
            }],
        ),

        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            condition=IfCondition(want_gui),
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_cfg],
            condition=IfCondition(rviz),
        ),
    ])
