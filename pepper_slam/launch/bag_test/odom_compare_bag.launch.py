# FAST-LIO odometry vs WHEEL odometry, side by side, on a recorded bag.
#
# Two terminals total -- this launch brings up the estimator, the comparison
# node and RViz:
#
#   ros2 launch pepper_slam odom_compare_bag.launch.py
#   ros2 bag play <bag> --clock \
#     --qos-profile-overrides-path config/play_qos.yaml \
#     --read-ahead-queue-size 1000 --disable-keyboard-controls --rate 3
#
# REPLAY /tf HERE. The other bag_test launches tell you to remap it away,
# because the bag's wheel odometry fights lio_map_odom_bridge for
# base_footprint's parent. This launch wants exactly that data: /pepper_odom is
# the thing under comparison, and the bag's /tf carries it.
#
# WHAT TO READ. The two paths start at the same point, so separation between
# them is accumulated disagreement -- but do NOT read that gap as the error.
# Wheel odometry's heading error integrates, so its path rotates away over a
# long run even when both agree on distance travelled. The trustworthy number
# is the ratio in the Odometer readout marker: PATH LENGTH is robust to heading
# error, since a wrong heading points a step the wrong way without changing its
# length. MEASURED on slam_20260823_merged: wheel 503 m, FAST-LIO 508 m
# (ratio 1.01), Point-LIO 607 m -- all of the latter's excess inside one 80 s
# window where it read 4.5x the wheel distance.
#
# Swap estimator with lio:=pointlio to compare that one instead; the comparison
# node reads /odom_lio either way.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('pepper_slam')
    bag_test = os.path.join(share, 'launch', 'bag_test')

    declare_lio_cmd = DeclareLaunchArgument(
        'lio', default_value='fastlio', choices=['fastlio', 'pointlio'],
        description='Which estimator to compare against wheel odometry.')
    declare_rviz_cmd = DeclareLaunchArgument('rviz', default_value='true')
    declare_rviz_cfg_cmd = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=os.path.join(share, 'rviz', 'odom_compare.rviz'))
    declare_min_step_cmd = DeclareLaunchArgument(
        'min_step', default_value='0.05',
        description='Metres between drawn poses. The raw streams are 100-200 Hz '
                    'and drawing every sample makes RViz crawl. Both odometers '
                    'are decimated equally, so the ratio is unaffected.')

    is_fastlio = PythonExpression(["'", LaunchConfiguration('lio'), "' == 'fastlio'"])

    # SCOPED. Without scoped=True the launch_arguments below are set in THIS
    # scope, not just inside the include -- so 'rviz': 'false' (meant to stop
    # the estimator opening its own RViz) overwrote this launch's own rviz
    # argument, and our RViz never started. The symptom is silent: the node is
    # simply never created, with nothing logged.
    fastlio = GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bag_test, 'fastlio_odometry_bag.launch.py')),
        launch_arguments={'rviz': 'false'}.items(),
        condition=IfCondition(is_fastlio))])

    pointlio = GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bag_test, 'pointlio_odometry_bag.launch.py')),
        launch_arguments={'rviz': 'false'}.items(),
        condition=IfCondition(PythonExpression(['not ', is_fastlio])))])

    compare = Node(
        package='pepper_slam',
        executable='odom_compare.py',
        name='odom_compare',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'min_step': LaunchConfiguration('min_step'),
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
    for a in (declare_lio_cmd, declare_rviz_cmd, declare_rviz_cfg_cmd,
              declare_min_step_cmd):
        ld.add_action(a)
    for n in (fastlio, pointlio, compare, rviz):
        ld.add_action(n)
    return ld
