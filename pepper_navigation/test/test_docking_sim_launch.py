#!/usr/bin/env python3
"""Docking end to end, against a simulated Pepper and charging station.

Runs the real pepper_docking.launch.py (docking server, dock_detector,
naoqi_battery_state) next to docking_world.py, which simulates the robot, its
L2 and bottom camera, and a station that sits 8 cm and 5 degrees away from
where the dock database says. The robot starts where Nav2 would leave it, at
the staging pose computed from the database. The test then checks that:

  * DockRobot succeeds, which in the simulation means the robot touched the
    station's TRUE contacts (within 2 cm and 5.7 degrees: docking_world.py),
    and it did so square, within 2 degrees, facing away from the station: the
    detection, the 180 degree turn and the blind backward approach all worked;
  * UndockRobot drives it back off the station.

Only the docking server's collision checking is off, as there is no costmap.

Safety: the docking server publishes cmd_vel_raw, which a live robot's
collision monitor would execute. The test picks a free ROS domain unless
ROS_DOMAIN_ID is set, and refuses to send a goal while any node outside the
test is visible.

Run via: colcon test --packages-select pepper_navigation
"""

import math
import os
import shutil
import sys
import tempfile
import time
import unittest

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing.actions
import pytest
import yaml

if 'ROS_DOMAIN_ID' not in os.environ:
    import domain_coordinator
    _domain = domain_coordinator.domain_id()
    os.environ['ROS_DOMAIN_ID'] = str(_domain.__enter__())

import rclpy  # noqa: E402,I100
from rclpy.action import ActionClient  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402,I100
from opennav_docking_msgs.action import DockRobot, UndockRobot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
from dock_perception import from_frame, Pose2D, station_from_dock  # noqa: E402

TEST_NODES = {'docking_server', 'lifecycle_manager_docking', 'dock_detector',
              'naoqi_battery_state', 'docking_world', 'docking_sim_test',
              'launch_ros', '_ros2cli_daemon'}

CONFIG = os.path.join(get_package_share_directory('pepper_navigation'), 'config')
with open(os.path.join(CONFIG, 'docking.yaml')) as f:
    PARAMS = yaml.safe_load(f)
DOCK_PARAMS = PARAMS['docking_server']['ros__parameters']['pepper_station']
STATION_OFFSET = -DOCK_PARAMS['external_detection_translation_x']

# Where the dock database says Pepper docks, and where the station really is:
# moved along / across its heading (m) and turned (deg). DOCKING_SIM_SHIFT
# overrides it, e.g. DOCKING_SIM_SHIFT="0.0,0.15,10" to try a harder case.
DOCK = Pose2D(2.0, 1.0, math.radians(120.0))
EXPECTED = station_from_dock(DOCK, STATION_OFFSET)
SHIFT = [float(v) for v in os.environ.get('DOCKING_SIM_SHIFT', '0.05,-0.12,8').split(',')]
STATION = Pose2D(*from_frame(SHIFT[0], SHIFT[1], EXPECTED), EXPECTED.yaw + math.radians(SHIFT[2]))
TRUE_DOCKED = Pose2D(*from_frame(-STATION_OFFSET, 0.0, STATION), STATION.yaw)
STAGING = Pose2D(*from_frame(DOCK_PARAMS['staging_x_offset'], 0.0, DOCK), DOCK.yaw)


@pytest.mark.launch_test
def generate_test_description():
    workdir = tempfile.mkdtemp(prefix='docking_sim_')
    PARAMS['docking_server']['ros__parameters']['controller']['use_collision_detection'] = False
    params_file = os.path.join(workdir, 'docking.yaml')
    with open(params_file, 'w') as f:
        yaml.safe_dump(PARAMS, f)
    dock_database = os.path.join(workdir, 'docks.yaml')
    with open(dock_database, 'w') as f:
        yaml.safe_dump({'docks': {'home_dock': {'type': 'pepper_station', 'frame': 'map',
                                                'pose': [DOCK.x, DOCK.y, DOCK.yaw]}}}, f)
    docking = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('pepper_navigation'), 'launch',
            'pepper_docking.launch.py')),
        launch_arguments={'params_file': params_file, 'dock_database': dock_database}.items())
    world = ExecuteProcess(
        cmd=[sys.executable, os.path.join(HERE, 'docking_world.py'), '--ros-args',
             '-p', f'station:=[{STATION.x}, {STATION.y}, {STATION.yaw}]',
             '-p', f'station_offset:={STATION_OFFSET}',
             '-p', f'start:=[{STAGING.x}, {STAGING.y}, {STAGING.yaw}]'],
        output='screen')
    return launch.LaunchDescription([
        docking, world, launch_testing.actions.ReadyToTest()]), {'workdir': workdir}


class TestDockingSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('docking_sim_test')
        cls.pose = cls.detected = None
        cls.node.create_subscription(PoseStamped, '/sim/robot_pose',
                                     lambda msg: setattr(cls, 'pose', msg.pose), 10)
        cls.node.create_subscription(PoseStamped, '/detected_dock_pose',
                                     lambda msg: setattr(cls, 'detected', msg.pose), 10)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def wait_for(self, future, timeout):
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        self.assertTrue(future.done(), 'timed out')
        return future.result()

    @staticmethod
    def pose2d(pose):
        q = pose.orientation
        return Pose2D(pose.position.x, pose.position.y, 2.0 * math.atan2(q.z, q.w))

    def robot(self):
        return self.pose2d(self.pose)

    def detection_report(self):
        """How far the last detection was from the true station, for messages."""
        if self.detected is None:
            return 'no detection received'
        d = self.pose2d(self.detected)
        return (f'detection off by {100 * math.hypot(d.x - STATION.x, d.y - STATION.y):.1f} cm, '
                f'{math.degrees(math.remainder(d.yaw - STATION.yaw, 2 * math.pi)):.1f} deg')

    def assert_only_test_nodes(self):
        self.spin_for(2.0)
        # rclpy's TransformListener adds a helper node per listener.
        foreign = {name for name in self.node.get_node_names()
                   if name not in TEST_NODES and not name.startswith('transform_listener_impl')}
        self.assertFalse(foreign, f'nodes outside the test are on this ROS domain '
                         f'({os.environ["ROS_DOMAIN_ID"]}): {sorted(foreign)}. Refusing to '
                         'send a dock goal that a live robot could execute.')

    def test_docks_on_the_moved_station_and_undocks(self):
        self.assert_only_test_nodes()
        dock = ActionClient(self.node, DockRobot, 'dock_robot')
        self.assertTrue(dock.wait_for_server(timeout_sec=60.0), 'docking server not up')
        self.spin_for(3.0)                            # let the detector see the station

        goal = DockRobot.Goal(use_dock_id=True, dock_id='home_dock',
                              navigate_to_staging_pose=False)
        handle = self.wait_for(dock.send_goal_async(goal), 10.0)
        self.assertTrue(handle.accepted)
        result = self.wait_for(handle.get_result_async(), 120.0).result
        self.assertTrue(result.success, f'docking failed, error {result.error_code}')

        robot = self.robot()
        miss = math.hypot(robot.x - TRUE_DOCKED.x, robot.y - TRUE_DOCKED.y)
        facing = math.remainder(robot.yaw - (TRUE_DOCKED.yaw + math.pi), 2 * math.pi)
        report = (f'docked {100 * miss:.1f} cm from the contacts, '
                  f'{math.degrees(facing):.1f} deg off; {self.detection_report()}')
        print(report)
        self.assertLess(miss, 0.025, report)     # the contacts register within 2 cm
        self.assertLess(abs(facing), math.radians(2.0), report)

        undock = ActionClient(self.node, UndockRobot, 'undock_robot')
        self.assertTrue(undock.wait_for_server(timeout_sec=10.0))
        handle = self.wait_for(undock.send_goal_async(UndockRobot.Goal(
            dock_type='pepper_station')), 10.0)
        result = self.wait_for(handle.get_result_async(), 60.0).result
        self.assertTrue(result.success, f'undocking failed, error {result.error_code}')
        robot = self.robot()
        self.assertGreater(math.hypot(robot.x - TRUE_DOCKED.x, robot.y - TRUE_DOCKED.y), 0.4)


@launch_testing.post_shutdown_test()
class TestCleanup(unittest.TestCase):

    def test_remove_workdir(self, workdir):
        shutil.rmtree(workdir, ignore_errors=True)
