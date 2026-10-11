#!/usr/bin/env python3
"""The docking configuration must describe one consistent physical setup.

The numbers that make a dock work are spread over docking.yaml, docks.yaml,
the sensor mount (pepper_slam's sensor_tf.yaml) and the launch file, and a
mistake in any of them only shows on the robot, as a dock that is never seen
or a robot that backs into the wrong spot. These tests tie them together:

  * the staging pose puts the L2 0.6 m from the station's front face, and the
    bottom camera sees the whole station from there;
  * the docking server enters backwards after turning around, and ignores
    collisions with the station itself;
  * the dock database, plugin names and frames agree between the server and
    the detector, and the launch file refuses a dock pose that was never set;
  * the battery relay reports "docked" while NAOqi says plugged in, full
    battery or not.
"""

import importlib.util
import math
import os
import sys

from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
import numpy as np
import pytest
import yaml

HERE = os.path.dirname(__file__)
PACKAGE = os.path.join(HERE, '..')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PACKAGE, 'scripts'))
from dock_perception import footprint_corners, from_frame, Pose2D, StationModel  # noqa: E402
from dock_sim import bottom_camera  # noqa: E402
from naoqi_battery_state import battery_states, parse_battery  # noqa: E402
from sensor_msgs.msg import BatteryState  # noqa: E402,I100


def load(name):
    with open(os.path.join(PACKAGE, 'config', name)) as f:
        return yaml.safe_load(f)


DOCKING = load('docking.yaml')
SERVER = DOCKING['docking_server']['ros__parameters']
DOCK = SERVER['pepper_station']
DETECTOR = DOCKING['dock_detector']['ros__parameters']
MODEL = StationModel(**DETECTOR['station'])
STATION_OFFSET = -DOCK['external_detection_translation_x']


def lidar_x_in_base():
    with open(os.path.join(PACKAGE, '..', 'pepper_slam', 'config', 'sensor_tf.yaml')) as f:
        transforms = yaml.safe_load(f)['transforms']
    mount = next(t for t in transforms
                 if t['parent'] == 'base_footprint' and t['child'] == 'l2lidar_frame')
    return mount['xyz'][0]


def staging_pose(dock=Pose2D(0.0, 0.0, 0.0)):
    x, y = from_frame(DOCK['staging_x_offset'], 0.0, dock)
    return Pose2D(x, y, dock.yaw + DOCK['staging_yaw_offset'])


def test_staging_puts_the_lidar_about_0_6_m_from_the_station():
    front_face = STATION_OFFSET - MODEL.depth / 2           # along the dock heading
    lidar = DOCK['staging_x_offset'] + lidar_x_in_base()
    assert front_face - lidar == pytest.approx(0.6, abs=0.05)


def test_bottom_camera_sees_the_whole_station_from_the_staging_pose():
    dock = Pose2D(0.0, 0.0, 0.0)
    station = Pose2D(*from_frame(STATION_OFFSET, 0.0, dock), dock.yaw)
    camera = bottom_camera(staging_pose(dock))
    corners = np.vstack([footprint_corners(station, MODEL, 0.0, z) for z in (0.0, MODEL.height)])
    uv = camera.project(corners)
    assert not np.isnan(uv).any()
    assert (uv[:, 0] > 20).all() and (uv[:, 0] < 620).all()
    assert (uv[:, 1] > 20).all() and (uv[:, 1] < 460).all()


def test_server_turns_around_and_backs_onto_the_station():
    assert SERVER['dock_backwards'] is True
    assert SERVER['rotate_to_dock'] is True
    assert DOCK['use_external_detection_pose'] is True


def test_collisions_with_the_station_itself_are_ignored():
    back_edge = STATION_OFFSET + MODEL.depth / 2
    assert SERVER['controller']['dock_collision_threshold'] >= back_edge


def test_detected_pose_is_used_as_published():
    # dock_detector publishes the station's centre on the floor, yaw into the
    # station: any rotation offset would turn it.
    for axis in ('yaw', 'pitch', 'roll'):
        assert DOCK[f'external_detection_rotation_{axis}'] == 0.0
    assert DOCK['external_detection_translation_y'] == 0.0


def test_server_and_detector_agree_on_frames_dock_and_charging_signal():
    docks = load('docks.yaml')['docks']
    assert DETECTOR['dock_id'] in docks
    dock = docks[DETECTOR['dock_id']]
    assert dock['type'] in SERVER['dock_plugins']
    assert dock['frame'] == SERVER['fixed_frame'] == DETECTOR['fixed_frame']
    assert SERVER['base_frame'] == DETECTOR['base_frame']
    assert DOCK['use_battery_status'] is True and DOCK['charging_threshold'] < 1.0


def load_launch():
    path = os.path.join(PACKAGE, 'launch', 'pepper_docking.launch.py')
    spec = importlib.util.spec_from_file_location('pepper_docking_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_refuses_a_dock_pose_that_was_never_set(tmp_path):
    launch = load_launch()
    shipped = os.path.join(PACKAGE, 'config', 'docks.yaml')
    with pytest.raises(RuntimeError, match='no pose'):
        launch.check_dock(shipped, DETECTOR['dock_id'])
    measured = tmp_path / 'docks.yaml'
    measured.write_text(yaml.safe_dump(
        {'docks': {'home_dock': {'type': 'pepper_station', 'frame': 'map',
                                 'pose': [1.0, 2.0, math.pi]}}}))
    launch.check_dock(str(measured), 'home_dock')


# --------------------------------------------------------------------------
# naoqi_battery_state
# --------------------------------------------------------------------------

def diagnostics(percentage=57, plugged=False, full=False, current=-1.5):
    status = DiagnosticStatus(name='naoqi_driver_battery:Status', values=[
        KeyValue(key='Percentage', value=str(percentage)),
        KeyValue(key='Charging', value=str(plugged)),
        KeyValue(key='Fully Charged', value=str(full))])
    current_status = DiagnosticStatus(name='naoqi_driver_battery:Current',
                                      values=[KeyValue(key='Current', value=str(current))])
    other = DiagnosticStatus(name='naoqi_driver_joints:Status')
    return [other, status, current_status]


def test_battery_is_reported_as_naoqi_gives_it():
    battery = parse_battery(diagnostics(percentage=57, current=-1.5))
    state, docking = battery_states(battery)
    assert state.percentage == pytest.approx(0.57)
    assert state.current == pytest.approx(-1.5)
    assert state.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
    assert docking.current < DOCK['charging_threshold']


@pytest.mark.parametrize('full', [False, True])
def test_plugged_in_counts_as_docked_even_when_full(full):
    battery = parse_battery(diagnostics(plugged=True, full=full, current=0.0))
    state, docking = battery_states(battery)
    assert docking.current > DOCK['charging_threshold']
    expected = (BatteryState.POWER_SUPPLY_STATUS_FULL if full
                else BatteryState.POWER_SUPPLY_STATUS_CHARGING)
    assert state.power_supply_status == expected


def test_current_sign_flips_naoqi_current():
    state, _ = battery_states(parse_battery(diagnostics(current=-2.0)), current_sign=-1.0)
    assert state.current == pytest.approx(2.0)


def test_no_battery_status_means_nothing_to_publish():
    assert parse_battery([DiagnosticStatus(name='naoqi_driver_joints:Status')]) is None
