#!/usr/bin/env python3
"""The station must be found where it really is, or not at all.

dock_perception measures how far the charging station has moved from where the
map says it is. Docking then backs onto that measurement blind, so the tests
pin both halves of the contract against scenes rendered by dock_sim's ray
caster from Pepper's real sensor poses:

  * accuracy: the lidar and the bottom camera each land within 1.5 cm and 1
    degree of the true pose, also when the station has moved, a white wall
    stands right behind it, legs stand next to it or the light is dim;
  * refusal: no station, a station moved beyond the search window, an object
    of the wrong size, or one mostly hidden yields None, never a guess;
  * fusion averages agreeing sensors and refuses disagreeing ones.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from dock_perception import (  # noqa: E402
    Estimate, FitLimits, from_frame, fuse, measure_with_camera, measure_with_lidar, Pose2D,
    station_from_dock, StationModel, wrap_angle)
from dock_sim import (  # noqa: E402
    bottom_camera, Box, lidar_scan, render, robot_at, Scene)

PRIOR = Pose2D(3.0, 2.0, math.radians(30.0))
XY_TOL, YAW_TOL = 0.015, math.radians(1.0)


def moved(du, dv, dyaw_deg, base=PRIOR):
    """The prior moved du along / dv across its heading and turned dyaw_deg."""
    return Pose2D(*from_frame(du, dv, base), base.yaw + math.radians(dyaw_deg))


def wall_behind(station, gap=0.005):
    x, y = from_frame(StationModel().depth / 2 + gap, 0.0, station)
    return ((x, y), (-math.cos(station.yaw), -math.sin(station.yaw)))


def legs_beside(station):
    return [(*from_frame(0.0, 0.40, station), 0.04, 0.9),
            (*from_frame(0.15, -0.42, station), 0.04, 0.9)]


def sense(scene, sensor, brightness=1.0):
    """Measure the scene from where the map puts the staging pose."""
    robot = robot_at(PRIOR, 0.6)
    if sensor == 'lidar':
        return measure_with_lidar(lidar_scan(scene, robot), PRIOR)
    camera = bottom_camera(robot)
    return measure_with_camera(render(scene, camera, brightness=brightness), camera, PRIOR)


def assert_found(fit, truth):
    assert fit is not None, 'station not found'
    assert math.hypot(fit.pose.x - truth.x, fit.pose.y - truth.y) < XY_TOL
    assert abs(wrap_angle(fit.pose.yaw - truth.yaw)) < YAW_TOL


SCENES = {
    'at_prior': (PRIOR, {}),
    'moved': (moved(0.10, -0.08, 8.0), {}),
    'moved_back_left_turned': (moved(-0.12, 0.06, -8.0), {}),
    'wall_behind': (PRIOR, {'wall': True}),
    'moved_with_wall': (moved(-0.12, 0.06, -8.0), {'wall': True}),
    'legs_beside': (PRIOR, {'legs': True}),
}


@pytest.mark.parametrize('sensor', ['lidar', 'camera'])
@pytest.mark.parametrize('name', sorted(SCENES))
def test_station_is_found_where_it_is(sensor, name):
    truth, extras = SCENES[name]
    scene = Scene(station=truth)
    if extras.get('wall'):
        scene.wall = wall_behind(truth)
    if extras.get('legs'):
        scene.legs = legs_beside(truth)
    assert_found(sense(scene, sensor), truth)


def test_camera_copes_with_dim_light():
    scene = Scene(station=PRIOR, wall=wall_behind(PRIOR))
    assert_found(sense(scene, 'camera', brightness=0.55), PRIOR)


def test_camera_ignores_white_things_outside_the_search_window():
    paper = Box(moved(-0.2, 0.9, 0.0), 0.30, 0.21, 0.001)
    scene = Scene(station=PRIOR, boxes=[paper])
    robot = robot_at(PRIOR, 0.6)
    camera = bottom_camera(robot)
    image = render(scene, camera, colours={5: (245, 245, 245)})
    assert_found(measure_with_camera(image, camera, PRIOR), PRIOR)


@pytest.mark.parametrize('sensor', ['lidar', 'camera'])
def test_no_station_is_not_found(sensor):
    assert sense(Scene(), sensor) is None


@pytest.mark.parametrize('sensor', ['lidar', 'camera'])
def test_station_moved_beyond_the_window_is_not_found(sensor):
    limits = FitLimits()
    scene = Scene(station=moved(0.0, limits.max_shift + 0.15, 0.0))
    assert sense(scene, sensor) is None


@pytest.mark.parametrize('sensor', ['lidar', 'camera'])
def test_object_of_the_wrong_size_is_not_taken_for_the_station(sensor):
    scene = Scene(station=PRIOR, model=StationModel(depth=0.37, width=0.60, height=0.075))
    assert sense(scene, sensor) is None


def test_lidar_refuses_a_station_mostly_hidden_behind_a_box():
    # A box standing in front of the station hides its front face and most of
    # its top from the L2: the fit must not guess from the sliver left over.
    blocker = Box(moved(-0.30, 0.0, 0.0), 0.08, 0.50, 0.25)
    scene = Scene(station=PRIOR, boxes=[blocker])
    assert sense(scene, 'lidar') is None


def test_lidar_picks_the_station_over_low_clutter_nearby():
    crate = Box(moved(0.05, 0.62, 15.0), 0.25, 0.25, 0.06)
    scene = Scene(station=PRIOR, boxes=[crate])
    assert_found(sense(scene, 'lidar'), PRIOR)


# --------------------------------------------------------------------------
# Fusion and dock geometry
# --------------------------------------------------------------------------

def test_fusion_weights_by_confidence():
    lidar = Estimate(Pose2D(1.00, 2.00, 0.10), sigma_xy=0.01, sigma_yaw=0.02)
    camera = Estimate(Pose2D(1.02, 2.00, 0.12), sigma_xy=0.02, sigma_yaw=0.02)
    fused = fuse([lidar, camera])
    assert fused.x == pytest.approx(1.004, abs=1e-6)     # 4:1 weights
    assert fused.yaw == pytest.approx(0.11, abs=1e-6)


def test_fusion_averages_headings_across_the_wrap():
    a = Estimate(Pose2D(0.0, 0.0, math.pi - 0.01), 0.01, 0.02)
    b = Estimate(Pose2D(0.0, 0.0, -math.pi + 0.01), 0.01, 0.02)
    assert abs(wrap_angle(fuse([a, b]).yaw - math.pi)) < 1e-9


@pytest.mark.parametrize('offset', [(0.06, 0.0, 0.0), (0.0, 0.0, 0.15)])
def test_fusion_refuses_disagreeing_sensors(offset):
    a = Estimate(Pose2D(1.0, 2.0, 0.1), 0.01, 0.02)
    b = Estimate(Pose2D(1.0 + offset[0], 2.0 + offset[1], 0.1 + offset[2]), 0.01, 0.02)
    assert fuse([a, b]) is None


def test_fusion_passes_a_single_sensor_through():
    only = Estimate(Pose2D(1.0, 2.0, 0.1), 0.01, 0.02)
    fused = fuse([only])
    assert (fused.x, fused.y, fused.yaw) == pytest.approx((1.0, 2.0, 0.1))
    assert fuse([]) is None


def test_station_lies_ahead_of_the_docked_pose_along_the_dock_heading():
    dock = Pose2D(1.0, 2.0, math.pi / 2)
    station = station_from_dock(dock, 0.25)
    assert (station.x, station.y, station.yaw) == pytest.approx((1.0, 2.25, math.pi / 2))
    assert np.isfinite([station.x, station.y]).all()
