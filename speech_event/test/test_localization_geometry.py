#!/usr/bin/env python3
"""
Unit tests for the angle helpers on SoundLocalizationNode: the circular mean
used to average azimuth estimates across frames, and the azimuth-to-direction
naming used for logging and RViz labels.

Both are pure functions of their arguments — they touch no node state — so they
are called unbound (``SoundLocalizationNode.method(None, ...)``) rather than
constructing a node, which would need rclpy.init() and a live ROS graph.

The module imports pyroomacoustics at import time, so this file skips when that
isn't installed (same convention as person_detection's bag-replay test skipping
on a missing ONNX model).

Run via: colcon test --packages-select speech_event

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: Aug 16, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import numpy as np
import pytest

# Deliberately NOT pytest.importorskip: under pytest 6.2.5 (the version ament
# ships) a module-level Skipped raised during collection aborts collection for
# the whole session, which silently disables the sibling flake8/pep257 tests.
# A pytestmark skipif collects normally and skips at run time instead.
try:
    from speech_event import speech_event_localization as localization
    Node = localization.SoundLocalizationNode
except ImportError:  # pragma: no cover - exercised only where deps are absent
    localization = None
    Node = None

pytestmark = pytest.mark.skipif(
    localization is None,
    reason='speech_event deps (pyroomacoustics) not installed')


def circular_mean(angles):
    return Node.circular_mean(None, angles)


def direction_name(azimuth):
    return Node.get_direction_name(None, azimuth)


def assert_same_bearing(actual, expected):
    """Compare two bearings modulo a full turn.

    circular_mean() can legitimately return either end of the wrap for a mean
    that lands on zero (see test_mean_on_the_wrap_may_return_360), so tests
    about *direction* must not care which end came back.
    """
    difference = (actual - expected + 180.0) % 360.0 - 180.0
    assert difference == pytest.approx(0.0, abs=1e-9), \
        f'{actual} is not the same bearing as {expected}'


# ─────────────────────────────────────────────────────────────────────────────
# circular_mean
# ─────────────────────────────────────────────────────────────────────────────

def test_mean_of_identical_angles_is_that_angle():
    assert circular_mean([90.0, 90.0, 90.0]) == pytest.approx(90.0)


def test_mean_of_a_single_angle():
    assert circular_mean([42.0]) == pytest.approx(42.0)


def test_simple_mean_away_from_the_wrap():
    assert circular_mean([80.0, 100.0]) == pytest.approx(90.0)


def test_handles_the_zero_360_wraparound():
    """The whole reason for a circular mean: the arithmetic mean of 350 and 10
    is 180 — exactly backwards from the true answer of 0.
    """
    assert_same_bearing(circular_mean([350.0, 10.0]), 0.0)
    assert np.mean([350.0, 10.0]) == pytest.approx(180.0)  # what NOT to do


def test_wraparound_with_several_angles():
    assert_same_bearing(circular_mean([355.0, 5.0, 0.0]), 0.0)


def test_mean_on_the_wrap_returns_zero_not_360():
    """Regression guard for the open upper end of the range.

    For a mean landing exactly on zero, floating-point error can put arctan2's
    result a hair BELOW zero; `% 360` then yields 360.0 rather than 0.0. That
    used to leak out of this function, breaking any caller that range-checks
    with `< 360` or bins with `int(azimuth // 45)` into an 8-element table.
    """
    result = circular_mean([350.0, 10.0])
    assert result == pytest.approx(0.0, abs=1e-9)
    assert result != 360.0
    assert direction_name(result) == 'Front'


def test_result_is_always_in_zero_to_360():
    """Property: the output is normalised to a half-open range, so downstream
    direction naming and marker orientation never see a negative angle or an
    out-of-range 360.0.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        angles = rng.uniform(-720, 720, size=rng.integers(1, 6)).tolist()
        mean = circular_mean(angles)
        assert 0.0 <= mean < 360.0, angles


def test_wrap_landing_on_zero_is_never_360_across_many_inputs():
    """Property: sweep symmetric pairs straddling 0 — each averages to a
    bearing of zero and is a candidate for the 360.0 rounding, so this covers
    the fix far more broadly than the single [350, 10] case.

    The sweep stops below 90 deg on purpose: at exactly 90 the pair is
    antipodal (90 and 270), the two unit vectors cancel, and the circular mean
    is mathematically undefined — the result there is arbitrary, not a bug.
    """
    for offset in np.arange(0.5, 89.5, 0.5):
        mean = circular_mean([(-offset) % 360.0, offset])
        assert 0.0 <= mean < 360.0, offset
        # Either end of the wrap is the same bearing; only the range matters.
        assert min(mean, 360.0 - mean) == pytest.approx(0.0, abs=1e-6), offset


def test_antipodal_input_stays_in_range():
    """The degenerate case must still return something in [0, 360), even
    though which bearing it picks is arbitrary.
    """
    for pair in ([90.0, 270.0], [0.0, 180.0], [45.0, 225.0]):
        assert 0.0 <= circular_mean(pair) < 360.0, pair


def test_bins_safely_into_an_eight_element_table():
    """The concrete downstream failure the 360.0 leak would have caused."""
    table = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    for angles in ([350.0, 10.0], [355.0, 5.0, 0.0], [180.0], [90.0]):
        index = int(circular_mean(angles) // 45)
        assert 0 <= index < len(table), angles


def test_negative_input_angles_are_normalized():
    assert circular_mean([-10.0]) == pytest.approx(350.0)
    assert circular_mean([-90.0]) == pytest.approx(270.0)


def test_equivalent_angles_give_the_same_mean():
    """Property: adding full turns to an input must not change the answer."""
    assert_same_bearing(circular_mean([370.0, 10.0]), circular_mean([10.0, 10.0]))


def test_returns_a_plain_float():
    """Publishers expect a float, not a numpy scalar."""
    assert type(circular_mean([1.0, 2.0])) is float


# ─────────────────────────────────────────────────────────────────────────────
# get_direction_name
# ─────────────────────────────────────────────────────────────────────────────

def test_cardinal_directions():
    assert direction_name(0.0) == 'Front'
    assert direction_name(90.0) == 'Left'
    assert direction_name(180.0) == 'Rear'
    assert direction_name(270.0) == 'Right'


def test_diagonal_directions():
    assert direction_name(45.0) == 'Front-Left'
    assert direction_name(135.0) == 'Rear-Left'
    assert direction_name(225.0) == 'Rear-Right'
    assert direction_name(315.0) == 'Front-Right'


def test_front_wraps_around_zero():
    """Front spans 337.5-360 and 0-22.5; both ends must name it the same."""
    assert direction_name(350.0) == 'Front'
    assert direction_name(10.0) == 'Front'
    assert direction_name(359.9) == 'Front'


def test_bin_edges_are_half_open_upward():
    """Each bin is [start, end), so an exact edge belongs to the higher bin."""
    assert direction_name(22.5) == 'Front-Left'
    assert direction_name(67.5) == 'Left'
    assert direction_name(112.5) == 'Rear-Left'
    assert direction_name(337.5) == 'Front'


def test_angles_outside_zero_360_are_normalized():
    assert direction_name(450.0) == direction_name(90.0)
    assert direction_name(-90.0) == direction_name(270.0)
    assert direction_name(720.0) == direction_name(0.0)


def test_every_angle_gets_a_name():
    """Property: the bins tile the whole circle with no gaps."""
    valid = {'Front', 'Front-Left', 'Left', 'Rear-Left',
             'Rear', 'Rear-Right', 'Right', 'Front-Right'}
    for angle in np.arange(0.0, 360.0, 0.5):
        assert direction_name(float(angle)) in valid, angle


def test_naming_agrees_with_the_nearest_cardinal_bin():
    """Property: a name's own bin centre must map back to that same name."""
    centres = {
        0.0: 'Front', 45.0: 'Front-Left', 90.0: 'Left', 135.0: 'Rear-Left',
        180.0: 'Rear', 225.0: 'Rear-Right', 270.0: 'Right', 315.0: 'Front-Right',
    }
    for centre, expected in centres.items():
        # Nudge either side of the centre; the bins are 45 deg wide.
        assert direction_name(centre - 10.0) == expected, centre
        assert direction_name(centre + 10.0) == expected, centre
