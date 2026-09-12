#!/usr/bin/env python3
"""The divergence guard must reject implausible poses rather than clip them.

These are the properties the guard exists for, and each one has a real failure
behind it:

  * a pose that moves faster than the base physically can is a broken estimate,
    not a fast robot, and must not reach odom -> base_footprint;
  * while rejecting, the pose must keep MOVING on wheel odometry -- freezing it
    makes obstacles stream past a robot that reports standing still;
  * coasting must be time bounded, or a permanent fault decays into unbounded
    silent dead reckoning;
  * a rejected excursion must never become the baseline the next check measures
    against, or one jump ratchets the guard along with it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from lio_odom_guard import ACCEPT, FAULT, HOLD, OdomGuard  # noqa: E402


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    """4x4 pose with a planar translation and a yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    m[:3, 3] = [x, y, z]
    return m


def guard(**kw):
    kw.setdefault('max_linear_speed', 0.7)
    kw.setdefault('max_angular_speed', 1.0)
    kw.setdefault('max_hold_duration', 3.0)
    return OdomGuard(**kw)


def test_first_pose_is_accepted_without_a_delta():
    g = guard()
    verdict, out = g.check(pose(1.0, 2.0), t=10.0)
    assert verdict == ACCEPT
    assert np.allclose(out, pose(1.0, 2.0))


def test_normal_motion_passes():
    g = guard()
    g.check(pose(0.0), t=0.0)
    # 0.3 m/s, comfortably inside Nav2's own 0.5 m/s command cap.
    verdict, _ = g.check(pose(0.03), t=0.1)
    assert verdict == ACCEPT


def test_jump_beyond_physical_bound_is_rejected():
    g = guard()
    g.check(pose(0.0), t=0.0)
    # 5 m in 100 ms = 50 m/s. Pepper cannot do this whatever the filter believes.
    verdict, _ = g.check(pose(5.0), t=0.1)
    assert verdict == HOLD
    assert 'exceeds' in g.last_reason


def test_rejected_pose_is_not_clipped_to_the_limit():
    """The whole point: discard, never saturate."""
    g = guard()
    g.check(pose(0.0), t=0.0)
    _, out = g.check(pose(5.0), t=0.1)
    # Without wheel odometry the held pose is the last good one, NOT the jump
    # scaled back to max_linear_speed * dt (which would be 0.07 m).
    assert np.allclose(out[:3, 3], [0.0, 0.0, 0.0])


def test_fast_rotation_is_rejected():
    g = guard()
    g.check(pose(yaw=0.0), t=0.0)
    verdict, _ = g.check(pose(yaw=1.5), t=0.1)  # 15 rad/s
    assert verdict == HOLD
    assert 'angular' in g.last_reason


def test_zero_velocity_cross_check_convicts_lio():
    """Wheels cannot report stationary while the base actually moves."""
    g = guard(max_linear_speed=10.0)  # speed bound deliberately out of the way
    g.check(pose(0.0), t=0.0, wheel_speed=0.0)
    verdict, _ = g.check(pose(0.05), t=0.1, wheel_speed=0.0)  # 0.5 m/s vs stopped
    assert verdict == HOLD
    assert 'stationary' in g.last_reason


def test_hold_dead_reckons_on_wheel_odometry():
    g = guard()
    g.check(pose(1.0), t=0.0, wheel_pose=pose(10.0))
    # LIO jumps; wheels advance 0.04 m over the same window.
    _, out = g.check(pose(9.0), t=0.1, wheel_pose=pose(10.0))
    assert np.allclose(out[:3, 3], [1.0, 0.0, 0.0])
    _, out = g.check(pose(9.5), t=0.2, wheel_pose=pose(10.04))
    # Anchored at 1.0 and carried forward by the wheel delta, not the jump.
    assert np.allclose(out[:3, 3], [1.04, 0.0, 0.0])


def test_dead_reckoning_is_frame_relative_not_absolute():
    """Wheel poses live in pepper_odom; only their DELTA may be borrowed."""
    g = guard()
    # Wheel origin is nowhere near the LIO origin -- disconnected trees.
    g.check(pose(0.0), t=0.0, wheel_pose=pose(100.0, 50.0))
    # Rejection starts here, anchoring the wheel origin at 100.1.
    g.check(pose(9.0), t=0.1, wheel_pose=pose(100.1, 50.0))
    _, out = g.check(pose(9.5), t=0.2, wheel_pose=pose(100.15, 50.0))
    # Only the 0.05 m travelled SINCE the hold began; the 100 m offset and the
    # motion that preceded the hold must both stay out.
    assert np.allclose(out[:3, 3], [0.05, 0.0, 0.0])


def test_hold_escalates_to_fault_after_max_hold_duration():
    g = guard(max_hold_duration=1.0)
    g.check(pose(0.0), t=0.0, wheel_pose=pose(0.0))
    assert g.check(pose(9.0), t=0.1, wheel_pose=pose(0.0))[0] == HOLD
    assert g.check(pose(9.1), t=0.9, wheel_pose=pose(0.0))[0] == HOLD
    # Past the bound, coasting must stop being reported as merely degraded.
    assert g.check(pose(9.2), t=1.5, wheel_pose=pose(0.0))[0] == FAULT


def test_rejected_pose_never_becomes_the_new_baseline():
    """Otherwise one jump ratchets the anchor and every later check passes."""
    g = guard()
    g.check(pose(0.0), t=0.0)
    g.check(pose(50.0), t=0.1)
    # Measured from the last GOOD pose (0.0), this is still a 50 m excursion.
    verdict, _ = g.check(pose(50.01), t=0.2)
    assert verdict == HOLD


def test_guard_releases_when_the_estimator_recovers():
    g = guard()
    g.check(pose(0.0), t=0.0, wheel_pose=pose(0.0))
    assert g.check(pose(9.0), t=0.1, wheel_pose=pose(0.0))[0] == HOLD
    assert g.holding
    # A plausible pose relative to the last good one clears the hold.
    verdict, out = g.check(pose(0.02), t=0.2, wheel_pose=pose(0.02))
    assert verdict == ACCEPT
    assert not g.holding
    assert np.allclose(out[:3, 3], [0.02, 0.0, 0.0])


def test_held_distance_is_zero_while_accepting():
    g = guard()
    g.check(pose(0.0), t=0.0, wheel_pose=pose(0.0))
    g.check(pose(0.02), t=0.1, wheel_pose=pose(0.02))
    assert g.held_distance == 0.0


def test_held_distance_accumulates_while_holding():
    g = guard()
    g.check(pose(0.0), t=0.0, wheel_pose=pose(0.0))
    g.check(pose(9.0), t=0.1, wheel_pose=pose(0.0))      # hold starts here
    g.check(pose(9.1), t=0.2, wheel_pose=pose(0.05))
    g.check(pose(9.2), t=0.3, wheel_pose=pose(0.12))
    assert np.isclose(g.held_distance, 0.12)


def test_held_distance_is_path_length_not_displacement():
    """Drift accrues with ground covered, so there-and-back still counts."""
    g = guard()
    g.check(pose(0.0), t=0.0, wheel_pose=pose(0.0))
    g.check(pose(9.0), t=0.1, wheel_pose=pose(0.0))      # hold starts here
    g.check(pose(9.1), t=0.2, wheel_pose=pose(0.30))     # out 0.30
    g.check(pose(9.2), t=0.3, wheel_pose=pose(0.0))      # and back 0.30
    assert np.isclose(g.held_distance, 0.60)


def test_held_distance_resets_when_the_guard_releases():
    g = guard()
    g.check(pose(0.0), t=0.0, wheel_pose=pose(0.0))
    g.check(pose(9.0), t=0.1, wheel_pose=pose(0.0))
    g.check(pose(9.1), t=0.2, wheel_pose=pose(0.10))
    assert g.held_distance > 0.0
    g.check(pose(0.01), t=0.3, wheel_pose=pose(0.11))    # accepted again
    assert g.held_distance == 0.0


def test_stationary_hold_accrues_no_distance():
    """Dead reckoning a stopped robot is exact, so the pose is still good."""
    g = guard(max_linear_speed=10.0)
    g.check(pose(0.0), t=0.0, wheel_speed=0.0, wheel_pose=pose(0.0))
    for i in range(1, 10):
        g.check(pose(0.5), t=0.1 * i, wheel_speed=0.0, wheel_pose=pose(0.0))
    assert g.held_distance == 0.0


def test_rejection_rate_tracks_gate_lockout():
    g = guard()
    g.check(pose(0.0), t=0.0)
    assert g.rejection_rate == 0.0
    for i in range(1, 21):
        g.check(pose(50.0 + i), t=0.1 * i)
    # Every update after the first was rejected: the thresholds are the suspect.
    assert g.rejection_rate > 0.9


def test_non_monotonic_stamp_is_ignored_not_rejected():
    """Out-of-order stamps are a transport artefact, not a divergence."""
    g = guard()
    g.check(pose(0.0), t=1.0)
    verdict, _ = g.check(pose(0.01), t=0.9)
    assert verdict == ACCEPT
    assert 'non-monotonic' in g.last_reason
