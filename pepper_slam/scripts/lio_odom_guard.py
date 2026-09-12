#!/usr/bin/env python3
"""Validation gate for a LIO estimator's pose, with wheel-odometry dead reckoning.

WHY THIS EXISTS
    When FAST-LIO's plane correspondences collapse -- a feature-poor corridor, a
    blank wall, a body-occluded lidar -- the iterated EKF stops correcting but
    keeps propagating on IMU. Accelerometer bias and gravity-misalignment then
    get DOUBLE integrated, so the pose does not drift, it accelerates: the
    estimate sprints out of the map in a second or two. l2.yaml's high acc_cov
    (2.0, carrying the motor-vibration picket fence) makes it worse by growing P
    quickly, which raises the Kalman gain and lets one bad correspondence set
    move the estimate a long way in a single scan.

    Unguarded, lio_odom_bridge republishes that straight onto
    odom -> base_footprint, i.e. INSIDE Nav2's control loop. Nav2's max_vel_x
    does not help: it clamps commanded velocity and never inspects the pose.

REJECT, DO NOT SATURATE
    This is measurement validation gating (Bar-Shalom's validation gate;
    robot_localization spells the same idea <topic>_pose_rejection_threshold).
    A failing update is DISCARDED, never clipped to the limit. Clipping would
    manufacture a plausible-looking but wrong pose, which is strictly harder to
    detect downstream than an obviously broken one. Saturation is the right
    treatment for an actuator command, which must be feasible; it is the wrong
    treatment for a measurement, which must be honest or absent.

TIER 1 ONLY
    The bound used here is PHYSICAL -- Pepper cannot exceed it whatever the
    filter believes -- so it needs no baseline, no warm-up and no tuning.
    Statistical degeneracy detection (a covariance spike, an effct_feat_num
    collapse; see utils/lio_health.py) is a separate tier: it should inform the
    health signal but must not reject on its own, because those signals look
    healthy after a confidently-wrong relock.

HOLD = DEAD RECKON, AND IT IS TIME BOUNDED
    While rejecting, the pose is carried forward on wheel odometry rather than
    frozen (a frozen odom makes obstacles stream past a robot that reports
    standing still) or dropped (that just moves the failure into tf2
    extrapolation errors). Coasting is bounded by max_hold_duration: a
    persistent fault must escalate to FAULT, not degrade into unbounded silent
    dead reckoning. Same reasoning as localization.health_bad_duration in
    FAST_LIO/src/laserLocalization.cpp.

GATE LOCKOUT
    If the gate starts rejecting MOST updates, the likely fault is this guard's
    thresholds, not the sensor -- an over-tight gate rejects exactly the
    measurements that would correct it and diverges with a clean conscience. The
    rejection rate over a rolling window is tracked and surfaced so that shows up
    as a diagnostic rather than as silence.

No ROS imports: everything here is plain numpy so it unit-tests without a spin.
"""

import collections

import numpy as np

# Verdicts returned by OdomGuard.check().
ACCEPT = 'accept'      # pose is plausible; publish it as-is
HOLD = 'hold'          # rejected, but inside max_hold_duration: dead reckon
FAULT = 'fault'        # rejected for longer than max_hold_duration


def relative_motion(m_from: np.ndarray, m_to: np.ndarray) -> np.ndarray:
    """Body-relative motion between two poses of the SAME child frame.

    The result is child(from) -> child(to). Because it is purely relative, it
    can be right-multiplied onto a pose expressed in a DIFFERENT parent frame,
    which is what lets wheel odometry (pepper_odom -> base_footprint) carry a
    pose in odom even though the two trees are deliberately disconnected. Both
    parents are gravity-aligned by physical construction -- the same argument
    pepper_odom_relabel.py's docstring makes for relabeling z/roll/pitch.
    """
    return np.linalg.inv(m_from) @ m_to


def rotation_angle(r: np.ndarray) -> float:
    """Magnitude of a rotation matrix's axis-angle, in radians."""
    # clip guards against a trace fractionally outside [-1, 3] from round-off.
    cos_theta = (np.trace(r[:3, :3]) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


class OdomGuard:
    """Gates LIO poses on a physical speed bound, holding on wheel odometry.

    Args:
        max_linear_speed: m/s the base cannot exceed. Anything faster is not a
            fast robot, it is a broken estimate.
        max_angular_speed: rad/s equivalent for rotation.
        max_hold_duration: seconds of continuous rejection before escalating to
            FAULT. Bounds how long the pose may coast on wheels alone.
        zero_velocity_eps: below this wheel speed (m/s) the base is treated as
            stationary, which turns the encoders into a zero-velocity detector.
        zero_velocity_lio_eps: LIO speed (m/s) that contradicts a stationary
            wheel reading. Needs no tuning against drift rates: if the wheels
            say stopped and LIO says moving, LIO is wrong.
        rate_window: how many recent verdicts the lockout rate is measured over.
    """

    def __init__(self, max_linear_speed=0.7, max_angular_speed=1.0,
                 max_hold_duration=3.0, zero_velocity_eps=0.02,
                 zero_velocity_lio_eps=0.10, rate_window=200):
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.max_hold_duration = max_hold_duration
        self.zero_velocity_eps = zero_velocity_eps
        self.zero_velocity_lio_eps = zero_velocity_lio_eps

        # Last ACCEPTED pose and its stamp -- the anchor both the speed check
        # and the dead reckoning measure from. Never updated while rejecting,
        # so a rejected excursion cannot become the next baseline.
        self._last_good = None
        self._last_good_t = None
        # Wheel pose sampled at the moment rejection began, i.e. W_0.
        self._hold_wheel_origin = None
        self._hold_start_t = None
        # PATH LENGTH dead reckoned since the hold began, and the sample it was
        # last measured from. Path length, not displacement from W_0: wheel
        # drift accumulates with distance driven, so a there-and-back excursion
        # has drifted even though it ends where it started -- the same quantity
        # pepper_odom_covariance grows its variance against.
        self._held_distance = 0.0
        self._hold_wheel_last = None

        self._verdicts = collections.deque(maxlen=rate_window)
        self.last_reason = ''

    @property
    def holding(self) -> bool:
        return self._hold_start_t is not None

    @property
    def held_distance(self) -> float:
        """Metres dead reckoned since the current hold began; 0.0 when accepting.

        How far the published pose has been carried on wheels alone with no
        exteroceptive correction, so it is what downstream should grow its
        uncertainty against. Deliberately NOT time: dead reckoning a stationary
        robot is exact, so a long hold that covered no ground has added no
        error. Elapsed time is reported separately, by the verdict.
        """
        return self._held_distance

    @property
    def rejection_rate(self) -> float:
        """Fraction of the recent window that was rejected (gate-lockout signal)."""
        if not self._verdicts:
            return 0.0
        return sum(self._verdicts) / len(self._verdicts)

    def check(self, m_odom_base: np.ndarray, t: float, wheel_pose=None,
              wheel_speed=None):
        """Classify one LIO pose. Returns (verdict, pose_to_publish).

        Args:
            m_odom_base: the candidate odom -> base_footprint as a 4x4.
            t: the ODOMETRY MESSAGE's stamp in seconds, never wall-now -- under
                bag replay the two differ by hours.
            wheel_pose: latest pepper_odom -> base_footprint as a 4x4, or None
                when wheel odometry is unavailable.
            wheel_speed: latest wheel linear speed (m/s), or None.

        The returned pose is the candidate on ACCEPT, and the dead-reckoned
        pose on HOLD/FAULT. On FAULT the caller should stop treating the pose
        as trustworthy even though a best-effort estimate is still returned.
        """
        # First pose: nothing to measure a delta against, so accept and anchor.
        if self._last_good is None or self._last_good_t is None:
            return self._accept(m_odom_base, t)

        dt = t - self._last_good_t
        if dt <= 0.0:
            # Out-of-order or duplicate stamp. Not a divergence; ignore it
            # rather than dividing by zero or a negative.
            self.last_reason = 'non-monotonic stamp'
            return HOLD if self.holding else ACCEPT, self._last_good

        reason = self._implausible(m_odom_base, dt, wheel_speed)
        if reason is None:
            return self._accept(m_odom_base, t)

        self.last_reason = reason
        self._verdicts.append(1)
        if self._hold_start_t is None:
            self._hold_start_t = t
            self._held_distance = 0.0
            if wheel_pose is None:
                self._hold_wheel_origin = None
                self._hold_wheel_last = None
            else:
                self._hold_wheel_origin = wheel_pose.copy()
                self._hold_wheel_last = wheel_pose.copy()

        held = self._dead_reckon(wheel_pose)
        if t - self._hold_start_t > self.max_hold_duration:
            return FAULT, held
        return HOLD, held

    def _implausible(self, m_odom_base, dt, wheel_speed):
        """Reason string if this pose breaks a physical bound, else None."""
        delta = relative_motion(self._last_good, m_odom_base)
        linear = float(np.linalg.norm(delta[:3, 3])) / dt
        angular = rotation_angle(delta) / dt

        if linear > self.max_linear_speed:
            return (f'linear {linear:.2f} m/s over {dt * 1e3:.0f} ms exceeds '
                    f'{self.max_linear_speed:.2f} m/s')
        if angular > self.max_angular_speed:
            return (f'angular {angular:.2f} rad/s over {dt * 1e3:.0f} ms exceeds '
                    f'{self.max_angular_speed:.2f} rad/s')
        # Zero-velocity cross-check. The encoders cannot report stationary
        # while the base actually moves, so a disagreement convicts LIO.
        if (wheel_speed is not None and wheel_speed < self.zero_velocity_eps
                and linear > self.zero_velocity_lio_eps):
            return (f'wheels stationary ({wheel_speed:.3f} m/s) while LIO '
                    f'reports {linear:.2f} m/s')
        return None

    def _accept(self, m_odom_base, t):
        self._verdicts.append(0)
        self._last_good = m_odom_base.copy()
        self._last_good_t = t
        self._hold_start_t = None
        self._hold_wheel_origin = None
        self._hold_wheel_last = None
        self._held_distance = 0.0
        self.last_reason = ''
        return ACCEPT, m_odom_base

    def _dead_reckon(self, wheel_pose):
        """P(t) = P_good @ (W_0^-1 @ W(t)); P_good alone without wheel odometry."""
        if wheel_pose is None or self._hold_wheel_origin is None:
            # No wheel source: the best available answer is the last good pose.
            # It freezes rather than coasts, which is why wheel odometry is a
            # requirement for the guard to do its job properly.
            return self._last_good
        # Accumulate path length before composing, so held_distance covers this
        # sample's motion too.
        step = relative_motion(self._hold_wheel_last, wheel_pose)
        self._held_distance += float(np.linalg.norm(step[:3, 3]))
        self._hold_wheel_last = wheel_pose.copy()
        return self._last_good @ relative_motion(self._hold_wheel_origin, wheel_pose)
