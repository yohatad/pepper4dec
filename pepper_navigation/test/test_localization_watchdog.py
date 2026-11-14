#!/usr/bin/env python3
"""Drive localization_watchdog.py through each of its branches on sim time.

Every branch is time-based (lost_duration, stale_timeout, recancel_period), so
the test owns /clock and steps it in 0.1 s ticks instead of sleeping. A fake
/navigate_to_pose/_action/cancel_goal service records each cancel with the sim
time it arrived. Diagnostics are handed to on_diagnostics() directly so their
timing is exact; test_diagnostics_topic_is_wired covers the subscription.

The test runs on a random domain over loopback only: it publishes /clock and
serves nav2's cancel service, and must never reach a live robot sharing the
shell's ROS_DOMAIN_ID.

    python3 -m pytest test/test_localization_watchdog.py     # or: colcon test
"""
import importlib.util
import os
import random
import time

from action_msgs.srv import CancelGoal
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

os.environ['CYCLONEDDS_URI'] = (
    '<CycloneDDS><Domain Id="any"><General><Interfaces>'
    '<NetworkInterface name="lo"/></Interfaces>'
    '<AllowMulticast>false</AllowMulticast></General>'
    '<Discovery><ParticipantIndex>auto</ParticipantIndex>'
    '<Peers><Peer address="127.0.0.1"/></Peers></Discovery>'
    '</Domain></CycloneDDS>')
# Cyclone rejects ROS_LOCALHOST_ONLY on top of an explicit lo interface
# ("the same interface may not be selected twice"); other RMWs need it.
if 'cyclonedds' in os.environ.get('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'):
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
else:
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
DOMAIN = random.randint(150, 220)

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts',
                     'localization_watchdog.py')
_spec = importlib.util.spec_from_file_location('localization_watchdog', _path)
watchdog_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(watchdog_mod)

LOCK = 'fastlio_localization: pose lock'
GATE = 'fastlio_localization: map -> odom gate'
OK, WARN, ERROR = DiagnosticStatus.OK, DiagnosticStatus.WARN, DiagnosticStatus.ERROR
T0 = 100.0   # sim time zero means "unset" to rclpy, so start well past it
STEP = 0.1


class Rig:
    """A watchdog, a /clock publisher and a recording cancel service."""

    def __init__(self, **params):
        params.setdefault('use_sim_time', True)
        args = ['--ros-args']
        for k, v in params.items():
            if isinstance(v, list):
                v = '[' + ','.join("'%s'" % s for s in v) + ']'
            elif isinstance(v, bool):
                v = str(v).lower()
            args += ['-p', '%s:=%s' % (k, v)]
        rclpy.init(args=args, domain_id=DOMAIN)
        self.wd = self.helper = self.ex = None
        try:
            self._setup()
        except BaseException:
            self.close()
            raise

    def _setup(self):
        self.helper = Node('watchdog_test_helper')
        self.clock_pub = self.helper.create_publisher(Clock, '/clock', 10)
        self.diag_pub = self.helper.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.cancels = []
        self.helper.create_service(
            CancelGoal, '/navigate_to_pose/_action/cancel_goal', self._on_cancel)
        self.t = T0
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.helper)
        self.wd = watchdog_mod.LocalizationWatchdog()
        self.ex.add_node(self.wd)
        self._sync_clock()
        self._wait(self.wd.cancel_cli.service_is_ready, 'cancel service not discovered')

    def close(self):
        if self.ex:
            self.ex.shutdown()
        for n in (self.wd, self.helper):
            if n:
                n.destroy_node()
        rclpy.shutdown()

    def _on_cancel(self, request, response):
        self.cancels.append(round(self.t, 1))
        return response

    def _now(self):
        return round(self.wd.get_clock().now().nanoseconds / 1e9, 1)

    def _publish_clock(self):
        sec = int(self.t)
        self.clock_pub.publish(Clock(clock=Time(
            sec=sec, nanosec=int(round((self.t - sec) * 1e9)))))

    def _sync_clock(self):
        # /clock is volatile: keep publishing until the watchdog has it, which
        # also covers the first sample being lost to discovery.
        end = time.monotonic() + 5.0
        while self._now() != round(self.t, 1):
            if time.monotonic() > end:
                raise AssertionError('watchdog clock did not reach %.1f' % self.t)
            self._publish_clock()
            self.ex.spin_once(timeout_sec=0.01)

    def _wait(self, cond, msg, timeout=5.0):
        end = time.monotonic() + timeout
        while not cond():
            if time.monotonic() > end:
                raise AssertionError(msg)
            self.ex.spin_once(timeout_sec=0.01)

    def settle(self, wall=0.05):
        """Spin without moving sim time, so timers and service calls finish."""
        end = time.monotonic() + wall
        while time.monotonic() < end:
            self.ex.spin_once(timeout_sec=0.005)

    def diag(self, **levels):
        """Hand the watchdog one DiagnosticArray, stamped at the current sim time."""
        names = {'lock': LOCK, 'gate': GATE}
        msg = DiagnosticArray()
        msg.status = [DiagnosticStatus(level=lvl, name=names[k]) for k, lvl in levels.items()]
        self.wd.on_diagnostics(msg)

    def run(self, seconds, every=1.0, on_tick=None, **levels):
        """Advance sim time, reporting `levels` every `every` s (none if empty)."""
        n = int(round(seconds / STEP))
        per = int(round(every / STEP))
        for i in range(n):
            if levels and i % per == 0:
                self.diag(**levels)
            self.t = round(self.t + STEP, 1)
            self._sync_clock()
            self.settle()
            if on_tick:
                on_tick(self)
        self.settle(0.3)


@pytest.fixture
def rig_factory():
    rigs = []

    def make(**params):
        rigs.append(Rig(**params))
        return rigs[-1]
    yield make
    for r in rigs:
        r.close()


BOTH = {'status_names': [LOCK, GATE]}


def test_01_nothing_published_never_cancels(rig_factory):
    r = rig_factory()
    r.run(30.0)
    assert not r.wd.acting
    assert r.cancels == []


def test_02_error_longer_than_lost_duration_cancels(rig_factory):
    r = rig_factory()
    r.run(2.0, lock=OK)
    first_error = r.t
    r.run(7.0, lock=ERROR)
    assert r.wd.acting
    assert r.cancels, 'no cancel after 7 s of ERROR'
    # lost at first_error + lost_duration, acted on at the next 1 s check
    assert first_error + 5.0 <= r.cancels[0] <= first_error + 6.0 + STEP


def test_03_error_shorter_than_lost_duration_does_not_cancel(rig_factory):
    r = rig_factory()
    r.run(2.0, lock=OK)
    r.run(3.5, lock=ERROR)
    r.run(10.0, lock=OK)
    assert not r.wd.acting
    assert r.cancels == []


def test_04_gate_never_published_is_ignored(rig_factory):
    r = rig_factory(**BOTH)
    r.run(30.0, lock=OK)    # 3x stale_timeout without a single gate status
    assert not r.wd.acting
    assert r.cancels == []


def test_05_lock_ok_gate_error_is_lost(rig_factory):
    r = rig_factory(**BOTH)
    r.run(2.0, lock=OK, gate=OK)
    r.run(7.0, lock=OK, gate=ERROR)
    lost, reason = r.wd._worst()
    assert lost and r.wd.acting and r.cancels
    assert GATE in reason


def test_06_both_error_names_one(rig_factory):
    r = rig_factory(**BOTH)
    r.run(2.0, lock=OK, gate=OK)
    r.run(7.0, lock=ERROR, gate=ERROR)
    lost, reason = r.wd._worst()
    assert lost and r.wd.acting and r.cancels
    assert (LOCK in reason) != (GATE in reason), reason


def test_07_recovery_clears_and_stops_cancelling(rig_factory):
    r = rig_factory()
    r.run(2.0, lock=OK)
    r.run(7.0, lock=ERROR)
    assert r.wd.acting and r.cancels
    n = len(r.cancels)
    r.run(1.0, lock=OK)
    assert not r.wd.acting, 'still holding navigation after a healthy status'
    r.run(15.0, lock=OK)
    assert len(r.cancels) == n, 'cancelled again after recovering'


def test_08_silence_after_first_sighting_is_lost(rig_factory):
    r = rig_factory()
    r.run(2.0, lock=OK)
    r.run(8.5)              # silent past lost_duration, short of stale_timeout
    assert not r.wd.acting, 'silence after OK counted as unhealthy'
    assert r.cancels == []
    r.run(2.5)              # now past stale_timeout (10 s since the last OK)
    lost, reason = r.wd._worst()
    assert lost and r.wd.acting and r.cancels
    assert reason.startswith('no "%s" status for' % LOCK), reason


def test_08b_reporting_gap_does_not_count_as_unhealthy(rig_factory):
    r = rig_factory()
    r.run(2.0, lock=OK)
    r.run(4.0)              # gap, still inside stale_timeout
    first_error = r.t
    r.run(4.0, lock=ERROR)
    assert r.cancels == [], 'gap before the ERROR was counted as unhealthy'
    r.run(3.0, lock=ERROR)
    assert r.cancels and r.cancels[0] >= first_error + 5.0
    _, reason = r.wd._worst()
    assert reason.startswith('"%s" unhealthy for' % LOCK), reason


def test_08c_error_then_silence_stays_lost(rig_factory):
    r = rig_factory()
    r.run(2.0, lock=OK)
    r.run(1.0, lock=ERROR)  # last word is ERROR, then nothing
    r.run(6.0)
    assert r.wd.acting and r.cancels


@pytest.mark.parametrize('warn_is_lost', [False, True])
def test_09_warn(rig_factory, warn_is_lost):
    r = rig_factory(treat_warn_as_lost=warn_is_lost)
    r.run(2.0, lock=OK)
    r.run(8.0, lock=WARN)
    assert r.wd.acting == warn_is_lost
    assert bool(r.cancels) == warn_is_lost


def test_10_first_cancel_immediate_then_every_recancel_period(rig_factory):
    r = rig_factory()
    became_lost = []

    def watch(rig):
        if rig.wd.acting and not became_lost:
            became_lost.append(rig.t)
    r.run(2.0, lock=OK)
    r.run(16.0, on_tick=watch, lock=ERROR)
    assert became_lost and r.cancels
    # same check tick that first declared LOST -- the last_cancel-is-None bug
    # this guards made the watchdog detect LOST and then never cancel at all
    # (cancels are stamped when the service answers, up to one STEP later)
    assert 0.0 <= r.cancels[0] - became_lost[0] <= STEP + 1e-6
    gaps = [round(b - a, 1) for a, b in zip(r.cancels, r.cancels[1:])]
    assert len(r.cancels) >= 4, r.cancels
    assert all(3.0 - STEP <= g <= 3.0 + STEP for g in gaps), gaps


def test_11_first_sighting_already_error_starts_the_clock(rig_factory):
    r = rig_factory()
    r.run(20.0)             # long silence before anything reports
    first = r.t
    r.run(4.0, lock=ERROR)
    assert r.cancels == [], 'cancelled before lost_duration of observed ERROR'
    r.run(3.0, lock=ERROR)
    assert r.cancels and r.cancels[0] >= first + 5.0


def test_12_cancel_goals_false_detects_but_never_cancels(rig_factory):
    r = rig_factory(cancel_goals=False)
    r.run(2.0, lock=OK)
    r.run(10.0, lock=ERROR)
    assert r.wd.acting
    assert r.cancels == []


def test_diagnostics_topic_is_wired(rig_factory):
    r = rig_factory()
    msg = DiagnosticArray(status=[DiagnosticStatus(level=ERROR, name=LOCK)])
    end = time.monotonic() + 5.0
    while r.wd.seen[LOCK]['at'] is None and time.monotonic() < end:
        r.diag_pub.publish(msg)
        r.settle(0.1)
    assert r.wd.seen[LOCK]['at'] is not None, '/diagnostics never reached the watchdog'
