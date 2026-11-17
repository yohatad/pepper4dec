#!/usr/bin/env python3
"""The LIO frame contract, asserted on the benchmark bag through the real stack.

Runs FAST-LIO odometry and lio_odom_bridge (fastlio_odometry_bag.launch.py) on
the first minute of slam_20260823_aligned, and the check_frame_contract test
executable against them. It passes only if the checker does:

  - the levelling offset (odom <-> lio_init z) equals the calibration
    prediction, base_footprint -> camera_imu_optical_frame z (0.3139 m);
  - base_footprint's mean roll and pitch in odom are within 3 deg;
  - the floor (median of the lowest 15% of the cloud in odom) is within 0.10 m
    of z = 0.

So a change that tilts the levelled frame, puts the floor off z = 0, or levels
by the wrong IMU fails here rather than on the robot.

The bag is ~/ros2_ws/bags/slam_20260823_aligned (override with
PEPPER_SLAM_TEST_BAG). The test skips when the bag is absent or FAST-LIO is not
built; fast_lio cannot be a test_depend, because it exec_depends on
pepper_slam and the dependency graph would close a cycle.

Run via: colcon test --packages-select pepper_slam

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 17, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os
import random
import signal
import subprocess
import time

import pytest
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

BAG = os.environ.get(
    'PEPPER_SLAM_TEST_BAG', os.path.expanduser('~/ros2_ws/bags/slam_20260823_aligned'))
CHECKER = os.environ.get('CHECK_FRAME_CONTRACT', '')

STARTUP = 8.0     # s for the launch to come up before the bag plays
WARMUP = 20.0     # s of bag before the checker starts: FAST-LIO init + leveling
CHECK = 25.0      # s the checker collects for
PLAY = 60.0       # s of bag played in total


def _stop(proc: subprocess.Popen) -> None:
    """SIGINT the whole process group, then kill whatever is left."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_frame_contract_on_the_benchmark_bag(tmp_path):
    if not os.path.isfile(os.path.join(BAG, 'metadata.yaml')):
        pytest.skip(f'benchmark bag not found: {BAG}')
    try:
        get_package_share_directory('fast_lio')
    except PackageNotFoundError:
        pytest.skip('fast_lio is not built in this workspace')
    assert CHECKER and os.access(CHECKER, os.X_OK), f'checker not built: {CHECKER!r}'

    # Its own domain: colcon runs packages' tests in parallel.
    env = dict(os.environ, ROS_DOMAIN_ID=str(random.randint(150, 200)))
    group = {'env': env, 'start_new_session': True}
    log = open(tmp_path / 'stack.log', 'w')

    stack = subprocess.Popen(
        ['ros2', 'launch', 'pepper_slam', 'fastlio_odometry_bag.launch.py', 'rviz:=false'],
        stdout=log, stderr=subprocess.STDOUT, **group)
    player = None
    try:
        time.sleep(STARTUP)
        player = subprocess.Popen(
            ['ros2', 'bag', 'play', BAG, '--clock', '--read-ahead-queue-size', '2000',
             '--disable-keyboard-controls',
             '--topics', '/points', '/camera/imu', '/tf', '/tf_static'],
            stdout=log, stderr=subprocess.STDOUT, **group)
        time.sleep(WARMUP)
        check = subprocess.run(
            [CHECKER, '--ros-args', '-p', 'level_frame:=odom', '-p', f'duration:={CHECK}',
             '-p', 'use_sim_time:=true'],
            env=env, capture_output=True, text=True, timeout=CHECK + 30)
        time.sleep(max(0.0, PLAY - WARMUP - CHECK))
    finally:
        for proc in (player, stack):
            if proc is not None:
                _stop(proc)
        log.close()

    print(check.stdout)
    assert check.returncode == 0, (
        f'frame contract FAILED:\n{check.stdout}\n{check.stderr[-2000:]}\n'
        f'stack log: {tmp_path / "stack.log"}')
