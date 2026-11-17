#!/usr/bin/env python3
"""The collision monitor's verdicts, checked on the benchmark bag.

Runs bag_test/collision_monitor_bag.launch.py on the first ~65 s of
slam_20260823_aligned: points_safety_filter and the real Nav2 collision
monitor, configured from nav2_params_fastloc.yaml, with cm_bag_judge scoring
every /points_safety scan against plain polygon geometry (the cloud in
base_footprint, the height band, points counted inside each polygon). It
passes only if:

  - the bag exercised the zones: the reference holds STOP and SLOWDOWN scans;
  - the monitor never reached a verdict the geometry does not support: each
    mismatch is either a scan with no monitor output yet (startup), or the
    verdict of the scan before or after it -- the monitor's answer for one
    scan is read when the next arrives, so a zone transition can land one
    scan late;
  - at least 99% of scans agree within one scan.

So a change to the polygons, the height band, the safety filter's cuts or the
monitor's sources that makes the monitor miss an obstacle (or stop for none)
fails here rather than on the robot. The monitor's velocity topics are renamed
to /cm_bag/*, so nothing here ever publishes /cmd_vel.

The bag is ~/ros2_ws/bags/slam_20260823_aligned (override with
PEPPER_NAV_TEST_BAG). The test skips when the bag is absent.

Run via: colcon test --packages-select pepper_navigation

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 17, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import csv
import json
import os
import random
import signal
import subprocess
import time

import pytest

BAG = os.environ.get(
    'PEPPER_NAV_TEST_BAG', os.path.expanduser('~/ros2_ws/bags/slam_20260823_aligned'))

STARTUP = 8.0     # s for the launch to come up before the bag plays
PLAY = 65.0       # s of bag: the first ~30 s are clear, then STOP and SLOWDOWN
MIN_SCANS = 300   # 10 Hz; fewer means the stack or the bag did not run


def _stop(proc: subprocess.Popen, timeout: float = 20.0) -> None:
    """SIGINT the whole process group, then kill whatever is left."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_collision_monitor_on_the_benchmark_bag(tmp_path):
    if not os.path.isfile(os.path.join(BAG, 'metadata.yaml')):
        pytest.skip(f'benchmark bag not found: {BAG}')

    # Its own domain: colcon runs packages' tests in parallel.
    env = dict(os.environ, ROS_DOMAIN_ID=str(random.randint(150, 200)))
    group = {'env': env, 'start_new_session': True}
    prefix = tmp_path / 'cm'
    log = open(tmp_path / 'stack.log', 'w')

    stack = subprocess.Popen(
        ['ros2', 'launch', 'pepper_navigation', 'collision_monitor_bag.launch.py',
         f'out_prefix:={prefix}'],
        stdout=log, stderr=subprocess.STDOUT, **group)
    player = None
    try:
        time.sleep(STARTUP)
        player = subprocess.Popen(
            ['ros2', 'bag', 'play', BAG, '--clock', '--disable-keyboard-controls',
             '--topics', '/points', '/tf_static'],
            stdout=log, stderr=subprocess.STDOUT, **group)
        time.sleep(PLAY)
    finally:
        # The judge writes its summary on SIGINT, so the stack stops first.
        for proc in (stack, player):
            if proc is not None:
                _stop(proc)
        log.close()

    summary_file = f'{prefix}_summary.json'
    assert os.path.isfile(summary_file), (
        f'the judge wrote no summary; stack log: {tmp_path / "stack.log"}')
    with open(summary_file) as f:
        s = json.load(f)
    print(json.dumps(s, indent=1))

    assert s['scans'] >= MIN_SCANS, f'only {s["scans"]} scans judged'
    ref = s['reference_pct']
    assert ref['stop'] > 0.0 and ref['slowdown'] > 0.0, (
        f'the bag never entered a zone, so nothing was tested: {ref}')

    # Every mismatch must be a scan the monitor had not answered yet, or a
    # transition landing one scan early or late -- never a verdict that no
    # neighbouring scan's geometry supports.
    with open(f'{prefix}_scans.csv') as f:
        rows = [(r['reference'], r['monitor']) for r in csv.DictReader(f)]
    wrong = []
    for k, (_, mon) in enumerate(rows):
        near = {rows[i][0] for i in range(max(0, k - 1), min(len(rows), k + 2))}
        if mon != 'no_output' and mon not in near:
            wrong.append(f'scan {k}: reference {rows[k][0]}, monitor {mon}')
    assert not wrong, 'the monitor disagreed with the geometry:\n' + '\n'.join(wrong[:20])
    assert s['agree_within_1_scan_pct'] >= 99.0, s
