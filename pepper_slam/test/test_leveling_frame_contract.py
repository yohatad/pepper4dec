#!/usr/bin/env python3
"""The map -> *_init leveling rotation must be built from the IMU the estimator
actually uses.

*_map_odom_bridge with level_source 'calibration' levels the whole map by the
static base_frame -> lidar_imu_frame mount. If that frame is not the same one
the LIO config names as publish.body_frame, the map is leveled by the WRONG
mount and comes out tilted by the angle between them -- silently. Nothing
errors, and the .pcd save still reports "frame map".

That is not hypothetical: fastlio_lc_l2.launch.py simply never passed
lidar_imu_frame, so the bridge kept its l2lidar_frame_imu default while
FAST-LIO estimated the RealSense IMU. The two mounts differ by 64.2 deg, and
every map saved that way was tilted by that much. Measuring height on such a
map reads a slanted axis through it -- the 2026-08-23 run showed an 80.30 m z
band that was really 8.05 m once leveled correctly.

The only visible symptom was the floor offset in the bridge's own log line:
+0.257 m (the L2 lidar IMU) where the RealSense IMU sits at +0.314 m.
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from resolve_launch import snap  # noqa: E402

SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# (launch file, LIO config whose publish.body_frame the bridge must agree with)
CASES = [
    ('fastlio_lc_pgo/launch/fastlio_lc_l2.launch.py',
     'FAST_LIO/config/l2_rsimu.yaml'),
    ('fastlio_lc_pgo/launch/pointlio_lc_l2.launch.py',
     'point_lio/config/l2lidar_rsimu.yaml'),
]


def _find(node, key):
    """publish.body_frame, wherever it sits under the ros__parameters nesting."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, str):
                return v
            found = _find(v, key)
            if found is not None:
                return found
    return None


def _params(entry):
    merged = {}
    for p in entry.get('params', []):
        if isinstance(p, dict):
            merged.update(p)
    return merged


@pytest.mark.parametrize('launch_rel,config_rel', CASES)
def test_bridge_levels_by_the_estimators_own_imu(launch_rel, config_rel):
    launch_path = os.path.join(SRC, launch_rel)
    config_path = os.path.join(SRC, config_rel)
    for path in (launch_path, config_path):
        if not os.path.exists(path):
            pytest.skip('not in this workspace: %s' % path)

    body_frame = _find(yaml.safe_load(open(config_path)), 'body_frame')
    assert body_frame, 'no publish.body_frame in %s' % config_rel

    bridges = [n for n in snap(launch_path, [])
               if str(n.get('exe', '')).endswith('map_odom_bridge.py')]
    assert bridges, (
        '%s starts no *_map_odom_bridge; this test needs updating for a '
        'relayout.' % launch_rel)

    # Every bridge must name the frame EXPLICITLY. Selecting only the ones that
    # already carry the parameter would silently excuse the node that dropped
    # it -- which is precisely the bug, since dropping it is what lets the
    # l2lidar_frame_imu default take over.
    for b in bridges:
        params = _params(b)
        assert 'lidar_imu_frame' in params, (
            "%s: %s does not set lidar_imu_frame, so it falls back to its "
            "l2lidar_frame_imu default while %s estimates '%s'. That tilts the "
            "whole map by the angle between the two mounts, logging nothing."
            % (launch_rel, b.get('name'), config_rel, body_frame))
        got = params['lidar_imu_frame']
        assert got == body_frame, (
            "%s: %s levels map by base_frame -> '%s', but %s estimates '%s'. "
            "The map will be tilted by the angle between the two mounts, with "
            "nothing logged except the floor offset."
            % (launch_rel, b.get('name'), got, config_rel, body_frame))
