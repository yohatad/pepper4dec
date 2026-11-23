#!/usr/bin/env python3
"""The rig URDF is the calibration source of truth; its copies must agree.

urdf/sensor_rig.xacro is what pepper_sensor_tf publishes. Three files hold
values derived from it by hand, and nothing regenerates them:

  FAST_LIO/config/l2_rsimu.yaml        extrinsic_R / extrinsic_T: the lidar in
  point_lio/config/l2lidar_rsimu.yaml  the estimator's IMU body frame, composed
                                       along the rig chain and inverted
  fastlio_lc_pgo/launch/
      fastlio_lc_l2.launch.py          sc_lidar_height: the lidar's height

A recalibration that edits only the URDF leaves the estimators fusing with
the old extrinsic. These tests fail until every copy is updated.

The URDF is expanded with xacro exactly as pepper_sensor_tf.launch.py does,
for both scopes. The sibling packages cannot be test_depends here (they
exec_depend on pepper_slam), so their cases skip when the file is not in the
workspace.
"""
import math
import os
import re
import xml.etree.ElementTree as ET

import pytest
import xacro
import yaml

PKG = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.normpath(os.path.join(PKG, '..', '..'))
RIG_URDF = os.path.join(PKG, 'urdf', 'pepper_sensor_rig.urdf.xacro')

BASE_FRAME = 'base_footprint'
# The frame the L2 stamps /points with (l2lidar_node.yaml: cloud_frame), i.e.
# the "lidar" side of every LIO extrinsic.
LIDAR_FRAME = 'l2lidar_frame'
# realsense2_camera publishes everything below this frame itself, from the
# device; scope:=mount must leave exactly that subtree out.
CAMERA_ROOT = 'camera_camera_link'

LIO_CONFIGS = [
    'FAST_LIO/config/l2_rsimu.yaml',
    'point_lio/config/l2lidar_rsimu.yaml',
]
SC_LAUNCH = 'fastlio_lc_pgo/launch/fastlio_lc_l2.launch.py'

# The LIO configs and sc_lidar_height are rounded to 6 / 4 decimals.
LIO_TOL = 1e-5          # m for T, unitless for R entries
HEIGHT_TOL = 1e-4       # m


def _rpy_to_matrix(rpy):
    """URDF convention: fixed-axis roll, pitch, yaw, i.e. Rz(y) Ry(p) Rx(r)."""
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def _matvec(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _joints(scope):
    """{(parent, child): (xyz, rpy)} from the expanded rig URDF."""
    doc = xacro.process_file(RIG_URDF, mappings={'scope': scope})
    root = ET.fromstring(doc.toxml())
    joints = {}
    for j in root.iter('joint'):
        assert j.get('type') == 'fixed', 'rig joint %s is not fixed' % j.get('name')
        origin = j.find('origin')
        key = (j.find('parent').get('link'), j.find('child').get('link'))
        joints[key] = ([float(v) for v in origin.get('xyz', '0 0 0').split()],
                       [float(v) for v in origin.get('rpy', '0 0 0').split()])
    return joints


def _chain(joints, ancestor, frame):
    """(R, t): pose of frame in ancestor, composed along the URDF joints."""
    parent_of = {child: parent for (parent, child) in joints}
    path, target = [], frame
    while frame != ancestor:
        assert frame in parent_of, (
            "the rig URDF has no chain from '%s' down to '%s' (it stops at '%s')"
            % (ancestor, target, frame))
        path.append((parent_of[frame], frame))
        frame = parent_of[frame]
    rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    trans = [0.0, 0.0, 0.0]
    for key in reversed(path):
        xyz, rpy = joints[key]
        trans = [a + b for a, b in zip(trans, _matvec(rot, xyz))]
        rot = _matmul(rot, _rpy_to_matrix(rpy))
    return rot, trans


def _find(node, key):
    """First value stored under key, wherever it sits in the param nesting."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                return v
            found = _find(v, key)
            if found is not None:
                return found
    return None


def _sibling(rel):
    path = os.path.join(SRC, rel)
    if not os.path.exists(path):
        pytest.skip('not in this workspace: %s' % path)
    return path


def test_mount_scope_leaves_out_exactly_the_driver_subtree():
    # Live, realsense2_camera publishes every edge below camera_camera_link
    # from the device. If scope:=mount also published one, that edge would have
    # two /tf_static publishers and whichever lands last silently wins; if it
    # dropped a rig edge, the tree would not close on bag replay.
    full, mount = _joints('all'), _joints('mount')
    assert set(mount) <= set(full)
    below = {CAMERA_ROOT}
    for _ in full:
        below |= {c for (p, c) in full if p in below}
    driver = {(p, c) for (p, c) in full if p in below}
    assert set(full) - set(mount) == driver, (
        'scope:=mount should drop exactly the edges below %s.\n  dropped: %s\n'
        '  below %s: %s' % (CAMERA_ROOT, sorted(set(full) - set(mount)),
                            CAMERA_ROOT, sorted(driver)))
    assert mount, 'scope:=mount publishes nothing'


@pytest.mark.parametrize('config_rel', LIO_CONFIGS)
def test_lio_extrinsic_matches_urdf(config_rel):
    with open(_sibling(config_rel), encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    body_frame = _find(doc, 'body_frame')
    ext_t, ext_r = _find(doc, 'extrinsic_T'), _find(doc, 'extrinsic_R')
    assert body_frame and ext_t and ext_r, (
        '%s: no publish.body_frame / extrinsic_T / extrinsic_R' % config_rel)

    # The config's convention: p_body = extrinsic_R * p_lidar + extrinsic_T,
    # the inverse of the body frame's pose in the lidar frame. scope 'all':
    # the body frame is the camera IMU, below the driver-owned subtree.
    rot, trans = _chain(_joints('all'), LIDAR_FRAME, body_frame)
    want_r = _transpose(rot)
    want_t = [-v for v in _matvec(want_r, trans)]
    got_r = [ext_r[3 * i:3 * i + 3] for i in range(3)]

    err_t = max(abs(a - b) for a, b in zip(ext_t, want_t))
    err_r = max(abs(got_r[i][j] - want_r[i][j]) for i in range(3) for j in range(3))
    assert err_t <= LIO_TOL and err_r <= LIO_TOL, (
        '%s: the %s -> %s extrinsic is off from the rig URDF by %.4f mm in T '
        'and %.2e in R. Recompose it from urdf/sensor_rig.xacro.\n'
        '  extrinsic_T: %s\n  extrinsic_R: %s'
        % (config_rel, LIDAR_FRAME, body_frame, err_t * 1000.0, err_r,
           ['%.6f' % v for v in want_t],
           ['%.6f' % v for row in want_r for v in row]))


def test_scan_context_height_matches_urdf():
    with open(_sibling(SC_LAUNCH), encoding='utf-8') as f:
        found = re.findall(r"'sc_lidar_height':\s*([-0-9.eE+]+)", f.read())
    assert len(found) == 1, '%s: expected one sc_lidar_height, found %d' % (
        SC_LAUNCH, len(found))
    _, trans = _chain(_joints('mount'), BASE_FRAME, LIDAR_FRAME)
    got = float(found[0])
    assert abs(got - trans[2]) <= HEIGHT_TOL, (
        '%s: sc_lidar_height is %.4f m, but the rig URDF puts %s at z = %.4f m '
        'in %s.' % (SC_LAUNCH, got, LIDAR_FRAME, trans[2], BASE_FRAME))
