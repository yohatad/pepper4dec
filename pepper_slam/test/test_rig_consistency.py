#!/usr/bin/env python3
"""The rig geometry is written down in four places; they must agree.

data/sensor_tf.yaml is the calibration source of truth, but nothing generates
the other copies from it:

  urdf/sensor_rig.xacro          what pepper_sensor_tf publishes by default
                                 (publisher:=urdf), its quaternions as rpy
  FAST_LIO/config/l2_rsimu.yaml  extrinsic_R / extrinsic_T, the lidar in the
  point_lio/config/              estimator's IMU body frame, composed from the
      l2lidar_rsimu.yaml         YAML chain and inverted

A recalibration that edits only the YAML changes nothing on the robot: the
URDF keeps publishing the old mount and the estimators keep fusing with the
old extrinsic. These tests fail until every copy is updated.

The LIO configs live in sibling repositories (fast_lio, point_lio), which
cannot be test_depends here (they exec_depend on pepper_slam). Those cases
skip when the file is not in the workspace.
"""
import math
import os
import xml.etree.ElementTree as ET

import pytest
import yaml

PKG = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.normpath(os.path.join(PKG, '..', '..'))
SENSOR_TF = os.path.join(PKG, 'data', 'sensor_tf.yaml')
RIG_XACRO = os.path.join(PKG, 'urdf', 'sensor_rig.xacro')
XACRO_NS = '{http://www.ros.org/wiki/xacro}'

# The frame the L2 stamps /points with (l2lidar_node.yaml: cloud_frame), i.e.
# the "lidar" side of every LIO extrinsic.
LIDAR_FRAME = 'l2lidar_frame'
LIO_CONFIGS = [
    'FAST_LIO/config/l2_rsimu.yaml',
    'point_lio/config/l2lidar_rsimu.yaml',
]

# The xacro holds the YAML's values verbatim (xyz) or converted to 9 decimals
# (rpy). The LIO configs are rounded to 6 decimals.
URDF_TRANS_TOL = 1e-6   # m
URDF_ROT_TOL = 1e-4     # deg
LIO_TOL = 1e-5          # m for T, unitless for R entries


def _quat_to_matrix(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


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


def _angle_deg(a, b):
    """Angle of the rotation taking a to b."""
    trace = sum(a[i][j] * b[i][j] for i in range(3) for j in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0))))


def _sensor_tf():
    """{(parent, child): edge} from data/sensor_tf.yaml."""
    with open(SENSOR_TF, encoding='utf-8') as f:
        edges = yaml.safe_load(f)['transforms']
    return {(e['parent'], e['child']): e for e in edges}


def _xacro_joints():
    """{(parent, child): (xyz, rpy, scope_all_only)} from the rig macro.

    Parsed as plain XML rather than expanded by xacro, so it needs no
    ament index; the origins are literal numbers, not ${} expressions.
    """
    root = ET.parse(RIG_XACRO).getroot()
    guarded = set()
    for block in root.iter(XACRO_NS + 'if'):
        guarded.update(id(j) for j in block.iter('joint'))
    joints = {}
    for j in root.iter('joint'):
        origin = j.find('origin')
        key = (j.find('parent').get('link'), j.find('child').get('link'))
        joints[key] = ([float(v) for v in origin.get('xyz').split()],
                       [float(v) for v in origin.get('rpy').split()],
                       id(j) in guarded)
    return joints


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


def _chain(edges, ancestor, frame):
    """Pose of frame in ancestor, composed along the sensor_tf.yaml tree."""
    parent_of = {child: parent for (parent, child) in edges}
    path, target = [], frame
    while frame != ancestor:
        assert frame in parent_of, (
            "sensor_tf.yaml has no chain from '%s' down to '%s' (it stops at '%s')"
            % (ancestor, target, frame))
        path.append((parent_of[frame], frame))
        frame = parent_of[frame]
    rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    trans = [0.0, 0.0, 0.0]
    for key in reversed(path):
        e = edges[key]
        trans = [a + b for a, b in zip(trans, _matvec(rot, e['xyz']))]
        rot = _matmul(rot, _quat_to_matrix(e['qxyzw']))
    return rot, trans


def test_urdf_has_the_same_edges_as_sensor_tf():
    edges, joints = _sensor_tf(), _xacro_joints()
    assert set(joints) == set(edges), (
        'sensor_rig.xacro and sensor_tf.yaml list different edges. Only in the '
        'xacro: %s. Only in the YAML: %s.'
        % (sorted(set(joints) - set(edges)), sorted(set(edges) - set(joints))))


def test_urdf_origins_match_sensor_tf():
    edges, joints = _sensor_tf(), _xacro_joints()
    for key in sorted(set(edges) & set(joints)):
        xyz, rpy, _ = joints[key]
        want = edges[key]
        dt = max(abs(a - b) for a, b in zip(xyz, want['xyz']))
        da = _angle_deg(_rpy_to_matrix(rpy), _quat_to_matrix(want['qxyzw']))
        assert dt <= URDF_TRANS_TOL and da <= URDF_ROT_TOL, (
            '%s -> %s: sensor_rig.xacro is %.4f mm / %.5f deg away from '
            'sensor_tf.yaml. With publisher:=urdf (the default) the robot '
            'publishes the xacro value; update its <origin> from the YAML.'
            % (key[0], key[1], dt * 1000.0, da))


def test_urdf_scope_guard_matches_driver_ownership():
    # scope:=mount drops the owner:driver edges in static_tf_publisher; the
    # xacro does the same with <xacro:if value="${scope == 'all'}">. If the
    # two sets differ, publisher:=urdf and publisher:=yaml publish different
    # trees for the same scope.
    edges, joints = _sensor_tf(), _xacro_joints()
    driver = {k for k, e in edges.items() if e.get('owner') == 'driver'}
    guarded = {k for k, j in joints.items() if j[2]}
    assert guarded == driver, (
        "Edges only the xacro's scope=='all' block publishes: %s. Edges "
        'sensor_tf.yaml tags owner: driver: %s.' % (sorted(guarded), sorted(driver)))


@pytest.mark.parametrize('config_rel', LIO_CONFIGS)
def test_lio_extrinsic_matches_sensor_tf(config_rel):
    config_path = os.path.join(SRC, config_rel)
    if not os.path.exists(config_path):
        pytest.skip('not in this workspace: %s' % config_path)
    with open(config_path, encoding='utf-8') as f:
        doc = yaml.safe_load(f)
    body_frame = _find(doc, 'body_frame')
    ext_t, ext_r = _find(doc, 'extrinsic_T'), _find(doc, 'extrinsic_R')
    assert body_frame and ext_t and ext_r, (
        '%s: no publish.body_frame / extrinsic_T / extrinsic_R' % config_rel)

    # The config's convention: p_body = extrinsic_R * p_lidar + extrinsic_T,
    # the inverse of the body frame's pose in the lidar frame.
    rot, trans = _chain(_sensor_tf(), LIDAR_FRAME, body_frame)
    want_r = _transpose(rot)
    want_t = [-v for v in _matvec(want_r, trans)]
    got_r = [ext_r[3 * i:3 * i + 3] for i in range(3)]

    err_t = max(abs(a - b) for a, b in zip(ext_t, want_t))
    err_r = max(abs(got_r[i][j] - want_r[i][j]) for i in range(3) for j in range(3))
    assert err_t <= LIO_TOL and err_r <= LIO_TOL, (
        '%s: the %s -> %s extrinsic is off from sensor_tf.yaml by %.4f mm in T '
        'and %.2e in R. Recompose it from the YAML.\n  extrinsic_T: %s\n'
        '  extrinsic_R: %s'
        % (config_rel, LIDAR_FRAME, body_frame, err_t * 1000.0, err_r,
           ['%.6f' % v for v in want_t],
           ['%.6f' % v for row in want_r for v in row]))
