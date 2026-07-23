#!/usr/bin/env python3
"""Compute the l2lidar_frame -> camera_camera_link static transform to publish
(or put in the URDF) from a direct_visual_lidar_calibration result.

WHY THIS EXISTS
---------------
direct_visual_lidar_calibration produces the extrinsic in the CAMERA'S OPTICAL
frame: l2lidar_frame <-> camera_color_optical_frame. But TF trees (and URDFs)
attach the camera at its MOUNT frame (camera_camera_link), not the optical leaf.
So the calibration must be "back-composed" through the RealSense internal chain:

    X = calib . (camera_camera_link -> camera_color_optical_frame)^-1

where the internal chain includes the ~90 deg optical-convention rotation
(color_frame -> optical_frame = [-0.5, 0.5, -0.5, 0.5]). Publishing the raw
calibration result directly onto camera_camera_link would be wrong by that 90 deg.

This script reads the camera's ACTUAL internal chain from a bag's /tf_static
(so it self-corrects for the real hardware -- a different RealSense model, firmware
or resolution changes the internals) and back-composes the calibration onto it.

USAGE
-----
    python3 compute_lidar_camera_bridge.py --bag ~/ros2_ws/bags/lidar_cam_calib2

    # override the calibration result (l2lidar <- camera_color_optical_frame):
    python3 compute_lidar_camera_bridge.py --bag <bag> \\
        --calib-xyz 0.015399 -0.038691 0.015828 \\
        --calib-quat -0.006724 -0.012757 0.535023 0.844715

    # different frame names:
    python3 compute_lidar_camera_bridge.py --bag <bag> \\
        --lidar-frame l2lidar_frame \\
        --cam-mount camera_camera_link \\
        --cam-optical camera_color_optical_frame

The bag must contain a /tf_static that captured the camera internal chain
(camera_camera_link -> ... -> camera_color_optical_frame). lidar_cam_calib2 has it;
slam_recording does NOT (its /tf_static is empty -- a rosbag2 transient_local
capture race). On the real robot, point --bag at a fresh bag that recorded
/tf_static, or extend this to read live TF.

Prints the result as URDF <origin> and as a ready-to-run
static_transform_publisher command.
"""

import argparse
import glob
import math
import sqlite3
import sys
from collections import deque

import numpy as np


def q2R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R2q(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def T(t, q):
    M = np.eye(4)
    M[:3, :3] = q2R(*q)
    M[:3, 3] = t
    return M


def quat_to_rpy(x, y, z, w):
    """URDF rpy (fixed-axis XYZ) from a quaternion."""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def read_static_tree(bag_path):
    """Read every /tf_static transform into an undirected adjacency map:
    frame -> [(neighbor, 4x4 T_this_to_neighbor)]."""
    from rosidl_runtime_py.utilities import get_message
    from rclpy.serialization import deserialize_message

    dbs = glob.glob(f'{bag_path}/*.db3')
    if not dbs:
        sys.exit(f'no .db3 in {bag_path}')
    con = sqlite3.connect(dbs[0])
    row = con.execute("select id,type from topics where name='/tf_static'").fetchone()
    if not row:
        sys.exit('bag has no /tf_static topic')
    tid, ttype = row
    msgcls = get_message(ttype)
    adj = {}
    n = 0
    for (raw,) in con.execute(f'select data from messages where topic_id={tid}'):
        m = deserialize_message(raw, msgcls)
        for tf in m.transforms:
            p, c = tf.header.frame_id, tf.child_frame_id
            tr, r = tf.transform.translation, tf.transform.rotation
            M = T([tr.x, tr.y, tr.z], [r.x, r.y, r.z, r.w])
            adj.setdefault(p, []).append((c, M))
            adj.setdefault(c, []).append((p, np.linalg.inv(M)))
            n += 1
    if n == 0:
        sys.exit('/tf_static is empty in this bag (the transient_local capture '
                 'race) -- use a bag that captured it, e.g. lidar_cam_calib2')
    return adj


def chain(adj, src, dst):
    """Compose the transform src -> dst by BFS over the static tree."""
    prev = {src: (None, None)}
    q = deque([src])
    while q:
        n = q.popleft()
        if n == dst:
            break
        for nb, M in adj.get(n, []):
            if nb not in prev:
                prev[nb] = (n, M)
                q.append(nb)
    if dst not in prev:
        sys.exit(f'no path {src} -> {dst} in /tf_static')
    steps = []
    n = dst
    while prev[n][0] is not None:
        parent, M = prev[n]
        steps.append(M)
        n = parent
    steps.reverse()
    out = np.eye(4)
    for M in steps:
        out = out @ M
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bag', required=True,
                    help='rosbag2 dir whose /tf_static has the camera internal chain')
    ap.add_argument('--lidar-frame', default='l2lidar_frame')
    ap.add_argument('--cam-mount', default='camera_camera_link',
                    help='frame the bridge/URDF joint attaches to')
    ap.add_argument('--cam-optical', default='camera_color_optical_frame',
                    help='frame the calibration was computed in')
    ap.add_argument('--calib-xyz', nargs=3, type=float,
                    default=[0.015399, -0.038691, 0.015828],
                    help='calibration translation (lidar <- optical)')
    ap.add_argument('--calib-quat', nargs=4, type=float,
                    default=[-0.006724, -0.012757, 0.535023, 0.844715],
                    help='calibration quaternion xyzw (lidar <- optical)')
    args = ap.parse_args()

    adj = read_static_tree(args.bag)
    cam = chain(adj, args.cam_mount, args.cam_optical)
    calib = T(args.calib_xyz, args.calib_quat)

    # X = calib . (cam_mount -> cam_optical)^-1  =>  lidar -> cam_mount
    X = calib @ np.linalg.inv(cam)
    t = X[:3, 3]
    q = R2q(X[:3, :3])
    rpy = quat_to_rpy(*q)

    # sanity: X . cam must reproduce calib
    err = np.linalg.norm((X @ cam)[:3, 3] - calib[:3, 3]) * 1000

    print(f'\ncamera internal chain {args.cam_mount} -> {args.cam_optical} read from '
          f'{args.bag}/tf_static')
    print(f'round-trip check (X . chain == calib): {err:.4f} mm\n')
    print(f'=== {args.lidar_frame} -> {args.cam_mount} ===')
    print(f'  xyz  = {t[0]:.6f} {t[1]:.6f} {t[2]:.6f}')
    print(f'  quat = {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}  (xyzw)')
    print(f'  rpy  = {rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}  (for URDF)\n')
    print('URDF joint origin:')
    print(f'  <origin xyz="{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}" '
          f'rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>\n')
    print('static_transform_publisher:')
    print(f'  ros2 run tf2_ros static_transform_publisher \\\n'
          f'    --x {t[0]:.6f} --y {t[1]:.6f} --z {t[2]:.6f} \\\n'
          f'    --qx {q[0]:.6f} --qy {q[1]:.6f} --qz {q[2]:.6f} --qw {q[3]:.6f} \\\n'
          f'    --frame-id {args.lidar_frame} --child-frame-id {args.cam_mount}\n')


if __name__ == '__main__':
    main()
