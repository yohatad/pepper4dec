#!/usr/bin/env python3
"""
measure_face_jitter.py: measure why the head shakes when a visitor is close.

Stand still in front of Pepper for the whole run, first close and then far
(or run it once per distance). For one window it reports:

  * face: depth, box width, and how much the face centroid jitters, in pixels
    and in degrees (angle = pixels / fx). Close faces are bigger, so the same
    relative jitter is more pixels and more degrees.
  * head commands from overt_attention (/joint_angles): how many per second,
    and how often the commanded yaw reverses direction (oscillation).
  * the 2 deg dead zone (min_angular_change_deg): the share of face updates
    whose target jump alone is larger than it, i.e. that send a command.

Usage (with the attention stack running):
    python3 measure_face_jitter.py --seconds 10 --label close
    python3 measure_face_jitter.py --seconds 10 --label far

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 27, 2026
Version: v1.0

Copyright (C) 2026 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import argparse
import math
import statistics
import time

from dec_interfaces.msg import FaceDetection
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

DEAD_ZONE_DEG = 2.0


def main():
    """Record one window and print the jitter and head-command report."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--label', default='')
    args = parser.parse_args()

    rclpy.init()
    node = Node('measure_face_jitter')
    faces, commands, info = [], [], {}

    def on_faces(msg):
        if msg.centroids:
            # the largest (closest) face
            i = max(range(len(msg.width)), key=lambda k: msg.width[k]) if msg.width else 0
            c = msg.centroids[i]
            faces.append((time.monotonic(), c.x, c.y, c.z,
                          msg.width[i] if i < len(msg.width) else float('nan')))

    def on_cmd(msg):
        if len(msg.joint_angles) >= 2:
            commands.append((time.monotonic(), msg.joint_angles[0], msg.joint_angles[1]))

    node.create_subscription(FaceDetection, '/face_detection/data', on_faces, 10)
    node.create_subscription(JointAnglesWithSpeed, '/joint_angles', on_cmd, 10)
    node.create_subscription(CameraInfo, '/pepper/front/camera_info',
                             lambda m: info.setdefault('fx', m.k[0]),
                             rclpy.qos.qos_profile_sensor_data)

    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    fx = info.get('fx', 525.0)

    label = f' [{args.label}]' if args.label else ''
    print(f'== {args.seconds:.0f} s window{label}')
    if len(faces) < 5:
        print(f'only {len(faces)} face detections: stand in view of the camera')
    else:
        xs = [f[1] for f in faces]
        ys = [f[2] for f in faces]
        depth = statistics.median(f[3] for f in faces)
        width = statistics.median(f[4] for f in faces)
        jumps = [math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(faces, faces[1:])]
        jumps_deg = [math.degrees(j / fx) for j in jumps]
        print(f'face: {len(faces)} detections, depth {depth:.2f} m, box width {width:.0f} px')
        print(f'centroid spread (std): x {statistics.pstdev(xs):.1f} px '
              f'({math.degrees(statistics.pstdev(xs) / fx):.2f} deg), '
              f'y {statistics.pstdev(ys):.1f} px '
              f'({math.degrees(statistics.pstdev(ys) / fx):.2f} deg)')
        print(f'frame-to-frame jump: median {statistics.median(jumps_deg):.2f} deg, '
              f'90th pct {sorted(jumps_deg)[int(0.9 * (len(jumps_deg) - 1))]:.2f} deg; '
              f'{100.0 * sum(j > DEAD_ZONE_DEG for j in jumps_deg) / len(jumps_deg):.0f}% '
              f'exceed the {DEAD_ZONE_DEG:.0f} deg dead zone')
    if commands:
        yaws = [c[1] for c in commands]
        steps = [b - a for a, b in zip(yaws, yaws[1:]) if abs(b - a) > 1e-4]
        reversals = sum(1 for a, b in zip(steps, steps[1:]) if a * b < 0)
        print(f'head commands: {len(commands) / args.seconds:.1f}/s, yaw range '
              f'{math.degrees(max(yaws) - min(yaws)):.1f} deg, '
              f'{reversals} direction reversals ({reversals / args.seconds:.1f}/s)')
    else:
        print('head commands: none')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
