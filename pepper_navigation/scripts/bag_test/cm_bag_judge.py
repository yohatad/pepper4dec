#!/usr/bin/env python3
"""Score nav2's collision monitor against plain polygon geometry on a bag.

Test 1b of the safety-chain plan. For every /points_safety scan this node
computes the verdict the monitor SHOULD reach -- the cloud moved into
base_footprint, the height band applied, points counted inside each polygon
with nav2's own ray-casting rule, "> max_points" triggers, STOP beats SLOWDOWN
-- and pairs it with the verdict the monitor DID reach, read off its output
velocity. Humble's monitor has no state topic, so the judge drives it with a
constant cmd_vel_in and decodes cmd_vel_out: vx unchanged = none, vx *
slowdown_ratio = slowdown, 0 = stop.

The polygons, height band and max_points are read from the same params file
the monitor loads, so the reference cannot drift from what is being tested.

The monitor's verdict for scan k is the last output received before scan k+1
arrives, so a transition can land one scan late; the summary reports exact
agreement and agreement within one scan.

Prints a summary every 30 s and at shutdown, and writes a per-scan CSV and a
JSON summary to out_prefix.
"""
import csv
import json
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
import yaml

NONE, SLOW, STOP = 'none', 'slowdown', 'stop'
ORDER = (NONE, SLOW, STOP)


def inside(poly, x, y):
    """Vectorised copy of nav2_collision_monitor Polygon::isPointInside (Humble)."""
    res = np.zeros(x.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (y <= yi) == (y > yj)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_inter = xi + (y - yi) * (xj - xi) / (yj - yi)
        res ^= crosses & (x_inter > x)
        j = i
    return res


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Judge(Node):
    def __init__(self):
        super().__init__('cm_bag_judge')
        self.declare_parameter('params_file', '')
        self.declare_parameter('cloud_topic', '/points_safety')
        self.declare_parameter('cmd_in_topic', '/cm_bag/cmd_vel_in')
        self.declare_parameter('cmd_out_topic', '/cm_bag/cmd_vel_out')
        self.declare_parameter('vx', 0.35)
        self.declare_parameter('rate', 50.0)
        self.declare_parameter('out_prefix', '/tmp/cm_bag')
        g = self.get_parameter

        with open(os.path.expanduser(g('params_file').value), encoding='utf-8') as f:
            cm = yaml.safe_load(f)['collision_monitor']['ros__parameters']
        self.base = cm['base_frame_id']
        src = cm[cm['observation_sources'][0]]
        self.zmin, self.zmax = src['min_height'], src['max_height']
        self.polys = []
        for name in cm['polygons']:
            p = cm[name]
            if not p.get('enabled', True):
                continue
            pts = p['points']
            self.polys.append({
                'name': name, 'action': p['action_type'], 'max': p['max_points'],
                'ratio': p.get('slowdown_ratio', 1.0),
                'poly': list(zip(pts[0::2], pts[1::2]))})
        self.vx = g('vx').value
        slow = [p['ratio'] for p in self.polys if p['action'] == 'slowdown']
        self.slow_vx = self.vx * slow[0] if slow else None

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.cmd_pub = self.create_publisher(Twist, g('cmd_in_topic').value, 10)
        self.create_subscription(Twist, g('cmd_out_topic').value, self.on_out, 50)
        self.create_subscription(PointCloud2, g('cloud_topic').value, self.on_cloud, 10)
        self.create_timer(1.0 / g('rate').value, self.tick)
        self.create_timer(30.0, self.report, clock=rclpy.clock.Clock())  # wall

        self.out_prefix = g('out_prefix').value
        self.csv_file = open(self.out_prefix + '_scans.csv', 'w', newline='')
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(['stamp', 'n_points', 'n_in_band'] +
                          ['n_' + p['name'] for p in self.polys] +
                          ['reference', 'monitor'])
        self.pending = None        # (row, reference) awaiting the monitor's verdict
        self.last_out = None
        self.outs_since_cloud = 0
        self.warmup = 0            # scans before the monitor's first output
        self.rows = []             # (reference, monitor)
        self.tf_fail = 0
        self.no_output = 0
        self.get_logger().info(
            f'judging {[p["name"] for p in self.polys]} in {self.base}, '
            f'z {self.zmin}..{self.zmax}, vx {self.vx}; writing {self.out_prefix}_*')

    def tick(self):
        t = Twist()
        t.linear.x = self.vx
        self.cmd_pub.publish(t)

    def on_out(self, msg):
        self.last_out = msg.linear.x
        self.outs_since_cloud += 1

    def decode(self, vx):
        if vx is None:
            return None
        if abs(vx) < 1e-6:
            return STOP
        if self.slow_vx is not None and abs(vx - self.slow_vx) < 1e-4:
            return SLOW
        if abs(vx - self.vx) < 1e-4:
            return NONE
        return f'other({vx:.4f})'

    def on_cloud(self, msg):
        if self.last_out is None:
            # Monitor not up yet (lifecycle activation, or waiting on TF):
            # scoring these would charge startup to the monitor.
            self.warmup += 1
            return
        # The previous scan's verdict is whatever the monitor said since then.
        if self.pending is not None:
            row, ref = self.pending
            mon = self.decode(self.last_out) if self.outs_since_cloud else None
            if mon is None:
                self.no_output += 1
                mon = 'no_output'
            self.csv.writerow(row + [ref, mon])
            self.rows.append((ref, mon))
            self.pending = None
        self.outs_since_cloud = 0

        try:
            tf = self.tf.lookup_transform(self.base, msg.header.frame_id, Time(),
                                          timeout=Duration(seconds=0.0))
        except TransformException:
            self.tf_fail += 1
            return
        xyz = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'))
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        r = tf.transform.translation
        xyz = xyz @ quat_to_matrix(tf.transform.rotation).T + np.array([r.x, r.y, r.z])
        band = (xyz[:, 2] >= self.zmin) & (xyz[:, 2] <= self.zmax)  # NaN fails both
        x, y = xyz[band, 0], xyz[band, 1]
        counts, ref = [], NONE
        for p in self.polys:
            n = int(inside(p['poly'], x, y).sum())
            counts.append(n)
            if n > p['max']:
                if p['action'] == 'stop':
                    ref = STOP
                elif p['action'] == 'slowdown' and ref != STOP:
                    ref = SLOW
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pending = ([f'{stamp:.3f}', len(xyz), int(band.sum())] + counts, ref)

    def summary(self):
        n = len(self.rows)
        labels = list(ORDER) + sorted({m for _, m in self.rows} - set(ORDER))
        matrix = {r: {m: 0 for m in labels} for r in ORDER}
        for r, m in self.rows:
            matrix[r][m] += 1
        exact = sum(1 for r, m in self.rows if r == m)
        # within one scan: the monitor's verdict matches the reference of the
        # same scan or a neighbour (a transition landing one scan late)
        near = 0
        for k, (_, m) in enumerate(self.rows):
            refs = {self.rows[i][0] for i in range(max(0, k - 1), min(n, k + 2))}
            near += m in refs

        def pct(c):
            return round(100.0 * c / n, 2) if n else 0.0
        return {
            'scans': n,
            'reference_pct': {a: pct(sum(1 for r, _ in self.rows if r == a)) for a in ORDER},
            'monitor_pct': {a: pct(sum(1 for _, m in self.rows if m == a)) for a in labels},
            'agree_exact_pct': pct(exact),
            'agree_within_1_scan_pct': pct(near),
            'matrix_reference_rows_monitor_cols': matrix,
            'scans_without_monitor_output': self.no_output,
            'scans_skipped_no_tf': self.tf_fail,
            'scans_before_first_monitor_output': self.warmup,
        }

    def report(self):
        s = self.summary()
        if s['scans']:
            self.get_logger().info(
                f"{s['scans']} scans | reference {s['reference_pct']} | "
                f"monitor {s['monitor_pct']} | agree {s['agree_exact_pct']}% "
                f"(+-1 scan {s['agree_within_1_scan_pct']}%)")

    def close(self):
        s = self.summary()
        self.csv_file.close()
        with open(self.out_prefix + '_summary.json', 'w') as f:
            json.dump(s, f, indent=1)
        print(json.dumps(s, indent=1), flush=True)


def main():
    rclpy.init()
    node = Judge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
