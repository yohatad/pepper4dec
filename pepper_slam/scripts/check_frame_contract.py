#!/usr/bin/env python3
"""
Asserts the LIO frame contract holds, whichever backend is running.

THE CONTRACT
  * odom          -- backend-native, OPAQUE. FAST-LIO leaves it at the IMU
                     mount orientation; Point-LIO can gravity-align it. Never
                     use it for geometry.
  * <level_frame> -- gravity-aligned AND floor-referenced (z=0 on the floor),
                     identical on every backend and every run. Everything
                     downstream standardises on this.

Because leveling is now derived from the rigid calibration rather than a
runtime sample (level_source=calibration), these are exact predictions, not
tolerances-of-convenience -- so this is a real assertion, not a smoke test.

    ros2 run pepper_slam check_frame_contract.py
    ros2 run pepper_slam check_frame_contract.py --ros-args \
        -p level_frame:=map -p expect_offset:=0.2571
"""
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
import rclpy.time


def rpy(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


class Check(Node):
    def __init__(self):
        super().__init__('check_frame_contract')
        self.declare_parameter('level_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('lidar_imu_frame', 'l2lidar_frame_imu')
        self.declare_parameter('cloud_topic', '/cloud_registered_body')
        # Predicted from the static calibration: t_base_imu.z
        self.declare_parameter('expect_offset', 0.2571)
        self.declare_parameter('tol_offset', 0.005)     # m
        self.declare_parameter('tol_rp', 3.0)           # deg
        self.declare_parameter('tol_floor', 0.10)       # m
        self.declare_parameter('duration', 25.0)

        def g(n):
            return self.get_parameter(n).value

        self.level, self.base = g('level_frame'), g('base_frame')
        self.imu_frame = g('lidar_imu_frame')
        self.expect, self.tol_o = g('expect_offset'), g('tol_offset')
        self.tol_rp, self.tol_floor = g('tol_rp'), g('tol_floor')

        self.buf = Buffer()
        self.lis = TransformListener(self.buf, self)
        self.create_subscription(PointCloud2, g('cloud_topic'), self.cb, 10)
        self.zs, self.rp = [], []
        self.duration = float(g('duration'))

    def cb(self, msg):
        try:
            t = self.buf.lookup_transform(self.level, msg.header.frame_id,
                                          rclpy.time.Time())
            b = self.buf.lookup_transform(self.level, self.base,
                                          rclpy.time.Time())
        except Exception:
            return
        q, tr = t.transform.rotation, t.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        pts = np.array([[p[0], p[1], p[2]] for p in
                        pc2.read_points(msg, ('x', 'y', 'z'), skip_nans=True)])
        if len(pts) == 0:
            return
        self.zs.append((pts @ R.T + [tr.x, tr.y, tr.z])[:, 2])
        self.rp.append(rpy(b.transform.rotation)[:2])

    def finish(self) -> int:
        fails = []
        print("\n===== LIO frame contract =====")
        try:
            lt = self.buf.lookup_transform(self.level, 'lio_init', rclpy.time.Time())
            off = abs(lt.transform.translation.z)
        except Exception:
            try:
                lt = self.buf.lookup_transform('lio_init', self.level, rclpy.time.Time())
                off = abs(lt.transform.translation.z)
            except Exception:
                off = None
        if off is None:
            fails.append(f"no transform between {self.level} and odom")
        else:
            ok = abs(off - self.expect) <= self.tol_o
            print(f"  leveling offset      {off:+.4f} m   expect {self.expect:+.4f}"
                  f"  [{'OK' if ok else 'FAIL'}]")
            if not ok:
                fails.append("leveling offset != calibration prediction")

        if self.rp:
            r, p = np.mean(self.rp, axis=0)
            ok = abs(r) <= self.tol_rp and abs(p) <= self.tol_rp
            print(f"  base roll/pitch      {r:+.2f} / {p:+.2f} deg   "
                  f"(<= {self.tol_rp})  [{'OK' if ok else 'FAIL'}]")
            if not ok:
                fails.append("base_frame not level in level_frame (drift or bad leveling)")
        else:
            fails.append("no base_frame samples")

        if self.zs:
            a = np.concatenate(self.zs)
            floor = float(np.median(a[a < np.percentile(a, 15)]))
            ok = abs(floor) <= self.tol_floor
            print(f"  floor plane z        {floor:+.3f} m   (|z| <= {self.tol_floor})"
                  f"  [{'OK' if ok else 'FAIL'}]")
            if not ok:
                fails.append("floor not at z=0 (level frame is not floor-referenced)")
        else:
            fails.append("no cloud samples -- is the cloud topic publishing?")

        print("=====", "PASS" if not fails else "FAIL: " + "; ".join(fails), "=====\n",
              flush=True)
        return 0 if not fails else 1


def main():
    import time
    rclpy.init()
    n = Check()
    rc = 1
    try:
        deadline = time.time() + n.duration
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(n, timeout_sec=0.1)
        rc = n.finish()
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()
