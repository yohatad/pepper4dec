#!/usr/bin/env python3
"""Rebuild the RealSense colour picture from /camera/depth/color/points.

The bags carry no image topic, but every point of the RealSense cloud has its
RGB and still sits on the depth camera's pixel grid (x/z and y/z step by one
pixel, 1/f). Projecting the points back gives the camera view, at the depth
stream's resolution; pixels without a valid depth stay black.

The focal length and grid offset are measured from the first cloud, so no
camera_info is needed. The camera streams 640x480, but the cloud holds every
second pixel (grid f = 194 px, half the D435i's ~385 px), so each point fills
an upsample x upsample block of the 640x480 output.

    python3 points_to_image.py --ros-args -p use_sim_time:=true
    -> /camera/points_rgb/image   (sensor_msgs/Image, rgb8, 640x480)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


class PointsToImage(Node):

    def __init__(self):
        super().__init__('points_to_image')
        self.declare_parameter('input', '/camera/depth/color/points')
        self.declare_parameter('output', '/camera/points_rgb/image')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('upsample', 2)       # camera pixels per cloud grid step
        self.declare_parameter('max_depth', 10.0)   # drops the 65.535 m invalid returns
        self.up = self.get_parameter('upsample').value
        # Render on the cloud's own grid, then scale up to the camera size.
        self.w = self.get_parameter('width').value // self.up
        self.h = self.get_parameter('height').value // self.up
        self.max_depth = self.get_parameter('max_depth').value
        self.cal = None   # (f, cx, cy)
        self.pub = self.create_publisher(Image, self.get_parameter('output').value, 5)
        self.create_subscription(PointCloud2, self.get_parameter('input').value,
                                 self.cb, qos_profile_sensor_data)

    def calibrate(self, u, v):
        # Pixel pitch = the smallest recurring gap between sorted x/z values.
        d = np.diff(np.sort(u))
        pitch = float(np.median(d[d > 1e-5]))
        f = 1.0 / pitch
        # Grid phase: u*f + cx must be an integer; centre the image on it.
        cx = self.w / 2 + np.angle(np.mean(np.exp(2j * np.pi * u * f))) / (2 * np.pi) * -1
        cy = self.h / 2 + np.angle(np.mean(np.exp(2j * np.pi * v * f))) / (2 * np.pi) * -1
        cx, cy = cx - round(cx - self.w / 2), cy - round(cy - self.h / 2)
        self.get_logger().info(f'calibrated from the cloud: f={f:.1f} px, '
                               f'cx={cx:.2f}, cy={cy:.2f}, grid {self.w}x{self.h}, '
                               f'output {self.w * self.up}x{self.h * self.up}')
        return f, cx, cy

    def cb(self, msg):
        a = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z', 'rgb'])
        x, y, z = a[:, 0], a[:, 1], a[:, 2]
        ok = (z > 0.05) & (z < self.max_depth)
        if ok.sum() < 100:
            return
        u, v = x[ok] / z[ok], y[ok] / z[ok]
        if self.cal is None:
            self.cal = self.calibrate(u, v)
        f, cx, cy = self.cal
        col = np.rint(u * f + cx).astype(int)
        row = np.rint(v * f + cy).astype(int)
        inside = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        rgb = a[ok, 3].astype(np.float32).view(np.uint32)[inside]
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[row[inside], col[inside]] = np.stack(
            [(rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255], axis=1).astype(np.uint8)
        if self.up > 1:
            img = img.repeat(self.up, axis=0).repeat(self.up, axis=1)

        out = Image()
        out.header = msg.header
        out.height, out.width = img.shape[:2]
        out.encoding = 'rgb8'
        out.step = img.shape[1] * 3
        out.data = img.tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(PointsToImage())


if __name__ == '__main__':
    main()
