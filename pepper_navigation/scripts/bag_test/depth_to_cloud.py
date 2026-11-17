#!/usr/bin/env python3
"""Make an RGB point cloud from aligned depth + colour images (old bags).

slam_20260823_aligned and other older recordings carry the RealSense as
images, not as /camera/depth/color/points:

    /camera/aligned_depth_to_color/image_raw   16UC1, millimetres
    /camera/color/image_raw                    rgb8
    /camera/color/camera_info                  intrinsics

This node projects them into the cloud the costmap expects. Same job as
depth_image_proc's point_cloud_xyzrgb, which is not installed here.

    python3 depth_to_cloud.py --ros-args -p use_sim_time:=true
    -> /bag/camera/depth/color/points   (feeds cloud_delay.py)

Set -p output:=/camera/depth/color/points to publish directly instead.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2


class DepthToCloud(Node):

    def __init__(self):
        super().__init__('depth_to_cloud')
        self.declare_parameter('depth', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('color', '/camera/color/image_raw')
        self.declare_parameter('info', '/camera/color/camera_info')
        self.declare_parameter('output', '/bag/camera/depth/color/points')
        self.declare_parameter('stride', 2)          # every 2nd pixel, like the live cloud
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('max_pair_dt', 0.05)  # s between depth and colour stamps
        self.stride = self.get_parameter('stride').value
        self.max_depth = self.get_parameter('max_depth').value
        self.max_dt = self.get_parameter('max_pair_dt').value
        self.K = None
        self.colors = []      # recent colour frames: (stamp, image)
        self.pub = self.create_publisher(PointCloud2, self.get_parameter('output').value,
                                         qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.get_parameter('info').value,
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.get_parameter('color').value,
                                 self.on_color, qos_profile_sensor_data)
        self.create_subscription(Image, self.get_parameter('depth').value,
                                 self.on_depth, qos_profile_sensor_data)

    @staticmethod
    def stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def on_info(self, msg):
        if self.K is None:
            self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])   # fx, fy, cx, cy
            self.get_logger().info('intrinsics: fx=%.1f fy=%.1f cx=%.1f cy=%.1f, %dx%d'
                                   % (self.K + (msg.width, msg.height)))

    def on_color(self, msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            self.get_logger().warn(f'colour encoding {msg.encoding} not handled',
                                   throttle_duration_sec=10.0)
            return
        # Honour msg.step: rows may be padded beyond width * 3.
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            img = img[:, :, ::-1]
        self.colors.append((self.stamp(msg), img))
        del self.colors[:-30]

    def on_depth(self, msg):
        if self.K is None or not self.colors:
            return
        t = self.stamp(msg)
        st, colour = min(self.colors, key=lambda c: abs(c[0] - t))
        if abs(st - t) > self.max_dt:
            return
        if msg.encoding not in ('16UC1', 'mono16'):
            self.get_logger().warn(f'depth encoding {msg.encoding} not handled',
                                   throttle_duration_sec=10.0)
            return
        d = np.frombuffer(msg.data, np.dtype(np.uint16).newbyteorder(
            '>' if msg.is_bigendian else '<')).reshape(msg.height, msg.step // 2)
        d = d[:, :msg.width]
        s = self.stride
        d = d[::s, ::s].astype(np.float32) * 0.001          # mm -> m
        colour = colour[::s, ::s]
        if colour.shape[:2] != d.shape:
            return
        fx, fy, cx, cy = self.K
        v, u = np.nonzero((d > 0.05) & (d < self.max_depth))  # 0 = no return (reflections)
        z = d[v, u]
        x = (u * s - cx) / fx * z
        y = (v * s - cy) / fy * z
        c = colour[v, u].astype(np.uint32)
        rgb = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)

        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate('xyz')]
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
        self.pub.publish(pc2.create_cloud(
            msg.header, fields,
            np.c_[x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), rgb]))


def main():
    rclpy.init()
    rclpy.spin(DepthToCloud())


if __name__ == '__main__':
    main()
