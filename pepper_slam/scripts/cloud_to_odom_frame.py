#!/usr/bin/env python3
"""
Republish a lidar scan expressed in an odometry frame.

global_localization assumes cloud_topic is ALREADY in odom_frame -- it forms
scan_in_map as (map -> odom) * scan and never consults TF for the scan itself.
With FAST-LIO that holds for free: /cloud_registered is published in the LIO
world frame, which is the odom frame. There is no such topic for wheel odometry,
so evaluating the localizer against /pepper_odom needs the raw scan carried into
the pepper_odom frame first. That is all this node does.

WHY BOTHER: FAST-LIO odometry is good enough that the ICP correction is almost
never asked to do real work, which makes it a weak test of the localizer.
Wheel odometry drifts hard, so it exercises the part under test.

NOT DESKEWED. FAST-LIO removes motion distortion using the IMU before
publishing /cloud_registered; the raw /points has none of that, so every scan is
smeared by whatever the robot did during the sweep -- worst while turning. Treat
a result from this node as a lower bound on the localizer's accuracy, not a
measurement of it.

Needs the bag's /tf (pepper_odom -> base_footprint) replayed, and the rig's
static transforms. Do NOT run lio_map_odom_bridge alongside it: that publishes
lio_init -> base_footprint, and a second live parent for base_footprint is a
broken tree (see pepper_odom_relabel.py).
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import tf2_ros
from sensor_msgs.msg import PointCloud2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class CloudToOdomFrame(Node):
    def __init__(self):
        super().__init__('cloud_to_odom_frame')
        self.declare_parameter('input_topic', '/points')
        self.declare_parameter('output_topic', '/cloud_in_odom')
        self.declare_parameter('target_frame', 'pepper_odom')
        # The scan and the TF that carries it are published by different
        # sources at different rates, so the exact stamp is often not in the
        # buffer yet. Waiting the full lidar period would stall the pipeline;
        # falling back to the latest available transform costs at most one
        # scan period of pose error, which is far below the drift being tested.
        self.declare_parameter('lookup_timeout', 0.05)
        self.declare_parameter('allow_latest_fallback', True)

        self.target = self.get_parameter('target_frame').value
        self.timeout = float(self.get_parameter('lookup_timeout').value)
        self.fallback = bool(self.get_parameter('allow_latest_fallback').value)

        self.buf = tf2_ros.Buffer()
        self.lis = tf2_ros.TransformListener(self.buf, self)

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(
            PointCloud2, self.get_parameter('output_topic').value, qos)
        self.create_subscription(
            PointCloud2, self.get_parameter('input_topic').value, self.cb, qos)

        self.n_out = 0
        self.n_drop = 0
        self.get_logger().info(
            f"Carrying {self.get_parameter('input_topic').value} into "
            f"'{self.target}' -> {self.get_parameter('output_topic').value}. "
            f"NOT deskewed; see the module docstring.")

    def cb(self, msg):
        try:
            tf = self.buf.lookup_transform(
                self.target, msg.header.frame_id, msg.header.stamp,
                rclpy.duration.Duration(seconds=self.timeout))
        except tf2_ros.TransformException:
            if not self.fallback:
                self._drop()
                return
            try:
                tf = self.buf.lookup_transform(
                    self.target, msg.header.frame_id, rclpy.time.Time())
            except tf2_ros.TransformException as ex:
                self._drop(ex)
                return
        out = do_transform_cloud(msg, tf)
        out.header.frame_id = self.target
        out.header.stamp = msg.header.stamp
        self.pub.publish(out)
        self.n_out += 1
        if self.n_out % 200 == 0:
            self.get_logger().info(
                f'{self.n_out} scans transformed, {self.n_drop} dropped.')

    def _drop(self, ex=None):
        self.n_drop += 1
        self.get_logger().warn(
            f'No transform {self.target} <- scan frame ({ex}); dropped '
            f'{self.n_drop}.', throttle_duration_sec=5.0)


def main():
    rclpy.init()
    n = CloudToOdomFrame()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()


if __name__ == '__main__':
    main()
