#!/usr/bin/env python3
"""
Republish a raw LIO odometry topic, rotated into the gravity-level odom frame.

Takes FAST-LIO's /Odometry, Point-LIO's /aft_mapped_to_init, or any backend
routed through lio_map_odom_bridge.py, and emits a full nav_msgs/Odometry for
feeding into an EKF (see ekf_lio_wheel.yaml) alongside wheel odometry.

Position-only leveling isn't enough for a state estimator: the ORIENTATION
also needs to be expressed in odom, not raw odom, or roll/pitch fusion
against wheel odometry is comparing two different reference frames. This is
the full-pose version of leveled_pose_publisher.py.

Usage:
    python3 leveled_odometry_publisher.py --ros-args \
        -p odom_topic:=/Odometry -p output_topic:=/odometry/lio_leveled
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
import tf2_ros


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r):
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)


class LeveledOdometryPublisher(Node):

    def __init__(self):
        super().__init__("leveled_odometry_publisher")
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("output_topic", "/odometry/lio_leveled")
        odom_topic = self.get_parameter("odom_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.R = None
        self.Tr = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(Odometry, output_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._cb, 50)
        self.get_logger().info(
            f"Republishing {odom_topic} (leveled into odom) as {output_topic}")

    def _ensure_transform(self):
        if self.R is not None:
            return True
        try:
            t = self.tf_buffer.lookup_transform("odom", "odom_lidar", Time())
        except Exception:
            return False
        q = t.transform.rotation
        self.R = quat_to_matrix(q.x, q.y, q.z, q.w)
        self.Tr = np.array([t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z])
        self.get_logger().info(f"Got odom->odom transform: R diag={np.diag(self.R)}")
        return True

    def _cb(self, msg):
        if not self._ensure_transform():
            return
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        raw_p = np.array([p.x, p.y, p.z])
        raw_r = quat_to_matrix(q.x, q.y, q.z, q.w)

        leveled_p = self.R @ raw_p + self.Tr
        leveled_r = self.R @ raw_r
        leveled_q = matrix_to_quat(leveled_r)

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "odom"
        out.child_frame_id = "base_footprint"
        out.pose.pose.position.x = float(leveled_p[0])
        out.pose.pose.position.y = float(leveled_p[1])
        out.pose.pose.position.z = float(leveled_p[2])
        out.pose.pose.orientation.x = float(leveled_q[0])
        out.pose.pose.orientation.y = float(leveled_q[1])
        out.pose.pose.orientation.z = float(leveled_q[2])
        out.pose.pose.orientation.w = float(leveled_q[3])
        # Modest, uniform covariance -- odom0_config in the EKF selects which
        # fields actually get fused (x, y, yaw only), so the exact values
        # here matter less than the config mask does.
        out.pose.covariance[0] = 0.01   # x
        out.pose.covariance[7] = 0.01   # y
        out.pose.covariance[14] = 0.05  # z (unused by config mask)
        out.pose.covariance[21] = 0.05  # roll (unused)
        out.pose.covariance[28] = 0.05  # pitch (unused)
        out.pose.covariance[35] = 0.02  # yaw
        self.pub.publish(out)


def main():
    rclpy.init()
    node = LeveledOdometryPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
