#!/usr/bin/env python3
"""
Overlay LIO odometry and wheel odometry in one frame, and report both odometers.

They are not comparable as published. /odom_lio is expressed in lio_init, which
is the IMU MOUNT orientation at t=0 and is NOT gravity-aligned -- on this rig it
sits about 90 deg off. /pepper_odom is gravity-aligned and base-referenced.
Displaying both raw gives two unrelated squiggles.

The rotation that relates them is the same one everything else here levels by:
base_frame <- lidar_imu_frame, read from TF rather than hardcoded, so it follows
the rig instead of going stale (a hardcoded axis 2.43 deg off cost 3 m of height
elsewhere in this workspace). lio_init IS the body frame at t=0, so applying
that rotation carries LIO positions into base-referenced coordinates, which is
what /pepper_odom already reports -- yaw included, since the mount rotation
carries the heading offset too.

Both odometers start counting only once BOTH streams are live, and are
re-origined at that moment. FAST-LIO publishes nothing until IMU init completes
-- about 20 s into a replay -- while wheel odometry counts from t=0, so
accumulating each from its own first message compared different spans and read
ratio 0.63 on a bag whose true ratio is 1.01.

PATH LENGTH is the useful number, not displacement: this route loops, so the
robot finishes ~0.2 m from where it started after ~500 m of driving. Path length
is also robust to heading error -- a wrong heading points a step the wrong way
but does not change its length -- which is why wheel odometry is a fair witness
for distance travelled even though its POSITION drifts badly.

Publishes:
  /compare/lio_path     nav_msgs/Path   (leveled /odom_lio)
  /compare/wheel_path   nav_msgs/Path   (/pepper_odom)
  /compare/report       visualization_msgs/Marker  live odometer readout
all in compare_frame, which is published as a static child of base_frame's
gravity-aligned twin so RViz needs no extra setup.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class OdomCompare(Node):
    def __init__(self):
        super().__init__('odom_compare')
        self.declare_parameter('lio_topic', '/odom_lio')
        self.declare_parameter('wheel_topic', '/pepper_odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('lidar_imu_frame', 'camera_imu_optical_frame')
        self.declare_parameter('compare_frame', 'compare')
        # One pose per this many metres. The raw streams are 100-200 Hz; drawing
        # every sample makes RViz crawl and hides the shape under overdraw.
        self.declare_parameter('min_step', 0.05)

        self.base = self.get_parameter('base_frame').value
        self.imuf = self.get_parameter('lidar_imu_frame').value
        self.frame = self.get_parameter('compare_frame').value
        self.min_step = float(self.get_parameter('min_step').value)

        self.buf = tf2_ros.Buffer()
        self.lis = tf2_ros.TransformListener(self.buf, self)
        self.static_bc = tf2_ros.StaticTransformBroadcaster(self)
        self.R = None                      # base <- lidar_imu, resolved lazily

        self.paths = {}
        self.origin = {}
        self.dist = {'lio': 0.0, 'wheel': 0.0}
        self.last = {}
        self.seen = set()          # streams that have delivered a message
        self.started = False       # both live -> start counting, from here

        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub = {
            'lio': self.create_publisher(Path, '/compare/lio_path', 1),
            'wheel': self.create_publisher(Path, '/compare/wheel_path', 1)}
        self.pub_report = self.create_publisher(Marker, '/compare/report', 1)
        self.create_subscription(
            Odometry, self.get_parameter('lio_topic').value,
            lambda m: self.cb(m, 'lio'), qos)
        self.create_subscription(
            Odometry, self.get_parameter('wheel_topic').value,
            lambda m: self.cb(m, 'wheel'), qos)
        self.create_timer(1.0, self.report)
        self._sent_static = False

    def resolve(self):
        """Resolve base <- lidar_imu, which levels lio_init and carries its yaw."""
        if self.R is not None:
            return True
        try:
            tf = self.buf.lookup_transform(self.base, self.imuf,
                                           rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(
                f'waiting for {self.base} <- {self.imuf}: {ex}',
                throttle_duration_sec=5.0)
            return False
        self.R = quat_to_R(tf.transform.rotation)
        self.get_logger().info(
            f'leveling {self.imuf} -> {self.base} resolved; LIO odometry will '
            f'be shown in "{self.frame}" alongside wheel odometry.')
        return True

    def send_static(self, stamp):
        if self._sent_static:
            return
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.base
        t.child_frame_id = self.frame
        t.transform.rotation.w = 1.0
        self.static_bc.sendTransform(t)
        self._sent_static = True

    def cb(self, msg, which):
        if which == 'lio' and not self.resolve():
            return
        p = msg.pose.pose.position
        v = np.array([p.x, p.y, p.z])
        if which == 'lio':
            v = self.R @ v

        # Wait for both. Counting from each stream's own first message compares
        # different time spans, because the estimator needs IMU init before it
        # publishes anything at all.
        self.seen.add(which)
        if not self.started:
            if len(self.seen) < 2:
                return
            self.started = True
            self.origin.clear()
            self.last.clear()
            self.dist = {'lio': 0.0, 'wheel': 0.0}
            self.get_logger().info(
                'both odometry streams live; counting from here so the two '
                'cover the same span.')

        if which not in self.origin:
            self.origin[which] = v
            self.paths[which] = Path()
            self.paths[which].header.frame_id = self.frame
        v = v - self.origin[which]

        if which in self.last:
            step = float(np.linalg.norm(v - self.last[which]))
            if step < self.min_step:
                return
            self.dist[which] += step
        self.last[which] = v

        ps = PoseStamped()
        ps.header.frame_id = self.frame
        ps.header.stamp = msg.header.stamp
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, v)
        ps.pose.orientation.w = 1.0
        self.paths[which].poses.append(ps)
        self.paths[which].header.stamp = msg.header.stamp
        self.send_static(msg.header.stamp)
        self.pub[which].publish(self.paths[which])

    def report(self):
        if not self.dist['wheel']:
            return
        lio, wheel = self.dist['lio'], self.dist['wheel']
        ratio = lio / wheel if wheel else float('nan')
        txt = (f'LIO   {lio:7.1f} m\n'
               f'wheel {wheel:7.1f} m\n'
               f'ratio {ratio:6.3f}')
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.z = 3.0
        m.pose.orientation.w = 1.0
        m.scale.z = 0.8
        m.color.r = m.color.g = m.color.b = m.color.a = 1.0
        m.text = txt
        self.pub_report.publish(m)
        self.get_logger().info(
            f'path length -- LIO {lio:.1f} m, wheel {wheel:.1f} m, '
            f'ratio {ratio:.3f}', throttle_duration_sec=10.0)


def main():
    rclpy.init()
    n = OdomCompare()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()


if __name__ == '__main__':
    main()
