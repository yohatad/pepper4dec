#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_srvs.srv import Empty


class OdomPathPublisher(Node):
    def __init__(self):
        super().__init__('odom_path_publisher')

        self.declare_parameter('min_distance', 0.02)  # metres between recorded poses
        self.declare_parameter('min_angle', 0.05)     # radians between recorded poses

        self.min_distance = self.get_parameter('min_distance').value
        self.min_angle = self.get_parameter('min_angle').value

        self.path = Path()
        self.path.header.frame_id = 'pepper_odom'
        self.start_pose = None
        self.pose_count = 0

        self.sub = self.create_subscription(Odometry, '/pepper_odom_filtered', self._odom_cb, 10)
        self.path_pub = self.create_publisher(Path, '/odom_path', 10)
        self.marker_pub = self.create_publisher(Marker, '/odom_markers', 10)
        self.create_service(Empty, 'reset_odom_path', self._reset_cb)

        self.get_logger().info(
            'Odometry path publisher started.\n'
            '  Path topic  : /odom_path\n'
            '  Markers     : /odom_markers  (green sphere = START)\n'
            '  Reset service: ros2 service call /reset_odom_path std_srvs/srv/Empty'
        )

    # ------------------------------------------------------------------ #
    def _odom_cb(self, msg: Odometry) -> None:
        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = 'pepper_odom'
        ps.pose = msg.pose.pose

        if not self.path.poses:
            self.start_pose = ps
            self.path.poses.append(ps)
            self._publish_start_markers(ps)
            self._publish_path(msg.header.stamp)
            return

        last = self.path.poses[-1].pose
        dx = ps.pose.position.x - last.position.x
        dy = ps.pose.position.y - last.position.y
        dist = math.hypot(dx, dy)
        dyaw = abs(self._yaw(ps.pose.orientation) - self._yaw(last.orientation))

        if dist < self.min_distance and dyaw < self.min_angle:
            return

        self.path.poses.append(ps)
        self.pose_count += 1
        self._publish_path(msg.header.stamp)

        # Update end marker and log deviation every 50 new poses
        if self.pose_count % 50 == 0:
            self._publish_end_marker(ps)
            self._log_deviation(ps)

    # ------------------------------------------------------------------ #
    def _publish_path(self, stamp) -> None:
        self.path.header.stamp = stamp
        self.path_pub.publish(self.path)

    def _publish_start_markers(self, ps: PoseStamped) -> None:
        # Green sphere at start position
        m = self._make_marker(ps, 0, Marker.SPHERE, (0.2, 0.2, 0.2), (0.0, 1.0, 0.0))
        self.marker_pub.publish(m)

        # "START" text label above the sphere
        t = self._make_marker(ps, 1, Marker.TEXT_VIEW_FACING, (0.0, 0.0, 0.15), (1.0, 1.0, 1.0))
        t.pose.position.z += 0.35
        t.text = 'START'
        self.marker_pub.publish(t)

    def _publish_end_marker(self, ps: PoseStamped) -> None:
        # Yellow sphere showing current position relative to start
        m = self._make_marker(ps, 2, Marker.SPHERE, (0.15, 0.15, 0.15), (1.0, 1.0, 0.0))
        self.marker_pub.publish(m)

        # Deviation line from start to current position
        if self.start_pose is None:
            return
        line = Marker()
        line.header = ps.header
        line.ns = 'odom_deviation'
        line.id = 3
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.03
        line.color.r = 1.0
        line.color.g = 0.4
        line.color.b = 0.0
        line.color.a = 0.9
        line.lifetime.sec = 0
        from geometry_msgs.msg import Point
        p0 = Point()
        p0.x = self.start_pose.pose.position.x
        p0.y = self.start_pose.pose.position.y
        p0.z = 0.0
        p1 = Point()
        p1.x = ps.pose.position.x
        p1.y = ps.pose.position.y
        p1.z = 0.0
        line.points = [p0, p1]
        self.marker_pub.publish(line)

    def _log_deviation(self, ps: PoseStamped) -> None:
        if self.start_pose is None:
            return
        sx = self.start_pose.pose.position.x
        sy = self.start_pose.pose.position.y
        cx = ps.pose.position.x
        cy = ps.pose.position.y
        dev = math.hypot(cx - sx, cy - sy)
        self.get_logger().info(
            f'Deviation from start: {dev:.4f} m  |  '
            f'pos=({cx:.3f}, {cy:.3f})  |  '
            f'path points: {len(self.path.poses)}'
        )

    # ------------------------------------------------------------------ #
    def _reset_cb(self, _req, response):
        self.path.poses.clear()
        self.start_pose = None
        self.pose_count = 0
        self.get_logger().info('Path reset — waiting for next /pepper_odom_filtered message.')
        return response

    # ------------------------------------------------------------------ #
    @staticmethod
    def _yaw(q) -> float:
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def _make_marker(self, ps: PoseStamped, mid: int, mtype: int,
                     scale: tuple, color: tuple) -> Marker:
        m = Marker()
        m.header = ps.header
        m.ns = 'odom_start'
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose = ps.pose
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color.r, m.color.g, m.color.b = color
        m.color.a = 1.0
        m.lifetime.sec = 0  # persistent until reset
        return m


def main(args=None):
    rclpy.init(args=args)
    node = OdomPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
