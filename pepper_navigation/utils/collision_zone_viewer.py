#!/usr/bin/env python3
"""Live view of what the collision monitor would count, from the L2 alone.

Runs the same two cuts as the safety chain -- points_safety_filter's sensor
range cut, then collision_monitor's height band -- and publishes the result
split by fate, plus the stop/slow circles and a live count. Needs only the L2
and the rig TF; Nav2 does not have to be running.

    ros2 run ... / python3 collision_zone_viewer.py
    rviz2: Fixed Frame base_footprint, add PointCloud2 for the three
           /zone_view/* clouds and MarkerArray /zone_view/markers.

Tune live, values take effect on the next scan:
    ros2 param set /collision_zone_viewer min_range 0.22
    ros2 param set /collision_zone_viewer min_height 0.22
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation


class ZoneViewer(Node):
    def __init__(self):
        super().__init__('collision_zone_viewer')
        for name, val in (('input_topic', '/points'), ('base_frame', 'base_footprint'),
                          ('min_range', 0.22), ('min_height', 0.22), ('max_height', 1.80),
                          ('stop_radius', 0.40), ('slow_radius', 0.80), ('max_points', 3)):
            self.declare_parameter(name, val)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.pub = {k: self.create_publisher(PointCloud2, '/zone_view/' + k, 5)
                    for k in ('counted', 'cut_by_range', 'cut_by_height')}
        self.markers = self.create_publisher(MarkerArray, '/zone_view/markers', 5)
        self.create_subscription(PointCloud2, self.get_parameter('input_topic').value,
                                 self.cb, qos_profile_sensor_data)

    def p(self, name):
        return self.get_parameter(name).value

    def cb(self, msg):
        base = self.p('base_frame')
        try:
            t = self.tf.lookup_transform(base, msg.header.frame_id, rclpy.time.Time())
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'no TF {base} <- {msg.header.frame_id}: {e}',
                                   throttle_duration_sec=5.0)
            return
        pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'))
        pts = pts[np.isfinite(pts).all(1)]
        # Range cut happens in the SENSOR frame, like cloud_range_filter.
        in_range = np.linalg.norm(pts, axis=1) >= self.p('min_range')
        q, tr = t.transform.rotation, t.transform.translation
        b = pts @ Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().T \
            + np.array([tr.x, tr.y, tr.z])
        in_band = (b[:, 2] >= self.p('min_height')) & (b[:, 2] <= self.p('max_height'))
        counted = in_range & in_band
        hdr = Header(stamp=msg.header.stamp, frame_id=base)
        r = np.hypot(b[:, 0], b[:, 1])
        near = r < self.p('slow_radius') + 0.5   # keep the debug clouds light
        for key, mask in (('counted', counted), ('cut_by_range', ~in_range),
                          ('cut_by_height', in_range & ~in_band)):
            self.pub[key].publish(pc2.create_cloud_xyz32(hdr, b[mask & near].astype(np.float32)))

        n_stop = int((counted & (r < self.p('stop_radius'))).sum())
        n_slow = int((counted & (r < self.p('slow_radius'))).sum())
        mp = self.p('max_points')
        ma = MarkerArray()
        for i, (rad, n, rgb) in enumerate(((self.p('stop_radius'), n_stop, (1.0, 0.0, 0.0)),
                                           (self.p('slow_radius'), n_slow, (1.0, 0.6, 0.0)))):
            m = Marker(header=hdr, ns='zones', id=i, type=Marker.LINE_STRIP)
            m.scale.x = 0.02 if n > mp else 0.008
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 1.0 if n > mp else 0.4
            m.points = [Point(x=rad * np.cos(a), y=rad * np.sin(a), z=0.02)
                        for a in np.linspace(0, 2 * np.pi, 73)]
            ma.markers.append(m)
        txt = Marker(header=hdr, ns='zones', id=2, type=Marker.TEXT_VIEW_FACING)
        txt.pose.position.z = 1.2
        txt.scale.z = 0.08
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = (f"STOP {n_stop}{' FIRES' if n_stop > mp else ''}   "
                    f"SLOW {n_slow}{' FIRES' if n_slow > mp else ''}\n"
                    f"min_range {self.p('min_range'):.2f}  min_height {self.p('min_height'):.2f}")
        ma.markers.append(txt)
        self.markers.publish(ma)


def main():
    rclpy.init()
    rclpy.spin(ZoneViewer())


if __name__ == '__main__':
    main()
