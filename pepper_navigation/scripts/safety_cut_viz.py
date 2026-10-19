#!/usr/bin/env python3
"""Colour every L2 point by what the safety chain does with it (for RViz).

Mirrors the two stages between the lidar and a STOP:
  points_safety_filter (pepper_slam/cloud_range_filter)
      min_range       3D range from the LIDAR, in l2lidar_frame
      self_cut_range  3D range from the LIDAR, cut only OUTSIDE the forward
                      cone of +-self_cut_angle (bearing in base_footprint)
      body_boxes      boxes on Pepper's own outline, in base_footprint
  collision_monitor (nav2, observation source "pointcloud")
      min/max_height  z in base_footprint
      PolygonStop / PolygonSlow, "> max_points" triggers, STOP beats SLOWDOWN
The polygons are read from the collision_monitor section of params_file, so
they match what the monitor loads. The filter values default to the ones in
pepper_nav2_fastloc.launch.py; change them there and here together.

Each point gets the colour of the FIRST stage that removes it:
  /safety_cut_viz/min_range  blue     cut by min_range (lidar housing, near body)
  /safety_cut_viz/self_cut   magenta  cut by the side-body cut
  /safety_cut_viz/body_box   orange   cut by the body-box cut (Pepper's front)
  /safety_cut_viz/height     grey     kept by the filter, outside the monitor's
                                      height band (floor, Pepper's base)
  /safety_cut_viz/stop       red      reaches the monitor, inside PolygonStop
  /safety_cut_viz/slow       yellow   reaches the monitor, inside PolygonSlow only
  /safety_cut_viz/clear      green    reaches the monitor, outside both polygons
  /safety_cut_viz/markers    the polygons, the min_range sphere, the self-cut
      ring at lidar height, the body boxes and the verdict the monitor would
      reach.
All clouds are published already in base_footprint, so RViz needs no TF.

TF: base_footprint <- l2lidar_frame comes from pepper_slam's
pepper_sensor_tf.launch.py. When it is not running, this node falls back to
sensor_tf.yaml and says so in the verdict -- the real collision monitor
would IGNORE every cloud in that state (no STOP at all).

    ros2 launch pepper_navigation safety_cut_viz.launch.py

Every threshold is live: `ros2 param set /safety_cut_viz min_range 0.12`
recolours the next scan, so a value can be tried on the robot before it goes
into the launch files.
"""
import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import ColorRGBA
import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
import yaml

CLASSES = {  # name: rgb, in the order the stages remove points
    'min_range': (30, 110, 255),
    'self_cut': (220, 40, 220),
    'body_box': (255, 140, 0),
    'height': (150, 150, 150),
    'stop': (230, 30, 30),
    'slow': (255, 200, 0),
    'clear': (40, 200, 60),
}


def inside(poly, x, y):
    """Vectorised copy of nav2_collision_monitor Polygon::isPointInside (Humble)."""
    res = np.zeros(x.shape, dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (y <= yi) == (y > yj)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_inter = xi + (y - yi) * (xj - xi) / (yj - yi)
        res ^= crosses & (x_inter > x)
        j = i
    return res


def rgb_float(rgb):
    r, g, b = rgb
    return np.array([(r << 16) | (g << 8) | b], np.uint32).view(np.float32)[0]


class SafetyCutViz(Node):

    def __init__(self):
        super().__init__('safety_cut_viz')
        nav = get_package_share_directory('pepper_navigation')
        slam = get_package_share_directory('pepper_slam')
        p = {'base_frame': 'base_footprint', 'cloud_topic': '/points',
             # points_safety_filter, as in pepper_nav2_fastloc.launch.py
             'min_range': 0.12, 'self_cut_range': 0.30, 'self_cut_angle': 25.0,
             'body_boxes': [0.10, 0.22, -0.13, 0.13, 0.60, 1.00],
             'params_file': os.path.join(nav, 'config', 'nav2_params_fastloc.yaml'),
             'transforms_file': os.path.join(slam, 'config', 'sensor_tf.yaml'),
             'robot_radius': 0.25}
        for k, v in p.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in p}
        self.load_monitor(self.p['params_file'])
        self.yaml_tf = self.load_rig(self.p['transforms_file'])
        self.add_on_set_parameters_callback(self.on_params)

        self.tf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf, self)
        self.pubs = {c: self.create_publisher(PointCloud2, f'/safety_cut_viz/{c}', 5)
                     for c in CLASSES}
        self.pub_mk = self.create_publisher(MarkerArray, '/safety_cut_viz/markers', 1)
        self.create_subscription(PointCloud2, self.p['cloud_topic'], self.on_cloud,
                                 qos_profile_sensor_data)
        self.get_logger().info(
            f"polygons {[q['name'] for q in self.polys]}, z {self.zmin}..{self.zmax}, "
            f"min_range {self.p['min_range']}, self_cut {self.p['self_cut_range']} "
            f"outside +-{self.p['self_cut_angle']} deg, "
            f"{len(self.boxes())} body boxes")

    def load_monitor(self, path):
        with open(os.path.expanduser(path), encoding='utf-8') as f:
            cm = yaml.safe_load(f)['collision_monitor']['ros__parameters']
        src = cm[cm['observation_sources'][0]]
        # Height band is a live parameter too; the yaml gives the default.
        for k in ('min_height', 'max_height'):
            if not self.has_parameter(k):
                self.declare_parameter(k, float(src[k]))
        self.zmin = self.get_parameter('min_height').value
        self.zmax = self.get_parameter('max_height').value
        self.polys = []
        for name in cm['polygons']:
            q = cm[name]
            if q.get('enabled', True):
                pts = q['points']
                self.polys.append({'name': name, 'action': q['action_type'],
                                   'max': q['max_points'],
                                   'poly': list(zip(pts[0::2], pts[1::2]))})

    def load_rig(self, path):
        """Return {child: 4x4} for the base_frame edges in sensor_tf.yaml."""
        with open(os.path.expanduser(path), encoding='utf-8') as f:
            edges = yaml.safe_load(f)['transforms']
        out = {}
        for e in edges:
            if e['parent'] == self.p['base_frame']:
                m = np.eye(4)
                m[:3, :3] = Rotation.from_quat(e['qxyzw']).as_matrix()
                m[:3, 3] = e['xyz']
                out[e['child']] = m
        return out

    def on_params(self, params):
        for prm in params:
            if prm.name in self.p:
                self.p[prm.name] = prm.value
            elif prm.name == 'min_height':
                self.zmin = prm.value
            elif prm.name == 'max_height':
                self.zmax = prm.value
        return SetParametersResult(successful=True)

    def boxes(self):
        b = list(self.p['body_boxes'])
        if len(b) % 6:
            self.get_logger().error('body_boxes needs 6 values per box; ignoring it',
                                    throttle_duration_sec=5.0)
            return []
        return [b[i:i + 6] for i in range(0, len(b), 6)]

    def sensor_to_base(self, frame):
        """(4x4, source) for base_frame <- frame: live TF, else sensor_tf.yaml."""
        try:
            t = self.tf.lookup_transform(self.p['base_frame'], frame, rclpy.time.Time())
        except tf2_ros.TransformException:
            if frame in self.yaml_tf:
                return self.yaml_tf[frame], 'yaml'
            return None, None
        q, tr = t.transform.rotation, t.transform.translation
        m = np.eye(4)
        m[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        m[:3, 3] = [tr.x, tr.y, tr.z]
        return m, 'tf'

    def on_cloud(self, msg):
        m, source = self.sensor_to_base(msg.header.frame_id)
        if m is None:
            self.get_logger().warn(
                f"no transform {self.p['base_frame']} <- {msg.header.frame_id}",
                throttle_duration_sec=5.0)
            return
        raw = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z']).astype(float)
        raw = raw[np.isfinite(raw).all(1)]
        pts = raw @ m[:3, :3].T + m[:3, 3]
        origin = m[:3, 3]
        p = self.p

        # Stage 1: rangeKeep, 3D range in the sensor frame.
        cut_range = np.linalg.norm(raw, axis=1) < p['min_range']
        # Stage 2: sideBodyKeep, range from the sensor, bearing in base_frame.
        d = pts - origin
        bearing = np.degrees(np.abs(np.arctan2(d[:, 1], d[:, 0])))
        cut_side = ((np.linalg.norm(d, axis=1) < p['self_cut_range'])
                    & (bearing > p['self_cut_angle']) & (p['self_cut_range'] > 0.0))
        cut_side &= ~cut_range
        # Stage 3: bodyBoxKeep, boxes in base_frame, bounds inclusive.
        in_box = np.zeros(len(pts), bool)
        for x0, x1, y0, y1, z0, z1 in self.boxes():
            in_box |= ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0)
                       & (pts[:, 1] <= y1) & (pts[:, 2] >= z0) & (pts[:, 2] <= z1))
        cut_box = in_box & ~cut_range & ~cut_side
        # Stage 4: the monitor's height band, in base_frame.
        reach = ~cut_range & ~cut_side & ~cut_box
        out_band = reach & ((pts[:, 2] < self.zmin) | (pts[:, 2] > self.zmax))
        inband = reach & ~out_band

        cls = {'min_range': cut_range, 'self_cut': cut_side, 'body_box': cut_box,
               'height': out_band}
        in_any = np.zeros(len(pts), bool)
        verdict, counts = 'clear', {}
        for action in ('stop', 'slowdown'):
            hit = np.zeros(len(pts), bool)
            for q in (q for q in self.polys if q['action'] == action):
                h = inband & inside(q['poly'], pts[:, 0], pts[:, 1])
                counts[q['name']] = int(h.sum())
                if h.sum() > q['max'] and verdict == 'clear':
                    verdict = 'STOP' if action == 'stop' else 'SLOWDOWN'
                hit |= h
            cls['stop' if action == 'stop' else 'slow'] = hit & ~in_any
            in_any |= hit
        cls['clear'] = inband & ~in_any

        header = msg.header
        header.frame_id = p['base_frame']
        for c, mask in cls.items():
            self.publish(self.pubs[c], header, pts[mask], CLASSES[c])
        self.markers(origin, verdict, counts, source,
                     {c: int(v.sum()) for c, v in cls.items()})

    def publish(self, pub, header, pts, rgb):
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate('xyz')]
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
        data = np.c_[pts.astype(np.float32), np.full(len(pts), rgb_float(rgb), np.float32)]
        pub.publish(pc2.create_cloud(header, fields, data))

    def marker(self, ns, mid, mtype, rgb, a=1.0):
        mk = Marker()
        mk.header.frame_id = self.p['base_frame']
        mk.ns, mk.id, mk.type, mk.action = ns, mid, mtype, Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.color = ColorRGBA(r=rgb[0] / 255, g=rgb[1] / 255, b=rgb[2] / 255, a=a)
        return mk

    def markers(self, origin, verdict, counts, source, n):
        p = self.p
        clear = Marker()
        clear.action = Marker.DELETEALL      # body boxes can be removed live
        ms = [clear]
        # Polygons, drawn on the floor and at both edges of the height band.
        for i, q in enumerate(self.polys):
            rgb = CLASSES['stop' if q['action'] == 'stop' else 'slow']
            for k, z in enumerate((0.01, self.zmin, self.zmax)):
                mk = self.marker('polygons', 3 * i + k, Marker.LINE_STRIP, rgb,
                                 1.0 if k < 2 else 0.4)
                mk.scale.x = 0.012 if k == 1 else 0.006
                mk.points = [Point(x=float(x), y=float(y), z=z)
                             for x, y in q['poly'] + q['poly'][:1]]
                ms.append(mk)
        # Pepper's footprint (robot_radius).
        mk = self.marker('robot', 0, Marker.LINE_STRIP, (255, 255, 255), 0.8)
        mk.scale.x = 0.008
        r = p['robot_radius']
        mk.points = [Point(x=r * np.cos(a), y=r * np.sin(a), z=0.01)
                     for a in np.linspace(0, 2 * np.pi, 73)]
        ms.append(mk)
        # min_range: a sphere around the lidar.
        mk = self.marker('min_range', 0, Marker.SPHERE, CLASSES['min_range'], 0.18)
        mk.pose.position = Point(x=origin[0], y=origin[1], z=origin[2])
        mk.scale.x = mk.scale.y = mk.scale.z = 2.0 * p['min_range']
        ms.append(mk)
        # self-cut: ring at lidar height outside the forward cone, plus the
        # two cone edges.
        mk = self.marker('self_cut', 0, Marker.LINE_STRIP, CLASSES['self_cut'])
        mk.scale.x = 0.01
        a0, rc = np.radians(p['self_cut_angle']), p['self_cut_range']
        arc = np.linspace(a0, 2 * np.pi - a0, 61)
        mk.points = ([Point(x=origin[0], y=origin[1], z=origin[2])]
                     + [Point(x=origin[0] + rc * np.cos(a), y=origin[1] + rc * np.sin(a),
                              z=origin[2]) for a in arc]
                     + [Point(x=origin[0], y=origin[1], z=origin[2])])
        ms.append(mk)
        # Body boxes, translucent.
        for i, (x0, x1, y0, y1, z0, z1) in enumerate(self.boxes()):
            mk = self.marker('body_box', i, Marker.CUBE, CLASSES['body_box'], 0.25)
            mk.pose.position = Point(x=(x0 + x1) / 2, y=(y0 + y1) / 2, z=(z0 + z1) / 2)
            mk.scale.x, mk.scale.y, mk.scale.z = x1 - x0, y1 - y0, z1 - z0
            ms.append(mk)
        # Verdict and per-class counts.
        mk = self.marker('verdict', 0, Marker.TEXT_VIEW_FACING,
                         {'STOP': CLASSES['stop'], 'SLOWDOWN': CLASSES['slow']}.get(
                             verdict, CLASSES['clear']))
        mk.pose.position = Point(x=0.0, y=0.0, z=1.25)
        mk.scale.z = 0.08
        poly_txt = '  '.join(f'{k} {v}' for k, v in counts.items())
        tf_txt = ('' if source == 'tf' else
                  '\nNO RIG TF (sensor_tf.yaml used): the real monitor ignores this cloud')
        mk.text = (f'{verdict}   ({poly_txt}; > max_points triggers)\n'
                   f"cut: min_range {n['min_range']}  self_cut {n['self_cut']}  "
                   f"body_box {n['body_box']}  height {n['height']}{tf_txt}")
        ms.append(mk)
        self.pub_mk.publish(MarkerArray(markers=ms))


def main():
    rclpy.init()
    node = SafetyCutViz()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
