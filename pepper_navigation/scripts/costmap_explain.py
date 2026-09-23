#!/usr/bin/env python3
"""Colour every sensor point by what the local costmap does with it (for RViz).

Mirrors the marking rules of nav2_params_fastloc.yaml (local costmap):
  L2         points_near: 0.4-0.7 m from the lidar, marks above 0.30 m
             points_far : 0.7-4.0 m from the lidar, marks above 0.10 m
  RealSense  realsense  : 0.1-4.0 m, marks above 0.18 m (voxel grid top 1.30 m)
Heights are in the global frame (map, levelled), ranges are 3D from the sensor,
as in nav2's ObstacleLayer. Change the parameters if the yaml changes.

Publishes (frame: map):
  /costmap_explain/l2   PointCloud2 rgb
      red    = marked by the L2
      orange = IGNORED by the near rule (within 0.7 m: below 0.30 m, or closer
               than 0.40 m -- Pepper's own body) that 0.10 m would have marked
      grey   = floor / out of band, never marked
  /costmap_explain/rs   PointCloud2 rgb
      magenta = marked by the RealSense
      blue    = below the floor (reflection / depth error), never marked
      grey    = floor / out of band
  /costmap_explain/markers   rings on the floor + labels: green = the 0.7 m
      near/far split, orange = the 0.40 m inner zone where the L2 marks nothing
  /costmap_explain/cells PointCloud2 rgb -- THE LOCAL COSTMAP's own lethal
      cells, coloured by which sensor is marking them right now:
      red = L2, blue = RealSense, purple = both,
      stale (neither has marked it within mark_memory seconds -- a leftover
      that nothing has cleared yet), by the sensor that marked it LAST:
      pink = left by the L2, light blue = left by the RealSense,
      grey = origin unknown (older than origin_memory, or before startup).
      The costmap itself cannot show this: it merges the L2 layer and the
      RealSense voxel layer into one grid.

    python3 costmap_explain.py --ros-args -p use_sim_time:=true

All thresholds are live parameters -- change one and the next scan is
recoloured, without restarting anything:
    ros2 param set /costmap_explain l2_near_min_h 0.35
    ros2 param set /costmap_explain rs_min_h 0.25
"""
import collections

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import GetParameters
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation
import tf2_ros

LETHAL = 100

RGB = {'red': (230, 30, 30), 'orange': (255, 150, 0), 'grey': (150, 150, 150),
       'pink': (255, 170, 200), 'lightblue': (150, 205, 255),
       'magenta': (220, 40, 220), 'blue': (30, 110, 255), 'purple': (170, 60, 230)}


def pack(names):
    """Colour names -> packed float32 rgb, vectorised (clouds reach 75k points)."""
    names = np.asarray(names)
    uniq, inv = np.unique(names, return_inverse=True)
    table = np.array([RGB[n] for n in uniq], np.uint32)
    c = table[inv]
    return ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)


class CostmapExplain(Node):

    def __init__(self):
        super().__init__('costmap_explain')
        p = {'global_frame': 'map', 'base_frame': 'base_footprint',
             'l2_split': 0.7, 'l2_near_min_h': 0.30, 'l2_far_min_h': 0.10,
             'l2_max_h': 1.70, 'l2_max_range': 4.0, 'l2_min_range': 0.1,
             'l2_near_min_range': 0.40,     # points_near.obstacle_min_range
             'rs_min_h': 0.18, 'rs_max_h': 1.30, 'rs_max_range': 4.0, 'rs_min_range': 0.1,
             'rs_stride': 2,
             'l2_topic': '/points', 'rs_topic': '/camera/depth/color/points',
             'costmap_topic': '/local_costmap/costmap',
             # Which sensors the COSTMAP actually uses. 'auto' asks the costmap
             # node which layers are in plugins and enabled, so a sensor whose
             # layer is switched off stops colouring cells (its points are still
             # coloured, to show what it WOULD mark). Override with l2/rs/both.
             'sensors': 'auto', 'costmap_node': '/local_costmap/local_costmap',
             # The costmap is the union of MANY scans; comparing it against a
             # single sparse L2 scan paints most cells grey. Keep this many
             # seconds of marks instead (0 = the latest scan only).
             'mark_memory': 3.0,
             # How far back to remember which sensor last marked a cell, to
             # colour stale cells by the sensor that left them.
             'origin_memory': 120.0, 'cell_res': 0.05}
        for k, v in p.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in p}
        # Every threshold is live: `ros2 param set /costmap_explain
        # l2_near_min_h 0.35` recolours the next scan, so a value can be tried
        # against real data before it goes into the yaml (where the per-source
        # heights and ranges are read only at startup, unlike inflation).
        self.add_on_set_parameters_callback(self.on_params)
        self.tf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf, self)
        self.pub_l2 = self.create_publisher(PointCloud2, '/costmap_explain/l2', 5)
        self.pub_rs = self.create_publisher(PointCloud2, '/costmap_explain/rs', 5)
        self.pub_mk = self.create_publisher(MarkerArray, '/costmap_explain/markers', 1)
        self.pub_cells = self.create_publisher(PointCloud2, '/costmap_explain/cells', 5)
        self.marked = {'l2': collections.deque(), 'rs': collections.deque()}
        self.last_mark = {'l2': {}, 'rs': {}}   # global cell key -> last time marked
        self.grid_off = (0.0, 0.0)              # costmap grid offset, from its origin
        self.use = {'l2': True, 'rs': True}     # layers the costmap really has
        self.layer_cli = self.create_client(
            GetParameters, self.p['costmap_node'] + '/get_parameters')
        self.create_timer(5.0, self.check_layers)
        self.create_subscription(
            OccupancyGrid, self.p['costmap_topic'], self.cells,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(PointCloud2, self.p['l2_topic'], self.on_l2,
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, self.p['rs_topic'], self.on_rs,
                                 qos_profile_sensor_data)
        self.create_timer(1.0, self.markers)

    def check_layers(self):
        """Ask the costmap which layers are loaded and enabled ('auto' mode)."""
        want = self.p['sensors']
        if want != 'auto':
            self.use = {'l2': want in ('l2', 'both'), 'rs': want in ('rs', 'both')}
            return
        if not self.layer_cli.service_is_ready():
            return
        # 'plugins' only: asking for a parameter the costmap does not have (a
        # layer that is not loaded) makes it answer with an EMPTY list, not a
        # not-set value. The enabled flags are fetched in a second call, for
        # the layers that are actually there.
        req = GetParameters.Request()
        req.names = ['plugins']
        self.layer_cli.call_async(req).add_done_callback(self.on_layers)

    def on_layers(self, future):
        try:
            vals = future.result().values
            if not vals:
                return
            plugins = list(vals[0].string_array_value)
        except Exception as err:                      # never take the node down
            self.get_logger().warn(f'layer query failed: {err}', once=True)
            return
        # Ask each layer for its enabled flag and its sources, rather than
        # guessing from the layer NAME: a layer may be called anything
        # (l2_voxel, obstacle_layer, ...). Which sensor it carries is decided
        # by the topics of its sources, below.
        names = [n for lay in plugins if lay != 'inflation_layer'
                 for n in (f'{lay}.enabled', f'{lay}.observation_sources')]
        req = GetParameters.Request()
        req.names = names
        self.layer_cli.call_async(req).add_done_callback(
            lambda f, p=plugins, n=names: self.on_sources(f, p, n))

    def on_sources(self, future, plugins, names):
        """Collect every source's topic parameter, for the enabled layers."""
        try:
            vals = future.result().values
            got = dict(zip(names, vals))
        except Exception as err:
            self.get_logger().warn(f'layer query failed: {err}', once=True)
            return
        topics = []
        for lay in plugins:
            enabled = got.get(f'{lay}.enabled')
            sources = got.get(f'{lay}.observation_sources')
            if not enabled or not enabled.bool_value or sources is None:
                continue
            topics += [f'{lay}.{s}.topic' for s in sources.string_value.split()]
        if not topics:
            self.set_use({'l2': False, 'rs': False}, plugins)
            return
        req = GetParameters.Request()
        req.names = topics
        self.layer_cli.call_async(req).add_done_callback(
            lambda f, p=plugins: self.on_topics(f, p))

    def on_topics(self, future, plugins):
        try:
            names = [v.string_value for v in future.result().values]
        except Exception as err:
            self.get_logger().warn(f'topic query failed: {err}', once=True)
            return
        base = [n.rsplit('_delayed', 1)[0] for n in names]
        use = {'l2': self.p['l2_topic'].rsplit('_delayed', 1)[0] in base,
               'rs': self.p['rs_topic'].rsplit('_delayed', 1)[0] in base}
        self.set_use(use, plugins)

    def set_use(self, use, plugins):
        if use != self.use:
            self.get_logger().info(
                f"costmap layers: L2 {'on' if use['l2'] else 'OFF'}, "
                f"RealSense {'on' if use['rs'] else 'OFF'} (plugins: {plugins})")
            self.use = use

    def on_params(self, params):
        for prm in params:
            if prm.name in self.p:
                self.p[prm.name] = prm.value
        return SetParametersResult(successful=True)

    def to_global(self, msg, stride=1):
        try:
            t = self.tf.lookup_transform(self.p['global_frame'], msg.header.frame_id,
                                         msg.header.stamp)
        except tf2_ros.TransformException:
            return None, None
        a = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z'])[::stride].astype(float)
        a = a[np.isfinite(a).all(1)]
        q, tr = t.transform.rotation, t.transform.translation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        origin = np.array([tr.x, tr.y, tr.z])
        return a @ rot.T + origin, origin

    def publish(self, pub, header, pts, colours):
        header.frame_id = self.p['global_frame']
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate('xyz')]
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1))
        data = np.c_[pts.astype(np.float32), colours]
        pub.publish(pc2.create_cloud(header, fields, data))

    def on_l2(self, msg):
        pts, o = self.to_global(msg)
        if pts is None:
            return
        p = self.p
        h = pts[:, 2]
        d = np.linalg.norm(pts - o, axis=1)
        inband = (d >= p['l2_min_range']) & (d < p['l2_max_range']) & (h <= p['l2_max_h'])
        near = d < p['l2_split']
        # points_near also starts at its own min range (Pepper's body is closer).
        inband &= ~near | (d >= p['l2_near_min_range'])
        marked = inband & np.where(near, h > p['l2_near_min_h'], h > p['l2_far_min_h'])
        near_only = (near & ~marked & (d >= p['l2_min_range'])
                     & (h > p['l2_far_min_h']) & (h <= p['l2_max_h']))
        names = np.where(marked, 'red', np.where(near_only, 'orange', 'grey'))
        self.remember('l2', pts[marked])
        self.publish(self.pub_l2, msg.header, pts, pack(names))

    def on_rs(self, msg):
        pts, o = self.to_global(msg, self.p['rs_stride'])
        if pts is None:
            return
        p = self.p
        h = pts[:, 2]
        d = np.linalg.norm(pts - o, axis=1)
        inrange = (d >= p['rs_min_range']) & (d < p['rs_max_range'])
        marked = inrange & (h > p['rs_min_h']) & (h <= p['rs_max_h'])
        below = h < -0.08
        names = np.where(marked, 'magenta', np.where(below, 'blue', 'grey'))
        self.remember('rs', pts[marked])
        self.publish(self.pub_rs, msg.header, pts, pack(names))

    def remember(self, src, pts):
        """Keep the last mark_memory seconds of marks, like the costmap does."""
        now = self.get_clock().now().nanoseconds * 1e-9
        q = self.marked[src]
        q.append((now, pts))
        while len(q) > 1 and now - q[0][0] > self.p['mark_memory']:
            q.popleft()
        if len(pts):
            keys = np.unique(self.cell_keys(pts[:, 0], pts[:, 1]))
            self.last_mark[src].update(dict.fromkeys(keys.tolist(), now))

    def cell_keys(self, x, y):
        """One integer per costmap cell, stable while the window rolls.

        The rolling window moves by whole cells, so its grid keeps a constant
        sub-cell offset from multiples of the resolution; subtracting that
        offset makes a point and the cell it falls in share one key.
        """
        r = self.p['cell_res']
        ox, oy = self.grid_off
        return ((np.floor((x - ox) / r).astype(np.int64) << 32)
                + np.floor((y - oy) / r).astype(np.int64))

    def cells(self, msg):
        """Colour the costmap's lethal cells by which sensor is marking them."""
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        grid = np.array(msg.data, np.int8).reshape(msg.info.height, msg.info.width)
        self.p['cell_res'] = res
        self.grid_off = (ox - res * round(ox / res), oy - res * round(oy / res))
        lethal = grid == LETHAL
        if not lethal.any():
            return
        by = np.zeros(grid.shape, np.uint8)             # bit 0 = L2, bit 1 = RealSense
        top = np.full(grid.shape, -np.inf)             # highest mark in the cell
        for bit, src in enumerate(('l2', 'rs')):
            q = self.marked[src]
            if not q or not self.use[src]:      # layer not in the costmap: don't credit it
                continue
            pts = np.vstack([p for _, p in q])
            col = np.floor((pts[:, 0] - ox) / res).astype(np.int64)
            row = np.floor((pts[:, 1] - oy) / res).astype(np.int64)
            ok = (col >= 0) & (col < grid.shape[1]) & (row >= 0) & (row < grid.shape[0])
            col, row, z = col[ok], row[ok], pts[ok, 2]
            hit = lethal[row, col]
            col, row, z = col[hit], row[hit], z[hit]
            by[row, col] |= np.uint8(1 << bit)
            np.maximum.at(top, (row, col), z)
        rows, cols = np.nonzero(lethal)
        b = by[rows, cols]
        z = top[rows, cols]
        pts = np.c_[ox + (cols + 0.5) * res, oy + (rows + 0.5) * res,
                    np.where(np.isfinite(z), z, 0.05)]
        names = np.array(['grey', 'red', 'blue', 'purple'])[b]
        # Stale cells: colour by whichever sensor marked them most recently.
        stale = np.nonzero(b == 0)[0]
        if len(stale):
            now = self.get_clock().now().nanoseconds * 1e-9
            keys = self.cell_keys(pts[stale, 0], pts[stale, 1]).tolist()
            neg = -np.inf
            t_l2 = np.array([self.last_mark['l2'].get(k, neg) if self.use['l2'] else neg
                             for k in keys])
            t_rs = np.array([self.last_mark['rs'].get(k, neg) if self.use['rs'] else neg
                             for k in keys])
            names = names.astype('<U9')
            names[stale] = np.where(np.isinf(np.maximum(t_l2, t_rs)), 'grey',
                                    np.where(t_rs >= t_l2, 'lightblue', 'pink'))
            for src in self.last_mark.values():        # forget very old marks
                if len(src) > 200000:
                    for k in [k for k, t in src.items()
                              if now - t > self.p['origin_memory']]:
                        del src[k]
        header = Header()
        header.stamp = msg.header.stamp
        self.publish(self.pub_cells, header, pts, pack(names))

    def markers(self):
        # The split is a 3D range from the lidar; on the floor it is a circle
        # of radius sqrt(split^2 - lidar_height^2) around the lidar.
        try:
            t = self.tf.lookup_transform(self.p['base_frame'], 'l2lidar_frame',
                                         rclpy.time.Time())
        except tf2_ros.TransformException:
            return
        lx, ly, lz = (t.transform.translation.x, t.transform.translation.y,
                      t.transform.translation.z)
        # Both rings are drawn where the 3D range reaches the FLOOR:
        # radius sqrt(range^2 - lidar_height^2) around the lidar.
        markers = []
        for i, (rng, rgba, text) in enumerate([
                (self.p['l2_split'], (0.1, 0.8, 0.2),
                 f"0.7 m: inside marks > {self.p['l2_near_min_h']:.2f} m, "
                 f"outside > {self.p['l2_far_min_h']:.2f} m"),
                (self.p['l2_near_min_range'], (1.0, 0.55, 0.0),
                 f"{self.p['l2_near_min_range']:.2f} m: L2 marks nothing inside "
                 "(Pepper's own body)")]):
            r = np.sqrt(max(rng ** 2 - lz ** 2, 0.0))
            ring = Marker()
            ring.header.frame_id = self.p['base_frame']
            ring.ns, ring.id, ring.type, ring.action = 'l2_split', 2 * i, Marker.LINE_STRIP, 0
            ring.scale.x = 0.015
            ring.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=1.0)
            ring.pose.orientation.w = 1.0
            ring.points = [Point(x=lx + r * np.cos(a), y=ly + r * np.sin(a), z=0.01)
                           for a in np.linspace(0, 2 * np.pi, 73)]
            label = Marker()
            label.header.frame_id = self.p['base_frame']
            label.ns, label.id, label.type = 'l2_split', 2 * i + 1, Marker.TEXT_VIEW_FACING
            label.action = 0
            label.pose.position.x = lx + r
            label.pose.position.y = -0.25 - 0.12 * i
            label.pose.position.z = 0.35 + 0.1 * i
            label.pose.orientation.w = 1.0
            label.scale.z = 0.07
            label.color = ring.color
            label.text = text
            markers += [ring, label]
        self.pub_mk.publish(MarkerArray(markers=markers))


def main():
    rclpy.init()
    rclpy.spin(CostmapExplain())


if __name__ == '__main__':
    main()
