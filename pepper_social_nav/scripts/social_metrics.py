#!/usr/bin/env python3
"""Measure socially-aware navigation, so "is it better?" has an answer.

WHY THIS EXISTS. Social navigation is easy to tune by vibes and hard to defend
without numbers. This is a CSSR4Africa research output, so "it looked better in
RViz" will not survive review. Run this alongside the DWB baseline and again
alongside the social stack, on the same route, and compare.

WHAT IT MEASURES, per run:
  min_distance          closest approach to any bystander, metres. The headline
                        safety number.
  intrusions            seconds spent inside the intimate (<0.45 m) and
                        personal (<1.2 m) zones. Hall's proxemic bands; the
                        social stack should cut these without inflating
                        duration much.
  path_length           metres travelled. Social navigation costs path length --
                        this is what quantifies how much.
  duration              seconds from first motion to last.
  interventions         collision-monitor activations. Should go DOWN: if the
                        social layer works, the safety layer stops firing,
                        because the robot yields before it gets that close.
  stopped               how long the robot sat at zero velocity. A robot that
                        achieves great proxemics by never moving is not better,
                        and this is the number that catches it.

Usage:
    ros2 run pepper_social_nav social_metrics.py --ros-args -p label:=baseline
    # drive the route, then Ctrl-C for the summary

Results also go to ~/.ros/social_metrics_<label>.json for later comparison.
"""
import json
import math
import os

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from social_nav_msgs.msg import Pedestrians
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException
from tf2_ros import LookupException, TransformListener

INTIMATE = 0.45
PERSONAL = 1.2


class SocialMetrics(Node):
    """Accumulate proxemic and efficiency metrics over one navigation run."""

    def __init__(self):
        super().__init__('social_metrics')
        self.declare_parameter('label', 'run')
        self.declare_parameter('pedestrians_topic', '/people_tracker/bystanders')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('rate', 10.0)

        self.label = self.get_parameter('label').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.dt = 1.0 / float(self.get_parameter('rate').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            Pedestrians, self.get_parameter('pedestrians_topic').value,
            self._peds_cb, qos)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)

        # CollisionMonitorState lives in nav2_msgs; import it lazily so this
        # script still runs (minus that one metric) where nav2 is not sourced.
        try:
            from nav2_msgs.msg import CollisionMonitorState
            self.create_subscription(
                CollisionMonitorState, '/collision_monitor_state',
                self._monitor_cb, 10)
        except ImportError:
            self.get_logger().warn(
                'nav2_msgs unavailable: intervention count disabled.')

        self.peds = []
        self.min_distance = math.inf
        self.intimate_s = 0.0
        self.personal_s = 0.0
        self.path_length = 0.0
        self.moving_s = 0.0
        self.stopped_s = 0.0
        self.interventions = 0
        self._last_monitor = 0
        self._last_xy = None
        self._started = False
        self._elapsed = 0.0

        self.create_timer(self.dt, self._tick)
        self.get_logger().info(f"social_metrics recording as '{self.label}'.")

    def _peds_cb(self, msg):
        self.peds = [(p.pose.x, p.pose.y) for p in msg.pedestrians]

    def _cmd_cb(self, msg):
        speed = math.hypot(msg.linear.x, msg.linear.y)
        if speed > 0.01 or abs(msg.angular.z) > 0.02:
            self._started = True

    def _monitor_cb(self, msg):
        # Count TRANSITIONS into a non-zero action, not every message, or one
        # sustained stop would score as hundreds of interventions.
        action = int(getattr(msg, 'action_type', 0))
        if action != 0 and self._last_monitor == 0:
            self.interventions += 1
        self._last_monitor = action

    def _tick(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame, self.robot_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return

        x = tf.transform.translation.x
        y = tf.transform.translation.y

        if self._last_xy is not None and self._started:
            step = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            self.path_length += step
            self._elapsed += self.dt
            if step / self.dt > 0.02:
                self.moving_s += self.dt
            else:
                self.stopped_s += self.dt
        self._last_xy = (x, y)

        if not self._started or not self.peds:
            return

        d = min(math.hypot(px - x, py - y) for px, py in self.peds)
        self.min_distance = min(self.min_distance, d)
        if d < INTIMATE:
            self.intimate_s += self.dt
        if d < PERSONAL:
            self.personal_s += self.dt

    def summary(self):
        """Return this run's metrics as a plain dict."""
        return {
            'label': self.label,
            'min_distance_m': (None if math.isinf(self.min_distance)
                               else round(self.min_distance, 3)),
            'intimate_zone_s': round(self.intimate_s, 2),
            'personal_zone_s': round(self.personal_s, 2),
            'path_length_m': round(self.path_length, 2),
            'duration_s': round(self._elapsed, 2),
            'moving_s': round(self.moving_s, 2),
            'stopped_s': round(self.stopped_s, 2),
            'collision_monitor_interventions': self.interventions,
        }

    def report(self):
        """Log the summary and write it beside the other runs' files."""
        s = self.summary()
        path = os.path.join(
            os.path.expanduser('~/.ros'), f'social_metrics_{self.label}.json')
        try:
            with open(path, 'w') as f:
                json.dump(s, f, indent=2)
        except OSError as ex:
            self.get_logger().error(f'could not write {path}: {ex}')
            path = '(not written)'

        width = max(len(k) for k in s)
        lines = '\n'.join(f'  {k:<{width}}  {v}' for k, v in s.items())
        self.get_logger().info(f'\n--- social metrics ---\n{lines}\n  -> {path}')


def main(args=None):
    rclpy.init(args=args)
    node = SocialMetrics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
