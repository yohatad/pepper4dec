#!/usr/bin/env python3
"""Put base_footprint -> SonarFront_frame on /tf_nav, as a single edge.

Why: collision_monitor reads /tf_nav, and a "range" source has to transform
its reading into base_frame_id. The real chain is

    base_footprint -> base_link -> torso -> Hip -> Pelvis -> Tibia
                                                         -> SonarFront_frame

and tf_nav_relay drops it at base_link (drop_children), so on /tf_nav the
sonar frame does not exist and the source is silently useless. Widening
keep_parents would relay five more joint edges at naoqi_driver's 50 Hz into
the TF buffer the relay exists to keep quiet; three of those joints are
revolute, so a static_transform_publisher would be a guess at the posture
rather than a transform.

This composes the chain off the FULL /tf -- where every edge is present --
and republishes the one edge Nav2 needs. /tf_nav gains 1 transform at
publish_rate instead of 5 at 50 Hz, and the value is the real one, so it
stays correct if the hip or knee moves.

Not a multiple-parent violation: on /tf_nav nothing else parents
SonarFront_frame, and /tf keeps its own unbroken chain for RViz.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: October 19, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener


class SonarTfBridge(Node):
    def __init__(self):
        super().__init__('sonar_tf_bridge')
        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('source_frames', ['SonarFront_frame'])
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('out_topic', '/tf_nav')

        self.target = self.get_parameter('target_frame').value
        self.sources = list(self.get_parameter('source_frames').value)
        rate = self.get_parameter('publish_rate').value

        # spin_thread: the buffer needs its own executor thread or the
        # lookups below starve while this node's timer holds the callback.
        self.buf = Buffer()
        TransformListener(self.buf, self, spin_thread=True)

        # /tf is RELIABLE by convention; a BEST_EFFORT publisher here would be
        # dropped by Nav2's RELIABLE subscription without a word.
        self.pub = self.create_publisher(
            TFMessage, self.get_parameter('out_topic').value,
            QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))

        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'bridging {self.sources} -> {self.target} onto '
            f'{self.get_parameter("out_topic").value} at {rate:.0f} Hz')

    def tick(self):
        out = []
        for frame in self.sources:
            try:
                # Latest, not a stamp: this edge only has to be fresher than
                # collision_monitor's transform_tolerance, and the sonar
                # reading it serves is already the newest one.
                t = self.buf.lookup_transform(self.target, frame, rclpy.time.Time())
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f'no TF {self.target} <- {frame}: {exc}',
                    throttle_duration_sec=5.0)
                continue
            t.header.stamp = self.get_clock().now().to_msg()
            out.append(t)
        if out:
            self.pub.publish(TFMessage(transforms=out))


def main():
    rclpy.init()
    node = SonarTfBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
