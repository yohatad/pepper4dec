#!/usr/bin/env python3
"""Publishes a whole static TF chain from ONE node.

Replaces N separate tf2_ros/static_transform_publisher processes with a single
node that emits every transform in one latched /tf_static message.

WHY: pepper_sensor_tf used to spawn 10 static_transform_publisher nodes -- one
per edge. Each is a full ROS node: its own executor, DDS participant, discovery
traffic and /tf_static publisher, to publish seven numbers that never change.
tf2 concatenates all /tf_static publishers anyway, so a single message carrying
the whole list is equivalent and costs one process instead of ten.

The transforms come from a YAML list so the rig geometry stays data, not code:

    transforms:
      - {parent: base_footprint, child: l2lidar_frame,
         xyz: [0.133, 0.0, 0.2582], qxyzw: [0.69313, -0.147438, 0.690138, -0.14677]}

Quaternions are normalised on load and a non-unit input is reported rather than
silently skewing the rig.
"""
import sys

import rclpy
import yaml
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster


class StaticTfPublisher(Node):

    def __init__(self):
        super().__init__('static_tf_publisher')
        self.declare_parameter('transforms_file', '')
        # Entries tagged 'owner: driver' are the RealSense's OWN internal
        # extrinsics (camera_link -> colour/depth/gyro and their optical
        # frames). realsense2_camera publishes those itself, read off the
        # device. Publishing them here too gives those edges two publishers:
        # /tf_static is latched and tf2 keys its static buffer on the child
        # frame, so whichever message lands last silently wins, and which one
        # that is varies between launches.
        #   mount (default) -- only the rig transforms nothing else provides.
        #                      Correct whenever the camera driver is running.
        #   all             -- everything, for replaying the older bags whose
        #                      /tf_static is empty (slam_recording*,
        #                      slam_bench_run*) and which therefore carry no
        #                      camera extrinsics of their own.
        self.declare_parameter('scope', 'mount')
        path = self.get_parameter('transforms_file').value
        scope = str(self.get_parameter('scope').value).lower()
        if scope not in ('mount', 'all'):
            self.get_logger().fatal(
                f"scope must be 'mount' or 'all', got '{scope}'.")
            raise SystemExit(1)
        if not path:
            self.get_logger().fatal("transforms_file parameter is empty.")
            raise SystemExit(1)

        try:
            with open(path) as fh:
                doc = yaml.safe_load(fh) or {}
        except OSError as ex:
            self.get_logger().fatal(f"Cannot read transforms_file '{path}': {ex}")
            raise SystemExit(1)

        entries = doc.get('transforms')
        if not entries:
            self.get_logger().fatal(
                f"'{path}' has no non-empty 'transforms:' list.")
            raise SystemExit(1)

        if scope == 'mount':
            kept = [e for e in entries if e.get('owner') != 'driver']
            skipped = len(entries) - len(kept)
            if skipped:
                self.get_logger().info(
                    f"scope=mount: publishing {len(kept)} rig transforms, "
                    f"leaving {skipped} camera-internal ones to the RealSense "
                    f"driver. Pass scope:=all when replaying a bag with no "
                    f"/tf_static.")
            entries = kept

        self.bc = StaticTransformBroadcaster(self)
        msgs, seen = [], {}
        now = self.get_clock().now().to_msg()

        for i, e in enumerate(entries):
            try:
                parent, child = e['parent'], e['child']
                x, y, z = (float(v) for v in e['xyz'])
                qx, qy, qz, qw = (float(v) for v in e['qxyzw'])
            except (KeyError, TypeError, ValueError) as ex:
                self.get_logger().fatal(
                    f"transform #{i} in '{path}' is malformed ({ex}); "
                    f"each entry needs parent, child, xyz[3], qxyzw[4].")
                raise SystemExit(1)

            # A frame with two parents silently breaks the tree, and the error
            # surfaces far away as "not part of the same tree". Catch it here.
            if child in seen:
                self.get_logger().fatal(
                    f"'{child}' is given two parents ('{seen[child]}' and "
                    f"'{parent}') in '{path}'. Each frame may have exactly one.")
                raise SystemExit(1)
            seen[child] = parent

            n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
            if n < 1e-9:
                self.get_logger().fatal(
                    f"{parent} -> {child}: zero-length quaternion.")
                raise SystemExit(1)
            if abs(n - 1.0) > 1e-3:
                self.get_logger().warn(
                    f"{parent} -> {child}: quaternion norm {n:.6f} != 1; "
                    f"normalising. Check the calibration source.")
            qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n

            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            msgs.append(t)

        # One latched message with the whole chain.
        self.bc.sendTransform(msgs)
        self.get_logger().info(
            f"Published {len(msgs)} static transforms from '{path}' in a single "
            f"/tf_static message: " +
            ", ".join(f"{m.header.frame_id}->{m.child_frame_id}" for m in msgs))


def main(args=None):
    rclpy.init(args=args)
    try:
        node = StaticTfPublisher()
    except SystemExit as ex:
        rclpy.shutdown()
        sys.exit(ex.code)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
