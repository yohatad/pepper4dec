#!/usr/bin/env python3
"""Start nav2's lifecycle manager once the map frame actually exists.

WHY. nav2 configures its costmaps at startup, and local_costmap blocks waiting
for a transform to its global_frame. With fastlio_localization that frame does
not exist yet: the node publishes map -> base_footprint only after ScanContext
locks, and locking now requires the robot to MOVE (init_require_motion), so the
wait is unbounded. Left to autostart, the bringup stalls with

    Timed out waiting for transform from base_footprint to map to become
    available, tf error: Invalid frame ID "map" ... frame does not exist

and planner/controller/bt_navigator sit inactive forever.

A fixed TimerAction cannot fix this -- there is no delay that is correct, since
the lock waits on the robot, not on the clock. So the lifecycle manager is
launched with autostart:=false and this node calls its startup service the
moment the transform appears.
"""
import rclpy
from rclpy.node import Node
from nav2_msgs.srv import ManageLifecycleNodes
import tf2_ros


class Waiter(Node):
    def __init__(self):
        super().__init__('wait_for_map_then_start')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('source_frame', 'base_footprint')
        self.declare_parameter('manager', '/lifecycle_manager_navigation/manage_nodes')
        self.target = self.get_parameter('target_frame').value
        self.source = self.get_parameter('source_frame').value
        manager = self.get_parameter('manager').value

        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.cli = self.create_client(ManageLifecycleNodes, manager)
        self.done = False
        self.ticks = 0
        self.timer = self.create_timer(1.0, self.tick)
        self.get_logger().info(
            f'waiting for {self.target} -> {self.source} before starting nav2; '
            f'with fastlio_localization this appears once ScanContext locks, '
            f'which requires the robot to move')

    def tick(self):
        if self.done:
            return
        self.ticks += 1
        try:
            self.buf.lookup_transform(self.target, self.source, rclpy.time.Time())
        except Exception:
            if self.ticks % 15 == 0:
                self.get_logger().info(
                    f'still waiting for {self.target} ({self.ticks}s) -- '
                    f'move the robot if it has not localized yet')
            return
        self.get_logger().info(f'{self.target} is up after {self.ticks}s; starting nav2')
        if not self.cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('lifecycle manager service never appeared')
            return
        req = ManageLifecycleNodes.Request()
        req.command = ManageLifecycleNodes.Request.STARTUP
        fut = self.cli.call_async(req)
        fut.add_done_callback(self.started)
        self.done = True

    def started(self, fut):
        try:
            ok = fut.result().success
        except Exception as e:
            self.get_logger().error(f'nav2 startup call failed: {e}')
            return
        self.get_logger().info('nav2 startup: ' + ('OK' if ok else 'REPORTED FAILURE'))


def main():
    rclpy.init()
    n = Waiter()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()


if __name__ == '__main__':
    main()
