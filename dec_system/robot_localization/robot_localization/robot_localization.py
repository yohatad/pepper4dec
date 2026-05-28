#!/usr/bin/env python3
"""
robot_localization.py

Simple robot localization node that converts relative odometry to absolute position.
Uses initial position and integrates odometry data to provide absolute pose.

Author: Assistant
Date: 2025
"""

import math
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D


class RobotLocalization(LifecycleNode):
    """
    Converts relative odometry to absolute robot pose.
    Integrates odometry data with an initial position to provide global localization.

    Lifecycle:
      configure  → read parameters, create publisher
      activate   → create subscription + timer
      deactivate → cancel timer, destroy subscription
      cleanup    → destroy publisher
    """

    def __init__(self):
        super().__init__('robot_localization')

        # Declare parameters here so they are settable via launch/CLI before configure
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_theta', 0.0)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('pose_topic', '/robotLocalization/pose')
        self.declare_parameter('publish_rate', 10.0)  # Hz

    # ── Lifecycle callbacks ─────────────────────────────────────────────────────

    def on_configure(self, state) -> TransitionCallbackReturn:
        """Read parameters, initialise state, create publisher."""
        initial_x     = self.get_parameter('initial_x').value
        initial_y     = self.get_parameter('initial_y').value
        initial_theta = self.get_parameter('initial_theta').value
        self._odom_topic    = self.get_parameter('odom_topic').value
        self._pose_topic    = self.get_parameter('pose_topic').value
        self._publish_rate  = self.get_parameter('publish_rate').value

        # Absolute pose (updated by odom_callback)
        self.absolute_x     = initial_x
        self.absolute_y     = initial_y
        self.absolute_theta = initial_theta

        # Anchor for the odom-frame → global-frame transform
        self._initial_x     = initial_x
        self._initial_y     = initial_y
        self._initial_theta = initial_theta

        # First odometry reading — captured once on startup
        self.first_odom_x     = None
        self.first_odom_y     = None
        self.first_odom_theta = None
        self.odom_initialized = False

        # Managed publisher — deactivated silently while node is INACTIVE
        self.pose_pub = self.create_lifecycle_publisher(Pose2D, self._pose_topic, 10)

        self.get_logger().info(
            f'RobotLocalization configured — '
            f'initial=({initial_x:.3f}, {initial_y:.3f}, '
            f'{math.degrees(initial_theta):.1f}°) | '
            f'odom={self._odom_topic} | pose={self._pose_topic} @ {self._publish_rate} Hz'
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        """Activate managed publishers, then create subscription and timer."""
        super().on_activate(state)

        self.odom_sub = self.create_subscription(
            Odometry, self._odom_topic, self.odom_callback, 10
        )
        self.pub_timer = self.create_timer(
            1.0 / self._publish_rate, self.publish_pose
        )

        self.get_logger().info('RobotLocalization activated')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        """Cancel timer, destroy subscription, then deactivate managed publishers."""
        self.pub_timer.cancel()
        self.destroy_timer(self.pub_timer)
        self.destroy_subscription(self.odom_sub)

        super().on_deactivate(state)
        self.get_logger().info('RobotLocalization deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        """Release publisher and reset odometry anchor."""
        self.destroy_lifecycle_publisher(self.pose_pub)
        self.odom_initialized = False
        self.get_logger().info('RobotLocalization cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        self.get_logger().info('RobotLocalization shutting down')
        return TransitionCallbackReturn.SUCCESS

    # ── Callbacks ───────────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        """Process odometry messages and update absolute position."""
        try:
            odom_x = msg.pose.pose.position.x
            odom_y = msg.pose.pose.position.y

            # Extract yaw from quaternion (z-axis rotation only for 2D)
            q = msg.pose.pose.orientation
            odom_theta = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )

            # Capture the first reading to anchor the odom→global transform
            if not self.odom_initialized:
                self.first_odom_x     = odom_x
                self.first_odom_y     = odom_y
                self.first_odom_theta = odom_theta
                self.odom_initialized = True
                self.get_logger().info(
                    f'Odometry anchor set at '
                    f'({odom_x:.3f}, {odom_y:.3f}, {math.degrees(odom_theta):.1f}°)'
                )
                return

            # Displacement in the odom frame since the first reading
            d_odom_x = odom_x - self.first_odom_x
            d_odom_y = odom_y - self.first_odom_y

            # Fixed rotation: odom frame → global frame
            rot = self._initial_theta - self.first_odom_theta

            self.absolute_x = (
                self._initial_x
                + d_odom_x * math.cos(rot)
                - d_odom_y * math.sin(rot)
            )
            self.absolute_y = (
                self._initial_y
                + d_odom_x * math.sin(rot)
                + d_odom_y * math.cos(rot)
            )
            self.absolute_theta = self.normalize_angle(
                self._initial_theta + (odom_theta - self.first_odom_theta)
            )

        except Exception as e:
            self.get_logger().error(f'Error processing odometry: {e}')

    def publish_pose(self):
        """Publish the current absolute pose at fixed rate."""
        msg = Pose2D()
        msg.x     = self.absolute_x
        msg.y     = self.absolute_y
        msg.theta = self.absolute_theta
        self.pose_pub.publish(msg)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = RobotLocalization()
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
