#!/usr/bin/env python3
""" pepper_odom_anchor.py

Entry point and lifecycle node implementation for absolute robot localization.
Converts relative odometry readings into an absolute robot pose using a fixed
initial position as the anchor.

On configure, the node reads the initial pose and topic/rate parameters and
creates a managed pose publisher. On activate, it subscribes to odometry and
starts a timer that periodically publishes the absolute pose. The first
odometry reading received after activation is captured as the odom-frame
anchor; subsequent readings are transformed (translated and rotated) into the
global frame relative to that anchor and the configured initial pose.

Subscribers:
    <odom_topic> (nav_msgs/Odometry)
        Relative odometry readings used to compute displacement and rotation
        since the odom-frame anchor.

Publishers:
    <pose_topic> (geometry_msgs/Pose2D)
        Absolute robot pose (x, y, theta) in the global frame, published at
        a fixed rate.

Parameters:
    initial_x (double, default: 0.0)
        Initial global x position of the robot.
    initial_y (double, default: 0.0)
        Initial global y position of the robot.
    initial_theta (double, default: 0.0)
        Initial global heading of the robot, in radians.
    odom_topic (string, default: "/pepper_odom_filtered")
        Topic to subscribe to for odometry messages.
    pose_topic (string, default: "/robot_localization/pose")
        Topic on which the absolute pose is published.
    publish_rate (double, default: 10.0)
        Rate, in Hz, at which the absolute pose is published.

Lifecycle:
    configure  -> read parameters, initialize the global pose/anchor state, and
                  create the managed pose publisher.
    activate   -> activate the pose publisher, subscribe to odometry, and start
                  the periodic pose-publishing timer.
    deactivate -> cancel and destroy the timer, destroy the odometry
                  subscription, and deactivate the pose publisher.
    cleanup    -> destroy the pose publisher and reset the odometry anchor
                  state.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: June 11, 2026
Version: v1.0
"""

import math
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D


class RobotLocalization(LifecycleNode):
    """Lifecycle node that fuses odometry into an absolute robot pose estimate."""

    def __init__(self):
        super().__init__('robot_localization')

        # Declare parameters here so they are settable via launch/CLI before configure
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_theta', 0.0)
        self.declare_parameter('odom_topic', '/pepper_odom_filtered')
        self.declare_parameter('pose_topic', '/robot_localization/pose')
        self.declare_parameter('publish_rate', 10.0)  # Hz

    # ── Lifecycle callbacks ─────────────────────────────────────────────────────

    def on_configure(self, state) -> TransitionCallbackReturn:
        """Read parameters and create the pose publisher."""
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
        """Log shutdown of the node."""
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
