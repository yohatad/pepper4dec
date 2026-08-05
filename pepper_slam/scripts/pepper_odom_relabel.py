#!/usr/bin/env python3
"""
Republishes /pepper_odom with header.frame_id overridden to odom, for
EKF fusion alongside leveled_odometry_publisher.py's output.

/pepper_odom lives on a TF tree (pepper_odom -> base_footprint) that is
deliberately disconnected from odom -> odom_lidar -> base_footprint (see
lio_map_odom_bridge.py's docstring and pepper_slam's README) -- two live
parents for base_footprint would be a broken tree. Since there's no TF path
between the two, robot_localization can't transform pepper_odom's data into
odom on its own. Relabeling sidesteps that: z/roll/pitch (the only
fields this source's EKF config mask actually fuses) don't depend on which
frame's horizontal origin/yaw you're nominally expressed in, only on shared
gravity alignment -- which both frames have by physical construction. x/y/yaw
are NOT relabeled-and-trusted this way; the EKF config mask excludes them
from this source entirely (wheel odom drifts there over distance).
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class PepperOdomRelabel(Node):

    def __init__(self):
        super().__init__("pepper_odom_relabel")
        self.declare_parameter("input_topic", "/pepper_odom")
        self.declare_parameter("output_topic", "/odometry/pepper_odom_relabeled")
        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.pub = self.create_publisher(Odometry, output_topic, 10)
        self.create_subscription(Odometry, input_topic, self._cb, 50)
        self.get_logger().info(f"Relabeling {input_topic} -> {output_topic} (frame_id=odom)")

    def _cb(self, msg):
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "odom"
        out.child_frame_id = "base_footprint"
        out.pose = msg.pose
        out.twist = msg.twist
        self.pub.publish(out)


def main():
    rclpy.init()
    node = PepperOdomRelabel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
