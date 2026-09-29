#!/usr/bin/env python3
"""Publish odom -> base_footprint from Pepper's wheel odometry, for the loc stacks.

The localization stacks (fastloc, pointloc) can broadcast REP-105
map -> odom instead of map -> base_footprint (publish.odom_frame, see
map_odom.hpp in fast_lio / point_lio). That needs a continuous
odom -> base_footprint, and in those stacks nothing else provides one: there is
no lio_odom_bridge, and the localizer's own filter IS the map pose.

naoqi_driver2 already publishes /pepper_odom, but its own TF broadcast is off on
purpose (publish_wheel_odom_tf) and names the parent pepper_odom. This node
relays the topic as TF under the frame names given, stamped with the message's
own stamp so the localizer's lookup at the scan stamp interpolates correctly.

Do NOT run this in the mapping or AMCL stacks: lio_odom_bridge already owns
odom -> lio_init -> base_footprint there, and a second parent splits the tree.
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class WheelOdomTf(Node):
    def __init__(self):
        super().__init__('wheel_odom_tf')
        self.declare_parameter('wheel_odom_topic', '/pepper_odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        topic = self.get_parameter('wheel_odom_topic').value

        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(Odometry, topic, self.on_odom, 50)
        self.get_logger().info(
            f'{topic} -> TF {self.odom_frame} -> {self.base_frame}')

    def on_odom(self, msg: Odometry):
        # The message's frame names (pepper_odom -> base_footprint) are
        # replaced, not trusted: the pose is what matters, and the odom frame
        # has to be the one the localizer composes against.
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        p = msg.pose.pose.position
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)


def main():
    rclpy.init()
    node = WheelOdomTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
