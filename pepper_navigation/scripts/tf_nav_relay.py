#!/usr/bin/env python3
"""Republish only the navigation edges of /tf on /tf_nav, for the Nav2 nodes.

Why: naoqi_driver publishes Pepper's whole joint tree on /tf -- MEASURED 83
frames at 50 Hz, ~4150 transforms/s -- while Nav2 needs ~11/s (the localizer's
map edge). tf2 re-checks every pending transform request on EVERY transform it
receives, and each costmap sensor cloud waits briefly for its TF, so the joint
tree kept the costmap's TF thread saturated until the local costmap stopped
updating. MEASURED with this relay: 4 min live, no freeze, 0 dropped clouds.

The Nav2 nodes remap /tf -> /tf_nav in the launch files. /tf_static is NOT
relayed or remapped: static transforms arrive once and cost nothing. RViz and
everything else keep the full /tf, so the robot model still renders.

Only remap nodes that LISTEN to TF. A node that also BROADCASTS on /tf (amcl,
rtabmap, the localizers) must not be remapped, or its broadcasts would go to
/tf_nav only and vanish from the real tree.

Parameters:
    keep_parents (default: map odom lio_init base_footprint)
        A transform is relayed when its PARENT frame is one of these...
    drop_children (default: base_link)
        ...unless its child is one of these. base_footprint -> base_link is
        naoqi_driver's root edge; nothing in Nav2 uses it.
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class TfNavRelay(Node):
    def __init__(self):
        super().__init__('tf_nav_relay')
        self.declare_parameter('keep_parents', ['map', 'odom', 'lio_init', 'base_footprint'])
        self.declare_parameter('drop_children', ['base_link'])
        self.keep = set(self.get_parameter('keep_parents').value)
        self.drop = set(self.get_parameter('drop_children').value)
        # Same QoS as tf2_ros's /tf publisher and listener.
        qos = QoSProfile(depth=100, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(TFMessage, '/tf_nav', qos)
        self.create_subscription(TFMessage, '/tf', self.cb, qos)
        self.get_logger().info(
            f'relaying /tf -> /tf_nav for parents {sorted(self.keep)}, '
            f'dropping children {sorted(self.drop)}')

    def cb(self, msg):
        keep = [t for t in msg.transforms
                if t.header.frame_id in self.keep and t.child_frame_id not in self.drop]
        if keep:
            self.pub.publish(TFMessage(transforms=keep))


def main():
    rclpy.init()
    node = TfNavRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # On Ctrl+C rclpy can shut the context down mid-take and raise
        # "Unable to convert call argument"; only a real error if still up.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
