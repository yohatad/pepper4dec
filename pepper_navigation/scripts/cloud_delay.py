#!/usr/bin/env python3
"""Republish bag clouds a fixed delay later, so their TF is already there.

Why: in the bags a cloud arrives ~20 ms BEFORE the /tf_nav pose for its
stamp. The costmap's tf2 MessageFilter then calls Buffer::waitForTransform,
and in Humble's tf2_ros (0.25.23) that path can deadlock against the TF
listener thread (addTransformableRequest vs testTransformableRequests take
the same two mutexes in opposite order). After that the local costmap never
takes another cloud and freezes. Seen on replay only; the robot's own
recorded costmap kept updating.

Holding each cloud back until the pose is already buffered makes the filter
take the canTransform() fast path and never wait.

    ros2 bag play <bag> --clock --remap /points:=/bag/points \
        /camera/depth/color/points:=/bag/camera/depth/color/points
    python3 cloud_delay.py --ros-args -p use_sim_time:=true

With a live localizer (FAST-LIO) the bag must NOT be remapped: the localizer
needs the clouds immediately, or its pose is late again by construction. Then
delay only the costmap's copy:
    python3 cloud_delay.py --ros-args -p use_sim_time:=true \
        -p in_prefix:='' -p out_suffix:=_delayed
and point the costmap's observation sources at <topic>_delayed.
"""
import collections

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

TOPICS = ['/points', '/camera/depth/color/points']


class CloudDelay(Node):

    def __init__(self):
        super().__init__('cloud_delay')
        self.declare_parameter('delay', 0.3)   # s of bag time; /tf_nav lags up to 0.33 s
        self.declare_parameter('in_prefix', '/bag')   # where the raw clouds come from
        self.declare_parameter('out_suffix', '')      # suffix of the delayed copies
        pre = self.get_parameter('in_prefix').value
        suf = self.get_parameter('out_suffix').value
        self.delay = rclpy.duration.Duration(seconds=self.get_parameter('delay').value)
        self.queues = {}
        for topic in TOPICS:
            pub = self.create_publisher(PointCloud2, topic + suf, qos_profile_sensor_data)
            q = collections.deque()
            self.queues[topic] = (q, pub)
            self.create_subscription(PointCloud2, pre + topic,
                                     lambda m, q=q: q.append((self.get_clock().now(), m)),
                                     qos_profile_sensor_data)
        self.create_timer(0.01, self.flush)

    def flush(self):
        now = self.get_clock().now()
        for q, pub in self.queues.values():
            while q and (now - q[0][0]) >= self.delay:
                pub.publish(q.popleft()[1])
            # A bag restart moves the clock backwards: drop what is queued.
            if q and q[0][0] > now:
                q.clear()


def main():
    rclpy.init()
    rclpy.spin(CloudDelay())


if __name__ == '__main__':
    main()
