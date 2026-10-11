#!/usr/bin/env python3
"""A simulated Pepper and charging station, for test_docking_sim_launch.py.

Stands in for everything outside pepper_docking.launch.py:

  * the robot: integrates cmd_vel_raw into its map pose and publishes it as
    TF map -> base_footprint (on /tf and /tf_nav) and Odometry
    (/localization/pose), as the localization profiles do;
  * the sensors: ray-cast L2 clouds on /points and bottom-camera images on
    /naoqi_driver/camera/bottom/*, from dock_sim, with the real mounts;
  * the station: where `station` says, which need not be where the dock
    database says. The robot cannot back through it, and naoqi_driver's
    battery diagnostics report "Charging" (plugged in) once the base reaches
    the contacts, i.e. the true docked pose;
  * /sim/robot_pose (PoseStamped) for the test to check where the robot ended.

Parameters:
    station [x, y, yaw]: the true station centre, yaw into the station.
    station_offset: docked base_footprint to station centre, along its yaw.
    start [x, y, yaw]: initial robot pose.
"""

import math
import os
import sys

from builtin_interfaces.msg import Time as TimeMsg
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import tf2_ros
from tf2_msgs.msg import TFMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from dock_perception import from_frame, Pose2D, to_frame  # noqa: E402
from dock_sim import (  # noqa: E402
    BOTTOM_CAMERA_AXIS, BOTTOM_CAMERA_D, BOTTOM_CAMERA_IN_BASE, BOTTOM_CAMERA_K, bottom_camera,
    LIDAR_IN_BASE, lidar_scan, render, Scene)

CONTACT_TOLERANCE = 0.02        # m: plugged in within this of the docked pose
CONTACT_YAW_TOLERANCE = 0.10    # rad


def quaternion_from_matrix(r):
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2.0,
                      r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2.0,
                      r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2.0,
                      r[1, 0] - r[0, 1])
    return x, y, z, w


def transform(parent, child, xyz, quaternion, stamp):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id, t.header.stamp = parent, child, stamp
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
    (t.transform.rotation.x, t.transform.rotation.y,
     t.transform.rotation.z, t.transform.rotation.w) = quaternion
    return t


class DockingWorld(Node):

    def __init__(self):
        super().__init__('docking_world')
        station = self.declare_parameter('station', [0.0, 0.0, 0.0]).value
        self.station = Pose2D(*station)
        offset = self.declare_parameter('station_offset', 0.25).value
        self.docked = Pose2D(*from_frame(-offset, 0.0, self.station), self.station.yaw)
        self.pose = Pose2D(*self.declare_parameter('start', [-1.0, 0.0, 0.0]).value)
        self.scene = Scene(station=self.station)
        self.cmd = Twist()
        self.bridge = CvBridge()
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)
        self.tf_nav_pub = self.create_publisher(TFMessage, '/tf_nav', 10)
        self.static = tf2_ros.StaticTransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, '/localization/pose', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/sim/robot_pose', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, '/points', 5)
        self.image_pub = self.create_publisher(Image, '/naoqi_driver/camera/bottom/image_raw', 2)
        self.info_pub = self.create_publisher(CameraInfo,
                                              '/naoqi_driver/camera/bottom/camera_info', 2)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.create_subscription(Twist, '/cmd_vel_raw', self.on_cmd, 10)
        self.publish_static()
        self.dt = 0.02
        # Rendering an image takes a while: the robot and TF get their own
        # thread, as do the lidar and the camera.
        self.create_timer(self.dt, self.step, MutuallyExclusiveCallbackGroup())
        self.create_timer(0.1, self.publish_cloud, MutuallyExclusiveCallbackGroup())
        self.create_timer(0.4, self.publish_image, MutuallyExclusiveCallbackGroup())
        self.create_timer(0.5, self.publish_diagnostics)
        self.scan_seed = 0

    def on_cmd(self, msg):
        self.cmd = msg

    def publish_static(self):
        stamp = TimeMsg()
        lidar = transform('base_footprint', 'l2lidar_frame', LIDAR_IN_BASE, (0.0, 0.0, 0.0, 1.0),
                          stamp)
        z = BOTTOM_CAMERA_AXIS / np.linalg.norm(BOTTOM_CAMERA_AXIS)
        x = np.array([0.0, -1.0, 0.0])
        rotation = np.stack([x, np.cross(z, x), z], axis=1)
        camera = transform('base_footprint', 'CameraBottom_optical_frame',
                           BOTTOM_CAMERA_IN_BASE, quaternion_from_matrix(rotation), stamp)
        self.static.sendTransform([lidar, camera])

    def plugged(self):
        local = to_frame(np.array([[self.pose.x, self.pose.y]]), self.docked)[0]
        heading = abs(math.remainder(self.pose.yaw - (self.docked.yaw + math.pi), 2 * math.pi))
        return (math.hypot(*local) <= CONTACT_TOLERANCE and heading <= CONTACT_YAW_TOLERANCE)

    def step(self):
        v, w = self.cmd.linear.x, self.cmd.angular.z
        if self.plugged() and v < 0.0:
            v = 0.0                       # the contacts: the station stops the base
        yaw = self.pose.yaw + w * self.dt
        x = self.pose.x + v * math.cos(yaw) * self.dt
        y = self.pose.y + v * math.sin(yaw) * self.dt
        # The base cannot back into the station beyond the contacts.
        u = to_frame(np.array([[x, y]]), self.docked)[0][0]
        if u > 0.0:
            x, y = from_frame(0.0, to_frame(np.array([[x, y]]), self.docked)[0][1], self.docked)
        self.pose = Pose2D(x, y, math.remainder(yaw, 2 * math.pi))
        stamp = self.get_clock().now().to_msg()
        q = (0.0, 0.0, math.sin(self.pose.yaw / 2), math.cos(self.pose.yaw / 2))
        tf = TFMessage(transforms=[transform('map', 'base_footprint',
                                             (self.pose.x, self.pose.y, 0.0), q, stamp)])
        self.tf_pub.publish(tf)
        self.tf_nav_pub.publish(tf)
        odom = Odometry()
        odom.header.frame_id, odom.child_frame_id = 'map', 'base_footprint'
        odom.header.stamp = stamp
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.pose.x, self.pose.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = q[2], q[3]
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = v, w
        self.odom_pub.publish(odom)
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)

    def publish_cloud(self):
        self.scan_seed += 1
        points = lidar_scan(self.scene, self.pose, n_rays=12000, seed=self.scan_seed)
        c, s = math.cos(self.pose.yaw), math.sin(self.pose.yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        origin = np.array([self.pose.x, self.pose.y, 0.0]) + rot @ LIDAR_IN_BASE
        local = (points - origin) @ rot                       # map -> l2lidar_frame
        header = Header(frame_id='l2lidar_frame', stamp=self.get_clock().now().to_msg())
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, local.astype(np.float32)))

    def publish_image(self):
        camera = bottom_camera(self.pose)
        image = render(self.scene, camera, seed=self.scan_seed)
        stamp = self.get_clock().now().to_msg()
        msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        msg.header.frame_id, msg.header.stamp = 'CameraBottom_optical_frame', stamp
        info = CameraInfo()
        info.header = msg.header
        info.width, info.height = 640, 480
        info.distortion_model = 'plumb_bob'
        info.k = BOTTOM_CAMERA_K.ravel().tolist()
        info.d = BOTTOM_CAMERA_D.tolist()
        self.info_pub.publish(info)
        self.image_pub.publish(msg)

    def publish_diagnostics(self):
        plugged = self.plugged()
        status = DiagnosticStatus(name='naoqi_driver_battery:Status', values=[
            KeyValue(key='Percentage', value='55'), KeyValue(key='Charging', value=str(plugged)),
            KeyValue(key='Fully Charged', value='False')])
        current = DiagnosticStatus(name='naoqi_driver_battery:Current', values=[
            KeyValue(key='Current', value='1.2' if plugged else '-1.5')])
        msg = DiagnosticArray(status=[status, current])
        msg.header.stamp = self.get_clock().now().to_msg()
        self.diag_pub.publish(msg)


def main():
    rclpy.init()
    node = DockingWorld()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
