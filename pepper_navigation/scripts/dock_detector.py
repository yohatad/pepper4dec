#!/usr/bin/env python3
"""Measure where the charging station really is, for opennav_docking.

The station's pose is known from the map (the dock database), but the station
can be nudged. While the robot is near it, this node measures the real pose
with the L2 lidar and Pepper's bottom camera (dock_perception), fuses the two
and publishes it for opennav_docking's SimpleChargingDock
(use_external_detection_pose). It measures only while the robot stands still
and publishes only while a sensor actually sees the station: during and after
the docking server's 180 degree turn it is silent, and the server backs in on
the last pose measured at the staging pose.

The published pose is the station's centre, its yaw pointing into the station;
SimpleChargingDock's external_detection_translation_x (= -station_offset)
moves it to where base_footprint stops when docked.

Subscribes:
    lidar_topic (sensor_msgs/PointCloud2), default /points
        The L2 cloud. Points near the station are kept in the fixed frame for
        lidar.accumulate_s, and measured at lidar.rate.
    camera.image_topic (sensor_msgs/Image) + camera.info_topic (CameraInfo)
        naoqi_driver's bottom camera (converters.bottom_camera, off by
        default: see config/naoqi_bottom_camera.yaml). Skipped while the head
        moves, so overt attention must be off during docking.
    /tf, /tf_static
        The full tree (the camera hangs off the head joints), not /tf_nav.

Publishes:
    detected_dock_pose (geometry_msgs/PoseStamped, fixed_frame)
        Fused station pose, for the docking server.
    dock_detector/lidar_pose, dock_detector/camera_pose (PoseStamped)
        Each sensor's own measurement.
    dock_detector/status (std_msgs/String)
        What each sensor saw and how far the station is from the map's pose.
    dock_detector/camera_debug (sensor_msgs/Image)
        The bottom camera image with the white blob and the fit drawn in.

Parameters: see the dock_detector section of config/docking.yaml.
"""

from collections import deque
import math

from cv_bridge import CvBridge
import cv2
from dock_perception import (
    Camera, Estimate, FitLimits, footprint_corners, fuse, lidar_station_points,
    largest_cluster, fit_station, camera_station_points, Pose2D, station_from_dock,
    StationModel, WhiteThresholds, wrap_angle)
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
import tf2_ros
import yaml


def load_dock(path, dock_id):
    """(frame, Pose2D) of dock_id in an opennav_docking dock database file."""
    with open(path) as f:
        docks = yaml.safe_load(f)['docks']
    dock = docks[dock_id]
    x, y, yaw = (float(v) for v in dock['pose'])
    return dock.get('frame', 'map'), Pose2D(x, y, yaw)


def matrix_from_transform(transform):
    """(rotation 3x3, translation 3) of a geometry_msgs Transform."""
    q, t = transform.rotation, transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return rotation, np.array([t.x, t.y, t.z])


class DockDetector(Node):

    def __init__(self):
        super().__init__('dock_detector')
        p = self.declare_parameter
        self.fixed_frame = p('fixed_frame', 'map').value
        self.base_frame = p('base_frame', 'base_footprint').value
        self.model = StationModel(p('station.depth', 0.37).value,
                                  p('station.width', 0.39).value,
                                  p('station.height', 0.075).value)
        self.limits = FitLimits(p('max_shift', 0.30).value, p('max_yaw_error', 0.35).value,
                                p('size_tolerance', 0.06).value,
                                p('min_visible_depth', 0.5).value)
        self.active_radius = p('active_radius', 2.5).value
        self.tf_tolerance = Duration(seconds=p('tf_tolerance_s', 0.2).value)
        self.accumulate = Duration(seconds=p('lidar.accumulate_s', 1.5).value)
        self.lidar_sigma = (p('lidar.sigma_xy', 0.01).value, p('lidar.sigma_yaw', 0.02).value)
        self.camera_sigma = (p('camera.sigma_xy', 0.01).value, p('camera.sigma_yaw', 0.02).value)
        self.thresholds = WhiteThresholds(p('camera.max_saturation', 60).value,
                                          p('camera.min_brightness', 60).value,
                                          p('camera.min_contrast', 40).value)
        self.camera_period = Duration(seconds=1.0 / p('camera.rate', 5.0).value)
        self.max_age = Duration(seconds=p('fusion.max_age_s', 0.5).value)
        self.max_disagreement = (p('fusion.max_disagreement_xy', 0.05).value,
                                 p('fusion.max_disagreement_yaw', 0.10).value)
        dock_frame, dock = load_dock(p('dock_database', '').value, p('dock_id', 'home_dock').value)
        if dock_frame != self.fixed_frame:
            raise RuntimeError(f'dock is in {dock_frame}, expected fixed_frame {self.fixed_frame}')
        self.prior = station_from_dock(dock, p('station_offset', 0.25).value)

        self.tf_buffer = tf2_ros.Buffer()
        # Its own thread: the cloud and image callbacks wait on TF at their
        # stamps, which never arrives if TF shares their executor.
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        self.bridge = CvBridge()
        self.cloud = deque()                   # (stamp, (N, 3) fixed-frame points)
        self.estimates = {}                    # sensor -> (stamp, Estimate)
        self.status = {'lidar': 'idle', 'camera': 'idle'}
        self.camera_info = None
        self.last_image = None
        self.last_camera_pose = None
        self.active = False
        self.still = False

        self.pose_pub = self.create_publisher(PoseStamped, 'detected_dock_pose', 10)
        self.sensor_pubs = {s: self.create_publisher(PoseStamped, f'dock_detector/{s}_pose', 10)
                            for s in ('lidar', 'camera')}
        self.status_pub = self.create_publisher(String, 'dock_detector/status', 10)
        self.debug_pub = self.create_publisher(Image, 'dock_detector/camera_debug', 2)
        self.create_subscription(PointCloud2, p('lidar_topic', '/points').value,
                                 self.on_cloud, qos_profile_sensor_data)
        if p('camera.enabled', True).value:
            image_topic = p('camera.image_topic', '/naoqi_driver/camera/bottom/image_raw').value
            info_topic = p('camera.info_topic', '/naoqi_driver/camera/bottom/camera_info').value
            self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, info_topic, self.on_camera_info,
                                     qos_profile_sensor_data)
        self.create_timer(1.0 / p('lidar.rate', 5.0).value, self.measure_lidar)
        self.create_timer(0.5, self.update_active)
        self.create_timer(0.1, self.update_motion)
        self.get_logger().info(
            f'station expected at ({self.prior.x:.2f}, {self.prior.y:.2f}) '
            f'heading {math.degrees(self.prior.yaw):.0f} deg in {self.fixed_frame}')

    # -- helpers -----------------------------------------------------------

    def lookup(self, target, source, stamp, timeout=0.0):
        try:
            return self.tf_buffer.lookup_transform(target, source, stamp,
                                                   timeout=Duration(seconds=timeout))
        except tf2_ros.TransformException:
            return None

    def lookup_at(self, target, source, stamp):
        """The transform at stamp, or the latest one if it is at most tf_tolerance older.

        L2 clouds arrive far more often than the localizer's map pose, so a
        cloud is usually newer than the latest transform. While the station is
        measured the robot stands still, so the latest pose is as good.
        """
        exact = self.lookup(target, source, stamp)
        if exact is not None:
            return exact
        latest = self.lookup(target, source, Time())
        if latest is None:
            return None
        lag = abs((stamp - Time.from_msg(latest.header.stamp)).nanoseconds)
        return latest if lag <= self.tf_tolerance.nanoseconds else None

    def update_active(self):
        """Measure only near the station, so nothing runs during normal driving."""
        tf = self.lookup(self.fixed_frame, self.base_frame, Time())
        if tf is None:
            self.active = False
        else:
            t = tf.transform.translation
            self.active = math.hypot(t.x - self.prior.x, t.y - self.prior.y) <= self.active_radius
        if not self.active:
            self.cloud.clear()
            self.estimates.clear()
        self.publish_status()

    def update_motion(self):
        """Measure only while the robot stands still.

        A measurement taken while it moves is placed with a pose that is a
        little off (the cloud or image is never stamped exactly like the
        localizer's pose), and the docking server keeps accepting detections
        for a second after its 180 degree turn. Moving also invalidates the
        accumulated cloud.
        """
        now = self.get_clock().now()
        current = self.lookup(self.fixed_frame, self.base_frame, Time())
        if current is None:
            self.still = False
            return
        stamp = Time.from_msg(current.header.stamp)
        before = self.lookup(self.fixed_frame, self.base_frame, stamp - Duration(seconds=0.3))
        if before is None or now - stamp > Duration(seconds=0.5):
            still = False
        else:
            r1, t1 = matrix_from_transform(current.transform)
            r0, t0 = matrix_from_transform(before.transform)
            turned = math.acos(max(-1.0, min(1.0, (np.trace(r0.T @ r1) - 1.0) / 2.0)))
            still = np.linalg.norm(t1[:2] - t0[:2]) < 0.01 and turned < math.radians(1.0)
        if not still and self.still:
            self.cloud.clear()
            self.estimates.clear()
        if not still:
            self.status['lidar'] = self.status['camera'] = 'robot moving: not measuring'
        self.still = still

    def offset_text(self, pose):
        du = (pose.x - self.prior.x) * math.cos(self.prior.yaw) + \
            (pose.y - self.prior.y) * math.sin(self.prior.yaw)
        dv = -(pose.x - self.prior.x) * math.sin(self.prior.yaw) + \
            (pose.y - self.prior.y) * math.cos(self.prior.yaw)
        return (f'{100 * du:+.1f} cm along, {100 * dv:+.1f} cm across, '
                f'{math.degrees(wrap_angle(pose.yaw - self.prior.yaw)):+.1f} deg from the map')

    def pose_msg(self, pose, stamp):
        msg = PoseStamped()
        msg.header.frame_id = self.fixed_frame
        msg.header.stamp = stamp.to_msg()
        msg.pose.position.x, msg.pose.position.y = pose.x, pose.y
        msg.pose.orientation.z = math.sin(pose.yaw / 2.0)
        msg.pose.orientation.w = math.cos(pose.yaw / 2.0)
        return msg

    def record(self, sensor, fit, stamp, sigma):
        if fit is None:
            return
        self.estimates[sensor] = (stamp, Estimate(fit.pose, *sigma))
        self.sensor_pubs[sensor].publish(self.pose_msg(fit.pose, stamp))
        self.status[sensor] = (f'found ({fit.points} points, width {100 * fit.width:.1f} cm): '
                               + self.offset_text(fit.pose))
        self.publish_fused()

    def publish_fused(self):
        now = self.get_clock().now()
        fresh = [(stamp, e) for stamp, e in self.estimates.values() if now - stamp <= self.max_age]
        pose = fuse([e for _, e in fresh], *self.max_disagreement)
        if pose is None:
            if len(fresh) > 1:
                self.status['fused'] = 'sensors disagree: not published'
            return
        newest = max(stamp for stamp, _ in fresh)
        self.pose_pub.publish(self.pose_msg(pose, newest))
        self.status['fused'] = f'published from {len(fresh)} sensor(s): ' + self.offset_text(pose)

    def publish_status(self):
        if not self.active:
            text = 'inactive: robot is not near the station'
        else:
            text = '; '.join(f'{k}: {v}' for k, v in self.status.items())
        self.status_pub.publish(String(data=text))

    # -- lidar --------------------------------------------------------------

    def on_cloud(self, msg):
        if not (self.active and self.still):
            return
        stamp = Time.from_msg(msg.header.stamp)
        tf = self.lookup_at(self.fixed_frame, msg.header.frame_id, stamp)
        if tf is None:
            self.status['lidar'] = f'no transform {msg.header.frame_id} -> {self.fixed_frame}'
            return
        rotation, translation = matrix_from_transform(tf.transform)
        xyz = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = xyz.astype(np.float64) @ rotation.T + translation
        reach = self.limits.max_shift + math.hypot(self.model.depth, self.model.width) + 0.3
        near = np.hypot(points[:, 0] - self.prior.x, points[:, 1] - self.prior.y) <= reach
        self.cloud.append((stamp, points[near]))
        while self.cloud and stamp - self.cloud[0][0] > self.accumulate:
            self.cloud.popleft()

    def measure_lidar(self):
        if not (self.active and self.still) or not self.cloud:
            return
        stamp = self.cloud[-1][0]
        points = np.vstack([pts for _, pts in self.cloud])
        candidates = lidar_station_points(points, self.prior, self.model, self.limits)
        fit = fit_station(largest_cluster(candidates, self.prior, self.model), self.prior,
                          self.model, self.limits)
        if fit is None:
            self.status['lidar'] = f'not found ({len(candidates)} candidate points)'
        self.record('lidar', fit, stamp, self.lidar_sigma)

    # -- bottom camera --------------------------------------------------------

    def on_camera_info(self, msg):
        self.camera_info = msg

    def on_image(self, msg):
        if not (self.active and self.still) or self.camera_info is None:
            return
        stamp = Time.from_msg(msg.header.stamp)
        if self.last_image is not None and stamp - self.last_image < self.camera_period:
            return
        self.last_image = stamp
        tf = self.lookup_at(self.fixed_frame, msg.header.frame_id, stamp)
        base = self.lookup_at(self.fixed_frame, self.base_frame, stamp)
        if tf is None or base is None:
            self.status['camera'] = f'no transform {msg.header.frame_id} -> {self.fixed_frame}'
            return
        rotation, position = matrix_from_transform(tf.transform)
        if self.head_moving(msg.header.frame_id, rotation, position, stamp):
            self.status['camera'] = 'head moving: skipped (turn overt attention off)'
            return
        info = self.camera_info
        camera = Camera(np.array(info.k).reshape(3, 3), np.array(info.d), rotation, position)
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        floor_z = base.transform.translation.z
        top, floor, blob = camera_station_points(image, camera, self.prior, self.model,
                                                 self.limits, self.thresholds, floor_z)
        fit = fit_station(top, self.prior, self.model, self.limits, near_candidates=floor)
        if fit is None:
            self.status['camera'] = ('no white blob where the station should be' if len(top) == 0
                                     else 'white blob does not fit the station')
        self.record('camera', fit, stamp, self.camera_sigma)
        self.publish_debug(image, blob, camera, fit, floor_z, msg.header)

    def head_moving(self, frame, rotation, position, stamp):
        """True if the camera moved more than 1 cm or 1 degree since 0.2 s ago."""
        before = self.lookup(self.fixed_frame, frame, stamp - Duration(seconds=0.2), timeout=0.0)
        if before is None:
            return False
        r0, t0 = matrix_from_transform(before.transform)
        angle = math.acos(max(-1.0, min(1.0, (np.trace(r0.T @ rotation) - 1.0) / 2.0)))
        return np.linalg.norm(position - t0) > 0.01 or angle > math.radians(1.0)

    def publish_debug(self, image, blob, camera, fit, floor_z, header):
        if self.debug_pub.get_subscription_count() == 0:
            return
        debug = image.copy()
        debug[blob > 0] = (0.5 * debug[blob > 0] + (0, 127, 0)).astype(np.uint8)
        for pose, colour in ((self.prior, (0, 200, 255)), (fit.pose if fit else None,
                                                           (0, 0, 255))):
            if pose is None:
                continue
            uv = camera.project(footprint_corners(pose, self.model, 0.0,
                                                  floor_z + self.model.height))
            if not np.isnan(uv).any():
                cv2.polylines(debug, [uv.astype(np.int32)], True, colour, 2)
        out = self.bridge.cv2_to_imgmsg(debug, 'bgr8')
        out.header = header
        self.debug_pub.publish(out)


def main():
    rclpy.init()
    node = DockDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
