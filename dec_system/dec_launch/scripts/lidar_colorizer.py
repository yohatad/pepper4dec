import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import tf2_ros
import numpy as np
import cv2
from cv_bridge import CvBridge
import struct

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


class LidarColorizer(Node):
    def __init__(self):
        super().__init__('lidar_colorizer')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub_info = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_cb,
            SENSOR_QOS)

        self.sub_points = Subscriber(self, PointCloud2, '/points')
        self.sub_image  = Subscriber(self, Image, '/camera/color/image_raw',
                                     qos_profile=SENSOR_QOS)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_points, self.sub_image], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)

        self.pub = self.create_publisher(PointCloud2, '/points_colored', 10)
        self.get_logger().info('lidar_colorizer ready')

    def camera_info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.img_w = msg.width
            self.img_h = msg.height

    def callback(self, cloud_msg, image_msg):
        if self.camera_matrix is None:
            return

        try:
            # Use the optical frame so Z=forward and cv2.projectPoints works correctly.
            # camera_camera_link uses REP-103 (X=forward, Z=up); the optical frame has Z=forward.
            tf = self.tf_buffer.lookup_transform(
                'camera_color_optical_frame',
                cloud_msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception:
            return

        # Build rotation matrix from LiDAR frame to camera_color_optical_frame
        t = tf.transform.translation
        q = tf.transform.rotation
        tx, ty, tz = t.x, t.y, t.z
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
        ])

        img = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')

        pts = np.array([[p[0], p[1], p[2]] for p in
                        pc2.read_points(cloud_msg, field_names=('x','y','z'),
                                        skip_nans=True)], dtype=np.float32)
        if len(pts) == 0:
            return

        # Transform points into camera frame
        pts_cam = (R @ pts.T).T + np.array([tx, ty, tz])

        # Keep only points in front of camera (z > 0.1)
        mask = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[mask]
        pts_orig = pts[mask]

        # Project onto image plane
        uvs, _ = cv2.projectPoints(
            pts_cam.reshape(-1, 1, 3),
            np.zeros(3), np.zeros(3),
            self.camera_matrix, self.dist_coeffs)
        uvs = uvs.reshape(-1, 2)

        # Sample color for each projected point
        colored = []
        for i, (u, v) in enumerate(uvs):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < self.img_w and 0 <= vi < self.img_h:
                b, g, r = img[vi, ui]
                rgb = struct.unpack('f', struct.pack('I',
                    (int(r) << 16) | (int(g) << 8) | int(b)))[0]
                colored.append([pts_orig[i, 0], pts_orig[i, 1],
                                pts_orig[i, 2], rgb])

        if not colored:
            return

        fields = [
            pc2.PointField(name='x', offset=0,  datatype=7, count=1),
            pc2.PointField(name='y', offset=4,  datatype=7, count=1),
            pc2.PointField(name='z', offset=8,  datatype=7, count=1),
            pc2.PointField(name='rgb', offset=12, datatype=7, count=1),
        ]
        out = pc2.create_cloud(cloud_msg.header, fields, colored)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LidarColorizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
