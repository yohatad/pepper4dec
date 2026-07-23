#!/usr/bin/env python3
"""
Calibrate L2 LiDAR extrinsic against RealSense depth using ICP.

Subscribes to the aligned depth image + camera_info directly so it does NOT
depend on the realsense pointcloud publisher (pointcloud.enable).

Usage:
  ros2 launch l2lidar_node l2lidar.launch.py
  ros2 launch dec_launch my_realsense_bottom.launch.py
  ros2 run dec_launch lidar_depth_calibrator.py

Point both sensors at a wall corner or cluttered area, then run.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf2_ros
import numpy as np
import open3d as o3d


# ── Helpers ──────────────────────────────────────────────────────────────────

def quat_to_matrix(qx, qy, qz, qw):
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def matrix_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        S = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw])
    if q[3] < 0:
        q = -q
    return q


def tf_to_matrix(tf_stamped):
    t = tf_stamped.transform.translation
    r = tf_stamped.transform.rotation
    M = np.eye(4)
    M[:3, :3] = quat_to_matrix(r.x, r.y, r.z, r.w)
    M[:3, 3] = [t.x, t.y, t.z]
    return M


def lidar_cloud_to_o3d(cloud_msg, min_range=0.3, max_range=4.0):
    pts = np.array(
        [[p[0], p[1], p[2]] for p in
         pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)],
        dtype=np.float64)
    if len(pts) == 0:
        return None
    d = np.linalg.norm(pts, axis=1)
    pts = pts[(d > min_range) & (d < max_range)]
    if len(pts) < 200:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def depth_image_to_o3d(depth_msg, camera_info, min_range=0.3, max_range=3.0):
    """Back-project aligned depth image to 3-D point cloud using camera intrinsics."""
    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]
    w  = camera_info.width
    h  = camera_info.height

    # Depth image is 16-bit unsigned in millimetres
    depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(h, w).astype(np.float64)
    depth *= 0.001  # mm → m

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth > min_range) & (depth < max_range)

    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    pts = np.column_stack([x, y, z])
    if len(pts) < 200:
        return None, camera_info.header.frame_id

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, camera_info.header.frame_id


def preprocess(pcd, voxel=0.05, normal_radius=0.15):
    ds = pcd.voxel_down_sample(voxel)
    ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    return ds


def crop_lidar_to_fov(pcd_lidar, M_init, fx, fy, cx, cy, width, height, margin=20):
    """
    Remove LiDAR points that don't project into the camera image.

    Out-of-FOV LiDAR points have no depth correspondences, which confuses ICP
    and inflates RMSE.  We use M_init (current TF estimate) to project each
    point and keep only those landing within the image bounds ± margin pixels.
    """
    pts = np.asarray(pcd_lidar.points)
    if len(pts) == 0:
        return pcd_lidar
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_cam = (M_init @ pts_h.T).T[:, :3]
    in_front = pts_cam[:, 2] > 0.1
    u = np.where(in_front, fx * pts_cam[:, 0] / np.where(pts_cam[:, 2] != 0, pts_cam[:, 2], 1) + cx, -1)
    v = np.where(in_front, fy * pts_cam[:, 1] / np.where(pts_cam[:, 2] != 0, pts_cam[:, 2], 1) + cy, -1)
    mask = (in_front &
            (u >= -margin) & (u < width  + margin) &
            (v >= -margin) & (v < height + margin))
    cropped = o3d.geometry.PointCloud()
    cropped.points = o3d.utility.Vector3dVector(pts[mask])
    return cropped


# ── Node ─────────────────────────────────────────────────────────────────────

class LidarDepthCalibrator(Node):

    LIDAR_TOPIC     = '/points'
    DEPTH_IMG_TOPIC = '/camera/aligned_depth_to_color/image_raw'
    DEPTH_INFO_TOPIC = '/camera/aligned_depth_to_color/camera_info'

    def __init__(self):
        super().__init__('lidar_depth_calibrator')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._pending = None

        self.sub_lidar = Subscriber(self, PointCloud2, self.LIDAR_TOPIC)
        self.sub_depth = Subscriber(self, Image,       self.DEPTH_IMG_TOPIC)
        self.sub_info  = Subscriber(self, CameraInfo,  self.DEPTH_INFO_TOPIC)

        self.sync = ApproximateTimeSynchronizer(
            [self.sub_lidar, self.sub_depth, self.sub_info],
            queue_size=5, slop=0.15)
        self.sync.registerCallback(self._on_sync)

        self.get_logger().info(
            f'Waiting for synced data on:\n'
            f'  LiDAR      : {self.LIDAR_TOPIC}\n'
            f'  Depth image: {self.DEPTH_IMG_TOPIC}\n'
            f'  Camera info: {self.DEPTH_INFO_TOPIC}\n'
            f'Point sensors at a wall corner or furniture for best results.')

    def _on_sync(self, lidar_msg, depth_msg, info_msg):
        if self._pending is None:
            self._pending = (lidar_msg, depth_msg, info_msg)

    def _lookup(self, target, source, timeout=5.0):
        return self.tf_buffer.lookup_transform(
            target, source,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=timeout))

    def run(self):
        while rclpy.ok() and self._pending is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not rclpy.ok():
            return

        lidar_msg, depth_msg, info_msg = self._pending
        self.get_logger().info('Got synchronized data — building point clouds...')

        lidar_frame = lidar_msg.header.frame_id
        pcd_lidar = lidar_cloud_to_o3d(lidar_msg, min_range=0.3, max_range=4.0)
        pcd_depth, depth_frame = depth_image_to_o3d(depth_msg, info_msg,
                                                     min_range=0.3, max_range=3.0)

        if pcd_lidar is None or pcd_depth is None:
            self.get_logger().error(
                'Too few valid points (<200). Move closer to geometry.')
            return

        self.get_logger().info(
            f'LiDAR frame: {lidar_frame} | Depth frame: {depth_frame}')

        # Spin for 2 s so the TF buffer receives /tf_static messages.
        # time.sleep() would freeze the executor — spin_once() keeps it alive.
        import time
        self.get_logger().info('Waiting 2 s for TF buffer to populate...')
        deadline = time.time() + 2.0
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

        try:
            M_init = tf_to_matrix(self._lookup(depth_frame, lidar_frame))
        except Exception as e:
            self.get_logger().error(f'TF lookup (initial guess) failed: {e}')
            return

        # Crop LiDAR to camera FOV before downsampling.
        # Out-of-FOV LiDAR points have no depth correspondences; keeping them
        # inflates RMSE and drives ICP toward wrong local minima.
        fx, fy = info_msg.k[0], info_msg.k[4]
        cx, cy = info_msg.k[2], info_msg.k[5]
        w, h   = info_msg.width, info_msg.height
        pcd_lidar_fov = crop_lidar_to_fov(pcd_lidar, M_init, fx, fy, cx, cy, w, h)

        # LiDAR: sparse raw data → 5 cm voxels are appropriate
        # Depth: dense 640×480 image → 1 cm voxels to preserve detail on flat walls
        src = preprocess(pcd_lidar_fov, voxel=0.05)
        tgt = preprocess(pcd_depth, voxel=0.01, normal_radius=0.05)
        self.get_logger().info(
            f'After downsampling: LiDAR={len(src.points)} pts (FOV-cropped), '
            f'Depth={len(tgt.points)} pts')

        if len(src.points) < 50:
            self.get_logger().error(
                'Too few LiDAR points in camera FOV (<50). '
                'Check that LiDAR and camera are both active and TF is plausible.')
            return

        icp = o3d.pipelines.registration.registration_icp
        p2p  = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        p2pl = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        # Stage 1: coarse point-to-point (robust far from optimum)
        r1 = icp(src, tgt, 0.15, M_init, p2p,
                 o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        self.get_logger().info(
            f'Coarse ICP (P2P):  fitness={r1.fitness:.3f}  '
            f'RMSE={r1.inlier_rmse*100:.1f} cm')

        # Stage 2: fine point-to-plane
        r2 = icp(src, tgt, 0.05, r1.transformation, p2pl,
                 o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        self.get_logger().info(
            f'Fine  ICP (P2Pl): fitness={r2.fitness:.3f}  '
            f'RMSE={r2.inlier_rmse*1000:.1f} mm')

        if r2.fitness < 0.40:
            self.get_logger().warn(
                f'Low fitness ({r2.fitness:.3f}). Try in a room corner with '
                'nearby objects for better geometry.')

        # M_DL maps lidar → depth_frame ; M_LD is its inverse
        M_DL = r2.transformation
        M_LD = np.linalg.inv(M_DL)

        # T_DC maps camera_camera_link → depth_frame
        # T_LC = M_LD @ T_DC  maps camera_camera_link → lidar_frame
        try:
            T_DC = tf_to_matrix(self._lookup(depth_frame, 'camera_camera_link'))
        except Exception as e:
            self.get_logger().error(f'Internal TF lookup failed: {e}')
            return

        T_LC = M_LD @ T_DC
        t = T_LC[:3, 3]
        q = matrix_to_quat(T_LC[:3, :3])

        try:
            tf_cur = self._lookup('camera_camera_link', lidar_frame)
            t_cur = np.array([tf_cur.transform.translation.x,
                              tf_cur.transform.translation.y,
                              tf_cur.transform.translation.z])
            shift_mm = np.linalg.norm(t - t_cur) * 1000
        except Exception:
            shift_mm = float('nan')

        sep = '=' * 64
        print(f'\n{sep}')
        print('  ICP CALIBRATION RESULT')
        print(f'  Depth frame    : {depth_frame}')
        print(f'  Fitness        : {r2.fitness:.3f}  (>0.6 good, >0.4 ok)')
        print(f'  RMSE           : {r2.inlier_rmse * 1000:.2f} mm  (<10 mm good)')
        print(f'  Shift from CAD : {shift_mm:.1f} mm')
        print(sep)
        print('\nPaste into my_realsense_bottom.launch.py '
              '(l2lidar_to_realsense_tf node):\n')
        print(f"                '--x',  '{t[0]:.6f}',")
        print(f"                '--y', '{t[1]:.6f}',")
        print(f"                '--z',  '{t[2]:.6f}',")
        print(f"                '--qx', '{q[0]:.6f}',")
        print(f"                '--qy', '{q[1]:.6f}',")
        print(f"                '--qz', '{q[2]:.6f}',")
        print(f"                '--qw', '{q[3]:.6f}',")
        print(f'\n{sep}\n')


def main(args=None):
    rclpy.init(args=args)
    node = LidarDepthCalibrator()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
