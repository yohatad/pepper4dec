# Static TF tree for the Pepper sensor rig: Unitree L2 + RealSense (colour,
# depth, IMU).
#
# WHY THIS EXISTS: bags/slam_recording, slam_recording2 and slam_bench_run* were
# all recorded with /tf_static containing ZERO messages, so the RealSense driver's
# internal extrinsics (camera_link -> color/depth/gyro/accel) and the rig mount
# transforms were never captured. Without them RTAB-Map (or any multi-sensor SLAM)
# cannot relate the camera to the lidar and refuses to start.
#
# The values below were recovered from bags/lidar_cam_calib2, which is one of the
# few bags whose /tf_static was captured (4 messages).
#
# ONE DELIBERATE DEVIATION -- l2lidar_frame -> camera_camera_link:
# the recorded value is the CAD-based initial guess. The refined result from
# direct_visual_lidar_calibration (koide3, NID-based, recorded in the header of
# FAST-LIVO2/config/unitree_l2_pepper.yaml) differs from it by 6.16 cm in
# translation -- mostly X/Z -- and 1.34 deg in rotation. The refined value is used
# here, back-composed through camera_camera_link -> camera_color_frame ->
# camera_color_optical_frame so the rest of the recorded chain still applies.
#
# Usage (include from another launch file, or run standalone):
#   ros2 launch pepper_slam pepper_sensor_tf.launch.py
#
# Play the bag WITH /tf if you want wheel odometry (pepper_odom -> base_footprint
# lives in /tf, not /tf_static).

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# (parent, child, x, y, z, qx, qy, qz, qw)
STATIC_TF = [
    # --- rig mount (recorded) ---
    ('base_footprint', 'l2lidar_frame',
     0.133000, 0.000000, 0.258200, 0.693130, -0.147438, 0.690138, -0.146770),

    # --- lidar -> camera body: REFINED, not the recorded CAD guess (see header) ---
    ('l2lidar_frame', 'camera_camera_link',
     0.021773, -0.025287, 0.016240, 0.680074, -0.159976, 0.698370, 0.155516),

    # --- RealSense internals (recorded) ---
    ('camera_camera_link', 'camera_color_frame',
     -0.000237, 0.014846, 0.000083, 0.004190, 0.000544, 0.001321, 0.999990),
    ('camera_color_frame', 'camera_color_optical_frame',
     0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5),
    ('camera_camera_link', 'camera_depth_frame',
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ('camera_depth_frame', 'camera_depth_optical_frame',
     0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5),
    ('camera_camera_link', 'camera_gyro_frame',
     -0.011740, -0.005520, 0.005100, 0.0, 0.0, 0.0, 1.0),
    ('camera_gyro_frame', 'camera_imu_frame',
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    ('camera_imu_frame', 'camera_imu_optical_frame',
     0.0, 0.0, 0.0, -0.5, 0.5, -0.5, 0.5),

    # --- L2 internal IMU (recorded). Published for completeness; note its
    #     timestamps carry a ~17 ms sawtooth from l2_sync_rate_ms:30, so prefer
    #     /camera/imu for anything timing-sensitive. ---
    ('l2lidar_frame', 'l2lidar_imu',
     -0.007698, -0.014655, 0.006670, 0.0, 0.0, 0.0, 1.0),
]


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        *[
            Node(
                package='tf2_ros', executable='static_transform_publisher',
                name=f'stf_{child}', output='log',
                parameters=[{'use_sim_time': use_sim_time}],
                arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                           '--qx', str(qx), '--qy', str(qy),
                           '--qz', str(qz), '--qw', str(qw),
                           '--frame-id', parent, '--child-frame-id', child],
            )
            for (parent, child, x, y, z, qx, qy, qz, qw) in STATIC_TF
        ],
    ])
