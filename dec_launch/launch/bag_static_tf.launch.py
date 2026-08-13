"""
Republish the static TF tree recorded in slam_august_8_bag.

Why this exists: the bag holds its 2 /tf_static messages at t+0.0066s, so any
playback with `--start-offset` seeks past them. rosbag2 (Humble) still creates
the TRANSIENT_LOCAL publisher from the bag metadata - /tf_static shows up in
`ros2 topic list` - but never writes a sample, so `ros2 run tf2_tools
view_frames` returns frame_yaml='[]' and RViz has no fixed frame.

Run this alongside the player to get the tree back regardless of the offset:

    ros2 launch dec_launch bag_static_tf.launch.py

The values below are dumped verbatim from the bag's /tf_static messages, not
re-measured, so the tree matches what the recording actually used. They differ
slightly from my_realsense_with_staticframes.launch.py, which carries an older
hand-measured camera_color_frame extrinsic - prefer these when replaying.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# (name, parent, child, x, y, z, qx, qy, qz, qw)
STATIC_TRANSFORMS = [
    ('camera_link_to_accel_frame', 'camera_camera_link', 'camera_accel_frame',
     '-0.011740', '-0.005520', '0.005100', '0.0', '0.0', '0.0', '1.0'),
    ('accel_frame_to_accel_optical', 'camera_accel_frame', 'camera_accel_optical_frame',
     '0.0', '0.0', '0.0', '-0.5', '0.5', '-0.5', '0.5'),
    ('camera_link_to_gyro_frame', 'camera_camera_link', 'camera_gyro_frame',
     '-0.011740', '-0.005520', '0.005100', '0.0', '0.0', '0.0', '1.0'),
    ('gyro_frame_to_gyro_optical', 'camera_gyro_frame', 'camera_gyro_optical_frame',
     '0.0', '0.0', '0.0', '-0.5', '0.5', '-0.5', '0.5'),
    ('gyro_frame_to_imu_frame', 'camera_gyro_frame', 'camera_imu_frame',
     '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0'),
    ('imu_frame_to_imu_optical', 'camera_imu_frame', 'camera_imu_optical_frame',
     '0.0', '0.0', '0.0', '-0.5', '0.5', '-0.5', '0.5'),
    ('camera_link_to_depth_frame', 'camera_camera_link', 'camera_depth_frame',
     '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '1.0'),
    ('depth_frame_to_depth_optical', 'camera_depth_frame', 'camera_depth_optical_frame',
     '0.0', '0.0', '0.0', '-0.5', '0.5', '-0.5', '0.5'),
    ('camera_link_to_color_frame', 'camera_camera_link', 'camera_color_frame',
     '-0.000237', '0.014846', '0.000083',
     '0.004190', '0.000544', '0.001321', '0.999990'),
    ('color_frame_to_color_optical', 'camera_color_frame', 'camera_color_optical_frame',
     '0.0', '0.0', '0.0', '-0.5', '0.5', '-0.5', '0.5'),
    ('l2lidar_frame_to_lidar_imu', 'l2lidar_frame', 'l2lidar_frame_imu',
     '-0.007698', '-0.014655', '0.006670', '0.0', '0.0', '0.0', '1.0'),
]


def generate_launch_description():
    # Defaults to true because this launch file exists for bag playback, which
    # is always run with --clock. tf2 ignores timestamps on static transforms,
    # so this only keeps the nodes consistent with the rest of the graph.
    use_sim_time = LaunchConfiguration('use_sim_time')

    nodes = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=name,
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[x, y, z, qx, qy, qz, qw, parent, child],
        )
        for name, parent, child, x, y, z, qx, qy, qz, qw in STATIC_TRANSFORMS
    ]

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use /clock from `ros2 bag play --clock`.',
        ),
        *nodes,
    ])
