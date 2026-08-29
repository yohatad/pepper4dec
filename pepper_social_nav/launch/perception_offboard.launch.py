# Offboard perception for socially-aware navigation -- RUN THIS ON THE LAPTOP.
#
# WHY OFFBOARD. YOLOv11m on two camera streams does not fit the Jetson budget
# next to FAST-LIO, the costmaps and MPPI. Compressed images go robot ->
# laptop, and what comes back is social_nav_msgs/Pedestrians: a few hundred
# bytes at 15 Hz. The expensive direction is the one that gets compressed.
#
# WHAT MUST NOT MOVE HERE. The controller, the costmaps and the collision
# monitor stay on the robot (pepper_nav2_social.launch.py). A WiFi dropout must
# never sit between a lidar return and the brakes. Everything this launch feeds
# degrades to "no social cost" when the link goes stale (message_timeout in
# nav2_params_social.yaml), leaving the on-robot lidar safety layer intact.
#
# LATENCY IS REAL AND IS HANDLED. Compress -> WiFi -> decompress -> YOLO is
# easily 100-300 ms. At a 0.5 m/s robot closing with a 1.4 m/s human that is
# tens of centimetres of error, so the tracker filters at the MEASUREMENT
# timestamp (from CameraInfo, not arrival time) and predicts forward to now,
# and the SFM critic starts its constant-velocity extrapolation at the
# message's actual age. Do not "fix" either by stamping with now().
#
# TWO CAMERAS, because neither alone is enough:
#   realsense  Body-mounted, has depth, but sits at 0.308 m with the optical
#              axis pitched DOWN 2.31 deg -- it sees waist-height at 2 m and
#              does not frame a whole person until ~4.5 m.
#   front      Pepper's head camera at ~1.2 m, pans with the head, frames whole
#              people at social range. No depth: the tracker ranges it by
#              ground-plane intersection through the feet.
#
# PREREQUISITES ON THE ROBOT SIDE:
#   * ROS_DOMAIN_ID identical on both machines, same DDS vendor, same subnet.
#   * The RealSense driver publishing .../compressed and .../compressedDepth.
#   * naoqi_driver2 running (front camera + camera_info + /joint_states, which
#     is what makes the head pan cancel out in TF).
#   * /tf and /tf_static reaching the laptop -- the tracker cannot deproject
#     without map -> camera at the image timestamp.
#
# Usage:
#   ros2 launch pepper_social_nav perception_offboard.launch.py
#   ros2 launch pepper_social_nav perception_offboard.launch.py front:=false

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Shared detector tuning. Passed as a dict rather than via
# person_detection's config yaml because that file is keyed on the node name
# 'person_detection', and two instances need two different names.
def detector_params(camera, use_sim_time):
    return {
        'camera': camera,
        'use_compressed': False,   # decompressed upstream by image_decompressor
        'image_timeout': 2.0,
        'verbose_mode': False,
        'confidence_threshold': 0.5,
        'target_classes': ['person'],
        'track_threshold': 0.45,
        'track_buffer': 30,
        'match_threshold': 0.8,
        'frame_rate': 15,
        'use_sim_time': use_sim_time,
    }


def generate_launch_description():
    social_share = get_package_share_directory('pepper_social_nav')
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use bag/simulation clock. Set true when replaying the '
                    'people bag recorded by scripts/record_people_bag.sh.')
    declare_realsense = DeclareLaunchArgument(
        'realsense', default_value='true',
        description='Run detection on the body-mounted RealSense stream.')
    declare_front = DeclareLaunchArgument(
        'front', default_value='true',
        description="Run detection on Pepper's head camera stream.")
    declare_tracker = DeclareLaunchArgument(
        'tracker', default_value='true',
        description='Run the people tracker here. Set false only if the robot '
                    'is running it (pepper_nav2_social run_tracker:=true).')

    # --- decompression ------------------------------------------------------
    # Only compressed images cross the link. The RealSense pair lands on the
    # *_custom topics person_detection already expects (person_detection/data/
    # pepper_topics.yaml), so nothing downstream needs rewiring.
    color_decompressor = Node(
        package='image_decompressor', executable='image_decompressor_node_exe',
        name='color_decompressor', output='screen',
        parameters=[{
            'compressed_image_topic': '/camera/color/image_raw/compressed',
            'output_topic': '/camera/color/image_raw_custom',
        }],
        condition=IfCondition(LaunchConfiguration('realsense')),
    )
    depth_decompressor = Node(
        package='image_decompressor', executable='depth_decompressor_node_exe',
        name='depth_decompressor', output='screen',
        parameters=[{
            'compressed_depth_topic':
                '/camera/aligned_depth_to_color/image_raw/compressedDepth',
            'output_topic': '/camera/aligned_depth_to_color/image_raw_custom',
        }],
        condition=IfCondition(LaunchConfiguration('realsense')),
    )
    # dec_common resolves the pepper camera to /pepper/front/image_raw from its
    # topics yaml, while naoqi_driver2 publishes camera/front/image_raw -- so
    # this decompressor also bridges that naming gap in one step.
    front_decompressor = Node(
        package='image_decompressor', executable='image_decompressor_node_exe',
        name='front_decompressor', output='screen',
        parameters=[{
            'compressed_image_topic': '/camera/front/image_raw/compressed',
            'output_topic': '/pepper/front/image_raw',
        }],
        condition=IfCondition(LaunchConfiguration('front')),
    )

    # --- detection ----------------------------------------------------------
    # Two instances of the SAME node. dec_common picks the input topics from
    # the camera type, and the output is remapped so the tracker can tell the
    # two streams apart (they need different ranging strategies).
    detector_realsense = Node(
        package='person_detection', executable='person_detection',
        name='person_detection_realsense', output='screen',
        parameters=[detector_params('realsense', use_sim_time)],
        remappings=[('/person_detection/data', '/person_detection/realsense/data'),
                    ('/person_detection/debug', '/person_detection/realsense/debug'),
                    ('/person_detection/depth_debug',
                     '/person_detection/realsense/depth_debug')],
        condition=IfCondition(LaunchConfiguration('realsense')),
    )
    detector_front = Node(
        package='person_detection', executable='person_detection',
        name='person_detection_front', output='screen',
        parameters=[detector_params('pepper', use_sim_time)],
        remappings=[('/person_detection/data', '/person_detection/front/data'),
                    ('/person_detection/debug', '/person_detection/front/debug'),
                    ('/person_detection/depth_debug',
                     '/person_detection/front/depth_debug')],
        condition=IfCondition(LaunchConfiguration('front')),
    )

    # person_detection is a managed lifecycle node and will sit in
    # 'unconfigured' forever without something to drive it. On the robot that is
    # dec_launch's manager; offboard, this is it.
    #
    # ONE MANAGER PER CAMERA, sharing the condition of the node it manages. A
    # single manager listing both would block in configure() forever whenever
    # one camera is disabled, waiting on a node nothing ever started.
    def perception_manager(suffix, managed, arg):
        return Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_perception_' + suffix, output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': True,
                # Generous: the first activation loads the ONNX model.
                'bond_timeout': 10.0,
                'node_names': [managed],
            }],
            condition=IfCondition(LaunchConfiguration(arg)),
        )

    manager_realsense = perception_manager(
        'realsense', 'person_detection_realsense', 'realsense')
    manager_front = perception_manager(
        'front', 'person_detection_front', 'front')

    # --- tracking -----------------------------------------------------------
    people_tracker = Node(
        package='pepper_social_nav', executable='people_tracker',
        name='people_tracker', output='screen',
        parameters=[
            os.path.join(social_share, 'config', 'people_tracker.yaml'),
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration('tracker')),
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_realsense,
        declare_front,
        declare_tracker,
        color_decompressor,
        depth_decompressor,
        front_decompressor,
        detector_realsense,
        detector_front,
        manager_realsense,
        manager_front,
        people_tracker,
    ])
