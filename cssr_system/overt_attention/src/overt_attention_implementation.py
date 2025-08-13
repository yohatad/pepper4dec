#!/usr/bin/env python3
import os
import yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# ROS 2 message & service imports
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D, Twist
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from cssr_system.srv import SetMode, GetMode

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
CONFIG_DEFAULTS = {
    'camera':                'RealSenseCamera',
    'realignment_threshold': 50,
    'x_offset_to_head_yaw':  0.0,
    'y_offset_to_head_pitch':0.0,
    'social_attention_mode': 'random',
    'num_faces_social_att':  3,
    'engagement_timeout':    12.0,
    'use_sound':             False,
    'use_compressed_images': False,
    'verbose_mode':          False,
}

TOPIC_DEFAULTS = {
    'RealSenseCamera':                '/camera/color/image_raw',
    'RealSenseCameraCompressed':      '/camera/color/image_raw/compressed',
    'RealSenseCameraDepth':           '/camera/depth/image_raw',
    'RealSenseCameraDepthCompressed': '/camera/depth/image_raw/compressed',
    'FrontCamera':                    '/naoqi_driver/camera/front/image_raw',
    'DepthCamera':                    '/naoqi_driver/camera/depth/image_raw',
    'JointAngles':                    '/joint_angles',
    'Wheels':                         '/cmd_vel',
    'JointStates':                    '/joint_states',
    'RobotPose':                      '/robotLocalization/pose',
    'FaceDetection':                  '/faceDetection/data',
    'SoundLocalization':              '/soundDetection/direction',
    'SetMode':                        '/overtAttention/set_mode',
    'GetMode':                        '/overtAttention/get_mode',
}


class OvertAttentionSystem(Node):
    def __init__(self):
        super().__init__('overt_attention')
        if not self.initialize():
            self.get_logger().error('Initialization failed; shutting down.')
            rclpy.shutdown()

    def initialize(self) -> bool:
        self.get_logger().info('Initializing OvertAttentionSystem...')

        # --- 1) Load configuration ---
        pkg_share = get_package_share_directory('cssr_system')
        cfg_path = os.path.join(pkg_share, 'overtAttention', 'config',
                                 'overtAttentionConfiguration.yaml')
        cfg = self.load_yaml(cfg_path, CONFIG_DEFAULTS)

        camera_key = cfg['camera'].lower()
        use_compr = cfg['use_compressed_images']
        # decide which camera topic key
        if camera_key == 'realsensecamera':
            camera_key = 'RealSenseCameraCompressed' if use_compr else 'RealSenseCamera'
        elif camera_key == 'peppercamera':
            camera_key = 'FrontCamera'
        else:
            self.get_logger().error(f'Unsupported camera type: {cfg["camera"]}')
            return False

        # --- 2) Load topics ---
        topics_path = os.path.join(pkg_share, 'overtAttention', 'data', 'pepperTopics.yaml')
        topics = self.load_yaml(topics_path, TOPIC_DEFAULTS)

        # --- 3) Subscriptions & Publishers ---
        self.camera_sub = self.create_subscription(Image, topics[camera_key], self.camera_callback, 10)
        self.joint_states_sub = self.create_subscription(JointState, topics['JointStates'], self.joint_states_callback, 10)
        self.sound_loc_sub = self.create_subscription(Float32, topics['SoundLocalization'], self.sound_localization_callback, 10)
        self.robot_pose_sub = self.create_subscription(Pose2D, topics['RobotPose'], self.robot_pose_callback, 10)

        self.velocity_pub = self.create_publisher(Twist, topics['Wheels'], 10)
        self.joint_angles_pub = self.create_publisher(JointAnglesWithSpeed, topics['JointAngles'], 10)

        # --- 4) Services ---
        self.set_mode_srv = self.create_service(SetMode, topics['SetMode'], self.set_mode_callback)
        self.get_mode_srv = self.create_service(GetMode, topics['GetMode'], self.get_mode_callback)

        self.get_logger().info('Initialization complete.')
        return True

    def load_yaml(self, path: str, defaults: dict) -> dict:
        """ Utility to load a YAML file and merge it onto defaults. """
        data = defaults.copy()
        try:
            with open(path, 'r') as f:
                override = yaml.safe_load(f) or {}
            data.update(override)
        except Exception as e:
            self.get_logger().error(f'Could not load YAML {path}: {e}')
        return data

    # --- Callbacks ---
    def camera_callback(self, msg: Image):
        # TODO: process incoming image
        pass

    def joint_states_callback(self, msg: JointState):
        # TODO: process joint state
        pass

    def sound_localization_callback(self, msg: Float32):
        # TODO: process sound direction
        pass

    def robot_pose_callback(self, msg: Pose2D):
        # TODO: process robot pose
        pass

    def set_mode_callback(self, request, response):
        # TODO: set your internal state
        # response.success = True
        return response

    def get_mode_callback(self, request, response):
        # TODO: fill response.mode = current_mode
        return response

