#!/usr/bin/env python3
"""
gesture_execution_implementation.py

ROS2 Action Server for Pepper robot gesture execution system — Lifecycle version.
Provides gesture execution with elapsed time feedback for synchronisation.

Lifecycle:
  configure  → load YAML configs, init kinematics, create publishers + action server
  activate   → create subscriptions (joint_states, robot_pose)
  deactivate → destroy subscriptions
  cleanup    → destroy publishers

Author: Yohannes Haile
Date: October 11, 2025
Version: v1.0
"""

import math
import time
import yaml
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose2D, Point
from visualization_msgs.msg import Marker
from naoqi_bridge_msgs.msg import JointAnglesTrajectory

from dec_interfaces.action import Gesture

from .pepper_kinematics_utilities import PepperKinematicsUtilities, RIGHT_ARM, LEFT_ARM

# ── Constants ───────────────────────────────────────────────────────────────────
MIN_GESTURE_DURATION = 1000   # ms
MAX_GESTURE_DURATION = 5000   # ms

DEICTIC_GESTURES  = 'deictic'
ICONIC_GESTURES   = 'iconic'
SYMBOLIC_GESTURES = 'symbolic'
BOWING_GESTURE    = 'bow'
NODDING_GESTURE   = 'nod'

UPPER_ARM_LENGTH   = 150.0
SHOULDER_OFFSET_X  = -57.0
SHOULDER_OFFSET_Y  = 149.74
SHOULDER_OFFSET_Z  = 86.82
TORSO_HEIGHT       = 0.0

MIN_RSHOULDER_PITCH = -2.0857
MAX_RSHOULDER_PITCH =  2.0857
MIN_RSHOULDER_ROLL  = -1.5621
MAX_RSHOULDER_ROLL  = -0.0087
MIN_LSHOULDER_PITCH = -2.0857
MAX_LSHOULDER_PITCH =  2.0857
MIN_LSHOULDER_ROLL  =  0.0087
MAX_LSHOULDER_ROLL  =  1.5621

MIN_BOW_ANGLE = 5
MAX_BOW_ANGLE = 45
MIN_NOD_ANGLE = 5
MAX_NOD_ANGLE = 30


@dataclass
class JointLimits:
    HEAD_YAW_RANGE:   Tuple[float, float] = (-2.0857, 2.0857)
    HEAD_PITCH_RANGE: Tuple[float, float] = (-0.7068, 0.6371)
    DEFAULT_HEAD_PITCH: float = -0.2
    DEFAULT_HEAD_YAW:   float = 0.0


@dataclass
class RobotPose:
    x:     float = 0.0
    y:     float = 0.0
    theta: float = 0.0


class ConfigManager:
    """Loads configuration and topic mappings from YAML."""

    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        self.config = self.load_config()
        self.topics = self.load_topics()

    def load_config(self) -> Dict:
        config = {
            'gestureDescriptors': 'gesture.yaml',
            'robotTopics': 'pepperTopics.yaml',
            'verboseMode': True,
        }
        config_path = self.package_path / 'config' / 'gesture_execution_configuration.yaml'
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config.update(yaml.safe_load(f) or {})
        except Exception as e:
            print(f'Warning: could not load config {config_path}: {e}')
        return config

    def load_topics(self) -> Dict:
        topics = {
            'JointAngles': '/joint_angles',
            'Wheels':      '/cmd_vel',
            'JointStates': '/joint_states',
            'RobotPose':   '/robotLocalization/pose',
        }
        topics_path = self.package_path / 'data' / 'pepper_topics.yaml'
        try:
            if topics_path.exists():
                with open(topics_path, 'r') as f:
                    yaml_data = yaml.safe_load(f) or {}
                    if 'topics' in yaml_data:
                        topics.update(yaml_data['topics'])
        except Exception as e:
            print(f'Warning: could not load topics {topics_path}: {e}')
        return topics


class GestureDescriptorManager:
    """Loads and queries gesture descriptors from YAML."""

    def __init__(self, package_path: Path, config: Dict):
        self.package_path = package_path
        self.config       = config
        self.kinematics   = PepperKinematicsUtilities()
        self.gestures_data = self.load_gestures_yaml()

        self.home_positions = {
            'RArm': [1.7410, -0.09664, 1.6981, 0.09664, -0.05679],
            'LArm': [1.7625,  0.09970, -1.7150, -0.1334, 0.06592],
            'Head': [-0.2, 0.0],
            'Leg':  [0.0, 0.0, 0.0],
        }

    def load_gestures_yaml(self) -> Dict:
        gesture_path = self.package_path / 'data' / 'gesture.yaml'
        try:
            with open(gesture_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f'Warning: could not load gestures {gesture_path}: {e}')
            return {}

    def get_gesture_by_name(self, gesture_name: str) -> Optional[Dict]:
        if 'gestures' not in self.gestures_data:
            return None
        gesture_data = self.gestures_data['gestures'].get(gesture_name)
        if gesture_data is not None:
            return {'name': gesture_name, 'data': gesture_data}
        return None

    def get_gesture_duration(self, gesture_name: str) -> float:
        gesture_info = self.get_gesture_by_name(gesture_name)
        if not gesture_info:
            return 0.0
        times_dict   = gesture_info['data'].get('times', {})
        max_duration = 0.0
        for arm_times in times_dict.values():
            if arm_times:
                max_duration = max(max_duration, arm_times[-1])
        return max_duration


# ── Main lifecycle node ─────────────────────────────────────────────────────────

class GestureExecutionSystem(LifecycleNode):
    """ROS2 Action Server for gesture execution with elapsed time feedback."""

    def __init__(self):
        super().__init__('gesture_action_server')

    # ── Lifecycle callbacks ───────────────────────────────────────────────────

    def on_configure(self, _state) -> TransitionCallbackReturn:
        """Load configs, init kinematics, create publishers + action server."""
        try:
            package_path = get_package_share_directory('gesture_execution')
            self.config_manager     = ConfigManager(package_path)
            self.descriptor_manager = GestureDescriptorManager(
                Path(package_path), self.config_manager.config
            )
            self.joint_limits = JointLimits()
            self.kinematics   = PepperKinematicsUtilities()
        except Exception as e:
            self.get_logger().error(f'Configuration failed: {e}')
            return TransitionCallbackReturn.FAILURE

        self.robot_pose   = RobotPose()
        self.joint_states: Dict[str, float] = {}
        self.verbose_mode = self.config_manager.config.get('verboseMode', False)
        self._executing   = False
        self._feedback_stop = threading.Event()
        self._lifecycle_active = False

        self._init_joint_states()

        topics = self.config_manager.topics

        # Managed publishers
        self.joint_traj_pub = self.create_lifecycle_publisher(
            JointAnglesTrajectory, '/joint_angles_trajectory', 10
        )
        self.marker_pub = self.create_lifecycle_publisher(
            Marker, '/gesture_execution/visualization', 10
        )

        # Action server — goal_callback guards inactive state
        self._action_server = ActionServer(
            self,
            Gesture,
            '/gesture_execution/execute',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info('GestureExecutionSystem configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, _state) -> TransitionCallbackReturn:
        """Activate publishers, then subscribe to joint_states and robot_pose."""
        super().on_activate(_state)
        self._lifecycle_active = True

        topics = self.config_manager.topics
        self._joint_sub = self.create_subscription(
            JointState, topics['JointStates'], self.joint_states_callback, 10
        )
        self._pose_sub = self.create_subscription(
            Pose2D, topics['RobotPose'], self.robot_pose_callback, 10
        )

        self.get_logger().info('GestureExecutionSystem activated — ready for goals on /gesture_execution/execute')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, _state) -> TransitionCallbackReturn:
        """Destroy subscriptions, then deactivate publishers."""
        self._lifecycle_active = False
        self.destroy_subscription(self._joint_sub)
        self.destroy_subscription(self._pose_sub)
        super().on_deactivate(_state)
        self.get_logger().info('GestureExecutionSystem deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, _state) -> TransitionCallbackReturn:
        self.destroy_lifecycle_publisher(self.joint_traj_pub)
        self.destroy_lifecycle_publisher(self.marker_pub)
        self.get_logger().info('GestureExecutionSystem cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state) -> TransitionCallbackReturn:
        self.get_logger().info('GestureExecutionSystem shutting down')
        return TransitionCallbackReturn.SUCCESS

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_joint_states(self):
        hp = self.descriptor_manager.home_positions
        self.joint_states = {
            'HeadPitch':       self.joint_limits.DEFAULT_HEAD_PITCH,
            'HeadYaw':         self.joint_limits.DEFAULT_HEAD_YAW,
            'RShoulderPitch':  hp['RArm'][0], 'RShoulderRoll': hp['RArm'][1],
            'RElbowYaw':       hp['RArm'][2], 'RElbowRoll':    hp['RArm'][3],
            'RWristYaw':       hp['RArm'][4],
            'LShoulderPitch':  hp['LArm'][0], 'LShoulderRoll': hp['LArm'][1],
            'LElbowYaw':       hp['LArm'][2], 'LElbowRoll':    hp['LArm'][3],
            'LWristYaw':       hp['LArm'][4],
            'RHand': 0.67, 'LHand': 0.67,
            'HipPitch': hp['Leg'][0], 'HipRoll': hp['Leg'][1], 'KneePitch': hp['Leg'][2],
        }

    # ── Subscription callbacks ─────────────────────────────────────────────────

    def joint_states_callback(self, msg: JointState):
        for name, position in zip(msg.name, msg.position):
            if name in self.joint_states:
                self.joint_states[name] = position

    def robot_pose_callback(self, msg: Pose2D):
        self.robot_pose.x     = msg.x
        self.robot_pose.y     = msg.y
        self.robot_pose.theta = msg.theta

    # ── Action server callbacks ────────────────────────────────────────────────

    def goal_callback(self, goal_request):
        if not self._lifecycle_active:
            self.get_logger().warn('Rejecting goal — node is not active')
            return GoalResponse.REJECT
        if self._executing:
            self.get_logger().warn('Gesture in progress — rejecting new goal')
            return GoalResponse.REJECT
        self.get_logger().info(
            f"Accepting goal: type='{goal_request.gesture_type}' "
            f"name='{goal_request.gesture_name}'"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle):
        self.get_logger().info('Cancel requested (NAOqi motion cannot be stopped mid-way)')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self._executing = True
        self._feedback_stop.clear()
        request = goal_handle.request
        result  = Gesture.Result()

        try:
            gesture_type = request.gesture_type.strip().lower()
            start_time   = time.time()

            if self.verbose_mode:
                self.get_logger().info(
                    f"Executing gesture — type='{gesture_type}', "
                    f"name='{request.gesture_name}', duration={request.gesture_duration}ms"
                )

            feedback_thread = threading.Thread(
                target=self.publish_elapsed_feedback,
                args=(goal_handle, start_time),
            )
            feedback_thread.start()

            success = self.execute_gesture(
                gesture_type=gesture_type,
                gesture_name=request.gesture_name,
                gesture_duration=request.gesture_duration,
                bow_nod_angle=request.bow_nod_angle,
                location_x=request.location_x,
                location_y=request.location_y,
                location_z=request.location_z,
            )

            self._feedback_stop.set()
            feedback_thread.join(timeout=1.0)

            result.success               = success
            result.actual_duration_seconds = time.time() - start_time
            result.message               = 'completed' if success else 'failed'

            if success:
                self.get_logger().info(f'Gesture completed in {result.actual_duration_seconds:.2f}s')
                goal_handle.succeed()
            else:
                self.get_logger().error('Gesture execution failed')
                goal_handle.abort()

        except Exception as e:
            self.get_logger().error(f'Gesture failed: {e}')
            self.get_logger().error(traceback.format_exc())
            self._feedback_stop.set()
            result.success = False
            result.message = str(e)
            goal_handle.abort()
        finally:
            self._executing = False

        return result

    def publish_elapsed_feedback(self, goal_handle, start_time: float):
        feedback = Gesture.Feedback()
        while not self._feedback_stop.is_set():
            feedback.elapsed_seconds = time.time() - start_time
            try:
                goal_handle.publish_feedback(feedback)
            except Exception:
                break
            time.sleep(0.1)

    # ── Gesture execution ──────────────────────────────────────────────────────

    def execute_gesture(self, gesture_type, gesture_name, gesture_duration,
                        bow_nod_angle, location_x, location_y, location_z) -> bool:
        gesture_duration = max(MIN_GESTURE_DURATION, min(gesture_duration, MAX_GESTURE_DURATION))

        if gesture_type in ['deictic']:
            return self.execute_deictic_gesture(location_x, location_y, location_z, gesture_duration)
        elif gesture_type == 'iconic':
            return self.execute_iconic_gesture(gesture_name, gesture_duration)
        elif gesture_type == 'symbolic':
            self.get_logger().warning('Symbolic gestures not implemented yet')
            return False
        elif gesture_type in ['bow']:
            return self.execute_bowing_gesture(bow_nod_angle, gesture_duration)
        elif gesture_type in ['nod']:
            return self.execute_nodding_gesture(bow_nod_angle, gesture_duration)
        else:
            return self.execute_iconic_gesture(gesture_type, gesture_duration)

    def execute_deictic_gesture(self, point_x, point_y, point_z, duration) -> bool:
        try:
            robot_x     = self.robot_pose.x * 1000.0
            robot_y     = self.robot_pose.y * 1000.0
            robot_theta = self.robot_pose.theta

            relative_x  = (point_x * 1000) - robot_x
            relative_y  = (point_y * 1000) - robot_y
            pointing_x  = relative_x * math.cos(-robot_theta) - relative_y * math.sin(-robot_theta)
            pointing_y  = relative_y * math.cos(-robot_theta) + relative_x * math.sin(-robot_theta)
            pointing_z  = point_z * 1000

            if self.verbose_mode:
                self.get_logger().info(
                    f'Pointing coordinates: ({pointing_x:.1f}, {pointing_y:.1f}, {pointing_z:.1f})'
                )

            if pointing_x < 0.0:
                self.get_logger().error(f'Target behind robot: x={pointing_x:.1f}mm')
                return False

            pointing_arm = RIGHT_ARM if pointing_y <= 0.0 else LEFT_ARM
            shoulder_x   = SHOULDER_OFFSET_X
            shoulder_y   = SHOULDER_OFFSET_Y if pointing_arm == LEFT_ARM else -SHOULDER_OFFSET_Y
            shoulder_z   = SHOULDER_OFFSET_Z + TORSO_HEIGHT

            distance = math.sqrt(
                (pointing_x - shoulder_x)**2 +
                (pointing_y - shoulder_y)**2 +
                (pointing_z - shoulder_z)**2
            )
            l_2 = distance - UPPER_ARM_LENGTH

            elbow_x = ((UPPER_ARM_LENGTH * pointing_x) + (l_2 * shoulder_x)) / distance
            elbow_y = ((UPPER_ARM_LENGTH * pointing_y) + (l_2 * shoulder_y)) / distance
            elbow_z = ((UPPER_ARM_LENGTH * pointing_z) + (l_2 * shoulder_z)) / distance
            elbow_z -= TORSO_HEIGHT

            shoulder_pitch, shoulder_roll = self.kinematics.get_arm_shoulder_angles(
                pointing_arm, elbow_x, elbow_y, elbow_z
            )

            if math.isnan(shoulder_pitch) or math.isnan(shoulder_roll):
                self.get_logger().error('Invalid joint angles (unreachable target)')
                return False

            if not self.check_joint_limits(pointing_arm, shoulder_pitch, shoulder_roll):
                return False

            self.publish_deictic_visualization(
                pointing_x, pointing_y, pointing_z,
                shoulder_x, shoulder_y, shoulder_z,
                pointing_arm,
            )

            return self.execute_pointing_motion(
                pointing_arm, shoulder_pitch, shoulder_roll, duration,
                pointing_x, pointing_y, pointing_z,
            )

        except Exception as e:
            self.get_logger().error(f'Deictic gesture failed: {e}')
            self.get_logger().error(traceback.format_exc())
            return False

    def check_joint_limits(self, arm, shoulder_pitch, shoulder_roll) -> bool:
        if arm == RIGHT_ARM:
            if (shoulder_pitch < MIN_RSHOULDER_PITCH or shoulder_pitch > MAX_RSHOULDER_PITCH or
                    shoulder_roll < MIN_RSHOULDER_ROLL or shoulder_roll > MAX_RSHOULDER_ROLL):
                self.get_logger().error(
                    f'Right arm limits exceeded: pitch={math.degrees(shoulder_pitch):.1f}°, '
                    f'roll={math.degrees(shoulder_roll):.1f}°'
                )
                return False
        else:
            if (shoulder_pitch < MIN_LSHOULDER_PITCH or shoulder_pitch > MAX_LSHOULDER_PITCH or
                    shoulder_roll < MIN_LSHOULDER_ROLL or shoulder_roll > MAX_LSHOULDER_ROLL):
                self.get_logger().error(
                    f'Left arm limits exceeded: pitch={math.degrees(shoulder_pitch):.1f}°, '
                    f'roll={math.degrees(shoulder_roll):.1f}°'
                )
                return False
        return True

    def execute_pointing_motion(self, arm, shoulder_pitch, shoulder_roll, duration,
                                pointing_x, pointing_y, pointing_z) -> bool:
        try:
            duration_sec = duration / 1000.0
            head_pitch, head_yaw = self.calculate_head_angles_to_target(pointing_x, pointing_y, pointing_z)

            if arm == RIGHT_ARM:
                arm_joint_names  = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
                pointing_angles  = [shoulder_pitch, shoulder_roll, 2.0857, 0.0, -1.0]
                home_position    = self.descriptor_manager.home_positions['RArm']
                hand_joint_name  = 'RHand'
            else:
                arm_joint_names  = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw']
                pointing_angles  = [shoulder_pitch, shoulder_roll, -1.5620, -0.0, -1.0]
                home_position    = self.descriptor_manager.home_positions['LArm']
                hand_joint_name  = 'LHand'

            head_joint_names   = ['HeadPitch', 'HeadYaw']
            head_home          = self.descriptor_manager.home_positions['Head']
            head_pointing      = [head_pitch, head_yaw]
            hand_home_position = [0.67]
            hand_open_position = [1.0]

            joint_names    = arm_joint_names + head_joint_names + [hand_joint_name]
            home_waypoint  = home_position    + head_home   + hand_home_position
            point_waypoint = pointing_angles  + head_pointing + hand_open_position
            return_waypoint = home_position   + head_home   + hand_home_position

            self.move_joints_bezier(joint_names, [home_waypoint, point_waypoint, return_waypoint],
                                    duration_sec * 2, use_bezier=True)
            time.sleep(duration_sec * 2)
            return True

        except Exception as e:
            self.get_logger().error(f'Pointing motion failed: {e}')
            self.get_logger().error(traceback.format_exc())
            return False

    def calculate_head_angles_to_target(self, target_x, target_y, target_z) -> Tuple[float, float]:
        distance_xy = math.sqrt(target_x**2 + target_y**2)
        head_yaw    = math.atan2(target_y, target_x)
        HEAD_HEIGHT = 300.0
        head_pitch  = -math.atan2(target_z - HEAD_HEIGHT, distance_xy)
        head_yaw    = max(self.joint_limits.HEAD_YAW_RANGE[0],   min(head_yaw,   self.joint_limits.HEAD_YAW_RANGE[1]))
        head_pitch  = max(self.joint_limits.HEAD_PITCH_RANGE[0], min(head_pitch, self.joint_limits.HEAD_PITCH_RANGE[1]))
        return head_pitch, head_yaw

    def execute_iconic_gesture(self, gesture_name, duration) -> bool:
        try:
            gesture_info = self.descriptor_manager.get_gesture_by_name(gesture_name)
            if not gesture_info:
                self.get_logger().error(f"Gesture '{gesture_name}' not found")
                return False

            gesture_data   = gesture_info['data']
            arms_to_execute = gesture_data.get('joints', [])
            joint_angles_dict = gesture_data.get('joint_angles', {})
            times_dict        = gesture_data.get('times', {})

            if not joint_angles_dict or not arms_to_execute:
                self.get_logger().error('Invalid gesture data')
                return False

            all_joint_names  = []
            all_joint_angles = []
            all_times        = []
            max_duration     = 0.0

            for arm_name in arms_to_execute:
                if arm_name not in joint_angles_dict:
                    continue
                joint_angles_list = joint_angles_dict[arm_name]
                if arm_name == 'RArm':
                    joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
                elif arm_name == 'LArm':
                    joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw']
                elif arm_name == 'Leg':
                    joint_names = ['HipPitch', 'HipRoll', 'KneePitch']
                else:
                    continue

                arm_times = times_dict.get(arm_name, None)
                if arm_times:
                    arm_duration = arm_times[-1]
                else:
                    arm_duration  = duration / 1000.0
                    num_waypoints = len(joint_angles_list)
                    arm_times     = [arm_duration * (i + 1) / num_waypoints for i in range(num_waypoints)]

                max_duration = max(max_duration, arm_duration)
                all_joint_names.extend(joint_names)

                num_joints    = len(joint_names)
                num_waypoints = len(joint_angles_list)
                for joint_idx in range(num_joints):
                    for wp_idx in range(num_waypoints):
                        all_joint_angles.append(float(joint_angles_list[wp_idx][joint_idx]))
                        all_times.append(float(arm_times[wp_idx]))

            if not all_joint_names:
                self.get_logger().error('No valid joints to execute')
                return False

            self.move_joints_bezier(all_joint_names, all_joint_angles, all_times, use_bezier=True)
            time.sleep(max_duration)
            return True

        except Exception as e:
            self.get_logger().error(f'Iconic gesture failed: {e}')
            self.get_logger().error(traceback.format_exc())
            return False

    def execute_bowing_gesture(self, bow_angle, duration) -> bool:
        try:
            duration_sec  = duration / 1000.0
            bow_angle     = max(MIN_BOW_ANGLE, min(bow_angle, MAX_BOW_ANGLE))
            bow_angle_rad = -math.radians(bow_angle)
            joint_names   = ['HipPitch', 'HipRoll', 'KneePitch']
            home_position = self.descriptor_manager.home_positions['Leg']
            bow_position  = [bow_angle_rad, -0.00766, 0.03221]
            self.move_joints_bezier(joint_names, [home_position, bow_position, home_position],
                                    duration_sec * 2, use_bezier=True)
            time.sleep(duration_sec * 2)
            return True
        except Exception as e:
            self.get_logger().error(f'Bowing gesture failed: {e}')
            self.get_logger().error(traceback.format_exc())
            return False

    def execute_nodding_gesture(self, nod_angle, duration) -> bool:
        try:
            duration_sec  = duration / 1000.0
            nod_angle     = max(MIN_NOD_ANGLE, min(nod_angle, MAX_NOD_ANGLE))
            nod_angle_rad = math.radians(nod_angle)
            joint_names   = ['HeadPitch', 'HeadYaw']
            home_position = self.descriptor_manager.home_positions['Head']
            nod_position  = [nod_angle_rad, 0.012271]
            self.move_joints_bezier(joint_names, [home_position, nod_position, home_position],
                                    duration_sec * 2, use_bezier=True)
            time.sleep(duration_sec * 2)
            return True
        except Exception as e:
            self.get_logger().error(f'Nodding gesture failed: {e}')
            self.get_logger().error(traceback.format_exc())
            return False

    def move_joints_bezier(self, joint_names: List[str],
                           joint_angles: Union[List[List[float]], List[float]],
                           times: Union[List[float], float],
                           use_bezier: bool = True):
        try:
            num_joints = len(joint_names)

            if isinstance(joint_angles, list) and len(joint_angles) > 0:
                if isinstance(joint_angles[0], (list, tuple)):
                    num_waypoints = len(joint_angles)
                    if not all(len(wp) == num_joints for wp in joint_angles):
                        self.get_logger().error('Waypoint dimension mismatch')
                        return

                    if isinstance(times, (int, float)):
                        times_list = [float(times * (i + 1) / num_waypoints) for i in range(num_waypoints)]
                    elif isinstance(times, list):
                        times_list = times
                        if len(times_list) != num_waypoints:
                            self.get_logger().error('Times/waypoints mismatch')
                            return
                    else:
                        self.get_logger().error('Invalid times type')
                        return

                    msg_joint_angles = []
                    msg_times        = []
                    for joint_idx in range(num_joints):
                        for wp_idx in range(num_waypoints):
                            msg_joint_angles.append(float(joint_angles[wp_idx][joint_idx]))
                            msg_times.append(float(times_list[wp_idx]))
                else:
                    msg_joint_angles = [float(a) for a in joint_angles]
                    if isinstance(times, list):
                        msg_times = [float(t) for t in times]
                    else:
                        self.get_logger().error('Times must be list for flat format')
                        return
                    if len(msg_joint_angles) != len(msg_times):
                        self.get_logger().error('Angles/times length mismatch')
                        return
            else:
                self.get_logger().error('No joint angles provided')
                return

            msg                 = JointAnglesTrajectory()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.joint_names     = joint_names
            msg.relative        = 0
            msg.use_bezier      = use_bezier
            msg.joint_angles    = msg_joint_angles
            msg.times           = msg_times
            self.joint_traj_pub.publish(msg)

            if self.verbose_mode:
                num_wp     = len(msg_joint_angles) // num_joints if num_joints > 0 else 0
                total_time = msg_times[-1] if msg_times else 0.0
                self.get_logger().info(
                    f'Trajectory: {num_joints} joints, {num_wp} waypoints over {total_time:.2f}s'
                )

        except Exception as e:
            self.get_logger().error(f'Trajectory failed: {e}')
            self.get_logger().error(traceback.format_exc())

    def publish_deictic_visualization(self, target_x, target_y, target_z,
                                      shoulder_x, shoulder_y, shoulder_z, arm):
        try:
            stamp = self.get_clock().now().to_msg()

            def make_marker(ns, mid, mtype):
                m = Marker()
                m.header.stamp    = stamp
                m.header.frame_id = 'base_link'
                m.ns              = ns
                m.id              = mid
                m.type            = mtype
                m.action          = Marker.ADD
                m.pose.orientation.w = 1.0
                m.lifetime.sec    = 10
                m.color.a         = 1.0
                return m

            # Target sphere
            tm = make_marker('deictic_target', 0, Marker.SPHERE)
            tm.pose.position.x = target_x / 1000.0
            tm.pose.position.y = target_y / 1000.0
            tm.pose.position.z = target_z / 1000.0
            tm.scale.x = tm.scale.y = tm.scale.z = 0.1
            tm.color.r = 1.0
            self.marker_pub.publish(tm)

            # Shoulder sphere
            sm = make_marker('deictic_shoulder', 1, Marker.SPHERE)
            sm.pose.position.x = shoulder_x / 1000.0
            sm.pose.position.y = shoulder_y / 1000.0
            sm.pose.position.z = shoulder_z / 1000.0
            sm.scale.x = sm.scale.y = sm.scale.z = 0.06
            if arm == LEFT_ARM:
                sm.color.g = 1.0
            else:
                sm.color.b = 1.0
            self.marker_pub.publish(sm)

            # Arrow
            lm = make_marker('deictic_line', 2, Marker.ARROW)
            lm.color.a    = 0.8
            lm.scale.x    = 0.03
            lm.scale.y    = 0.06
            lm.scale.z    = 0.12
            start         = Point(x=shoulder_x / 1000.0, y=shoulder_y / 1000.0, z=shoulder_z / 1000.0)
            end           = Point(x=target_x  / 1000.0, y=target_y  / 1000.0, z=target_z  / 1000.0)
            lm.points     = [start, end]
            if arm == LEFT_ARM:
                lm.color.g = 1.0
            else:
                lm.color.b = 1.0
            self.marker_pub.publish(lm)

        except Exception as e:
            self.get_logger().error(f'Visualization failed: {e}')
