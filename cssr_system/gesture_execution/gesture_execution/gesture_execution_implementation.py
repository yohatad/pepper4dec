#!/usr/bin/env python3
"""
gesture_execution.py

ROS2 Python implementation of Pepper robot gesture execution system
Simplified version using direct joint control with JointAnglesWithSpeed messages

Author: Yohannes Haile
Date: October 11, 2025
Version: v1.0
"""

import math
import time
import yaml
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# ROS2 messages and services
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose2D, Twist, Point, Vector3
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from naoqi_bridge_msgs.msg import JointAnglesTrajectory
from cssr_interfaces.srv import PerformGesture
from .pepper_kinematics_utilities import PepperKinematicsUtilities, RIGHT_ARM, LEFT_ARM

# Import builtin_interfaces for time
from builtin_interfaces.msg import Time

# Constants
MIN_GESTURE_DURATION = 1000  # milliseconds
MAX_GESTURE_DURATION = 10000  # milliseconds

# Gesture types
DEICTIC_GESTURES    = "deictic"
ICONIC_GESTURES     = "iconic"
SYMBOLIC_GESTURES   = "symbolic"
BOWING_GESTURE      = "bow"
NODDING_GESTURE     = "nod"

# Robot physical constants (in millimeters)
UPPER_ARM_LENGTH = 150.0
SHOULDER_OFFSET_X = -57.0
SHOULDER_OFFSET_Y = 149.74  # Will be negated for right arm
SHOULDER_OFFSET_Z = 86.82
TORSO_HEIGHT = 0.0  # Adjust based on your robot setup

# Joint limits (radians)
MIN_RSHOULDER_PITCH = -2.0857
MAX_RSHOULDER_PITCH = 2.0857
MIN_RSHOULDER_ROLL = -1.5621
MAX_RSHOULDER_ROLL = -0.0087
MIN_LSHOULDER_PITCH = -2.0857
MAX_LSHOULDER_PITCH = 2.0857
MIN_LSHOULDER_ROLL = 0.0087
MAX_LSHOULDER_ROLL = 1.5621

# Bow/Nod angle limits (degrees)
MIN_BOW_ANGLE = 5
MAX_BOW_ANGLE = 45
MIN_NOD_ANGLE = 5
MAX_NOD_ANGLE = 30

@dataclass
class JointLimits:
    """Robot joint limits and constraints"""
    HEAD_YAW_RANGE: Tuple[float, float] = (-2.0857, 2.0857)
    HEAD_PITCH_RANGE: Tuple[float, float] = (-0.7068, 0.6371)
    DEFAULT_HEAD_PITCH: float = -0.2
    DEFAULT_HEAD_YAW: float = 0.0

@dataclass
class RobotPose:
    """Robot pose in the environment"""
    x: float = 0.0  # meters
    y: float = 0.0  # meters
    theta: float = 0.0  # radians

class ConfigManager:
    """Handles configuration and topic loading using actual YAML format"""
    
    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        self.config = self.load_config()
        self.topics = self.load_topics()
    
    def load_config(self) -> Dict:
        """Load configuration from gesture_execution_configuration.yaml"""
        config_path = self.package_path / "config" / "gesture_execution_configuration.yaml"
        
        # Default values
        config = {
            "gestureDescriptors": "gesture.yaml",
            "robotTopics": "pepperTopics.yaml", 
            "verboseMode": True  # Changed to True for debugging
        }
        
        try:
            if config_path.exists():
                with open(config_path, "r") as f:
                    yaml_config = yaml.safe_load(f) or {}
                    config.update(yaml_config)
        except Exception as e:
            print(f"Warning: Could not load config {config_path}: {e}")
        
        return config
    
    def load_topics(self) -> Dict:
        """Load topic mappings from pepper_topics.yaml"""
        topics_path = self.package_path / "data" / "pepper_topics.yaml"
        
        # Default topics
        topics = {
            "JointAngles": "/joint_angles",
            "Wheels": "/cmd_vel",
            "JointStates": "/joint_states",
            "RobotPose": "/robotLocalization/pose"
        }
        
        try:
            if topics_path.exists():
                with open(topics_path, "r") as f:
                    yaml_data = yaml.safe_load(f) or {}
                    if "topics" in yaml_data:
                        topics.update(yaml_data["topics"])
        except Exception as e:
            print(f"Warning: Could not load topics {topics_path}: {e}")
        
        return topics

class GestureDescriptorManager:
    """Manages gesture descriptor files using YAML format"""
    def __init__(self, package_path: Path, config: Dict):
        self.package_path = package_path
        self.config = config
        self.kinematics = PepperKinematicsUtilities()
        
        # Load gesture data from YAML
        self.gestures_data = self.load_gestures_yaml()
        
        # Home positions for each actuator
        # Joint order - ShoulderPitch, ShoulderRoll, ElbowYaw, ElbowRoll, WristYaw
        self.home_positions = {
            'RArm': [1.7410, -0.09664, 1.6981, 0.09664, -0.05679],
            'LArm': [1.7625, 0.09970, -1.7150, -0.1334, 0.06592],
            'Head': [-0.2, 0.0],
            'Leg' : [0.0, 0.0, 0.0]
        }
    
    def load_gestures_yaml(self) -> Dict:
        """Load gesture data from gesture.yaml"""
        gesture_path = self.package_path / "data" / "gesture.yaml"
        
        try:
            with open(gesture_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load gestures {gesture_path}: {e}")
            return {}
    
    def get_gesture_by_id(self, gesture_id: int) -> Optional[Dict]:
        """Get gesture data by ID"""
        if "gestures" not in self.gestures_data:
            return None
            
        for gesture_name, gesture_data in self.gestures_data["gestures"].items():
            if gesture_data.get("id") == gesture_id:
                return {
                    "name": gesture_name,
                    "data": gesture_data
                }
        return None

class GestureExecutionSystem(Node):
    """Main ROS2 node for gesture execution"""  
    def __init__(self):
        super().__init__("gesture_execution")
        
        # Initialize components
        package_path = get_package_share_directory("gesture_execution") 
        self.config_manager = ConfigManager(package_path)
        self.descriptor_manager = GestureDescriptorManager(Path(package_path), self.config_manager.config)
        self.joint_limits = JointLimits()
        self.kinematics = PepperKinematicsUtilities()
        
        # State
        self.robot_pose = RobotPose()
        self.joint_states = {}
        self.verbose_mode = self.config_manager.config.get("verboseMode", False)
        
        # Initialize joint state storage
        self.init_joint_states()
        
        self.setup_ros_interfaces()
        self.get_logger().info("Gesture Execution System started - waiting for service calls")
    
    def init_joint_states(self):
        """Initialize joint state storage"""
        # FIXED: Consistent joint order matching home_positions
        # Pepper hand joints: RHand and LHand range from 0.0 (closed) to 1.0 (open)
        # Home position is typically 0.67 (partially open)
        self.joint_states = {
            'HeadPitch': self.joint_limits.DEFAULT_HEAD_PITCH,
            'HeadYaw': self.joint_limits.DEFAULT_HEAD_YAW,
            'RShoulderPitch': self.descriptor_manager.home_positions['RArm'][0],
            'RShoulderRoll': self.descriptor_manager.home_positions['RArm'][1], 
            'RElbowYaw': self.descriptor_manager.home_positions['RArm'][2],
            'RElbowRoll': self.descriptor_manager.home_positions['RArm'][3],
            'RWristYaw': self.descriptor_manager.home_positions['RArm'][4],
            'LShoulderPitch': self.descriptor_manager.home_positions['LArm'][0],
            'LShoulderRoll': self.descriptor_manager.home_positions['LArm'][1],
            'LElbowYaw': self.descriptor_manager.home_positions['LArm'][2],
            'LElbowRoll': self.descriptor_manager.home_positions['LArm'][3],
            'LWristYaw': self.descriptor_manager.home_positions['LArm'][4],
            'RHand': 0.67,  # Pepper hand home position (partially open)
            'LHand': 0.67,  # Pepper hand home position (partially open)
            'HipPitch': self.descriptor_manager.home_positions['Leg'][0],
            'HipRoll': self.descriptor_manager.home_positions['Leg'][1],
            'KneePitch': self.descriptor_manager.home_positions['Leg'][2]
        }
    
    def setup_ros_interfaces(self):
        """Initialize ROS publishers, subscribers, and services"""
        topics = self.config_manager.topics

        # Subscribers
        self.create_subscription(JointState, topics["JointStates"], self.joint_states_callback, 10)
        self.create_subscription(Pose2D, topics["RobotPose"], self.robot_pose_callback, 10)

        # Publishers - FIXED: Correct topic name
        self.joint_traj_pub = self.create_publisher(JointAnglesTrajectory, '/joint_angles_trajectory', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, topics["Wheels"], 10)
        
        # RViz2 visualization publishers
        self.marker_pub = self.create_publisher(Marker, "/gesture_execution/visualization", 10)

        # Services
        self.create_service(PerformGesture, "/gesture_execution/perform_gesture", self.perform_gesture_callback)
    
    def joint_states_callback(self, msg: JointState):
        """Handle joint state"""
        for name, position in zip(msg.name, msg.position):
            if name in self.joint_states:
                self.joint_states[name] = position
    
    def robot_pose_callback(self, msg: Pose2D):
        """Handle robot pose"""
        self.robot_pose.x = msg.x
        self.robot_pose.y = msg.y
        self.robot_pose.theta = msg.theta
    
    def perform_gesture_callback(self, request: PerformGesture.Request,
                             response: PerformGesture.Response):
        """Handle gesture execution requests"""
        try:
            success = self.execute_gesture(
                gesture_type=request.gesture_type,
                gesture_id=request.gesture_id,
                gesture_duration=request.speed,
                bow_nod_angle=request.bow_nod_angle,
                location_x=request.location_x,
                location_y=request.location_y,
                location_z=request.location_z
            )
            
            response.gesture_success = 1 if success else 0
            
            if success and self.verbose_mode:
                self.get_logger().info("Gesture executed successfully")
            elif not success:
                self.get_logger().error("Gesture execution failed")
                
        except Exception as e:
            self.get_logger().error(f"Gesture execution error: {e}")
            self.get_logger().error(traceback.format_exc())
            response.gesture_success = 0
        
        return response
    
    def execute_gesture(self, gesture_type: str, gesture_id: int, gesture_duration: int,
                   bow_nod_angle: int, location_x: float, location_y: float, location_z: float) -> bool:
        """Main gesture execution logic"""
        
        # FIXED: Normalize gesture type - strip whitespace and convert to lowercase
        gesture_type = gesture_type.strip().lower()
        
        if self.verbose_mode:
            self.get_logger().info(
                f"Gesture request - Type: '{gesture_type}', ID: {gesture_id}, "
                f"Duration: {gesture_duration}ms, Angle: {bow_nod_angle}°, "
                f"Location: ({location_x:.2f}, {location_y:.2f}, {location_z:.2f})"
            )
        
        # Clamp gesture duration to limits
        gesture_duration = max(MIN_GESTURE_DURATION, min(gesture_duration, MAX_GESTURE_DURATION))
        
        # Execute based on gesture type - Use normalized lowercase
        if gesture_type in ["deictic", "pointing"]:
            return self.execute_deictic_gesture(location_x, location_y, location_z, gesture_duration)
        
        elif gesture_type == "iconic":
            return self.execute_iconic_gesture(gesture_id, gesture_duration)
        
        elif gesture_type == "symbolic":
            self.get_logger().warning("Symbolic gestures not implemented yet")
            return False
        
        elif gesture_type in ["bow", "bowing"]:
            return self.execute_bowing_gesture(bow_nod_angle, gesture_duration)
        
        elif gesture_type in ["nod", "nodding"]:
            return self.execute_nodding_gesture(bow_nod_angle, gesture_duration)
        
        else:
            self.get_logger().warning(f"Unsupported gesture type: '{gesture_type}'")
            return False
    
    def execute_pointing_motion(self, arm: int, shoulder_pitch: float, shoulder_roll: float, 
                           duration: int, pointing_x: float, pointing_y: float, 
                           pointing_z: float) -> bool:
        """Execute the actual pointing motion with smooth Bezier interpolation including head movement"""
        try:
            duration_sec = duration / 1000.0
            
            # Calculate head angles to look at target
            head_pitch, head_yaw = self.calculate_head_angles_to_target(pointing_x, pointing_y, pointing_z)
            
            if self.verbose_mode:
                self.get_logger().info(
                    f"Head angles - Pitch: {math.degrees(head_pitch):.1f}°, "
                    f"Yaw: {math.degrees(head_yaw):.1f}°"
                )
            
            # FIXED: Consistent joint order - ShoulderPitch, ShoulderRoll, ElbowYaw, ElbowRoll, WristYaw
            if arm == RIGHT_ARM:
                arm_joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
                # For deictic gestures, open the hand by setting wrist yaw to 0 (straight hand)
                # Home wrist yaw is -0.05679, pointing wrist yaw should be 0 for open hand
                pointing_angles = [shoulder_pitch, shoulder_roll, 2.0857, 0.0, -1.0]
                home_position = self.descriptor_manager.home_positions['RArm']
                # Add right hand joint for opening/closing
                hand_joint_name = 'RHand'
            else:
                arm_joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw']
                # For deictic gestures, open the hand by setting wrist yaw to 0 (straight hand)
                # Home wrist yaw is 0.06592, pointing wrist yaw should be 0 for open hand
                pointing_angles = [shoulder_pitch, shoulder_roll, -1.5620, -0.0, -1.0]
                home_position = self.descriptor_manager.home_positions['LArm']
                # Add left hand joint for opening/closing
                hand_joint_name = 'LHand'
            
            # Add head joints
            head_joint_names = ['HeadPitch', 'HeadYaw']
            head_home = self.descriptor_manager.home_positions['Head']
            head_pointing = [head_pitch, head_yaw]
            
            # Hand positions: 0.0 = closed, 1.0 = fully open, 0.67 = home (partially open)
            # For pointing: hand should be fully open (1.0)
            hand_home_position = [0.67]  # Pepper hand home position (partially open)
            hand_open_position = [1.0]    # Fully open for pointing
            
            # Combine all joints: arm + head + hand
            joint_names = arm_joint_names + head_joint_names + [hand_joint_name]
            
            # Create synchronized trajectory: home -> pointing -> home
            # Home waypoint: arm home + head home + hand home (partially open)
            home_waypoint = home_position + head_home + hand_home_position
            # Pointing waypoint: pointing angles + head pointing + hand open
            pointing_waypoint = pointing_angles + head_pointing + hand_open_position
            # Return waypoint: arm home + head home + hand home (partially open)
            return_waypoint = home_position + head_home + hand_home_position
            
            waypoints = [home_waypoint, pointing_waypoint, return_waypoint]
            
            # Execute smooth pointing gesture with head tracking and hand opening
            self.move_joints_bezier(joint_names, waypoints, duration_sec * 2, use_bezier=True)
            time.sleep(duration_sec * 2)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Pointing motion failed: {e}")
            self.get_logger().error(traceback.format_exc())
            return False
    
    
    def execute_deictic_gesture(self, point_x: float, point_y: float, point_z: float, duration: int) -> bool:
        """Execute pointing gesture to specific location"""
        try:
            # Convert robot pose to millimeters and compute relative pointing coordinates
            robot_x = self.robot_pose.x * 1000.0
            robot_y = self.robot_pose.y * 1000.0
            robot_theta = self.robot_pose.theta
            
            # Transform to robot's local coordinate frame
            relative_x = (point_x * 1000) - robot_x
            relative_y = (point_y * 1000) - robot_y
            
            pointing_x = (relative_x * math.cos(-robot_theta)) - (relative_y * math.sin(-robot_theta))
            pointing_y = (relative_y * math.cos(-robot_theta)) + (relative_x * math.sin(-robot_theta))
            pointing_z = point_z * 1000
            
            if self.verbose_mode:
                self.get_logger().info(f"Pointing coordinates: ({pointing_x:.1f}, {pointing_y:.1f}, {pointing_z:.1f})")
            
            # Check if target is in front of robot
            if pointing_x < 0.0:
                self.get_logger().error(
                    f"Pointing target is out of bounds (behind robot): "
                    f"x={pointing_x:.1f}mm. Robot cannot point backwards."
                )
                return False
            
            # Determine which arm to use based on y coordinate
            pointing_arm = RIGHT_ARM if pointing_y <= 0.0 else LEFT_ARM
            
            # Calculate elbow position for inverse kinematics
            shoulder_x = SHOULDER_OFFSET_X
            shoulder_y = SHOULDER_OFFSET_Y if pointing_arm == LEFT_ARM else -SHOULDER_OFFSET_Y
            shoulder_z = SHOULDER_OFFSET_Z + TORSO_HEIGHT
            
            distance = math.sqrt((pointing_x - shoulder_x)**2 + (pointing_y - shoulder_y)**2 + (pointing_z - shoulder_z)**2)
            l_2 = distance - UPPER_ARM_LENGTH
            
            elbow_x = ((UPPER_ARM_LENGTH * pointing_x) + (l_2 * shoulder_x)) / distance
            elbow_y = ((UPPER_ARM_LENGTH * pointing_y) + (l_2 * shoulder_y)) / distance
            elbow_z = ((UPPER_ARM_LENGTH * pointing_z) + (l_2 * shoulder_z)) / distance
            elbow_z -= TORSO_HEIGHT  # Convert to joint coordinate frame
            
            # Calculate joint angles using inverse kinematics
            shoulder_pitch, shoulder_roll = self.kinematics.get_arm_shoulder_angles(
                pointing_arm, elbow_x, elbow_y, elbow_z
            )
            
            # Check for invalid angles
            if math.isnan(shoulder_pitch) or math.isnan(shoulder_roll):
                self.get_logger().error(
                    f"Pointing target is out of bounds (unreachable): "
                    f"Invalid joint angles calculated for target ({pointing_x:.1f}, {pointing_y:.1f}, {pointing_z:.1f})mm"
                )
                return False
            
            # Check joint limits
            if pointing_arm == RIGHT_ARM:
                if (shoulder_pitch < MIN_RSHOULDER_PITCH or shoulder_pitch > MAX_RSHOULDER_PITCH or
                    shoulder_roll < MIN_RSHOULDER_ROLL or shoulder_roll > MAX_RSHOULDER_ROLL):
                    self.get_logger().error(
                        f"Pointing target is out of bounds (joint limits exceeded): "
                        f"Right arm - Pitch: {math.degrees(shoulder_pitch):.1f}° "
                        f"[{math.degrees(MIN_RSHOULDER_PITCH):.1f}° to {math.degrees(MAX_RSHOULDER_PITCH):.1f}°], "
                        f"Roll: {math.degrees(shoulder_roll):.1f}° "
                        f"[{math.degrees(MIN_RSHOULDER_ROLL):.1f}° to {math.degrees(MAX_RSHOULDER_ROLL):.1f}°]"
                    )
                    return False
            else:
                if (shoulder_pitch < MIN_LSHOULDER_PITCH or shoulder_pitch > MAX_LSHOULDER_PITCH or
                    shoulder_roll < MIN_LSHOULDER_ROLL or shoulder_roll > MAX_LSHOULDER_ROLL):
                    self.get_logger().error(
                        f"Pointing target is out of bounds (joint limits exceeded): "
                        f"Left arm - Pitch: {math.degrees(shoulder_pitch):.1f}° "
                        f"[{math.degrees(MIN_LSHOULDER_PITCH):.1f}° to {math.degrees(MAX_LSHOULDER_PITCH):.1f}°], "
                        f"Roll: {math.degrees(shoulder_roll):.1f}° "
                        f"[{math.degrees(MIN_LSHOULDER_ROLL):.1f}° to {math.degrees(MAX_LSHOULDER_ROLL):.1f}°]"
                    )
                    return False
            
            # Publish visualization markers before executing the gesture
            self._publish_deictic_visualization(
                pointing_x, pointing_y, pointing_z,
                shoulder_x, shoulder_y, shoulder_z,
                pointing_arm
            )
            
            # Execute pointing motion with head tracking
            success = self.execute_pointing_motion(
                pointing_arm, shoulder_pitch, shoulder_roll, duration,
                pointing_x, pointing_y, pointing_z
            )
            
            return success
            
        except Exception as e:
            self.get_logger().error(f"Deictic gesture execution failed: {e}")
            self.get_logger().error(traceback.format_exc())
            return False
    
    def execute_iconic_gesture(self, gesture_id: int, duration: int) -> bool:
        """
        Execute iconic gesture from YAML configuration with independent arm timing
        
        Args:
            gesture_id: ID of the gesture to execute
            duration: Duration in milliseconds (fallback if no times specified)
        """
        try:
            gesture_info = self.descriptor_manager.get_gesture_by_id(gesture_id)
            if not gesture_info:
                self.get_logger().error(f"Gesture ID {gesture_id} not found")
                return False
            
            gesture_data = gesture_info["data"]
            arms_to_execute = gesture_data.get("joints", [])
            joint_angles_dict = gesture_data.get("joint_angles", {})
            times_dict = gesture_data.get("times", {})  # Dictionary per arm
            
            if not joint_angles_dict:
                self.get_logger().error("No joint angles found in gesture data")
                return False
            
            if not arms_to_execute:
                self.get_logger().error("No arms specified in gesture data")
                return False
            
            if self.verbose_mode:
                self.get_logger().info(
                    f"Executing gesture '{gesture_info['name']}' on arms: {arms_to_execute}"
                )
            
            # Collect all joint names and trajectories for simultaneous execution
            all_joint_names = []
            all_joint_angles = []
            all_times = []
            
            for arm_name in arms_to_execute:
                if arm_name not in joint_angles_dict:
                    self.get_logger().warning(f"No angles defined for {arm_name}")
                    continue
                
                joint_angles_list = joint_angles_dict[arm_name]
                
                # Get joint names for this arm
                if arm_name == "RArm":
                    joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
                elif arm_name == "LArm":
                    joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw']
                elif arm_name == "Leg":
                    joint_names = ['HipPitch', 'HipRoll', 'KneePitch']
                else:
                    self.get_logger().warning(f"Unknown arm: {arm_name}")
                    continue
                
                # Get timing for this specific arm
                arm_times = times_dict.get(arm_name, None)
                
                # Calculate duration for this arm
                if arm_times:
                    arm_duration = arm_times[-1]
                else:
                    arm_duration = duration / 1000.0
                    # Generate evenly spaced times
                    num_waypoints = len(joint_angles_list)
                    arm_times = [arm_duration * (i + 1) / num_waypoints for i in range(num_waypoints)]
                
                if self.verbose_mode:
                    self.get_logger().info(
                        f"  {arm_name}: {len(joint_angles_list)} waypoints, "
                        f"duration: {arm_duration:.2f}s"
                    )
                
                # Add this arm's data to the combined lists
                all_joint_names.extend(joint_names)
                
                # Flatten the waypoints for this arm: [j1_wp1, j1_wp2, ..., j2_wp1, j2_wp2, ...]
                num_joints = len(joint_names)
                num_waypoints = len(joint_angles_list)
                
                for joint_idx in range(num_joints):
                    for waypoint_idx in range(num_waypoints):
                        all_joint_angles.append(float(joint_angles_list[waypoint_idx][joint_idx]))
                        all_times.append(float(arm_times[waypoint_idx]))
            
            if not all_joint_names:
                self.get_logger().error("No valid joints to execute")
                return False
            
            # Execute all arms simultaneously in one call
            self.move_joints_bezier(all_joint_names, all_joint_angles,all_times,use_bezier=True)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Iconic gesture execution failed: {e}")
            self.get_logger().error(traceback.format_exc())
            return False
    
    def execute_bowing_gesture(self, bow_angle: int, duration: int) -> bool:
        """Execute bowing gesture with smooth Bezier interpolation"""
        try:    
            duration_sec = duration / 1000.0
            
            # Clamp bow angle to limits
            bow_angle = max(MIN_BOW_ANGLE, min(bow_angle, MAX_BOW_ANGLE))
            bow_angle_rad = -math.radians(bow_angle)
            
            joint_names = ['HipPitch', 'HipRoll', 'KneePitch']
            home_position = self.descriptor_manager.home_positions['Leg']
            bow_position = [bow_angle_rad, -0.00766, 0.03221]
            
            # Smooth bow: home -> bow -> home
            waypoints = [home_position, bow_position, home_position]
            self.move_joints_bezier(joint_names, waypoints, duration_sec * 2, use_bezier=True)
            time.sleep(duration_sec * 2)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Bowing gesture failed: {e}")
            self.get_logger().error(traceback.format_exc())
            return False

    def calculate_head_angles_to_target(self, target_x: float, target_y: float, target_z: float) -> Tuple[float, float]:
        """
        Calculate head yaw and pitch angles to look at a target point
        
        Args:
            target_x, target_y, target_z: Target coordinates in robot's local frame (mm)
        
        Returns:
            Tuple of (head_pitch, head_yaw) in radians
        """
        # Calculate distance in XY plane
        distance_xy = math.sqrt(target_x**2 + target_y**2)
        
        # Calculate head yaw (rotation around Z axis)
        head_yaw = math.atan2(target_y, target_x)
        
        # Calculate head pitch (rotation around Y axis)
        # Adjust target_z for head height (approximately 300mm from base)
        HEAD_HEIGHT = 300.0  # mm - adjust based on Pepper's actual head height
        adjusted_z = target_z - HEAD_HEIGHT
        
        # FIX: Head pitch sign was reversed
        # For Pepper robot: 
        # - Positive pitch = looking down (head tilting forward)
        # - Negative pitch = looking up (head tilting back)
        # When target is above head (adjusted_z > 0), robot should look UP (negative pitch)
        # When target is below head (adjusted_z < 0), robot should look DOWN (positive pitch)
        head_pitch = -math.atan2(adjusted_z, distance_xy)
        
        # Clamp to joint limits
        head_yaw = max(self.joint_limits.HEAD_YAW_RANGE[0], 
                    min(head_yaw, self.joint_limits.HEAD_YAW_RANGE[1]))
        head_pitch = max(self.joint_limits.HEAD_PITCH_RANGE[0], 
                        min(head_pitch, self.joint_limits.HEAD_PITCH_RANGE[1]))
        
        return head_pitch, head_yaw
    
    def execute_nodding_gesture(self, nod_angle: int, duration: int) -> bool:
        """Execute nodding gesture with smooth Bezier interpolation"""
        try:
            duration_sec = duration / 1000.0
            
            # Clamp nod angle to limits
            nod_angle = max(MIN_NOD_ANGLE, min(nod_angle, MAX_NOD_ANGLE))
            nod_angle_rad = math.radians(nod_angle)
            
            joint_names = ['HeadPitch', 'HeadYaw']
            home_position = self.descriptor_manager.home_positions['Head']
            nod_position = [nod_angle_rad, 0.012271]
            
            # Smooth nod: home -> nod -> home
            waypoints = [home_position, nod_position, home_position]
            self.move_joints_bezier(joint_names, waypoints, duration_sec * 2, use_bezier=True)
            time.sleep(duration_sec * 2)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Nodding gesture failed: {e}")
            self.get_logger().error(traceback.format_exc())
            return False
    
    def move_joints_bezier(self, joint_names: List[str], 
                   joint_angles: Union[List[List[float]], List[float]], 
                   times: Union[List[float], float],
                   use_bezier: bool = True):
        """
        Move joints through multiple waypoints using trajectory control
        
        Args:
            joint_names: List of joint names
            joint_angles: Flat list of angles [j1_wp1, j1_wp2, ..., j2_wp1, ...] OR list of waypoints
            times: Flat list of times [t1, t2, ...] OR single duration
            use_bezier: If True, use Bezier interpolation for smooth motion
        """
        try:
            num_joints = len(joint_names)
            
            # Handle different input formats
            if isinstance(joint_angles, list) and len(joint_angles) > 0:
                # Check if it's a flat list or list of lists
                if isinstance(joint_angles[0], (list, tuple)):
                    # List of waypoints format: [[wp1], [wp2], ...]
                    num_waypoints = len(joint_angles)
                    
                    if not all(len(wp) == num_joints for wp in joint_angles):
                        self.get_logger().error(
                            f"Waypoint dimension mismatch: expected {num_joints} joints"
                        )
                        return
                    
                    # Convert to flat format for message
                    msg_joint_angles = []
                    msg_times = []
                    
                    # Handle times parameter
                    if isinstance(times, (int, float)):
                        # Generate evenly spaced times
                        times_list = [float(times * (i + 1) / num_waypoints) 
                                    for i in range(num_waypoints)]
                    elif isinstance(times, list):
                        times_list = times
                        if len(times_list) != num_waypoints:
                            self.get_logger().error(
                                f"Times list length ({len(times_list)}) doesn't match waypoints ({num_waypoints})"
                            )
                            return
                    else:
                        self.get_logger().error("Invalid times type")
                        return
                    
                    # NAOqi format: [j1_wp1, j1_wp2, ..., j2_wp1, j2_wp2, ...]
                    for joint_idx in range(num_joints):
                        for waypoint_idx in range(num_waypoints):
                            msg_joint_angles.append(float(joint_angles[waypoint_idx][joint_idx]))
                            msg_times.append(float(times_list[waypoint_idx]))
                else:
                    # Already flat format: [j1_wp1, j1_wp2, ..., j2_wp1, j2_wp2, ...]
                    msg_joint_angles = [float(a) for a in joint_angles]
                    
                    if isinstance(times, list):
                        msg_times = [float(t) for t in times]
                    else:
                        self.get_logger().error("Times must be a list for flat angle format")
                        return
                    
                    if len(msg_joint_angles) != len(msg_times):
                        self.get_logger().error(
                            f"Angles length ({len(msg_joint_angles)}) != times length ({len(msg_times)})"
                        )
                        return
            else:
                self.get_logger().error("No joint angles provided")
                return
            
            msg = JointAnglesTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = joint_names
            msg.relative = 0
            msg.use_bezier = use_bezier
            msg.joint_angles = msg_joint_angles
            msg.times = msg_times
            
            self.joint_traj_pub.publish(msg)
            
            if self.verbose_mode:
                num_waypoints = len(msg_joint_angles) // num_joints if num_joints > 0 else 0
                total_time = msg_times[-1] if msg_times else 0.0
                self.get_logger().info(
                    f"Published {'Bezier' if use_bezier else 'linear'} trajectory: "
                    f"{num_joints} joints, {num_waypoints} waypoints over {total_time:.2f}s"
                )
                
        except Exception as e:
            self.get_logger().error(f"Bezier trajectory failed: {e}")
            self.get_logger().error(traceback.format_exc())

    def _publish_deictic_visualization(self, 
                                    target_x: float, target_y: float, target_z: float,
                                    shoulder_x: float, shoulder_y: float, shoulder_z: float,
                                    arm: int):
        """
        Publish RViz2 markers for deictic gesture visualization
        
        Args:
            target_x, target_y, target_z: Target point coordinates in mm
            shoulder_x, shoulder_y, shoulder_z: Shoulder position in mm  
            arm: RIGHT_ARM or LEFT_ARM indicating which arm is pointing
        """
        try:
            stamp = self.get_clock().now().to_msg()
            
            # 1. Target point marker (sphere) - bright red for better visibility
            target_marker = Marker()
            target_marker.header.stamp = stamp
            target_marker.header.frame_id = "base_link"  # Robot base frame
            target_marker.ns = "deictic_target"
            target_marker.id = 0
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            target_marker.pose.position.x = target_x / 1000.0  # Convert mm to meters
            target_marker.pose.position.y = target_y / 1000.0
            target_marker.pose.position.z = target_z / 1000.0
            target_marker.pose.orientation.w = 1.0
            target_marker.scale.x = 0.1  # 10cm sphere (increased from 5cm)
            target_marker.scale.y = 0.1
            target_marker.scale.z = 0.1
            target_marker.color.r = 1.0  # Bright red for target
            target_marker.color.g = 0.0
            target_marker.color.b = 0.0
            target_marker.color.a = 1.0  # Fully opaque
            # Set lifetime properly with both sec and nanosec
            target_marker.lifetime.sec = 10  # 10 seconds (increased from 3)
            target_marker.lifetime.nanosec = 0
            
            self.marker_pub.publish(target_marker)
            
            # 2. Shoulder position marker (sphere) - bright colors for better visibility
            shoulder_marker = Marker()
            shoulder_marker.header.stamp = stamp
            shoulder_marker.header.frame_id = "base_link"
            shoulder_marker.ns = "deictic_shoulder"
            shoulder_marker.id = 1
            shoulder_marker.type = Marker.SPHERE
            shoulder_marker.action = Marker.ADD
            shoulder_marker.pose.position.x = shoulder_x / 1000.0
            shoulder_marker.pose.position.y = shoulder_y / 1000.0
            shoulder_marker.pose.position.z = shoulder_z / 1000.0
            shoulder_marker.pose.orientation.w = 1.0
            shoulder_marker.scale.x = 0.06  # 6cm sphere (increased from 3cm)
            shoulder_marker.scale.y = 0.06
            shoulder_marker.scale.z = 0.06
            # Bright colors based on arm (green for left, blue for right)
            if arm == LEFT_ARM:
                shoulder_marker.color.r = 0.0
                shoulder_marker.color.g = 1.0  # Bright green
                shoulder_marker.color.b = 0.0
            else:  # RIGHT_ARM
                shoulder_marker.color.r = 0.0
                shoulder_marker.color.g = 0.0
                shoulder_marker.color.b = 1.0  # Bright blue
            shoulder_marker.color.a = 1.0  # Fully opaque
            shoulder_marker.lifetime.sec = 10
            shoulder_marker.lifetime.nanosec = 0
            
            self.marker_pub.publish(shoulder_marker)
            
            # 3. Pointing line from shoulder to target (arrow) - thicker and brighter
            line_marker = Marker()
            line_marker.header.stamp = stamp
            line_marker.header.frame_id = "base_link"
            line_marker.ns = "deictic_line"
            line_marker.id = 2
            line_marker.type = Marker.ARROW
            line_marker.action = Marker.ADD
            
            start_point = Point()
            start_point.x = shoulder_x / 1000.0
            start_point.y = shoulder_y / 1000.0
            start_point.z = shoulder_z / 1000.0
            end_point = Point()
            end_point.x = target_x / 1000.0
            end_point.y = target_y / 1000.0
            end_point.z = target_z / 1000.0
            
            line_marker.points.append(start_point)
            line_marker.points.append(end_point)
            
            # Bright colors based on arm (green for left, blue for right)
            if arm == LEFT_ARM:
                line_marker.color.r = 0.0
                line_marker.color.g = 1.0  # Bright green
                line_marker.color.b = 0.0
            else:  # RIGHT_ARM
                line_marker.color.r = 0.0
                line_marker.color.g = 0.0
                line_marker.color.b = 1.0  # Bright blue
            line_marker.color.a = 0.8  # Slightly transparent for better visibility
            line_marker.scale.x = 0.03  # Shaft diameter (increased from 0.02)
            line_marker.scale.y = 0.06  # Head diameter (increased from 0.04)
            line_marker.scale.z = 0.12  # Head length (increased from 0.1)
            line_marker.lifetime.sec = 10
            line_marker.lifetime.nanosec = 0
            
            self.marker_pub.publish(line_marker)
            
            # 4. Text label showing coordinates - larger and brighter
            text_marker = Marker()
            text_marker.header.stamp = stamp
            text_marker.header.frame_id = "base_link"
            text_marker.ns = "deictic_text"
            text_marker.id = 3
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            # Position text slightly above target
            text_marker.pose.position.x = target_x / 1000.0
            text_marker.pose.position.y = target_y / 1000.0
            text_marker.pose.position.z = (target_z / 1000.0) + 0.15  # 15cm above target
            text_marker.pose.orientation.w = 1.0
            text_marker.text = f"Target: ({target_x/1000:.2f}, {target_y/1000:.2f}, {target_z/1000:.2f}) m"
            text_marker.scale.z = 0.07  # Text height (increased from 0.05)
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0  # White text for better visibility
            text_marker.color.a = 1.0  # Fully opaque
            text_marker.lifetime.sec = 10
            text_marker.lifetime.nanosec = 0
            
            self.marker_pub.publish(text_marker)
            
            if self.verbose_mode:
                arm_name = "left" if arm == LEFT_ARM else "right"
                self.get_logger().info(f"Published deictic visualization markers for {arm_name} arm pointing")
                self.get_logger().info(f"  Target: ({target_x/1000:.2f}, {target_y/1000:.2f}, {target_z/1000:.2f}) m")
                self.get_logger().info(f"  Shoulder: ({shoulder_x/1000:.2f}, {shoulder_y/1000:.2f}, {shoulder_z/1000:.2f}) m")
                self.get_logger().info(f"  Markers will be visible for 10 seconds")
                
        except Exception as e:
            self.get_logger().error(f"Failed to publish deictic visualization: {e}")
            self.get_logger().error(traceback.format_exc())
