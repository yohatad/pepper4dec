#!/usr/bin/env python3
"""
gesture_execution_ros2.py

ROS2 Python implementation of Pepper robot gesture execution system
Simplified version using direct joint control with JointAnglesWithSpeed messages

Author: Converted from C++ (Adedayo Akinade)
Date: September 24, 2025
Version: v2.0
"""

import os
import math
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# ROS2 messages and services
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose2D, Twist
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from cssr_interfaces.srv import PerformGesture
from pepper_kinematics_utilities import PepperKinematicsUtilities, RIGHT_ARM, LEFT_ARM

# Constants
MIN_GESTURE_DURATION = 1000  # milliseconds
MAX_GESTURE_DURATION = 10000  # milliseconds

# Gesture types
DIECTIC_GESTURES = "diectic"
ICONIC_GESTURES = "iconic"
SYMBOLIC_GESTURES = "symbolic"
BOWING_GESTURE = "bow"
NODDING_GESTURE = "nod"

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
            "gestureDescriptors": "gestureDescriptors.dat",
            "robotTopics": "pepperTopics.dat", 
            "verboseMode": False
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
        self.home_positions = {
            'RArm': [1.7410, -0.09664, 0.09664, 1.6981, -0.05679],
            'LArm': [1.7625, 0.09970, -0.1334, -1.7150, 0.06592],
            'Head': [-0.2, 0.0],
            'Leg': [0.0, 0.0, 0.0]
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
        self.get_logger().info("Gesture Execution System initialized")
    
    def init_joint_states(self):
        """Initialize joint state storage"""
        self.joint_states = {
            'HeadPitch': self.joint_limits.DEFAULT_HEAD_PITCH,
            'HeadYaw': self.joint_limits.DEFAULT_HEAD_YAW,
            'RShoulderPitch': self.descriptor_manager.home_positions['RArm'][0],
            'RShoulderRoll': self.descriptor_manager.home_positions['RArm'][1], 
            'RElbowRoll': self.descriptor_manager.home_positions['RArm'][2],
            'RElbowYaw': self.descriptor_manager.home_positions['RArm'][3],
            'RWristYaw': self.descriptor_manager.home_positions['RArm'][4],
            'LShoulderPitch': self.descriptor_manager.home_positions['LArm'][0],
            'LShoulderRoll': self.descriptor_manager.home_positions['LArm'][1],
            'LElbowRoll': self.descriptor_manager.home_positions['LArm'][2],
            'LElbowYaw': self.descriptor_manager.home_positions['LArm'][3],
            'LWristYaw': self.descriptor_manager.home_positions['LArm'][4],
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
        
        # Publishers
        self.joint_angles_pub = self.create_publisher(JointAnglesWithSpeed, topics["JointAngles"], 10)
        self.cmd_vel_pub = self.create_publisher(Twist, topics["Wheels"], 10)
        
        # Services
        self.create_service(PerformGesture, "/gestureExecution/perform_gesture", self.perform_gesture_callback)
    
    def joint_states_callback(self, msg: JointState):
        """Handle joint state updates"""
        for name, position in zip(msg.name, msg.position):
            if name in self.joint_states:
                self.joint_states[name] = position
    
    def robot_pose_callback(self, msg: Pose2D):
        """Handle robot pose updates"""
        self.robot_pose.x = msg.x
        self.robot_pose.y = msg.y
        self.robot_pose.theta = msg.theta
    
    def perform_gesture_callback(self, request, response):
        """Handle gesture execution requests"""
        try:
            success = self.execute_gesture(
                gesture_type=request.gesture_type,
                gesture_id=request.gesture_id,
                gesture_duration=request.gesture_duration,
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
            response.gesture_success = 0
        
        return response
    
    def execute_gesture(self, gesture_type: str, gesture_id: int, gesture_duration: int,
                       bow_nod_angle: int, location_x: float, location_y: float, location_z: float) -> bool:
        """Main gesture execution logic"""
        
        if self.verbose_mode:
            self.get_logger().info(
                f"Gesture request - Type: {gesture_type}, ID: {gesture_id}, "
                f"Duration: {gesture_duration}ms, Angle: {bow_nod_angle}°, "
                f"Location: ({location_x:.2f}, {location_y:.2f}, {location_z:.2f})"
            )
        
        # Clamp gesture duration to limits
        gesture_duration = max(MIN_GESTURE_DURATION, min(gesture_duration, MAX_GESTURE_DURATION))
        
        # Execute based on gesture type
        if gesture_type == DIECTIC_GESTURES:
            return self.execute_deictic_gesture(location_x, location_y, location_z, gesture_duration)
        
        elif gesture_type == ICONIC_GESTURES:
            return self.execute_iconic_gesture(gesture_id, gesture_duration)
        
        elif gesture_type == SYMBOLIC_GESTURES:
            self.get_logger().warning("Symbolic gestures not implemented yet")
            return False
        
        elif gesture_type == BOWING_GESTURE:
            return self.execute_bowing_gesture(bow_nod_angle, gesture_duration)
        
        elif gesture_type == NODDING_GESTURE:
            return self.execute_nodding_gesture(bow_nod_angle, gesture_duration)
        
        else:
            self.get_logger().warning(f"Unsupported gesture type: {gesture_type}")
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
            
            # Determine which arm to use and if rotation is needed
            rotation_angle = 0.0
            pose_achievable = True
            
            if pointing_x >= 0.0:
                # Target in front - choose arm based on y coordinate
                pointing_arm = RIGHT_ARM if pointing_y <= 0.0 else LEFT_ARM
            else:
                # Target behind - need to rotate robot
                pose_achievable = False
                if pointing_y <= 0.0:
                    rotation_angle = -90.0  # Rotate right
                    pointing_arm = RIGHT_ARM
                    temp = pointing_x
                    pointing_x = -pointing_y
                    pointing_y = temp
                else:
                    rotation_angle = 90.0   # Rotate left
                    pointing_arm = LEFT_ARM
                    temp = pointing_x
                    pointing_x = pointing_y
                    pointing_y = -temp
            
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
            
            # Check joint limits
            if pointing_arm == RIGHT_ARM:
                if (shoulder_pitch < MIN_RSHOULDER_PITCH or shoulder_pitch > MAX_RSHOULDER_PITCH or
                    shoulder_roll < MIN_RSHOULDER_ROLL or shoulder_roll > MAX_RSHOULDER_ROLL):
                    self.get_logger().error("Pointing target outside joint limits")
                    return False
            else:
                if (shoulder_pitch < MIN_LSHOULDER_PITCH or shoulder_pitch > MAX_LSHOULDER_PITCH or
                    shoulder_roll < MIN_LSHOULDER_ROLL or shoulder_roll > MAX_LSHOULDER_ROLL):
                    self.get_logger().error("Pointing target outside joint limits")
                    return False
            
            if math.isnan(shoulder_pitch) or math.isnan(shoulder_roll):
                self.get_logger().error("Invalid joint angles calculated")
                return False
            
            # Rotate robot if needed
            if not pose_achievable:
                self.rotate_robot(rotation_angle)
            
            # Execute pointing motion
            success = self.execute_pointing_motion(pointing_arm, shoulder_pitch, shoulder_roll, duration)
            
            # Rotate back if needed
            if not pose_achievable:
                self.rotate_robot(-rotation_angle)
            
            return success
            
        except Exception as e:
            self.get_logger().error(f"Deictic gesture execution failed: {e}")
            return False
    
    def execute_pointing_motion(self, arm: int, shoulder_pitch: float, shoulder_roll: float, duration: int) -> bool:
        """Execute the actual pointing motion"""
        try:
            duration_sec = duration / 1000.0
            
            # Set fixed values for other joints during pointing
            if arm == RIGHT_ARM:
                joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowRoll', 'RElbowYaw', 'RWristYaw']
                joint_angles = [shoulder_pitch, shoulder_roll, 0.0, 2.0857, -0.05679]
            else:
                joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowRoll', 'LElbowYaw', 'LWristYaw']
                joint_angles = [shoulder_pitch, shoulder_roll, -0.0, -1.5620, 0.06592]
            
            # Move to pointing position
            self.move_joints(joint_names, joint_angles, 0.2)
            time.sleep(duration_sec)
            
            # Return to home position
            home_position = self.descriptor_manager.home_positions['RArm' if arm == RIGHT_ARM else 'LArm']
            self.move_joints(joint_names, home_position, 0.2)
            time.sleep(duration_sec)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Pointing motion failed: {e}")
            return False
    
    def execute_iconic_gesture(self, gesture_id: int, duration: int) -> bool:
        """Execute iconic gesture from YAML configuration"""
        try:
            gesture_info = self.descriptor_manager.get_gesture_by_id(gesture_id)
            if not gesture_info:
                self.get_logger().error(f"Gesture ID {gesture_id} not found")
                return False
            
            gesture_data = gesture_info["data"]
            arm_name = gesture_data.get("arm")
            joint_angles_list = gesture_data.get("joint_angles", [])
            
            if not joint_angles_list:
                self.get_logger().error("No joint angles found in gesture data")
                return False
            
            # Determine joint names based on arm
            if arm_name == "RArm":
                joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
            elif arm_name == "LArm":
                joint_names = ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw']
            else:
                self.get_logger().error(f"Unknown arm: {arm_name}")
                return False
            
            # Execute gesture motion
            return self.execute_iconic_motion(joint_names, joint_angles_list, arm_name, duration)
            
        except Exception as e:
            self.get_logger().error(f"Iconic gesture execution failed: {e}")
            return False
    
    def execute_iconic_motion(self, joint_names: List[str], joint_angles_list: List[List[float]], 
                             arm_name: str, duration: int) -> bool:
        """Execute the iconic gesture motion"""
        try:
            duration_sec = duration / 1000.0
            num_waypoints = len(joint_angles_list)
            waypoint_duration = duration_sec / num_waypoints
            
            # Execute waypoints sequentially
            for waypoint in joint_angles_list:
                self.move_joints(joint_names, waypoint, 0.2)
                time.sleep(waypoint_duration)
            
            # Return to home position
            home_position = self.descriptor_manager.home_positions[arm_name]
            self.move_joints(joint_names, home_position, 0.2)
            time.sleep(waypoint_duration)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Iconic motion execution failed: {e}")
            return False
    
    def execute_bowing_gesture(self, bow_angle: int, duration: int) -> bool:
        """Execute bowing gesture"""
        try:
            duration_sec = duration / 1000.0
            
            # Clamp bow angle to limits
            bow_angle = max(MIN_BOW_ANGLE, min(bow_angle, MAX_BOW_ANGLE))
            bow_angle_rad = -math.radians(bow_angle)  # Negative for forward bow
            
            joint_names = ['HipPitch', 'HipRoll', 'KneePitch']
            bow_position = [bow_angle_rad, -0.00766, 0.03221]
            
            # Move to bow position
            self.move_joints(joint_names, bow_position, 0.2)
            time.sleep(duration_sec)
            
            # Return to home position
            home_position = self.descriptor_manager.home_positions['Leg']
            self.move_joints(joint_names, home_position, 0.2)
            time.sleep(duration_sec)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Bowing gesture failed: {e}")
            return False
    
    def execute_nodding_gesture(self, nod_angle: int, duration: int) -> bool:
        """Execute nodding gesture"""
        try:
            duration_sec = duration / 1000.0
            
            # Clamp nod angle to limits
            nod_angle = max(MIN_NOD_ANGLE, min(nod_angle, MAX_NOD_ANGLE))
            nod_angle_rad = math.radians(nod_angle)
            
            joint_names = ['HeadPitch', 'HeadYaw']
            nod_position = [nod_angle_rad, 0.012271]
            
            # Move to nod position
            self.move_joints(joint_names, nod_position, 0.2)
            time.sleep(duration_sec)
            
            # Return to home position
            home_position = self.descriptor_manager.home_positions['Head']
            self.move_joints(joint_names, home_position, 0.2)
            time.sleep(duration_sec)
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Nodding gesture failed: {e}")
            return False
    
    def rotate_robot(self, angle_degrees: float):
        """Rotate robot by specified angle"""
        try:
            duration = abs(angle_degrees) / 30.0  # 30 degrees per second
            angular_velocity = math.radians(30.0) if angle_degrees > 0 else -math.radians(30.0)
            
            twist = Twist()
            twist.angular.z = angular_velocity
            
            start_time = time.time()
            while time.time() - start_time < duration:
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.1)
            
            # Stop rotation
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            
            if self.verbose_mode:
                self.get_logger().info(f"Robot rotated {angle_degrees} degrees")
                
        except Exception as e:
            self.get_logger().error(f"Robot rotation failed: {e}")
    
    def move_joints(self, joint_names: List[str], joint_angles: List[float], speed: float = 0.2):
        """Move specified joints to target angles using JointAnglesWithSpeed"""
        try:
            msg = JointAnglesWithSpeed()
            msg.joint_names = joint_names
            msg.joint_angles = [float(angle) for angle in joint_angles]
            msg.speed = speed
            msg.relative = False
            
            self.joint_angles_pub.publish(msg)
            
            # Update internal joint state tracking
            for name, angle in zip(joint_names, joint_angles):
                if name in self.joint_states:
                    self.joint_states[name] = angle
                    
        except Exception as e:
            self.get_logger().error(f"Joint movement failed: {e}")