#!/usr/bin/env python3
"""
pepper_kinematics_utilities.py

Author: Yohannes Haile
Date: September 24, 2025
Version: v1.0

Pure Python implementation of Pepper robot kinematics utilities
"""

import math
from typing import Tuple

# Constants for arm identification
RIGHT_ARM = 0
LEFT_ARM = 1

class PepperKinematicsUtilities:
    """
    Utility class for Pepper robot kinematics calculations
    Provides forward and inverse kinematics functions for Pepper's arms and head
    """

    # --------------------------------------------------
    #           CONVERSION FUNCTIONS 
    # --------------------------------------------------
    
    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        """
        Convert angle from degrees to radians
        
        Args:
            degrees: Angle in degrees
            
        Returns:
            Angle in radians
        """
        return degrees * math.pi / 180.0
    
    @staticmethod
    def radians_to_degrees(radians: float) -> float:
        """
        Convert angle from radians to degrees
        
        Args:
            radians: Angle in radians
            
        Returns:
            Angle in degrees
        """
        return radians * 180.0 / math.pi

    # --------------------------------------------------
    #           FORWARD KINEMATICS FUNCTIONS 
    # --------------------------------------------------
    
    @staticmethod
    def get_elbow_position(arm: int, theta_1: float, theta_2: float) -> Tuple[float, float, float]:
        """
        Get the position of the elbow given shoulder angles
        
        Args:
            arm: RIGHT_ARM or LEFT_ARM
            theta_1: Shoulder pitch angle (radians)
            theta_2: Shoulder roll angle (radians)
            
        Returns:
            Tuple of (x, y, z) position of the elbow
        """
        # Define the lengths of the arm segments
        l_1 = -57.0
        l_2 = -149.74 if arm == RIGHT_ARM else 149.74
        l_3 = 86.82
        l_4 = 181.2
        l_5 = -15.0 if arm == RIGHT_ARM else 15.0
        l_6 = 0.13

        # Calculate trigonometric values
        sin_theta_1 = math.sin(theta_1)
        cos_theta_1 = math.cos(theta_1)
        sin_theta_2 = math.sin(theta_2)
        cos_theta_2 = math.cos(theta_2)

        # Calculate elbow position
        f_1 = l_6 * sin_theta_1
        f_2 = l_4 * cos_theta_2
        f_3 = l_5 * sin_theta_2
        f_4 = f_2 - f_3
        f_5 = cos_theta_1 * f_4
        position_x = l_1 + f_1 + f_5

        # Y position
        f_1 = l_5 * cos_theta_2
        f_2 = l_4 * sin_theta_2
        position_y = l_2 + f_1 + f_2

        # Z position
        f_1 = l_6 * cos_theta_1
        f_2 = sin_theta_1 * f_4
        position_z = l_3 + f_1 - f_2

        return (position_x, position_y, position_z)
    
    @staticmethod
    def get_wrist_position(arm: int, theta_1: float, theta_2: float) -> Tuple[float, float, float]:
        """
        Get the position of the wrist given shoulder angles
        
        Args:
            arm: RIGHT_ARM or LEFT_ARM
            theta_1: Shoulder pitch angle (radians)
            theta_2: Shoulder roll angle (radians)
            
        Returns:
            Tuple of (x, y, z) position of the wrist
        """
        # Define the lengths of the arm segments
        l_1 = -57.0
        l_2 = -149.74 if arm == RIGHT_ARM else 149.74
        l_3 = 86.82
        l_4 = 181.2
        l_5 = -15.0 if arm == RIGHT_ARM else 15.0
        l_6 = 0.13

        # Calculate trigonometric values
        sin_theta_1 = math.sin(theta_1)
        cos_theta_1 = math.cos(theta_1)
        sin_theta_2 = math.sin(theta_2)
        cos_theta_2 = math.cos(theta_2)

        # Calculate wrist position
        f_1 = l_6 * sin_theta_1
        f_2 = l_4 * cos_theta_2
        f_3 = l_5 * sin_theta_2
        f_4 = f_2 - f_3
        f_5 = cos_theta_1 * f_4
        position_x = l_1 + f_1 + f_5

        # Y position
        f_1 = l_5 * cos_theta_2
        f_2 = l_4 * sin_theta_2
        position_y = l_2 + f_1 + f_2

        # Z position
        f_1 = l_6 * cos_theta_1
        f_2 = sin_theta_1 * f_4
        position_z = l_3 + f_1 - f_2

        return (position_x, position_y, position_z)

    # --------------------------------------------------
    #           INVERSE KINEMATICS FUNCTIONS 
    # --------------------------------------------------
    
    @staticmethod
    def get_arm_shoulder_angles(arm: int, elbow_x: float, elbow_y: float, elbow_z: float) -> Tuple[float, float]:
        """
        Calculate shoulder pitch and roll angles given elbow position
        
        Args:
            arm: RIGHT_ARM or LEFT_ARM
            elbow_x: X position of elbow
            elbow_y: Y position of elbow
            elbow_z: Z position of elbow
            
        Returns:
            Tuple of (shoulder_pitch, shoulder_roll) angles in radians
        """
        # Define the lengths of the arm segments
        l_1 = -57.0
        l_2 = -149.74 if arm == RIGHT_ARM else 149.74
        l_3 = 86.82
        l_4 = 181.2
        l_5 = -15.0 if arm == RIGHT_ARM else 15.0
        l_6 = 0.13

        # Calculate shoulder_roll (theta_2)
        f_1 = elbow_y - l_2
        f_2 = math.sqrt(l_4**2 + l_5**2)
        f_3 = math.asin(f_1 / f_2)
        f_4 = math.atan(l_5 / l_4)
        t_2_temp = f_3 - f_4

        if arm == RIGHT_ARM:
            if (t_2_temp + f_4) > (-math.pi/2 - f_4):
                shoulder_roll = t_2_temp
            else:
                shoulder_roll = -math.pi - f_3 - f_4
                if shoulder_roll < -1.5630:
                    shoulder_roll = t_2_temp
            
            # Check if solution is within Pepper's range
            if shoulder_roll < -1.58 or shoulder_roll >= -0.0087:
                shoulder_roll = -0.0087
        else:  # LEFT_ARM
            if t_2_temp + f_4 < math.pi/2 - f_4:
                shoulder_roll = t_2_temp
            else:
                shoulder_roll = math.pi - f_3 - f_4
                if shoulder_roll > 1.5630:
                    shoulder_roll = t_2_temp
            
            # Check if solution is within Pepper's range
            if shoulder_roll > 1.58 or shoulder_roll <= 0.0087:
                shoulder_roll = 0.0087

        # Calculate shoulder_pitch (theta_1)
        n = (l_4 * math.cos(shoulder_roll)) - (l_5 * math.sin(shoulder_roll))
        f_1 = elbow_x - l_1
        f_2 = elbow_z - l_3
        f_3 = math.atan2(f_1, f_2)
        f_4 = math.sqrt(f_1**2 + f_2**2 - l_6**2)
        f_4 = math.atan2(f_4, l_6)
        t_1_1 = f_3 - f_4
        
        f_3 = (l_6 * f_1) - (n * f_2)
        f_4 = (l_6 * f_2) + (n * f_1)
        t_1_2 = math.atan2(f_3, f_4)

        # Check if solutions are within Pepper's range
        if t_1_1 < -2.1 or t_1_1 > 2.1:
            t_1_1 = float('nan')
        if t_1_2 < -2.1 or t_1_2 > 2.1:
            t_1_2 = float('nan')

        # Calculate positions for both solutions
        pos_1 = PepperKinematicsUtilities.get_elbow_position(arm, t_1_1, shoulder_roll) if not math.isnan(t_1_1) else (float('inf'), float('inf'), float('inf'))
        pos_2 = PepperKinematicsUtilities.get_elbow_position(arm, t_1_2, shoulder_roll) if not math.isnan(t_1_2) else (float('inf'), float('inf'), float('inf'))

        # Choose the solution with smaller distance error
        dist_1 = math.sqrt((pos_1[0] - elbow_x)**2 + (pos_1[1] - elbow_y)**2 + (pos_1[2] - elbow_z)**2)
        dist_2 = math.sqrt((pos_2[0] - elbow_x)**2 + (pos_2[1] - elbow_y)**2 + (pos_2[2] - elbow_z)**2)

        if dist_1 <= dist_2 or math.isnan(dist_2):
            shoulder_pitch = t_1_1
        else:
            shoulder_pitch = t_1_2

        return (shoulder_pitch, shoulder_roll)
    
    @staticmethod
    def get_arm_elbow_roll_angle(arm: int, shoulder_pitch: float, shoulder_roll: float, 
                                wrist_x: float, wrist_y: float, wrist_z: float) -> float:
        """
        Calculate elbow roll angle given shoulder angles and wrist position
        
        Args:
            arm: RIGHT_ARM or LEFT_ARM
            shoulder_pitch: Shoulder pitch angle
            shoulder_roll: Shoulder roll angle
            wrist_x: X position of wrist
            wrist_y: Y position of wrist
            wrist_z: Z position of wrist
            
        Returns:
            Elbow roll angle in radians
        """
        # Define the lengths of the arm segments
        l_1 = -57.0
        l_2 = -149.74 if arm == RIGHT_ARM else 149.74
        l_3 = 86.82
        l_4 = 150
        d_3 = 181.2
        z_3 = 0.13
        alpha = math.radians(9)  # 9 degrees

        # Calculate the ElbowRoll angle
        t_2 = shoulder_roll - math.pi/2
        term_3 = ((wrist_x - l_1) * math.sin(shoulder_pitch)) + ((wrist_z - l_3) * math.cos(shoulder_pitch)) - z_3
        term_4 = ((wrist_z - l_3) * math.sin(shoulder_pitch) * math.sin(t_2)) + ((wrist_y - l_2) * math.cos(t_2)) - d_3 - ((wrist_x - l_1) * math.sin(t_2) * math.cos(shoulder_pitch))
        term_2 = (math.sin(alpha) * term_3) + (math.cos(alpha) * term_4)
        term_1 = (1.0 / l_4) * term_2

        if term_1 > 1.0:
            elbow_roll = float('nan')
        else:
            if arm == RIGHT_ARM:
                elbow_roll = math.acos(term_1)
            else:  # LEFT_ARM
                elbow_roll = -math.acos(term_1)

        # Check if solution is within Pepper's range
        if arm == RIGHT_ARM:
            if elbow_roll > 1.58 or elbow_roll < 0.0087 or math.isnan(elbow_roll):
                elbow_roll = 0.0087
        else:  # LEFT_ARM
            if elbow_roll > -0.0087 or elbow_roll < -1.58 or math.isnan(elbow_roll):
                elbow_roll = -0.0087

        return elbow_roll
    
    @staticmethod
    def get_arm_elbow_yaw_angle(arm: int, shoulder_pitch: float, shoulder_roll: float, 
                               elbow_roll: float, wrist_x: float, wrist_y: float, wrist_z: float) -> float:
        """
        Calculate elbow yaw angle given other joint angles and wrist position
        
        Args:
            arm: RIGHT_ARM or LEFT_ARM
            shoulder_pitch: Shoulder pitch angle
            shoulder_roll: Shoulder roll angle
            elbow_roll: Elbow roll angle
            wrist_x: X position of wrist
            wrist_y: Y position of wrist
            wrist_z: Z position of wrist
            
        Returns:
            Elbow yaw angle in radians
        """
        # Define the lengths of the arm segments
        l_1 = -57.0
        l_2 = -149.74 if arm == RIGHT_ARM else 149.74
        l_3 = 86.82
        a_3 = 15.0 if arm == RIGHT_ARM else -15.0
        d_3 = 181.2
        d_5 = 150
        alpha = math.radians(9)  # 9 degrees

        # Calculate the ElbowYaw angle
        t_2 = shoulder_roll - math.pi/2

        try:
            a_term = (d_3 + (d_5 * math.cos(alpha) * math.cos(elbow_roll)) + 
                     (math.cos(shoulder_pitch) * math.sin(t_2) * (wrist_x - l_1)) - 
                     (math.sin(shoulder_pitch) * math.sin(t_2) * (wrist_z - l_3)) - 
                     (math.cos(t_2) * (wrist_y - l_2))) / (d_5 * math.sin(elbow_roll) * math.sin(alpha))
            
            b_term = ((math.cos(t_2) * math.sin(shoulder_pitch) * (wrist_z - l_3)) + a_3 - 
                     (math.cos(shoulder_pitch) * math.cos(t_2) * (wrist_x - l_1)) - 
                     (math.sin(t_2) * (wrist_y - l_2))) / (d_5 * math.sin(elbow_roll))

            elbow_yaw = math.atan2(a_term, b_term)
        except (ZeroDivisionError, ValueError):
            elbow_yaw = 0.0

        return elbow_yaw
    
    @staticmethod
    def get_arm_elbow_angles(arm: int, shoulder_pitch: float, shoulder_roll: float,
                            wrist_x: float, wrist_y: float, wrist_z: float) -> Tuple[float, float]:
        """
        Calculate both elbow angles given shoulder angles and wrist position
        
        Args:
            arm: RIGHT_ARM or LEFT_ARM
            shoulder_pitch: Shoulder pitch angle
            shoulder_roll: Shoulder roll angle
            wrist_x: X position of wrist
            wrist_y: Y position of wrist
            wrist_z: Z position of wrist
            
        Returns:
            Tuple of (elbow_yaw, elbow_roll) angles in radians
        """
        elbow_roll = PepperKinematicsUtilities.get_arm_elbow_roll_angle(arm, shoulder_pitch, shoulder_roll, wrist_x, wrist_y, wrist_z)
        elbow_yaw = PepperKinematicsUtilities.get_arm_elbow_yaw_angle(arm, shoulder_pitch, shoulder_roll, elbow_roll, wrist_x, wrist_y, wrist_z)
        return (elbow_yaw, elbow_roll)
    
    @staticmethod
    def get_arm_angles(arm: int, elbow_x: float, elbow_y: float, elbow_z: float,
                      wrist_x: float, wrist_y: float, wrist_z: float) -> Tuple[float, float, float, float]:
        """
        Calculate all arm joint angles given elbow and wrist positions
        
        Args:
            arm: RIGHT_ARM or LEFT_ARM
            elbow_x: X position of elbow
            elbow_y: Y position of elbow
            elbow_z: Z position of elbow
            wrist_x: X position of wrist
            wrist_y: Y position of wrist
            wrist_z: Z position of wrist
            
        Returns:
            Tuple of (shoulder_pitch, shoulder_roll, elbow_yaw, elbow_roll) angles in radians
        """
        shoulder_pitch, shoulder_roll = PepperKinematicsUtilities.get_arm_shoulder_angles(arm, elbow_x, elbow_y, elbow_z)
        
        # For basic implementation, elbow angles can be calculated if needed
        # elbow_yaw, elbow_roll = PepperKinematicsUtilities.get_arm_elbow_angles(arm, shoulder_pitch, shoulder_roll, wrist_x, wrist_y, wrist_z)
        elbow_yaw = 0.0  # Default values as mentioned in original code
        elbow_roll = 0.0
        
        return (shoulder_pitch, shoulder_roll, elbow_yaw, elbow_roll)

    # --------------------------------------------------
    #           HEAD KINEMATICS FUNCTIONS 
    # --------------------------------------------------
    
    @staticmethod
    def get_head_angles(camera_x: float, camera_y: float, camera_z: float) -> Tuple[float, float]:
        """
        Calculate head angles given camera position
        
        Args:
            camera_x: X position of camera
            camera_y: Y position of camera
            camera_z: Z position of camera
            
        Returns:
            Tuple of (head_yaw, head_pitch) angles in radians
        """
        # Define the lengths of the head chain
        l_1 = -38.0
        l_2 = 169.9
        l_3 = 93.6
        l_4 = 61.6

        # Calculate head angles
        head_yaw = math.atan2(camera_y, (camera_x - l_1))
        head_pitch = math.asin((l_2 - camera_z) / math.sqrt(l_4**2 + l_3**2)) + math.atan(l_4 / l_3)

        # Check if angles are within Pepper's range
        if math.isnan(head_yaw) or head_yaw < -2.1 or head_yaw > 2.1:
            head_yaw = 0.0
        if math.isnan(head_pitch) or head_pitch < -0.71 or head_pitch > 0.6371:
            head_pitch = 0.0

        return (head_yaw, head_pitch)