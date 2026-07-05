/* pepper_kinematics_utilities.h
 *
 * Pure forward and inverse kinematics helpers for Pepper's arms and head,
 * used by gesture_execution to compute joint angles for deictic pointing
 * gestures.
 *
 * Author: Yohannes Tadesse Haile
 * Date: Jul 05, 2026
 * Version: v1.0 - C++ port of pepper_kinematics_utilities.py
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#ifndef GESTURE_EXECUTION_PEPPER_KINEMATICS_UTILITIES_H
#define GESTURE_EXECUTION_PEPPER_KINEMATICS_UTILITIES_H

#include <array>

namespace pepper_kinematics {

constexpr int RIGHT_ARM = 0;
constexpr int LEFT_ARM = 1;

double degreesToRadians(double degrees);
double radiansToDegrees(double radians);

// Get the position of the elbow given shoulder angles. Returns {x, y, z}.
std::array<double, 3> getElbowPosition(int arm, double theta_1, double theta_2);

// Get the position of the wrist given shoulder angles. Returns {x, y, z}.
std::array<double, 3> getWristPosition(int arm, double theta_1, double theta_2);

// Calculate shoulder pitch/roll angles given the elbow position. Returns {pitch, roll}.
std::array<double, 2> getArmShoulderAngles(int arm, double elbow_x, double elbow_y, double elbow_z);

// Calculate elbow roll angle given shoulder angles and wrist position.
double getArmElbowRollAngle(int arm, double shoulder_pitch, double shoulder_roll,
                             double wrist_x, double wrist_y, double wrist_z);

// Calculate elbow yaw angle given other joint angles and wrist position.
double getArmElbowYawAngle(int arm, double shoulder_pitch, double shoulder_roll, double elbow_roll,
                            double wrist_x, double wrist_y, double wrist_z);

// Calculate both elbow angles given shoulder angles and wrist position. Returns {yaw, roll}.
std::array<double, 2> getArmElbowAngles(int arm, double shoulder_pitch, double shoulder_roll,
                                         double wrist_x, double wrist_y, double wrist_z);

// Calculate all arm joint angles given elbow and wrist positions.
// Returns {shoulder_pitch, shoulder_roll, elbow_yaw, elbow_roll}.
// NOTE: elbow_yaw/elbow_roll are always 0.0 here — mirrors the Python
// implementation, which leaves them as placeholder defaults.
std::array<double, 4> getArmAngles(int arm, double elbow_x, double elbow_y, double elbow_z,
                                    double wrist_x, double wrist_y, double wrist_z);

// Calculate head angles given camera position. Returns {yaw, pitch}.
std::array<double, 2> getHeadAngles(double camera_x, double camera_y, double camera_z);

}  // namespace pepper_kinematics

#endif  // GESTURE_EXECUTION_PEPPER_KINEMATICS_UTILITIES_H
