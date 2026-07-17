/* pepper_kinematics_utilities.h
 *
 * Pure forward and inverse kinematics helpers for Pepper's arms and head,
 * used by gesture_execution to compute joint angles for deictic pointing
 * gestures.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 05, 2026
 * Version: v1.0 - C++ port of pepper_kinematics_utilities.py
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#ifndef GESTURE_EXECUTION_PEPPER_KINEMATICS_UTILITIES_H
#define GESTURE_EXECUTION_PEPPER_KINEMATICS_UTILITIES_H

#include <array>

namespace pepper_kinematics {

constexpr int RIGHT_ARM = 0;
constexpr int LEFT_ARM = 1;

/** @brief Convert an angle from degrees to radians. */
double degreesToRadians(double degrees);

/** @brief Convert an angle from radians to degrees. */
double radiansToDegrees(double radians);

/**
 * @brief Get the position of the elbow given shoulder angles.
 * @param arm RIGHT_ARM or LEFT_ARM
 * @param theta_1 Shoulder pitch angle (radians)
 * @param theta_2 Shoulder roll angle (radians)
 * @return {x, y, z} position of the elbow
 */
std::array<double, 3> getElbowPosition(int arm, double theta_1, double theta_2);

/**
 * @brief Get the position of the wrist given shoulder angles.
 * @param arm RIGHT_ARM or LEFT_ARM
 * @param theta_1 Shoulder pitch angle (radians)
 * @param theta_2 Shoulder roll angle (radians)
 * @return {x, y, z} position of the wrist
 */
std::array<double, 3> getWristPosition(int arm, double theta_1, double theta_2);

/**
 * @brief Calculate shoulder pitch and roll angles given the elbow position.
 * @param arm RIGHT_ARM or LEFT_ARM
 * @param elbow_x X position of elbow
 * @param elbow_y Y position of elbow
 * @param elbow_z Z position of elbow
 * @return {shoulder_pitch, shoulder_roll} angles in radians
 */
std::array<double, 2> getArmShoulderAngles(int arm, double elbow_x, double elbow_y, double elbow_z);

/**
 * @brief Calculate elbow roll angle given shoulder angles and wrist position.
 * @param arm RIGHT_ARM or LEFT_ARM
 * @param shoulder_pitch Shoulder pitch angle
 * @param shoulder_roll Shoulder roll angle
 * @param wrist_x X position of wrist
 * @param wrist_y Y position of wrist
 * @param wrist_z Z position of wrist
 * @return Elbow roll angle in radians
 */
double getArmElbowRollAngle(int arm, double shoulder_pitch, double shoulder_roll,
                             double wrist_x, double wrist_y, double wrist_z);

/**
 * @brief Calculate elbow yaw angle given other joint angles and wrist position.
 * @param arm RIGHT_ARM or LEFT_ARM
 * @param shoulder_pitch Shoulder pitch angle
 * @param shoulder_roll Shoulder roll angle
 * @param elbow_roll Elbow roll angle
 * @param wrist_x X position of wrist
 * @param wrist_y Y position of wrist
 * @param wrist_z Z position of wrist
 * @return Elbow yaw angle in radians
 */
double getArmElbowYawAngle(int arm, double shoulder_pitch, double shoulder_roll, double elbow_roll,
                            double wrist_x, double wrist_y, double wrist_z);

/**
 * @brief Calculate both elbow angles given shoulder angles and wrist position.
 * @param arm RIGHT_ARM or LEFT_ARM
 * @param shoulder_pitch Shoulder pitch angle
 * @param shoulder_roll Shoulder roll angle
 * @param wrist_x X position of wrist
 * @param wrist_y Y position of wrist
 * @param wrist_z Z position of wrist
 * @return {elbow_yaw, elbow_roll} angles in radians
 */
std::array<double, 2> getArmElbowAngles(int arm, double shoulder_pitch, double shoulder_roll,
                                         double wrist_x, double wrist_y, double wrist_z);

/**
 * @brief Calculate all arm joint angles given elbow and wrist positions.
 * @param arm RIGHT_ARM or LEFT_ARM
 * @param elbow_x X position of elbow
 * @param elbow_y Y position of elbow
 * @param elbow_z Z position of elbow
 * @param wrist_x X position of wrist
 * @param wrist_y Y position of wrist
 * @param wrist_z Z position of wrist
 * @return {shoulder_pitch, shoulder_roll, elbow_yaw, elbow_roll} angles in
 *         radians. NOTE: elbow_yaw/elbow_roll are always 0.0 here — mirrors
 *         the Python implementation, which leaves them as placeholder
 *         defaults.
 */
std::array<double, 4> getArmAngles(int arm, double elbow_x, double elbow_y, double elbow_z,
                                    double wrist_x, double wrist_y, double wrist_z);

/**
 * @brief Calculate head angles given camera position.
 * @param camera_x X position of camera
 * @param camera_y Y position of camera
 * @param camera_z Z position of camera
 * @return {head_yaw, head_pitch} angles in radians
 */
std::array<double, 2> getHeadAngles(double camera_x, double camera_y, double camera_z);

}  // namespace pepper_kinematics

#endif  // GESTURE_EXECUTION_PEPPER_KINEMATICS_UTILITIES_H
