/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: July 05, 2026
Version: v1.0 - C++ port of pepper_kinematics_utilities.py
*/

#include "gesture_execution/pepper_kinematics_utilities.h"

#include <cmath>
#include <limits>

namespace pepper_kinematics {

double degreesToRadians(double degrees) {
    return degrees * M_PI / 180.0;
}

double radiansToDegrees(double radians) {
    return radians * 180.0 / M_PI;
}

std::array<double, 3> getElbowPosition(int arm, double theta_1, double theta_2) {
    // Arm segment lengths
    double l_1 = -57.0;
    double l_2 = (arm == RIGHT_ARM) ? -149.74 : 149.74;
    double l_3 = 86.82;
    double l_4 = 181.2;
    double l_5 = (arm == RIGHT_ARM) ? -15.0 : 15.0;
    double l_6 = 0.13;

    double sin_theta_1 = std::sin(theta_1);
    double cos_theta_1 = std::cos(theta_1);
    double sin_theta_2 = std::sin(theta_2);
    double cos_theta_2 = std::cos(theta_2);

    double f_1 = l_6 * sin_theta_1;
    double f_2 = l_4 * cos_theta_2;
    double f_3 = l_5 * sin_theta_2;
    double f_4 = f_2 - f_3;
    double f_5 = cos_theta_1 * f_4;
    double position_x = l_1 + f_1 + f_5;

    f_1 = l_5 * cos_theta_2;
    f_2 = l_4 * sin_theta_2;
    double position_y = l_2 + f_1 + f_2;

    f_1 = l_6 * cos_theta_1;
    f_2 = sin_theta_1 * f_4;
    double position_z = l_3 + f_1 - f_2;

    return {position_x, position_y, position_z};
}

std::array<double, 3> getWristPosition(int arm, double theta_1, double theta_2) {
    // Identical formulation to getElbowPosition in the reference implementation.
    return getElbowPosition(arm, theta_1, theta_2);
}

std::array<double, 2> getArmShoulderAngles(int arm, double elbow_x, double elbow_y, double elbow_z) {
    double l_1 = -57.0;
    double l_2 = (arm == RIGHT_ARM) ? -149.74 : 149.74;
    double l_3 = 86.82;
    double l_4 = 181.2;
    double l_5 = (arm == RIGHT_ARM) ? -15.0 : 15.0;
    double l_6 = 0.13;

    // Shoulder roll (theta_2)
    double f_1 = elbow_y - l_2;
    double f_2 = std::sqrt(l_4 * l_4 + l_5 * l_5);
    double f_3 = std::asin(f_1 / f_2);
    double f_4 = std::atan(l_5 / l_4);
    double t_2_temp = f_3 - f_4;

    double shoulder_roll;
    if (arm == RIGHT_ARM) {
        if ((t_2_temp + f_4) > (-M_PI / 2.0 - f_4)) {
            shoulder_roll = t_2_temp;
        } else {
            shoulder_roll = -M_PI - f_3 - f_4;
            if (shoulder_roll < -1.5630) {
                shoulder_roll = t_2_temp;
            }
        }
        if (shoulder_roll < -1.58 || shoulder_roll >= -0.0087) {
            shoulder_roll = -0.0087;
        }
    } else {  // LEFT_ARM
        if (t_2_temp + f_4 < M_PI / 2.0 - f_4) {
            shoulder_roll = t_2_temp;
        } else {
            shoulder_roll = M_PI - f_3 - f_4;
            if (shoulder_roll > 1.5630) {
                shoulder_roll = t_2_temp;
            }
        }
        if (shoulder_roll > 1.58 || shoulder_roll <= 0.0087) {
            shoulder_roll = 0.0087;
        }
    }

    // Shoulder pitch (theta_1)
    double n = (l_4 * std::cos(shoulder_roll)) - (l_5 * std::sin(shoulder_roll));
    f_1 = elbow_x - l_1;
    f_2 = elbow_z - l_3;
    f_3 = std::atan2(f_1, f_2);
    f_4 = std::sqrt(f_1 * f_1 + f_2 * f_2 - l_6 * l_6);
    f_4 = std::atan2(f_4, l_6);
    double t_1_1 = f_3 - f_4;

    f_3 = (l_6 * f_1) - (n * f_2);
    f_4 = (l_6 * f_2) + (n * f_1);
    double t_1_2 = std::atan2(f_3, f_4);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (t_1_1 < -2.1 || t_1_1 > 2.1) t_1_1 = nan;
    if (t_1_2 < -2.1 || t_1_2 > 2.1) t_1_2 = nan;

    auto pos_1 = std::isnan(t_1_1)
        ? std::array<double, 3>{std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity()}
        : getElbowPosition(arm, t_1_1, shoulder_roll);
    auto pos_2 = std::isnan(t_1_2)
        ? std::array<double, 3>{std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity()}
        : getElbowPosition(arm, t_1_2, shoulder_roll);

    double dist_1 = std::sqrt(std::pow(pos_1[0] - elbow_x, 2) + std::pow(pos_1[1] - elbow_y, 2) +
                               std::pow(pos_1[2] - elbow_z, 2));
    double dist_2 = std::sqrt(std::pow(pos_2[0] - elbow_x, 2) + std::pow(pos_2[1] - elbow_y, 2) +
                               std::pow(pos_2[2] - elbow_z, 2));

    double shoulder_pitch = (dist_1 <= dist_2 || std::isnan(dist_2)) ? t_1_1 : t_1_2;

    return {shoulder_pitch, shoulder_roll};
}

double getArmElbowRollAngle(int arm, double shoulder_pitch, double shoulder_roll,
                             double wrist_x, double wrist_y, double wrist_z) {
    double l_1 = -57.0;
    double l_2 = (arm == RIGHT_ARM) ? -149.74 : 149.74;
    double l_3 = 86.82;
    double l_4 = 150.0;
    double d_3 = 181.2;
    double z_3 = 0.13;
    double alpha = degreesToRadians(9.0);

    double t_2 = shoulder_roll - M_PI / 2.0;
    double term_3 = ((wrist_x - l_1) * std::sin(shoulder_pitch)) +
                     ((wrist_z - l_3) * std::cos(shoulder_pitch)) - z_3;
    double term_4 = ((wrist_z - l_3) * std::sin(shoulder_pitch) * std::sin(t_2)) +
                     ((wrist_y - l_2) * std::cos(t_2)) - d_3 -
                     ((wrist_x - l_1) * std::sin(t_2) * std::cos(shoulder_pitch));
    double term_2 = (std::sin(alpha) * term_3) + (std::cos(alpha) * term_4);
    double term_1 = (1.0 / l_4) * term_2;

    double elbow_roll;
    if (term_1 > 1.0) {
        elbow_roll = std::numeric_limits<double>::quiet_NaN();
    } else {
        elbow_roll = (arm == RIGHT_ARM) ? std::acos(term_1) : -std::acos(term_1);
    }

    if (arm == RIGHT_ARM) {
        if (elbow_roll > 1.58 || elbow_roll < 0.0087 || std::isnan(elbow_roll)) {
            elbow_roll = 0.0087;
        }
    } else {  // LEFT_ARM
        if (elbow_roll > -0.0087 || elbow_roll < -1.58 || std::isnan(elbow_roll)) {
            elbow_roll = -0.0087;
        }
    }

    return elbow_roll;
}

double getArmElbowYawAngle(int arm, double shoulder_pitch, double shoulder_roll, double elbow_roll,
                            double wrist_x, double wrist_y, double wrist_z) {
    double l_1 = -57.0;
    double l_2 = (arm == RIGHT_ARM) ? -149.74 : 149.74;
    double l_3 = 86.82;
    double a_3 = (arm == RIGHT_ARM) ? 15.0 : -15.0;
    double d_3 = 181.2;
    double d_5 = 150.0;
    double alpha = degreesToRadians(9.0);

    double t_2 = shoulder_roll - M_PI / 2.0;

    double denom_a = d_5 * std::sin(elbow_roll) * std::sin(alpha);
    double denom_b = d_5 * std::sin(elbow_roll);
    if (denom_a == 0.0 || denom_b == 0.0) {
        return 0.0;
    }

    double a_term = (d_3 + (d_5 * std::cos(alpha) * std::cos(elbow_roll)) +
                      (std::cos(shoulder_pitch) * std::sin(t_2) * (wrist_x - l_1)) -
                      (std::sin(shoulder_pitch) * std::sin(t_2) * (wrist_z - l_3)) -
                      (std::cos(t_2) * (wrist_y - l_2))) / denom_a;

    double b_term = ((std::cos(t_2) * std::sin(shoulder_pitch) * (wrist_z - l_3)) + a_3 -
                      (std::cos(shoulder_pitch) * std::cos(t_2) * (wrist_x - l_1)) -
                      (std::sin(t_2) * (wrist_y - l_2))) / denom_b;

    return std::atan2(a_term, b_term);
}

std::array<double, 2> getArmElbowAngles(int arm, double shoulder_pitch, double shoulder_roll,
                                         double wrist_x, double wrist_y, double wrist_z) {
    double elbow_roll = getArmElbowRollAngle(arm, shoulder_pitch, shoulder_roll, wrist_x, wrist_y, wrist_z);
    double elbow_yaw = getArmElbowYawAngle(arm, shoulder_pitch, shoulder_roll, elbow_roll, wrist_x, wrist_y, wrist_z);
    return {elbow_yaw, elbow_roll};
}

std::array<double, 4> getArmAngles(int arm, double elbow_x, double elbow_y, double elbow_z,
                                    double wrist_x, double wrist_y, double wrist_z) {
    (void)wrist_x;
    (void)wrist_y;
    (void)wrist_z;
    auto shoulder = getArmShoulderAngles(arm, elbow_x, elbow_y, elbow_z);
    // Elbow angles default to 0.0 here — mirrors the reference implementation,
    // which leaves this computation commented out.
    return {shoulder[0], shoulder[1], 0.0, 0.0};
}

std::array<double, 2> getHeadAngles(double camera_x, double camera_y, double camera_z) {
    double l_1 = -38.0;
    double l_2 = 169.9;
    double l_3 = 93.6;
    double l_4 = 61.6;

    double head_yaw = std::atan2(camera_y, (camera_x - l_1));
    double head_pitch = std::asin((l_2 - camera_z) / std::sqrt(l_4 * l_4 + l_3 * l_3)) + std::atan(l_4 / l_3);

    if (std::isnan(head_yaw) || head_yaw < -2.1 || head_yaw > 2.1) {
        head_yaw = 0.0;
    }
    if (std::isnan(head_pitch) || head_pitch < -0.71 || head_pitch > 0.6371) {
        head_pitch = 0.0;
    }

    return {head_yaw, head_pitch};
}

}  // namespace pepper_kinematics
