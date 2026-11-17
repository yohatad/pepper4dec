/* transform_utils.h
 *
 * Pose and transform helpers shared by pepper_slam's nodes (lio_odom_bridge,
 * pgo_map_odom_bridge, the frame-contract test): message <-> tf2::Transform
 * conversions, and the one-time levelling both bridges publish.
 *
 * Levelling: a LIO estimator's world frame starts at identity rotation, so its
 * axes are the IMU's axes, tilted by however the IMU is mounted. The level
 * frame above it is gravity-aligned, with z = 0 on the floor. Two sources:
 *
 *   calibration  the rotation is the static base_footprint -> IMU mount
 *                rotation and the floor offset is the mount height: exact,
 *                and identical on every run and backend.
 *   pose         (legacy) the inverse of the base's rotation in the world
 *                frame at one instant; any motion before it leaks in.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: November 17, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#pragma once

#include <tf2/LinearMath/Transform.h>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

namespace pepper_slam
{

// A pose as a transform. A zero-length quaternion becomes identity rotation
// (an uninitialised message); any other is normalised.
tf2::Transform toTransform(
  const geometry_msgs::msg::Point & position, const geometry_msgs::msg::Quaternion & orientation);

tf2::Transform toTransform(const geometry_msgs::msg::TransformStamped & transform);

// Write a transform into a message's translation and (normalised) rotation.
void fill(geometry_msgs::msg::TransformStamped & out, const tf2::Transform & transform);

double toSec(const builtin_interfaces::msg::Time & stamp);

// level_frame -> world frame from the static base -> IMU mount. With
// on_floor, the origin drops by the mount height so z = 0 is the floor.
tf2::Transform levelFromCalibration(const tf2::Transform & base_imu, bool on_floor);

// level_frame -> world frame from the base's pose in the world frame at one
// instant (legacy). With on_floor, base's levelled height is cancelled.
tf2::Transform levelFromPose(const tf2::Transform & world_base, bool on_floor);

}  // namespace pepper_slam
