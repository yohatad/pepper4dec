/* transform_utils.cpp
 *
 * See include/pepper_slam/transform_utils.h.
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

#include "pepper_slam/transform_utils.h"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

namespace pepper_slam
{

tf2::Transform toTransform(
  const geometry_msgs::msg::Point & p, const geometry_msgs::msg::Quaternion & q)
{
  const double n = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  const tf2::Quaternion rotation = n < 1e-12 ?
    tf2::Quaternion(0.0, 0.0, 0.0, 1.0) :
    tf2::Quaternion(q.x, q.y, q.z, q.w).normalized();
  return tf2::Transform(rotation, tf2::Vector3(p.x, p.y, p.z));
}

tf2::Transform toTransform(const geometry_msgs::msg::TransformStamped & t)
{
  geometry_msgs::msg::Point p;
  p.x = t.transform.translation.x;
  p.y = t.transform.translation.y;
  p.z = t.transform.translation.z;
  return toTransform(p, t.transform.rotation);
}

void fill(geometry_msgs::msg::TransformStamped & out, const tf2::Transform & t)
{
  const tf2::Vector3 & o = t.getOrigin();
  tf2::Quaternion q;
  t.getBasis().getRotation(q);
  q.normalize();
  out.transform.translation.x = o.x();
  out.transform.translation.y = o.y();
  out.transform.translation.z = o.z();
  out.transform.rotation.x = q.x();
  out.transform.rotation.y = q.y();
  out.transform.rotation.z = q.z();
  out.transform.rotation.w = q.w();
}

double toSec(const builtin_interfaces::msg::Time & stamp)
{
  return stamp.sec + stamp.nanosec * 1e-9;
}

tf2::Transform levelFromCalibration(const tf2::Transform & base_imu, bool on_floor)
{
  return tf2::Transform(
    base_imu.getBasis(), tf2::Vector3(0.0, 0.0, on_floor ? base_imu.getOrigin().z() : 0.0));
}

tf2::Transform levelFromPose(const tf2::Transform & world_base, bool on_floor)
{
  const tf2::Matrix3x3 r = world_base.getBasis().transpose();  // inverse rotation -> upright
  // Rotation alone leaves the origin at the start pose, about lidar-mount
  // height above the ground; cancel the base's levelled height.
  const double z = on_floor ? -(r * world_base.getOrigin()).z() : 0.0;
  return tf2::Transform(r, tf2::Vector3(0.0, 0.0, z));
}

}  // namespace pepper_slam
