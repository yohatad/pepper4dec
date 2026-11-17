/* lio_odom_guard.cpp
 *
 * See include/pepper_slam/lio_odom_guard.h.
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

#include "pepper_slam/lio_odom_guard.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>

namespace pepper_slam
{

const char * toString(GuardVerdict verdict)
{
  switch (verdict) {
    case GuardVerdict::kAccept: return "accept";
    case GuardVerdict::kHold: return "hold";
    case GuardVerdict::kFault: return "fault";
  }
  return "unknown";
}

tf2::Transform relativeMotion(const tf2::Transform & from, const tf2::Transform & to)
{
  return from.inverse() * to;
}

double rotationAngle(const tf2::Transform & t)
{
  const tf2::Matrix3x3 & r = t.getBasis();
  // Clamp: round-off can put the trace fractionally outside [-1, 3].
  const double cos_theta = (r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5;
  return std::acos(std::clamp(cos_theta, -1.0, 1.0));
}

OdomGuard::OdomGuard(const GuardParams & params)
: params_(params) {}

double OdomGuard::rejectionRate() const
{
  if (verdicts_.empty()) {
    return 0.0;
  }
  return static_cast<double>(std::accumulate(verdicts_.begin(), verdicts_.end(), 0)) /
         static_cast<double>(verdicts_.size());
}

GuardResult OdomGuard::check(
  const tf2::Transform & odom_base, double t,
  const std::optional<tf2::Transform> & wheel_pose, std::optional<double> wheel_speed)
{
  // First pose: nothing to measure a delta against, so accept and anchor.
  if (!last_good_) {
    return accept(odom_base, t);
  }

  const double dt = t - last_good_t_;
  if (dt <= 0.0) {
    // Out-of-order or duplicate stamp: a transport artefact, not a divergence.
    last_reason_ = "non-monotonic stamp";
    return {holding() ? GuardVerdict::kHold : GuardVerdict::kAccept, *last_good_};
  }

  const std::string reason = implausible(odom_base, dt, wheel_speed);
  if (reason.empty()) {
    return accept(odom_base, t);
  }

  last_reason_ = reason;
  verdicts_.push_back(1);
  if (verdicts_.size() > params_.rate_window) {
    verdicts_.pop_front();
  }
  if (!hold_start_t_) {
    hold_start_t_ = t;
    held_distance_ = 0.0;
    hold_wheel_origin_ = wheel_pose;
    hold_wheel_last_ = wheel_pose;
  }

  const tf2::Transform held = deadReckon(wheel_pose);
  if (t - *hold_start_t_ > params_.max_hold_duration) {
    return {GuardVerdict::kFault, held};
  }
  return {GuardVerdict::kHold, held};
}

std::string OdomGuard::implausible(
  const tf2::Transform & odom_base, double dt, std::optional<double> wheel_speed) const
{
  const tf2::Transform delta = relativeMotion(*last_good_, odom_base);
  const double linear = delta.getOrigin().length() / dt;
  const double angular = rotationAngle(delta) / dt;

  char reason[160];
  if (linear > params_.max_linear_speed) {
    std::snprintf(
      reason, sizeof(reason), "linear %.2f m/s over %.0f ms exceeds %.2f m/s", linear,
      dt * 1e3, params_.max_linear_speed);
    return reason;
  }
  if (angular > params_.max_angular_speed) {
    std::snprintf(
      reason, sizeof(reason), "angular %.2f rad/s over %.0f ms exceeds %.2f rad/s", angular,
      dt * 1e3, params_.max_angular_speed);
    return reason;
  }
  // Zero-velocity cross-check: the encoders cannot report stationary while the
  // base actually moves, so a disagreement convicts the LIO.
  if (wheel_speed && *wheel_speed < params_.zero_velocity_eps &&
    linear > params_.zero_velocity_lio_eps)
  {
    std::snprintf(
      reason, sizeof(reason), "wheels stationary (%.3f m/s) while LIO reports %.2f m/s",
      *wheel_speed, linear);
    return reason;
  }
  return {};
}

GuardResult OdomGuard::accept(const tf2::Transform & odom_base, double t)
{
  verdicts_.push_back(0);
  if (verdicts_.size() > params_.rate_window) {
    verdicts_.pop_front();
  }
  last_good_ = odom_base;
  last_good_t_ = t;
  hold_start_t_.reset();
  hold_wheel_origin_.reset();
  hold_wheel_last_.reset();
  held_distance_ = 0.0;
  last_reason_.clear();
  return {GuardVerdict::kAccept, odom_base};
}

tf2::Transform OdomGuard::deadReckon(const std::optional<tf2::Transform> & wheel_pose)
{
  if (!wheel_pose || !hold_wheel_origin_) {
    // No wheel source: the best answer is the last good pose. It freezes
    // rather than coasts, which is why the guard needs wheel odometry.
    return *last_good_;
  }
  // Accumulate path length before composing, so it covers this sample too.
  held_distance_ += relativeMotion(*hold_wheel_last_, *wheel_pose).getOrigin().length();
  hold_wheel_last_ = wheel_pose;
  return *last_good_ * relativeMotion(*hold_wheel_origin_, *wheel_pose);
}

}  // namespace pepper_slam
