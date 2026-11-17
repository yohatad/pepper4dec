/* lio_odom_guard.h
 *
 * Validation gate for a LIO pose, with wheel-odometry dead reckoning: the
 * divergence guard inside lio_odom_bridge.
 *
 * Rejects (never saturates) poses that imply motion above a physical speed
 * bound, or motion while the wheels report standing still, and dead-reckons on
 * wheel odometry for up to max_hold_duration before escalating to FAULT. A
 * failing update is discarded, not clipped. A rejected pose never becomes the
 * baseline the next check measures from.
 *
 * Free of ROS nodes (poses are tf2::Transform) so it unit-tests without a spin
 * (test/test_lio_odom_guard.cpp).
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

#include <cstddef>
#include <deque>
#include <optional>
#include <string>

namespace pepper_slam
{

enum class GuardVerdict
{
  kAccept,  // plausible: publish it as-is
  kHold,    // rejected, but inside max_hold_duration: dead reckon
  kFault,   // rejected for longer than max_hold_duration
};

// "accept", "hold" or "fault", as reported on /diagnostics.
const char * toString(GuardVerdict verdict);

// Body-relative motion between two poses of the same child frame:
// child(from) -> child(to). Being purely relative, it can be applied to a pose
// in a different parent frame, which is what lets wheel odometry
// (pepper_odom -> base_footprint) carry a pose in odom although the two trees
// are deliberately disconnected; both parents are gravity-aligned.
tf2::Transform relativeMotion(const tf2::Transform & from, const tf2::Transform & to);

// Magnitude of a rotation's axis-angle, in radians.
double rotationAngle(const tf2::Transform & t);

struct GuardParams
{
  double max_linear_speed = 0.7;       // m/s the base cannot exceed
  double max_angular_speed = 1.0;      // rad/s
  double max_hold_duration = 3.0;      // s of rejection before FAULT
  double zero_velocity_eps = 0.02;     // m/s: wheels below this are stopped
  double zero_velocity_lio_eps = 0.10;  // m/s: LIO above this while stopped is wrong
  std::size_t rate_window = 200;       // verdicts the rejection rate covers
};

struct GuardResult
{
  GuardVerdict verdict;
  tf2::Transform pose;  // the candidate on accept; the dead-reckoned pose otherwise
};

class OdomGuard
{
public:
  explicit OdomGuard(const GuardParams & params = GuardParams());

  // Classify one LIO pose. `t` is the odometry message's stamp in seconds,
  // never wall time: under bag replay they differ by hours. `wheel_pose` is
  // the latest pepper_odom -> base_footprint and `wheel_speed` its linear
  // speed; either may be absent. On FAULT the returned pose is a best effort
  // the caller should stop trusting.
  GuardResult check(
    const tf2::Transform & odom_base, double t,
    const std::optional<tf2::Transform> & wheel_pose = std::nullopt,
    std::optional<double> wheel_speed = std::nullopt);

  bool holding() const {return hold_start_t_.has_value();}

  // Metres dead reckoned since the current hold began; 0 while accepting.
  // Path length, not displacement: wheel drift grows with ground covered, so a
  // there-and-back excursion has drifted although it ends where it started.
  double heldDistance() const {return held_distance_;}

  // Fraction of the recent window that was rejected (gate lockout signal).
  double rejectionRate() const;

  const std::string & lastReason() const {return last_reason_;}
  const GuardParams & params() const {return params_;}

private:
  // Why this pose breaks a physical bound, or empty if it does not.
  std::string implausible(
    const tf2::Transform & odom_base, double dt,
    std::optional<double> wheel_speed) const;
  GuardResult accept(const tf2::Transform & odom_base, double t);
  // P(t) = P_good * (W_0^-1 * W(t)); P_good alone without wheel odometry.
  tf2::Transform deadReckon(const std::optional<tf2::Transform> & wheel_pose);

  GuardParams params_;
  // Last ACCEPTED pose and its stamp: the anchor both the speed check and the
  // dead reckoning measure from. Never updated while rejecting.
  std::optional<tf2::Transform> last_good_;
  double last_good_t_ = 0.0;
  // Wheel pose when the rejection began (W_0), and the last one measured.
  std::optional<tf2::Transform> hold_wheel_origin_;
  std::optional<tf2::Transform> hold_wheel_last_;
  std::optional<double> hold_start_t_;
  double held_distance_ = 0.0;
  std::deque<int> verdicts_;  // 1 = rejected
  std::string last_reason_;
};

}  // namespace pepper_slam
