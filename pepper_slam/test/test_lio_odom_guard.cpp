/* test_lio_odom_guard.cpp
 *
 * The divergence guard must reject implausible poses rather than clip them.
 * Each property below is one the guard exists for, with a real failure behind
 * it:
 *
 *   - a pose that moves faster than the base physically can is a broken
 *     estimate, not a fast robot, and must not reach odom -> base_footprint;
 *   - while rejecting, the pose must keep MOVING on wheel odometry; freezing it
 *     makes obstacles stream past a robot that reports standing still;
 *   - coasting must be time bounded, or a permanent fault decays into
 *     unbounded silent dead reckoning;
 *   - a rejected excursion must never become the baseline the next check
 *     measures against, or one jump ratchets the guard along with it.
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

#include <gtest/gtest.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

#include <string>

#include "pepper_slam/lio_odom_guard.h"

using pepper_slam::GuardParams;
using pepper_slam::GuardVerdict;
using pepper_slam::OdomGuard;

namespace
{

// A pose with a planar translation and a yaw.
tf2::Transform pose(double x = 0.0, double y = 0.0, double yaw = 0.0)
{
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw);
  return tf2::Transform(q, tf2::Vector3(x, y, 0.0));
}

OdomGuard guard(double max_linear_speed = 0.7, double max_hold_duration = 3.0)
{
  GuardParams p;
  p.max_linear_speed = max_linear_speed;
  p.max_angular_speed = 1.0;
  p.max_hold_duration = max_hold_duration;
  return OdomGuard(p);
}

void expectAt(const tf2::Transform & t, double x, double y = 0.0)
{
  EXPECT_NEAR(t.getOrigin().x(), x, 1e-9);
  EXPECT_NEAR(t.getOrigin().y(), y, 1e-9);
  EXPECT_NEAR(t.getOrigin().z(), 0.0, 1e-9);
}

bool contains(const std::string & text, const char * part)
{
  return text.find(part) != std::string::npos;
}

}  // namespace

TEST(LioOdomGuard, FirstPoseIsAcceptedWithoutADelta)
{
  auto g = guard();
  const auto r = g.check(pose(1.0, 2.0), 10.0);
  EXPECT_EQ(r.verdict, GuardVerdict::kAccept);
  expectAt(r.pose, 1.0, 2.0);
}

TEST(LioOdomGuard, NormalMotionPasses)
{
  auto g = guard();
  g.check(pose(0.0), 0.0);
  // 0.3 m/s, inside Nav2's own command cap.
  EXPECT_EQ(g.check(pose(0.03), 0.1).verdict, GuardVerdict::kAccept);
}

TEST(LioOdomGuard, JumpBeyondPhysicalBoundIsRejected)
{
  auto g = guard();
  g.check(pose(0.0), 0.0);
  // 5 m in 100 ms = 50 m/s: Pepper cannot do this whatever the filter believes.
  EXPECT_EQ(g.check(pose(5.0), 0.1).verdict, GuardVerdict::kHold);
  EXPECT_TRUE(contains(g.lastReason(), "exceeds"));
}

TEST(LioOdomGuard, RejectedPoseIsNotClippedToTheLimit)
{
  auto g = guard();
  g.check(pose(0.0), 0.0);
  // Without wheel odometry the held pose is the last good one, not the jump
  // scaled back to max_linear_speed * dt (which would be 0.07 m).
  expectAt(g.check(pose(5.0), 0.1).pose, 0.0);
}

TEST(LioOdomGuard, FastRotationIsRejected)
{
  auto g = guard();
  g.check(pose(0.0, 0.0, 0.0), 0.0);
  EXPECT_EQ(g.check(pose(0.0, 0.0, 1.5), 0.1).verdict, GuardVerdict::kHold);  // 15 rad/s
  EXPECT_TRUE(contains(g.lastReason(), "angular"));
}

TEST(LioOdomGuard, ZeroVelocityCrossCheckConvictsLio)
{
  auto g = guard(10.0);  // speed bound deliberately out of the way
  g.check(pose(0.0), 0.0, std::nullopt, 0.0);
  // 0.5 m/s while the wheels say stopped.
  EXPECT_EQ(g.check(pose(0.05), 0.1, std::nullopt, 0.0).verdict, GuardVerdict::kHold);
  EXPECT_TRUE(contains(g.lastReason(), "stationary"));
}

TEST(LioOdomGuard, HoldDeadReckonsOnWheelOdometry)
{
  auto g = guard();
  g.check(pose(1.0), 0.0, pose(10.0));
  // The LIO jumps; the wheels advance 0.04 m over the same window.
  expectAt(g.check(pose(9.0), 0.1, pose(10.0)).pose, 1.0);
  // Anchored at 1.0 and carried forward by the wheel delta, not the jump.
  expectAt(g.check(pose(9.5), 0.2, pose(10.04)).pose, 1.04);
}

TEST(LioOdomGuard, DeadReckoningIsFrameRelativeNotAbsolute)
{
  // Wheel poses live in pepper_odom; only their DELTA may be borrowed.
  auto g = guard();
  g.check(pose(0.0), 0.0, pose(100.0, 50.0));  // far from the LIO origin
  g.check(pose(9.0), 0.1, pose(100.1, 50.0));  // the hold starts here
  // Only the 0.05 m travelled since the hold began; the 100 m offset and the
  // motion before the hold must both stay out.
  expectAt(g.check(pose(9.5), 0.2, pose(100.15, 50.0)).pose, 0.05);
}

TEST(LioOdomGuard, HoldEscalatesToFaultAfterMaxHoldDuration)
{
  auto g = guard(0.7, 1.0);
  g.check(pose(0.0), 0.0, pose(0.0));
  EXPECT_EQ(g.check(pose(9.0), 0.1, pose(0.0)).verdict, GuardVerdict::kHold);
  EXPECT_EQ(g.check(pose(9.1), 0.9, pose(0.0)).verdict, GuardVerdict::kHold);
  // Past the bound, coasting must stop being reported as merely degraded.
  EXPECT_EQ(g.check(pose(9.2), 1.5, pose(0.0)).verdict, GuardVerdict::kFault);
}

TEST(LioOdomGuard, RejectedPoseNeverBecomesTheNewBaseline)
{
  // Otherwise one jump ratchets the anchor and every later check passes.
  auto g = guard();
  g.check(pose(0.0), 0.0);
  g.check(pose(50.0), 0.1);
  // Measured from the last GOOD pose (0.0), this is still a 50 m excursion.
  EXPECT_EQ(g.check(pose(50.01), 0.2).verdict, GuardVerdict::kHold);
}

TEST(LioOdomGuard, GuardReleasesWhenTheEstimatorRecovers)
{
  auto g = guard();
  g.check(pose(0.0), 0.0, pose(0.0));
  EXPECT_EQ(g.check(pose(9.0), 0.1, pose(0.0)).verdict, GuardVerdict::kHold);
  EXPECT_TRUE(g.holding());
  // A plausible pose relative to the last good one clears the hold.
  const auto r = g.check(pose(0.02), 0.2, pose(0.02));
  EXPECT_EQ(r.verdict, GuardVerdict::kAccept);
  EXPECT_FALSE(g.holding());
  expectAt(r.pose, 0.02);
}

TEST(LioOdomGuard, HeldDistanceIsZeroWhileAccepting)
{
  auto g = guard();
  g.check(pose(0.0), 0.0, pose(0.0));
  g.check(pose(0.02), 0.1, pose(0.02));
  EXPECT_EQ(g.heldDistance(), 0.0);
}

TEST(LioOdomGuard, HeldDistanceAccumulatesWhileHolding)
{
  auto g = guard();
  g.check(pose(0.0), 0.0, pose(0.0));
  g.check(pose(9.0), 0.1, pose(0.0));  // the hold starts here
  g.check(pose(9.1), 0.2, pose(0.05));
  g.check(pose(9.2), 0.3, pose(0.12));
  EXPECT_NEAR(g.heldDistance(), 0.12, 1e-9);
}

TEST(LioOdomGuard, HeldDistanceIsPathLengthNotDisplacement)
{
  // Drift accrues with ground covered, so there-and-back still counts.
  auto g = guard();
  g.check(pose(0.0), 0.0, pose(0.0));
  g.check(pose(9.0), 0.1, pose(0.0));   // the hold starts here
  g.check(pose(9.1), 0.2, pose(0.30));  // out 0.30
  g.check(pose(9.2), 0.3, pose(0.0));   // and back 0.30
  EXPECT_NEAR(g.heldDistance(), 0.60, 1e-9);
}

TEST(LioOdomGuard, HeldDistanceResetsWhenTheGuardReleases)
{
  auto g = guard();
  g.check(pose(0.0), 0.0, pose(0.0));
  g.check(pose(9.0), 0.1, pose(0.0));
  g.check(pose(9.1), 0.2, pose(0.10));
  EXPECT_GT(g.heldDistance(), 0.0);
  g.check(pose(0.01), 0.3, pose(0.11));  // accepted again
  EXPECT_EQ(g.heldDistance(), 0.0);
}

TEST(LioOdomGuard, StationaryHoldAccruesNoDistance)
{
  // Dead reckoning a stopped robot is exact, so the pose is still good.
  auto g = guard(10.0);
  g.check(pose(0.0), 0.0, pose(0.0), 0.0);
  for (int i = 1; i < 10; ++i) {
    g.check(pose(0.5), 0.1 * i, pose(0.0), 0.0);
  }
  EXPECT_EQ(g.heldDistance(), 0.0);
}

TEST(LioOdomGuard, RejectionRateTracksGateLockout)
{
  auto g = guard();
  g.check(pose(0.0), 0.0);
  EXPECT_EQ(g.rejectionRate(), 0.0);
  for (int i = 1; i <= 20; ++i) {
    g.check(pose(50.0 + i), 0.1 * i);
  }
  // Every update after the first was rejected: the thresholds are the suspect.
  EXPECT_GT(g.rejectionRate(), 0.9);
}

TEST(LioOdomGuard, NonMonotonicStampIsIgnoredNotRejected)
{
  // Out-of-order stamps are a transport artefact, not a divergence.
  auto g = guard();
  g.check(pose(0.0), 1.0);
  EXPECT_EQ(g.check(pose(0.01), 0.9).verdict, GuardVerdict::kAccept);
  EXPECT_TRUE(contains(g.lastReason(), "non-monotonic"));
}

TEST(LioOdomGuard, RotationAngleOfAKnownRotation)
{
  EXPECT_NEAR(pepper_slam::rotationAngle(pose(0.0, 0.0, 0.5)), 0.5, 1e-9);
  EXPECT_NEAR(pepper_slam::rotationAngle(pose(1.0, 2.0, 0.0)), 0.0, 1e-9);
}
