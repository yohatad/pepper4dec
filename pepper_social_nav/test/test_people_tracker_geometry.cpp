// Copyright (C) 2026 Upanzi Network. BSD-3-Clause.
/**
 * Unit tests for the tracker's filter behaviour.
 *
 * The property that actually matters downstream is VELOCITY: the SFM critic
 * extrapolates people at constant velocity, so a track that reports the right
 * position but a wrong velocity steers the robot into the space someone is
 * about to occupy. These tests pin that down.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "pepper_social_nav/people_tracker.hpp"

using pepper_social_nav::Observation;
using pepper_social_nav::Track;

namespace
{

Observation makeObs(double x, double y, double t)
{
  Observation o;
  o.x = x;
  o.y = y;
  o.range = std::hypot(x, y);
  o.bearing = std::atan2(y, x);
  o.range_stddev = 0.12;
  o.cross_stddev = 0.04;
  o.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9), RCL_ROS_TIME);
  o.source = "test";
  return o;
}

/** A near-bearing-only observation, as Pepper's front camera produces close in. */
Observation makeBearingObs(double x, double y, double t)
{
  Observation o = makeObs(x, y, t);
  o.range_stddev = 1.20;   // range is a guess
  o.cross_stddev = 0.04;   // direction is not
  return o;
}

}  // namespace

/** A person walking at a steady 1 m/s must be reported as walking at 1 m/s. */
TEST(TrackTest, ConvergesToConstantVelocity)
{
  const double v = 1.0;
  const double dt = 0.1;

  Track track(1, makeObs(0.0, 0.0, 0.0), 1.0);
  for (int i = 1; i <= 30; ++i) {
    const double t = i * dt;
    track.predict(rclcpp::Time(static_cast<int64_t>(t * 1e9), RCL_ROS_TIME), 0.6);
    track.update(makeObs(v * t, 0.0, t));
  }

  EXPECT_NEAR(track.speed(), v, 0.12);
  EXPECT_NEAR(track.x(), v * 3.0, 0.15);
  EXPECT_NEAR(track.y(), 0.0, 0.1);
}

/** A standing person must not be reported as drifting. */
TEST(TrackTest, StationaryPersonHasNearZeroVelocity)
{
  const double dt = 0.1;
  Track track(1, makeObs(2.0, 1.0, 0.0), 1.0);
  for (int i = 1; i <= 30; ++i) {
    const double t = i * dt;
    track.predict(rclcpp::Time(static_cast<int64_t>(t * 1e9), RCL_ROS_TIME), 0.6);
    track.update(makeObs(2.0, 1.0, t));
  }
  EXPECT_LT(track.speed(), 0.15);
}

/**
 * stateAt must extrapolate forward without mutating the filter. This is what
 * compensates for offboard-detection latency at publish time, so it has to be
 * side-effect free -- otherwise every publish would corrupt the estimate.
 */
TEST(TrackTest, StateAtExtrapolatesWithoutMutating)
{
  const double dt = 0.1;
  Track track(1, makeObs(0.0, 0.0, 0.0), 1.0);
  for (int i = 1; i <= 30; ++i) {
    const double t = i * dt;
    track.predict(rclcpp::Time(static_cast<int64_t>(t * 1e9), RCL_ROS_TIME), 0.6);
    track.update(makeObs(1.0 * t, 0.0, t));
  }

  const double x_before = track.x();
  const auto future = track.stateAt(rclcpp::Time(static_cast<int64_t>(4.0 * 1e9), RCL_ROS_TIME));

  EXPECT_DOUBLE_EQ(track.x(), x_before) << "stateAt must not mutate the filter";
  EXPECT_GT(future(0), x_before) << "extrapolation should advance a moving track";
  EXPECT_NEAR(future(0), x_before + track.speed() * 1.0, 0.2);
}

/**
 * Two cameras with different pipeline latencies deliver measurements out of
 * order. That must be a no-op, not a backwards time jump.
 */
TEST(TrackTest, OutOfOrderPredictIsIgnored)
{
  Track track(1, makeObs(0.0, 0.0, 1.0), 1.0);
  const auto before = track.lastUpdate();
  track.predict(rclcpp::Time(static_cast<int64_t>(0.5 * 1e9), RCL_ROS_TIME), 0.6);
  EXPECT_EQ(track.lastUpdate().nanoseconds(), before.nanoseconds());
  EXPECT_TRUE(std::isfinite(track.x()));
  EXPECT_TRUE(std::isfinite(track.speed()));
}

/**
 * The point of the anisotropic update: ONE front-camera detection whose range
 * is a guess but whose bearing is good must correct the track's lateral error
 * almost fully while barely moving it along the line of sight.
 *
 * Asserted on a single update, because that is where the anisotropy lives. Over
 * many updates the filter legitimately converges on the range too -- the
 * property being pinned here is the per-update RATIO, which is what makes the
 * front camera useful close in. Collapse this back to an isotropic sigma and
 * the cross-range correction gets diluted to match the worst axis.
 */
TEST(TrackTest, BearingOnlyObservationCorrectsLateralNotRange)
{
  // Track believes the person is off to one side, at 3 m.
  Track track(1, makeObs(3.0, 0.6, 0.0), 1.0);

  // Camera sees them straight ahead (bearing correct), range overestimated.
  track.predict(rclcpp::Time(static_cast<int64_t>(0.1 * 1e9), RCL_ROS_TIME), 0.6);
  track.update(makeBearingObs(4.0, 0.0, 0.1));

  EXPECT_LT(std::fabs(track.y()), 0.1)
    << "an accurate bearing should correct nearly all of the lateral error";
  EXPECT_LT(track.x(), 3.5)
    << "a weak range measurement should be adopted only slowly";
  EXPECT_GT(track.x(), 3.0)
    << "...but still in the right direction";
}

/**
 * The same machinery must not make a confident depth measurement sluggish:
 * a RealSense observation should move the track essentially onto the reading.
 */
TEST(TrackTest, ConfidentObservationIsAdoptedQuickly)
{
  Track track(1, makeObs(3.0, 0.0, 0.0), 1.0);
  for (int i = 1; i <= 10; ++i) {
    const double t = i * 0.1;
    track.predict(rclcpp::Time(static_cast<int64_t>(t * 1e9), RCL_ROS_TIME), 0.6);
    Observation o = makeObs(3.0, 0.5, t);
    o.range_stddev = 0.10;
    o.cross_stddev = 0.04;
    track.update(o);
  }
  EXPECT_NEAR(track.y(), 0.5, 0.12);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
