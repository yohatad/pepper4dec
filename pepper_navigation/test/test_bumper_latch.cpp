// Latch logic for bumper_guard: any press is a full stop until clear().
// Time is passed in, so every case is exact rather than slept through.

#include <gtest/gtest.h>

#include "pepper_navigation/bumper_latch.h"

using pepper_navigation::BumperLatch;
using pepper_navigation::kBack;
using pepper_navigation::kLeft;
using pepper_navigation::kRight;

namespace
{
BumperLatch fresh() {return BumperLatch(10.0);}
}  // namespace

TEST(BumperLatch, IdleStopsNothing)
{
  auto l = fresh();
  EXPECT_FALSE(l.stop());
  EXPECT_FALSE(l.latched(kRight));
}

TEST(BumperLatch, AnyBumperStops)
{
  for (int id : {kRight, kLeft, kBack}) {
    auto l = fresh();
    l.set(id, true, 0.0);
    EXPECT_TRUE(l.stop()) << "bumper " << id;
    EXPECT_TRUE(l.latched(id));
  }
}

TEST(BumperLatch, ReleaseDoesNotLiftTheStop)
{
  auto l = fresh();
  l.set(kRight, true, 0.0);
  l.set(kRight, false, 0.1);
  EXPECT_TRUE(l.stop());                // no time-based release any more
  EXPECT_TRUE(l.latched(kRight));
  EXPECT_FALSE(l.pressed(kRight));
}

TEST(BumperLatch, BounceKeepsTheStop)
{
  auto l = fresh();
  l.set(kBack, true, 0.0);
  l.set(kBack, false, 0.30);
  l.set(kBack, true, 0.35);             // ~50 ms bounce, seen on the robot
  l.set(kBack, false, 0.55);
  EXPECT_TRUE(l.stop());
}

TEST(BumperLatch, ClearRefusedWhileStillPressed)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  EXPECT_FALSE(l.clear());
  EXPECT_TRUE(l.stop());
}

TEST(BumperLatch, ClearRefusedWhileAnotherBumperIsPressed)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  l.set(kLeft, false, 0.1);
  l.set(kBack, true, 0.2);
  EXPECT_FALSE(l.clear());
  EXPECT_TRUE(l.stop());
}

TEST(BumperLatch, ClearAfterReleaseLiftsTheStop)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  l.set(kLeft, false, 0.1);
  EXPECT_TRUE(l.clear());
  EXPECT_FALSE(l.stop());
  EXPECT_FALSE(l.latched(kLeft));
}

TEST(BumperLatch, PressAfterClearStopsAgain)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  l.set(kLeft, false, 0.1);
  ASSERT_TRUE(l.clear());
  l.set(kRight, true, 5.0);
  EXPECT_TRUE(l.stop());
}

TEST(BumperLatch, BlindDoesNotInventOrLiftAStop)
{
  auto l = fresh();
  l.set_blind(true);
  EXPECT_FALSE(l.stop());               // blind alone is not a contact
  l.set(kRight, true, 0.0);
  l.set(kRight, false, 0.1);
  l.set_blind(false);
  EXPECT_TRUE(l.stop());                // nor does regaining liveness clear it
}

TEST(BumperLatch, StuckOnlyAfterTheTimeout)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  EXPECT_FALSE(l.stuck(5.0));
  EXPECT_TRUE(l.stuck(10.1));
}

TEST(BumperLatch, ReleasedSwitchIsNotStuck)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  l.set(kLeft, false, 0.2);
  EXPECT_FALSE(l.stuck(600.0));
}

TEST(BumperLatch, RepeatedPressDoesNotRestartTheStuckClock)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  l.set(kLeft, true, 9.0);    // a duplicate press event, switch never released
  EXPECT_TRUE(l.stuck(10.1));
}

TEST(BumperLatch, OutOfRangeIdIsIgnored)
{
  auto l = fresh();
  l.set(7, true, 0.0);        // the id comes off the wire; do not trust it
  l.set(-1, true, 0.0);
  EXPECT_FALSE(l.stop());
  EXPECT_FALSE(l.latched(7));
}

TEST(BumperLatch, ReleaseWithoutPressDoesNotLatch)
{
  auto l = fresh();
  l.set(kLeft, false, 0.0);   // a release event we never saw the press for
  EXPECT_FALSE(l.stop());
}
