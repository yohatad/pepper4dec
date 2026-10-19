// Latch and direction logic for bumper_guard. Time is passed in, so every
// case is exact rather than slept through.

#include <gtest/gtest.h>

#include "pepper_navigation/bumper_latch.h"

using pepper_navigation::BumperLatch;
using pepper_navigation::kBack;
using pepper_navigation::kLeft;
using pepper_navigation::kRight;

namespace
{
BumperLatch fresh() {return BumperLatch(1.0, 10.0);}
}  // namespace

TEST(BumperLatch, IdleBlocksNothing)
{
  auto l = fresh();
  const auto c = l.clamp(0.0);
  EXPECT_FALSE(c.block_forward);
  EXPECT_FALSE(c.block_reverse);
}

TEST(BumperLatch, FrontPressBlocksForwardOnly)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  const auto c = l.clamp(0.0);
  EXPECT_TRUE(c.block_forward);
  // Reverse must stay open or BackUp cannot drive off the obstacle.
  EXPECT_FALSE(c.block_reverse);
}

TEST(BumperLatch, BackPressBlocksReverseOnly)
{
  auto l = fresh();
  l.set(kBack, true, 0.0);
  const auto c = l.clamp(0.0);
  EXPECT_FALSE(c.block_forward);
  EXPECT_TRUE(c.block_reverse);
}

TEST(BumperLatch, FrontAndBackBlockBoth)
{
  auto l = fresh();
  l.set(kRight, true, 0.0);
  l.set(kBack, true, 0.0);
  const auto c = l.clamp(0.0);
  EXPECT_TRUE(c.block_forward);
  EXPECT_TRUE(c.block_reverse);
}

TEST(BumperLatch, EitherFrontBumperBlocksForward)
{
  auto l = fresh();
  l.set(kRight, true, 0.0);
  EXPECT_TRUE(l.clamp(0.0).block_forward);
  auto r = fresh();
  r.set(kLeft, true, 0.0);
  EXPECT_TRUE(r.clamp(0.0).block_forward);
}

TEST(BumperLatch, LatchOutlivesTheRelease)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  l.set(kLeft, false, 0.1);           // a rebound releases the switch at once
  EXPECT_TRUE(l.clamp(0.5).block_forward);   // still latched inside clear_delay
  EXPECT_TRUE(l.clamp(1.0).block_forward);   // boundary is exclusive
  EXPECT_FALSE(l.clamp(1.2).block_forward);  // 1.0 s after the release
}

TEST(BumperLatch, HeldSwitchNeverAgesOut)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  EXPECT_TRUE(l.clamp(600.0).block_forward);
}

TEST(BumperLatch, ClearDropsTheLatchTail)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  l.set(kLeft, false, 0.1);
  EXPECT_TRUE(l.clear());
  EXPECT_FALSE(l.clamp(0.2).block_forward);
}

TEST(BumperLatch, ClearRefusedWhileStillPressed)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  EXPECT_FALSE(l.clear());
  EXPECT_TRUE(l.clamp(0.1).block_forward);
}

TEST(BumperLatch, BlindHoldsTheLatchInsteadOfAgeingIt)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  l.set(kLeft, false, 0.1);
  l.set_blind(true);
  // Without the driver alive, /bumper silence proves nothing, so the latch
  // must not expire on schedule.
  EXPECT_TRUE(l.clamp(99.0).block_forward);
  l.set_blind(false);
  EXPECT_FALSE(l.clamp(99.0).block_forward);
}

TEST(BumperLatch, BlindDoesNotInventALatch)
{
  auto l = fresh();
  l.set_blind(true);
  const auto c = l.clamp(99.0);
  EXPECT_FALSE(c.block_forward);
  EXPECT_FALSE(c.block_reverse);
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
  const auto c = l.clamp(0.0);
  EXPECT_FALSE(c.block_forward);
  EXPECT_FALSE(c.block_reverse);
  EXPECT_FALSE(l.latched(7, 0.0));
}

TEST(BumperLatch, ReleaseWithoutPressDoesNotLatch)
{
  auto l = fresh();
  l.set(kLeft, false, 0.0);   // a release event we never saw the press for
  EXPECT_FALSE(l.clamp(0.0).block_forward);
}

TEST(BumperLatch, RePressAfterExpiryLatchesAgain)
{
  auto l = fresh();
  l.set(kLeft, true, 0.0);
  l.set(kLeft, false, 0.1);
  ASSERT_FALSE(l.clamp(1.2).block_forward);
  l.set(kLeft, true, 2.0);
  EXPECT_TRUE(l.clamp(2.0).block_forward);
}
