/* test_animate_behavior_motion.cpp
 *
 * Unit tests for the pure motion math in animate_behavior_motion.cpp: joint
 * soft-limit clamping, randomized gesture target computation, and the
 * exponential smoothing applied before joint angles are published.
 *
 * No ROS runtime, no robot, no RNG — the caller supplies the random value, so
 * every case here is deterministic.
 *
 * Expected values come from three sources, noted per test:
 *   (a) hand-derived arithmetic,
 *   (b) properties that must hold for any input (bounds, monotonicity,
 *       convergence, symmetry),
 *   (c) documented behaviour at degenerate inputs.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Aug 16, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include <gtest/gtest.h>

#include <cmath>

#include "animate_behavior/animate_behavior_motion.h"

namespace motion = animate_behavior_motion;

//=============================================================================
// clampToLimits
//=============================================================================

TEST(ClampToLimits, PassesThroughValuesInsideTheLimits) {
    EXPECT_DOUBLE_EQ(motion::clampToLimits(0.0, -1.0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(motion::clampToLimits(0.5, -1.0, 1.0), 0.5);
    EXPECT_DOUBLE_EQ(motion::clampToLimits(-0.9, -1.0, 1.0), -0.9);
}

TEST(ClampToLimits, ClampsAtBothEnds) {
    EXPECT_DOUBLE_EQ(motion::clampToLimits(5.0, -1.0, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(motion::clampToLimits(-5.0, -1.0, 1.0), -1.0);
}

TEST(ClampToLimits, BoundariesAreInclusive) {
    EXPECT_DOUBLE_EQ(motion::clampToLimits(-1.0, -1.0, 1.0), -1.0);
    EXPECT_DOUBLE_EQ(motion::clampToLimits(1.0, -1.0, 1.0), 1.0);
}

// Pepper's joint ranges are asymmetric (e.g. LShoulderPitch is roughly
// -2.08..2.08 but LElbowRoll is -1.56..-0.01), so a negative-only range has
// to behave.
TEST(ClampToLimits, HandlesEntirelyNegativeRanges) {
    EXPECT_DOUBLE_EQ(motion::clampToLimits(0.0, -1.56, -0.01), -0.01);
    EXPECT_DOUBLE_EQ(motion::clampToLimits(-3.0, -1.56, -0.01), -1.56);
    EXPECT_DOUBLE_EQ(motion::clampToLimits(-0.5, -1.56, -0.01), -0.5);
}

// (c) A degenerate joint definition (min == max) pins the joint.
TEST(ClampToLimits, EqualLimitsPinTheValue) {
    for (double value : {-10.0, 0.0, 0.25, 10.0}) {
        EXPECT_DOUBLE_EQ(motion::clampToLimits(value, 0.25, 0.25), 0.25);
    }
}

// (c) Documented behaviour for a malformed joint definition: with min > max
// the lower bound wins. The point is that the result is still one of the two
// supplied limits rather than an unbounded target reaching the robot.
TEST(ClampToLimits, InvertedLimitsFallBackToTheLowerBound) {
    EXPECT_DOUBLE_EQ(motion::clampToLimits(0.0, 1.0, -1.0), 1.0);
    EXPECT_DOUBLE_EQ(motion::clampToLimits(99.0, 1.0, -1.0), 1.0);
}

// (b) Property: the result is always within the limits, for any input.
TEST(ClampToLimits, ResultAlwaysWithinLimits) {
    const double min_limit = -1.2, max_limit = 0.8;
    for (double value = -5.0; value <= 5.0; value += 0.1) {
        const double clamped = motion::clampToLimits(value, min_limit, max_limit);
        EXPECT_GE(clamped, min_limit) << "value=" << value;
        EXPECT_LE(clamped, max_limit) << "value=" << value;
    }
}

//=============================================================================
// gestureTarget
//=============================================================================

// (a) Hand-derived: home 0.5, noise 1.0, range 0.2, factor 0.5
//     -> 0.5 + 1.0 * 0.2 * 0.5 = 0.6, comfortably inside the limits.
TEST(GestureTarget, ComputesHomePlusScaledNoise) {
    EXPECT_DOUBLE_EQ(
        motion::gestureTarget(0.5, 1.0, 0.2, 0.5, -2.0, 2.0), 0.6);
    EXPECT_DOUBLE_EQ(
        motion::gestureTarget(0.5, -1.0, 0.2, 0.5, -2.0, 2.0), 0.4);
}

// Zero noise means "sit at home" — the midpoint every gesture oscillates about.
TEST(GestureTarget, ZeroNoiseReturnsHome) {
    EXPECT_DOUBLE_EQ(motion::gestureTarget(0.3, 0.0, 0.5, 1.0, -2.0, 2.0), 0.3);
}

// range 0 disables motion entirely regardless of noise — this is what an
// action goal with range 0 must do.
TEST(GestureTarget, ZeroRangeHoldsHome) {
    for (double noise : {-1.0, -0.4, 0.0, 0.7, 1.0}) {
        EXPECT_DOUBLE_EQ(motion::gestureTarget(0.3, noise, 0.0, 1.0, -2.0, 2.0), 0.3)
            << "noise=" << noise;
    }
}

// factor 0 excludes one joint from the gesture while its limb still animates.
TEST(GestureTarget, ZeroFactorHoldsHome) {
    for (double noise : {-1.0, 0.0, 1.0}) {
        EXPECT_DOUBLE_EQ(motion::gestureTarget(0.3, noise, 0.5, 0.0, -2.0, 2.0), 0.3)
            << "noise=" << noise;
    }
}

// (b) The safety property that matters most: no combination of noise, range
// and factor may drive a joint past its soft limits.
TEST(GestureTarget, NeverExceedsLimitsForAnyNoise) {
    const double min_limit = -0.5, max_limit = 0.5;
    // A range far larger than the limits allow, to force saturation.
    for (double noise = -1.0; noise <= 1.0; noise += 0.05) {
        const double target =
            motion::gestureTarget(0.0, noise, 10.0, 1.0, min_limit, max_limit);
        EXPECT_GE(target, min_limit) << "noise=" << noise;
        EXPECT_LE(target, max_limit) << "noise=" << noise;
    }
}

// (b) A home pose already outside the limits is pulled back inside rather than
// passed through — a bad joint definition cannot command an illegal angle.
TEST(GestureTarget, ClampsEvenWhenHomeIsOutOfRange) {
    EXPECT_DOUBLE_EQ(motion::gestureTarget(5.0, 0.0, 0.2, 1.0, -1.0, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(motion::gestureTarget(-5.0, 0.0, 0.2, 1.0, -1.0, 1.0), -1.0);
}

// (b) Symmetry: opposite noise values sit equidistant either side of home,
// so the gesture is unbiased when it is not saturating.
TEST(GestureTarget, IsSymmetricAboutHomeWhenUnclamped) {
    const double home = 0.1;
    for (double noise : {0.2, 0.5, 0.9}) {
        const double high = motion::gestureTarget(home, noise, 0.4, 1.0, -2.0, 2.0);
        const double low = motion::gestureTarget(home, -noise, 0.4, 1.0, -2.0, 2.0);
        EXPECT_NEAR(high - home, home - low, 1e-12) << "noise=" << noise;
    }
}

// (b) Monotonic in noise: larger noise never produces a smaller target.
TEST(GestureTarget, IsMonotonicInNoise) {
    double previous = -1e9;
    for (double noise = -1.0; noise <= 1.0; noise += 0.05) {
        const double target =
            motion::gestureTarget(0.0, noise, 0.5, 1.0, -2.0, 2.0);
        EXPECT_GE(target, previous) << "noise=" << noise;
        previous = target;
    }
}

// A negative factor mirrors that joint's motion — used to make paired joints
// (e.g. left/right shoulders) move in opposition rather than in lockstep.
TEST(GestureTarget, NegativeFactorMirrorsTheMotion) {
    const double positive = motion::gestureTarget(0.0, 1.0, 0.3, 1.0, -2.0, 2.0);
    const double mirrored = motion::gestureTarget(0.0, 1.0, 0.3, -1.0, -2.0, 2.0);
    EXPECT_DOUBLE_EQ(mirrored, -positive);
}

//=============================================================================
// smoothToward
//=============================================================================

// (a) Hand-derived: 0 + 0.15 * (1 - 0) = 0.15, the package's default factor.
TEST(SmoothToward, TakesAFractionalStepTowardTheTarget) {
    EXPECT_DOUBLE_EQ(motion::smoothToward(0.0, 1.0, 0.15), 0.15);
    EXPECT_DOUBLE_EQ(motion::smoothToward(0.0, 1.0, 0.5), 0.5);
}

// (c) The two documented endpoints of the factor.
TEST(SmoothToward, FactorZeroHoldsAndFactorOneJumps) {
    EXPECT_DOUBLE_EQ(motion::smoothToward(0.2, 0.9, 0.0), 0.2);
    EXPECT_DOUBLE_EQ(motion::smoothToward(0.2, 0.9, 1.0), 0.9);
}

TEST(SmoothToward, AlreadyAtTargetDoesNotMove) {
    for (double factor : {0.0, 0.15, 0.5, 1.0}) {
        EXPECT_DOUBLE_EQ(motion::smoothToward(0.42, 0.42, factor), 0.42)
            << "factor=" << factor;
    }
}

// (b) Property: a smoothed step never overshoots and never moves away — the
// output stays between where the joint is and where it is going.
TEST(SmoothToward, StaysBetweenCurrentAndTarget) {
    for (double factor = 0.0; factor <= 1.0; factor += 0.05) {
        const double forward = motion::smoothToward(-1.0, 2.0, factor);
        EXPECT_GE(forward, -1.0) << "factor=" << factor;
        EXPECT_LE(forward, 2.0) << "factor=" << factor;

        // Same in the descending direction.
        const double backward = motion::smoothToward(2.0, -1.0, factor);
        EXPECT_GE(backward, -1.0) << "factor=" << factor;
        EXPECT_LE(backward, 2.0) << "factor=" << factor;
    }
}

// (b) Property: repeated application converges on the target. This is what
// makes the joint stream settle instead of oscillating or stalling short.
TEST(SmoothToward, ConvergesOnTheTarget) {
    const double target = 1.0;
    double current = 0.0;
    for (int step = 0; step < 500; ++step) {
        current = motion::smoothToward(current, target, 0.15);
    }
    EXPECT_NEAR(current, target, 1e-9);
}

// (b) Property: each step strictly reduces the remaining distance for any
// factor in (0, 1) — no stalling partway.
TEST(SmoothToward, MonotonicallyReducesDistance) {
    const double target = 1.0;
    double current = 0.0;
    double previous_distance = std::fabs(target - current);
    for (int step = 0; step < 50; ++step) {
        current = motion::smoothToward(current, target, 0.15);
        const double distance = std::fabs(target - current);
        EXPECT_LT(distance, previous_distance) << "step=" << step;
        previous_distance = distance;
    }
}

// Works the same approaching from above, including across zero.
TEST(SmoothToward, WorksDescendingAndAcrossZero) {
    EXPECT_DOUBLE_EQ(motion::smoothToward(1.0, 0.0, 0.25), 0.75);
    EXPECT_DOUBLE_EQ(motion::smoothToward(-1.0, 1.0, 0.5), 0.0);
}

// The first published value for a joint uses current == target (the node seeds
// it that way when no /joint_states feedback has arrived yet), which must be a
// no-op rather than a jump from zero.
TEST(SmoothToward, SeedingWithCurrentEqualToTargetIsANoOp) {
    const double seeded = 0.7;
    EXPECT_DOUBLE_EQ(motion::smoothToward(seeded, seeded, 0.15), seeded);
}
