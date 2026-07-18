/* test_pepper_kinematics.cpp
 *
 * Unit tests for the pure forward/inverse kinematics helpers in
 * pepper_kinematics_utilities.cpp. No ROS runtime needed — everything here
 * is numbers in, numbers out.
 *
 * Expected values come from three sources, noted per test:
 *   (a) hand-derived geometry at simple poses (e.g. both shoulder angles 0),
 *   (b) mathematical properties that must hold (mirror symmetry, round trips),
 *   (c) documented clamping behavior at joint limits.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 18, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "gesture_execution/pepper_kinematics_utilities.h"

namespace pk = pepper_kinematics;

TEST(AngleConversions, KnownValuesAndRoundTrip) {
    EXPECT_NEAR(pk::degreesToRadians(0.0), 0.0, 1e-12);
    EXPECT_NEAR(pk::degreesToRadians(180.0), M_PI, 1e-12);
    EXPECT_NEAR(pk::degreesToRadians(-90.0), -M_PI / 2.0, 1e-12);
    EXPECT_NEAR(pk::radiansToDegrees(M_PI), 180.0, 1e-12);
    for (double deg : {-720.0, -33.3, 0.0, 9.0, 123.456, 1080.0}) {
        EXPECT_NEAR(pk::radiansToDegrees(pk::degreesToRadians(deg)), deg, 1e-9);
    }
}

// (a) Hand-derived: with theta_1 = theta_2 = 0 the trig collapses and the
// elbow position reduces to the link offsets alone:
//   x = l_1 + l_4          = -57.0 + 181.2        = 124.2
//   y = l_2 + l_5          = ±(149.74 + 15.0)     = ±164.74
//   z = l_3 + l_6          = 86.82 + 0.13         = 86.95
// (l_6 really is 0.13 — the meters-vs-mm quirk inherited from the Python
// reference; see the constants block in pepper_kinematics_utilities.cpp.)
TEST(ElbowPosition, ZeroPoseMatchesLinkOffsets) {
    auto left = pk::getElbowPosition(pk::LEFT_ARM, 0.0, 0.0);
    EXPECT_NEAR(left[0], 124.2, 1e-9);
    EXPECT_NEAR(left[1], 164.74, 1e-9);
    EXPECT_NEAR(left[2], 86.95, 1e-9);

    auto right = pk::getElbowPosition(pk::RIGHT_ARM, 0.0, 0.0);
    EXPECT_NEAR(right[0], 124.2, 1e-9);
    EXPECT_NEAR(right[1], -164.74, 1e-9);
    EXPECT_NEAR(right[2], 86.95, 1e-9);
}

// (b) Property: the arms are mirror images across the x-z plane, so the right
// arm at (theta_1, -theta_2) must land at the y-negated position of the left
// arm at (theta_1, theta_2).
TEST(ElbowPosition, LeftRightMirrorSymmetry) {
    for (double t1 : {-1.2, 0.0, 0.3, 1.5}) {
        for (double t2 : {0.2, 0.8, 1.4}) {
            auto left = pk::getElbowPosition(pk::LEFT_ARM, t1, t2);
            auto right = pk::getElbowPosition(pk::RIGHT_ARM, t1, -t2);
            EXPECT_NEAR(right[0], left[0], 1e-9) << "t1=" << t1 << " t2=" << t2;
            EXPECT_NEAR(right[1], -left[1], 1e-9) << "t1=" << t1 << " t2=" << t2;
            EXPECT_NEAR(right[2], left[2], 1e-9) << "t1=" << t1 << " t2=" << t2;
        }
    }
}

// The reference implementation defines the wrist position with the identical
// formulation as the elbow; this pins that (documented) equivalence.
TEST(WristPosition, MatchesElbowFormulation) {
    auto elbow = pk::getElbowPosition(pk::LEFT_ARM, 0.7, 0.9);
    auto wrist = pk::getWristPosition(pk::LEFT_ARM, 0.7, 0.9);
    EXPECT_DOUBLE_EQ(wrist[0], elbow[0]);
    EXPECT_DOUBLE_EQ(wrist[1], elbow[1]);
    EXPECT_DOUBLE_EQ(wrist[2], elbow[2]);
}

// (b) Round trip: forward kinematics to a position, inverse kinematics back —
// the recovered angles must reproduce the same elbow position. Comparing
// positions (not angles) keeps the test robust to multiple valid IK solutions.
// Angles are chosen strictly inside the joint limits enforced by
// getArmShoulderAngles (pitch within ±2.1, roll within ±[0.0087, 1.58]).
TEST(ShoulderAngles, ForwardInverseRoundTrip) {
    struct Case { int arm; double t1; double t2; };
    const Case cases[] = {
        {pk::LEFT_ARM, -1.0, 0.1}, {pk::LEFT_ARM, 0.0, 0.6},
        {pk::LEFT_ARM, 0.9, 1.2},  {pk::LEFT_ARM, 1.8, 0.3},
        {pk::RIGHT_ARM, -1.0, -0.1}, {pk::RIGHT_ARM, 0.0, -0.6},
        {pk::RIGHT_ARM, 0.9, -1.2},  {pk::RIGHT_ARM, 1.8, -0.3},
    };
    for (const auto& c : cases) {
        auto target = pk::getElbowPosition(c.arm, c.t1, c.t2);
        auto angles = pk::getArmShoulderAngles(c.arm, target[0], target[1], target[2]);
        ASSERT_FALSE(std::isnan(angles[0]))
            << "arm=" << c.arm << " t1=" << c.t1 << " t2=" << c.t2;
        auto reached = pk::getElbowPosition(c.arm, angles[0], angles[1]);
        EXPECT_NEAR(reached[0], target[0], 1e-6)
            << "arm=" << c.arm << " t1=" << c.t1 << " t2=" << c.t2;
        EXPECT_NEAR(reached[1], target[1], 1e-6)
            << "arm=" << c.arm << " t1=" << c.t1 << " t2=" << c.t2;
        EXPECT_NEAR(reached[2], target[2], 1e-6)
            << "arm=" << c.arm << " t1=" << c.t1 << " t2=" << c.t2;
    }
}

// (c) Clamp: a target that implies a negative shoulder roll on the LEFT arm
// (here: the pose the arm would reach with roll = -0.3, which only the right
// arm can do) must clamp the roll to the +0.0087 limit, not go negative.
TEST(ShoulderAngles, LeftRollClampsAtLowerLimit) {
    auto target = pk::getElbowPosition(pk::LEFT_ARM, 0.5, -0.3);
    auto angles = pk::getArmShoulderAngles(pk::LEFT_ARM, target[0], target[1], target[2]);
    EXPECT_NEAR(angles[1], 0.0087, 1e-9);
}

// (c) Clamp: a wrist far outside the reachable workspace drives the acos
// argument above 1 (NaN), which must clamp elbow roll to the near-zero limit
// for the respective arm rather than propagate NaN.
TEST(ElbowRoll, ClampsForUnreachableWrist) {
    double right = pk::getArmElbowRollAngle(pk::RIGHT_ARM, 0.5, -0.5, 1.0e6, 0.0, 0.0);
    EXPECT_NEAR(right, 0.0087, 1e-9);
    double left = pk::getArmElbowRollAngle(pk::LEFT_ARM, 0.5, 0.5, 1.0e6, 0.0, 0.0);
    EXPECT_NEAR(left, -0.0087, 1e-9);
}

// Pins the documented placeholder behavior: getArmAngles leaves elbow
// yaw/roll at 0.0 (mirrors the Python reference, where that computation is
// commented out). If someone implements it, this test failing is the prompt
// to update both the doc comment and the callers' assumptions.
TEST(ArmAngles, ElbowAnglesAreDocumentedPlaceholderZeros) {
    auto target = pk::getElbowPosition(pk::LEFT_ARM, 0.4, 0.7);
    auto angles = pk::getArmAngles(pk::LEFT_ARM, target[0], target[1], target[2],
                                   target[0] + 100.0, target[1], target[2]);
    EXPECT_DOUBLE_EQ(angles[2], 0.0);
    EXPECT_DOUBLE_EQ(angles[3], 0.0);
}

// (a) Hand-derived head angles:
//   yaw = atan2(y, x - l_1) with l_1 = -38, so camera at x=62, y=100 gives
//         atan2(100, 100) = pi/4.
//   pitch at camera_z = l_2 = 169.9 collapses the asin term to 0, leaving
//         atan(l_4 / l_3) = atan(61.6 / 93.6).
TEST(HeadAngles, KnownGeometry) {
    auto straight = pk::getHeadAngles(500.0, 0.0, 169.9);
    EXPECT_NEAR(straight[0], 0.0, 1e-12);
    EXPECT_NEAR(straight[1], std::atan(61.6 / 93.6), 1e-9);

    auto diagonal = pk::getHeadAngles(62.0, 100.0, 169.9);
    EXPECT_NEAR(diagonal[0], M_PI / 4.0, 1e-9);
}

// (c) Clamp: a camera behind the robot yields |yaw| > 2.1 -> forced to 0;
// a camera far below yields an asin argument > 1 (NaN pitch) -> forced to 0.
TEST(HeadAngles, OutOfRangeClampsToZero) {
    auto behind = pk::getHeadAngles(-500.0, 1.0, 169.9);
    EXPECT_DOUBLE_EQ(behind[0], 0.0);

    auto far_below = pk::getHeadAngles(500.0, 0.0, -500.0);
    EXPECT_DOUBLE_EQ(far_below[1], 0.0);
}
