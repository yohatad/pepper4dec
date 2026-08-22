/* test_overt_attention_utilities.cpp
 *
 * Unit tests for the shared helpers in overt_attention_utilities.cpp and the
 * inline helpers in overt_attention_interface.h: camera-variant topic
 * selection, compression-suffix handling, pixel-to-angle projection, clamping,
 * the image QoS profile, and the face-ID colour generator.
 *
 * All pure functions — no ROS graph, no camera, no node instantiation. The
 * one ROS type touched (rclcpp::QoS) is a plain value object that needs no
 * rclcpp::init().
 *
 * Expected values come from two sources, noted per test:
 *   (a) hand-derived arithmetic at simple inputs,
 *   (b) properties that must hold for any input (determinism, symmetry,
 *       documented ordering).
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
#include <functional>
#include <string>

#include "overt_attention/overt_attention_interface.h"

//=============================================================================
// selectCameraTopic
//=============================================================================

TEST(SelectCameraTopic, PicksTheRequestedVariant) {
    const std::string pepper = "/pepper/front/image_raw";
    const std::string realsense = "/camera/color/image_raw";
    EXPECT_EQ(selectCameraTopic("pepper", pepper, realsense), pepper);
    EXPECT_EQ(selectCameraTopic("realsense", pepper, realsense), realsense);
}

// The node constructors rely on this throwing rather than silently defaulting —
// a typo'd camera_type must fail loudly at startup, not subscribe to nothing.
TEST(SelectCameraTopic, ThrowsOnUnknownCameraType) {
    EXPECT_THROW(selectCameraTopic("webcam", "a", "b"), std::invalid_argument);
    EXPECT_THROW(selectCameraTopic("", "a", "b"), std::invalid_argument);
    // Case matters: the parameter is compared verbatim.
    EXPECT_THROW(selectCameraTopic("Pepper", "a", "b"), std::invalid_argument);
}

//=============================================================================
// getImageTopic
//=============================================================================

// (a) Hand-derived: colour images get "/compressed", depth images get
// "/compressedDepth" — image_transport's two different suffixes. Getting these
// swapped subscribes to a topic nothing publishes, which is silent at runtime.
TEST(GetImageTopic, AppendsTheCorrectCompressionSuffix) {
    const std::string base = "/camera/color/image_raw";
    EXPECT_EQ(getImageTopic(base, false), base);
    EXPECT_EQ(getImageTopic(base, true), base + "/compressed");

    const std::string depth_base = "/camera/depth/image_rect_raw";
    EXPECT_EQ(getImageTopic(depth_base, false, true), depth_base);
    EXPECT_EQ(getImageTopic(depth_base, true, true), depth_base + "/compressedDepth");
}

// is_depth must not matter when compression is off — the raw topic is the raw
// topic either way.
TEST(GetImageTopic, IsDepthIrrelevantWhenUncompressed) {
    const std::string base = "/some/topic";
    EXPECT_EQ(getImageTopic(base, false, false), getImageTopic(base, false, true));
}

//=============================================================================
// clamp
//=============================================================================

TEST(Clamp, BoundsAndPassthrough) {
    EXPECT_DOUBLE_EQ(clamp(0.5, 0.0, 1.0), 0.5);   // inside -> unchanged
    EXPECT_DOUBLE_EQ(clamp(-3.0, 0.0, 1.0), 0.0);  // below -> lo
    EXPECT_DOUBLE_EQ(clamp(7.0, 0.0, 1.0), 1.0);   // above -> hi
    EXPECT_DOUBLE_EQ(clamp(0.0, 0.0, 1.0), 0.0);   // on the boundary
    EXPECT_DOUBLE_EQ(clamp(1.0, 0.0, 1.0), 1.0);
}

// The head-limit clamps are asymmetric (e.g. pitch_up 0.4 / pitch_dn -0.7),
// so negative ranges must behave.
TEST(Clamp, NegativeRange) {
    EXPECT_DOUBLE_EQ(clamp(0.0, -0.7, 0.4), 0.0);
    EXPECT_DOUBLE_EQ(clamp(-1.5, -0.7, 0.4), -0.7);
    EXPECT_DOUBLE_EQ(clamp(1.5, -0.7, 0.4), 0.4);
}

//=============================================================================
// pixelToAngles
//=============================================================================

// (a) Hand-derived: a pixel exactly at the principal point projects to the
// optical axis, i.e. zero yaw and zero pitch.
TEST(PixelToAngles, PrincipalPointIsZeroZero) {
    auto [yaw, pitch] = pixelToAngles(320.0, 240.0, 500.0, 500.0, 320.0, 240.0);
    EXPECT_NEAR(yaw, 0.0, 1e-12);
    EXPECT_NEAR(pitch, 0.0, 1e-12);
}

// (a) Hand-derived: one focal length off-centre is exactly 45 degrees.
// Yaw is negated (image +x is to the right, robot +yaw is to the left), so a
// pixel right of centre must produce a NEGATIVE yaw. A sign flip here makes
// the robot turn away from whatever it is looking at.
TEST(PixelToAngles, OneFocalLengthOffCentreIs45Degrees) {
    const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;

    auto [yaw_right, pitch_right] = pixelToAngles(cx + fx, cy, fx, fy, cx, cy);
    EXPECT_NEAR(yaw_right, -M_PI / 4.0, 1e-12);
    EXPECT_NEAR(pitch_right, 0.0, 1e-12);

    auto [yaw_left, pitch_left] = pixelToAngles(cx - fx, cy, fx, fy, cx, cy);
    EXPECT_NEAR(yaw_left, M_PI / 4.0, 1e-12);
    EXPECT_NEAR(pitch_left, 0.0, 1e-12);

    // Image +y is downward; pitch is NOT negated, so below-centre is +pitch.
    auto [yaw_down, pitch_down] = pixelToAngles(cx, cy + fy, fx, fy, cx, cy);
    EXPECT_NEAR(yaw_down, 0.0, 1e-12);
    EXPECT_NEAR(pitch_down, M_PI / 4.0, 1e-12);
}

// (b) Property: the projection is odd about the principal point — mirroring a
// pixel across the centre must negate both angles.
TEST(PixelToAngles, OddSymmetryAboutPrincipalPoint) {
    const double fx = 615.0, fy = 617.0, cx = 324.0, cy = 241.0;
    for (double du : {-200.0, -37.0, 5.0, 180.0}) {
        for (double dv : {-150.0, -12.0, 9.0, 133.0}) {
            auto [yaw_a, pitch_a] = pixelToAngles(cx + du, cy + dv, fx, fy, cx, cy);
            auto [yaw_b, pitch_b] = pixelToAngles(cx - du, cy - dv, fx, fy, cx, cy);
            EXPECT_NEAR(yaw_b, -yaw_a, 1e-12) << "du=" << du << " dv=" << dv;
            EXPECT_NEAR(pitch_b, -pitch_a, 1e-12) << "du=" << du << " dv=" << dv;
        }
    }
}

// (b) Property: angles are monotonic in pixel coordinate, and bounded by
// +/- pi/2 — the attention controller clamps against joint limits assuming
// this range.
TEST(PixelToAngles, MonotonicAndBounded) {
    const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;
    double previous_yaw = 10.0;  // larger than any achievable yaw
    for (double u = 0.0; u <= 640.0; u += 40.0) {
        auto [yaw, pitch] = pixelToAngles(u, cy, fx, fy, cx, cy);
        EXPECT_LT(yaw, previous_yaw) << "yaw must decrease as u increases, u=" << u;
        EXPECT_GT(yaw, -M_PI / 2.0);
        EXPECT_LT(yaw, M_PI / 2.0);
        previous_yaw = yaw;
    }
}

//=============================================================================
// getImageQoS
//=============================================================================

// Pins the WiFi-friendly profile. If this ever silently becomes Reliable, the
// Pepper image stream stalls behind retransmits instead of dropping frames —
// a failure mode that looks like "the robot froze", not like a QoS bug.
TEST(GetImageQoS, IsBestEffortVolatileKeepLastOne) {
    auto qos = getImageQoS();
    const auto& profile = qos.get_rmw_qos_profile();
    EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
    EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
    EXPECT_EQ(profile.depth, 1u);
}

//=============================================================================
// saliencyBorderPad
//=============================================================================

// (a) Hand-derived at the configured downsample size: 5% of the shorter side.
TEST(SaliencyBorderPad, ScalesWithTheShorterSide) {
    auto [pad_y, pad_x] = saliencyBorderPad(120, 160);  // package default
    EXPECT_EQ(pad_y, 6);
    EXPECT_EQ(pad_x, 6);
}

// The 3px floor keeps a usable margin on small frames.
TEST(SaliencyBorderPad, HasAThreePixelFloor) {
    auto [pad_y, pad_x] = saliencyBorderPad(20, 20);  // 5% would be 1
    EXPECT_EQ(pad_y, 3);
    EXPECT_EQ(pad_x, 3);
}

// (b) The property that matters: the mask ranges findPeaks builds from these
// pads — [0, pad) and [dim - pad, dim) — must stay inside the image on both
// axes. An unclamped pad made the second range start negative and OpenCV threw,
// so a small enough down_w/down_h crashed the saliency node outright.
TEST(SaliencyBorderPad, NeverExceedsHalfTheImageOnEitherAxis) {
    for (int h = 1; h <= 40; ++h) {
        for (int w = 1; w <= 40; ++w) {
            auto [pad_y, pad_x] = saliencyBorderPad(h, w);
            EXPECT_GE(pad_y, 0) << "h=" << h << " w=" << w;
            EXPECT_GE(pad_x, 0) << "h=" << h << " w=" << w;
            EXPECT_LE(pad_y, h / 2) << "h=" << h << " w=" << w;
            EXPECT_LE(pad_x, w / 2) << "h=" << h << " w=" << w;
            // The ranges findPeaks derives must be valid and ordered.
            EXPECT_GE(h - pad_y, pad_y) << "h=" << h << " w=" << w;
            EXPECT_GE(w - pad_x, pad_x) << "h=" << h << " w=" << w;
        }
    }
}

// The degenerate sizes that used to throw. A 1px image gets no margin at all,
// which is harmless; the point is that it returns rather than crashing.
TEST(SaliencyBorderPad, HandlesDegenerateImageSizes) {
    EXPECT_EQ(saliencyBorderPad(1, 1), std::make_pair(0, 0));
    EXPECT_EQ(saliencyBorderPad(2, 2), std::make_pair(1, 1));
    EXPECT_EQ(saliencyBorderPad(4, 4), std::make_pair(2, 2));
}

// Non-square frames clamp each axis independently, not by the shorter side.
TEST(SaliencyBorderPad, ClampsEachAxisIndependently) {
    auto [pad_y, pad_x] = saliencyBorderPad(2, 200);
    EXPECT_EQ(pad_y, 1);   // clamped by the 2px height
    EXPECT_EQ(pad_x, 3);   // the 3px floor, well inside 200
}

//=============================================================================
// generateColorFromId
//=============================================================================

namespace {

double brightnessOf(const cv::Scalar& bgr) {
    // Same weighting the implementation uses, with the BGR ordering undone.
    return 0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0];
}

}  // namespace

// (b) Property: the same face keeps the same colour across frames — the whole
// point of hashing the ID rather than assigning colours in arrival order.
TEST(GenerateColorFromId, IsDeterministic) {
    for (const std::string id : {"face_0", "face_17", "", "a-very-long-face-id-42"}) {
        EXPECT_EQ(generateColorFromId(id), generateColorFromId(id)) << "id=" << id;
    }
}

// (b) Property: every channel is a valid 8-bit value. The brightness boost
// multiplies channels by up to 150x, so a missing clamp would wrap or overflow
// when cv::Scalar is narrowed to a pixel.
TEST(GenerateColorFromId, ChannelsStayInByteRange) {
    for (int i = 0; i < 200; ++i) {
        auto color = generateColorFromId("face_" + std::to_string(i));
        for (int c = 0; c < 3; ++c) {
            EXPECT_GE(color[c], 0.0) << "i=" << i << " channel=" << c;
            EXPECT_LE(color[c], 255.0) << "i=" << i << " channel=" << c;
        }
    }
}

// Pins the documented channel ordering: the hash's low byte is RED, and the
// returned Scalar is BGR (so red lands in index 2). Expected values are
// recomputed from std::hash here rather than hardcoded, because the hash is
// implementation-defined — what this pins is the ordering and the boost rule,
// not libstdc++'s hash values.
TEST(GenerateColorFromId, UsesBgrOrderAndBoostsOnlyDarkColors) {
    for (int i = 0; i < 100; ++i) {
        const std::string id = "face_" + std::to_string(i);
        std::size_t h = std::hash<std::string>{}(id);
        const double r = static_cast<double>(h & 0xFF);
        const double g = static_cast<double>((h >> 8) & 0xFF);
        const double b = static_cast<double>((h >> 16) & 0xFF);
        const double raw_brightness = 0.299 * r + 0.587 * g + 0.114 * b;

        auto color = generateColorFromId(id);
        if (raw_brightness >= 100.0) {
            // Bright enough already: passed through untouched, in BGR order.
            EXPECT_DOUBLE_EQ(color[0], b) << "id=" << id;
            EXPECT_DOUBLE_EQ(color[1], g) << "id=" << id;
            EXPECT_DOUBLE_EQ(color[2], r) << "id=" << id;
        } else {
            // Too dark to read on the overlay: scaled up, never down.
            EXPECT_GE(brightnessOf(color), raw_brightness - 1e-9)
                << "id=" << id << " must not get darker";
            EXPECT_GE(color[0], b - 1e-9) << "id=" << id;
            EXPECT_GE(color[1], g - 1e-9) << "id=" << id;
            EXPECT_GE(color[2], r - 1e-9) << "id=" << id;
        }
    }
}
