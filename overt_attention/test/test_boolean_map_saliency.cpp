/* test_boolean_map_saliency.cpp
 *
 * Unit tests for BooleanMapSaliency — the bottom-up saliency operator behind
 * SaliencyNode. Synthetic images only: no ROS graph, no camera, no node.
 *
 * BMS thresholds the normalized Lab channels into boolean maps, suppresses
 * whatever touches the image border by flood-fill, and averages the surviving
 * regions. So the properties worth pinning are structural rather than exact
 * pixel values:
 *   (a) output shape/type/range invariants,
 *   (b) a border-connected region scores lower than an enclosed one — the
 *       whole point of the flood-fill activation step,
 *   (c) degenerate inputs (flat images) produce zeros rather than NaNs.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Aug 16, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "overt_attention/overt_attention_interface.h"

namespace {

// A mid-grey canvas — uniform, so it carries no saliency of its own.
cv::Mat background(int width = 64, int height = 48) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
}

double meanInside(const cv::Mat& saliency, const cv::Rect& roi) {
    return cv::mean(saliency(roi))[0];
}

}  // namespace

//=============================================================================
// Output invariants
//=============================================================================

TEST(BooleanMapSaliency, OutputMatchesInputSizeAndIsFloat) {
    BooleanMapSaliency bms;
    cv::Mat frame = background(80, 60);
    cv::rectangle(frame, cv::Rect(30, 20, 20, 20), cv::Scalar(255, 255, 255), cv::FILLED);

    cv::Mat saliency = bms.computeSaliency(frame);
    EXPECT_EQ(saliency.rows, frame.rows);
    EXPECT_EQ(saliency.cols, frame.cols);
    EXPECT_EQ(saliency.type(), CV_32F);
}

// The map is documented as normalized to [0, 1]; downstream peak-finding
// compares against a min_peak threshold in those units.
TEST(BooleanMapSaliency, OutputStaysInUnitRange) {
    BooleanMapSaliency bms;
    cv::Mat frame = background();
    cv::circle(frame, cv::Point(32, 24), 8, cv::Scalar(0, 0, 255), cv::FILLED);
    cv::rectangle(frame, cv::Rect(4, 4, 10, 10), cv::Scalar(255, 255, 0), cv::FILLED);

    cv::Mat saliency = bms.computeSaliency(frame);
    double min_value = 0.0, max_value = 0.0;
    cv::minMaxLoc(saliency, &min_value, &max_value);
    EXPECT_GE(min_value, 0.0);
    EXPECT_LE(max_value, 1.0);
}

TEST(BooleanMapSaliency, OutputIsFinite) {
    BooleanMapSaliency bms;
    cv::Mat frame = background();
    cv::circle(frame, cv::Point(32, 24), 10, cv::Scalar(20, 200, 40), cv::FILLED);

    cv::Mat saliency = bms.computeSaliency(frame);
    EXPECT_EQ(cv::countNonZero(saliency != saliency), 0) << "NaNs present";
    cv::Mat infinite = (cv::abs(saliency) == std::numeric_limits<float>::infinity());
    EXPECT_EQ(cv::countNonZero(infinite), 0) << "infinities present";
}

//=============================================================================
// Degenerate inputs
//=============================================================================

// A perfectly flat image has zero Lab range; the implementation short-circuits
// to zeros rather than dividing by ~0 and producing NaNs.
TEST(BooleanMapSaliency, UniformImageIsAllZeros) {
    BooleanMapSaliency bms;
    cv::Mat saliency = bms.computeSaliency(background());

    double min_value = 0.0, max_value = 0.0;
    cv::minMaxLoc(saliency, &min_value, &max_value);
    EXPECT_DOUBLE_EQ(min_value, 0.0);
    EXPECT_DOUBLE_EQ(max_value, 0.0);
}

TEST(BooleanMapSaliency, BlackAndWhiteUniformImagesAreAlsoZero) {
    BooleanMapSaliency bms;
    for (const auto& shade : {cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255)}) {
        cv::Mat frame(32, 32, CV_8UC3, shade);
        double min_value = 0.0, max_value = 0.0;
        cv::minMaxLoc(bms.computeSaliency(frame), &min_value, &max_value);
        EXPECT_DOUBLE_EQ(max_value, 0.0);
    }
}

// A single-pixel frame is a legitimate degenerate case for the border
// flood-fill (every pixel is a border pixel); it must not crash or index
// out of bounds.
TEST(BooleanMapSaliency, HandlesTinyImages) {
    BooleanMapSaliency bms;
    for (int size : {1, 2, 3}) {
        cv::Mat frame(size, size, CV_8UC3, cv::Scalar(10, 120, 240));
        cv::Mat saliency;
        ASSERT_NO_THROW(saliency = bms.computeSaliency(frame)) << "size=" << size;
        EXPECT_EQ(saliency.rows, size);
        EXPECT_EQ(saliency.cols, size);
    }
}

//=============================================================================
// The flood-fill activation step
//=============================================================================

// (b) The defining behaviour of BMS: a region enclosed by the image must score
// higher than an otherwise identical region bleeding off the border, because
// border-connected regions are flood-filled away. Without the activation step
// these two would score the same, and the robot would fixate on background
// walls as readily as on objects.
TEST(BooleanMapSaliency, EnclosedRegionOutscoresBorderConnectedRegion) {
    BooleanMapSaliency bms;
    const cv::Scalar patch_color(255, 255, 255);

    cv::Mat centred = background(64, 64);
    const cv::Rect centre_roi(26, 26, 12, 12);
    cv::rectangle(centred, centre_roi, patch_color, cv::FILLED);

    // Same-sized patch, but flush against the left edge.
    cv::Mat edged = background(64, 64);
    const cv::Rect edge_roi(0, 26, 12, 12);
    cv::rectangle(edged, edge_roi, patch_color, cv::FILLED);

    const double centre_score = meanInside(bms.computeSaliency(centred), centre_roi);
    const double edge_score = meanInside(bms.computeSaliency(edged), edge_roi);

    EXPECT_GT(centre_score, edge_score)
        << "centre=" << centre_score << " edge=" << edge_score;
}

// (b) A centred object must stand out against the uniform background
// surrounding it — the basic "is this operator finding anything at all" check.
TEST(BooleanMapSaliency, CentredObjectOutscoresItsSurroundings) {
    BooleanMapSaliency bms;
    cv::Mat frame = background(64, 64);
    const cv::Rect object_roi(24, 24, 16, 16);
    cv::rectangle(frame, object_roi, cv::Scalar(255, 255, 255), cv::FILLED);

    cv::Mat saliency = bms.computeSaliency(frame);
    const double object_score = meanInside(saliency, object_roi);
    const double corner_score = meanInside(saliency, cv::Rect(0, 0, 10, 10));

    EXPECT_GT(object_score, corner_score)
        << "object=" << object_score << " corner=" << corner_score;
}

//=============================================================================
// Determinism and threshold count
//=============================================================================

// Frame-to-frame stability: the same image must always score the same, or the
// attention controller would chase noise between identical frames.
TEST(BooleanMapSaliency, IsDeterministic) {
    BooleanMapSaliency bms;
    cv::Mat frame = background();
    cv::circle(frame, cv::Point(32, 24), 9, cv::Scalar(0, 0, 255), cv::FILLED);

    cv::Mat first = bms.computeSaliency(frame);
    cv::Mat second = bms.computeSaliency(frame);
    EXPECT_EQ(cv::countNonZero(first != second), 0);
}

// The same instance must not accumulate state across calls — a different image
// in between cannot change the result for a repeated one.
TEST(BooleanMapSaliency, IsStatelessAcrossCalls) {
    BooleanMapSaliency bms;
    cv::Mat frame = background();
    cv::circle(frame, cv::Point(32, 24), 9, cv::Scalar(0, 0, 255), cv::FILLED);

    cv::Mat baseline = bms.computeSaliency(frame);

    cv::Mat other = background();
    cv::rectangle(other, cv::Rect(2, 2, 30, 30), cv::Scalar(255, 0, 0), cv::FILLED);
    bms.computeSaliency(other);

    cv::Mat repeated = bms.computeSaliency(frame);
    EXPECT_EQ(cv::countNonZero(baseline != repeated), 0);
}

// The threshold count is a quality/cost dial. Both settings must stay within
// the documented range and agree on which region is the salient one.
TEST(BooleanMapSaliency, ThresholdCountIsConfigurableAndConsistent) {
    cv::Mat frame = background(64, 64);
    const cv::Rect object_roi(24, 24, 16, 16);
    cv::rectangle(frame, object_roi, cv::Scalar(255, 255, 255), cv::FILLED);

    for (int thresholds : {1, 4, 10, 20}) {
        BooleanMapSaliency bms(thresholds);
        cv::Mat saliency = bms.computeSaliency(frame);

        double min_value = 0.0, max_value = 0.0;
        cv::minMaxLoc(saliency, &min_value, &max_value);
        EXPECT_GE(min_value, 0.0) << "n_thresholds=" << thresholds;
        EXPECT_LE(max_value, 1.0) << "n_thresholds=" << thresholds;

        EXPECT_GT(meanInside(saliency, object_roi),
                  meanInside(saliency, cv::Rect(0, 0, 10, 10)))
            << "n_thresholds=" << thresholds;
    }
}
