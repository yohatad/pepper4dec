/* test_byte_tracker.cpp
 *
 * Unit tests for the shared ByteTrack port in dec_common: the Hungarian
 * assignment and IoU helpers against hand-computed matrices, and the full
 * tracker against synthetic detection sequences (stable IDs while tracking,
 * ID retention through short occlusions, low-confidence filtering).
 *
 * Everything is synthetic — no ROS, no camera, no model inference.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 18, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <utility>
#include <vector>

#include "dec_common/byte_tracker.h"

using byte_tracker::ByteTrack;
using byte_tracker::Detections;

namespace {

// Builds a single-detection Detections at (x, y) with the given size/score.
Detections makeDetections(const std::vector<Eigen::Vector4d>& boxes, float score) {
    Detections d;
    d.xyxy = boxes;
    d.confidence.assign(boxes.size(), score);
    d.class_id.assign(boxes.size(), 0);
    return d;
}

}  // namespace

//=============================================================================
// Matching helpers — hand-computed expected values
//=============================================================================

TEST(HungarianAssignment, PicksGloballyOptimalMatching) {
    // Greedy would pick (0,0) cost 1 then (1,1) cost 4 = 5 total; the optimal
    // assignment is (0,1)+(1,0) = 4 total. Pins that this is Hungarian, not greedy.
    Eigen::MatrixXd cost(2, 2);
    cost << 1.0, 2.0,
            2.0, 4.0;
    auto matches = byte_tracker::matching::hungarianAssignment(cost);
    ASSERT_EQ(matches.size(), 2u);
    std::sort(matches.begin(), matches.end());
    EXPECT_EQ(matches[0], std::make_pair(0, 1));
    EXPECT_EQ(matches[1], std::make_pair(1, 0));
}

TEST(HungarianAssignment, RectangularReturnsMinDimensionPairs) {
    // 2 rows x 3 cols: every row matched, exactly one column left out.
    Eigen::MatrixXd cost(2, 3);
    cost << 5.0, 1.0, 9.0,
            1.0, 5.0, 9.0;
    auto matches = byte_tracker::matching::hungarianAssignment(cost);
    ASSERT_EQ(matches.size(), 2u);
    std::sort(matches.begin(), matches.end());
    EXPECT_EQ(matches[0], std::make_pair(0, 1));
    EXPECT_EQ(matches[1], std::make_pair(1, 0));
}

TEST(BoxIou, KnownOverlaps) {
    std::vector<Eigen::Vector4d> a{{0.0, 0.0, 10.0, 10.0}};
    std::vector<Eigen::Vector4d> b{
        {0.0, 0.0, 10.0, 10.0},    // identical -> IoU 1
        {5.0, 0.0, 15.0, 10.0},    // half-shifted: inter 50, union 150 -> 1/3
        {20.0, 20.0, 30.0, 30.0},  // disjoint -> 0
    };
    auto iou = byte_tracker::matching::boxIouBatch(a, b);
    ASSERT_EQ(iou.rows(), 1);
    ASSERT_EQ(iou.cols(), 3);
    EXPECT_NEAR(iou(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(iou(0, 1), 1.0 / 3.0, 1e-9);
    EXPECT_NEAR(iou(0, 2), 0.0, 1e-9);
}

TEST(LinearAssignment, ThresholdExcludesExpensiveMatches) {
    // Row 1's best option costs 0.9 > thresh 0.5, so it must come back
    // unmatched even though the solver would pair it.
    Eigen::MatrixXd cost(2, 2);
    cost << 0.1, 0.9,
            0.9, 0.9;
    auto result = byte_tracker::matching::linearAssignment(cost, 0.5);
    ASSERT_EQ(result.matches.size(), 1u);
    EXPECT_EQ(result.matches[0], std::make_pair(0, 0));
    ASSERT_EQ(result.unmatched_rows.size(), 1u);
    EXPECT_EQ(result.unmatched_rows[0], 1);
    ASSERT_EQ(result.unmatched_cols.size(), 1u);
    EXPECT_EQ(result.unmatched_cols[0], 1);
}

//=============================================================================
// ByteTrack end-to-end on synthetic sequences
//=============================================================================

// Two objects crossing paths horizontally. Each must keep its own ID for the
// whole sequence — ID stability is the property the whole system depends on
// (face IDs, person IDs feeding the behavior tree).
TEST(ByteTrack, StableIdsForTwoCrossingObjects) {
    ByteTrack tracker;  // defaults: activation 0.25, matching 0.8

    int id_a = -1;
    int id_b = -1;
    for (int frame = 0; frame < 12; ++frame) {
        double xa = 0.0 + 10.0 * frame;     // A moves right
        double xb = 200.0 - 10.0 * frame;   // B moves left
        auto dets = makeDetections({{xa, 0.0, xa + 50.0, 100.0},
                                    {xb, 150.0, xb + 50.0, 250.0}},
                                   0.9f);
        auto tracked = tracker.updateWithDetections(dets);
        ASSERT_EQ(tracked.tracker_id.size(), 2u) << "frame " << frame;

        // Identify which output row is A by its y-extent (A lives at y 0-100).
        int idx_a = (tracked.xyxy[0][1] < 100.0) ? 0 : 1;
        int idx_b = 1 - idx_a;
        if (frame == 0) {
            id_a = tracked.tracker_id[idx_a];
            id_b = tracked.tracker_id[idx_b];
            EXPECT_GT(id_a, 0);
            EXPECT_GT(id_b, 0);
            EXPECT_NE(id_a, id_b);
        } else {
            EXPECT_EQ(tracked.tracker_id[idx_a], id_a) << "frame " << frame;
            EXPECT_EQ(tracked.tracker_id[idx_b], id_b) << "frame " << frame;
        }
    }
}

// A stationary object that drops out for two frames (occlusion, missed
// detection) must come back with the SAME ID — that's ByteTrack's lost-track
// re-association, and it's why the robot doesn't re-greet a person after a
// blink of missed detections.
TEST(ByteTrack, ReassociatesAfterShortDropout) {
    ByteTrack tracker;
    const Eigen::Vector4d box{100.0, 100.0, 150.0, 200.0};

    auto first = tracker.updateWithDetections(makeDetections({box}, 0.9f));
    ASSERT_EQ(first.tracker_id.size(), 1u);
    int original_id = first.tracker_id[0];

    // A couple of solid frames, then two empty frames (object missed).
    tracker.updateWithDetections(makeDetections({box}, 0.9f));
    tracker.updateWithDetections(makeDetections({box}, 0.9f));
    tracker.updateWithDetections(makeDetections({}, 0.9f));
    tracker.updateWithDetections(makeDetections({}, 0.9f));

    auto back = tracker.updateWithDetections(makeDetections({box}, 0.9f));
    ASSERT_EQ(back.tracker_id.size(), 1u);
    EXPECT_EQ(back.tracker_id[0], original_id);
}

// Detections below the activation threshold must not spawn tracks; the
// output drops them entirely (mirrors supervision's update_with_detections).
TEST(ByteTrack, LowConfidenceDetectionsDoNotCreateTracks) {
    ByteTrack tracker;  // activation threshold 0.25
    auto out = tracker.updateWithDetections(
        makeDetections({{0.0, 0.0, 50.0, 50.0}}, 0.1f));
    EXPECT_EQ(out.tracker_id.size(), 0u);
    EXPECT_EQ(out.xyxy.size(), 0u);
}

// reset() must restart ID assignment from 1 — used between runs/streams.
TEST(ByteTrack, ResetRestartsExternalIds) {
    ByteTrack tracker;
    const Eigen::Vector4d box{0.0, 0.0, 50.0, 50.0};

    auto first = tracker.updateWithDetections(makeDetections({box}, 0.9f));
    ASSERT_EQ(first.tracker_id.size(), 1u);
    int first_id = first.tracker_id[0];

    tracker.reset();

    auto after = tracker.updateWithDetections(makeDetections({box}, 0.9f));
    ASSERT_EQ(after.tracker_id.size(), 1u);
    EXPECT_EQ(after.tracker_id[0], first_id);  // counter restarted, same first ID
}
