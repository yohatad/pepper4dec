/* test_age_gender_profile.cpp
 *
 * Unit tests for the temporal-smoothing logic in AgeGenderPersonProfile
 * (median age + confidence-weighted majority gender vote over a sliding
 * 5-estimate window), its JSON serialization, and AgeGenderBoundingBox's
 * centroid-to-corners conversion.
 *
 * Pure data structures — no ROS graph, no MiVOLO model, no camera. The one
 * ROS type touched (geometry_msgs::msg::Point) is a plain message struct that
 * needs no rclcpp::init().
 *
 * Expected values come from three sources, noted per test:
 *   (a) hand-computed medians and weighted votes,
 *   (b) properties that must hold (window bounding, tie-break direction),
 *   (c) the documented JSON shape and numeric precision.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Aug 16, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include <gtest/gtest.h>

#include <string>

#include "face_detection/age_gender_detection_interface.h"

//=============================================================================
// AgeGenderBoundingBox::fromCentroid
//=============================================================================

// (a) Hand-derived: the centroid sits at the box centre, so the corners are
// the centroid plus/minus half the extents, and z carries through as depth.
TEST(AgeGenderBoundingBox, FromCentroidComputesCorners) {
    geometry_msgs::msg::Point centroid;
    centroid.x = 100.0;
    centroid.y = 200.0;
    centroid.z = 1.5;  // metres

    auto bbox = AgeGenderBoundingBox::fromCentroid(centroid, 40.0f, 60.0f, true);
    EXPECT_DOUBLE_EQ(bbox.x1, 80.0);
    EXPECT_DOUBLE_EQ(bbox.x2, 120.0);
    EXPECT_DOUBLE_EQ(bbox.y1, 170.0);
    EXPECT_DOUBLE_EQ(bbox.y2, 230.0);
    EXPECT_DOUBLE_EQ(bbox.depth, 1.5);
    EXPECT_TRUE(bbox.mutual_gaze);
}

// mutual_gaze defaults to false — it is only meaningful for face detections,
// and person boxes must not claim eye contact they never measured.
TEST(AgeGenderBoundingBox, MutualGazeDefaultsToFalse) {
    geometry_msgs::msg::Point centroid;
    auto bbox = AgeGenderBoundingBox::fromCentroid(centroid, 10.0f, 10.0f);
    EXPECT_FALSE(bbox.mutual_gaze);
}

// (b) Property: corners stay correctly ordered (x1 < x2, y1 < y2) for any
// positive extent — downstream crop extraction assumes this ordering.
TEST(AgeGenderBoundingBox, CornersStayOrdered) {
    geometry_msgs::msg::Point centroid;
    for (double cx : {-50.0, 0.0, 640.0}) {
        centroid.x = cx;
        centroid.y = cx;
        auto bbox = AgeGenderBoundingBox::fromCentroid(centroid, 1.0f, 1.0f);
        EXPECT_LT(bbox.x1, bbox.x2) << "cx=" << cx;
        EXPECT_LT(bbox.y1, bbox.y2) << "cx=" << cx;
    }
}

//=============================================================================
// AgeGenderPersonProfile::addEstimate — age median
//=============================================================================

TEST(AgeGenderProfile, StartsEmpty) {
    AgeGenderPersonProfile profile;
    EXPECT_FALSE(profile.has_valid_estimate);
    EXPECT_FALSE(profile.age.has_value());
    EXPECT_FALSE(profile.gender.has_value());
    EXPECT_EQ(profile.estimation_count, 0);
}

// (a) Hand-computed: a single estimate is its own median.
TEST(AgeGenderProfile, FirstEstimateBecomesTheMedian) {
    AgeGenderPersonProfile profile;
    profile.addEstimate(30.0, "male", 0.9);
    ASSERT_TRUE(profile.age.has_value());
    EXPECT_DOUBLE_EQ(*profile.age, 30.0);
    EXPECT_TRUE(profile.has_valid_estimate);
    EXPECT_EQ(profile.estimation_count, 1);
}

// (a) Hand-computed: odd count takes the middle element, even count averages
// the two middle elements. Median (not mean) is what makes a single wild
// outlier frame unable to move the reported age much.
TEST(AgeGenderProfile, AgeIsMedianNotMean) {
    AgeGenderPersonProfile profile;
    profile.addEstimate(20.0, "male", 0.9);
    profile.addEstimate(30.0, "male", 0.9);
    // Two values -> mean of the middle pair.
    EXPECT_DOUBLE_EQ(*profile.age, 25.0);

    profile.addEstimate(100.0, "male", 0.9);  // outlier
    // Sorted {20, 30, 100} -> median 30, not the mean of 50.
    EXPECT_DOUBLE_EQ(*profile.age, 30.0);
}

// (a) Hand-computed: insertion order must not matter to the median.
TEST(AgeGenderProfile, MedianIgnoresInsertionOrder) {
    AgeGenderPersonProfile ascending;
    for (double a : {10.0, 20.0, 30.0, 40.0, 50.0}) ascending.addEstimate(a, "male", 0.9);

    AgeGenderPersonProfile shuffled;
    for (double a : {30.0, 50.0, 10.0, 40.0, 20.0}) shuffled.addEstimate(a, "male", 0.9);

    EXPECT_DOUBLE_EQ(*ascending.age, 30.0);
    EXPECT_DOUBLE_EQ(*shuffled.age, *ascending.age);
}

// (b) Property: the window holds at most kHistoryMaxLen entries, so old
// estimates age out. Without this the profile would drift toward whatever the
// person looked like when first seen and never recover.
TEST(AgeGenderProfile, HistoryIsBoundedAndOldEstimatesAgeOut) {
    AgeGenderPersonProfile profile;
    // Five estimates at 10, then five at 80 — the window should hold only 80s.
    for (int i = 0; i < 5; ++i) profile.addEstimate(10.0, "male", 0.9);
    EXPECT_DOUBLE_EQ(*profile.age, 10.0);
    EXPECT_EQ(profile.age_history.size(), AgeGenderPersonProfile::kHistoryMaxLen);

    for (int i = 0; i < 5; ++i) profile.addEstimate(80.0, "male", 0.9);
    EXPECT_EQ(profile.age_history.size(), AgeGenderPersonProfile::kHistoryMaxLen);
    EXPECT_DOUBLE_EQ(*profile.age, 80.0);

    // estimation_count is a lifetime counter, NOT the window size.
    EXPECT_EQ(profile.estimation_count, 10);
}

//=============================================================================
// addEstimate — confidence-weighted gender vote
//=============================================================================

TEST(AgeGenderProfile, SingleGenderEstimateIsFullyConfident) {
    AgeGenderPersonProfile profile;
    profile.addEstimate(30.0, "male", 0.8);
    ASSERT_TRUE(profile.gender.has_value());
    EXPECT_EQ(*profile.gender, "male");
    // Only one vote, so it holds the entire share regardless of its raw score.
    ASSERT_TRUE(profile.gender_confidence.has_value());
    EXPECT_DOUBLE_EQ(*profile.gender_confidence, 1.0);
}

// (a) Hand-computed: the vote is weighted by confidence, not by count.
// Two low-confidence male votes (0.2 + 0.2 = 0.4) must lose to one
// high-confidence female vote (0.9). Confidence share = 0.9 / 1.3.
TEST(AgeGenderProfile, VoteIsWeightedByConfidenceNotCount) {
    AgeGenderPersonProfile profile;
    profile.addEstimate(30.0, "male", 0.2);
    profile.addEstimate(30.0, "male", 0.2);
    profile.addEstimate(30.0, "female", 0.9);

    EXPECT_EQ(*profile.gender, "female");
    EXPECT_NEAR(*profile.gender_confidence, 0.9 / 1.3, 1e-12);
}

// (b) Property: reported confidence is a share of the total vote, so it is
// always in [0.5, 1.0] — the winner holds at least half by definition.
TEST(AgeGenderProfile, ConfidenceIsANormalizedShare) {
    AgeGenderPersonProfile profile;
    profile.addEstimate(30.0, "male", 0.6);
    profile.addEstimate(30.0, "female", 0.4);
    // male 0.6 of 1.0 total.
    EXPECT_EQ(*profile.gender, "male");
    EXPECT_NEAR(*profile.gender_confidence, 0.6, 1e-12);
    EXPECT_GE(*profile.gender_confidence, 0.5);
    EXPECT_LE(*profile.gender_confidence, 1.0);
}

// (b) Documented tie-break: the comparison is `male_score > female_score`, so
// an exact tie resolves to FEMALE. Pinned so the bias is a deliberate choice
// rather than something a refactor can silently flip.
TEST(AgeGenderProfile, ExactTieResolvesToFemale) {
    AgeGenderPersonProfile profile;
    profile.addEstimate(30.0, "male", 0.5);
    profile.addEstimate(30.0, "female", 0.5);
    EXPECT_EQ(*profile.gender, "female");
    EXPECT_NEAR(*profile.gender_confidence, 0.5, 1e-12);
}

// An unrecognized gender label contributes no weight to either side. With no
// valid votes at all, gender stays unset rather than defaulting to a guess.
TEST(AgeGenderProfile, UnknownGenderLabelsContributeNoWeight) {
    AgeGenderPersonProfile profile;
    profile.addEstimate(30.0, "unknown", 0.9);
    EXPECT_FALSE(profile.gender.has_value());
    // The age half of the estimate still landed.
    ASSERT_TRUE(profile.age.has_value());
    EXPECT_DOUBLE_EQ(*profile.age, 30.0);

    // A real vote alongside it wins outright, unpolluted by the unknown label.
    profile.addEstimate(30.0, "male", 0.4);
    EXPECT_EQ(*profile.gender, "male");
    EXPECT_DOUBLE_EQ(*profile.gender_confidence, 1.0);
}

// A person who turns away and is re-estimated as the other gender should flip
// once the old votes leave the 5-entry window — the same aging-out property as
// age, but through the weighted vote.
TEST(AgeGenderProfile, GenderFlipsAfterWindowTurnsOver) {
    AgeGenderPersonProfile profile;
    for (int i = 0; i < 5; ++i) profile.addEstimate(30.0, "male", 0.9);
    EXPECT_EQ(*profile.gender, "male");

    for (int i = 0; i < 5; ++i) profile.addEstimate(30.0, "female", 0.9);
    EXPECT_EQ(*profile.gender, "female");
    EXPECT_DOUBLE_EQ(*profile.gender_confidence, 1.0);
}

//=============================================================================
// toJson
//=============================================================================

// (c) An untouched profile serializes its optionals as JSON null — not as
// "0", not as an empty string. Consumers distinguish "not yet estimated"
// from "estimated as zero".
TEST(AgeGenderProfileJson, EmptyProfileUsesNulls) {
    AgeGenderPersonProfile profile;
    profile.label_id = "person_1";
    const std::string json = profile.toJson();

    EXPECT_NE(json.find("\"label_id\":\"person_1\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"age\":null"), std::string::npos) << json;
    EXPECT_NE(json.find("\"gender\":null"), std::string::npos) << json;
    EXPECT_NE(json.find("\"gender_confidence\":null"), std::string::npos) << json;
    EXPECT_NE(json.find("\"estimation_count\":0"), std::string::npos) << json;
    EXPECT_NE(json.find("\"person_bbox\":null"), std::string::npos) << json;
}

// (c) Documented precision: age at 1 decimal, confidence at 3.
TEST(AgeGenderProfileJson, FormatsNumbersAtDocumentedPrecision) {
    AgeGenderPersonProfile profile;
    profile.label_id = "person_2";
    profile.addEstimate(31.25, "male", 0.9);
    const std::string json = profile.toJson();

    EXPECT_NE(json.find("\"age\":31.2"), std::string::npos) << json;
    EXPECT_NE(json.find("\"gender\":\"male\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"gender_confidence\":1.000"), std::string::npos) << json;
    EXPECT_NE(json.find("\"estimation_count\":1"), std::string::npos) << json;
}

// (c) The bbox is nested as an object with 2-decimal corners once present.
TEST(AgeGenderProfileJson, SerializesBoundingBoxWhenPresent) {
    AgeGenderPersonProfile profile;
    profile.label_id = "person_3";

    geometry_msgs::msg::Point centroid;
    centroid.x = 100.0;
    centroid.y = 200.0;
    profile.last_person_bbox = AgeGenderBoundingBox::fromCentroid(centroid, 40.0f, 60.0f);

    const std::string json = profile.toJson();
    EXPECT_NE(json.find("\"person_bbox\":{\"x1\":80.00,\"y1\":170.00,\"x2\":120.00,\"y2\":230.00}"),
              std::string::npos) << json;
}

// A label_id containing a quote or backslash must be escaped, or the emitted
// string stops being parseable JSON at that character.
TEST(AgeGenderProfileJson, EscapesQuotesAndBackslashesInLabelId) {
    AgeGenderPersonProfile profile;
    profile.label_id = "we\"ird\\id";
    const std::string json = profile.toJson();
    EXPECT_NE(json.find("\"label_id\":\"we\\\"ird\\\\id\""), std::string::npos) << json;
}

// (b) Property: the serialized form is balanced and starts/ends as an object.
// A cheap structural guard that survives future field additions.
TEST(AgeGenderProfileJson, IsAWellFormedObject) {
    AgeGenderPersonProfile profile;
    profile.label_id = "person_4";
    profile.addEstimate(42.0, "female", 0.75);

    const std::string json = profile.toJson();
    ASSERT_FALSE(json.empty());
    EXPECT_EQ(json.front(), '{');
    EXPECT_EQ(json.back(), '}');

    int depth = 0;
    bool in_string = false;
    for (size_t i = 0; i < json.size(); ++i) {
        const char c = json[i];
        if (in_string) {
            if (c == '\\') { ++i; continue; }   // skip the escaped character
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == '{') ++depth;
        else if (c == '}') --depth;
        ASSERT_GE(depth, 0) << "unbalanced at index " << i << ": " << json;
    }
    EXPECT_EQ(depth, 0) << json;
    EXPECT_FALSE(in_string) << "unterminated string: " << json;
}
