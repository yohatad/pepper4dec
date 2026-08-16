/* test_class_indices.cpp
 *
 * Unit tests for getClassIndices() — the resolver that turns the
 * target_classes parameter (COCO class names, or numeric indices written as
 * strings) into the set of class indices the detector is allowed to publish.
 *
 * Pure lookup logic: no ROS graph, no model, no camera. The rclcpp::Logger it
 * takes is obtained from rclcpp::get_logger(), which needs no rclcpp::init().
 *
 * This is the unit-level counterpart to test_bag_replay_launch.py — that test
 * exercises the whole node against recorded frames, this one pins the class
 * filtering that decides what those frames are allowed to produce.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Aug 16, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "person_detection/person_detection_interface.h"

namespace {

rclcpp::Logger testLogger() {
    return rclcpp::get_logger("test_class_indices");
}

// Index of a COCO class name, or -1. Keeps the tests readable without
// hardcoding indices that would silently rot if COCO_CLASSES were reordered.
int indexOf(const std::string& name) {
    auto it = std::find(COCO_CLASSES.begin(), COCO_CLASSES.end(), name);
    return it == COCO_CLASSES.end()
        ? -1
        : static_cast<int>(std::distance(COCO_CLASSES.begin(), it));
}

}  // namespace

// The COCO class table is a fixed 80-entry contract with the exported model:
// if these drift, every published class label is wrong by an offset.
TEST(CocoClasses, HasTheExpectedShape) {
    ASSERT_EQ(COCO_CLASSES.size(), 80u);
    EXPECT_EQ(COCO_CLASSES.front(), "person");  // index 0 — the default target
    EXPECT_NE(indexOf("chair"), -1);
    EXPECT_NE(indexOf("bottle"), -1);
}

//=============================================================================
// "track everything" forms
//=============================================================================

// Empty and {"all"} are both documented as "track everything". An empty list
// resolving to an empty set instead would silently disable all detection.
TEST(GetClassIndices, EmptyMeansEverything) {
    auto indices = getClassIndices({}, testLogger());
    EXPECT_EQ(indices.size(), COCO_CLASSES.size());
    EXPECT_EQ(*indices.begin(), 0);
    EXPECT_EQ(*indices.rbegin(), static_cast<int>(COCO_CLASSES.size()) - 1);
}

TEST(GetClassIndices, AllKeywordMeansEverything) {
    EXPECT_EQ(getClassIndices({"all"}, testLogger()).size(), COCO_CLASSES.size());
}

// "all" anywhere in the list wins, even mixed with specific names.
TEST(GetClassIndices, AllKeywordWinsWhenMixedWithNames) {
    EXPECT_EQ(getClassIndices({"person", "all", "chair"}, testLogger()).size(),
              COCO_CLASSES.size());
}

//=============================================================================
// Name resolution
//=============================================================================

TEST(GetClassIndices, ResolvesNamesToIndices) {
    auto indices = getClassIndices({"person", "chair"}, testLogger());
    EXPECT_EQ(indices, (std::set<int>{indexOf("person"), indexOf("chair")}));
}

// Names are matched case-insensitively, so a YAML file written with
// "Person" behaves the same as "person".
TEST(GetClassIndices, NameMatchingIsCaseInsensitive) {
    const std::set<int> expected{indexOf("person")};
    EXPECT_EQ(getClassIndices({"Person"}, testLogger()), expected);
    EXPECT_EQ(getClassIndices({"PERSON"}, testLogger()), expected);
    EXPECT_EQ(getClassIndices({"pErSoN"}, testLogger()), expected);
}

// The result is a set: duplicates collapse rather than producing repeats.
TEST(GetClassIndices, DuplicatesCollapse) {
    auto indices = getClassIndices({"person", "person", "PERSON"}, testLogger());
    EXPECT_EQ(indices, (std::set<int>{indexOf("person")}));
}

// Unknown names are warned about and skipped — a typo must not take the whole
// filter down with it, and must not fall back to "track everything".
TEST(GetClassIndices, UnknownNamesAreSkippedNotFatal) {
    auto indices = getClassIndices({"person", "unicorn"}, testLogger());
    EXPECT_EQ(indices, (std::set<int>{indexOf("person")}));

    // Every name unknown -> empty set, NOT the everything set.
    EXPECT_TRUE(getClassIndices({"unicorn", "dragon"}, testLogger()).empty());
}

//=============================================================================
// Numeric entries
//=============================================================================

// Numeric strings are direct class indices (the isinstance(cls, int) branch of
// the Python reference), not names to look up.
TEST(GetClassIndices, NumericStringsAreDirectIndices) {
    EXPECT_EQ(getClassIndices({"0"}, testLogger()), (std::set<int>{0}));
    EXPECT_EQ(getClassIndices({"0", "56"}, testLogger()), (std::set<int>{0, 56}));
}

// A numeric index and the matching name must resolve identically.
TEST(GetClassIndices, NumericAndNameAgree) {
    const int person = indexOf("person");
    EXPECT_EQ(getClassIndices({std::to_string(person)}, testLogger()),
              getClassIndices({"person"}, testLogger()));
}

// Out-of-range indices are warned about and dropped rather than inserted —
// an index >= 80 would read past COCO_CLASSES when labels are looked up.
TEST(GetClassIndices, OutOfRangeIndicesAreDropped) {
    EXPECT_TRUE(getClassIndices({"80"}, testLogger()).empty());
    EXPECT_TRUE(getClassIndices({"999"}, testLogger()).empty());
    // The last valid index still gets through.
    EXPECT_EQ(getClassIndices({"79"}, testLogger()), (std::set<int>{79}));
    // A bad index must not discard the good entries alongside it.
    EXPECT_EQ(getClassIndices({"0", "999"}, testLogger()), (std::set<int>{0}));
}

// "-1" is not all-digits, so it takes the name branch, finds nothing, and is
// skipped. Pinned because the obvious alternative (parsing it as an int) would
// insert a negative index and read out of bounds.
TEST(GetClassIndices, NegativeNumbersAreTreatedAsUnknownNames) {
    EXPECT_TRUE(getClassIndices({"-1"}, testLogger()).empty());
}

// The package default. Guards against a refactor that changes what a
// stock launch (target_classes: ["person"]) actually tracks.
TEST(GetClassIndices, PackageDefaultResolvesToPersonOnly) {
    PersonDetectionConfig config;
    EXPECT_EQ(getClassIndices(config.target_classes, testLogger()),
              (std::set<int>{indexOf("person")}));
}
