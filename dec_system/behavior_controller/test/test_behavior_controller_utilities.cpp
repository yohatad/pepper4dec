/* test_behavior_controller_utilities.cpp
 *
 * Unit tests for the non-ROS-runtime utilities in
 * behavior_controller_utilities.cpp: TextUtils, the environment knowledge
 * base validator, ConfigManager, and KnowledgeManager. YAML fixtures live in
 * test/fixtures/ (the path is injected at compile time as TEST_FIXTURE_DIR).
 *
 * ORDERING NOTE: ConfigManager and KnowledgeManager are process-wide
 * singletons, so their state persists across tests in this binary. All tests
 * that touch them live in the single `Singletons` suite below, in a
 * deliberate declaration order (gtest runs tests in declaration order by
 * default — do not run this binary with --gtest_shuffle). This order
 * dependence is a direct consequence of the singleton design; if the
 * managers ever become injectable, these tests can be made independent.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 18, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "behavior_controller/behavior_controller_interface.h"

namespace {
const std::string kFixtureDir = TEST_FIXTURE_DIR;
}

//=============================================================================
// TextUtils — pure functions, no ordering concerns
//=============================================================================

TEST(TextUtils, ToLowerCase) {
    EXPECT_EQ(TextUtils::toLowerCase("HeLLo World"), "hello world");
    EXPECT_EQ(TextUtils::toLowerCase("already lower"), "already lower");
    EXPECT_EQ(TextUtils::toLowerCase("MIXED 123 !?"), "mixed 123 !?");
    EXPECT_EQ(TextUtils::toLowerCase(""), "");
}

TEST(TextUtils, SplitSkipsEmptyTokens) {
    EXPECT_EQ(TextUtils::split("a,b,c", ','),
              (std::vector<std::string>{"a", "b", "c"}));
    // Consecutive/leading/trailing delimiters produce no empty tokens.
    EXPECT_EQ(TextUtils::split(",a,,b,", ','),
              (std::vector<std::string>{"a", "b"}));
    EXPECT_EQ(TextUtils::split("", ','), (std::vector<std::string>{}));
    EXPECT_EQ(TextUtils::split("single", ','),
              (std::vector<std::string>{"single"}));
}

TEST(TextUtils, Trim) {
    EXPECT_EQ(TextUtils::trim("  hello  "), "hello");
    EXPECT_EQ(TextUtils::trim("\t\n hi \r\n"), "hi");
    EXPECT_EQ(TextUtils::trim("no-trim"), "no-trim");
    EXPECT_EQ(TextUtils::trim("   "), "");
    EXPECT_EQ(TextUtils::trim(""), "");
    // Interior whitespace is preserved.
    EXPECT_EQ(TextUtils::trim("  a b  "), "a b");
}

// containsAnyWord uses \b word boundaries and lowercases both sides — this is
// what gates yes/no style intent matching in the behavior tree, so pin the
// semantics: substrings inside larger words must NOT match.
TEST(TextUtils, ContainsAnyWordUsesWordBoundaries) {
    const std::vector<std::string> words{"yes", "sure"};
    EXPECT_TRUE(TextUtils::containsAnyWord("Yes please", words));
    EXPECT_TRUE(TextUtils::containsAnyWord("oh SURE, why not", words));
    EXPECT_TRUE(TextUtils::containsAnyWord("yes", words));
    // "eyes" contains "yes" as a substring but not as a word.
    EXPECT_FALSE(TextUtils::containsAnyWord("my eyes hurt", words));
    EXPECT_FALSE(TextUtils::containsAnyWord("measure twice", words));
    EXPECT_FALSE(TextUtils::containsAnyWord("", words));
    EXPECT_FALSE(TextUtils::containsAnyWord("anything", {}));
}

//=============================================================================
// validateEnvironmentKnowledgeBase — stateless, takes a full file path
//=============================================================================

TEST(KnowledgeBaseValidator, AcceptsValidFixture) {
    EXPECT_TRUE(behavior_controller::validateEnvironmentKnowledgeBase(
        kFixtureDir + "/data/testEnvKB.yaml"));
}

TEST(KnowledgeBaseValidator, AcceptsShippedKnowledgeBases) {
    // The KBs actually deployed on the robot must always validate — this
    // catches accidental edits to the production data files.
    EXPECT_TRUE(behavior_controller::validateEnvironmentKnowledgeBase(
        kFixtureDir + "/../../data/decEnvironmentKnowledgeBase.yaml"));
    EXPECT_TRUE(behavior_controller::validateEnvironmentKnowledgeBase(
        kFixtureDir + "/../../data/labEnvironmentKnowledgeBase.yaml"));
}

TEST(KnowledgeBaseValidator, RejectsTourReferencingUndefinedLocation) {
    EXPECT_FALSE(behavior_controller::validateEnvironmentKnowledgeBase(
        kFixtureDir + "/data/invalid_tour_ref.yaml"));
}

TEST(KnowledgeBaseValidator, RejectsThetaOutOfRange) {
    EXPECT_FALSE(behavior_controller::validateEnvironmentKnowledgeBase(
        kFixtureDir + "/data/theta_out_of_range.yaml"));
}

TEST(KnowledgeBaseValidator, RejectsMissingFile) {
    EXPECT_FALSE(behavior_controller::validateEnvironmentKnowledgeBase(
        kFixtureDir + "/data/does_not_exist.yaml"));
}

//=============================================================================
// Singletons — order-dependent, see file header note
//=============================================================================

// Declared FIRST among singleton tests: must run before any loadFromPackage.
TEST(Singletons, KnowledgeManagerThrowsBeforeLoad) {
    EXPECT_THROW(KnowledgeManager::instance().getUtilityPhrase("1"),
                 std::runtime_error);
    EXPECT_THROW(KnowledgeManager::instance().getLocationInfo("1"),
                 std::runtime_error);
    EXPECT_THROW(KnowledgeManager::instance().getTourSpecification(),
                 std::runtime_error);
}

TEST(Singletons, ConfigManagerLoadsGoodConfig) {
    auto& config = ConfigManager::instance();
    ASSERT_TRUE(config.loadFromFile(kFixtureDir + "/config_good.yaml"));
    EXPECT_EQ(config.getScenarioSpecification(), "test_tour");
    EXPECT_EQ(config.getCultureKnowledgeBasePath(), "testCultureKB.yaml");
    EXPECT_EQ(config.getEnvironmentKnowledgeBasePath(), "testEnvKB.yaml");
    EXPECT_FALSE(config.isVerbose());
}

TEST(Singletons, ConfigManagerAppliesDefaultsForMissingKeys) {
    auto& config = ConfigManager::instance();
    ASSERT_TRUE(config.loadFromFile(kFixtureDir + "/config_minimal.yaml"));
    EXPECT_TRUE(config.isVerbose());
    // Absent keys fall back to the documented defaults, replacing whatever
    // the previous load set.
    EXPECT_EQ(config.getScenarioSpecification(), "lab_tour");
    EXPECT_EQ(config.getCultureKnowledgeBasePath(), "cultureKnowledgeBase.yaml");
    EXPECT_EQ(config.getEnvironmentKnowledgeBasePath(),
              "labEnvironmentKnowledgeBase.yaml");
}

TEST(Singletons, ConfigManagerRejectsMissingFile) {
    EXPECT_FALSE(ConfigManager::instance().loadFromFile(
        kFixtureDir + "/no_such_config.yaml"));
}

TEST(Singletons, KnowledgeManagerLoadsFixturePackage) {
    // Point the config at the test KBs, then load the fixture "package"
    // (loadFromPackage appends /data/<filename> itself).
    ASSERT_TRUE(ConfigManager::instance().loadFromFile(
        kFixtureDir + "/config_good.yaml"));
    ASSERT_TRUE(KnowledgeManager::instance().loadFromPackage(kFixtureDir));

    auto& km = KnowledgeManager::instance();

    // Utility phrases: integer and string keys both usable as string IDs.
    EXPECT_EQ(km.getUtilityPhrase("1"), "Hello from the test fixture");
    EXPECT_EQ(km.getUtilityPhrase("greeting"), "Welcome to the test lab");
    EXPECT_THROW(km.getUtilityPhrase("nonexistent"), std::runtime_error);

    // Locations: full field round trip from YAML to LocationInfo.
    auto start = km.getLocationInfo("1");
    EXPECT_EQ(start.description, "Test starting location");
    EXPECT_DOUBLE_EQ(start.robotPose.x, 4.0);
    EXPECT_DOUBLE_EQ(start.robotPose.y, 0.5);
    EXPECT_DOUBLE_EQ(start.robotPose.theta, 90.0);
    EXPECT_DOUBLE_EQ(start.gestureTarget.x, 1.0);
    EXPECT_DOUBLE_EQ(start.gestureTarget.y, 2.0);
    EXPECT_DOUBLE_EQ(start.gestureTarget.z, 0.75);
    EXPECT_EQ(start.gestureMessage, "This is the test start.");

    auto booth = km.getLocationInfo("booth");
    EXPECT_EQ(booth.description, "Test booth");
    EXPECT_EQ(booth.gestureMessage, "");  // empty message is allowed

    EXPECT_THROW(km.getLocationInfo("ghost"), std::runtime_error);

    // Tour specification preserves order and coerces integer IDs to strings.
    auto tour = km.getTourSpecification();
    ASSERT_EQ(tour.getLocationCount(), 2u);
    EXPECT_EQ(tour.locationIds[0], "1");
    EXPECT_EQ(tour.locationIds[1], "booth");
}
