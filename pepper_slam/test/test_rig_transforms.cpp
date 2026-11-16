/* test_rig_transforms.cpp
 *
 * static_tf_publisher's parser (rig_transforms.cpp) on hand-written YAML and on
 * the shipped config/sensor_tf.yaml.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: November 17, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <string>

#include "pepper_slam/rig_transforms.h"

using pepper_slam::parseRigTransforms;
using pepper_slam::RigTransformError;

namespace
{

const char * kRig =
  R"(
transforms:
  - {parent: base_footprint, child: l2lidar_frame,
     xyz: [0.133, 0.0, 0.2582], qxyzw: [0.0, 0.0, 0.0, 1.0]}
  - {parent: l2lidar_frame, child: camera_link, owner: driver,
     xyz: [0.01, 0.02, 0.03], qxyzw: [0.0, 0.0, 0.0, 1.0]}
)";

}  // namespace

TEST(RigTransforms, MountLeavesDriverEdgesToTheDriver)
{
  const auto rig = parseRigTransforms(kRig, "mount");
  ASSERT_EQ(rig.transforms.size(), 1u);
  EXPECT_EQ(rig.skipped_driver_edges, 1u);
  EXPECT_EQ(rig.transforms[0].parent, "base_footprint");
  EXPECT_EQ(rig.transforms[0].child, "l2lidar_frame");
  EXPECT_DOUBLE_EQ(rig.transforms[0].xyz[2], 0.2582);
}

TEST(RigTransforms, AllPublishesEveryEdge)
{
  const auto rig = parseRigTransforms(kRig, "all");
  EXPECT_EQ(rig.transforms.size(), 2u);
  EXPECT_EQ(rig.skipped_driver_edges, 0u);
}

TEST(RigTransforms, NormalisesAndWarnsAboutANonUnitQuaternion)
{
  const auto rig = parseRigTransforms(
    "transforms:\n  - {parent: a, child: b, xyz: [0, 0, 0], qxyzw: [0, 0, 0, 2]}\n", "all");
  ASSERT_EQ(rig.transforms.size(), 1u);
  EXPECT_DOUBLE_EQ(rig.transforms[0].qxyzw[3], 1.0);
  ASSERT_EQ(rig.warnings.size(), 1u);
  EXPECT_NE(rig.warnings[0].find("2.000000"), std::string::npos);
}

TEST(RigTransforms, ANearlyUnitQuaternionIsNormalisedSilently)
{
  const auto rig = parseRigTransforms(
    "transforms:\n  - {parent: a, child: b, xyz: [0, 0, 0], qxyzw: [0, 0, 0, 1.0005]}\n",
    "all");
  EXPECT_TRUE(rig.warnings.empty());
  EXPECT_DOUBLE_EQ(rig.transforms[0].qxyzw[3], 1.0);
}

TEST(RigTransforms, RejectsAFrameWithTwoParents)
{
  EXPECT_THROW(
    parseRigTransforms(
      "transforms:\n"
      "  - {parent: a, child: c, xyz: [0, 0, 0], qxyzw: [0, 0, 0, 1]}\n"
      "  - {parent: b, child: c, xyz: [0, 0, 0], qxyzw: [0, 0, 0, 1]}\n", "all"),
    RigTransformError);
}

TEST(RigTransforms, RejectsAZeroQuaternion)
{
  EXPECT_THROW(
    parseRigTransforms(
      "transforms:\n  - {parent: a, child: b, xyz: [0, 0, 0], qxyzw: [0, 0, 0, 0]}\n", "all"),
    RigTransformError);
}

TEST(RigTransforms, RejectsMalformedEntries)
{
  // Missing child, short xyz, non-numeric quaternion, not a mapping.
  for (const char * entry : {
    "{parent: a, xyz: [0, 0, 0], qxyzw: [0, 0, 0, 1]}",
    "{parent: a, child: b, xyz: [0, 0], qxyzw: [0, 0, 0, 1]}",
    "{parent: a, child: b, xyz: [0, 0, 0], qxyzw: [0, 0, x, 1]}",
    "just a string"})
  {
    EXPECT_THROW(
      parseRigTransforms(std::string("transforms:\n  - ") + entry + "\n", "all"),
      RigTransformError) << entry;
  }
}

TEST(RigTransforms, RejectsAnEmptyOrMissingList)
{
  EXPECT_THROW(parseRigTransforms("transforms: []\n", "all"), RigTransformError);
  EXPECT_THROW(parseRigTransforms("other: 1\n", "all"), RigTransformError);
  EXPECT_THROW(parseRigTransforms("", "all"), RigTransformError);
}

TEST(RigTransforms, RejectsAnUnknownScope)
{
  EXPECT_THROW(parseRigTransforms(kRig, "everything"), RigTransformError);
}

TEST(RigTransforms, LoadReportsAMissingFile)
{
  EXPECT_THROW(pepper_slam::loadRigTransforms("", "mount"), RigTransformError);
  EXPECT_THROW(
    pepper_slam::loadRigTransforms("/nonexistent/sensor_tf.yaml", "mount"), RigTransformError);
}

// The shipped rig: one tree, unit quaternions, and the edges every launch needs.
TEST(RigTransforms, ShippedSensorTfIsAValidTree)
{
  const auto rig = pepper_slam::loadRigTransforms(SENSOR_TF_YAML, "all");
  EXPECT_TRUE(rig.warnings.empty());
  std::set<std::string> children;
  for (const auto & t : rig.transforms) {
    children.insert(t.child);
  }
  for (const char * frame : {"l2lidar_frame", "camera_camera_link", "l2lidar_frame_imu",
      "camera_imu_optical_frame"})
  {
    EXPECT_EQ(children.count(frame), 1u) << frame;
  }
  const auto mount = pepper_slam::loadRigTransforms(SENSOR_TF_YAML, "mount");
  EXPECT_LT(mount.transforms.size(), rig.transforms.size());
}
