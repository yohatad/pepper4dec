/* test_cloud_filters.cpp
 *
 * Unit tests for the cloud_range_filter filters (cloud_filters.h), on
 * synthetic clouds: a floor with obstacles on it for the ground fit, and a
 * dense cluster beside lone spikes for the outlier filter.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: October 11, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "pepper_slam/cloud_filters.h"

using pepper_slam::GroundPlaneParams;
using pepper_slam::KeepMask;
using pepper_slam::Point;

namespace
{

std::size_t kept(const KeepMask & mask)
{
  std::size_t n = 0;
  for (const auto k : mask) {
    n += k;
  }
  return n;
}

// Floor at height floor_z (scattered by noise) plus box_points of obstacles
// standing 0.15-1.5 m above it.
struct Scene
{
  std::vector<Point> points;
  std::vector<bool> is_floor;
};

Scene floorWithBoxes(double floor_z, double tilt_rad, int box_points, std::mt19937 & rng)
{
  std::uniform_real_distribution<double> xy(-5.0, 5.0);
  std::normal_distribution<double> noise(0.0, 0.01);
  Scene s;
  for (int i = 0; i < 3000; ++i) {
    const double x = xy(rng), y = xy(rng);
    // A slight tilt about y, as from a slightly mis-levelled sensor.
    s.points.push_back({x, y, floor_z + std::tan(tilt_rad) * x + noise(rng)});
    s.is_floor.push_back(true);
  }
  std::uniform_real_distribution<double> h(0.15, 1.5);
  for (int i = 0; i < box_points; ++i) {
    const double x = xy(rng), y = xy(rng);
    s.points.push_back({x, y, floor_z + std::tan(tilt_rad) * x + h(rng)});
    s.is_floor.push_back(false);
  }
  return s;
}

}  // namespace

TEST(RangeKeep, CutsBelowMinAndAboveMax)
{
  const std::vector<Point> pts = {
    {0.1, 0.0, 0.0}, {0.0, 0.3, 0.0}, {0.0, 0.0, 1.0}, {3.0, 4.0, 0.0}, {12.0, 0.0, 0.0}};
  EXPECT_EQ(pepper_slam::rangeKeep(pts, 0.22, 0.0), (KeepMask{0, 1, 1, 1, 1}));
  EXPECT_EQ(pepper_slam::rangeKeep(pts, 0.22, 5.0), (KeepMask{0, 1, 1, 1, 0}));
}

TEST(RangeKeep, BoundaryIsInclusive)
{
  const std::vector<Point> pts = {{0.8, 0.0, 0.0}, {0.0, 0.0, 2.0}};
  EXPECT_EQ(pepper_slam::rangeKeep(pts, 0.8, 2.0), (KeepMask{1, 1}));
}

TEST(SideBodyKeep, DisabledKeepsEverything)
{
  const std::vector<Point> pts = {{0.0, 0.3, 0.0}, {0.0, -0.3, 0.0}};
  EXPECT_EQ(pepper_slam::sideBodyKeep(pts, {0.0, 0.0, 0.0}, 0.0, 0.44), (KeepMask{1, 1}));
}
TEST(SideBodyKeep, CutsCloseFlanksKeepsForwardAndFar)
{
  const double cone = 25.0 * M_PI / 180.0;
  const std::vector<Point> pts = {
    {0.30, 0.0, 0.0},     // close, straight ahead -- kept
    {0.0, 0.30, 0.0},     // close, left flank -- cut
    {0.0, -0.30, 0.0},    // close, right flank -- cut
    {-0.30, 0.0, 0.0},    // close, behind -- cut
    {0.0, 2.0, 0.0},      // far, to the side -- kept
    {0.30, 0.12, 0.0},    // close, 21.8 deg off -- inside the cone, kept
    {0.30, 0.16, 0.0},    // close, 28.1 deg off -- outside, cut
  };
  EXPECT_EQ(
    pepper_slam::sideBodyKeep(pts, {0.0, 0.0, 0.0}, 0.35, cone),
    (KeepMask{1, 0, 0, 0, 1, 1, 0}));
}
TEST(SideBodyKeep, RangeAndBearingAreMeasuredFromTheSensor)
{
  // Sensor forward of the body origin: the cut follows the sensor, but the
  // bearing is still the body's +x.
  const double cone = 25.0 * M_PI / 180.0;
  const std::vector<Point> pts = {{0.50, 0.30, 0.0}, {0.50, 0.0, 0.0}};
  const Point sensor{0.50, 0.0, 0.0};
  EXPECT_EQ(pepper_slam::sideBodyKeep(pts, sensor, 0.35, cone), (KeepMask{0, 1}));
  // Same points measured from the body origin are far enough to survive.
  EXPECT_EQ(pepper_slam::sideBodyKeep(pts, {0.0, 0.0, 0.0}, 0.35, cone), (KeepMask{1, 1}));
}
TEST(RadiusOutlierKeep, DisabledOrTinyCloudKeepsEverything)
{
  const std::vector<Point> pts = {{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, {20.0, 0.0, 0.0}};
  EXPECT_EQ(kept(pepper_slam::radiusOutlierKeep(pts, 0, 0.5)), 3u);
  EXPECT_EQ(kept(pepper_slam::radiusOutlierKeep(pts, 3, 0.5)), 3u);
}

TEST(RadiusOutlierKeep, DropsIsolatedPointsKeepsCluster)
{
  std::vector<Point> pts;
  for (int i = 0; i < 10; ++i) {
    pts.push_back({1.0 + 0.02 * i, 0.0, 0.0});         // a tight line of 10
  }
  pts.push_back({5.0, 5.0, 5.0});                      // lone spikes
  pts.push_back({-3.0, 0.0, 1.0});
  const KeepMask keep = pepper_slam::radiusOutlierKeep(pts, 3, 0.1);
  for (int i = 0; i < 10; ++i) {
    EXPECT_TRUE(keep[i]) << "cluster point " << i;
  }
  EXPECT_FALSE(keep[10]);
  EXPECT_FALSE(keep[11]);
}

TEST(RadiusOutlierKeep, MatchesBruteForceAcrossCellBoundaries)
{
  // Points straddling negative and positive cell indices, where an
  // off-by-one in the grid lookup would show up as a missed neighbour.
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<Point> pts;
  for (int i = 0; i < 400; ++i) {
    pts.push_back({u(rng), u(rng), u(rng)});
  }
  const int k = 4;
  const double r = 0.2;
  const KeepMask keep = pepper_slam::radiusOutlierKeep(pts, k, r);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    int count = 0;
    for (std::size_t j = 0; j < pts.size(); ++j) {
      const double dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y, dz = pts[i].z - pts[j].z;
      count += j != i && dx * dx + dy * dy + dz * dz <= r * r;
    }
    EXPECT_EQ(keep[i] != 0, count >= k) << "point " << i;
  }
}

TEST(GroundPlaneKeep, RemovesFloorKeepsObstacles)
{
  std::mt19937 rng(1);
  const Scene s = floorWithBoxes(0.03, 0.02, 1000, rng);
  GroundPlaneParams params;
  params.z_thresh = 0.20;
  params.iterations = 300;
  const KeepMask keep = pepper_slam::groundPlaneKeep(s.points, params, rng);

  std::size_t floor_kept = 0, boxes_kept = 0, floor_total = 0;
  for (std::size_t i = 0; i < s.points.size(); ++i) {
    floor_total += s.is_floor[i];
    (s.is_floor[i] ? floor_kept : boxes_kept) += keep[i];
  }
  EXPECT_LT(floor_kept, floor_total / 50);   // > 98% of the floor removed
  EXPECT_EQ(boxes_kept, 1000u);              // boxes start 0.15 m up, outside the band
}

TEST(GroundPlaneKeep, FloorTooFarFromZeroIsKept)
{
  // The only plane sits 0.5 m below the ground frame's z = 0: a wrong TF,
  // not the floor. Fail safe and keep everything.
  std::mt19937 rng(2);
  const Scene s = floorWithBoxes(-0.5, 0.0, 0, rng);
  GroundPlaneParams params;
  params.z_thresh = 0.20;
  params.iterations = 300;
  EXPECT_EQ(kept(pepper_slam::groundPlaneKeep(s.points, params, rng)), s.points.size());
}

TEST(GroundPlaneKeep, WallIsNotGround)
{
  // A vertical wall through the origin: its normal is horizontal, so it is
  // rejected however many points it has.
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> u(-3.0, 3.0);
  std::vector<Point> wall;
  for (int i = 0; i < 2000; ++i) {
    wall.push_back({0.0, u(rng), u(rng)});
  }
  GroundPlaneParams params;
  params.iterations = 300;
  EXPECT_EQ(kept(pepper_slam::groundPlaneKeep(wall, params, rng)), wall.size());
}

TEST(GroundPlaneKeep, TooFewInliersKeepsEverything)
{
  std::mt19937 rng(4);
  const std::vector<Point> pts = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 1.0, 0.01}, {0.5, 0.5, 1.0}};
  GroundPlaneParams params;          // min_points 30 > 5 points
  EXPECT_EQ(kept(pepper_slam::groundPlaneKeep(pts, params, rng)), pts.size());
  params.min_points = 3;
  EXPECT_EQ(kept(pepper_slam::groundPlaneKeep(pts, params, rng)), 1u);   // only the 1 m point
}

TEST(GroundPlaneKeep, FewerThanThreePointsKeepsEverything)
{
  std::mt19937 rng(5);
  const std::vector<Point> pts = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  EXPECT_EQ(kept(pepper_slam::groundPlaneKeep(pts, GroundPlaneParams{}, rng)), 2u);
}

TEST(ApplyMask, KeepsOrder)
{
  const std::vector<Point> pts = {{1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};
  const auto out = pepper_slam::applyMask(pts, KeepMask{1, 0, 1});
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].x, 1.0);
  EXPECT_EQ(out[1].x, 3.0);
}
