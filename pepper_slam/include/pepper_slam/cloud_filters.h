/* cloud_filters.h
 *
 * The point filters behind cloud_range_filter, kept free of ROS so the tests
 * can drive them with synthetic clouds:
 *
 *   rangeKeep          -- drop the robot's own body (and optionally far returns)
 *   sideBodyKeep       -- drop close returns outside the forward cone
 *   bodyBoxKeep        -- drop returns inside boxes on the robot's own outline
 *   radiusOutlierKeep  -- drop isolated spike returns
 *   groundPlaneKeep    -- RANSAC-fit and drop the near-z=0 floor plane
 *
 * Each returns a keep-mask (1 = keep) parallel to its input, so the node can
 * apply them in sequence to the points it will publish while the ground fit
 * runs on a transformed copy.
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

#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace pepper_slam
{

struct Point
{
  double x;
  double y;
  double z;
};

using KeepMask = std::vector<std::uint8_t>;

/// Keep points whose distance from the sensor origin is at least min_range
/// and, when max_range > 0, at most max_range.
KeepMask rangeKeep(const std::vector<Point> & points, double min_range, double max_range);

/// Keep points unless they are both closer to sensor than cut_range and
/// bearing further than cut_angle (rad) off the body's +x. points and sensor
/// must be in the body frame. cut_range <= 0 keeps everything.
KeepMask sideBodyKeep(
  const std::vector<Point> & points, const Point & sensor, double cut_range, double cut_angle);

/// Axis-aligned box in the body frame (m), bounds inclusive.
struct Box
{
  double x_min;
  double x_max;
  double y_min;
  double y_max;
  double z_min;
  double z_max;
};

/// Keep points outside every box. points must be in the body frame. No
/// boxes keeps everything.
KeepMask bodyBoxKeep(const std::vector<Point> & points, const std::vector<Box> & boxes);

/// Keep points with at least min_neighbors other points within radius.
/// min_neighbors <= 0, or a cloud of at most min_neighbors points, keeps all.
KeepMask radiusOutlierKeep(const std::vector<Point> & points, int min_neighbors, double radius);

struct GroundPlaneParams
{
  double distance_thresh = 0.05;  // inlier band around the fitted plane (m)
  double angle_thresh = 0.15;     // plane normal vs +Z tolerance (rad)
  double z_thresh = 0.12;         // plane must sit within this of z = 0 (m)
  int iterations = 60;
  int min_points = 30;            // fewer inliers than this: keep everything
};

/// Drop the inliers of the dominant near-horizontal plane near z = 0.
/// points must already be in the ground frame. Fails safe: when no plane
/// satisfies the constraints with at least min_points inliers, keeps all.
KeepMask groundPlaneKeep(
  const std::vector<Point> & points, const GroundPlaneParams & params,
  std::mt19937 & rng);

/// Copy of points with only the entries mask keeps.
std::vector<Point> applyMask(const std::vector<Point> & points, const KeepMask & mask);

}  // namespace pepper_slam
