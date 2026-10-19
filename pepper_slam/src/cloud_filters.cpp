/* cloud_filters.cpp
 *
 * Implements the point filters declared in cloud_filters.h.
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

#include "pepper_slam/cloud_filters.h"

#include <cmath>
#include <cstddef>
#include <unordered_map>

namespace pepper_slam
{

KeepMask rangeKeep(const std::vector<Point> & points, double min_range, double max_range)
{
  KeepMask keep(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Point & p = points[i];
    const double r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    keep[i] = r >= min_range && (max_range <= 0.0 || r <= max_range);
  }
  return keep;
}

KeepMask sideBodyKeep(
  const std::vector<Point> & points, const Point & sensor, double cut_range, double cut_angle)
{
  if (cut_range <= 0.0) {
    return KeepMask(points.size(), 1);
  }
  KeepMask keep(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const double dx = points[i].x - sensor.x;
    const double dy = points[i].y - sensor.y;
    const double dz = points[i].z - sensor.z;
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    // Bearing off the body's +x, so the forward cone survives at every range:
    // only the flanks, where the lidar sees Pepper itself, are cut.
    keep[i] = r >= cut_range || std::fabs(std::atan2(dy, dx)) <= cut_angle;
  }
  return keep;
}

KeepMask radiusOutlierKeep(const std::vector<Point> & points, int min_neighbors, double radius)
{
  const std::size_t n = points.size();
  if (min_neighbors <= 0 || n <= static_cast<std::size_t>(min_neighbors) || radius <= 0.0) {
    return KeepMask(n, 1);
  }

  // Bucket the points into cubes of side `radius`: every neighbour within
  // radius then lies in the point's own cube or one of the 26 around it.
  // 21 bits per axis wrap only 2^21 cubes apart, so the 27 cubes searched
  // never share a key; a wrapped collision with a far cube only costs an
  // extra distance check.
  const auto cell = [radius](double v) {
      return static_cast<std::int64_t>(std::floor(v / radius));
    };
  const auto key = [](std::int64_t cx, std::int64_t cy, std::int64_t cz) {
      constexpr std::uint64_t mask = (1ULL << 21) - 1;
      return ((static_cast<std::uint64_t>(cx) & mask) << 42) |
             ((static_cast<std::uint64_t>(cy) & mask) << 21) |
             (static_cast<std::uint64_t>(cz) & mask);
    };
  std::unordered_map<std::uint64_t, std::vector<std::size_t>> grid;
  grid.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    grid[key(cell(points[i].x), cell(points[i].y), cell(points[i].z))].push_back(i);
  }

  const double r2 = radius * radius;
  KeepMask keep(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const Point & p = points[i];
    const std::int64_t cx = cell(p.x), cy = cell(p.y), cz = cell(p.z);
    int count = 0;
    for (int dx = -1; dx <= 1 && count < min_neighbors; ++dx) {
      for (int dy = -1; dy <= 1 && count < min_neighbors; ++dy) {
        for (int dz = -1; dz <= 1 && count < min_neighbors; ++dz) {
          const auto it = grid.find(key(cx + dx, cy + dy, cz + dz));
          if (it == grid.end()) {
            continue;
          }
          for (const std::size_t j : it->second) {
            if (j == i) {
              continue;
            }
            const double ex = points[j].x - p.x;
            const double ey = points[j].y - p.y;
            const double ez = points[j].z - p.z;
            if (ex * ex + ey * ey + ez * ez <= r2 && ++count >= min_neighbors) {
              break;
            }
          }
        }
      }
    }
    keep[i] = count >= min_neighbors;
  }
  return keep;
}

KeepMask groundPlaneKeep(
  const std::vector<Point> & points, const GroundPlaneParams & params,
  std::mt19937 & rng)
{
  const std::size_t n = points.size();
  if (n < 3) {
    return KeepMask(n, 1);
  }

  const double cos_thresh = std::cos(params.angle_thresh);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);
  int best_count = 0;
  double best_nx = 0.0, best_ny = 0.0, best_nz = 0.0, best_d = 0.0;

  for (int iter = 0; iter < params.iterations; ++iter) {
    const std::size_t i0 = pick(rng);
    std::size_t i1, i2;
    do {
      i1 = pick(rng);
    } while (i1 == i0);
    do {
      i2 = pick(rng);
    } while (i2 == i0 || i2 == i1);
    const Point & p0 = points[i0];
    const Point & p1 = points[i1];
    const Point & p2 = points[i2];

    // normal = (p1 - p0) x (p2 - p0)
    const double ux = p1.x - p0.x, uy = p1.y - p0.y, uz = p1.z - p0.z;
    const double vx = p2.x - p0.x, vy = p2.y - p0.y, vz = p2.z - p0.z;
    double nx = uy * vz - uz * vy;
    double ny = uz * vx - ux * vz;
    double nz = ux * vy - uy * vx;
    const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (norm < 1e-9) {
      continue;                                   // collinear sample
    }
    nx /= norm;
    ny /= norm;
    nz /= norm;
    if (nz < 0.0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    if (nz < cos_thresh) {
      continue;                                   // not near-horizontal
    }
    const double d = -(nx * p0.x + ny * p0.y + nz * p0.z);
    if (std::abs(d) > params.z_thresh) {
      continue;                                   // not near z = 0
    }

    int count = 0;
    for (const Point & p : points) {
      count += std::abs(nx * p.x + ny * p.y + nz * p.z + d) <= params.distance_thresh;
    }
    if (count > best_count) {
      best_count = count;
      best_nx = nx;
      best_ny = ny;
      best_nz = nz;
      best_d = d;
    }
  }

  // best_count stays 0 when no sample passed the constraints.
  if (best_count == 0 || best_count < params.min_points) {
    return KeepMask(n, 1);
  }
  KeepMask keep(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Point & p = points[i];
    keep[i] =
      std::abs(best_nx * p.x + best_ny * p.y + best_nz * p.z + best_d) > params.distance_thresh;
  }
  return keep;
}

std::vector<Point> applyMask(const std::vector<Point> & points, const KeepMask & mask)
{
  std::vector<Point> out;
  out.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    if (mask[i]) {
      out.push_back(points[i]);
    }
  }
  return out;
}

}  // namespace pepper_slam
