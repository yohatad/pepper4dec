/* rig_transforms.h
 *
 * Reads the sensor rig's static transforms from data/sensor_tf.yaml, kept
 * free of ROS so the tests can drive it directly. static_tf_publisher publishes
 * the result as one latched /tf_static message.
 *
 * The file holds one key, `transforms`, a list of edges:
 *
 *   - {parent: base_footprint, child: l2lidar_frame,
 *      xyz: [x, y, z], qxyzw: [qx, qy, qz, qw], owner: driver}
 *
 * `owner: driver` (optional) marks an edge a sensor driver also publishes (the
 * RealSense's internal extrinsics). With scope "mount" those are left to the
 * driver: /tf_static is latched and tf2 keys it on the child frame, so two
 * publishers of one edge race and the last one silently wins. Scope "all"
 * publishes them too, for bags recorded without /tf_static.
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

#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace pepper_slam
{

struct RigTransform
{
  std::string parent;
  std::string child;
  std::array<double, 3> xyz;
  std::array<double, 4> qxyzw;  // normalised on load
};

struct RigTransforms
{
  std::vector<RigTransform> transforms;
  // Edges left to their driver under scope "mount".
  std::size_t skipped_driver_edges = 0;
  // Non-fatal problems to log, e.g. a quaternion that needed normalising.
  std::vector<std::string> warnings;
};

// The file or an entry is unusable. The message says which and why.
class RigTransformError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

// Parse YAML text. `scope` is "mount" or "all"; `source` names the text in
// error messages. Throws RigTransformError for a scope other than those two,
// a missing or empty `transforms` list, a malformed entry, a frame given two
// parents, or a zero-length quaternion.
RigTransforms parseRigTransforms(
  const std::string & yaml_text, const std::string & scope,
  const std::string & source = "<text>");

// The same, reading the file at `path`.
RigTransforms loadRigTransforms(const std::string & path, const std::string & scope);

}  // namespace pepper_slam
