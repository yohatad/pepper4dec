/* rig_transforms.cpp
 *
 * See include/pepper_slam/rig_transforms.h.
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

#include "pepper_slam/rig_transforms.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

namespace pepper_slam
{

namespace
{

template<std::size_t N>
std::array<double, N> readVector(const YAML::Node & node, const char * key)
{
  const YAML::Node values = node[key];
  if (!values || !values.IsSequence() || values.size() != N) {
    throw std::invalid_argument(
            std::string(key) + " must be a list of " + std::to_string(N) + " numbers");
  }
  std::array<double, N> out{};
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = values[i].as<double>();
  }
  return out;
}

std::string readString(const YAML::Node & node, const char * key)
{
  const YAML::Node value = node[key];
  if (!value || !value.IsScalar()) {
    throw std::invalid_argument(std::string("missing ") + key);
  }
  return value.as<std::string>();
}

}  // namespace

RigTransforms parseRigTransforms(
  const std::string & yaml_text, const std::string & scope, const std::string & source)
{
  if (scope != "mount" && scope != "all") {
    throw RigTransformError("scope must be 'mount' or 'all', got '" + scope + "'.");
  }

  YAML::Node doc;
  try {
    doc = YAML::Load(yaml_text);
  } catch (const YAML::Exception & ex) {
    throw RigTransformError("Cannot parse '" + source + "': " + ex.what());
  }
  const YAML::Node entries = doc.IsMap() ? doc["transforms"] : YAML::Node();
  if (!entries || !entries.IsSequence() || entries.size() == 0) {
    throw RigTransformError("'" + source + "' has no non-empty 'transforms:' list.");
  }

  RigTransforms out;
  std::map<std::string, std::string> parent_of;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const YAML::Node e = entries[i];
    if (scope == "mount" && e.IsMap() && e["owner"] &&
      e["owner"].as<std::string>("") == "driver")
    {
      ++out.skipped_driver_edges;
      continue;
    }

    RigTransform t;
    try {
      if (!e.IsMap()) {
        throw std::invalid_argument("not a mapping");
      }
      t.parent = readString(e, "parent");
      t.child = readString(e, "child");
      t.xyz = readVector<3>(e, "xyz");
      t.qxyzw = readVector<4>(e, "qxyzw");
    } catch (const std::exception & ex) {
      throw RigTransformError(
              "transform #" + std::to_string(i) + " in '" + source + "' is malformed (" +
              ex.what() + "); each entry needs parent, child, xyz[3], qxyzw[4].");
    }

    // A frame with two parents silently breaks the tree, and the error
    // surfaces far away as "not part of the same tree". Catch it here.
    const auto seen = parent_of.find(t.child);
    if (seen != parent_of.end()) {
      throw RigTransformError(
              "'" + t.child + "' is given two parents ('" + seen->second + "' and '" + t.parent +
              "') in '" + source + "'. Each frame may have exactly one.");
    }
    parent_of[t.child] = t.parent;

    auto & q = t.qxyzw;
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n < 1e-9) {
      throw RigTransformError(t.parent + " -> " + t.child + ": zero-length quaternion.");
    }
    if (std::abs(n - 1.0) > 1e-3) {
      char norm[32];
      std::snprintf(norm, sizeof(norm), "%.6f", n);
      out.warnings.push_back(
        t.parent + " -> " + t.child + ": quaternion norm " + norm +
        " != 1; normalising. Check the calibration source.");
    }
    for (auto & c : q) {
      c /= n;
    }
    out.transforms.push_back(t);
  }
  return out;
}

RigTransforms loadRigTransforms(const std::string & path, const std::string & scope)
{
  if (path.empty()) {
    throw RigTransformError("transforms_file parameter is empty.");
  }
  std::ifstream in(path);
  if (!in) {
    throw RigTransformError("Cannot read transforms_file '" + path + "'.");
  }
  std::stringstream text;
  text << in.rdbuf();
  return parseRigTransforms(text.str(), scope, path);
}

}  // namespace pepper_slam
