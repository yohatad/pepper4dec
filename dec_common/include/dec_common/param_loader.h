/* param_loader.h
 *
 * Shared helper for declaring and reading ROS2 lifecycle-node parameters.
 * Declaring a parameter with a default makes it visible to `ros2 param
 * get/set` and lets rclcpp populate it from the launch file's YAML.
 *
 * A wrong-typed value throws InvalidParameterTypeException from
 * declare_parameter, which is intentional: config errors should fail node
 * startup loudly rather than being silently swallowed.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 23, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#pragma once

#include <string>

#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace dec_common
{

template<typename T>
T declareAndGetParameter(
  rclcpp_lifecycle::LifecycleNode * node, const std::string & name,
  const T & default_value)
{
  if (!node->has_parameter(name)) {
    node->declare_parameter(name, default_value);
  }
  return node->get_parameter(name).get_value<T>();
}

}  // namespace dec_common
