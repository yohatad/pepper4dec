/* param_loader.h
 *
 * Shared helper for declaring and reading ROS2 lifecycle-node parameters.
 * Replaces the pattern of manually parsing a node's config YAML with
 * yaml-cpp: declaring the parameter (with a default) makes it visible to
 * `ros2 param get/set` and lets rclcpp's own parameter-loading machinery
 * (fed by `parameters=[...]` in the launch file) populate it from YAML.
 *
 * A malformed value in the YAML (wrong type) throws
 * rclcpp::exceptions::InvalidParameterTypeException from declare_parameter,
 * which is intentional: config errors should fail node startup loudly
 * rather than being silently swallowed.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 */

#ifndef DEC_COMMON_PARAM_LOADER_H
#define DEC_COMMON_PARAM_LOADER_H

#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <string>

namespace dec_common {

template <typename T>
T declareAndGetParameter(rclcpp_lifecycle::LifecycleNode* node, const std::string& name,
                         const T& default_value) {
    if (!node->has_parameter(name)) {
        node->declare_parameter(name, default_value);
    }
    return node->get_parameter(name).get_value<T>();
}

}  // namespace dec_common

#endif  // DEC_COMMON_PARAM_LOADER_H
