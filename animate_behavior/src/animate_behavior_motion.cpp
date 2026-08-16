/* animate_behavior_motion.cpp
 *
 * Pure motion math for animate_behavior — see animate_behavior_motion.h.
 * Extracted from randomGesture()/publishJoints() in
 * animate_behavior_implementation.cpp so the arithmetic is testable without a
 * ROS graph, a robot, or the node's RNG and wall-clock.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Aug 16, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include "animate_behavior/animate_behavior_motion.h"

#include <algorithm>

namespace animate_behavior_motion {

double clampToLimits(double value, double min_limit, double max_limit) {
    // Ordering matches the original std::max(min, std::min(max, value)):
    // with inverted limits the lower bound wins.
    return std::max(min_limit, std::min(max_limit, value));
}

double gestureTarget(double home, double noise, double range, double factor,
                     double min_limit, double max_limit) {
    return clampToLimits(home + noise * range * factor, min_limit, max_limit);
}

double smoothToward(double current, double target, double factor) {
    return current + factor * (target - current);
}

}  // namespace animate_behavior_motion
