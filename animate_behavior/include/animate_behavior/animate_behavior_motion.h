/* animate_behavior_motion.h
 *
 * Pure motion math for animate_behavior: joint-limit clamping for randomized
 * gesture targets, and the exponential smoothing applied between the current
 * joint position and its target before publishing.
 *
 * Deliberately free of ROS and of the randomness/wall-clock the node itself
 * carries, so the arithmetic that decides what the arms actually do can be
 * tested directly (mirrors gesture_pepper_kinematics.h in gesture_execution).
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Aug 16, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#pragma once

namespace animate_behavior_motion {

/**
 * @brief Clamp a joint value into its soft limits.
 *
 * If min > max the limits are inverted (a malformed joint definition); the
 * lower bound wins so the result is still one of the two supplied values
 * rather than an unbounded target.
 */
double clampToLimits(double value, double min_limit, double max_limit);

/**
 * @brief Compute a randomized gesture target for one joint, clamped to limits.
 *
 * @param home       Neutral pose for the joint.
 * @param noise      Random value in [-1, 1], supplied by the caller so this
 *                   stays deterministic and testable.
 * @param range      Overall gesture amplitude (the action's `range` field).
 * @param factor     Per-joint scaling from the limb's JointDef.
 * @param min_limit  Soft lower limit.
 * @param max_limit  Soft upper limit.
 */
double gestureTarget(double home, double noise, double range, double factor,
                     double min_limit, double max_limit);

/**
 * @brief One step of exponential smoothing from current toward target.
 *
 * factor 0 holds position, factor 1 jumps straight to the target. Values in
 * between approach it asymptotically; this is what keeps the published joint
 * stream from stepping discontinuously when a new gesture target is picked.
 */
double smoothToward(double current, double target, double factor);

}  // namespace animate_behavior_motion
