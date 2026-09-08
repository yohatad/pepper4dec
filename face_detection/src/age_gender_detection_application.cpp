/* age_gender_detection_application.cpp
 *
 * Entry point for the MiVOLO age/gender estimation lifecycle node. Loads
 * configuration, spins the node, and cleans up (stopping the estimation
 * worker thread) on shutdown; the node classes themselves are implemented
 * in age_gender_detection_implementation.cpp.
 *
 * Subscribers:
 *   <image_topic> (sensor_msgs/msg/Image)
 *     Raw color camera frames, cached for crop extraction.
 *   <face_topic> (dec_interfaces/msg/FaceDetection)
 *     Per-frame face tracking results, incl. the mutual_gaze flag.
 *   <person_topic> (dec_interfaces/msg/PersonDetection)
 *     Tracked person detections, used for the body crop.
 *
 * Publishers:
 *   <output_topic> (std_msgs/msg/String)
 *     Per-person JSON profile: label_id, age, gender, gender_confidence,
 *     estimation_count, person_bbox.
 *
 * Parameters (config/age_gender_detection_configuration.yaml, under
 * age_gender_detection/ros__parameters):
 *   mivolo_model_path (string, default: <face_detection share dir>/models/
 *                      face_detection_mivolo_agegender.onnx)
 *   device (string, default: "cuda")
 *   face_only (bool, default: false)
 *   face_topic (string, default: "/face_detection/data")
 *   person_topic (string, default: "/person_detection/data")
 *   image_topic (string, default: "/camera/color/image_raw")
 *   output_topic (string, default: "/face_detection/age_gender_results")
 *   max_cache_age_sec (double, default: 2.0)
 *   min_estimate_interval_sec (double, default: 0.5)
 *   re_estimate_interval_sec (double, default: 30.0)
 *   person_class_name (string, default: "person")
 *   max_depth_m (double, default: 4.0)
 *
 * Lifecycle:
 *   configure  -> load the MiVOLO ONNX model, reset caches
 *   activate   -> create the publisher and subscriptions, start the estimation
 *                 worker thread, start the cleanup/debug timers
 *   deactivate -> destroy subscriptions, stop and join the worker thread
 *   cleanup    -> release the ONNX session, clear caches
 *   shutdown   -> log shutdown
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 29, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include "face_detection/age_gender_detection_interface.h"

#include "dec_common/node_runner.h"

int main(int argc, char** argv) {
    return dec_common::runNode<AgeGenderDetectionNode>(
        argc, argv,
        {"age_gender_detection v1.0 — This program comes with ABSOLUTELY NO WARRANTY.", "age_gender_detection"},
        nullptr,
        [](AgeGenderDetectionNode& node) { node.cleanup(); });
}
