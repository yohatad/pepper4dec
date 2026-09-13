/* person_detection_application.cpp
 *
 * Entry point for the Yolov11Node person detection lifecycle node. Loads
 * configuration, spins the node, and cleans up on shutdown; the node classes
 * themselves are implemented in person_detection_implementation.cpp.
 *
 * Subscribers:
 *   <camera color topic> (sensor_msgs/msg/Image or CompressedImage)
 *     Color camera frames (topic resolved from data/pepper_topics.yaml based
 *     on the configured camera type).
 *   <camera depth topic> (sensor_msgs/msg/Image or CompressedImage)
 *     Depth camera frames used to estimate distance to detected objects.
 *
 * Publishers:
 *   /person_detection/data (dec_interfaces/msg/PersonDetection)
 *     Tracked object detections: track IDs, class names/IDs, confidences,
 *     centroids (with depth), widths, and heights.
 *   /person_detection/debug (sensor_msgs/msg/Image)
 *     Annotated color image showing tracked bounding boxes, labels, and depth.
 *   /person_detection/depth_debug (sensor_msgs/msg/Image)
 *     Colorized visualization of the raw depth image.
 *
 * Parameters (config/person_detection_configuration.yaml, under
 * person_detection/ros__parameters):
 *   camera (string, default: "realsense")
 *   use_compressed (bool, default: false)
 *   image_timeout (double, default: 2.0)
 *   verbose_mode (bool, default: true)
 *   confidence_threshold (double, default: 0.5)
 *   target_classes (string[], default: ["person"]; ["all"] tracks every class)
 *   track_threshold (double, default: 0.45)
 *   track_buffer (int, default: 30)
 *   match_threshold (double, default: 0.8)
 *   frame_rate (int, default: 30)
 *
 * Lifecycle:
 *   configure  -> create lifecycle publishers, load camera/config settings
 *                 and target classes (base); load ONNX model + ByteTrack
 *                 (Yolov11Node)
 *   activate   -> start the visualization and status timers (base); create
 *                 camera subscriptions and start the timeout monitor
 *                 (Yolov11Node)
 *   deactivate -> cancel the visualization and status timers (base); destroy
 *                 camera subscriptions (Yolov11Node)
 *   cleanup    -> destroy the lifecycle publishers (base); release the ONNX
 *                 session (Yolov11Node)
 *   shutdown   -> log that the node is shutting down
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 6, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include "person_detection/person_detection_interface.h"

#include "dec_common/node_runner.h"

namespace {
constexpr const char* kBanner = R"(
================================================================================
                        Person Detection v1.0
================================================================================
  - YOLOv11 person detection with ByteTrack multi-object tracking
  - Configurable target classes via person_detection_configuration.yaml
  - Supported classes: person, car, bottle, chair, and 76 more COCO classes

  This program comes with ABSOLUTELY NO WARRANTY.
================================================================================
)";
}  // namespace

int main(int argc, char** argv) {
    return dec_common::runNode<Yolov11Node>(argc, argv, {kBanner, "person_detection"});
}
