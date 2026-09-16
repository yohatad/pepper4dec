/* face_detection_application.cpp
 *
 * Entry point for the SixDRepNet face and mutual gaze detection lifecycle
 * node. Loads configuration, spins the node, and cleans up (closing any
 * debug windows) on shutdown; the node classes themselves are implemented in
 * face_detection_implementation.cpp.
 *
 * Subscribers:
 *   <camera color topic> (sensor_msgs/msg/Image or CompressedImage)
 *     Synchronized RGB camera frames (topic resolved from data/pepper_topics.yaml
 *     for the configured camera).
 *   <camera depth topic> (sensor_msgs/msg/Image or CompressedImage)
 *     Synchronized depth camera frames (topic resolved from data/pepper_topics.yaml
 *     for the configured camera).
 *   /person_detection/data (dec_interfaces/msg/PersonDetection)
 *     Tracked person detections used to constrain and match faces (only when
 *     require_person_detection is true).
 *
 * Publishers:
 *   /face_detection/data (dec_interfaces/msg/FaceDetection)
 *     Per-frame face tracking results: face IDs, centroids, sizes, and mutual
 *     gaze flags.
 *   /face_detection/debug (sensor_msgs/msg/Image)
 *     Debug visualization of the color frame with face boxes and head-pose axes.
 *   /face_detection/depth_debug (sensor_msgs/msg/Image)
 *     Colorized depth visualization for debugging.
 *
 * Parameters (config/face_detection_configuration.yaml, under
 * face_detection/ros__parameters):
 *   use_compressed (bool, default: false)
 *   camera (string, default: "pepper")
 *   verbose_mode (bool, default: false)
 *   image_timeout (double, default: 2.0)
 *   face_detection_confidence (double, default: 0.90)
 *   sixdrepnet_headpose_angle (double, default: 10.0)
 *   require_person_detection (bool, default: true)
 *   person_detection_timeout (double, default: 0.5)
 *   prioritize_face_depth (bool, default: true)
 *
 * Lifecycle:
 *   configure  -> create lifecycle publishers and initialize state, incl. the
 *                 standalone-mode ByteTrack face tracker (base); load YOLO +
 *                 SixDRepNet ONNX models (SixDRepNet)
 *   activate   -> subscribe to person detection (if enabled) and start the
 *                 debug visualization timer (base); create camera
 *                 subscriptions and start the image timeout monitor (SixDRepNet)
 *   deactivate -> stop the visualization timer and destroy the person
 *                 detection subscription (base); destroy camera
 *                 subscriptions (SixDRepNet)
 *   cleanup    -> destroy lifecycle publishers (base); release the loaded
 *                 ONNX models (SixDRepNet)
 *   shutdown   -> log shutdown (base)
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

#include "face_detection/face_detection_interface.h"

#include "dec_common/node_runner.h"

int main(int argc, char** argv) {
    return dec_common::runNode<SixDRepNet>(
        argc, argv,
        {"face_detection v1.0 — This program comes with ABSOLUTELY NO WARRANTY.", "face_detection"},
        nullptr,
        [](SixDRepNet& node) { node.cleanup(); });
}
