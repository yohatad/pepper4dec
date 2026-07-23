/* face_detection_interface.h
 *
 * Lifecycle node(s) for face and mutual-gaze detection: a base
 * FaceDetectionNode that manages publishers, person-detection subscription,
 * and debug visualization, and a SixDrepNet subclass that loads the YOLO
 * (face detector) and SixDrepNet (head pose) ONNX models and runs
 * head-pose/mutual-gaze inference on synchronized RGB-D camera frames. Faces
 * are matched to tracked persons (from /person_detection/data) via a
 * Hungarian-assignment cost function when require_person_detection is true;
 * otherwise faces are tracked directly with ByteTrack.
 *
 * Subscribers:
 *   <camera color topic> (sensor_msgs/Image or CompressedImage)
 *     Synchronized RGB camera frames (topic resolved from camera config).
 *   <camera depth topic> (sensor_msgs/Image or CompressedImage)
 *     Synchronized depth camera frames (topic resolved from camera config).
 *   /person_detection/data (dec_interfaces/PersonDetection)
 *     Tracked person detections used to constrain and match faces (only if
 *     requirePersonDetection is true).
 *
 * Publishers:
 *   /face_detection/data (dec_interfaces/FaceDetection)
 *     Per-frame face tracking results: face IDs, centroids, sizes, and
 *     mutual gaze flags.
 *   /face_detection/debug (sensor_msgs/Image)
 *     Debug visualization of the color frame with face boxes and head-pose axes.
 *   /face_detection/depth_debug (sensor_msgs/Image)
 *     Colorized depth visualization for debugging.
 *
 * Parameters (ROS2 parameters, loaded from face_detection_configuration.yaml
 * via the launch file):
 *   use_compressed, camera, verbose_mode, image_timeout,
 *   sixdrepnet_confidence, sixdrepnet_headpose_angle, require_person_detection,
 *   person_detection_timeout, prioritize_face_depth.
 *
 * Lifecycle:
 *   configure  -> create lifecycle publishers and initialize state, incl. the
 *                 standalone-mode ByteTrack face tracker (base); load YOLO +
 *                 SixDrepNet ONNX models (SixDrepNet)
 *   activate   -> subscribe to person detection (if enabled) and start the
 *                 debug visualization timer (base); create camera
 *                 subscriptions and start the image timeout monitor (SixDrepNet)
 *   deactivate -> stop the visualization timer and destroy the person
 *                 detection subscription (base); destroy camera
 *                 subscriptions (SixDrepNet)
 *   cleanup    -> destroy lifecycle publishers (base); release the loaded
 *                 ONNX models (SixDrepNet)
 *   shutdown   -> log shutdown (base)
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 06, 2026
 * Version: v1.0 - C++ port of face_detection_implementation.py / face_detection_application.py
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#ifndef FACE_DETECTION_INTERFACE_H
#define FACE_DETECTION_INTERFACE_H

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <dec_interfaces/msg/face_detection.hpp>
#include <dec_interfaces/msg/person_detection.hpp>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "dec_common/byte_tracker.h"
#include "dec_common/camera_lifecycle_node.h"

// Sentinel cost for impossible face-person matches. A large finite value
// (rather than infinity) keeps the cost matrix feasible for the Hungarian
// solver.
constexpr double kImpossibleMatchCost = 1e6;

struct FaceDetectionConfig {
    bool use_compressed = false;
    std::string camera = "realsense";
    bool verbose_mode = true;
    double image_timeout = 2.0;
    double sixdrepnet_confidence = 0.65;
    double sixdrepnet_headpose_angle = 10.0;
    bool require_person_detection = true;
    double person_detection_timeout = 0.5;
    bool prioritize_face_depth = true;
};

// Declares and reads this node's ROS2 parameters (see param_loader.h),
// falling back to the FaceDetectionConfig defaults above for any parameter
// not set by the launch file's YAML.
FaceDetectionConfig loadConfiguration(rclcpp_lifecycle::LifecycleNode* node);

// Cached snapshot of the latest /person_detection/data message.
struct PersonSnapshot {
    std::vector<std::string> person_label_id;
    std::vector<std::string> class_names;
    std::vector<geometry_msgs::msg::Point> centroids;
    std::vector<float> width;
    std::vector<float> height;
};

// One finalized face tracking record, ready to publish/draw.
struct FaceTrackingDatum {
    std::string face_id;
    geometry_msgs::msg::Point centroid;
    float width = 0.0f;
    float height = 0.0f;
    bool mutual_gaze = false;
};

//=============================================================================
// YOLOONNX
//
// Thin wrapper around the goldYOLO face-detector ONNX model. NMS is baked
// into the exported graph, so postprocessing here is just a confidence
// filter + coordinate rescale (unlike Yolov11Node's detector, which performs
// its own NMS).
//=============================================================================

class YOLOONNX {
public:
    YOLOONNX(const std::string& model_path, double class_score_th);

    // Returns (boxes as xyxy in image pixel coords, scores).
    std::pair<std::vector<cv::Rect2d>, std::vector<float>> detect(const cv::Mat& image);

private:
    cv::Mat preprocess(const cv::Mat& image);
    std::pair<std::vector<cv::Rect2d>, std::vector<float>> postprocess(
        const cv::Mat& image, const std::vector<float>& raw_boxes, int64_t num_boxes, int64_t num_attrs);

    double class_score_th_;
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    int64_t input_height_ = 0;
    int64_t input_width_ = 0;
};

//=============================================================================
// FaceDetectionNode
//=============================================================================

// Camera plumbing (topic resolution, subscriptions, depth decode, debug
// visualization, timeout monitor) is inherited from
// dec_common::CameraLifecycleNode; this class adds the face publishers and
// the person-detection subscription used for face-person matching.
class FaceDetectionNode : public dec_common::CameraLifecycleNode {
public:
    explicit FaceDetectionNode(const std::string& node_name = "faceDetection");

    // ── Lifecycle callbacks ─────────────────────────────────────────────────
    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_shutdown  (const rclcpp_lifecycle::State& state) override;

    void cleanup();

protected:
    void publishFaceDetection(const std::vector<FaceTrackingDatum>& tracking_data);
    void personDetectionCallback(const dec_interfaces::msg::PersonDetection& msg);

    FaceDetectionConfig config_;

    rclcpp_lifecycle::LifecyclePublisher<dec_interfaces::msg::FaceDetection>::SharedPtr pub_gaze_;

    bool require_person_detection_ = true;
    double person_detection_timeout_ = 0.5;
    bool prioritize_face_depth_ = true;

    std::mutex person_detections_mutex_;
    std::optional<PersonSnapshot> latest_person_detections_;
    std::optional<rclcpp::Time> latest_person_detections_timestamp_;

    std::unordered_map<std::string, cv::Scalar> face_colors_;
    byte_tracker::ByteTrack face_tracker_;

    rclcpp::Subscription<dec_interfaces::msg::PersonDetection>::SharedPtr person_detection_sub_;
};

//=============================================================================
// SixDrepNet
//=============================================================================

class SixDrepNet : public FaceDetectionNode {
public:
    SixDrepNet();

    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;

protected:
    void processImages() override;

private:
    void drawAxis(cv::Mat& img, double yaw, double pitch, double roll, double tdx, double tdy, double size = 100.0);

    struct FaceCandidate {
        double x1, y1, x2, y2;
        double cx, cy;
        double w, h;
        float score;
    };
    struct PersonCandidate {
        std::string tracking_id;
        double x1, y1, x2, y2;
        double depth;
        int assigned_faces = 0;
    };

    double calculateMatchingCost(const FaceCandidate& face, const PersonCandidate& person) const;
    std::vector<std::pair<int, int>> matchFacesToPersonsHungarian(
        const std::vector<FaceCandidate>& faces, const std::vector<PersonCandidate>& persons);
    float getBestDepthEstimate(double face_cx, double face_cy, double face_width, double face_height,
                               double person_depth) const;

    cv::Mat processFrameStandalone(const cv::Mat& cv_image);
    cv::Mat processFrameWithPersonDetection(const cv::Mat& cv_image);

    // Runs SixDrepNet on a cropped face image, returning (yaw, pitch, roll) in degrees.
    std::optional<std::array<double, 3>> estimateHeadPose(const cv::Mat& face_crop);

    double sixdrep_angle_ = 10.0;

    std::unique_ptr<YOLOONNX> yolo_model_;
    std::unique_ptr<Ort::Env> sixdrepnet_env_;
    std::unique_ptr<Ort::Session> sixdrepnet_session_;
    std::vector<std::string> sixdrepnet_input_names_;
    std::vector<std::string> sixdrepnet_output_names_;

    cv::Scalar mean_{0.485, 0.456, 0.406};
    cv::Scalar std_{0.229, 0.224, 0.225};
};

#endif  // FACE_DETECTION_INTERFACE_H
