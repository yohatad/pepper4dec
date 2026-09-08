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
 * The node's complete ROS2 interface (subscribers, publishers, parameters,
 * and lifecycle transitions) is documented in
 * face_detection_application.cpp.
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

#pragma once

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

/** @brief Tunable settings for the face-detection node (see the YAML config). */
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

/**
 * @brief Cached snapshot of the latest /person_detection/data message.
 */
struct PersonSnapshot {
    std::vector<std::string> person_label_id;
    std::vector<std::string> class_names;
    std::vector<geometry_msgs::msg::Point> centroids;
    std::vector<float> width;
    std::vector<float> height;
};

/**
 * @brief One finalized face tracking record, ready to publish/draw.
 */
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

/**
 * @class YOLOONNX
 * @brief ONNX Runtime wrapper around the YOLO face detector.
 *
 * NMS is baked into the exported graph, so postprocessing is only a confidence
 * filter plus coordinate rescale.
 */
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

/**
 * @class FaceDetectionNode
 * @brief Base lifecycle node owning the face publishers and person matching.
 *
 * Camera plumbing (topic resolution, subscriptions, depth decode, debug
 * visualization, timeout monitor) is inherited from
 * dec_common::CameraLifecycleNode; this class adds the face publishers and the
 * person-detection subscription used for face-person matching.
 */
class FaceDetectionNode : public dec_common::CameraLifecycleNode {
public:
    explicit FaceDetectionNode(const std::string& node_name = "face_detection");

    /** @brief Create the lifecycle publishers and the standalone-mode face tracker. */
    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;

    /** @brief Subscribe to person detection (if enabled) and start the debug
     *         visualization timer. */
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;

    /** @brief Stop the visualization timer and drop the person-detection
     *         subscription. */
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;

    /** @brief Destroy the lifecycle publishers. */
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;

    /** @brief Log that the node is shutting down. */
    CallbackReturn on_shutdown  (const rclcpp_lifecycle::State& state) override;

    /** @brief Close any open debug windows before the process exits. */
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

/**
 * @class SixDrepNet
 * @brief Face-detection node adding SixDRepNet head-pose and mutual gaze.
 *
 * Loads the YOLO face detector and the SixDRepNet head-pose model, runs both
 * over the synchronized RGB-D frames, and marks a face as engaged when its
 * head-pose angle falls within sixdrepnet_headpose_angle of the camera.
 */
class SixDrepNet : public FaceDetectionNode {
public:
    SixDrepNet();

    /** @brief Load the YOLO face detector and the SixDRepNet head-pose model. */
    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;

    /** @brief Create the camera subscriptions and start the image-timeout monitor. */
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;

    /** @brief Destroy the camera subscriptions. */
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;

    /** @brief Release the loaded ONNX models. */
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;

protected:
    void processImages() override;

private:
    void drawAxis(cv::Mat& img, double yaw, double pitch, double roll, double tdx, double tdy, double size = 100.0);

    /** @brief One detected face awaiting assignment to a tracked person. */
    struct FaceCandidate {
        double x1, y1, x2, y2;
        double cx, cy;
        double w, h;
        float score;
    };

    /** @brief One tracked person available to receive a detected face. */
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

