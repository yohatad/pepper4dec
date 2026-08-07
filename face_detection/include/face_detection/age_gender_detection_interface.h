/* age_gender_detection_interface.h
 *
 * Lifecycle node for age/gender estimation of tracked persons exhibiting
 * mutual gaze: MiVOLOONNX (an ONNX Runtime wrapper around the MiVOLO
 * face+body age/gender classifier) and AgeGenderDetectionNode, which caches
 * recent /face_detection/data and /person_detection/data detections plus
 * the latest camera frame, and whenever a tracked person exhibits mutual
 * gaze within a configurable depth range, queues that person's track ID for
 * inference. A single worker thread runs MiVOLO on the cached face+body
 * crop, applies temporal smoothing across repeated estimates, and publishes
 * the resulting per-person profile as a JSON string.
 *
 * Subscribers:
 *   <image_topic> (sensor_msgs/Image)
 *     Raw color camera frames, cached for crop extraction.
 *   <face_topic> (dec_interfaces/FaceDetection)
 *     Per-frame face tracking results, incl. mutual_gaze flag.
 *   <person_topic> (dec_interfaces/PersonDetection)
 *     Tracked person detections, used for the body crop.
 *
 * Publishers:
 *   <output_topic> (std_msgs/String)
 *     Per-person JSON profile: label_id, age, gender, gender_confidence,
 *     estimation_count, person_bbox.
 *
 * Parameters (ROS2 parameters, loaded from
 * age_gender_detection_configuration.yaml via the launch file):
 *   mivolo_model_path, device, face_only, face_topic, person_topic,
 *   image_topic, output_topic, max_cache_age_sec, min_estimate_interval_sec,
 *   re_estimate_interval_sec, person_class_name, max_depth_m.
 *
 * Lifecycle:
 *   configure  -> load the MiVOLO ONNX model, reset caches
 *   activate   -> create publisher + subscriptions, start the estimation
 *                 worker thread, start the cleanup/debug timers
 *   deactivate -> destroy subscriptions, stop and join the worker thread
 *   cleanup    -> release the ONNX session, clear caches
 *   shutdown   -> log shutdown
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 29, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <dec_interfaces/msg/face_detection.hpp>
#include <dec_interfaces/msg/person_detection.hpp>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

struct AgeGenderDetectionConfig {
    std::string mivolo_model_path;
    std::string device = "cuda";
    // Only the exported face+body (6-channel) MiVOLO model is currently
    // packaged; a face-only (3-channel) model would need its own ONNX
    // export before this flag does anything.
    bool face_only = false;
    std::string face_topic = "/face_detection/data";
    std::string person_topic = "/person_detection/data";
    std::string image_topic = "/camera/color/image_raw";
    std::string output_topic = "/face_detection/age_gender_results";
    double max_cache_age_sec = 2.0;
    double min_estimate_interval_sec = 0.5;
    double re_estimate_interval_sec = 30.0;
    std::string person_class_name = "person";
    double max_depth_m = 4.0;
};

// Declares and reads this node's ROS2 parameters (see param_loader.h),
// falling back to the AgeGenderDetectionConfig defaults above for any
// parameter not set by the launch file's YAML.
AgeGenderDetectionConfig loadAgeGenderConfiguration(rclcpp_lifecycle::LifecycleNode* node);

//=============================================================================
// MiVOLOONNX
//
// Thin wrapper around the MiVOLO face+body age/gender ONNX model (exported
// from the original MiVOLO TorchScript checkpoint). Input is a single
// 6-channel [1, 6, 224, 224] tensor: face crop channels 0-2, body crop
// channels 3-5, each letterboxed to 224x224 and ImageNet-normalized. Output
// is [1, 3]: (gender_male_logit, gender_female_logit, age_normalized).
//=============================================================================

class MiVOLOONNX {
public:
    struct Estimate {
        double age = 0.0;
        std::string gender;
        double gender_confidence = 0.0;
    };

    MiVOLOONNX(const std::string& model_path, bool use_cuda);

    // face_crop and body_crop must both be non-empty BGR images (the
    // packaged model requires face+body). Returns std::nullopt on invalid
    // input.
    std::optional<Estimate> predict(const cv::Mat& face_crop, const cv::Mat& body_crop);

private:
    // Letterbox-resizes `crop` to kInputSize^2, normalizes, and writes 3
    // planes of CHW float32 into dst. Writes zeros if crop is empty.
    void writeCropPlanes(const cv::Mat& crop, float* dst) const;

    static constexpr int kInputSize = 224;

    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    cv::Scalar mean_{0.485, 0.456, 0.406};
    cv::Scalar std_{0.229, 0.224, 0.225};

    // Age decoding parameters (see MiVOLOInference in age_gender_detection.py).
    double min_age_ = 0.0;
    double max_age_ = 95.0;
    double avg_age_ = 48.0;
};

//=============================================================================
// Data structures
//=============================================================================

// Snapshot of a face or person detection, cached for later crop extraction.
struct AgeGenderBoundingBox {
    double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
    double depth = 0.0;        // metres (z of centroid), 0.0 if unavailable
    bool mutual_gaze = false;  // only meaningful for face detections
    std::chrono::steady_clock::time_point timestamp;

    static AgeGenderBoundingBox fromCentroid(const geometry_msgs::msg::Point& centroid, float width, float height,
                                             bool mutual_gaze = false);
};

// Stores age/gender estimates with temporal smoothing (median age, confidence-
// weighted majority gender vote over the last 5 estimates).
struct AgeGenderPersonProfile {
    static constexpr size_t kHistoryMaxLen = 5;

    std::string label_id;

    std::deque<double> age_history;
    std::deque<std::pair<std::string, double>> gender_history;

    std::optional<double> age;
    std::optional<std::string> gender;
    std::optional<double> gender_confidence;

    std::chrono::steady_clock::time_point last_updated{std::chrono::steady_clock::now()};
    int estimation_count = 0;
    bool has_valid_estimate = false;

    std::optional<AgeGenderBoundingBox> last_person_bbox;

    void addEstimate(double age_val, const std::string& gender_val, double gender_conf);
    std::string toJson() const;
};

//=============================================================================
// AgeGenderDetectionNode
//=============================================================================

class AgeGenderDetectionNode : public rclcpp_lifecycle::LifecycleNode {
public:
    using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

    explicit AgeGenderDetectionNode(const std::string& node_name = "age_gender_detection");
    ~AgeGenderDetectionNode() override;

    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_shutdown  (const rclcpp_lifecycle::State& state) override;

    void cleanup();

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    void personCallback(const dec_interfaces::msg::PersonDetection::ConstSharedPtr& msg);
    void faceCallback(const dec_interfaces::msg::FaceDetection::ConstSharedPtr& msg);

    void cleanupStaleData();
    void debugStatus();

    void scheduleEstimation(const std::string& label_id);
    void estimationWorker();
    void estimateForPerson(const std::string& label_id);

    bool shouldReEstimate(const AgeGenderPersonProfile& profile, std::chrono::steady_clock::time_point now) const;
    std::optional<cv::Mat> extractCrop(const cv::Mat& image, const AgeGenderBoundingBox& bbox) const;
    void publishResult(const std::string& json_result);

    AgeGenderDetectionConfig config_;
    std::unique_ptr<MiVOLOONNX> mivolo_;

    // Detection caches and per-person profiles, guarded by data_mutex_.
    std::mutex data_mutex_;
    std::unordered_map<std::string, AgeGenderBoundingBox> recent_persons_;
    std::unordered_map<std::string, AgeGenderBoundingBox> recent_faces_;
    std::unordered_map<std::string, AgeGenderPersonProfile> person_profiles_;
    std::set<std::string> known_label_ids_;

    // Latest camera frame, guarded by image_mutex_.
    std::mutex image_mutex_;
    cv::Mat latest_image_;
    std::chrono::steady_clock::time_point image_timestamp_;
    bool has_image_ = false;

    // Single worker thread with a dedup'd queue (mirrors the Python node's
    // Queue + pending-set pattern, avoiding unbounded thread creation).
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<std::string> estimation_queue_;
    std::set<std::string> pending_estimates_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};

    rclcpp::Subscription<dec_interfaces::msg::PersonDetection>::SharedPtr person_sub_;
    rclcpp::Subscription<dec_interfaces::msg::FaceDetection>::SharedPtr face_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>::SharedPtr result_pub_;

    rclcpp::TimerBase::SharedPtr cleanup_timer_;
    rclcpp::TimerBase::SharedPtr debug_timer_;
};

