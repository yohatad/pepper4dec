/* person_detection_interface.h
 *
 * Lifecycle node(s) for person (and other configurable COCO-class) detection
 * and tracking. PersonDetectionNode provides the base publishers, debug
 * visualization, camera topic resolution, and depth lookup; Yolov11Node
 * loads a YOLOv11 ONNX detection model and a ByteTrack tracker to detect
 * and track configurable COCO classes from synchronized color/depth camera
 * streams.
 *
 * Subscribers:
 *   <camera color topic> (sensor_msgs/Image or CompressedImage)
 *     Color camera frames (topic resolved from pepper_topics.yaml based on
 *     the configured camera type).
 *   <camera depth topic> (sensor_msgs/Image or CompressedImage)
 *     Depth camera frames used to estimate distance to detected objects.
 *
 * Publishers:
 *   /person_detection/data (dec_interfaces/PersonDetection)
 *     Tracked object detections: track IDs, class names/IDs, confidences,
 *     centroids (with depth), widths, and heights.
 *   /person_detection/debug (sensor_msgs/Image)
 *     Annotated color image showing tracked bounding boxes, labels, and depth.
 *   /person_detection/depth_debug (sensor_msgs/Image)
 *     Colorized visualization of the raw depth image.
 *
 * Parameters (ROS2 parameters, loaded from person_detection_configuration.yaml
 * via the launch file):
 *   camera, use_compressed, image_timeout, verbose_mode, confidence_threshold,
 *   target_classes, track_threshold, track_buffer, match_threshold, frame_rate.
 *
 * Lifecycle:
 *   configure  -> create lifecycle publishers, load camera/config settings
 *                 and target classes (base); load ONNX model + ByteTrack (Yolov11Node)
 *   activate   -> start the visualization and status timers (base); create
 *                 camera subscriptions and start the timeout monitor (Yolov11Node)
 *   deactivate -> cancel the visualization and status timers (base); destroy
 *                 camera subscriptions (Yolov11Node)
 *   cleanup    -> destroy the lifecycle publishers (base); release the ONNX
 *                 session (Yolov11Node)
 *   shutdown   -> log that the node is shutting down
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 06, 2026
 * Version: v1.0 - C++ port of person_detection_implementation.py / person_detection_application.py
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#ifndef PERSON_DETECTION_INTERFACE_H
#define PERSON_DETECTION_INTERFACE_H

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <dec_interfaces/msg/person_detection.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "person_detection/byte_tracker.h"

// The 80 COCO class names, in model-output order.
extern const std::array<std::string, 80> COCO_CLASSES;

struct PersonDetectionConfig {
    std::string camera = "realsense";
    bool use_compressed = false;
    double image_timeout = 2.0;
    bool verbose_mode = true;
    double confidence_threshold = 0.5;

    // ByteTrack parameters
    float track_threshold = 0.45f;
    int track_buffer = 30;
    float match_threshold = 0.8f;
    int frame_rate = 30;

    // Class names (or numeric indices, as strings) to track; {"all"} tracks everything.
    std::vector<std::string> target_classes = {"person"};
};

// Declares and reads this node's ROS2 parameters (see dec_common/param_loader.h),
// falling back to the PersonDetectionConfig defaults above for any parameter
// not set by the launch file's YAML.
PersonDetectionConfig loadConfiguration(rclcpp_lifecycle::LifecycleNode* node);

// Resolves target_classes (names or numeric-string indices) to a set of COCO
// class indices. Empty or {"all"} means "track everything".
std::set<int> getClassIndices(const std::vector<std::string>& target_classes, const rclcpp::Logger& logger);

// One finalized tracked-object record, ready to publish/draw.
struct TrackingDatum {
    std::string track_id;
    int class_id = 0;
    std::string class_name;
    float confidence = 0.0f;
    geometry_msgs::msg::Point centroid;
    float width = 0.0f;
    float height = 0.0f;
};

//=============================================================================
// PersonDetectionNode
//=============================================================================

class PersonDetectionNode : public rclcpp_lifecycle::LifecycleNode {
public:
    using CallbackReturn =
        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

    explicit PersonDetectionNode(const std::string& node_name = "personDetection");

    // ── Lifecycle callbacks ─────────────────────────────────────────────────
    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_shutdown  (const rclcpp_lifecycle::State& state) override;

protected:
    void statusCallback();
    void updateLatestFrame(const cv::Mat& frame);
    void visualizationCallback();

    // Resolves the RGB/depth topic names for the configured camera type.
    std::pair<std::string, std::string> getTopicNames();
    std::optional<std::string> extractTopic(const std::string& image_topic_key);

    void synchronizedCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data,
                              const sensor_msgs::msg::Image::ConstSharedPtr& depth_data);
    void synchronizedCallbackCompressed(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& color_data,
                                        const sensor_msgs::msg::CompressedImage::ConstSharedPtr& depth_data);
    void rgbOnlyCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data);
    void rgbOnlyCallbackCompressed(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& color_data);

    std::optional<cv::Mat> processDepthImageMsg(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    std::optional<cv::Mat> processDepthCompressedMsg(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg);

    void startTimeoutMonitor();
    void checkTimeout();
    bool checkCameraResolution(const cv::Mat& color_image, const cv::Mat& depth_image) const;
    std::optional<cv::Mat> makeDepthVis(const cv::Mat& depth) const;
    std::optional<float> getDepthInRegion(double centroid_x, double centroid_y, double box_width,
                                          double box_height, double region_scale = 0.1) const;
    cv::Scalar generateDarkColor();

    // Runs detection + tracking on the current color/depth frame. Subclasses
    // implement detectObject(); this drives the shared post-processing
    // (class filtering, tracking, publishing, visualization).
    void processImages();
    virtual bool detectObject(const cv::Mat& image, std::vector<Eigen::Vector4d>& boxes,
                              std::vector<float>& scores, std::vector<int>& class_ids) = 0;
    virtual byte_tracker::Detections updateTracker(const std::vector<Eigen::Vector4d>& boxes,
                                                    const std::vector<float>& scores,
                                                    const std::vector<int>& class_ids) = 0;

    std::vector<TrackingDatum> prepareTrackingData(const byte_tracker::Detections& tracked);
    void publishObjectDetection(const std::vector<TrackingDatum>& tracking_data);
    cv::Mat drawTrackedObjects(const cv::Mat& frame, const byte_tracker::Detections& tracked,
                              const std::vector<TrackingDatum>& tracking_data);

    PersonDetectionConfig config_;
    std::string node_name_;

    rclcpp_lifecycle::LifecyclePublisher<dec_interfaces::msg::PersonDetection>::SharedPtr pub_objects_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr depth_debug_pub_;

    cv::Mat color_image_;
    cv::Mat depth_image_;

    bool use_compressed_ = false;
    std::string camera_type_ = "realsense";
    bool verbose_mode_ = true;
    double image_timeout_ = 2.0;
    std::set<int> target_class_indices_;

    rclcpp::Time timer_start_;
    std::optional<double> last_image_time_;

    std::mutex frame_mutex_;
    cv::Mat latest_frame_;

    std::unordered_map<int, cv::Scalar> object_colors_;

    rclcpp::TimerBase::SharedPtr vis_timer_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    rclcpp::TimerBase::SharedPtr timeout_timer_;

    // Synchronized (uncompressed) subscription pair.
    using ApproxSync = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>> color_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproxSync>> sync_;

    // Synchronized (compressed) subscription pair.
    using ApproxSyncCompressed = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage>;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CompressedImage, rclcpp_lifecycle::LifecycleNode>>
        color_sub_compressed_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CompressedImage, rclcpp_lifecycle::LifecycleNode>>
        depth_sub_compressed_;
    std::shared_ptr<message_filters::Synchronizer<ApproxSyncCompressed>> sync_compressed_;

    // Pepper RGB-only subscription (plain, not message_filters-wrapped).
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_plain_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr color_sub_plain_compressed_;
};

//=============================================================================
// Yolov11Node
//
// Lifecycle:
//   configure  -> super().on_configure() -> load ONNX model + ByteTrack tracker
//   activate   -> super().on_activate()  -> create camera subscribers + timeout monitor
//   deactivate -> destroy camera subscribers -> super().on_deactivate()
//   cleanup    -> release ONNX session -> super().on_cleanup()
//=============================================================================

class Yolov11Node : public PersonDetectionNode {
public:
    Yolov11Node();

    CallbackReturn on_configure (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_activate  (const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
    CallbackReturn on_cleanup   (const rclcpp_lifecycle::State& state) override;

protected:
    bool detectObject(const cv::Mat& image, std::vector<Eigen::Vector4d>& boxes,
                      std::vector<float>& scores, std::vector<int>& class_ids) override;
    byte_tracker::Detections updateTracker(const std::vector<Eigen::Vector4d>& boxes,
                                           const std::vector<float>& scores,
                                           const std::vector<int>& class_ids) override;

private:
    bool createCameraSubscriptions();
    bool initModel();

    Ort::Value prepareInput(const cv::Mat& image);
    void processOutput(const Ort::Value& output, std::vector<Eigen::Vector4d>& boxes,
                       std::vector<float>& scores, std::vector<int>& class_ids);
    static Eigen::Vector4d rescaleBox(const Eigen::Vector4d& box_cxcywh, double orig_w, double orig_h,
                                      double input_w, double input_h);
    static Eigen::Vector4d xywhToXyxy(const Eigen::Vector4d& box);
    static double computeIou(const Eigen::Vector4d& a, const Eigen::Vector4d& b);
    static std::vector<int> nms(const std::vector<Eigen::Vector4d>& boxes, const std::vector<float>& scores,
                                double iou_threshold);
    static std::vector<int> multiclassNms(const std::vector<Eigen::Vector4d>& boxes,
                                          const std::vector<float>& scores, const std::vector<int>& class_ids,
                                          double iou_threshold);

    double confidence_threshold_ = 0.5;
    float track_thresh_ = 0.45f;
    int track_buffer_ = 30;
    float match_thresh_ = 0.8f;
    int frame_rate_ = 30;

    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    int64_t input_height_ = 0;
    int64_t input_width_ = 0;
    double orig_width_ = 0.0;
    double orig_height_ = 0.0;

    std::unique_ptr<byte_tracker::ByteTrack> tracker_;
};

#endif  // PERSON_DETECTION_INTERFACE_H
