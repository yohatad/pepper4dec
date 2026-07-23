/* camera_lifecycle_node.cpp
 *
 * Implements the shared camera plumbing declared in camera_lifecycle_node.h.
 * Bodies were unified from the previously duplicated implementations in
 * face_detection_implementation.cpp and person_detection_implementation.cpp;
 * see the header for the deliberate behavioral knobs and unifications.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 18, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#include "dec_common/camera_lifecycle_node.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

namespace dec_common {

CameraLifecycleNode::CameraLifecycleNode(const std::string& node_name, CameraNodeBehavior behavior)
    : rclcpp_lifecycle::LifecycleNode(node_name), behavior_(std::move(behavior)) {}

// ── Debug visualization ──────────────────────────────────────────────────────

void CameraLifecycleNode::updateLatestFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_frame_ = frame.clone();
    // Snapshot the depth frame under the same lock so visualizationCallback
    // (timer thread) never reads depth_image_ while a camera callback writes it.
    latest_depth_ = depth_image_.empty() ? cv::Mat() : depth_image_.clone();
}

void CameraLifecycleNode::visualizationCallback() {
    cv::Mat color_frame, depth_vis;
    bool have_color = false, have_depth_vis = false;
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!latest_frame_.empty()) {
            color_frame = latest_frame_.clone();
            have_color = true;
        }
        if (!latest_depth_.empty()) {
            auto vis = makeDepthVis(latest_depth_);
            if (vis) {
                depth_vis = *vis;
                have_depth_vis = true;
            }
        }
    }

    if (!have_color && !have_depth_vis) return;

    const bool display = verbose_mode_ && std::getenv("DISPLAY") != nullptr;

    if (display) {
        try {
            if (have_color) cv::imshow(behavior_.debug_window_prefix + " Debug (RGB)", color_frame);
            if (have_depth_vis) cv::imshow(behavior_.debug_window_prefix + " Debug (Depth)", depth_vis);
            int key = cv::waitKey(1) & 0xFF;
            if (behavior_.quit_on_q && key == 'q') {
                RCLCPP_INFO(get_logger(), "%s: User requested shutdown", node_name_.c_str());
                rclcpp::shutdown();
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "imshow failed (likely headless): %s", e.what());
        }
    }

    if (behavior_.always_publish_debug || display) {
        try {
            if (have_color) {
                auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", color_frame).toImageMsg();
                debug_pub_->publish(*msg);
            }
            if (have_depth_vis) {
                auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", depth_vis).toImageMsg();
                depth_debug_pub_->publish(*msg);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to publish debug images: %s", e.what());
        }
    }
}

// ── Camera topic resolution ──────────────────────────────────────────────────

std::pair<std::string, std::string> CameraLifecycleNode::getTopicNames() {
    std::string rgb_key, depth_key;
    if (camera_type_ == "realsense" || camera_type_ == "video") {
        rgb_key = "RealSenseCameraRGB";
        depth_key = "RealSenseCameraDepth";
    } else if (camera_type_ == "pepper") {
        rgb_key = "PepperFrontCamera";
        depth_key = "PepperDepthCamera";
    } else {
        throw std::invalid_argument("Invalid camera type: " + camera_type_);
    }

    auto rgb_topic = extractTopic(rgb_key);
    auto depth_topic = extractTopic(depth_key);
    if (!rgb_topic || !depth_topic) {
        throw std::runtime_error("Failed to extract camera topics");
    }
    return {*rgb_topic, *depth_topic};
}

std::optional<std::string> CameraLifecycleNode::extractTopic(const std::string& image_topic_key) {
    try {
        std::string package = ament_index_cpp::get_package_share_directory(behavior_.topics_package);
        std::string config_path = package + "/data/pepper_topics.yaml";
        YAML::Node topics = YAML::LoadFile(config_path);
        if (topics[image_topic_key]) {
            return topics[image_topic_key].as<std::string>();
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error extracting topic '%s': %s", image_topic_key.c_str(), e.what());
    }
    return std::nullopt;
}

// ── Subscriptions ────────────────────────────────────────────────────────────

bool CameraLifecycleNode::createCameraSubscriptions() {
    try {
        auto [rgb_topic, depth_topic] = getTopicNames();

        std::string color_topic, depth_topic_sub;
        bool compressed = false;
        if (use_compressed_ && camera_type_ == "realsense") {
            color_topic = rgb_topic + "/compressed";
            depth_topic_sub = depth_topic + "/compressedDepth";
            compressed = true;
        } else if (use_compressed_ && camera_type_ == "pepper") {
            RCLCPP_WARN(get_logger(), "Compressed images not available for Pepper cameras");
            color_topic = rgb_topic;
        } else {
            color_topic = rgb_topic;
            depth_topic_sub = depth_topic;
        }

        rmw_qos_profile_t qos = rmw_qos_profile_sensor_data;

        if (camera_type_ == "pepper") {
            if (compressed) {
                color_sub_plain_compressed_ = create_subscription<sensor_msgs::msg::CompressedImage>(
                    color_topic, rclcpp::SensorDataQoS(),
                    std::bind(&CameraLifecycleNode::rgbOnlyCallbackCompressed, this, std::placeholders::_1));
            } else {
                color_sub_plain_ = create_subscription<sensor_msgs::msg::Image>(
                    color_topic, rclcpp::SensorDataQoS(),
                    std::bind(&CameraLifecycleNode::rgbOnlyCallback, this, std::placeholders::_1));
            }
            RCLCPP_INFO(get_logger(), "%s: subscribed to %s (depth disabled for Pepper)",
                node_name_.c_str(), color_topic.c_str());
            return true;
        }

        if (compressed) {
            color_sub_compressed_ = std::make_shared<
                message_filters::Subscriber<sensor_msgs::msg::CompressedImage, rclcpp_lifecycle::LifecycleNode>>(
                this, color_topic, qos);
            depth_sub_compressed_ = std::make_shared<
                message_filters::Subscriber<sensor_msgs::msg::CompressedImage, rclcpp_lifecycle::LifecycleNode>>(
                this, depth_topic_sub, qos);
            sync_compressed_ = std::make_shared<message_filters::Synchronizer<ApproxSyncCompressed>>(
                ApproxSyncCompressed(10), *color_sub_compressed_, *depth_sub_compressed_);
            sync_compressed_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
            sync_compressed_->registerCallback(
                std::bind(&CameraLifecycleNode::synchronizedCallbackCompressed, this,
                    std::placeholders::_1, std::placeholders::_2));
        } else {
            color_sub_ = std::make_shared<
                message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>>(
                this, color_topic, qos);
            depth_sub_ = std::make_shared<
                message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>>(
                this, depth_topic_sub, qos);
            sync_ = std::make_shared<message_filters::Synchronizer<ApproxSync>>(
                ApproxSync(10), *color_sub_, *depth_sub_);
            sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
            sync_->registerCallback(
                std::bind(&CameraLifecycleNode::synchronizedCallback, this,
                    std::placeholders::_1, std::placeholders::_2));
        }

        RCLCPP_INFO(get_logger(), "%s: subscribed to %s, %s", node_name_.c_str(), color_topic.c_str(),
            depth_topic_sub.c_str());
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "%s: camera subscription failed: %s", node_name_.c_str(), e.what());
        return false;
    }
}

// ── Frame callbacks ──────────────────────────────────────────────────────────

void CameraLifecycleNode::synchronizedCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data,
                                               const sensor_msgs::msg::Image::ConstSharedPtr& depth_data) {
    last_image_time_ = get_clock()->now().seconds();
    try {
        color_image_ = cv_bridge::toCvCopy(color_data, "bgr8")->image;
        auto depth = processDepthImageMsg(depth_data);
        depth_image_ = depth ? *depth : cv::Mat();

        if (color_image_.empty() || depth_image_.empty()) {
            RCLCPP_WARN(get_logger(), "Failed to decode images");
            return;
        }
        if (!checkCameraResolution(color_image_, depth_image_)) {
            RCLCPP_ERROR(get_logger(), "%s: Color camera and depth camera have different resolutions.",
                node_name_.c_str());
            rclcpp::shutdown();
            return;
        }
        processImages();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error in synchronizedCallback: %s", e.what());
    }
}

void CameraLifecycleNode::synchronizedCallbackCompressed(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& color_data,
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& depth_data) {
    last_image_time_ = get_clock()->now().seconds();
    try {
        color_image_ = cv::imdecode(cv::Mat(color_data->data), cv::IMREAD_COLOR);
        auto depth = processDepthCompressedMsg(depth_data);
        depth_image_ = depth ? *depth : cv::Mat();

        if (color_image_.empty() || depth_image_.empty()) {
            RCLCPP_WARN(get_logger(), "Failed to decode images");
            return;
        }
        if (!checkCameraResolution(color_image_, depth_image_)) {
            RCLCPP_ERROR(get_logger(), "%s: Color camera and depth camera have different resolutions.",
                node_name_.c_str());
            rclcpp::shutdown();
            return;
        }
        processImages();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error in synchronizedCallback: %s", e.what());
    }
}

void CameraLifecycleNode::rgbOnlyCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data) {
    last_image_time_ = get_clock()->now().seconds();
    try {
        color_image_ = cv_bridge::toCvCopy(color_data, "bgr8")->image;
        if (color_image_.empty()) {
            RCLCPP_WARN(get_logger(), "Failed to decode color image");
            return;
        }
        depth_image_ = cv::Mat();  // depth intentionally unused for Pepper
        processImages();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error in rgbOnlyCallback: %s", e.what());
    }
}

void CameraLifecycleNode::rgbOnlyCallbackCompressed(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& color_data) {
    last_image_time_ = get_clock()->now().seconds();
    try {
        color_image_ = cv::imdecode(cv::Mat(color_data->data), cv::IMREAD_COLOR);
        if (color_image_.empty()) {
            RCLCPP_WARN(get_logger(), "Failed to decode color image");
            return;
        }
        depth_image_ = cv::Mat();
        processImages();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error in rgbOnlyCallback: %s", e.what());
    }
}

std::optional<cv::Mat> CameraLifecycleNode::processDepthImageMsg(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
        return cv_bridge::toCvCopy(msg, "passthrough")->image;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Depth image processing error: %s", e.what());
        return std::nullopt;
    }
}

std::optional<cv::Mat> CameraLifecycleNode::processDepthCompressedMsg(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg) {
    try {
        if (!msg->format.empty() && msg->format.find("compressedDepth") != std::string::npos) {
            constexpr size_t kHeaderSize = 12;
            if (msg->data.size() <= kHeaderSize) {
                RCLCPP_WARN(get_logger(), "compressedDepth message too small to contain a header");
                return std::nullopt;
            }
            std::vector<uint8_t> img_data(msg->data.begin() + kHeaderSize, msg->data.end());
            cv::Mat depth = cv::imdecode(img_data, cv::IMREAD_ANYDEPTH);
            if (depth.empty()) {
                RCLCPP_WARN(get_logger(), "Failed to decode compressedDepth image, format: %s", msg->format.c_str());
            }
            return depth;
        }
        cv::Mat depth = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_UNCHANGED);
        if (depth.empty()) {
            RCLCPP_WARN(get_logger(), "Failed to decode compressed image, format: %s", msg->format.c_str());
        }
        return depth;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Depth image processing error: %s", e.what());
        return std::nullopt;
    }
}

// ── Monitoring / helpers ─────────────────────────────────────────────────────

void CameraLifecycleNode::startTimeoutMonitor() {
    timeout_timer_ = create_wall_timer(std::chrono::seconds(1),
                                       std::bind(&CameraLifecycleNode::checkTimeout, this));
}

void CameraLifecycleNode::checkTimeout() {
    if (last_image_time_) {
        double elapsed = get_clock()->now().seconds() - *last_image_time_;
        if (elapsed > image_timeout_) {
            RCLCPP_WARN(get_logger(), "No images received for %.1fs (timeout=%.1fs)", elapsed, image_timeout_);
        }
    }
}

bool CameraLifecycleNode::checkCameraResolution(const cv::Mat& color_image, const cv::Mat& depth_image) const {
    if (color_image.empty() || depth_image.empty()) return false;
    return color_image.rows == depth_image.rows && color_image.cols == depth_image.cols;
}

std::optional<cv::Mat> CameraLifecycleNode::makeDepthVis(const cv::Mat& depth) const {
    if (depth.empty()) return std::nullopt;
    try {
        cv::Mat depth_f32;
        depth.convertTo(depth_f32, CV_32F);
        for (int r = 0; r < depth_f32.rows; ++r) {
            float* row = depth_f32.ptr<float>(r);
            for (int c = 0; c < depth_f32.cols; ++c) {
                if (!std::isfinite(row[c])) row[c] = 0.0f;
            }
        }

        double max_val = 0.0;
        if (depth_f32.total() > 0) cv::minMaxLoc(depth_f32, nullptr, &max_val);
        if (max_val > 1000.0) depth_f32 /= 1000.0;
        if (depth_f32.total() > 0) cv::minMaxLoc(depth_f32, nullptr, &max_val);
        if (max_val <= 0.0) return std::nullopt;

        cv::Mat norm;
        depth_f32.convertTo(norm, CV_8U, 255.0 / max_val);
        cv::Mat colored;
        cv::applyColorMap(norm, colored, cv::COLORMAP_JET);
        return colored;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Depth visualization failed: %s", e.what());
        return std::nullopt;
    }
}

std::optional<float> CameraLifecycleNode::getDepthInRegion(double centroid_x, double centroid_y, double box_width,
                                                           double box_height, double region_scale) const {
    if (depth_image_.empty()) return std::nullopt;

    int region_width = std::max(5, static_cast<int>(box_width * region_scale));
    int region_height = std::max(5, static_cast<int>(box_height * region_scale));

    int x_start = std::max(0, static_cast<int>(centroid_x - region_width / 2.0));
    int y_start = std::max(0, static_cast<int>(centroid_y - region_height / 2.0));
    int x_end = std::min(depth_image_.cols, x_start + region_width);
    int y_end = std::min(depth_image_.rows, y_start + region_height);

    if (x_start >= x_end || y_start >= y_end) {
        RCLCPP_WARN(get_logger(), "Invalid region coordinates (%d, %d, %d, %d).", x_start, y_start, x_end, y_end);
        return std::nullopt;
    }

    cv::Mat roi = depth_image_(cv::Range(y_start, y_end), cv::Range(x_start, x_end));
    cv::Mat roi_f;
    roi.convertTo(roi_f, CV_32F);

    std::vector<float> valid;
    for (int r = 0; r < roi_f.rows; ++r) {
        const float* row = roi_f.ptr<float>(r);
        for (int c = 0; c < roi_f.cols; ++c) {
            if (std::isfinite(row[c]) && row[c] > 0.0f) valid.push_back(row[c]);
        }
    }
    if (valid.empty()) return std::nullopt;

    if (behavior_.median_depth) {
        std::sort(valid.begin(), valid.end());
        size_t n = valid.size();
        float median = (n % 2 == 1) ? valid[n / 2] : (valid[n / 2 - 1] + valid[n / 2]) / 2.0f;
        return median / 1000.0f;
    }
    double sum = 0.0;
    for (float v : valid) sum += v;
    return static_cast<float>((sum / valid.size()) / 1000.0);
}

cv::Scalar CameraLifecycleNode::generateDarkColor() {
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_int_distribution<int> dist(0, 150);
    while (true) {
        int b = dist(gen), g = dist(gen), r = dist(gen);
        double brightness = 0.299 * r + 0.587 * g + 0.114 * b;
        if (brightness < 130.0) return cv::Scalar(b, g, r);
    }
}

}  // namespace dec_common
