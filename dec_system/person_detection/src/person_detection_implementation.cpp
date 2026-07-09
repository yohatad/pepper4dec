/* person_detection_implementation.cpp
 *
 * Implements PersonDetectionNode (publishers, debug visualization, camera
 * topic resolution, depth lookup) and Yolov11Node (ONNX detection + ByteTrack
 * tracking). See person_detection_interface.h for the full subscriber/
 * publisher/parameter reference and the lifecycle state-machine diagram.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 06, 2026
 * Version: v1.0 - C++ port of person_detection_implementation.py
 */

#include "person_detection/person_detection_interface.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <dec_common/param_loader.h>
#include <yaml-cpp/yaml.h>

#include <rmw/qos_profiles.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>

const std::array<std::string, 80> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"};

PersonDetectionConfig loadConfiguration(rclcpp_lifecycle::LifecycleNode* node) {
    PersonDetectionConfig config;
    config.camera = dec_common::declareAndGetParameter(node, "camera", config.camera);
    config.use_compressed = dec_common::declareAndGetParameter(node, "use_compressed", config.use_compressed);
    config.image_timeout = dec_common::declareAndGetParameter(node, "image_timeout", config.image_timeout);
    config.verbose_mode = dec_common::declareAndGetParameter(node, "verbose_mode", config.verbose_mode);
    config.confidence_threshold =
        dec_common::declareAndGetParameter(node, "confidence_threshold", config.confidence_threshold);
    config.track_threshold = dec_common::declareAndGetParameter(node, "track_threshold", config.track_threshold);
    config.track_buffer = dec_common::declareAndGetParameter(node, "track_buffer", config.track_buffer);
    config.match_threshold = dec_common::declareAndGetParameter(node, "match_threshold", config.match_threshold);
    config.frame_rate = dec_common::declareAndGetParameter(node, "frame_rate", config.frame_rate);
    config.target_classes = dec_common::declareAndGetParameter(node, "target_classes", config.target_classes);
    return config;
}

std::set<int> getClassIndices(const std::vector<std::string>& target_classes, const rclcpp::Logger& logger) {
    if (target_classes.empty() ||
        std::find(target_classes.begin(), target_classes.end(), "all") != target_classes.end()) {
        std::set<int> all;
        for (size_t i = 0; i < COCO_CLASSES.size(); ++i) all.insert(static_cast<int>(i));
        return all;
    }

    std::set<int> indices;
    for (const auto& cls : target_classes) {
        // Numeric entries (e.g. "5") are treated as direct class indices,
        // matching the Python isinstance(cls, int) branch.
        bool is_numeric = !cls.empty() &&
            std::all_of(cls.begin(), cls.end(), [](unsigned char c) { return std::isdigit(c); });
        if (is_numeric) {
            int idx = std::stoi(cls);
            if (idx >= 0 && idx < static_cast<int>(COCO_CLASSES.size())) {
                indices.insert(idx);
            } else {
                RCLCPP_WARN(logger, "Class index %d out of range", idx);
            }
            continue;
        }
        std::string lower = cls;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
        auto it = std::find_if(COCO_CLASSES.begin(), COCO_CLASSES.end(), [&](const std::string& name) {
            std::string n = name;
            std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return std::tolower(c); });
            return n == lower;
        });
        if (it != COCO_CLASSES.end()) {
            indices.insert(static_cast<int>(std::distance(COCO_CLASSES.begin(), it)));
        } else {
            RCLCPP_WARN(logger, "Class name '%s' not found in COCO classes", cls.c_str());
        }
    }
    return indices;
}

// ── PersonDetectionNode ──────────────────────────────────────────────────────

PersonDetectionNode::PersonDetectionNode(const std::string& node_name)
    : rclcpp_lifecycle::LifecycleNode(node_name) {}

PersonDetectionNode::CallbackReturn PersonDetectionNode::on_configure(const rclcpp_lifecycle::State&) {
    config_ = loadConfiguration(this);

    pub_objects_ = create_publisher<dec_interfaces::msg::PersonDetection>("/person_detection/data", 10);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/person_detection/debug", 1);
    depth_debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/person_detection/depth_debug", 1);

    color_image_ = cv::Mat();
    depth_image_ = cv::Mat();

    use_compressed_ = config_.use_compressed;
    camera_type_ = config_.camera;
    verbose_mode_ = config_.verbose_mode;
    image_timeout_ = config_.image_timeout;
    target_class_indices_ = getClassIndices(config_.target_classes, get_logger());

    if (verbose_mode_) {
        std::string names;
        for (int idx : target_class_indices_) names += COCO_CLASSES[idx] + " ";
        RCLCPP_INFO(get_logger(), "Tracking classes: %s", names.c_str());
    }

    node_name_ = get_name();
    timer_start_ = get_clock()->now();
    last_image_time_.reset();

    object_colors_.clear();
    return CallbackReturn::SUCCESS;
}

PersonDetectionNode::CallbackReturn PersonDetectionNode::on_activate(const rclcpp_lifecycle::State& state) {
    LifecycleNode::on_activate(state);
    vis_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / 30.0)),
        std::bind(&PersonDetectionNode::visualizationCallback, this));
    status_timer_ = create_wall_timer(
        std::chrono::seconds(10), std::bind(&PersonDetectionNode::statusCallback, this));
    return CallbackReturn::SUCCESS;
}

PersonDetectionNode::CallbackReturn PersonDetectionNode::on_deactivate(const rclcpp_lifecycle::State& state) {
    if (vis_timer_) {
        vis_timer_->cancel();
        vis_timer_.reset();
    }
    if (status_timer_) {
        status_timer_->cancel();
        status_timer_.reset();
    }
    LifecycleNode::on_deactivate(state);
    return CallbackReturn::SUCCESS;
}

PersonDetectionNode::CallbackReturn PersonDetectionNode::on_cleanup(const rclcpp_lifecycle::State&) {
    pub_objects_.reset();
    debug_pub_.reset();
    depth_debug_pub_.reset();
    return CallbackReturn::SUCCESS;
}

PersonDetectionNode::CallbackReturn PersonDetectionNode::on_shutdown(const rclcpp_lifecycle::State&) {
    RCLCPP_INFO(get_logger(), "%s shutting down", get_name());
    return CallbackReturn::SUCCESS;
}

void PersonDetectionNode::statusCallback() {
    RCLCPP_INFO(get_logger(), "%s: running.", node_name_.c_str());
}

void PersonDetectionNode::updateLatestFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_frame_ = frame.clone();
}

void PersonDetectionNode::visualizationCallback() {
    cv::Mat color_frame, depth_vis;
    bool have_color = false, have_depth_vis = false;
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!latest_frame_.empty()) {
            color_frame = latest_frame_.clone();
            have_color = true;
        }
        if (!depth_image_.empty()) {
            auto vis = makeDepthVis(depth_image_);
            if (vis) {
                depth_vis = *vis;
                have_depth_vis = true;
            }
        }
    }

    if (!have_color && !have_depth_vis) return;

    // Both imshow and debug-image publishing are gated behind
    // verboseMode+DISPLAY here — matches the reference Python implementation
    // exactly (unlike face_detection, which always publishes).
    if (verbose_mode_ && std::getenv("DISPLAY") != nullptr) {
        try {
            if (have_color) cv::imshow("Person Detection Debug (RGB)", color_frame);
            if (have_depth_vis) cv::imshow("Person Detection Debug (Depth)", depth_vis);
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q') {
                RCLCPP_INFO(get_logger(), "%s: User requested shutdown", node_name_.c_str());
                rclcpp::shutdown();
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "imshow failed (likely headless): %s", e.what());
        }

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

std::pair<std::string, std::string> PersonDetectionNode::getTopicNames() {
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

std::optional<std::string> PersonDetectionNode::extractTopic(const std::string& image_topic_key) {
    try {
        std::string package = ament_index_cpp::get_package_share_directory("person_detection");
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

void PersonDetectionNode::synchronizedCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data,
                                               const sensor_msgs::msg::Image::ConstSharedPtr& depth_data) {
    last_image_time_ = get_clock()->now().seconds();
    try {
        if (!depth_image_.empty() && !color_image_.empty()) {
            if (!checkCameraResolution(color_image_, depth_image_)) {
                RCLCPP_ERROR(get_logger(), "%s: Color camera and depth camera have different resolutions.",
                    node_name_.c_str());
                rclcpp::shutdown();
                return;
            }
        }

        color_image_ = cv_bridge::toCvCopy(color_data, "bgr8")->image;
        auto depth = processDepthImageMsg(depth_data);
        depth_image_ = depth ? *depth : cv::Mat();

        if (color_image_.empty() || depth_image_.empty()) {
            RCLCPP_WARN(get_logger(), "Failed to decode images");
            return;
        }
        processImages();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error in synchronizedCallback: %s", e.what());
    }
}

void PersonDetectionNode::synchronizedCallbackCompressed(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& color_data,
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& depth_data) {
    last_image_time_ = get_clock()->now().seconds();
    try {
        if (!depth_image_.empty() && !color_image_.empty()) {
            if (!checkCameraResolution(color_image_, depth_image_)) {
                RCLCPP_ERROR(get_logger(), "%s: Color camera and depth camera have different resolutions.",
                    node_name_.c_str());
                rclcpp::shutdown();
                return;
            }
        }

        color_image_ = cv::imdecode(cv::Mat(color_data->data), cv::IMREAD_COLOR);
        auto depth = processDepthCompressedMsg(depth_data);
        depth_image_ = depth ? *depth : cv::Mat();

        if (color_image_.empty() || depth_image_.empty()) {
            RCLCPP_WARN(get_logger(), "Failed to decode images");
            return;
        }
        processImages();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error in synchronizedCallback: %s", e.what());
    }
}

void PersonDetectionNode::rgbOnlyCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data) {
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

void PersonDetectionNode::rgbOnlyCallbackCompressed(
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

std::optional<cv::Mat> PersonDetectionNode::processDepthImageMsg(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
        return cv_bridge::toCvCopy(msg, "passthrough")->image;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Depth image processing error: %s", e.what());
        return std::nullopt;
    }
}

std::optional<cv::Mat> PersonDetectionNode::processDepthCompressedMsg(
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

void PersonDetectionNode::startTimeoutMonitor() {
    timeout_timer_ = create_wall_timer(std::chrono::seconds(1), std::bind(&PersonDetectionNode::checkTimeout, this));
}

void PersonDetectionNode::checkTimeout() {
    if (last_image_time_) {
        double elapsed = get_clock()->now().seconds() - *last_image_time_;
        if (elapsed > image_timeout_) {
            RCLCPP_WARN(get_logger(), "No images received for %.1fs (timeout=%.1fs)", elapsed, image_timeout_);
        }
    }
}

bool PersonDetectionNode::checkCameraResolution(const cv::Mat& color_image, const cv::Mat& depth_image) const {
    if (color_image.empty() || depth_image.empty()) return false;
    return color_image.rows == depth_image.rows && color_image.cols == depth_image.cols;
}

std::optional<cv::Mat> PersonDetectionNode::makeDepthVis(const cv::Mat& depth) const {
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

std::optional<float> PersonDetectionNode::getDepthInRegion(double centroid_x, double centroid_y, double box_width,
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

    double sum = 0.0;
    int count = 0;
    for (int r = 0; r < roi_f.rows; ++r) {
        const float* row = roi_f.ptr<float>(r);
        for (int c = 0; c < roi_f.cols; ++c) {
            if (std::isfinite(row[c]) && row[c] > 0.0f) {
                sum += row[c];
                ++count;
            }
        }
    }
    if (count == 0) return std::nullopt;
    return static_cast<float>((sum / count) / 1000.0);
}

cv::Scalar PersonDetectionNode::generateDarkColor() {
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_int_distribution<int> dist(0, 150);
    while (true) {
        int b = dist(gen), g = dist(gen), r = dist(gen);
        double brightness = 0.299 * r + 0.587 * g + 0.114 * b;
        if (brightness < 130.0) return cv::Scalar(b, g, r);
    }
}

void PersonDetectionNode::processImages() {
    bool depth_required = camera_type_ != "pepper";
    if (color_image_.empty() || (depth_required && depth_image_.empty())) return;

    if (!depth_image_.empty() && !checkCameraResolution(color_image_, depth_image_)) {
        RCLCPP_WARN(get_logger(), "%s: Color and depth image resolutions do not match", node_name_.c_str());
        return;
    }

    cv::Mat frame = color_image_.clone();
    std::vector<Eigen::Vector4d> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    detectObject(frame, boxes, scores, class_ids);

    // Filter detections by target classes.
    if (!boxes.empty() && target_class_indices_.size() < COCO_CLASSES.size()) {
        std::vector<Eigen::Vector4d> filtered_boxes;
        std::vector<float> filtered_scores;
        std::vector<int> filtered_class_ids;
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (target_class_indices_.count(class_ids[i])) {
                filtered_boxes.push_back(boxes[i]);
                filtered_scores.push_back(scores[i]);
                filtered_class_ids.push_back(class_ids[i]);
            }
        }
        boxes = std::move(filtered_boxes);
        scores = std::move(filtered_scores);
        class_ids = std::move(filtered_class_ids);
    }

    if (!boxes.empty()) {
        auto tracked = updateTracker(boxes, scores, class_ids);
        auto tracking_data = prepareTrackingData(tracked);
        cv::Mat annotated = drawTrackedObjects(frame, tracked, tracking_data);
        updateLatestFrame(annotated);
        publishObjectDetection(tracking_data);
    } else {
        updateLatestFrame(frame);
    }
}

std::vector<TrackingDatum> PersonDetectionNode::prepareTrackingData(const byte_tracker::Detections& tracked) {
    std::vector<TrackingDatum> tracking_data;
    for (size_t i = 0; i < tracked.xyxy.size(); ++i) {
        double x1 = tracked.xyxy[i][0], y1 = tracked.xyxy[i][1];
        double x2 = tracked.xyxy[i][2], y2 = tracked.xyxy[i][3];
        double width = x2 - x1;
        double height = y2 - y1;
        double centroid_x = (x1 + x2) / 2.0;
        double centroid_y = (y1 + y2) / 2.0;

        auto depth = getDepthInRegion(centroid_x, centroid_y, width, height);

        TrackingDatum datum;
        datum.track_id = std::to_string(tracked.tracker_id[i]);
        datum.class_id = tracked.class_id[i];
        datum.class_name = datum.class_id < static_cast<int>(COCO_CLASSES.size())
            ? COCO_CLASSES[datum.class_id] : "unknown";
        datum.confidence = tracked.confidence[i];
        datum.centroid.x = centroid_x;
        datum.centroid.y = centroid_y;
        datum.centroid.z = depth.value_or(0.0f);
        datum.width = static_cast<float>(width);
        datum.height = static_cast<float>(height);
        tracking_data.push_back(datum);
    }
    return tracking_data;
}

void PersonDetectionNode::publishObjectDetection(const std::vector<TrackingDatum>& tracking_data) {
    if (tracking_data.empty()) return;

    dec_interfaces::msg::PersonDetection msg;
    for (const auto& d : tracking_data) {
        msg.person_label_id.push_back(d.track_id);
        msg.class_names.push_back(d.class_name);
        msg.class_ids.push_back(d.class_id);
        msg.confidences.push_back(d.confidence);
        msg.centroids.push_back(d.centroid);
        msg.width.push_back(d.width);
        msg.height.push_back(d.height);
    }
    pub_objects_->publish(msg);
}

cv::Mat PersonDetectionNode::drawTrackedObjects(const cv::Mat& frame, const byte_tracker::Detections& tracked,
                                                const std::vector<TrackingDatum>& tracking_data) {
    cv::Mat output = frame.clone();
    for (size_t i = 0; i < tracked.xyxy.size(); ++i) {
        int x1 = static_cast<int>(tracked.xyxy[i][0]);
        int y1 = static_cast<int>(tracked.xyxy[i][1]);
        int x2 = static_cast<int>(tracked.xyxy[i][2]);
        int y2 = static_cast<int>(tracked.xyxy[i][3]);
        int track_id = tracked.tracker_id[i];
        int class_id = tracked.class_id[i];
        float confidence = tracked.confidence[i];
        std::string class_name = class_id < static_cast<int>(COCO_CLASSES.size())
            ? COCO_CLASSES[class_id] : "unknown";

        if (object_colors_.find(track_id) == object_colors_.end()) {
            object_colors_[track_id] = generateDarkColor();
        }
        cv::Scalar color = object_colors_[track_id];

        cv::rectangle(output, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        char label[128];
        std::snprintf(label, sizeof(label), "%s #%d (%.2f)", class_name.c_str(), track_id, confidence);
        cv::putText(output, label, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

        float depth = 0.0f;
        for (const auto& d : tracking_data) {
            if (d.track_id == std::to_string(track_id)) {
                depth = d.centroid.z;
                break;
            }
        }
        std::string depth_str = depth > 0.0f
            ? "Depth: " + std::to_string(depth).substr(0, 4) + " m" : "Depth: Unknown";
        cv::putText(output, depth_str, cv::Point(x1, y2 + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
    return output;
}

// ── Yolov11Node ──────────────────────────────────────────────────────────────

Yolov11Node::Yolov11Node() : PersonDetectionNode() {}

Yolov11Node::CallbackReturn Yolov11Node::on_configure(const rclcpp_lifecycle::State& state) {
    auto ret = PersonDetectionNode::on_configure(state);
    if (ret != CallbackReturn::SUCCESS) return ret;

    confidence_threshold_ = config_.confidence_threshold;
    track_thresh_ = config_.track_threshold;
    track_buffer_ = config_.track_buffer;
    match_thresh_ = config_.match_threshold;
    frame_rate_ = config_.frame_rate;

    if (!initModel()) {
        RCLCPP_ERROR(get_logger(), "%s: failed to load ONNX model", node_name_.c_str());
        return CallbackReturn::FAILURE;
    }

    tracker_ = std::make_unique<byte_tracker::ByteTrack>(track_thresh_, track_buffer_, match_thresh_, frame_rate_);

    RCLCPP_INFO(get_logger(), "%s: configured — YOLOv11 + ByteTrack ready", node_name_.c_str());
    return CallbackReturn::SUCCESS;
}

Yolov11Node::CallbackReturn Yolov11Node::on_activate(const rclcpp_lifecycle::State& state) {
    auto ret = PersonDetectionNode::on_activate(state);
    if (ret != CallbackReturn::SUCCESS) return ret;

    if (!createCameraSubscriptions()) {
        RCLCPP_ERROR(get_logger(), "%s: failed to set up camera subscriptions", node_name_.c_str());
        return CallbackReturn::FAILURE;
    }

    startTimeoutMonitor();
    RCLCPP_INFO(get_logger(), "%s: activated — processing frames", node_name_.c_str());
    return CallbackReturn::SUCCESS;
}

Yolov11Node::CallbackReturn Yolov11Node::on_deactivate(const rclcpp_lifecycle::State& state) {
    sync_.reset();
    color_sub_.reset();
    depth_sub_.reset();
    sync_compressed_.reset();
    color_sub_compressed_.reset();
    depth_sub_compressed_.reset();
    color_sub_plain_.reset();
    color_sub_plain_compressed_.reset();
    if (timeout_timer_) {
        timeout_timer_->cancel();
        timeout_timer_.reset();
    }
    return PersonDetectionNode::on_deactivate(state);
}

Yolov11Node::CallbackReturn Yolov11Node::on_cleanup(const rclcpp_lifecycle::State& state) {
    session_.reset();
    ort_env_.reset();
    tracker_.reset();
    return PersonDetectionNode::on_cleanup(state);
}

bool Yolov11Node::createCameraSubscriptions() {
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
                    std::bind(&Yolov11Node::rgbOnlyCallbackCompressed, this, std::placeholders::_1));
            } else {
                color_sub_plain_ = create_subscription<sensor_msgs::msg::Image>(
                    color_topic, rclcpp::SensorDataQoS(),
                    std::bind(&Yolov11Node::rgbOnlyCallback, this, std::placeholders::_1));
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
                std::bind(&Yolov11Node::synchronizedCallbackCompressed, this,
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
                std::bind(&Yolov11Node::synchronizedCallback, this,
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

bool Yolov11Node::initModel() {
    try {
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "person_detection");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        bool cuda_enabled = false;
        try {
            OrtCUDAProviderOptions cuda_options{};
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            cuda_enabled = true;
        } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "%s: CUDAExecutionProvider unavailable (%s); running on CPU",
                node_name_.c_str(), e.what());
        }

        std::string package_path = ament_index_cpp::get_package_share_directory("person_detection");
        std::string model_path = package_path + "/models/person_detection_yolov11m.onnx";
        if (!std::filesystem::exists(model_path)) {
            RCLCPP_ERROR(get_logger(), "Model file not found: %s", model_path.c_str());
            return false;
        }

        session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            input_names_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            output_names_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
        }

        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();  // [N, C, H, W]
        input_height_ = input_shape[2];
        input_width_ = input_shape[3];

        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "%s: CUDAExecutionProvider %s", node_name_.c_str(),
                cuda_enabled ? "is active — running on GPU for faster inference" : "not available — running on CPU");
        }

        // Warmup run to load model weights into memory.
        std::vector<float> dummy(static_cast<size_t>(input_height_ * input_width_ * 3), 0.0f);
        std::array<int64_t, 4> shape = {1, 3, input_height_, input_width_};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, dummy.data(), dummy.size(), shape.data(), shape.size());

        std::vector<const char*> input_ptrs;
        for (auto& n : input_names_) input_ptrs.push_back(n.c_str());
        std::vector<const char*> output_ptrs;
        for (auto& n : output_names_) output_ptrs.push_back(n.c_str());

        session_->Run(Ort::RunOptions{nullptr}, input_ptrs.data(), &input_tensor, 1, output_ptrs.data(),
            output_ptrs.size());

        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "%s: ONNX model loaded successfully.", node_name_.c_str());
        }
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "%s: Failed to initialize ONNX model: %s", node_name_.c_str(), e.what());
        return false;
    }
}

Ort::Value Yolov11Node::prepareInput(const cv::Mat& image) {
    orig_height_ = image.rows;
    orig_width_ = image.cols;

    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(static_cast<int>(input_width_), static_cast<int>(input_height_)));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // HWC -> CHW
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    static thread_local std::vector<float> chw;
    chw.resize(static_cast<size_t>(3 * input_height_ * input_width_));
    size_t plane = static_cast<size_t>(input_height_ * input_width_);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(chw.data() + c * plane, channels[c].ptr<float>(), plane * sizeof(float));
    }

    std::array<int64_t, 4> shape = {1, 3, input_height_, input_width_};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(mem_info, chw.data(), chw.size(), shape.data(), shape.size());
}

Eigen::Vector4d Yolov11Node::rescaleBox(const Eigen::Vector4d& box, double orig_w, double orig_h, double input_w,
                                        double input_h) {
    return {box[0] * (orig_w / input_w), box[1] * (orig_h / input_h), box[2] * (orig_w / input_w),
            box[3] * (orig_h / input_h)};
}

Eigen::Vector4d Yolov11Node::xywhToXyxy(const Eigen::Vector4d& box) {
    double x = box[0], y = box[1], w = box[2], h = box[3];
    return {x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0};
}

double Yolov11Node::computeIou(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
    double x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]);
    double x2 = std::min(a[2], b[2]), y2 = std::min(a[3], b[3]);
    double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    double area_a = (a[2] - a[0]) * (a[3] - a[1]);
    double area_b = (b[2] - b[0]) * (b[3] - b[1]);
    return inter / (area_a + area_b - inter + 1e-6);
}

std::vector<int> Yolov11Node::nms(const std::vector<Eigen::Vector4d>& boxes, const std::vector<float>& scores,
                                  double iou_threshold) {
    std::vector<int> order(boxes.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<int> keep;
    std::vector<bool> removed(boxes.size(), false);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        int i = order[oi];
        if (removed[i]) continue;
        keep.push_back(i);
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            int j = order[oj];
            if (removed[j]) continue;
            if (computeIou(boxes[i], boxes[j]) >= iou_threshold) removed[j] = true;
        }
    }
    return keep;
}

std::vector<int> Yolov11Node::multiclassNms(const std::vector<Eigen::Vector4d>& boxes,
                                            const std::vector<float>& scores, const std::vector<int>& class_ids,
                                            double iou_threshold) {
    std::set<int> unique_classes(class_ids.begin(), class_ids.end());
    std::vector<int> final_keep;
    for (int cid : unique_classes) {
        std::vector<int> idx;
        for (size_t i = 0; i < class_ids.size(); ++i) {
            if (class_ids[i] == cid) idx.push_back(static_cast<int>(i));
        }
        std::vector<Eigen::Vector4d> cls_boxes;
        std::vector<float> cls_scores;
        for (int i : idx) {
            cls_boxes.push_back(boxes[i]);
            cls_scores.push_back(scores[i]);
        }
        auto keep = nms(cls_boxes, cls_scores, iou_threshold);
        for (int k : keep) final_keep.push_back(idx[k]);
    }
    return final_keep;
}

bool Yolov11Node::detectObject(const cv::Mat& image, std::vector<Eigen::Vector4d>& out_boxes,
                               std::vector<float>& out_scores, std::vector<int>& out_class_ids) {
    Ort::Value input_tensor = prepareInput(image);

    std::vector<const char*> input_ptrs;
    for (auto& n : input_names_) input_ptrs.push_back(n.c_str());
    std::vector<const char*> output_ptrs;
    for (auto& n : output_names_) output_ptrs.push_back(n.c_str());

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_ptrs.data(), &input_tensor, 1, output_ptrs.data(), output_ptrs.size());

    processOutput(outputs[0], out_boxes, out_scores, out_class_ids);
    return true;
}

void Yolov11Node::processOutput(const Ort::Value& output, std::vector<Eigen::Vector4d>& out_boxes,
                                std::vector<float>& out_scores, std::vector<int>& out_class_ids) {
    // YOLOv11 output: [1, 84, 8400] — 4 box coords (cx,cy,w,h, input-pixel
    // scale) + 80 class logits (need sigmoid), for 8400 candidate predictions.
    auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    int64_t num_attrs = shape[1];
    int64_t num_preds = shape[2];
    int num_classes = static_cast<int>(num_attrs - 4);
    const float* data = output.GetTensorData<float>();

    std::vector<Eigen::Vector4d> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    for (int64_t i = 0; i < num_preds; ++i) {
        float best_prob = -1.0f;
        int best_class = 0;
        for (int c = 0; c < num_classes; ++c) {
            float logit = data[(4 + c) * num_preds + i];
            float prob = 1.0f / (1.0f + std::exp(-logit));
            if (prob > best_prob) {
                best_prob = prob;
                best_class = c;
            }
        }
        if (best_prob <= confidence_threshold_) continue;

        double cx = data[0 * num_preds + i];
        double cy = data[1 * num_preds + i];
        double w = data[2 * num_preds + i];
        double h = data[3 * num_preds + i];

        Eigen::Vector4d rescaled = rescaleBox({cx, cy, w, h}, orig_width_, orig_height_,
            static_cast<double>(input_width_), static_cast<double>(input_height_));
        boxes.push_back(xywhToXyxy(rescaled));
        scores.push_back(best_prob);
        class_ids.push_back(best_class);
    }

    if (boxes.empty()) return;

    // NOTE: confidence_threshold_ is reused as the NMS IoU threshold here,
    // matching the reference Python implementation's (likely unintentional
    // but functioning) reuse of the same config value for both purposes.
    auto keep = multiclassNms(boxes, scores, class_ids, confidence_threshold_);
    for (int idx : keep) {
        out_boxes.push_back(boxes[idx]);
        out_scores.push_back(scores[idx]);
        out_class_ids.push_back(class_ids[idx]);
    }
}

byte_tracker::Detections Yolov11Node::updateTracker(const std::vector<Eigen::Vector4d>& boxes,
                                                    const std::vector<float>& scores,
                                                    const std::vector<int>& class_ids) {
    byte_tracker::Detections det;
    det.xyxy = boxes;
    det.confidence = scores;
    det.class_id = class_ids;
    return tracker_->updateWithDetections(det);
}
