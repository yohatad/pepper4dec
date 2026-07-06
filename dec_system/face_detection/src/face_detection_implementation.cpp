/* face_detection_implementation.cpp
 *
 * Implements YOLOONNX (the goldYOLO face-detector wrapper), FaceDetectionNode
 * (publishers, debug visualization, camera topic resolution, depth lookup),
 * and SixDrepNet (YOLO face detection + SixDrepNet head-pose/mutual-gaze
 * inference, with Hungarian face-to-person matching). See
 * face_detection_interface.h for the full subscriber/publisher/parameter
 * reference and the lifecycle state-machine diagram.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 06, 2026
 * Version: v1.0 - C++ port of face_detection_implementation.py
 */

#include "face_detection/face_detection_interface.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <yaml-cpp/yaml.h>
#include <rmw/qos_profiles.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <set>
#include <thread>

FaceDetectionConfig loadConfiguration() {
    FaceDetectionConfig config;
    try {
        std::string package_path = ament_index_cpp::get_package_share_directory("face_detection");
        std::string config_file = package_path + "/config/face_detection_configuration.yaml";
        if (std::filesystem::exists(config_file)) {
            YAML::Node yaml = YAML::LoadFile(config_file);
            if (yaml["algorithm"]) config.algorithm = yaml["algorithm"].as<std::string>();
            if (yaml["useCompressed"]) config.use_compressed = yaml["useCompressed"].as<bool>();
            if (yaml["camera"]) config.camera = yaml["camera"].as<std::string>();
            if (yaml["verboseMode"]) config.verbose_mode = yaml["verboseMode"].as<bool>();
            if (yaml["imageTimeout"]) config.image_timeout = yaml["imageTimeout"].as<double>();
            if (yaml["sixdrepnetConfidence"]) config.sixdrepnet_confidence = yaml["sixdrepnetConfidence"].as<double>();
            if (yaml["sixdrepnetHeadposeAngle"]) {
                config.sixdrepnet_headpose_angle = yaml["sixdrepnetHeadposeAngle"].as<double>();
            }
            if (yaml["requirePersonDetection"]) {
                config.require_person_detection = yaml["requirePersonDetection"].as<bool>();
            }
            if (yaml["personDetectionTimeout"]) {
                config.person_detection_timeout = yaml["personDetectionTimeout"].as<double>();
            }
            if (yaml["prioritizeFaceDepth"]) config.prioritize_face_depth = yaml["prioritizeFaceDepth"].as<bool>();
            std::cout << "Loaded configuration from " << config_file << std::endl;
        } else {
            std::cout << "Warning: Configuration file not found at " << config_file << ", using defaults"
                      << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Error reading configuration file: " << e.what() << std::endl;
        std::cout << "Using default configuration values" << std::endl;
    }
    return config;
}

// ── YOLOONNX ─────────────────────────────────────────────────────────────────

YOLOONNX::YOLOONNX(const std::string& model_path, double class_score_th) : class_score_th_(class_score_th) {
    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "face_detection_yolo");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
        OrtCUDAProviderOptions cuda_options{};
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    } catch (const std::exception&) {
        // Falls back to CPU; the caller logs provider status separately.
    }

    session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        input_names_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        output_names_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
    }

    auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_height_ = input_shape[2];
    input_width_ = input_shape[3];

    // Warmup run to load model weights into memory.
    std::vector<float> dummy(static_cast<size_t>(3 * input_height_ * input_width_), 0.0f);
    std::array<int64_t, 4> shape = {1, 3, input_height_, input_width_};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, dummy.data(), dummy.size(), shape.data(), shape.size());
    std::vector<const char*> in_ptrs;
    for (auto& n : input_names_) in_ptrs.push_back(n.c_str());
    std::vector<const char*> out_ptrs;
    for (auto& n : output_names_) out_ptrs.push_back(n.c_str());
    session_->Run(Ort::RunOptions{nullptr}, in_ptrs.data(), &input_tensor, 1, out_ptrs.data(), out_ptrs.size());
}

cv::Mat YOLOONNX::preprocess(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(static_cast<int>(input_width_), static_cast<int>(input_height_)));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    return rgb;
}

std::pair<std::vector<cv::Rect2d>, std::vector<float>> YOLOONNX::detect(const cv::Mat& image) {
    cv::Mat pre = preprocess(image);
    std::vector<cv::Mat> channels(3);
    cv::split(pre, channels);

    std::vector<float> chw(static_cast<size_t>(3 * input_height_ * input_width_));
    size_t plane = static_cast<size_t>(input_height_ * input_width_);
    for (int c = 0; c < 3; ++c) {
        std::memcpy(chw.data() + c * plane, channels[c].ptr<float>(), plane * sizeof(float));
    }

    std::array<int64_t, 4> shape = {1, 3, input_height_, input_width_};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, chw.data(), chw.size(), shape.data(), shape.size());

    std::vector<const char*> in_ptrs;
    for (auto& n : input_names_) in_ptrs.push_back(n.c_str());
    std::vector<const char*> out_ptrs;
    for (auto& n : output_names_) out_ptrs.push_back(n.c_str());

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr}, in_ptrs.data(), &input_tensor, 1, out_ptrs.data(), out_ptrs.size());

    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t num_boxes = out_shape[0];
    int64_t num_attrs = out_shape.size() > 1 ? out_shape[1] : 0;
    const float* data = outputs[0].GetTensorData<float>();

    std::vector<float> raw(data, data + num_boxes * num_attrs);
    return postprocess(image, raw, num_boxes, num_attrs);
}

std::pair<std::vector<cv::Rect2d>, std::vector<float>> YOLOONNX::postprocess(
    const cv::Mat& image, const std::vector<float>& raw, int64_t num_boxes, int64_t num_attrs) {
    // goldYOLO output rows: [batch_idx, class_id, x1, y1, x2, y2, score]
    // (NMS is baked into the exported graph, unlike Yolov11Node's raw output).
    int img_h = image.rows, img_w = image.cols;
    std::vector<cv::Rect2d> result_boxes;
    std::vector<float> result_scores;
    if (num_boxes > 0 && num_attrs >= 7) {
        for (int64_t i = 0; i < num_boxes; ++i) {
            const float* row = raw.data() + i * num_attrs;
            float score = row[6];
            if (score <= class_score_th_) continue;

            int x_min = static_cast<int>(std::max(static_cast<double>(row[2]), 0.0) * img_w / input_width_);
            int y_min = static_cast<int>(std::max(static_cast<double>(row[3]), 0.0) * img_h / input_height_);
            int x_max = static_cast<int>(
                std::min(static_cast<double>(row[4]), static_cast<double>(input_width_)) * img_w / input_width_);
            int y_max = static_cast<int>(
                std::min(static_cast<double>(row[5]), static_cast<double>(input_height_)) * img_h / input_height_);

            result_boxes.emplace_back(cv::Point2d(x_min, y_min), cv::Point2d(x_max, y_max));
            result_scores.push_back(score);
        }
    }
    return {result_boxes, result_scores};
}

// ── FaceDetectionNode ────────────────────────────────────────────────────────

FaceDetectionNode::FaceDetectionNode(const FaceDetectionConfig& config, const std::string& node_name)
    : rclcpp_lifecycle::LifecycleNode(node_name),
      config_(config),
      face_tracker_(0.5f, 30, 0.3f, 15) {}

FaceDetectionNode::CallbackReturn FaceDetectionNode::on_configure(const rclcpp_lifecycle::State&) {
    pub_gaze_ = create_publisher<dec_interfaces::msg::FaceDetection>("/face_detection/data", 10);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/face_detection/debug", 1);
    depth_debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/face_detection/depth_debug", 1);

    color_image_ = cv::Mat();
    depth_image_ = cv::Mat();

    use_compressed_ = config_.use_compressed;
    camera_type_ = config_.camera;
    verbose_mode_ = config_.verbose_mode;
    image_timeout_ = config_.image_timeout;
    require_person_detection_ = config_.require_person_detection;
    person_detection_timeout_ = config_.person_detection_timeout;
    prioritize_face_depth_ = config_.prioritize_face_depth;

    node_name_ = get_name();
    timer_start_ = get_clock()->now();
    last_image_time_.reset();

    latest_person_detections_.reset();
    latest_person_detections_timestamp_.reset();
    face_colors_.clear();
    face_tracker_.reset();

    return CallbackReturn::SUCCESS;
}

FaceDetectionNode::CallbackReturn FaceDetectionNode::on_activate(const rclcpp_lifecycle::State& state) {
    LifecycleNode::on_activate(state);

    if (require_person_detection_) {
        person_detection_sub_ = create_subscription<dec_interfaces::msg::PersonDetection>(
            "/person_detection/data", 10,
            std::bind(&FaceDetectionNode::personDetectionCallback, this, std::placeholders::_1));
        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "%s: person detection ENABLED — subscribed to /person_detection/data",
                node_name_.c_str());
        }
    } else if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "%s: person detection DISABLED — standalone mode", node_name_.c_str());
    }

    vis_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / 30.0)),
        std::bind(&FaceDetectionNode::visualizationCallback, this));
    return CallbackReturn::SUCCESS;
}

FaceDetectionNode::CallbackReturn FaceDetectionNode::on_deactivate(const rclcpp_lifecycle::State& state) {
    if (vis_timer_) {
        vis_timer_->cancel();
        vis_timer_.reset();
    }
    if (require_person_detection_ && person_detection_sub_) {
        person_detection_sub_.reset();
    }
    LifecycleNode::on_deactivate(state);
    return CallbackReturn::SUCCESS;
}

FaceDetectionNode::CallbackReturn FaceDetectionNode::on_cleanup(const rclcpp_lifecycle::State&) {
    pub_gaze_.reset();
    debug_pub_.reset();
    depth_debug_pub_.reset();
    return CallbackReturn::SUCCESS;
}

FaceDetectionNode::CallbackReturn FaceDetectionNode::on_shutdown(const rclcpp_lifecycle::State&) {
    RCLCPP_INFO(get_logger(), "%s shutting down", get_name());
    return CallbackReturn::SUCCESS;
}

void FaceDetectionNode::cleanup() {
    try {
        cv::destroyAllWindows();
        RCLCPP_INFO(get_logger(), "Cleanup completed");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error during cleanup: %s", e.what());
    }
}

void FaceDetectionNode::updateLatestFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_frame_ = frame.clone();
    latest_depth_ = depth_image_.empty() ? cv::Mat() : depth_image_.clone();
}

void FaceDetectionNode::visualizationCallback() {
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

    if (verbose_mode_ && std::getenv("DISPLAY") != nullptr) {
        try {
            if (have_color) cv::imshow("Face Detection Debug (RGB)", color_frame);
            if (have_depth_vis) cv::imshow("Face Detection Debug (Depth)", depth_vis);
            cv::waitKey(1);
        } catch (const std::exception& e) {
            RCLCPP_WARN(get_logger(), "imshow failed (likely headless): %s", e.what());
        }
    }

    // Unlike person_detection, debug images are always published here,
    // regardless of verboseMode — matches the reference implementation.
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

std::pair<std::string, std::string> FaceDetectionNode::getTopicNames() {
    std::string rgb_key, depth_key;
    if (camera_type_ == "realsense") {
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

std::optional<std::string> FaceDetectionNode::extractTopic(const std::string& image_topic_key) {
    try {
        std::string package = ament_index_cpp::get_package_share_directory("face_detection");
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

void FaceDetectionNode::synchronizedCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data,
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

void FaceDetectionNode::synchronizedCallbackCompressed(
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

void FaceDetectionNode::rgbOnlyCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data) {
    last_image_time_ = get_clock()->now().seconds();
    try {
        color_image_ = cv_bridge::toCvCopy(color_data, "bgr8")->image;
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

void FaceDetectionNode::rgbOnlyCallbackCompressed(
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

std::optional<cv::Mat> FaceDetectionNode::processDepthImageMsg(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
        return cv_bridge::toCvCopy(msg, "passthrough")->image;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Depth image processing error: %s", e.what());
        return std::nullopt;
    }
}

std::optional<cv::Mat> FaceDetectionNode::processDepthCompressedMsg(
    const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg) {
    try {
        if (!msg->format.empty() && msg->format.find("compressedDepth") != std::string::npos) {
            constexpr size_t kHeaderSize = 12;
            if (msg->data.size() <= kHeaderSize) return std::nullopt;
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

void FaceDetectionNode::startTimeoutMonitor() {
    timeout_timer_ = create_wall_timer(std::chrono::seconds(1), std::bind(&FaceDetectionNode::checkTimeout, this));
}

void FaceDetectionNode::checkTimeout() {
    if (last_image_time_) {
        double elapsed = get_clock()->now().seconds() - *last_image_time_;
        if (elapsed > image_timeout_) {
            RCLCPP_WARN(get_logger(), "No images received for %.1fs (timeout=%.1fs)", elapsed, image_timeout_);
        }
    }
}

bool FaceDetectionNode::checkCameraResolution(const cv::Mat& color_image, const cv::Mat& depth_image) const {
    if (color_image.empty() || depth_image.empty()) return false;
    return color_image.rows == depth_image.rows && color_image.cols == depth_image.cols;
}

std::optional<cv::Mat> FaceDetectionNode::makeDepthVis(const cv::Mat& depth) const {
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

std::optional<float> FaceDetectionNode::getDepthInRegion(double centroid_x, double centroid_y, double box_width,
                                                          double box_height, double region_scale) const {
    if (depth_image_.empty()) return std::nullopt;

    int region_width = std::max(5, static_cast<int>(box_width * region_scale));
    int region_height = std::max(5, static_cast<int>(box_height * region_scale));
    int x_start = std::max(0, static_cast<int>(centroid_x - region_width / 2.0));
    int y_start = std::max(0, static_cast<int>(centroid_y - region_height / 2.0));
    int x_end = std::min(depth_image_.cols, x_start + region_width);
    int y_end = std::min(depth_image_.rows, y_start + region_height);
    if (x_start >= x_end || y_start >= y_end) return std::nullopt;

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

    // Face detection uses the median depth (person_detection uses the mean).
    std::sort(valid.begin(), valid.end());
    size_t n = valid.size();
    float median = (n % 2 == 1) ? valid[n / 2] : (valid[n / 2 - 1] + valid[n / 2]) / 2.0f;
    return median / 1000.0f;
}

cv::Scalar FaceDetectionNode::generateDarkColor() {
    static thread_local std::mt19937 gen{std::random_device{}()};
    std::uniform_int_distribution<int> dist(0, 150);
    while (true) {
        int b = dist(gen), g = dist(gen), r = dist(gen);
        double brightness = 0.299 * r + 0.587 * g + 0.114 * b;
        if (brightness < 130.0) return cv::Scalar(b, g, r);
    }
}

void FaceDetectionNode::publishFaceDetection(const std::vector<FaceTrackingDatum>& tracking_data) {
    dec_interfaces::msg::FaceDetection msg;
    if (tracking_data.empty()) {
        pub_gaze_->publish(msg);
        return;
    }
    for (const auto& d : tracking_data) {
        msg.face_label_id.push_back(d.face_id);
        msg.centroids.push_back(d.centroid);
        msg.width.push_back(d.width);
        msg.height.push_back(d.height);
        msg.mutual_gaze.push_back(d.mutual_gaze);
    }
    pub_gaze_->publish(msg);
}

void FaceDetectionNode::personDetectionCallback(const dec_interfaces::msg::PersonDetection& msg) {
    std::lock_guard<std::mutex> lock(person_detections_mutex_);
    PersonSnapshot snapshot;
    snapshot.person_label_id.assign(msg.person_label_id.begin(), msg.person_label_id.end());
    snapshot.class_names.assign(msg.class_names.begin(), msg.class_names.end());
    snapshot.centroids.assign(msg.centroids.begin(), msg.centroids.end());
    snapshot.width.assign(msg.width.begin(), msg.width.end());
    snapshot.height.assign(msg.height.begin(), msg.height.end());
    latest_person_detections_ = snapshot;
    latest_person_detections_timestamp_ = get_clock()->now();
}

// ── SixDrepNet ───────────────────────────────────────────────────────────────

SixDrepNet::SixDrepNet(const FaceDetectionConfig& config) : FaceDetectionNode(config) {}

SixDrepNet::CallbackReturn SixDrepNet::on_configure(const rclcpp_lifecycle::State& state) {
    auto ret = FaceDetectionNode::on_configure(state);
    if (ret != CallbackReturn::SUCCESS) return ret;

    sixdrep_angle_ = config_.sixdrepnet_headpose_angle;

    RCLCPP_INFO(get_logger(), "%s: loading ONNX models...", node_name_.c_str());

    std::string package_path;
    try {
        package_path = ament_index_cpp::get_package_share_directory("face_detection");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "%s: failed to locate model files: %s", node_name_.c_str(), e.what());
        return CallbackReturn::FAILURE;
    }

    std::string yolo_model_path = package_path + "/models/face_detection_goldYOLO.onnx";
    std::string sixdrepnet_model_path = package_path + "/models/face_detection_sixdrepnet360.onnx";

    try {
        yolo_model_ = std::make_unique<YOLOONNX>(yolo_model_path, config_.sixdrepnet_confidence);
        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "%s: YOLOONNX loaded successfully", node_name_.c_str());
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "%s: YOLOONNX init failed: %s", node_name_.c_str(), e.what());
        return CallbackReturn::FAILURE;
    }

    try {
        sixdrepnet_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "face_detection_sixdrepnet");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        bool cuda_enabled = false;
        try {
            OrtCUDAProviderOptions cuda_options{};
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            cuda_enabled = true;
        } catch (const std::exception&) {
            // CPU fallback.
        }

        sixdrepnet_session_ = std::make_unique<Ort::Session>(*sixdrepnet_env_, sixdrepnet_model_path.c_str(),
            session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < sixdrepnet_session_->GetInputCount(); ++i) {
            sixdrepnet_input_names_.emplace_back(sixdrepnet_session_->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < sixdrepnet_session_->GetOutputCount(); ++i) {
            sixdrepnet_output_names_.emplace_back(sixdrepnet_session_->GetOutputNameAllocated(i, allocator).get());
        }

        // Warmup
        std::vector<float> dummy(1 * 3 * 224 * 224, 0.0f);
        std::array<int64_t, 4> shape = {1, 3, 224, 224};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, dummy.data(), dummy.size(), shape.data(), shape.size());
        std::vector<const char*> in_ptrs;
        for (auto& n : sixdrepnet_input_names_) in_ptrs.push_back(n.c_str());
        std::vector<const char*> out_ptrs;
        for (auto& n : sixdrepnet_output_names_) out_ptrs.push_back(n.c_str());
        sixdrepnet_session_->Run(Ort::RunOptions{nullptr}, in_ptrs.data(), &input_tensor, 1, out_ptrs.data(),
            out_ptrs.size());

        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "%s: SixDrepNet loaded — %s", node_name_.c_str(),
                cuda_enabled ? "GPU (CUDA)" : "CPU");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "%s: SixDrepNet ONNX init failed: %s", node_name_.c_str(), e.what());
        return CallbackReturn::FAILURE;
    }

    RCLCPP_INFO(get_logger(), "%s: configured — models ready", node_name_.c_str());
    return CallbackReturn::SUCCESS;
}

SixDrepNet::CallbackReturn SixDrepNet::on_activate(const rclcpp_lifecycle::State& state) {
    auto ret = FaceDetectionNode::on_activate(state);
    if (ret != CallbackReturn::SUCCESS) return ret;

    if (!createCameraSubscriptions()) {
        RCLCPP_ERROR(get_logger(), "%s: failed to set up camera subscriptions", node_name_.c_str());
        return CallbackReturn::FAILURE;
    }

    startTimeoutMonitor();
    RCLCPP_INFO(get_logger(), "%s: activated — processing frames", node_name_.c_str());
    return CallbackReturn::SUCCESS;
}

SixDrepNet::CallbackReturn SixDrepNet::on_deactivate(const rclcpp_lifecycle::State& state) {
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
    return FaceDetectionNode::on_deactivate(state);
}

SixDrepNet::CallbackReturn SixDrepNet::on_cleanup(const rclcpp_lifecycle::State& state) {
    yolo_model_.reset();
    sixdrepnet_session_.reset();
    sixdrepnet_env_.reset();
    return FaceDetectionNode::on_cleanup(state);
}

bool SixDrepNet::createCameraSubscriptions() {
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

        std::string person_det = require_person_detection_ ? "/person_detection/data" : "";
        rmw_qos_profile_t qos = rmw_qos_profile_sensor_data;

        if (camera_type_ == "pepper") {
            if (compressed) {
                color_sub_plain_compressed_ = create_subscription<sensor_msgs::msg::CompressedImage>(
                    color_topic, rclcpp::SensorDataQoS(),
                    std::bind(&SixDrepNet::rgbOnlyCallbackCompressed, this, std::placeholders::_1));
            } else {
                color_sub_plain_ = create_subscription<sensor_msgs::msg::Image>(
                    color_topic, rclcpp::SensorDataQoS(),
                    std::bind(&SixDrepNet::rgbOnlyCallback, this, std::placeholders::_1));
            }
            RCLCPP_INFO(get_logger(), "%s: subscribed to %s (depth disabled for Pepper)%s",
                node_name_.c_str(), color_topic.c_str(),
                person_det.empty() ? "" : (", " + person_det).c_str());
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
                std::bind(&SixDrepNet::synchronizedCallbackCompressed, this,
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
                std::bind(&SixDrepNet::synchronizedCallback, this,
                    std::placeholders::_1, std::placeholders::_2));
        }

        RCLCPP_INFO(get_logger(), "%s: subscribed to %s, %s%s", node_name_.c_str(), color_topic.c_str(),
            depth_topic_sub.c_str(), person_det.empty() ? "" : (", " + person_det).c_str());
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "%s: camera subscription failed: %s", node_name_.c_str(), e.what());
        return false;
    }
}

void SixDrepNet::drawAxis(cv::Mat& img, double yaw, double pitch, double roll, double tdx, double tdy, double size) {
    double pitch_r = pitch * M_PI / 180.0;
    double yaw_r = -yaw * M_PI / 180.0;
    double roll_r = roll * M_PI / 180.0;

    double x1 = size * (std::cos(yaw_r) * std::cos(roll_r)) + tdx;
    double y1 = size * (std::cos(pitch_r) * std::sin(roll_r) +
        std::sin(pitch_r) * std::sin(yaw_r) * std::cos(roll_r)) + tdy;
    double x2 = size * (-std::cos(yaw_r) * std::sin(roll_r)) + tdx;
    double y2 = size * (std::cos(pitch_r) * std::cos(roll_r) -
        std::sin(pitch_r) * std::sin(yaw_r) * std::sin(roll_r)) + tdy;
    double x3 = size * std::sin(yaw_r) + tdx;
    double y3 = size * (-std::cos(yaw_r) * std::sin(pitch_r)) + tdy;

    cv::Point origin(static_cast<int>(tdx), static_cast<int>(tdy));
    cv::line(img, origin, cv::Point(static_cast<int>(x1), static_cast<int>(y1)), cv::Scalar(0, 0, 255), 2);
    cv::line(img, origin, cv::Point(static_cast<int>(x2), static_cast<int>(y2)), cv::Scalar(0, 255, 0), 2);
    cv::line(img, origin, cv::Point(static_cast<int>(x3), static_cast<int>(y3)), cv::Scalar(255, 0, 0), 2);
}

double SixDrepNet::calculateMatchingCost(const FaceCandidate& face, const PersonCandidate& person) const {
    if (!(person.x1 <= face.cx && face.cx <= person.x2 && person.y1 <= face.cy && face.cy <= person.y2)) {
        return kImpossibleMatchCost;
    }

    double person_cx = (person.x1 + person.x2) / 2.0;
    double person_cy = (person.y1 + person.y2) / 2.0;
    double person_width = person.x2 - person.x1;
    double person_height = person.y2 - person.y1;
    double person_area = person_width * person_height;
    if (person_area == 0.0) return kImpossibleMatchCost;

    double distance = std::sqrt(std::pow(face.cx - person_cx, 2) + std::pow(face.cy - person_cy, 2));
    double normalized_distance = distance / std::sqrt(person_area);

    double person_upper_third_y = person.y1 + person_height * 0.33;
    double vertical_penalty;
    if (face.cy < person_upper_third_y) {
        vertical_penalty = 0.0;
    } else if (face.cy < person.y1 + person_height * 0.5) {
        vertical_penalty = 0.5;
    } else {
        vertical_penalty = 2.0;
    }

    double expected_ratio = face.h / person_height;
    double size_penalty = (expected_ratio >= 0.1 && expected_ratio <= 0.4) ? 0.0 : 1.0;
    double confidence_cost = 1.0 - face.score;

    return normalized_distance * 2.0 + vertical_penalty * 1.5 + size_penalty * 1.0 + confidence_cost * 0.5;
}

std::vector<std::pair<int, int>> SixDrepNet::matchFacesToPersonsHungarian(
    const std::vector<FaceCandidate>& faces, const std::vector<PersonCandidate>& persons) {
    if (faces.empty() || persons.empty()) return {};

    int n_faces = static_cast<int>(faces.size());
    int n_persons = static_cast<int>(persons.size());
    Eigen::MatrixXd cost(n_faces, n_persons);
    for (int f = 0; f < n_faces; ++f) {
        for (int p = 0; p < n_persons; ++p) {
            cost(f, p) = calculateMatchingCost(faces[f], persons[p]);
        }
    }

    try {
        auto raw_matches = byte_tracker::matching::hungarianAssignment(cost);
        std::vector<std::pair<int, int>> matches;
        std::set<int> matched_face_indices;
        for (const auto& [f_idx, p_idx] : raw_matches) {
            if (cost(f_idx, p_idx) < kImpossibleMatchCost) {
                matches.push_back({f_idx, p_idx});
                matched_face_indices.insert(f_idx);
            }
        }
        for (int f_idx = 0; f_idx < n_faces; ++f_idx) {
            if (matched_face_indices.count(f_idx)) continue;
            double best_cost = kImpossibleMatchCost;
            int best_person_idx = -1;
            for (int p_idx = 0; p_idx < n_persons; ++p_idx) {
                double c = cost(f_idx, p_idx);
                if (c < best_cost) {
                    best_cost = c;
                    best_person_idx = p_idx;
                }
            }
            if (best_person_idx >= 0) {
                matches.push_back({f_idx, best_person_idx});
                if (verbose_mode_) {
                    RCLCPP_INFO(get_logger(), "Additional face %d matched to person %s (multiple faces in same box)",
                        f_idx, persons[best_person_idx].tracking_id.c_str());
                }
            }
        }
        return matches;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Hungarian algorithm failed: %s", e.what());
        return {};
    }
}

float SixDrepNet::getBestDepthEstimate(double face_cx, double face_cy, double face_width, double face_height,
                                       double person_depth) const {
    auto face_depth = getDepthInRegion(face_cx, face_cy, face_width, face_height);
    if (prioritize_face_depth_) {
        if (face_depth && *face_depth > 0.0f) return *face_depth;
        if (person_depth > 0.0) return static_cast<float>(person_depth);
        return 0.0f;
    }
    if (person_depth > 0.0) return static_cast<float>(person_depth);
    if (face_depth && *face_depth > 0.0f) return *face_depth;
    return 0.0f;
}

std::optional<std::array<double, 3>> SixDrepNet::estimateHeadPose(const cv::Mat& face_crop) {
    try {
        cv::Mat resized;
        cv::resize(face_crop, resized, cv::Size(224, 224));
        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        double mean_arr[3] = {mean_[0], mean_[1], mean_[2]};
        double std_arr[3] = {std_[0], std_[1], std_[2]};
        for (int c = 0; c < 3; ++c) {
            channels[c] = (channels[c] - mean_arr[c]) / std_arr[c];
        }

        std::vector<float> chw(3 * 224 * 224);
        size_t plane = 224 * 224;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(chw.data() + c * plane, channels[c].ptr<float>(), plane * sizeof(float));
        }

        std::array<int64_t, 4> shape = {1, 3, 224, 224};
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, chw.data(), chw.size(), shape.data(), shape.size());

        std::vector<const char*> in_ptrs;
        for (auto& n : sixdrepnet_input_names_) in_ptrs.push_back(n.c_str());
        std::vector<const char*> out_ptrs;
        for (auto& n : sixdrepnet_output_names_) out_ptrs.push_back(n.c_str());

        auto outputs = sixdrepnet_session_->Run(
            Ort::RunOptions{nullptr}, in_ptrs.data(), &input_tensor, 1, out_ptrs.data(), out_ptrs.size());
        const float* data = outputs[0].GetTensorData<float>();
        return std::array<double, 3>{data[0], data[1], data[2]};
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Head pose estimation failed: %s", e.what());
        return std::nullopt;
    }
}

void SixDrepNet::processImages() {
    bool depth_required = camera_type_ != "pepper";
    if (color_image_.empty() || (depth_required && depth_image_.empty())) return;

    cv::Mat frame = require_person_detection_
        ? processFrameWithPersonDetection(color_image_)
        : processFrameStandalone(color_image_);
    if (!frame.empty()) updateLatestFrame(frame);
}

cv::Mat SixDrepNet::processFrameStandalone(const cv::Mat& cv_image) {
    cv::Mat debug_image = cv_image.clone();
    int img_h = debug_image.rows, img_w = debug_image.cols;
    std::vector<FaceTrackingDatum> tracking_data;

    auto [face_boxes, face_scores] = yolo_model_->detect(cv_image);
    if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "Standalone mode: %zu faces detected", face_boxes.size());
    }

    std::vector<Eigen::Vector4d> valid_boxes;
    std::vector<float> valid_scores;
    for (size_t i = 0; i < face_boxes.size(); ++i) {
        double fx1 = face_boxes[i].x, fy1 = face_boxes[i].y;
        double fx2 = fx1 + face_boxes[i].width, fy2 = fy1 + face_boxes[i].height;
        if ((fx2 - fx1) < 20 || (fy2 - fy1) < 20) continue;
        if (!(fx1 >= 0 && fx1 < fx2 && fx2 <= img_w && fy1 >= 0 && fy1 < fy2 && fy2 <= img_h)) continue;
        valid_boxes.push_back({fx1, fy1, fx2, fy2});
        valid_scores.push_back(face_scores[i]);
    }

    byte_tracker::Detections tracked;
    if (!valid_boxes.empty()) {
        byte_tracker::Detections det;
        det.xyxy = valid_boxes;
        det.confidence = valid_scores;
        det.class_id.assign(valid_boxes.size(), 0);
        tracked = face_tracker_.updateWithDetections(det);
    }

    std::set<std::string> active_face_ids;
    for (size_t i = 0; i < tracked.xyxy.size(); ++i) {
        int track_id = tracked.tracker_id[i];
        int fx1 = static_cast<int>(tracked.xyxy[i][0]);
        int fy1 = static_cast<int>(tracked.xyxy[i][1]);
        int fx2 = static_cast<int>(tracked.xyxy[i][2]);
        int fy2 = static_cast<int>(tracked.xyxy[i][3]);
        double face_cx = (fx1 + fx2) / 2.0;
        double face_cy = (fy1 + fy2) / 2.0;
        double face_width = fx2 - fx1, face_height = fy2 - fy1;
        std::string face_id = "face_" + std::to_string(track_id);
        active_face_ids.insert(face_id);

        if (face_colors_.find(face_id) == face_colors_.end()) face_colors_[face_id] = generateDarkColor();
        cv::Scalar face_color = face_colors_[face_id];

        int cfx1 = std::max(0, fx1), cfy1 = std::max(0, fy1);
        int cfx2 = std::min(cv_image.cols, fx2), cfy2 = std::min(cv_image.rows, fy2);
        if (cfx2 <= cfx1 || cfy2 <= cfy1) continue;
        cv::Mat face_image = cv_image(cv::Range(cfy1, cfy2), cv::Range(cfx1, cfx2));

        auto pose = estimateHeadPose(face_image);
        if (!pose) continue;
        double yaw_deg = (*pose)[0], pitch_deg = (*pose)[1], roll_deg = (*pose)[2];

        drawAxis(debug_image, yaw_deg, pitch_deg, roll_deg, face_cx, face_cy, 50.0);

        float cz = getDepthInRegion(face_cx, face_cy, face_width, face_height).value_or(0.0f);
        bool mutual_gaze = std::abs(yaw_deg) < sixdrep_angle_ && std::abs(pitch_deg) < sixdrep_angle_;

        FaceTrackingDatum datum;
        datum.face_id = face_id;
        datum.centroid.x = face_cx;
        datum.centroid.y = face_cy;
        datum.centroid.z = cz;
        datum.width = static_cast<float>(face_width);
        datum.height = static_cast<float>(face_height);
        datum.mutual_gaze = mutual_gaze;
        tracking_data.push_back(datum);

        cv::rectangle(debug_image, cv::Point(fx1, fy1), cv::Point(fx2, fy2), face_color, 2);
        cv::putText(debug_image, mutual_gaze ? "Engaged" : "Not Engaged", cv::Point(fx1 + 10, fy1 - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2);
        cv::putText(debug_image, "ID: " + face_id, cv::Point(fx1 + 10, fy1 - 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            face_color, 2);
        if (cz > 0.0f) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "Depth: %.2fm", cz);
            cv::putText(debug_image, buf, cv::Point(fx1 + 10, fy2 + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2);
        }
    }

    std::vector<std::string> stale;
    for (const auto& [id, _] : face_colors_) {
        if (!active_face_ids.count(id)) stale.push_back(id);
    }
    for (const auto& id : stale) face_colors_.erase(id);

    publishFaceDetection(tracking_data);
    return debug_image;
}

cv::Mat SixDrepNet::processFrameWithPersonDetection(const cv::Mat& cv_image) {
    cv::Mat debug_image = cv_image.clone();
    int img_h = debug_image.rows, img_w = debug_image.cols;
    std::vector<FaceTrackingDatum> tracking_data;

    PersonSnapshot snapshot;
    double detection_age = 0.0;
    {
        std::lock_guard<std::mutex> lock(person_detections_mutex_);
        if (!latest_person_detections_ || !latest_person_detections_timestamp_) {
            if (verbose_mode_) RCLCPP_WARN(get_logger(), "No person detection data available");
            publishFaceDetection({});
            return debug_image;
        }
        detection_age = (get_clock()->now() - *latest_person_detections_timestamp_).seconds();
        if (detection_age > person_detection_timeout_) {
            if (verbose_mode_) {
                RCLCPP_WARN(get_logger(), "Person detection data is stale (%.2fs old, timeout=%.2fs)",
                    detection_age, person_detection_timeout_);
            }
            publishFaceDetection({});
            return debug_image;
        }
        snapshot = *latest_person_detections_;
    }

    size_t n_detections = snapshot.person_label_id.size();
    if (verbose_mode_ && n_detections > 0) {
        RCLCPP_INFO(get_logger(), "Person detection: %zu persons (age: %.2fs)", n_detections, detection_age);
    }

    if (n_detections == 0) {
        publishFaceDetection({});
        face_colors_.clear();
        return debug_image;
    }

    if (snapshot.class_names.size() != n_detections || snapshot.centroids.size() != n_detections ||
        snapshot.width.size() != n_detections || snapshot.height.size() != n_detections) {
        RCLCPP_ERROR(get_logger(), "Inconsistent person detection array lengths");
        publishFaceDetection({});
        return debug_image;
    }

    std::vector<PersonCandidate> persons;
    std::set<std::string> active_person_ids;
    for (size_t i = 0; i < n_detections; ++i) {
        if (snapshot.class_names[i] != "person") continue;
        const auto& centroid = snapshot.centroids[i];
        double width = snapshot.width[i], height = snapshot.height[i];

        int x1 = std::max(0, static_cast<int>(centroid.x - width / 2.0));
        int y1 = std::max(0, static_cast<int>(centroid.y - height / 2.0));
        int x2 = std::min(img_w, static_cast<int>(centroid.x + width / 2.0));
        int y2 = std::min(img_h, static_cast<int>(centroid.y + height / 2.0));

        int box_width = x2 - x1, box_height = y2 - y1;
        int box_area = box_width * box_height;
        if (x2 <= x1 || y2 <= y1 || box_area < 100) continue;
        if (x1 >= img_w || y1 >= img_h || x2 <= 0 || y2 <= 0) continue;

        std::string tracking_id = snapshot.person_label_id[i];
        active_person_ids.insert(tracking_id);

        if (verbose_mode_) {
            RCLCPP_INFO(get_logger(), "Person tracking_id=%s: box=(%d,%d,%d,%d), depth=%.2fm",
                tracking_id.c_str(), x1, y1, x2, y2, centroid.z);
        }

        PersonCandidate person;
        person.tracking_id = tracking_id;
        person.x1 = x1; person.y1 = y1; person.x2 = x2; person.y2 = y2;
        person.depth = centroid.z;
        persons.push_back(person);
    }

    std::vector<std::string> stale_ids;
    for (const auto& [id, _] : face_colors_) {
        if (!active_person_ids.count(id)) stale_ids.push_back(id);
    }
    for (const auto& id : stale_ids) face_colors_.erase(id);

    if (persons.empty()) {
        if (verbose_mode_) RCLCPP_INFO(get_logger(), "No valid persons detected (total objects: %zu)", n_detections);
        publishFaceDetection({});
        return debug_image;
    }

    auto [face_boxes, face_scores] = yolo_model_->detect(cv_image);
    if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "Face detection: %zu raw detections", face_boxes.size());
    }

    std::vector<FaceCandidate> faces;
    for (size_t idx = 0; idx < face_boxes.size(); ++idx) {
        double fx1 = face_boxes[idx].x, fy1 = face_boxes[idx].y;
        double fx2 = fx1 + face_boxes[idx].width, fy2 = fy1 + face_boxes[idx].height;
        double face_cx = std::floor((fx1 + fx2) / 2.0);
        double face_cy = std::floor((fy1 + fy2) / 2.0);
        double face_width = fx2 - fx1, face_height = fy2 - fy1;

        if (face_width < 20 || face_height < 20) continue;
        if (!(fx1 >= 0 && fx1 < fx2 && fx2 <= img_w && fy1 >= 0 && fy1 < fy2 && fy2 <= img_h)) continue;

        FaceCandidate face;
        face.x1 = fx1; face.y1 = fy1; face.x2 = fx2; face.y2 = fy2;
        face.cx = face_cx; face.cy = face_cy;
        face.w = face_width; face.h = face_height;
        face.score = face_scores[idx];
        faces.push_back(face);
    }

    if (faces.empty()) {
        if (verbose_mode_) RCLCPP_INFO(get_logger(), "No valid faces detected (persons: %zu)", persons.size());
        publishFaceDetection({});
        if (verbose_mode_) {
            for (const auto& person : persons) {
                cv::rectangle(debug_image, cv::Point(static_cast<int>(person.x1), static_cast<int>(person.y1)),
                    cv::Point(static_cast<int>(person.x2), static_cast<int>(person.y2)), cv::Scalar(128, 128, 128), 1);
                cv::putText(debug_image, "P:" + person.tracking_id,
                    cv::Point(static_cast<int>(person.x1) + 5, static_cast<int>(person.y1) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(128, 128, 128), 1);
            }
        }
        return debug_image;
    }

    auto matches = matchFacesToPersonsHungarian(faces, persons);
    if (verbose_mode_) {
        RCLCPP_INFO(get_logger(), "Hungarian matching: Persons=%zu, Faces=%zu, Matches=%zu",
            persons.size(), faces.size(), matches.size());
    }

    if (matches.empty()) {
        publishFaceDetection({});
        if (verbose_mode_) {
            RCLCPP_WARN(get_logger(), "Faces detected (%zu) but none matched to persons (%zu)", faces.size(),
                persons.size());
            for (const auto& face : faces) {
                cv::rectangle(debug_image, cv::Point(static_cast<int>(face.x1), static_cast<int>(face.y1)),
                    cv::Point(static_cast<int>(face.x2), static_cast<int>(face.y2)), cv::Scalar(0, 0, 255), 1);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "Unmatched (score=%.2f)", face.score);
                cv::putText(debug_image, buf, cv::Point(static_cast<int>(face.x1), static_cast<int>(face.y1) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);
            }
            for (const auto& person : persons) {
                cv::rectangle(debug_image, cv::Point(static_cast<int>(person.x1), static_cast<int>(person.y1)),
                    cv::Point(static_cast<int>(person.x2), static_cast<int>(person.y2)), cv::Scalar(255, 0, 0), 2);
                cv::putText(debug_image, "Person:" + person.tracking_id,
                    cv::Point(static_cast<int>(person.x1) + 5, static_cast<int>(person.y1) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }
        return debug_image;
    }

    for (const auto& [face_idx, person_idx] : matches) {
        const auto& face = faces[face_idx];
        auto& person = persons[person_idx];

        int fx1 = static_cast<int>(face.x1), fy1 = static_cast<int>(face.y1);
        int fx2 = static_cast<int>(face.x2), fy2 = static_cast<int>(face.y2);
        double face_cx = face.cx, face_cy = face.cy;
        double face_width = face.w, face_height = face.h;

        person.assigned_faces++;
        std::string face_id = person.assigned_faces == 1
            ? person.tracking_id : person.tracking_id + "_f" + std::to_string(person.assigned_faces);

        const std::string& base_id = person.tracking_id;
        if (face_colors_.find(base_id) == face_colors_.end()) face_colors_[base_id] = generateDarkColor();
        cv::Scalar face_color = face_colors_[base_id];

        int cfx1 = std::max(0, fx1), cfy1 = std::max(0, fy1);
        int cfx2 = std::min(cv_image.cols, fx2), cfy2 = std::min(cv_image.rows, fy2);
        if (cfx2 <= cfx1 || cfy2 <= cfy1) continue;
        cv::Mat face_image = cv_image(cv::Range(cfy1, cfy2), cv::Range(cfx1, cfx2));

        auto pose = estimateHeadPose(face_image);
        if (!pose) continue;
        double yaw_deg = (*pose)[0], pitch_deg = (*pose)[1], roll_deg = (*pose)[2];

        drawAxis(debug_image, yaw_deg, pitch_deg, roll_deg, face_cx, face_cy, 50.0);

        float cz = getBestDepthEstimate(face_cx, face_cy, face_width, face_height, person.depth);
        bool mutual_gaze = std::abs(yaw_deg) < sixdrep_angle_ && std::abs(pitch_deg) < sixdrep_angle_;

        FaceTrackingDatum datum;
        datum.face_id = face_id;
        datum.centroid.x = face_cx;
        datum.centroid.y = face_cy;
        datum.centroid.z = cz;
        datum.width = static_cast<float>(face_width);
        datum.height = static_cast<float>(face_height);
        datum.mutual_gaze = mutual_gaze;
        tracking_data.push_back(datum);

        cv::rectangle(debug_image, cv::Point(fx1, fy1), cv::Point(fx2, fy2), face_color, 2);
        cv::putText(debug_image, mutual_gaze ? "Engaged" : "Not Engaged", cv::Point(fx1 + 10, fy1 - 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2);
        cv::putText(debug_image, "ID: " + face_id, cv::Point(fx1 + 10, fy1 - 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
            face_color, 2);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Depth: %.2fm", cz);
        cv::putText(debug_image, buf, cv::Point(fx1 + 10, fy2 + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2);
    }

    if (verbose_mode_) {
        for (const auto& person : persons) {
            cv::Scalar color = person.assigned_faces > 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128);
            cv::rectangle(debug_image, cv::Point(static_cast<int>(person.x1), static_cast<int>(person.y1)),
                cv::Point(static_cast<int>(person.x2), static_cast<int>(person.y2)), color, 1);
            std::string label = "P:" + person.tracking_id;
            if (person.assigned_faces > 0) label += " (" + std::to_string(person.assigned_faces) + " faces)";
            cv::putText(debug_image, label,
                cv::Point(static_cast<int>(person.x1) + 5, static_cast<int>(person.y1) - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
    }

    publishFaceDetection(tracking_data);
    return debug_image;
}
