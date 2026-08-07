/* age_gender_detection_implementation.cpp
 *
 * Implements MiVOLOONNX (the MiVOLO age/gender ONNX wrapper) and
 * AgeGenderDetectionNode (caching, mutual-gaze-triggered estimation worker,
 * temporal smoothing, JSON publishing). See age_gender_detection_interface.h
 * for the full subscriber/publisher/parameter reference and the lifecycle
 * state-machine diagram.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 29, 2026
 * Version: v1.0
 */

#include "face_detection/age_gender_detection_interface.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cv_bridge/cv_bridge.h>
#include <dec_common/param_loader.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

namespace {

std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

}  // namespace

AgeGenderDetectionConfig loadAgeGenderConfiguration(rclcpp_lifecycle::LifecycleNode* node) {
    AgeGenderDetectionConfig config;

    std::string default_model_path;
    try {
        default_model_path =
            ament_index_cpp::get_package_share_directory("face_detection") + "/models/face_detection_mivolo_agegender.onnx";
    } catch (const std::exception&) {
        default_model_path = "";
    }

    config.mivolo_model_path = dec_common::declareAndGetParameter(node, "mivolo_model_path", default_model_path);
    config.device = dec_common::declareAndGetParameter(node, "device", config.device);
    config.face_only = dec_common::declareAndGetParameter(node, "face_only", config.face_only);
    config.face_topic = dec_common::declareAndGetParameter(node, "face_topic", config.face_topic);
    config.person_topic = dec_common::declareAndGetParameter(node, "person_topic", config.person_topic);
    config.image_topic = dec_common::declareAndGetParameter(node, "image_topic", config.image_topic);
    config.output_topic = dec_common::declareAndGetParameter(node, "output_topic", config.output_topic);
    config.max_cache_age_sec =
        dec_common::declareAndGetParameter(node, "max_cache_age_sec", config.max_cache_age_sec);
    config.min_estimate_interval_sec =
        dec_common::declareAndGetParameter(node, "min_estimate_interval_sec", config.min_estimate_interval_sec);
    config.re_estimate_interval_sec =
        dec_common::declareAndGetParameter(node, "re_estimate_interval_sec", config.re_estimate_interval_sec);
    config.person_class_name =
        dec_common::declareAndGetParameter(node, "person_class_name", config.person_class_name);
    config.max_depth_m = dec_common::declareAndGetParameter(node, "max_depth_m", config.max_depth_m);
    return config;
}

// ── AgeGenderBoundingBox ─────────────────────────────────────────────────────

AgeGenderBoundingBox AgeGenderBoundingBox::fromCentroid(const geometry_msgs::msg::Point& centroid, float width,
                                                        float height, bool mutual_gaze) {
    AgeGenderBoundingBox bbox;
    bbox.x1 = centroid.x - width / 2.0;
    bbox.y1 = centroid.y - height / 2.0;
    bbox.x2 = centroid.x + width / 2.0;
    bbox.y2 = centroid.y + height / 2.0;
    bbox.depth = centroid.z;
    bbox.mutual_gaze = mutual_gaze;
    bbox.timestamp = std::chrono::steady_clock::now();
    return bbox;
}

// ── AgeGenderPersonProfile ───────────────────────────────────────────────────

void AgeGenderPersonProfile::addEstimate(double age_val, const std::string& gender_val, double gender_conf) {
    if (age_history.size() >= kHistoryMaxLen) age_history.pop_front();
    age_history.push_back(age_val);
    if (gender_history.size() >= kHistoryMaxLen) gender_history.pop_front();
    gender_history.emplace_back(gender_val, gender_conf);

    estimation_count++;
    last_updated = std::chrono::steady_clock::now();
    has_valid_estimate = true;

    std::vector<double> sorted_ages(age_history.begin(), age_history.end());
    std::sort(sorted_ages.begin(), sorted_ages.end());
    size_t n = sorted_ages.size();
    age = (n % 2 == 1) ? sorted_ages[n / 2] : (sorted_ages[n / 2 - 1] + sorted_ages[n / 2]) / 2.0;

    double male_score = 0.0, female_score = 0.0;
    for (const auto& [g, conf] : gender_history) {
        if (g == "male") {
            male_score += conf;
        } else if (g == "female") {
            female_score += conf;
        }
    }
    double total = male_score + female_score;
    if (total > 0.0) {
        if (male_score > female_score) {
            gender = "male";
            gender_confidence = male_score / total;
        } else {
            gender = "female";
            gender_confidence = female_score / total;
        }
    }
}

std::string AgeGenderPersonProfile::toJson() const {
    char num_buf[64];

    std::string age_json = "null";
    if (age) {
        std::snprintf(num_buf, sizeof(num_buf), "%.1f", *age);
        age_json = num_buf;
    }

    std::string gender_json = gender ? ("\"" + *gender + "\"") : "null";

    std::string conf_json = "null";
    if (gender_confidence) {
        std::snprintf(num_buf, sizeof(num_buf), "%.3f", *gender_confidence);
        conf_json = num_buf;
    }

    std::string bbox_json = "null";
    if (last_person_bbox) {
        char bbox_buf[256];
        std::snprintf(bbox_buf, sizeof(bbox_buf), "{\"x1\":%.2f,\"y1\":%.2f,\"x2\":%.2f,\"y2\":%.2f}",
            last_person_bbox->x1, last_person_bbox->y1, last_person_bbox->x2, last_person_bbox->y2);
        bbox_json = bbox_buf;
    }

    return "{\"label_id\":\"" + jsonEscape(label_id) + "\"," + "\"age\":" + age_json + "," +
           "\"gender\":" + gender_json + "," + "\"gender_confidence\":" + conf_json + "," +
           "\"estimation_count\":" + std::to_string(estimation_count) + "," + "\"person_bbox\":" + bbox_json + "}";
}

// ── MiVOLOONNX ───────────────────────────────────────────────────────────────

MiVOLOONNX::MiVOLOONNX(const std::string& model_path, bool use_cuda) {
    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "age_gender_detection_mivolo");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(static_cast<int>(std::thread::hardware_concurrency()));
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (use_cuda) {
        try {
            OrtCUDAProviderOptions cuda_options{};
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        } catch (const std::exception&) {
            // Falls back to CPU; the caller logs provider status separately.
        }
    }

    session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        input_names_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        output_names_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
    }

    // Warmup run (6-channel face+body input) to load model weights into memory.
    std::vector<float> dummy(static_cast<size_t>(6 * kInputSize * kInputSize), 0.0f);
    std::array<int64_t, 4> shape = {1, 6, kInputSize, kInputSize};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(mem_info, dummy.data(), dummy.size(), shape.data(), shape.size());
    std::vector<const char*> in_ptrs;
    for (auto& n : input_names_) in_ptrs.push_back(n.c_str());
    std::vector<const char*> out_ptrs;
    for (auto& n : output_names_) out_ptrs.push_back(n.c_str());
    session_->Run(Ort::RunOptions{nullptr}, in_ptrs.data(), &input_tensor, 1, out_ptrs.data(), out_ptrs.size());
}

void MiVOLOONNX::writeCropPlanes(const cv::Mat& crop, float* dst) const {
    size_t plane = static_cast<size_t>(kInputSize) * kInputSize;
    if (crop.empty()) {
        std::fill(dst, dst + 3 * plane, 0.0f);
        return;
    }

    // class_letterbox: resize maintaining aspect ratio, pad with black to
    // kInputSize x kInputSize (see class_letterbox in age_gender_detection.py).
    cv::Mat letterboxed;
    if (crop.rows == kInputSize && crop.cols == kInputSize) {
        letterboxed = crop;
    } else {
        double r = std::min(static_cast<double>(kInputSize) / crop.rows, static_cast<double>(kInputSize) / crop.cols);
        int new_unpad_w = static_cast<int>(std::lround(crop.cols * r));
        int new_unpad_h = static_cast<int>(std::lround(crop.rows * r));
        double dw = (kInputSize - new_unpad_w) / 2.0;
        double dh = (kInputSize - new_unpad_h) / 2.0;

        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

        int top = static_cast<int>(std::lround(dh - 0.1));
        int bottom = static_cast<int>(std::lround(dh + 0.1));
        int left = static_cast<int>(std::lround(dw - 0.1));
        int right = static_cast<int>(std::lround(dw + 0.1));
        cv::copyMakeBorder(resized, letterboxed, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    cv::Mat rgb;
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - mean_[c]) / std_[c];
        std::memcpy(dst + c * plane, channels[c].ptr<float>(), plane * sizeof(float));
    }
}

std::optional<MiVOLOONNX::Estimate> MiVOLOONNX::predict(const cv::Mat& face_crop, const cv::Mat& body_crop) {
    if (face_crop.empty() || body_crop.empty()) return std::nullopt;

    size_t plane = static_cast<size_t>(kInputSize) * kInputSize;
    std::vector<float> chw(6 * plane);
    writeCropPlanes(face_crop, chw.data());
    writeCropPlanes(body_crop, chw.data() + 3 * plane);

    std::array<int64_t, 4> shape = {1, 6, kInputSize, kInputSize};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(mem_info, chw.data(), chw.size(), shape.data(), shape.size());

    std::vector<const char*> in_ptrs;
    for (auto& n : input_names_) in_ptrs.push_back(n.c_str());
    std::vector<const char*> out_ptrs;
    for (auto& n : output_names_) out_ptrs.push_back(n.c_str());

    auto outputs =
        session_->Run(Ort::RunOptions{nullptr}, in_ptrs.data(), &input_tensor, 1, out_ptrs.data(), out_ptrs.size());
    const float* data = outputs[0].GetTensorData<float>();

    // Output layout: [gender_male_logit, gender_female_logit, age_normalized].
    double male_logit = data[0], female_logit = data[1];
    double max_logit = std::max(male_logit, female_logit);
    double m = std::exp(male_logit - max_logit);
    double f = std::exp(female_logit - max_logit);
    double male_prob = m / (m + f);
    double female_prob = f / (m + f);

    Estimate estimate;
    if (male_prob >= female_prob) {
        estimate.gender = "male";
        estimate.gender_confidence = male_prob;
    } else {
        estimate.gender = "female";
        estimate.gender_confidence = female_prob;
    }

    double age = static_cast<double>(data[2]) * (max_age_ - min_age_) + avg_age_;
    estimate.age = std::round(age * 100.0) / 100.0;

    return estimate;
}

// ── AgeGenderDetectionNode ───────────────────────────────────────────────────

AgeGenderDetectionNode::AgeGenderDetectionNode(const std::string& node_name) : rclcpp_lifecycle::LifecycleNode(node_name) {}

AgeGenderDetectionNode::~AgeGenderDetectionNode() { cleanup(); }

AgeGenderDetectionNode::CallbackReturn AgeGenderDetectionNode::on_configure(const rclcpp_lifecycle::State&) {
    config_ = loadAgeGenderConfiguration(this);

    if (config_.mivolo_model_path.empty()) {
        RCLCPP_ERROR(get_logger(), "%s: failed to locate MiVOLO model file", get_name());
        return CallbackReturn::FAILURE;
    }

    RCLCPP_INFO(get_logger(), "%s: loading MiVOLO ONNX model from %s (device=%s)", get_name(),
        config_.mivolo_model_path.c_str(), config_.device.c_str());
    try {
        mivolo_ = std::make_unique<MiVOLOONNX>(config_.mivolo_model_path, config_.device == "cuda");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "%s: MiVOLOONNX init failed: %s", get_name(), e.what());
        return CallbackReturn::FAILURE;
    }

    result_pub_ = create_publisher<std_msgs::msg::String>(config_.output_topic, 10);

    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        recent_persons_.clear();
        recent_faces_.clear();
        person_profiles_.clear();
        known_label_ids_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        latest_image_ = cv::Mat();
        has_image_ = false;
    }

    RCLCPP_INFO(get_logger(),
        "%s: configured\n"
        "  Person topic: %s\n"
        "  Face topic: %s\n"
        "  Image topic: %s\n"
        "  Output topic: %s\n"
        "  Max depth: %.2fm",
        get_name(), config_.person_topic.c_str(), config_.face_topic.c_str(), config_.image_topic.c_str(),
        config_.output_topic.c_str(), config_.max_depth_m);

    return CallbackReturn::SUCCESS;
}

AgeGenderDetectionNode::CallbackReturn AgeGenderDetectionNode::on_activate(const rclcpp_lifecycle::State& state) {
    LifecycleNode::on_activate(state);

    rclcpp::QoS sensor_qos(rclcpp::KeepLast(1));
    sensor_qos.best_effort();

    person_sub_ = create_subscription<dec_interfaces::msg::PersonDetection>(
        config_.person_topic, sensor_qos, std::bind(&AgeGenderDetectionNode::personCallback, this, std::placeholders::_1));
    face_sub_ = create_subscription<dec_interfaces::msg::FaceDetection>(
        config_.face_topic, sensor_qos, std::bind(&AgeGenderDetectionNode::faceCallback, this, std::placeholders::_1));
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        config_.image_topic, sensor_qos, std::bind(&AgeGenderDetectionNode::imageCallback, this, std::placeholders::_1));

    running_ = true;
    worker_thread_ = std::thread(&AgeGenderDetectionNode::estimationWorker, this);

    cleanup_timer_ =
        create_wall_timer(std::chrono::seconds(1), std::bind(&AgeGenderDetectionNode::cleanupStaleData, this));
    debug_timer_ = create_wall_timer(std::chrono::seconds(5), std::bind(&AgeGenderDetectionNode::debugStatus, this));

    RCLCPP_INFO(get_logger(), "%s: activated", get_name());
    return CallbackReturn::SUCCESS;
}

AgeGenderDetectionNode::CallbackReturn AgeGenderDetectionNode::on_deactivate(const rclcpp_lifecycle::State& state) {
    if (cleanup_timer_) {
        cleanup_timer_->cancel();
        cleanup_timer_.reset();
    }
    if (debug_timer_) {
        debug_timer_->cancel();
        debug_timer_.reset();
    }

    person_sub_.reset();
    face_sub_.reset();
    image_sub_.reset();

    running_ = false;
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();

    LifecycleNode::on_deactivate(state);
    return CallbackReturn::SUCCESS;
}

AgeGenderDetectionNode::CallbackReturn AgeGenderDetectionNode::on_cleanup(const rclcpp_lifecycle::State&) {
    result_pub_.reset();
    mivolo_.reset();

    std::lock_guard<std::mutex> lock(data_mutex_);
    recent_persons_.clear();
    recent_faces_.clear();
    person_profiles_.clear();
    known_label_ids_.clear();
    return CallbackReturn::SUCCESS;
}

AgeGenderDetectionNode::CallbackReturn AgeGenderDetectionNode::on_shutdown(const rclcpp_lifecycle::State&) {
    RCLCPP_INFO(get_logger(), "%s shutting down", get_name());
    return CallbackReturn::SUCCESS;
}

void AgeGenderDetectionNode::cleanup() {
    running_ = false;
    queue_cv_.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
}

void AgeGenderDetectionNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
        cv::Mat image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        std::lock_guard<std::mutex> lock(image_mutex_);
        latest_image_ = image;
        image_timestamp_ = std::chrono::steady_clock::now();
        has_image_ = true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "%s: cv_bridge conversion failed: %s", get_name(), e.what());
    }
}

void AgeGenderDetectionNode::personCallback(const dec_interfaces::msg::PersonDetection::ConstSharedPtr& msg) {
    size_t n = msg->person_label_id.size();
    if (msg->class_names.size() != n || msg->centroids.size() != n || msg->width.size() != n ||
        msg->height.size() != n || msg->confidences.size() != n) {
        RCLCPP_WARN(get_logger(), "%s: person_callback: inconsistent array lengths, skipping message", get_name());
        return;
    }

    std::vector<std::string> ids_to_estimate;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        for (size_t i = 0; i < n; ++i) {
            if (msg->class_names[i] != config_.person_class_name) continue;

            const std::string& label_id = msg->person_label_id[i];
            AgeGenderBoundingBox person_bbox =
                AgeGenderBoundingBox::fromCentroid(msg->centroids[i], msg->width[i], msg->height[i]);
            recent_persons_[label_id] = person_bbox;

            bool is_new = known_label_ids_.find(label_id) == known_label_ids_.end();
            if (is_new) {
                known_label_ids_.insert(label_id);
                person_profiles_[label_id].label_id = label_id;
                RCLCPP_INFO(get_logger(), "%s: new person detected: %s", get_name(), label_id.c_str());
            }
            person_profiles_[label_id].last_person_bbox = person_bbox;

            if (is_new) {
                auto face_it = recent_faces_.find(label_id);
                if (face_it != recent_faces_.end() && face_it->second.mutual_gaze) {
                    bool depth_ok = face_it->second.depth > 0.0 && face_it->second.depth < config_.max_depth_m;
                    if (depth_ok) {
                        RCLCPP_INFO(get_logger(), "%s: new person %s has mutual gaze at %.2fm, triggering estimation",
                            get_name(), label_id.c_str(), face_it->second.depth);
                        ids_to_estimate.push_back(label_id);
                    }
                }
            }
        }
    }

    for (const auto& label_id : ids_to_estimate) scheduleEstimation(label_id);
}

void AgeGenderDetectionNode::faceCallback(const dec_interfaces::msg::FaceDetection::ConstSharedPtr& msg) {
    size_t n = msg->face_label_id.size();
    if (msg->centroids.size() != n || msg->width.size() != n || msg->height.size() != n ||
        msg->mutual_gaze.size() != n) {
        RCLCPP_WARN(get_logger(), "%s: face_callback: inconsistent array lengths, skipping message", get_name());
        return;
    }

    std::vector<std::string> ids_to_estimate;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        for (size_t i = 0; i < n; ++i) {
            const std::string& label_id = msg->face_label_id[i];
            bool mutual_gaze = msg->mutual_gaze[i];
            double depth = msg->centroids[i].z;

            AgeGenderBoundingBox face_bbox =
                AgeGenderBoundingBox::fromCentroid(msg->centroids[i], msg->width[i], msg->height[i], mutual_gaze);
            recent_faces_[label_id] = face_bbox;

            bool depth_ok = depth > 0.0 && depth < config_.max_depth_m;
            if (mutual_gaze && depth_ok && known_label_ids_.count(label_id)) {
                auto profile_it = person_profiles_.find(label_id);
                if (profile_it != person_profiles_.end() &&
                    shouldReEstimate(profile_it->second, std::chrono::steady_clock::now())) {
                    RCLCPP_INFO(get_logger(), "%s: mutual gaze detected for %s at %.2fm (< %.2fm), triggering estimation",
                        get_name(), label_id.c_str(), depth, config_.max_depth_m);
                    ids_to_estimate.push_back(label_id);
                }
            }
        }
    }

    for (const auto& label_id : ids_to_estimate) scheduleEstimation(label_id);
}

bool AgeGenderDetectionNode::shouldReEstimate(const AgeGenderPersonProfile& profile,
                                              std::chrono::steady_clock::time_point now) const {
    if (!profile.has_valid_estimate) return true;
    double elapsed = std::chrono::duration<double>(now - profile.last_updated).count();
    return elapsed > config_.re_estimate_interval_sec;
}

void AgeGenderDetectionNode::scheduleEstimation(const std::string& label_id) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (pending_estimates_.count(label_id)) return;
        pending_estimates_.insert(label_id);
        estimation_queue_.push(label_id);
    }
    queue_cv_.notify_one();
}

void AgeGenderDetectionNode::estimationWorker() {
    while (true) {
        std::string label_id;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !estimation_queue_.empty() || !running_; });
            if (!running_ && estimation_queue_.empty()) return;
            label_id = estimation_queue_.front();
            estimation_queue_.pop();
            pending_estimates_.erase(label_id);
        }
        try {
            estimateForPerson(label_id);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "%s: estimation worker error: %s", get_name(), e.what());
        }
    }
}

void AgeGenderDetectionNode::estimateForPerson(const std::string& label_id) {
    AgeGenderBoundingBox face_bbox;
    std::optional<AgeGenderBoundingBox> person_bbox;

    // Snapshot bboxes together to ensure they are temporally aligned.
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        auto profile_it = person_profiles_.find(label_id);
        if (profile_it == person_profiles_.end()) return;
        const auto& profile = profile_it->second;

        if (profile.estimation_count > 0) {
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - profile.last_updated).count();
            if (elapsed < config_.min_estimate_interval_sec) return;
        }

        auto face_it = recent_faces_.find(label_id);
        if (face_it == recent_faces_.end()) {
            RCLCPP_DEBUG(get_logger(), "%s: no face bbox for %s", get_name(), label_id.c_str());
            return;
        }
        face_bbox = face_it->second;

        auto person_it = recent_persons_.find(label_id);
        if (person_it == recent_persons_.end() && !config_.face_only) {
            RCLCPP_DEBUG(get_logger(), "%s: no person bbox for %s", get_name(), label_id.c_str());
            return;
        }
        if (person_it != recent_persons_.end()) person_bbox = person_it->second;
    }

    // Get image and verify temporal alignment with bboxes.
    cv::Mat image;
    std::chrono::steady_clock::time_point image_ts;
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        if (!has_image_) {
            RCLCPP_WARN(get_logger(), "%s: no image available", get_name());
            return;
        }
        image = latest_image_.clone();
        image_ts = image_timestamp_;
    }

    double gap = std::abs(std::chrono::duration<double>(image_ts - face_bbox.timestamp).count());
    if (gap > config_.max_cache_age_sec) {
        RCLCPP_WARN(get_logger(), "%s: skipping %s: image/bbox time gap %.2fs exceeds %.2fs", get_name(),
            label_id.c_str(), gap, config_.max_cache_age_sec);
        return;
    }

    auto face_crop = extractCrop(image, face_bbox);
    if (!face_crop) {
        RCLCPP_WARN(get_logger(), "%s: invalid face crop for %s", get_name(), label_id.c_str());
        return;
    }

    cv::Mat person_crop;
    if (!config_.face_only && person_bbox) {
        auto crop = extractCrop(image, *person_bbox);
        if (!crop) {
            RCLCPP_WARN(get_logger(), "%s: invalid person crop for %s", get_name(), label_id.c_str());
            return;
        }
        person_crop = *crop;
    }

    try {
        RCLCPP_INFO(get_logger(), "%s: running inference for %s", get_name(), label_id.c_str());
        auto result = mivolo_->predict(*face_crop, person_crop);

        if (result) {
            std::string json_result;
            {
                std::lock_guard<std::mutex> lock(data_mutex_);
                auto profile_it = person_profiles_.find(label_id);
                if (profile_it == person_profiles_.end()) return;
                profile_it->second.addEstimate(result->age, result->gender, result->gender_confidence);
                json_result = profile_it->second.toJson();
            }

            RCLCPP_INFO(get_logger(), "%s: [%s] age=%.1f (raw=%.1f), gender=%s (%.1f%%)", get_name(),
                label_id.c_str(), result->age, result->age, result->gender.c_str(), result->gender_confidence * 100.0);
            publishResult(json_result);
        } else {
            RCLCPP_WARN(get_logger(), "%s: inference returned nullopt for %s", get_name(), label_id.c_str());
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "%s: estimation failed for %s: %s", get_name(), label_id.c_str(), e.what());
    }
}

std::optional<cv::Mat> AgeGenderDetectionNode::extractCrop(const cv::Mat& image, const AgeGenderBoundingBox& bbox) const {
    int h = image.rows, w = image.cols;
    int x1 = std::max(0, static_cast<int>(bbox.x1));
    int y1 = std::max(0, static_cast<int>(bbox.y1));
    int x2 = std::min(w, static_cast<int>(bbox.x2));
    int y2 = std::min(h, static_cast<int>(bbox.y2));

    if (x2 <= x1 || y2 <= y1) return std::nullopt;
    return image(cv::Range(y1, y2), cv::Range(x1, x2)).clone();
}

void AgeGenderDetectionNode::publishResult(const std::string& json_result) {
    std_msgs::msg::String msg;
    msg.data = json_result;
    result_pub_->publish(msg);
}

void AgeGenderDetectionNode::cleanupStaleData() {
    auto now = std::chrono::steady_clock::now();
    double profile_timeout = config_.re_estimate_interval_sec * 3.0;

    std::lock_guard<std::mutex> lock(data_mutex_);
    for (auto it = recent_persons_.begin(); it != recent_persons_.end();) {
        double age = std::chrono::duration<double>(now - it->second.timestamp).count();
        it = (age > config_.max_cache_age_sec) ? recent_persons_.erase(it) : std::next(it);
    }
    for (auto it = recent_faces_.begin(); it != recent_faces_.end();) {
        double age = std::chrono::duration<double>(now - it->second.timestamp).count();
        it = (age > config_.max_cache_age_sec) ? recent_faces_.erase(it) : std::next(it);
    }

    for (auto it = person_profiles_.begin(); it != person_profiles_.end();) {
        const std::string& label_id = it->first;
        bool seen_recently =
            recent_persons_.count(label_id) > 0 || recent_faces_.count(label_id) > 0;
        double idle = std::chrono::duration<double>(now - it->second.last_updated).count();
        if (!seen_recently && idle > profile_timeout) {
            RCLCPP_INFO(get_logger(), "%s: cleaned up stale profile: %s", get_name(), label_id.c_str());
            known_label_ids_.erase(label_id);
            it = person_profiles_.erase(it);
        } else {
            ++it;
        }
    }
}

void AgeGenderDetectionNode::debugStatus() {
    bool has_image;
    double image_age;
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        has_image = has_image_;
        image_age = has_image_ ? std::chrono::duration<double>(std::chrono::steady_clock::now() - image_timestamp_).count() : -1.0;
    }

    size_t num_persons, num_faces, num_known, gaze_count = 0, gaze_depth_ok = 0;
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        num_persons = recent_persons_.size();
        num_faces = recent_faces_.size();
        num_known = known_label_ids_.size();
        for (const auto& [id, f] : recent_faces_) {
            if (f.mutual_gaze) {
                gaze_count++;
                if (f.depth > 0.0 && f.depth < config_.max_depth_m) gaze_depth_ok++;
            }
        }
    }

    RCLCPP_INFO(get_logger(), "%s: [DEBUG] Image: %s (age=%.1fs), Persons: %zu, Faces: %zu (%zu gaze, %zu <%.2fm), Known IDs: %zu",
        get_name(), has_image ? "YES" : "NO", image_age, num_persons, num_faces, gaze_count, gaze_depth_ok,
        config_.max_depth_m, num_known);
}
