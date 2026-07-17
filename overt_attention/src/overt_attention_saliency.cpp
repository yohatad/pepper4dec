/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: June 12, 2026
Version: v1.0 - C++ port of overt_attention_saliency.py

Saliency Node - Computes bottom-up visual attention using Boolean Map Saliency (BMS)
*/

#include "overt_attention/overt_attention_interface.h"
#include <iomanip>
#include <sstream>

//=============================================================================
// BooleanMapSaliency
//=============================================================================

BooleanMapSaliency::BooleanMapSaliency(int n_thresholds) : n_thresholds_(n_thresholds) {
    // Mirrors np.linspace(0, 1, n_thresholds + 1, endpoint=False)[1:]
    thresholds_.reserve(n_thresholds_);
    for (int i = 1; i <= n_thresholds_; ++i) {
        thresholds_.push_back(static_cast<double>(i) / (n_thresholds_ + 1));
    }
}

cv::Mat BooleanMapSaliency::activateBooleanMap(const cv::Mat& bool_map) {
    cv::Mat activation = bool_map.clone();
    int h = activation.rows;
    int w = activation.cols;
    cv::Mat ffill_mask = cv::Mat::zeros(h + 2, w + 2, CV_8UC1);

    for (int y = 0; y < h; ++y) {
        if (activation.at<uchar>(y, 0)) {
            cv::floodFill(activation, ffill_mask, cv::Point(0, y), cv::Scalar(0));
        }
        if (activation.at<uchar>(y, w - 1)) {
            cv::floodFill(activation, ffill_mask, cv::Point(w - 1, y), cv::Scalar(0));
        }
    }
    for (int x = 0; x < w; ++x) {
        if (activation.at<uchar>(0, x)) {
            cv::floodFill(activation, ffill_mask, cv::Point(x, 0), cv::Scalar(0));
        }
        if (activation.at<uchar>(h - 1, x)) {
            cv::floodFill(activation, ffill_mask, cv::Point(x, h - 1), cv::Scalar(0));
        }
    }

    return activation;
}

cv::Mat BooleanMapSaliency::computeSaliency(const cv::Mat& frame_bgr) {
    cv::Mat lab;
    cv::cvtColor(frame_bgr, lab, cv::COLOR_BGR2Lab);
    lab.convertTo(lab, CV_32F);

    double lab_min, lab_max;
    cv::minMaxLoc(lab.reshape(1), &lab_min, &lab_max);
    double lab_range = lab_max - lab_min;
    if (lab_range < 1e-6) {
        return cv::Mat::zeros(frame_bgr.rows, frame_bgr.cols, CV_32F);
    }
    lab = (lab - lab_min) / lab_range;

    int h = lab.rows, w = lab.cols;
    cv::Mat saliency = cv::Mat::zeros(h, w, CV_32F);

    std::vector<cv::Mat> lab_ch;
    cv::split(lab, lab_ch);

    for (double thresh : thresholds_) {
        for (int c = 0; c < 3; ++c) {
            cv::Mat bool_map = (lab_ch[c] > thresh);
            cv::Mat activation = activateBooleanMap(bool_map);
            cv::Mat activation_f;
            activation.convertTo(activation_f, CV_32F, 1.0 / 255.0);
            saliency += activation_f;
        }
    }

    saliency /= (n_thresholds_ * 3);
    return saliency;
}

//=============================================================================
// SaliencyNode
//=============================================================================

SaliencyNode::SaliencyNode() : Node("saliency_node") {
    try {
        topics_config_ = loadTopicsConfig("overt_attention", "data/pepper_topics.yaml");
        RCLCPP_INFO(get_logger(), "Loaded topics configuration from overt_attention package");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to load topics configuration: %s", e.what());
        throw;
    }

    setupParameters();
    loadParameters();

    auto qos = getImageQoS();

    // Publishers
    pub_peak_ = create_publisher<std_msgs::msg::Float32MultiArray>(topics_config_.saliency.peak, 10);

    if (publish_map_flag_) {
        pub_map_ = create_publisher<sensor_msgs::msg::CompressedImage>(topics_config_.saliency.map, 1);
    }

    // Image subscriber
    std::string image_topic = getImageTopic(image_topic_base_, use_compressed_, false);
    if (use_compressed_) {
        sub_image_compressed_ = create_subscription<sensor_msgs::msg::CompressedImage>(
            image_topic, qos, std::bind(&SaliencyNode::onImageCompressed, this, std::placeholders::_1));
    } else {
        sub_image_raw_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic, qos, std::bind(&SaliencyNode::onImageRaw, this, std::placeholders::_1));
    }
    RCLCPP_INFO(get_logger(), "Subscribing to image: %s", image_topic.c_str());

    // Depth subscriber
    if (use_depth_weighting_) {
        std::string depth_topic = getImageTopic(depth_topic_base_, use_compressed_, true);
        if (use_compressed_) {
            sub_depth_compressed_ = create_subscription<sensor_msgs::msg::CompressedImage>(
                depth_topic, qos, std::bind(&SaliencyNode::onDepthCompressed, this, std::placeholders::_1));
        } else {
            sub_depth_raw_ = create_subscription<sensor_msgs::msg::Image>(
                depth_topic, qos, std::bind(&SaliencyNode::onDepthRaw, this, std::placeholders::_1));
        }
        RCLCPP_INFO(get_logger(), "Subscribing to depth: %s", depth_topic.c_str());
    }

    // Processing timer
    auto timer_period = std::chrono::duration<double>(1.0 / process_hz_);
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
        std::bind(&SaliencyNode::timerCallback, this));

    RCLCPP_INFO(get_logger(), "Saliency node ready @ %.1f Hz (depth_weighting=%s)",
                process_hz_, use_depth_weighting_ ? "ON" : "OFF");
}

void SaliencyNode::setupParameters() {
    declare_parameter("camera_type", std::string("pepper"));
    declare_parameter("use_compressed", true);
    declare_parameter("use_depth_weighting", true);
    declare_parameter("depth_min_m", 0.3);
    declare_parameter("depth_max_m", 10.0);
    declare_parameter("depth_weight_min", 0.2);
    declare_parameter("publish_map", true);
    declare_parameter("down_w", 160);
    declare_parameter("down_h", 120);
    declare_parameter("min_peak", 0.25);
    declare_parameter("overlay_alpha", 0.4);
    declare_parameter("num_peaks", 10);
    declare_parameter("peak_min_distance_px", 50);
    declare_parameter("process_hz", 1.0);
}

void SaliencyNode::loadParameters() {
    camera_type_ = get_parameter("camera_type").as_string();
    use_compressed_ = get_parameter("use_compressed").as_bool();

    image_topic_base_ = selectCameraTopic(camera_type_, topics_config_.image.pepper, topics_config_.image.realsense);
    depth_topic_base_ = selectCameraTopic(camera_type_, topics_config_.depth.pepper, topics_config_.depth.realsense);

    // Pepper depth is unreliable — force depth weighting off regardless of config
    if (camera_type_ == "pepper") {
        use_depth_weighting_ = false;
        RCLCPP_INFO(get_logger(), "Pepper camera: depth weighting disabled");
    } else {
        use_depth_weighting_ = get_parameter("use_depth_weighting").as_bool();
    }
    depth_min_m_ = get_parameter("depth_min_m").as_double();
    depth_max_m_ = get_parameter("depth_max_m").as_double();
    depth_weight_min_ = get_parameter("depth_weight_min").as_double();
    publish_map_flag_ = get_parameter("publish_map").as_bool();
    down_w_ = static_cast<int>(get_parameter("down_w").as_int());
    down_h_ = static_cast<int>(get_parameter("down_h").as_int());
    min_peak_ = get_parameter("min_peak").as_double();
    overlay_alpha_ = get_parameter("overlay_alpha").as_double();
    num_peaks_ = static_cast<int>(get_parameter("num_peaks").as_int());
    peak_min_dist_ = static_cast<int>(get_parameter("peak_min_distance_px").as_int());
    process_hz_ = get_parameter("process_hz").as_double();
}

//============ Image Callbacks (just cache frames) ============

void SaliencyNode::onImageRaw(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::string encoding = msg->encoding;
    std::transform(encoding.begin(), encoding.end(), encoding.begin(), ::tolower);

    if (encoding == "bgr8" || encoding == "rgb8") {
        cv::Mat img(msg->height, msg->width, CV_8UC3, const_cast<uint8_t*>(msg->data.data()));
        cv::Mat bgr = img.clone();
        if (encoding == "rgb8") {
            cv::cvtColor(bgr, bgr, cv::COLOR_RGB2BGR);
        }
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = bgr;
        latest_stamp_ = msg->header.stamp;
        latest_size_ = cv::Size(msg->width, msg->height);
        has_frame_ = true;
    }
}

void SaliencyNode::onImageCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    cv::Mat bgr = cv::imdecode(msg->data, cv::IMREAD_COLOR);
    if (!bgr.empty()) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = bgr;
        latest_stamp_ = msg->header.stamp;
        latest_size_ = cv::Size(bgr.cols, bgr.rows);
        has_frame_ = true;
    }
}

void SaliencyNode::onDepthRaw(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::string encoding = msg->encoding;
    std::transform(encoding.begin(), encoding.end(), encoding.begin(), ::tolower);

    cv::Mat depth_m;
    if (encoding == "16uc1" || encoding == "mono16") {
        cv::Mat depth(msg->height, msg->width, CV_16UC1, const_cast<uint8_t*>(msg->data.data()));
        depth.convertTo(depth_m, CV_32F, 1.0 / 1000.0);
    } else if (encoding == "32fc1") {
        cv::Mat depth(msg->height, msg->width, CV_32FC1, const_cast<uint8_t*>(msg->data.data()));
        depth_m = depth.clone();
    } else {
        return;
    }
    cacheDepth(depth_m);
}

void SaliencyNode::onDepthCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    if (msg->data.size() <= 12) {
        return;
    }
    std::vector<uint8_t> buf(msg->data.begin() + 12, msg->data.end());
    cv::Mat depth = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
    if (!depth.empty()) {
        cv::Mat depth_m;
        depth.convertTo(depth_m, CV_32F, 1.0 / 1000.0);
        cacheDepth(depth_m);
    }
}

void SaliencyNode::cacheDepth(const cv::Mat& depth_m) {
    cv::Mat depth_small;
    cv::resize(depth_m, depth_small, cv::Size(down_w_, down_h_), 0, 0, cv::INTER_NEAREST);
    std::lock_guard<std::mutex> lock(depth_mutex_);
    depth_small_ = depth_small;
    has_depth_ = true;
}

//============ Timer Callback ============

void SaliencyNode::timerCallback() {
    cv::Mat bgr;
    cv::Size full_size;
    builtin_interfaces::msg::Time stamp;
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (!has_frame_) {
            return;
        }
        bgr = latest_frame_;
        full_size = latest_size_;
        stamp = latest_stamp_;
    }
    processFrame(bgr, full_size.width, full_size.height, stamp);
}

//============ Processing ============

cv::Mat SaliencyNode::computeDepthWeight() {
    cv::Mat depth;
    {
        std::lock_guard<std::mutex> lock(depth_mutex_);
        if (!has_depth_) {
            return cv::Mat::ones(down_h_, down_w_, CV_32F);
        }
        depth = depth_small_.clone();
    }

    cv::Mat invalid = (depth <= 0) | (depth > 10.0);
    depth.setTo(depth_max_m_, invalid);

    double depth_range = depth_max_m_ - depth_min_m_;
    if (depth_range < 1e-6) {
        return cv::Mat::ones(down_h_, down_w_, CV_32F);
    }

    cv::Mat normalized = (depth - depth_min_m_) / depth_range;
    cv::min(normalized, 1.0, normalized);
    cv::max(normalized, 0.0, normalized);

    cv::Mat weight = 1.0 - normalized * (1.0 - depth_weight_min_);
    cv::Mat weight_f;
    weight.convertTo(weight_f, CV_32F);
    return weight_f;
}

std::vector<cv::Vec3f> SaliencyNode::findPeaks(const cv::Mat& saliency) {
    std::vector<cv::Vec3f> peaks;
    cv::Mat S_work = saliency.clone();
    int h = S_work.rows, w = S_work.cols;

    double scale_x = W_ / static_cast<double>(down_w_);
    double scale_y = H_ / static_cast<double>(down_h_);
    double avg_scale = (scale_x + scale_y) / 2.0;
    double min_dist_down = std::max(1.0, peak_min_dist_ / avg_scale);

    int pad = std::max(3, static_cast<int>(std::min(h, w) * 0.05));

    for (int i = 0; i < num_peaks_; ++i) {
        cv::Mat S_masked = S_work.clone();
        S_masked.rowRange(0, pad).setTo(0);
        S_masked.rowRange(h - pad, h).setTo(0);
        S_masked.colRange(0, pad).setTo(0);
        S_masked.colRange(w - pad, w).setTo(0);

        double max_val;
        cv::Point max_loc;
        cv::minMaxLoc(S_masked, nullptr, &max_val, nullptr, &max_loc);

        if (max_val < min_peak_) {
            break;
        }

        int u_s = max_loc.x;
        int v_s = max_loc.y;
        float u = static_cast<float>(u_s * scale_x);
        float v = static_cast<float>(v_s * scale_y);
        peaks.emplace_back(u, v, static_cast<float>(max_val));

        // Suppress region around the peak so subsequent peaks are spatially separated
        cv::circle(S_work, cv::Point(u_s, v_s), static_cast<int>(min_dist_down), cv::Scalar(0), -1);
    }

    return peaks;
}

void SaliencyNode::processFrame(const cv::Mat& bgr, int width, int height,
                                 const builtin_interfaces::msg::Time& stamp) {
    W_ = width;
    H_ = height;

    cv::Mat small_bgr;
    cv::resize(bgr, small_bgr, cv::Size(down_w_, down_h_), 0, 0, cv::INTER_AREA);

    cv::Mat S = bms_.computeSaliency(small_bgr);

    if (use_depth_weighting_) {
        S = S.mul(computeDepthWeight());
    }

    cv::GaussianBlur(S, S, cv::Size(5, 5), 0);

    double s_min, s_max;
    cv::minMaxLoc(S, &s_min, &s_max);
    if ((s_max - s_min) > 1e-6) {
        S = (S - s_min) / (s_max - s_min);
    } else {
        S = cv::Mat::zeros(S.size(), S.type());
    }

    std::vector<cv::Vec3f> peaks = findPeaks(S);

    // Publish peaks (u, v, score) - 3 values per peak
    std_msgs::msg::Float32MultiArray msg;
    for (const auto& peak : peaks) {
        msg.data.push_back(peak[0]);
        msg.data.push_back(peak[1]);
        msg.data.push_back(peak[2]);
    }
    pub_peak_->publish(msg);

    if (pub_map_) {
        publishVisualization(bgr, S, peaks, stamp);
    }
}

void SaliencyNode::publishVisualization(const cv::Mat& bgr, const cv::Mat& S,
                                         const std::vector<cv::Vec3f>& peaks,
                                         const builtin_interfaces::msg::Time& stamp) {
    cv::Mat scaled = S * 255;
    cv::Mat S_u8;
    scaled.convertTo(S_u8, CV_8U);

    cv::Mat vis;
    cv::resize(S_u8, vis, cv::Size(W_, H_), 0, 0, cv::INTER_LINEAR);

    cv::Mat vis_color;
    cv::applyColorMap(vis, vis_color, cv::COLORMAP_JET);
    cv::addWeighted(bgr, 1.0, vis_color, overlay_alpha_, 0, vis_color);

    static const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 255, 255), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255),
        cv::Scalar(0, 255, 0), cv::Scalar(255, 128, 0)
    };

    for (size_t idx = 0; idx < peaks.size(); ++idx) {
        float u = peaks[idx][0];
        float v = peaks[idx][1];
        float score = peaks[idx][2];
        cv::Scalar color = idx < colors.size() ? colors[idx] : cv::Scalar(200, 200, 200);

        cv::drawMarker(vis_color, cv::Point(static_cast<int>(u), static_cast<int>(v)), color,
                       cv::MARKER_TILTED_CROSS, 16, 2);

        std::ostringstream label;
        label << (idx + 1) << ":" << std::fixed << std::setprecision(2) << score;
        cv::putText(vis_color, label.str(),
                    cv::Point(static_cast<int>(u) + 12, static_cast<int>(v)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }

    sensor_msgs::msg::CompressedImage out;
    out.format = "png";
    out.header.stamp = stamp;
    cv::imencode(".png", vis_color, out.data);
    pub_map_->publish(out);
}

//=============================================================================
// main
//=============================================================================

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<SaliencyNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("saliency_node"), "Exception: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
