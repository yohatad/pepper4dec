/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: June 12, 2026
Version: v1.0 - C++ port of overt_attention_visualization.py

Improved Visualization for Overt Attention System
Shows faces with tracking IDs, engagement status, depth, saliency peaks, and current head target
*/

#include "overt_attention/overt_attention_interface.h"
#include <iomanip>
#include <limits>
#include <sstream>

VisualizationNode::VisualizationNode() : Node("visualization_node") {
    try {
        topics_config_ = loadTopicsConfig("overt_attention", "data/pepper_topics.yaml");
        RCLCPP_INFO(get_logger(), "Loaded topics configuration from overt_attention package");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to load topics configuration: %s", e.what());
        throw;
    }

    // Parameters
    declare_parameter("camera_type", std::string("pepper"));
    declare_parameter("show_face_ids", true);
    declare_parameter("show_depth", true);
    declare_parameter("show_engagement", true);

    std::string camera_type = get_parameter("camera_type").as_string();
    show_face_ids_ = get_parameter("show_face_ids").as_bool();
    show_depth_ = get_parameter("show_depth").as_bool();
    show_engagement_ = get_parameter("show_engagement").as_bool();

    // Load topics from YAML config
    std::string face_topic = topics_config_.face;
    std::string saliency_topic = topics_config_.saliency.peak;
    std::string target_topic = topics_config_.target_angles;
    std::string camera_info_topic = selectCameraTopic(
        camera_type, topics_config_.camera_info.pepper, topics_config_.camera_info.realsense);

    use_compressed_ = topics_config_.image.use_compressed;
    image_topic_ = getImageTopic(
        selectCameraTopic(camera_type, topics_config_.image.pepper, topics_config_.image.realsense),
        use_compressed_);

    // QoS
    rclcpp::QoS qos = getImageQoS();

    // Subscriptions
    if (use_compressed_) {
        sub_image_compressed_ = create_subscription<sensor_msgs::msg::CompressedImage>(
            image_topic_, qos,
            std::bind(&VisualizationNode::onImageCompressed, this, std::placeholders::_1));
    } else {
        sub_image_raw_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic_, qos,
            std::bind(&VisualizationNode::onImageRaw, this, std::placeholders::_1));
    }

    RCLCPP_INFO(get_logger(), "Subscribing to image: %s", image_topic_.c_str());

    sub_faces_ = create_subscription<dec_interfaces::msg::FaceDetection>(
        face_topic, 10, std::bind(&VisualizationNode::onFaces, this, std::placeholders::_1));
    sub_saliency_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        saliency_topic, 10, std::bind(&VisualizationNode::onSaliency, this, std::placeholders::_1));
    sub_target_ = create_subscription<geometry_msgs::msg::Vector3>(
        target_topic, 10, std::bind(&VisualizationNode::onTarget, this, std::placeholders::_1));
    sub_caminfo_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic, qos, std::bind(&VisualizationNode::onCamInfo, this, std::placeholders::_1));

    // Publisher
    pub_overlay_ = create_publisher<sensor_msgs::msg::Image>("/overt_attention/visualization", 10);

    RCLCPP_INFO(get_logger(), "Improved visualization ready (Faces w/ Tracking + Engagement + Saliency)");
}

//============ Camera info / detections / target ============

void VisualizationNode::onCamInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (!fx_.has_value()) {
        fx_ = msg->k[0];
        fy_ = msg->k[4];
        cx_ = msg->k[2];
        cy_ = msg->k[5];
        RCLCPP_INFO(get_logger(), "Camera info: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f",
                    *fx_, *fy_, *cx_, *cy_);
    }
}

void VisualizationNode::onFaces(const dec_interfaces::msg::FaceDetection::SharedPtr msg) {
    face_count_++;
    faces_.clear();

    size_t n = msg->centroids.size();
    for (size_t i = 0; i < n; ++i) {
        double u = msg->centroids[i].x;
        double v = msg->centroids[i].y;
        double depth = msg->centroids[i].z;
        int w = (i < msg->width.size()) ? static_cast<int>(msg->width[i]) : 80;
        int h = (i < msg->height.size()) ? static_cast<int>(msg->height[i]) : 80;
        std::string face_id = (i < msg->face_label_id.size()) ? msg->face_label_id[i]
                                                                : ("unknown_" + std::to_string(i));
        bool engaged = (i < msg->mutual_gaze.size()) ? msg->mutual_gaze[i] : false;

        if (face_colors_.find(face_id) == face_colors_.end()) {
            face_colors_[face_id] = generateColorFromId(face_id);
        }

        FaceInfo face;
        face.u = static_cast<int>(u);
        face.v = static_cast<int>(v);
        face.w = w;
        face.h = h;
        face.depth = static_cast<float>(depth);
        face.face_id = face_id;
        face.engaged = engaged;
        face.color = face_colors_[face_id];
        faces_.push_back(face);
    }

    if (face_count_ == 1) {
        RCLCPP_INFO(get_logger(), "First face message received: %zu faces", faces_.size());
    }
}

void VisualizationNode::onSaliency(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    saliency_count_++;
    saliency_peaks_.clear();

    const auto& data = msg->data;
    for (size_t i = 0; i + 2 < data.size(); i += 3) {
        SaliencyPeak peak;
        peak.u = static_cast<int>(data[i]);
        peak.v = static_cast<int>(data[i + 1]);
        peak.score = data[i + 2];
        saliency_peaks_.push_back(peak);
    }

    if (saliency_count_ == 1) {
        RCLCPP_INFO(get_logger(), "First saliency message received: %zu peaks", saliency_peaks_.size());
    }
}

void VisualizationNode::onTarget(const geometry_msgs::msg::Vector3::SharedPtr msg) {
    target_count_++;

    TargetInfo target;
    target.yaw = msg->x;
    target.pitch = msg->y;
    target.score = msg->z;
    current_target_ = target;

    // Try to determine which face is being targeted (if any)
    updateTargetFace();

    if (target_count_ == 1) {
        RCLCPP_INFO(get_logger(), "First target received: yaw=%.1f\xc2\xb0, pitch=%.1f\xc2\xb0",
                    msg->x * 180.0 / M_PI, msg->y * 180.0 / M_PI);
    }
}

void VisualizationNode::updateTargetFace() {
    if (!current_target_.has_value() || faces_.empty() || !fx_.has_value()) {
        target_face_id_.reset();
        return;
    }

    double yaw = current_target_->yaw;
    double pitch = current_target_->pitch;

    double x_norm = -std::tan(yaw);
    double y_norm = std::tan(pitch);
    double target_u = x_norm * (*fx_) + (*cx_);
    double target_v = y_norm * (*fy_) + (*cy_);

    double min_dist = std::numeric_limits<double>::infinity();
    std::optional<std::string> closest_face_id;

    for (const auto& face : faces_) {
        double dist = std::sqrt(std::pow(face.u - target_u, 2) + std::pow(face.v - target_v, 2));
        if (dist < min_dist && dist < 100.0) {
            min_dist = dist;
            closest_face_id = face.face_id;
        }
    }

    target_face_id_ = closest_face_id;
}

//============ Image callbacks ============

void VisualizationNode::onImageCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    try {
        cv::Mat frame = cv::imdecode(msg->data, cv::IMREAD_COLOR);
        if (frame.empty()) {
            RCLCPP_WARN(get_logger(), "Failed to decode compressed image");
            return;
        }
        processFrame(frame, msg->header.stamp);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error processing compressed image: %s", e.what());
    }
}

void VisualizationNode::onImageRaw(const sensor_msgs::msg::Image::SharedPtr msg) {
    try {
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        processFrame(frame, msg->header.stamp);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error processing raw image: %s", e.what());
    }
}

void VisualizationNode::processFrame(const cv::Mat& frame, const builtin_interfaces::msg::Time& stamp) {
    image_count_++;

    if (image_count_ == 1) {
        RCLCPP_INFO(get_logger(), "First image received: %dx%d", frame.cols, frame.rows);
    }

    cv::Mat vis = frame.clone();
    int H = vis.rows, W = vis.cols;

    // Draw saliency peaks first (so they're under faces)
    for (size_t i = 0; i < saliency_peaks_.size(); ++i) {
        drawSaliencyPeak(vis, saliency_peaks_[i], static_cast<int>(i));
    }

    // Draw faces (Priority 1)
    for (const auto& face : faces_) {
        bool is_targeted = target_face_id_.has_value() && (face.face_id == *target_face_id_);
        drawFace(vis, face, is_targeted);
    }

    // Draw current target
    if (current_target_.has_value() && fx_.has_value()) {
        drawTarget(vis, W, H);
    }

    // Draw info overlay
    drawInfo(vis);

    // Publish as RAW Image
    try {
        cv_bridge::CvImage cv_image;
        cv_image.header.stamp = stamp;
        cv_image.header.frame_id = "camera_color_optical_frame";
        cv_image.encoding = "bgr8";
        cv_image.image = vis;

        auto out_msg = cv_image.toImageMsg();
        pub_overlay_->publish(*out_msg);

        if (image_count_ == 1) {
            RCLCPP_INFO(get_logger(), "First visualization published to /overt_attention/visualization");
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error publishing visualization: %s", e.what());
    }
}

//============ Drawing helpers ============

void VisualizationNode::drawFace(cv::Mat& vis, const FaceInfo& face, bool is_targeted) {
    int u = face.u, v = face.v;
    int w2 = face.w / 2, h2 = face.h / 2;

    const cv::Scalar& color = face.color;
    bool engaged = face.engaged;
    const std::string& face_id = face.face_id;
    float depth = face.depth;

    // Thicker box if targeted or engaged
    int thickness = (is_targeted || engaged) ? 5 : 3;

    // Draw bounding box
    int x1 = u - w2, y1 = v - h2;
    int x2 = u + w2, y2 = v + h2;
    cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness);

    // Draw engagement indicator (glowing circles) if engaged
    if (engaged) {
        cv::circle(vis, cv::Point(u, v), 25, cv::Scalar(0, 255, 0), 3);
        cv::circle(vis, cv::Point(u, v), 15, cv::Scalar(0, 255, 0), 2);
        cv::putText(vis, "ENGAGED", cv::Point(x1, y1 - 60),
                     cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    // Center cross
    cv::drawMarker(vis, cv::Point(u, v), color, cv::MARKER_CROSS, 15, 3);

    // Draw face ID if enabled
    if (show_face_ids_) {
        std::string id_text = "ID: " + face_id;
        if (is_targeted) {
            id_text = ">>> " + id_text + " <<<";
        }
        cv::putText(vis, id_text, cv::Point(x1, y1 - 35),
                     cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }

    // Draw depth if enabled and available
    if (show_depth_ && depth > 0) {
        std::ostringstream depth_text;
        depth_text << std::fixed << std::setprecision(2) << depth << "m";
        cv::putText(vis, depth_text.str(), cv::Point(x1, y2 + 25),
                     cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
    }

    // Draw "FACE" label
    std::string label = is_targeted ? "TARGETED FACE" : "FACE";
    cv::putText(vis, label, cv::Point(x1, y1 - 10),
                 cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv::LINE_AA);
}

void VisualizationNode::drawSaliencyPeak(cv::Mat& vis, const SaliencyPeak& peak, int rank) {
    int u = peak.u, v = peak.v;
    float score = peak.score;

    cv::Scalar color;
    int radius;
    if (rank == 0) {
        color = cv::Scalar(0, 255, 255);  // Yellow - highest
        radius = static_cast<int>(20 + score * 30);
    } else {
        double alpha = std::max(0.3, 1.0 - rank * 0.2);
        color = cv::Scalar(0, alpha * 200, alpha * 200);
        radius = static_cast<int>(15 + score * 20);
    }

    cv::circle(vis, cv::Point(u, v), radius, color, 2);
    cv::circle(vis, cv::Point(u, v), 3, color, -1);

    std::ostringstream label;
    label << "SAL #" << (rank + 1) << ": " << std::fixed << std::setprecision(2) << score;
    cv::putText(vis, label.str(), cv::Point(u + 25, v + 5 + rank * 20),
                 cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

void VisualizationNode::drawTarget(cv::Mat& vis, int width, int height) {
    double yaw = current_target_->yaw;
    double pitch = current_target_->pitch;
    double score = current_target_->score;

    // Project angles back to pixel coordinates
    double x_norm = -std::tan(yaw);
    double y_norm = std::tan(pitch);
    int u = static_cast<int>(x_norm * (*fx_) + (*cx_));
    int v = static_cast<int>(y_norm * (*fy_) + (*cy_));

    // Draw even if off-screen (with indicator)
    if (!(u >= 0 && u < width && v >= 0 && v < height)) {
        cv::putText(vis, "TARGET OFF-SCREEN", cv::Point(width / 2 - 100, 30),
                     cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        return;
    }

    // Draw target reticle
    cv::Scalar color(0, 0, 255);  // Red
    int size = 40;

    cv::line(vis, cv::Point(u - size, v), cv::Point(u + size, v), color, 3);
    cv::line(vis, cv::Point(u, v - size), cv::Point(u, v + size), color, 3);
    cv::circle(vis, cv::Point(u, v), size, color, 3);
    cv::circle(vis, cv::Point(u, v), 5, color, -1);

    // Label
    std::ostringstream label;
    label << "TARGET: (" << std::fixed << std::setprecision(1)
          << (yaw * 180.0 / M_PI) << "\xc2\xb0, " << (pitch * 180.0 / M_PI) << "\xc2\xb0)";
    if (score > 0) {
        label << " s=" << std::setprecision(2) << score;
    }

    // Add face ID if targeting a face
    if (target_face_id_.has_value()) {
        label << " [" << *target_face_id_ << "]";
    }

    cv::putText(vis, label.str(), cv::Point(u + 50, v - 50),
                 cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv::LINE_AA);
}

void VisualizationNode::drawInfo(cv::Mat& vis) {
    // Count engaged faces
    int engaged_count = 0;
    for (const auto& face : faces_) {
        if (face.engaged) {
            engaged_count++;
        }
    }

    // Semi-transparent background
    cv::Mat overlay = vis.clone();
    int info_height = (engaged_count > 0) ? 200 : 180;
    cv::rectangle(overlay, cv::Point(10, 10), cv::Point(450, info_height), cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.7, vis, 0.3, 0, vis);

    // Info text
    int y = 35;
    std::vector<std::string> info;

    {
        std::ostringstream oss;
        oss << "Faces: " << faces_.size() << " (" << engaged_count << " engaged)";
        info.push_back(oss.str());
    }
    {
        std::ostringstream oss;
        oss << "Saliency peaks: " << saliency_peaks_.size();
        info.push_back(oss.str());
    }
    {
        std::string target_str;
        if (target_face_id_.has_value()) {
            target_str = *target_face_id_;
        } else if (current_target_.has_value()) {
            target_str = "Saliency";
        } else {
            target_str = "None";
        }
        info.push_back("Target: " + target_str);
    }

    // Add tracked face IDs
    if (!faces_.empty()) {
        std::ostringstream oss;
        oss << "Tracked IDs: ";
        size_t shown = std::min<size_t>(5, faces_.size());
        for (size_t i = 0; i < shown; ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << faces_[i].face_id;
        }
        info.push_back(oss.str());
        if (faces_.size() > 5) {
            info.push_back("  ... and " + std::to_string(faces_.size() - 5) + " more");
        }
    }

    for (const auto& text : info) {
        cv::putText(vis, text, cv::Point(20, y),
                     cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        y += 30;
    }

    // Legend
    y += 10;
    cv::putText(vis, "Legend:", cv::Point(20, y),
                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    y += 20;

    struct LegendItem {
        std::string label;
        std::string desc;
        cv::Scalar color;
    };
    std::vector<LegendItem> legend_items = {
        {"Green Glow", "Engaged (mutual gaze)", cv::Scalar(0, 255, 0)},
        {"Colored Box", "Tracked face", cv::Scalar(255, 255, 255)},
        {"Red Cross", "Attention target", cv::Scalar(0, 0, 255)},
        {"Yellow Circle", "Top saliency", cv::Scalar(0, 255, 255)},
    };

    for (const auto& item : legend_items) {
        cv::rectangle(vis, cv::Point(20, y - 10), cv::Point(35, y + 5), item.color, -1);
        cv::putText(vis, item.label + ": " + item.desc, cv::Point(45, y),
                     cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
        y += 18;
    }
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<VisualizationNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("visualization_node"), "Exception: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
