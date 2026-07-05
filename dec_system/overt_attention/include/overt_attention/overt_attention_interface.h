/* overt_attention_interface.h
 *
 * Author: Yohannes Tadesse Haile
 * Date: Jun 12, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 CyLab Carnegie Mellon University Africa
 */

#ifndef OVERT_ATTENTION_INTERFACE_H
#define OVERT_ATTENTION_INTERFACE_H

// ROS includes
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <builtin_interfaces/msg/time.hpp>

// Message includes
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <naoqi_bridge_msgs/msg/joint_angles_with_speed.hpp>
#include <dec_interfaces/msg/face_detection.hpp>

// cv_bridge / OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// YAML
#include <yaml-cpp/yaml.h>

// Standard includes
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <functional>
#include <stdexcept>

//=============================================================================
// Shared configuration / helpers
//=============================================================================

// Mirrors the structure of data/pepper_topics.yaml
struct TopicsConfig {
    struct {
        std::string pepper;
        std::string realsense;
        bool use_compressed = false;
    } image;

    struct {
        std::string pepper;
        std::string realsense;
        bool use_compressed = false;
    } depth;

    struct {
        std::string peak;
        std::string map;
    } saliency;

    struct {
        std::string pepper;
        std::string realsense;
    } camera_info;

    std::string face;
    std::string audio;
    std::string target_angles;
    std::string joint_state;
    std::string joint_angles;
    std::string trajectory;
};

// Load topics configuration from a YAML file located under a package's share directory.
TopicsConfig loadTopicsConfig(const std::string& package_name, const std::string& relative_path);

// Select the pepper/realsense variant of a topic pair based on the camera_type parameter.
// Throws std::invalid_argument if camera_type is neither "pepper" nor "realsense".
inline const std::string& selectCameraTopic(
    const std::string& camera_type, const std::string& pepper, const std::string& realsense) {
    if (camera_type == "pepper") return pepper;
    if (camera_type == "realsense") return realsense;
    throw std::invalid_argument("Invalid camera_type: " + camera_type);
}

// QoS profile suitable for image transport over WiFi (best-effort, volatile, keep-last 1).
rclcpp::QoS getImageQoS();

// Build the full image topic name based on the compression setting.
std::string getImageTopic(const std::string& base_topic, bool use_compressed, bool is_depth = false);

// Clamp x to [lo, hi].
inline double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

// Convert a pixel coordinate to camera-relative (yaw, pitch) angles.
inline std::pair<double, double> pixelToAngles(double u, double v, double fx, double fy, double cx, double cy) {
    double x = (u - cx) / fx;
    double y = (v - cy) / fy;
    return {-std::atan2(x, 1.0), std::atan2(y, 1.0)};
}

// Generate a consistent BGR color for a face ID using a hash of the string.
cv::Scalar generateColorFromId(const std::string& face_id);

//=============================================================================
// Boolean Map Saliency (BMS)
//=============================================================================

// Boolean Map Saliency (BMS) - Frame-based
//  - Threshold-based boolean maps (per BMS paper)
//  - Flood-fill based region activation
//  - Output normalized to [0, 1]
class BooleanMapSaliency {
public:
    explicit BooleanMapSaliency(int n_thresholds = 10);

    // Compute saliency map for a downsampled BGR frame. Returns a CV_32F map in [0, 1].
    cv::Mat computeSaliency(const cv::Mat& frame_bgr);

private:
    // Suppress background regions of a boolean map using flood-fill from the borders.
    cv::Mat activateBooleanMap(const cv::Mat& bool_map);

    int n_thresholds_;
    std::vector<double> thresholds_;
};

//=============================================================================
// Saliency Node
//=============================================================================

// Computes bottom-up visual attention using Boolean Map Saliency (BMS).
// Publishes the top-N saliency peaks (pixel coords + score) and, optionally,
// a saliency-overlay visualization.
class SaliencyNode : public rclcpp::Node {
public:
    SaliencyNode();

private:
    void setupParameters();
    void loadParameters();

    // Image / depth callbacks (just cache the latest frame)
    void onImageRaw(const sensor_msgs::msg::Image::SharedPtr msg);
    void onImageCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void onDepthRaw(const sensor_msgs::msg::Image::SharedPtr msg);
    void onDepthCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void cacheDepth(const cv::Mat& depth_m);

    // Processing
    void timerCallback();
    cv::Mat computeDepthWeight();
    std::vector<cv::Vec3f> findPeaks(const cv::Mat& saliency);
    void processFrame(const cv::Mat& bgr, int width, int height, const builtin_interfaces::msg::Time& stamp);
    void publishVisualization(const cv::Mat& bgr, const cv::Mat& saliency,
                               const std::vector<cv::Vec3f>& peaks,
                               const builtin_interfaces::msg::Time& stamp);

    TopicsConfig topics_config_;

    // Parameters
    std::string camera_type_;
    bool use_compressed_ = true;
    bool use_depth_weighting_ = true;
    double depth_min_m_ = 0.3;
    double depth_max_m_ = 10.0;
    double depth_weight_min_ = 0.2;
    bool publish_map_flag_ = true;
    int down_w_ = 160;
    int down_h_ = 120;
    double min_peak_ = 0.25;
    double overlay_alpha_ = 0.4;
    int num_peaks_ = 10;
    int peak_min_dist_ = 50;
    double process_hz_ = 1.0;

    std::string image_topic_base_;
    std::string depth_topic_base_;

    // Full-resolution frame size (updated from incoming images)
    int W_ = 640;
    int H_ = 480;

    // Thread-safe frame storage
    std::mutex frame_mutex_;
    cv::Mat latest_frame_;
    cv::Size latest_size_;
    builtin_interfaces::msg::Time latest_stamp_;
    bool has_frame_ = false;

    // Thread-safe depth storage
    std::mutex depth_mutex_;
    cv::Mat depth_small_;
    bool has_depth_ = false;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_raw_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_image_compressed_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_raw_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_depth_compressed_;

    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_peak_;
    rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_map_;

    rclcpp::TimerBase::SharedPtr timer_;

    BooleanMapSaliency bms_;
};

//=============================================================================
// Unified Attention Node
//=============================================================================

// Improved attention controller for robot overt attention.
// Priority 1: Engaged faces | Priority 2: Detected faces | Priority 3: Saliency (with cooldown + IOR)
class UnifiedAttentionNode : public rclcpp::Node {
public:
    UnifiedAttentionNode();

private:
    struct FaceCandidate {
        std::string face_id;
        geometry_msgs::msg::Point centroid;
        double world_yaw;
        double world_pitch;
        bool mutual_gaze;
        double priority;
        bool is_current;
    };

    struct VisitedLocation {
        double yaw;
        double pitch;
        double timestamp;
    };

    void handleSetEnabled(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                           std::shared_ptr<std_srvs::srv::SetBool::Response> response);

    void onCamInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void onJointStates(const sensor_msgs::msg::JointState::SharedPtr msg);
    void onFaces(const dec_interfaces::msg::FaceDetection::SharedPtr msg);
    void onSaliency(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

    void moveHeadToDefault();
    double calculateFacePriority(const geometry_msgs::msg::Point& centroid, bool mutual_gaze, bool is_current_face);
    double calculateIorSuppression(double age_seconds);
    double applyIorFilter(double world_yaw, double world_pitch, double score);
    void cleanupWeakIor();
    void publishHead(double yaw, double pitch, double score = 0.0,
                     const std::string& source = "unknown", bool force = false);

    TopicsConfig topics_config_;

    // System parameters
    bool move_to_default_on_disable_ = true;
    double default_yaw_ = 0.0;
    double default_pitch_ = -0.2;
    double default_move_speed_ = 0.1;

    // Joint limits for face tracking
    double face_yaw_lim_ = 1.8;
    double face_pitch_up_ = 0.4;
    double face_pitch_dn_ = -0.7;

    // Joint limits for saliency
    double saliency_yaw_lim_ = 1.8;
    double saliency_pitch_up_ = 0.4;
    double saliency_pitch_dn_ = -0.7;

    // Face parameters
    double face_timeout_ = 2.0;
    double engaged_bonus_ = 2.0;
    double face_switch_cooldown_ = 1.0;
    double same_face_threshold_ = 0.0;  // radians
    bool prefer_closer_ = true;
    double max_face_distance_ = 5.0;

    // Stability parameters (anti-jitter)
    double min_angular_change_ = 0.0;  // radians
    double target_smoothing_alpha_ = 0.4;

    // Saliency parameters
    double saliency_min_ = 0.30;
    double min_cooldown_ = 1.5;
    double max_dwell_ = 3.0;
    double switch_ratio_ = 1.4;
    double same_target_threshold_ = 0.0;  // radians

    // IOR parameters
    bool enable_ior_ = true;
    double ior_max_suppression_ = 0.9;
    double ior_half_life_ = 3.0;
    double ior_radius_ = 0.0;  // radians
    double ior_cleanup_threshold_ = 0.05;
    int ior_max_locations_ = 20;

    // Enable/disable state
    bool attention_enabled_ = true;

    // Camera intrinsics
    std::optional<double> fx_, fy_, cx_, cy_;

    // Head state
    std::optional<double> head_yaw_, head_pitch_;

    // Smoothed target state
    std::optional<double> target_yaw_, target_pitch_;

    // Face state
    double last_face_time_ = 0.0;
    double last_face_switch_time_ = 0.0;
    std::optional<std::string> current_face_id_;
    std::optional<std::pair<double, double>> current_face_location_;

    // Saliency state
    double last_saliency_cmd_time_ = 0.0;
    std::optional<std::pair<double, double>> current_saliency_target_;
    double current_saliency_score_ = 0.0;

    // IOR state
    std::vector<VisitedLocation> visited_locations_;

    rclcpp::Subscription<dec_interfaces::msg::FaceDetection>::SharedPtr sub_faces_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_saliency_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_caminfo_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;

    rclcpp::Publisher<naoqi_bridge_msgs::msg::JointAnglesWithSpeed>::SharedPtr pub_head_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr pub_target_;

    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr srv_enable_;
};

//=============================================================================
// Visualization Node
//=============================================================================

// Shows faces with tracking IDs, engagement status, depth, saliency peaks,
// and the current head target overlaid on the camera image.
class VisualizationNode : public rclcpp::Node {
public:
    VisualizationNode();

private:
    struct FaceInfo {
        int u, v, w, h;
        float depth;
        std::string face_id;
        bool engaged;
        cv::Scalar color;
    };

    struct SaliencyPeak {
        int u, v;
        float score;
    };

    struct TargetInfo {
        double yaw, pitch, score;
    };

    void onCamInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
    void onFaces(const dec_interfaces::msg::FaceDetection::SharedPtr msg);
    void onSaliency(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void onTarget(const geometry_msgs::msg::Vector3::SharedPtr msg);
    void updateTargetFace();

    void onImageCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    void onImageRaw(const sensor_msgs::msg::Image::SharedPtr msg);
    void processFrame(const cv::Mat& frame, const builtin_interfaces::msg::Time& stamp);

    void drawFace(cv::Mat& vis, const FaceInfo& face, bool is_targeted);
    void drawSaliencyPeak(cv::Mat& vis, const SaliencyPeak& peak, int rank);
    void drawTarget(cv::Mat& vis, int width, int height);
    void drawInfo(cv::Mat& vis);

    TopicsConfig topics_config_;

    // Parameters
    bool show_face_ids_ = true;
    bool show_depth_ = true;
    bool show_engagement_ = true;

    std::string image_topic_;
    bool use_compressed_ = false;

    // State
    std::vector<FaceInfo> faces_;
    std::vector<SaliencyPeak> saliency_peaks_;
    std::optional<TargetInfo> current_target_;
    std::optional<std::string> target_face_id_;

    std::optional<double> fx_, fy_, cx_, cy_;

    std::unordered_map<std::string, cv::Scalar> face_colors_;

    // Counters for debugging
    int image_count_ = 0;
    int face_count_ = 0;
    int saliency_count_ = 0;
    int target_count_ = 0;

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_image_compressed_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_raw_;
    rclcpp::Subscription<dec_interfaces::msg::FaceDetection>::SharedPtr sub_faces_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_saliency_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr sub_target_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_caminfo_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_overlay_;
};

#endif // OVERT_ATTENTION_INTERFACE_H
