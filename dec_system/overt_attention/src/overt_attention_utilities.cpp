/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: June 12, 2026
Version: v1.0 - C++ port of the overt_attention package
*/

#include "overt_attention/overt_attention_interface.h"

TopicsConfig loadTopicsConfig(const std::string& package_name, const std::string& relative_path) {
    std::string package_share = ament_index_cpp::get_package_share_directory(package_name);
    std::string config_path = package_share + "/" + relative_path;

    YAML::Node yaml = YAML::LoadFile(config_path);
    YAML::Node topics = yaml["topics"];

    TopicsConfig cfg;
    cfg.image.base = topics["image"]["base"].as<std::string>();
    cfg.image.use_compressed = topics["image"]["use_compressed"].as<bool>();
    cfg.depth.base = topics["depth"]["base"].as<std::string>();
    cfg.depth.use_compressed = topics["depth"]["use_compressed"].as<bool>();
    cfg.saliency.peak = topics["saliency"]["peak"].as<std::string>();
    cfg.saliency.map = topics["saliency"]["map"].as<std::string>();
    cfg.face = topics["face"].as<std::string>();
    cfg.audio = topics["audio"].as<std::string>();
    cfg.camera_info = topics["camera_info"].as<std::string>();
    cfg.target_angles = topics["target_angles"].as<std::string>();
    cfg.joint_state = topics["joint_state"].as<std::string>();
    cfg.joint_angles = topics["joint_angles"].as<std::string>();
    cfg.trajectory = topics["trajectory"].as<std::string>();
    return cfg;
}

rclcpp::QoS getImageQoS() {
    return rclcpp::QoS(rclcpp::KeepLast(1))
        .reliability(rclcpp::ReliabilityPolicy::BestEffort)
        .durability(rclcpp::DurabilityPolicy::Volatile);
}

std::string getImageTopic(const std::string& base_topic, bool use_compressed, bool is_depth) {
    if (use_compressed) {
        return base_topic + (is_depth ? "/compressedDepth" : "/compressed");
    }
    return base_topic;
}

cv::Scalar generateColorFromId(const std::string& face_id) {
    std::size_t hash_val = std::hash<std::string>{}(face_id);

    int r = static_cast<int>(hash_val & 0xFF);
    int g = static_cast<int>((hash_val >> 8) & 0xFF);
    int b = static_cast<int>((hash_val >> 16) & 0xFF);

    double brightness = 0.299 * r + 0.587 * g + 0.114 * b;
    if (brightness < 100.0) {
        double scale = 150.0 / std::max(brightness, 1.0);
        r = std::min(255, static_cast<int>(r * scale));
        g = std::min(255, static_cast<int>(g * scale));
        b = std::min(255, static_cast<int>(b * scale));
    }

    return cv::Scalar(b, g, r);  // BGR for OpenCV
}
