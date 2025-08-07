#pragma once

// Include the necessary libraries
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <naoqi_bridge_msgs/msg/joint_angles_with_speed.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <string.h>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>  
#include <ctype.h>
#include <iostream>
#include <math.h>
#include <complex>
#include <algorithm>
#include <signal.h>
#include <csignal>
#include <fstream>
#include <sstream>
#include <iomanip> 
#include <regex>
#include <yaml-cpp/yaml.h>
#include <rclcpp/logger.hpp>

#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <unordered_map>
#include <optional>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/saliency.hpp>
#include <cv_bridge/cv_bridge.h>

// ROS2 message and service includes
// #include "cssr_system/msg/face_detection/data.hpp"
#include "cssr_system/msg/mode.hpp"
#include "cssr_system/srv/set_mode.hpp"
#include "cssr_system/srv/get_mode.hpp"

#define SOFTWARE_VERSION            "v1.0"
#pragma once

#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
#include <unordered_map>
#include <string>

/***
 * Generic YAML loader that merges compile-time defaults with
 * optional overrides from a YAML file, storing all values as strings.
 *
 * KeyT: the key type (e.g. std::string)
 ***/
template<typename KeyT>
class YamlLoaderString {
public:
  using Map = std::unordered_map<KeyT, std::string>;

  /// Merge compile-time defaults with overrides from a YAML node
  static YamlLoaderString fromYaml(const YAML::Node & node, const Map &defaults) {
    YamlLoaderString loader;
    loader.data_ = defaults;
    for (auto &kv : defaults) {
      const KeyT &key = kv.first;
      if (node[key]) {
        loader.data_[key] = node[key].template as<std::string>();
      }
    }
    return loader;
  }

  /// Load from file, log errors, fallback to defaults
  static YamlLoaderString loadFromFile(
    const std::string &path,
    const rclcpp::Logger &logger,
    const Map &defaults
  ) {
    YAML::Node root;
    try {
      root = YAML::LoadFile(path);
    } catch (const YAML::BadFile &e) {
      RCLCPP_ERROR_STREAM(logger, "Could not open YAML [" << path << "]: " << e.what());
      return fromYaml(YAML::Node{}, defaults);
    }
    try {
      return fromYaml(root, defaults);
    } catch (const YAML::ParserException &e) {
      RCLCPP_ERROR_STREAM(logger, "YAML parse error in [" << path << "]: " << e.what());
      return fromYaml(YAML::Node{}, defaults);
    }
  }

  /// Retrieve raw string value (throws if key missing)
  const std::string &getString(const KeyT &key) const {
    auto it = data_.find(key);
    if (it == data_.end()) {
      throw std::out_of_range("Missing key: " + key);
    }
    return it->second;
  }

private:
  Map data_;
};

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------
using ConfigLoader = YamlLoaderString<std::string>;

// compile-time defaults for config (all values as strings)
static const ConfigLoader::Map CONFIG_DEFAULTS = {
  {"camera",                "RealSenseCamera"},
  {"realignment_threshold", "50"},
  {"x_offset_to_head_yaw",  "0"},
  {"y_offset_to_head_pitch","0"},
  {"social_attention_mode", "random"},
  {"num_faces_social_att",  "3"},
  {"engagement_timeout",    "12.0"},
  {"use_sound",             "false"},
  {"use_compressed_images", "false"},
  {"verbose_mode",          "false"},
};

// ---------------------------------------------------------------------------
// TOPICS
// ---------------------------------------------------------------------------
using TopicLoader = YamlLoaderString<std::string>;

// compile-time defaults for topics (PascalCase keys)
static const TopicLoader::Map TOPIC_DEFAULTS = {
  {"RealSenseCamera",                "/camera/color/image_raw"},
  {"RealSenseCameraCompressed",      "/camera/color/image_raw/compressed"},
  {"RealSenseCameraDepth",           "/camera/depth/image_raw"},
  {"RealSenseCameraDepthCompressed", "/camera/depth/image_raw/compressed"},
  
  {"FrontCamera",                    "/naoqi_driver/camera/front/image_raw"},
  {"DepthCamera",                    "/naoqi_driver/camera/depth/image_raw"},
  
  {"JointAngles",                    "/joint_angles"},
  {"Wheels",                         "/pepper_dcm/cmd_moveto"},
  {"JointStates",                    "/joint_states"},
  
  {"RobotPose",                      "/robotLocalization/pose"},
  {"FaceDetection",                  "/faceDetection/data"},
  {"SoundLocalization",              "/soundDetection/direction"},
  
  {"SetMode",                        "/overtAttention/set_mode"},
  {"GetMode",                        "/overtAttention/get_mode"},
};

struct AngleChange {
    double delta_yaw;
    double delta_pitch;
};

struct PixelCoordinates {
    int x;
    int y;
    
};

class AttentionState {
public:
    enum class Mode { DISABLED, SOCIAL, SCANNING, SEEKING, LOCATION };

private:
    Mode current_mode{Mode::DISABLED};

public:
    AttentionState() = default;
    // Current attention mode and sensory flags
    bool face_detected{false};
    bool sound_detected{false};
    bool mutual_gaze_detected{false};
    bool face_within_range{false};
    double sound_angle{0.0};

    std::mt19937 random_generator{};
    std::uniform_int_distribution<int> random_distribution{0, 100};

    std::array<double, 3> robot_pose{0.0, 0.0, 0.0};
    std::array<double, 2> head_joint_states{0.0, 0.0};

    // Set the current attention mode
    Mode getMode() const { return current_mode;}
    void setMode(Mode m) { current_mode = m; }

    // Handle location-based attention behavior
    int locationAttention(float point_x,float point_y,float point_z);

    // Handle social attention behavior
    int socialAttention(int realignment_threshold,int social_control);

    // Handle scanning attention behavior
    int scanningAttention(double control_head_yaw, double control_head_pitch);

    // Handle seeking attention behavior
    int seekingAttention(int realignment_threshold);
};

struct HeadJointLimits {
    static constexpr double MIN_HEAD_YAW = -2.0857;
    static constexpr double MAX_HEAD_YAW = 2.0857;
    static constexpr double MIN_HEAD_PITCH = -0.7068;
    static constexpr double MAX_HEAD_PITCH = 0.6371;
    
    static constexpr double MIN_HEAD_PITCH_SCANNING = -0.3;
    static constexpr double MAX_HEAD_PITCH_SCANNING = 0.0;
    static constexpr double MIN_HEAD_YAW_SCANNING = -0.58353;  // ~33.4 degrees
    static constexpr double MAX_HEAD_YAW_SCANNING = 0.58353;   // ~33.4 degrees
    
    static constexpr double DEFAULT_HEAD_PITCH = -0.2;
    static constexpr double DEFAULT_HEAD_YAW = 0.0;
    
    static constexpr double DROP_HEAD_PITCH = 0.1;
    static constexpr double MAX_HEAD_PITCH_PERSON = -0.25;
};
struct RobotConstants {
    static constexpr double TORSO_HEIGHT = 820.0;  // mm
    
    // Pepper robot links lengths (mm)
    static constexpr double LINK_1 = -38.0;
    static constexpr double LINK_2 = 169.9;
    static constexpr double LINK_3 = 93.6;
    static constexpr double LINK_4 = 61.6;
    static constexpr double L_HEAD_1 = 112.051;
};

namespace camera_info {
struct Specs {
  double vfov_deg;   // vertical field of view in degrees
  double hfov_deg;   // horizontal field of view in degrees
  int    width_px;   // image width in pixels
  int    height_px;  // image height in pixels
  constexpr Specs(double v, double h, int w, int ht)
    : vfov_deg(v), hfov_deg(h), width_px(w), height_px(ht) {}
};

inline constexpr Specs PepperFront{ 44.30, 55.20, 640, 480 };
inline constexpr Specs RealSense{  42.50, 69.50, 640, 480 };

}  // namespace camera_info

struct SaliencyConstants {
    static constexpr int PATCH_RADIUS = 15;  // Radius for the saliency patch
    static constexpr int IOR_LIMIT = 50;  // Inhibition of return limit
    static constexpr double HABITUATION_RATE = 0.1;  // Rate of habituation for saliency map
};

struct InfoPeriods {
    static constexpr double INITIALIZATION_INFO_PERIOD = 5.0;  // Period for initialization info messages
    static constexpr double OPERATION_INFO_PERIOD = 10.0;      // Period for operation info messages
};

class OvertAttentionSystem : public rclcpp::Node
{
public:
    explicit OvertAttentionSystem();
    ~OvertAttentionSystem() override = default;

    bool initialize();
    
    void run();
    
private:
    AttentionState state;
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr                      camera_sub;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr                 joint_states_sub;
    // rclcpp::Subscription<cssr_system::msg::FaceDetectionData>::SharedPtr          face_detection_sub;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr                       sound_localization_sub;
    rclcpp::Subscription<geometry_msgs::msg::Pose2D>::SharedPtr                   robot_pose_sub;

    // publishers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr                      velocity_pub;
    rclcpp::Publisher<naoqi_bridge_msgs::msg::JointAnglesWithSpeed>::SharedPtr   joint_angles_pub;

    // service
    rclcpp::Service<cssr_system::srv::SetMode>::SharedPtr                        set_mode_service;
    rclcpp::Service<cssr_system::srv::GetMode>::SharedPtr                        get_mode_service;

    // callbacks
    void camera_callback(const sensor_msgs::msg::Image::SharedPtr msg);
    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
    // void face_detection_callback(const cssr_system::msg::FaceDetectionData::SharedPtr msg);
    void sound_localization_callback(const std_msgs::msg::Float32::SharedPtr msg);
    void robot_pose_callback(const geometry_msgs::msg::Pose2D::SharedPtr msg);

    void set_mode_callback(const std::shared_ptr<cssr_system::srv::SetMode::Request>  request,
        std::shared_ptr<cssr_system::srv::SetMode::Response>       response);

    void get_mode(const std::shared_ptr<cssr_system::srv::GetMode::Request>  request,
        std::shared_ptr<cssr_system::srv::GetMode::Response>       response);

    void topic_availability_check(const std::string & topic_name);

    void move_head_to_angles(double pitch, double yaw);
    void publish_velocity(const geometry_msgs::msg::Twist & twist_msg);

    void get_head_angles(double camera_x, double camera_y, double camera_z, double* head_yaw, double* head_pitch);
    // void update_attention_mode(const cssr_system::msg::OvertAttentionMode & mode_msg);
};

// Camera configuration struct
struct CameraConfig {
    double vfov_deg;
    double hfov_deg;
    int width_px;
    int height_px;
    
    CameraConfig(double v, double h, int w, int ht)
        : vfov_deg(v), hfov_deg(h), width_px(w), height_px(ht) {}
};

class SaliencyProcessor {
public:
    explicit SaliencyProcessor(const CameraConfig& camera_config);
  
    cv::Mat compute_saliency_map(const cv::Mat& image);
    std::pair<int, int> find_winner_takes_all(const cv::Mat& saliency_map);
    AngleChange pixel_to_angle(double center_x, double center_y) const;
    void habituation(cv::Mat& saliency_map, const std::vector<cv::Point>& previous_points);
    
    std::pair<cv::Mat, std::vector<std::tuple<double, double, int>>> habituation(cv::Mat& saliency_map, cv::Mat& wta_map, const std::vector<std::tuple<double, double, int>>& previous_locations); 
    std::pair<cv::Mat, std::vector<std::tuple<double, double, int>>> inhibition_of_return(cv::Mat& saliency_map, cv::Mat& wta_map, const std::vector<std::tuple<double, double, int>>& previous_locations);

private:
    cv::Mat faces_map;                                                          // Stores the map of the faces detected
    cv::Mat camera_image;                                                       // Hold image from the robot camera for scanning mode
    std::vector<std::tuple<double, double, double>> previous_locations;         // Stores locations that won in the WTA
    std::vector<std::tuple<double, double, double>> face_locations;             // Stores the locations of the detected faces saliency
};
