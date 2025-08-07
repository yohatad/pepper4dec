
#include "overtAttention/overtAttentionInterface.h"

// Constructor: set up all pubs/subs/services/timers
OvertAttentionSystem::OvertAttentionSystem()
: Node("overtAttention")
{
  if (!initialize()) {
    RCLCPP_ERROR(get_logger(), "Initialization failed; shutting down.");
    rclcpp::shutdown();
  }
}

bool OvertAttentionSystem::initialize()
{
    RCLCPP_INFO(get_logger(), "Initializing OvertAttentionSystem...");

    // 1) Load our node‐specific config (all values are strings)
    const auto pkg      = ament_index_cpp::get_package_share_directory("cssr_system");
    const auto cfg_file = pkg + "/overtAttention/config/overtAttentionConfiguration.yaml";
    auto cfg = ConfigLoader::loadFromFile(cfg_file, get_logger(), CONFIG_DEFAULTS);

    std::string camera_key = cfg.getString("camera");
    std::string use_compressed_images = cfg.getString("use_compressed_images");

    std::string depth_key;
    
    // Make camera_key lowercase for consistency
    std::transform(camera_key.begin(), camera_key.end(), camera_key.begin(), ::tolower);
    std::transform(use_compressed_images.begin(), use_compressed_images.end(), use_compressed_images.begin(), ::tolower);

    // The compressed image topic is only used if the camera is RealSense
    if (camera_key == "realsensecamera" && use_compressed_images == "true") {
        camera_key = "RealSenseCameraCompressed";
    } else if (camera_key == "realsensecamera" && use_compressed_images == "false") {
        camera_key = "RealSenseCamera";
    } else if (camera_key == "peppercamera") {
        camera_key = "FrontCamera";
    } else {
        RCLCPP_ERROR(get_logger(), "Unsupported camera type: %s", camera_key.c_str());
        return false;
    }

    const auto topics_file = pkg + "/overtAttention/data/pepperTopics.yaml";
    auto topics = TopicLoader::loadFromFile(topics_file, get_logger(), TOPIC_DEFAULTS);

    camera_sub = create_subscription<sensor_msgs::msg::Image>(topics.getString(camera_key), 10,
        std::bind(&OvertAttentionSystem::camera_callback, this, std::placeholders::_1));
        
    velocity_pub = create_publisher<geometry_msgs::msg::Twist>(topics.getString("Wheels"), 10);
    
    joint_states_sub = create_subscription<sensor_msgs::msg::JointState>(topics.getString("JointStates"), 10,
        std::bind(&OvertAttentionSystem::joint_states_callback, this, std::placeholders::_1));

    sound_localization_sub = create_subscription<std_msgs::msg::Float32>(topics.getString("SoundLocalization"), 10,
        std::bind(&OvertAttentionSystem::sound_localization_callback, this, std::placeholders::_1));

    robot_pose_sub = create_subscription<geometry_msgs::msg::Pose2D>(topics.getString("RobotPose"), 10,
        std::bind(&OvertAttentionSystem::robot_pose_callback, this, std::placeholders::_1));

    // face_detection_sub = create_subscription<cssr_system::msg::FaceDetectionData>(topics.getString("FaceDetection"), 10,
    //     std::bind(&OvertAttentionSystem::face_detection_callback, this, std::placeholders::_1));

    joint_angles_pub = create_publisher<naoqi_bridge_msgs::msg::JointAnglesWithSpeed>(topics.getString("JointAngles"), 10);

    // Create services for setting and getting modes
    set_mode_service = create_service<cssr_system::srv::SetMode>(topics.getString("SetMode"),
        std::bind(&OvertAttentionSystem::set_mode_callback, this,
                std::placeholders::_1, std::placeholders::_2));

    get_mode_service = create_service<cssr_system::srv::GetMode>(topics.getString("GetMode"),
        std::bind(&OvertAttentionSystem::get_mode, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(), "Initialization complete.");
    return true;
}

void OvertAttentionSystem::run()
{
    
}

void OvertAttentionSystem::camera_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
   
}

void OvertAttentionSystem::joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    // Process joint states
}

void OvertAttentionSystem::sound_localization_callback(const std_msgs::msg::Float32::SharedPtr msg)
{
    // Process sound localization data
}

void OvertAttentionSystem::robot_pose_callback(const geometry_msgs::msg::Pose2D::SharedPtr msg)
{
    // Process robot pose data
}

void OvertAttentionSystem::set_mode_callback(
    const std::shared_ptr<cssr_system::srv::SetMode::Request> request,
    std::shared_ptr<cssr_system::srv::SetMode::Response> response)
{
    // // Set the attention mode based on the request
    // state.setMode(static_cast<AttentionState::Mode>(request->mode));
    // response->success = true;
}

void OvertAttentionSystem::get_mode(
    const std::shared_ptr<cssr_system::srv::GetMode::Request> request,
    std::shared_ptr<cssr_system::srv::GetMode::Response> response)
{
    // // Get the current attention mode
    // response->mode = static_cast<int>(state.getMode());
}



