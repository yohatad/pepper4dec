/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: June 12, 2026
Version: v1.0 - C++ port of overt_attention_unified_attention.py

Improved Attention Controller for Robot Overt Attention
Priority 1: Engaged Faces | Priority 2: Detected Faces | Priority 3: Saliency (with cooldown + IOR)
*/

#include "overt_attention/overt_attention_interface.h"

UnifiedAttentionNode::UnifiedAttentionNode() : Node("simple_attention") {
    try {
        topics_config_ = loadTopicsConfig("overt_attention", "data/pepper_topics.yaml");
        RCLCPP_INFO(get_logger(), "Loaded topics configuration from overt_attention package");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to load topics configuration: %s", e.what());
        throw;
    }

    // System parameters
    declare_parameter("camera_type", std::string("pepper"));
    declare_parameter("start_enabled", true);
    declare_parameter("move_to_default_on_disable", true);
    declare_parameter("default_yaw", 0.0);
    declare_parameter("default_pitch", -0.2);
    declare_parameter("default_move_speed", 0.1);

    // Joint limits for face tracking
    declare_parameter("face_yaw_lim", 1.8);
    declare_parameter("face_pitch_up", 0.4);
    declare_parameter("face_pitch_dn", -0.7);

    // Joint limits for saliency
    declare_parameter("saliency_yaw_lim", 1.8);
    declare_parameter("saliency_pitch_up", 0.4);
    declare_parameter("saliency_pitch_dn", -0.7);

    // Face parameters
    declare_parameter("face_timeout", 2.0);
    declare_parameter("engaged_priority_bonus", 2.0);
    declare_parameter("face_switch_cooldown", 1.0);
    declare_parameter("same_face_threshold_deg", 8.0);
    declare_parameter("prefer_closer_faces", true);
    declare_parameter("max_face_distance", 5.0);

    // Stability parameters (anti-jitter)
    declare_parameter("min_angular_change_deg", 2.0);
    declare_parameter("target_smoothing_alpha", 0.4);

    // Saliency parameters
    declare_parameter("saliency_min_score", 0.30);
    declare_parameter("saliency_min_cooldown", 1.5);
    declare_parameter("saliency_max_dwell", 3.0);
    declare_parameter("switch_score_ratio", 1.4);
    declare_parameter("same_target_threshold_deg", 5.0);

    // IOR parameters
    declare_parameter("enable_ior", true);
    declare_parameter("ior_max_suppression", 0.9);
    declare_parameter("ior_half_life", 3.0);
    declare_parameter("ior_radius_deg", 15.0);
    declare_parameter("ior_cleanup_threshold", 0.05);
    declare_parameter("ior_max_locations", 20);

    // Load topics from YAML config
    std::string camera_type = get_parameter("camera_type").as_string();
    std::string face_topic = topics_config_.face;
    std::string saliency_topic = topics_config_.saliency.peak;
    std::string camera_info_topic = selectCameraTopic(
        camera_type, topics_config_.camera_info.pepper, topics_config_.camera_info.realsense);
    std::string head_cmd_topic = topics_config_.joint_angles;
    std::string target_topic = topics_config_.target_angles;

    // Load parameters
    move_to_default_on_disable_ = get_parameter("move_to_default_on_disable").as_bool();
    default_yaw_ = get_parameter("default_yaw").as_double();
    default_pitch_ = get_parameter("default_pitch").as_double();
    default_move_speed_ = get_parameter("default_move_speed").as_double();

    face_yaw_lim_ = get_parameter("face_yaw_lim").as_double();
    face_pitch_up_ = get_parameter("face_pitch_up").as_double();
    face_pitch_dn_ = get_parameter("face_pitch_dn").as_double();

    saliency_yaw_lim_ = get_parameter("saliency_yaw_lim").as_double();
    saliency_pitch_up_ = get_parameter("saliency_pitch_up").as_double();
    saliency_pitch_dn_ = get_parameter("saliency_pitch_dn").as_double();

    face_timeout_ = get_parameter("face_timeout").as_double();
    engaged_bonus_ = get_parameter("engaged_priority_bonus").as_double();
    face_switch_cooldown_ = get_parameter("face_switch_cooldown").as_double();
    same_face_threshold_ = get_parameter("same_face_threshold_deg").as_double() * M_PI / 180.0;
    prefer_closer_ = get_parameter("prefer_closer_faces").as_bool();
    max_face_distance_ = get_parameter("max_face_distance").as_double();

    min_angular_change_ = get_parameter("min_angular_change_deg").as_double() * M_PI / 180.0;
    target_smoothing_alpha_ = get_parameter("target_smoothing_alpha").as_double();

    saliency_min_ = get_parameter("saliency_min_score").as_double();
    min_cooldown_ = get_parameter("saliency_min_cooldown").as_double();
    max_dwell_ = get_parameter("saliency_max_dwell").as_double();
    switch_ratio_ = get_parameter("switch_score_ratio").as_double();
    same_target_threshold_ = get_parameter("same_target_threshold_deg").as_double() * M_PI / 180.0;

    enable_ior_ = get_parameter("enable_ior").as_bool();
    ior_max_suppression_ = get_parameter("ior_max_suppression").as_double();
    ior_half_life_ = get_parameter("ior_half_life").as_double();
    ior_radius_ = get_parameter("ior_radius_deg").as_double() * M_PI / 180.0;
    ior_cleanup_threshold_ = get_parameter("ior_cleanup_threshold").as_double();
    ior_max_locations_ = static_cast<int>(get_parameter("ior_max_locations").as_int());

    // Enable/disable state
    attention_enabled_ = get_parameter("start_enabled").as_bool();

    // Subscriptions
    auto qos_img = getImageQoS();
    sub_faces_ = create_subscription<dec_interfaces::msg::FaceDetection>(
        face_topic, 10, std::bind(&UnifiedAttentionNode::onFaces, this, std::placeholders::_1));
    sub_saliency_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        saliency_topic, 10, std::bind(&UnifiedAttentionNode::onSaliency, this, std::placeholders::_1));
    sub_caminfo_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic, qos_img, std::bind(&UnifiedAttentionNode::onCamInfo, this, std::placeholders::_1));
    sub_joint_states_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10, std::bind(&UnifiedAttentionNode::onJointStates, this, std::placeholders::_1));

    // Publishers
    pub_head_ = create_publisher<naoqi_bridge_msgs::msg::JointAnglesWithSpeed>(head_cmd_topic, 10);
    pub_target_ = create_publisher<geometry_msgs::msg::Vector3>(target_topic, 10);

    // Services
    srv_enable_ = create_service<std_srvs::srv::SetBool>(
        "/overt_attention/set_enabled",
        std::bind(&UnifiedAttentionNode::handleSetEnabled, this, std::placeholders::_1, std::placeholders::_2));

    std::string status = attention_enabled_ ? "ENABLED" : "DISABLED";
    std::string default_mode = move_to_default_on_disable_ ? "move to default" : "hold position";
    RCLCPP_INFO(get_logger(), "Improved attention controller ready (%s)", status.c_str());
    RCLCPP_INFO(get_logger(), "Service: /overt_attention/set_enabled (std_srvs/SetBool)");
    RCLCPP_INFO(get_logger(), "Disable mode: %s (yaw=%.1f\xc2\xb0, pitch=%.1f\xc2\xb0)",
                default_mode.c_str(), default_yaw_ * 180.0 / M_PI, default_pitch_ * 180.0 / M_PI);
}

void UnifiedAttentionNode::handleSetEnabled(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                                             std::shared_ptr<std_srvs::srv::SetBool::Response> response) {
    bool old_state = attention_enabled_;
    attention_enabled_ = request->data;

    if (old_state != attention_enabled_) {
        if (attention_enabled_) {
            RCLCPP_INFO(get_logger(), "Attention system ENABLED");
            response->success = true;
            response->message = "Attention system enabled";
        } else {
            // Clear all tracking state
            current_face_id_.reset();
            current_face_location_.reset();
            current_saliency_target_.reset();
            visited_locations_.clear();

            if (move_to_default_on_disable_) {
                moveHeadToDefault();
                RCLCPP_INFO(get_logger(),
                            "Attention system DISABLED - moving to default position (yaw=%.1f\xc2\xb0, pitch=%.1f\xc2\xb0)",
                            default_yaw_ * 180.0 / M_PI, default_pitch_ * 180.0 / M_PI);
                response->success = true;
                response->message =
                    "Attention system disabled - moving to default position (yaw=" +
                    std::to_string(default_yaw_ * 180.0 / M_PI) + "\xc2\xb0, pitch=" +
                    std::to_string(default_pitch_ * 180.0 / M_PI) + "\xc2\xb0)";
            } else {
                double current_yaw = head_yaw_.value_or(0.0);
                double current_pitch = head_pitch_.value_or(0.0);
                RCLCPP_INFO(get_logger(),
                            "Attention system DISABLED - holding current position (yaw=%.1f\xc2\xb0, pitch=%.1f\xc2\xb0)",
                            current_yaw * 180.0 / M_PI, current_pitch * 180.0 / M_PI);
                response->success = true;
                response->message =
                    "Attention system disabled - holding position at yaw=" +
                    std::to_string(current_yaw * 180.0 / M_PI) + "\xc2\xb0, pitch=" +
                    std::to_string(current_pitch * 180.0 / M_PI) + "\xc2\xb0";
            }
        }
    } else {
        std::string status = attention_enabled_ ? "enabled" : "disabled";
        response->success = true;
        response->message = "Attention system already " + status;
    }
}

void UnifiedAttentionNode::onCamInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (!fx_.has_value()) {
        fx_ = msg->k[0];
        fy_ = msg->k[4];
        cx_ = msg->k[2];
        cy_ = msg->k[5];
        RCLCPP_INFO(get_logger(), "Camera: fx=%.1f, fy=%.1f", *fx_, *fy_);
    }
}

void UnifiedAttentionNode::onJointStates(const sensor_msgs::msg::JointState::SharedPtr msg) {
    auto yaw_it = std::find(msg->name.begin(), msg->name.end(), "HeadYaw");
    auto pitch_it = std::find(msg->name.begin(), msg->name.end(), "HeadPitch");

    if (yaw_it != msg->name.end() && pitch_it != msg->name.end()) {
        size_t yaw_idx = std::distance(msg->name.begin(), yaw_it);
        size_t pitch_idx = std::distance(msg->name.begin(), pitch_it);
        if (yaw_idx < msg->position.size() && pitch_idx < msg->position.size()) {
            head_yaw_ = msg->position[yaw_idx];
            head_pitch_ = msg->position[pitch_idx];
        }
    }
}

void UnifiedAttentionNode::moveHeadToDefault() {
    naoqi_bridge_msgs::msg::JointAnglesWithSpeed msg;
    msg.header.stamp = get_clock()->now();
    msg.joint_names = {"HeadYaw", "HeadPitch"};
    msg.joint_angles = {static_cast<float>(default_yaw_), static_cast<float>(default_pitch_)};
    msg.speed = static_cast<float>(default_move_speed_);
    msg.relative = 0;
    pub_head_->publish(msg);

    // Also publish to target topic for visualization (camera-relative, default is centered)
    geometry_msgs::msg::Vector3 target_msg;
    target_msg.x = 0.0;
    target_msg.y = 0.0;
    target_msg.z = 0.0;
    pub_target_->publish(target_msg);
}

double UnifiedAttentionNode::calculateFacePriority(const geometry_msgs::msg::Point& centroid,
                                                    bool mutual_gaze, bool is_current_face) {
    double score = 1.0;

    // Factor 1: Engagement bonus
    if (mutual_gaze) {
        score *= engaged_bonus_;
    }

    // Factor 2: Distance from center
    double dist_from_center = std::sqrt(std::pow(centroid.x - *cx_, 2) + std::pow(centroid.y - *cy_, 2));
    double max_dist = std::sqrt(std::pow(*cx_, 2) + std::pow(*cy_, 2));
    double center_score = 1.0 - (dist_from_center / max_dist);
    score *= (0.5 + 0.5 * center_score);

    // Factor 3: Depth bonus
    if (prefer_closer_ && centroid.z > 0) {
        if (centroid.z <= max_face_distance_) {
            double depth_bonus = 1.5 - (centroid.z / max_face_distance_);
            score *= std::max(0.5, depth_bonus);
        } else {
            score *= 0.3;
        }
    }

    // Factor 4: Continuity bonus
    if (is_current_face) {
        double time_since_switch = get_clock()->now().seconds() - last_face_switch_time_;
        if (time_since_switch < face_switch_cooldown_) {
            score *= 1.5;
        } else {
            score *= 1.1;
        }
    }

    return score;
}

void UnifiedAttentionNode::onFaces(const dec_interfaces::msg::FaceDetection::SharedPtr msg) {
    if (!attention_enabled_) {
        return;
    }

    if (!fx_.has_value() || !head_yaw_.has_value() || !head_pitch_.has_value()) {
        return;
    }

    if (msg->centroids.empty()) {
        if (current_face_id_.has_value()) {
            RCLCPP_INFO(get_logger(), "Lost face: %s", current_face_id_->c_str());
            current_face_id_.reset();
            current_face_location_.reset();
        }
        return;
    }

    double current_time = get_clock()->now().seconds();

    // Build candidate list with priorities
    std::vector<FaceCandidate> candidates;
    for (size_t i = 0; i < msg->centroids.size(); ++i) {
        FaceCandidate cand;
        cand.face_id = (i < msg->face_label_id.size()) ? msg->face_label_id[i] : ("unknown_" + std::to_string(i));
        cand.mutual_gaze = (i < msg->mutual_gaze.size()) ? msg->mutual_gaze[i] : false;
        cand.centroid = msg->centroids[i];
        cand.is_current = current_face_id_.has_value() && (cand.face_id == *current_face_id_);
        cand.priority = calculateFacePriority(cand.centroid, cand.mutual_gaze, cand.is_current);

        auto [cam_yaw, cam_pitch] = pixelToAngles(cand.centroid.x, cand.centroid.y, *fx_, *fy_, *cx_, *cy_);
        cand.world_yaw = cam_yaw + *head_yaw_;
        cand.world_pitch = cam_pitch + *head_pitch_;

        candidates.push_back(cand);
    }

    // Select best face
    const FaceCandidate* best_face = &candidates[0];
    for (const auto& c : candidates) {
        if (c.priority > best_face->priority) {
            best_face = &c;
        }
    }

    // Check if we should switch faces
    bool should_switch = false;
    std::string switch_reason;

    if (!current_face_id_.has_value()) {
        should_switch = true;
        switch_reason = "initial";
    } else if (best_face->is_current) {
        should_switch = true;
        switch_reason = "refresh";
    } else {
        double time_since_switch = current_time - last_face_switch_time_;
        const FaceCandidate* current_face = nullptr;
        for (const auto& c : candidates) {
            if (c.is_current) {
                current_face = &c;
                break;
            }
        }

        if (current_face == nullptr) {
            should_switch = true;
            switch_reason = "lost_current";
        } else if (time_since_switch < face_switch_cooldown_) {
            if (best_face->priority > current_face->priority * 1.5) {
                should_switch = true;
                switch_reason = "much_better";
            }
        } else {
            if (best_face->priority > current_face->priority * 1.1) {
                should_switch = true;
                switch_reason = "better";
            }
        }
    }

    if (!should_switch) {
        last_face_time_ = current_time;
        return;
    }

    // Update state
    bool is_new_face = !current_face_id_.has_value() || (best_face->face_id != *current_face_id_);

    if (is_new_face) {
        last_face_switch_time_ = current_time;
        // Reset smoothed target so we jump immediately to the new face
        target_yaw_.reset();
        target_pitch_.reset();
        RCLCPP_INFO(get_logger(), "Switching to face: %s (engaged=%s, depth=%.2fm, priority=%.2f, reason=%s)",
                    best_face->face_id.c_str(), best_face->mutual_gaze ? "True" : "False",
                    best_face->centroid.z, best_face->priority, switch_reason.c_str());
    }

    current_face_id_ = best_face->face_id;
    current_face_location_ = std::make_pair(best_face->world_yaw, best_face->world_pitch);
    last_face_time_ = current_time;

    // Clamp and publish
    double yaw = clamp(best_face->world_yaw, -face_yaw_lim_, face_yaw_lim_);
    double pitch = clamp(best_face->world_pitch, face_pitch_dn_, face_pitch_up_);

    std::string source = "face[" + best_face->face_id + "]";
    if (best_face->mutual_gaze) {
        source += "_engaged";
    }
    if (!switch_reason.empty()) {
        source += "(" + switch_reason + ")";
    }

    publishHead(yaw, pitch, best_face->priority, source);
}

double UnifiedAttentionNode::calculateIorSuppression(double age_seconds) {
    double decay = std::log(2.0) / ior_half_life_;
    return ior_max_suppression_ * std::exp(-decay * age_seconds);
}

double UnifiedAttentionNode::applyIorFilter(double world_yaw, double world_pitch, double score) {
    if (!enable_ior_ || visited_locations_.empty()) {
        return score;
    }

    double current_time = get_clock()->now().seconds();
    double max_suppression = 0.0;

    for (const auto& loc : visited_locations_) {
        double age = current_time - loc.timestamp;
        double dist = std::sqrt(std::pow(world_yaw - loc.yaw, 2) + std::pow(world_pitch - loc.pitch, 2));

        if (dist < ior_radius_) {
            double time_supp = calculateIorSuppression(age);
            double space_decay = 1.0 - (dist / ior_radius_);
            max_suppression = std::max(max_suppression, time_supp * space_decay);
        }
    }

    return score * (1.0 - max_suppression);
}

void UnifiedAttentionNode::cleanupWeakIor() {
    if (!enable_ior_) {
        return;
    }

    double current_time = get_clock()->now().seconds();
    std::vector<VisitedLocation> kept;
    for (const auto& loc : visited_locations_) {
        if (calculateIorSuppression(current_time - loc.timestamp) >= ior_cleanup_threshold_) {
            kept.push_back(loc);
        }
        if (static_cast<int>(kept.size()) >= ior_max_locations_) {
            break;
        }
    }
    visited_locations_ = std::move(kept);
}

void UnifiedAttentionNode::onSaliency(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    if (!attention_enabled_) {
        return;
    }

    if (!fx_.has_value() || !head_yaw_.has_value()) {
        return;
    }

    if (get_clock()->now().seconds() - last_face_time_ < face_timeout_) {
        return;
    }

    if (current_face_id_.has_value()) {
        RCLCPP_INFO(get_logger(), "No recent faces, switching to saliency (was tracking: %s)",
                    current_face_id_->c_str());
        current_face_id_.reset();
        current_face_location_.reset();

        // Reset smoothed target so the face position doesn't bias saliency EMA
        target_yaw_.reset();
        target_pitch_.reset();

        // If head is outside saliency limits (e.g. from face tracking), pull it back in
        double head_yaw_val = *head_yaw_;
        double head_pitch_val = head_pitch_.value_or(0.0);
        double clamped_yaw = clamp(head_yaw_val, -saliency_yaw_lim_, saliency_yaw_lim_);
        double clamped_pitch = clamp(head_pitch_val, saliency_pitch_dn_, saliency_pitch_up_);
        if (clamped_yaw != head_yaw_val || clamped_pitch != head_pitch_val) {
            RCLCPP_INFO(get_logger(),
                        "Head outside saliency limits (yaw=%.1f\xc2\xb0, pitch=%.1f\xc2\xb0) "
                        "\xe2\x86\x92 returning to (yaw=%.1f\xc2\xb0, pitch=%.1f\xc2\xb0)",
                        head_yaw_val * 180.0 / M_PI, head_pitch_val * 180.0 / M_PI,
                        clamped_yaw * 180.0 / M_PI, clamped_pitch * 180.0 / M_PI);
            publishHead(clamped_yaw, clamped_pitch, 0.0, "saliency_reenter", true);
        }
    }

    if (msg->data.size() < 3) {
        return;
    }

    cleanupWeakIor();

    // Convert all candidates to world angles with IOR
    std::vector<std::tuple<double, double, double>> candidates;
    for (size_t i = 0; i + 2 < msg->data.size(); i += 3) {
        double u = msg->data[i];
        double v = msg->data[i + 1];
        double score = msg->data[i + 2];

        if (score < saliency_min_) {
            continue;
        }

        auto [cam_yaw, cam_pitch] = pixelToAngles(u, v, *fx_, *fy_, *cx_, *cy_);
        double world_yaw = cam_yaw + *head_yaw_;
        double world_pitch = cam_pitch + *head_pitch_;

        // Skip candidates outside saliency joint limits so the head never
        // gets commanded to (and stuck at) a boundary
        if (world_yaw < -saliency_yaw_lim_ || world_yaw > saliency_yaw_lim_ ||
            world_pitch < saliency_pitch_dn_ || world_pitch > saliency_pitch_up_) {
            continue;
        }

        score = applyIorFilter(world_yaw, world_pitch, score);

        if (score >= saliency_min_) {
            candidates.emplace_back(world_yaw, world_pitch, score);
        }
    }

    if (candidates.empty()) {
        return;
    }

    auto best = *std::max_element(candidates.begin(), candidates.end(),
                                   [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
    double best_yaw = std::get<0>(best);
    double best_pitch = std::get<1>(best);
    double best_score = std::get<2>(best);

    // Check if same target as current
    bool is_same = false;
    if (current_saliency_target_.has_value()) {
        double curr_yaw = current_saliency_target_->first;
        double curr_pitch = current_saliency_target_->second;
        double dist = std::sqrt(std::pow(best_yaw - curr_yaw, 2) + std::pow(best_pitch - curr_pitch, 2));
        is_same = dist < same_target_threshold_;
    }

    // Cooldown logic
    double time_on_target = get_clock()->now().seconds() - last_saliency_cmd_time_;

    bool should_switch = false;
    std::string reason;

    if (!current_saliency_target_.has_value()) {
        should_switch = true;
        reason = "initial";
    } else if (is_same) {
        should_switch = true;
        reason = "refresh";
    } else if (time_on_target < min_cooldown_) {
        should_switch = best_score > current_saliency_score_ * switch_ratio_;
        if (should_switch) reason = "early";
    } else if (time_on_target > max_dwell_) {
        should_switch = !is_same;
        if (should_switch) reason = "max_dwell";
    } else {
        should_switch = best_score > current_saliency_score_ * 1.15;
        if (should_switch) reason = "better";
    }

    if (!should_switch) {
        return;
    }

    // Update state
    current_saliency_target_ = std::make_pair(best_yaw, best_pitch);
    current_saliency_score_ = best_score;
    last_saliency_cmd_time_ = get_clock()->now().seconds();

    if (enable_ior_) {
        VisitedLocation loc;
        loc.yaw = best_yaw;
        loc.pitch = best_pitch;
        loc.timestamp = get_clock()->now().seconds();
        visited_locations_.push_back(loc);
    }

    // Clamp and publish
    double yaw = clamp(best_yaw, -saliency_yaw_lim_, saliency_yaw_lim_);
    double pitch = clamp(best_pitch, saliency_pitch_dn_, saliency_pitch_up_);
    publishHead(yaw, pitch, best_score, "saliency(" + reason + ")");
}

void UnifiedAttentionNode::publishHead(double yaw, double pitch, double score,
                                        const std::string& source, bool force) {
    if (force) {
        // Immediate move: reset smoothed target to destination and skip dead-zone
        target_yaw_ = yaw;
        target_pitch_ = pitch;
    } else {
        // Apply exponential moving average smoothing to reduce noise
        if (!target_yaw_.has_value()) {
            target_yaw_ = yaw;
            target_pitch_ = pitch;
        } else {
            double alpha = target_smoothing_alpha_;
            target_yaw_ = alpha * yaw + (1.0 - alpha) * (*target_yaw_);
            target_pitch_ = alpha * pitch + (1.0 - alpha) * (*target_pitch_);
        }

        yaw = *target_yaw_;
        pitch = *target_pitch_;

        // Dead-zone: skip command if head is already close enough to the smoothed target
        if (head_yaw_.has_value() && head_pitch_.has_value()) {
            if (std::abs(yaw - *head_yaw_) < min_angular_change_ &&
                std::abs(pitch - *head_pitch_) < min_angular_change_) {
                return;
            }
        }
    }

    naoqi_bridge_msgs::msg::JointAnglesWithSpeed msg;
    msg.header.stamp = get_clock()->now();
    msg.joint_names = {"HeadYaw", "HeadPitch"};
    msg.joint_angles = {static_cast<float>(yaw), static_cast<float>(pitch)};
    msg.speed = 0.1f;
    msg.relative = 0;
    pub_head_->publish(msg);

    // Publish target for visualization
    double cam_relative_yaw, cam_relative_pitch;
    if (head_yaw_.has_value() && head_pitch_.has_value()) {
        cam_relative_yaw = yaw - *head_yaw_;
        cam_relative_pitch = pitch - *head_pitch_;
    } else {
        cam_relative_yaw = yaw;
        cam_relative_pitch = pitch;
    }

    geometry_msgs::msg::Vector3 target_msg;
    target_msg.x = cam_relative_yaw;
    target_msg.y = cam_relative_pitch;
    target_msg.z = score;
    pub_target_->publish(target_msg);

    RCLCPP_INFO(get_logger(), "[%s] \xe2\x86\x92 yaw=%.1f\xc2\xb0, pitch=%.1f\xc2\xb0, score=%.2f",
                source.c_str(), yaw * 180.0 / M_PI, pitch * 180.0 / M_PI, score);
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<UnifiedAttentionNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("simple_attention"), "Exception: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
