// Copyright (C) 2026 Upanzi Network. BSD-3-Clause.
/**
 * Implementation of the world-frame multi-camera people tracker.
 * See include/pepper_social_nav/people_tracker.hpp for the design rationale.
 */

#include "pepper_social_nav/people_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "tf2/utils.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace pepper_social_nav
{

// ---------------------------------------------------------------------------
// Track
// ---------------------------------------------------------------------------

Track::Track(int id, const Observation & obs, double initial_velocity_variance)
: id_(id), last_update_(obs.stamp)
{
  x_ << obs.x, obs.y, 0.0, 0.0;
  P_ = Eigen::Matrix4d::Zero();
  // Position starts confident (we just measured it); velocity starts unknown,
  // which is what lets the first few updates swing it freely.
  P_(0, 0) = P_(1, 1) = 0.25;
  P_(2, 2) = P_(3, 3) = initial_velocity_variance;
}

namespace
{

Eigen::Matrix4d transitionMatrix(double dt)
{
  Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
  F(0, 2) = dt;
  F(1, 3) = dt;
  return F;
}

/** Piecewise-white acceleration noise: people change velocity, not position. */
Eigen::Matrix4d processNoise(double dt, double sigma_a)
{
  const double q = sigma_a * sigma_a;
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;

  Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
  Q(0, 0) = Q(1, 1) = dt4 / 4.0 * q;
  Q(2, 2) = Q(3, 3) = dt2 * q;
  Q(0, 2) = Q(2, 0) = Q(1, 3) = Q(3, 1) = dt3 / 2.0 * q;
  return Q;
}

}  // namespace

void Track::predict(const rclcpp::Time & to, double process_noise_accel)
{
  const double dt = (to - last_update_).seconds();
  if (dt <= 0.0) {
    // Out-of-order measurement (two cameras, different pipeline latencies).
    // Updating in place without rewinding is a small, bounded approximation;
    // a full OOSM retrodiction is not worth the complexity at these dts.
    return;
  }
  x_ = transitionMatrix(dt) * x_;
  const Eigen::Matrix4d F = transitionMatrix(dt);
  P_ = F * P_ * F.transpose() + processNoise(dt, process_noise_accel);
  last_update_ = to;
}

void Track::update(const Observation & obs)
{
  Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
  H(0, 0) = 1.0;
  H(1, 1) = 1.0;

  // Anisotropic measurement noise, built along the line of sight and rotated
  // into the tracking frame. This is what lets a bearing-accurate but
  // range-poor detection (Pepper's front camera close in, where the feet are
  // out of frame) still sharpen the track's DIRECTION without dragging its
  // distance around. An isotropic sigma would have to be as bad as the worst
  // axis, throwing the good axis away.
  const double c = std::cos(obs.bearing);
  const double sn = std::sin(obs.bearing);
  Eigen::Matrix2d Rot;
  Rot << c, -sn,
    sn, c;
  Eigen::Matrix2d D = Eigen::Matrix2d::Zero();
  D(0, 0) = obs.range_stddev * obs.range_stddev;
  D(1, 1) = obs.cross_stddev * obs.cross_stddev;
  const Eigen::Matrix2d R = Rot * D * Rot.transpose();

  const Eigen::Vector2d z(obs.x, obs.y);
  const Eigen::Vector2d y = z - H * x_;
  const Eigen::Matrix2d S = H * P_ * H.transpose() + R;
  const Eigen::Matrix<double, 4, 2> K = P_ * H.transpose() * S.inverse();

  x_ = x_ + K * y;
  P_ = (Eigen::Matrix4d::Identity() - K * H) * P_;
  last_update_ = obs.stamp;
  ++hits_;
}

Eigen::Vector4d Track::stateAt(const rclcpp::Time & to) const
{
  const double dt = (to - last_update_).seconds();
  if (dt <= 0.0) {
    return x_;
  }
  return transitionMatrix(dt) * x_;
}

// ---------------------------------------------------------------------------
// PeopleTracker
// ---------------------------------------------------------------------------

PeopleTracker::PeopleTracker(const rclcpp::NodeOptions & options)
: rclcpp::Node("people_tracker", options)
{
  declareParameters();

  // A generous buffer: when detection runs offboard the TF a measurement needs
  // may be seconds old by the time the detection lands.
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock(), tf2::durationFromSec(30.0));
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  pub_all_ = create_publisher<social_nav_msgs::msg::Pedestrians>("~/pedestrians", 10);
  pub_bystanders_ = create_publisher<social_nav_msgs::msg::Pedestrians>("~/bystanders", 10);
  pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/markers", 1);

  sub_tour_active_ = create_subscription<std_msgs::msg::Bool>(
    "/behavior_controller/tour_active", 10,
    [this](const std_msgs::msg::Bool::SharedPtr msg) {tour_active_ = msg->data;});

  setupSources();

  timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / publish_rate_),
    std::bind(&PeopleTracker::cycle, this));

  RCLCPP_INFO(
    get_logger(), "people_tracker up: %zu camera source(s), tracking frame '%s'",
    sources_.size(), tracking_frame_.c_str());
}

void PeopleTracker::declareParameters()
{
  tracking_frame_ = declare_parameter<std::string>("tracking_frame", "map");
  robot_frame_ = declare_parameter<std::string>("robot_frame", "base_footprint");
  ground_z_ = declare_parameter<double>("ground_z", 0.0);
  association_distance_ = declare_parameter<double>("association_distance", 0.9);
  process_noise_accel_ = declare_parameter<double>("process_noise_accel", 0.6);
  depth_range_stddev_ = declare_parameter<double>("depth_range_stddev", 0.10);
  feet_range_stddev_ = declare_parameter<double>("feet_range_stddev", 0.18);
  head_plane_range_stddev_ =
    declare_parameter<double>("head_plane_range_stddev", 0.55);
  cross_stddev_ = declare_parameter<double>("cross_stddev", 0.04);
  initial_velocity_variance_ = declare_parameter<double>("initial_velocity_variance", 1.0);
  max_track_age_ = declare_parameter<double>("max_track_age", 1.5);
  confirm_hits_ = declare_parameter<int>("confirm_hits", 3);
  max_speed_ = declare_parameter<double>("max_speed", 2.5);
  tf_timeout_ = declare_parameter<double>("tf_timeout", 0.15);
  publish_rate_ = declare_parameter<double>("publish_rate", 15.0);
  border_margin_px_ = declare_parameter<double>("border_margin_px", 6.0);
  audience_radius_ = declare_parameter<double>("audience_radius", 3.0);
  audience_min_bearing_ = declare_parameter<double>("audience_min_bearing", 1.919862);
}

void PeopleTracker::setupSources()
{
  const auto names = declare_parameter<std::vector<std::string>>(
    "sources", std::vector<std::string>{});

  if (names.empty()) {
    RCLCPP_ERROR(
      get_logger(),
      "No camera sources configured. Set the 'sources' parameter -- with none, "
      "this node has nothing to track and every downstream social cost is zero.");
    return;
  }

  for (const auto & name : names) {
    CameraSource src;
    src.name = name;
    src.detections_topic =
      declare_parameter<std::string>("source." + name + ".detections_topic", "");
    src.camera_info_topic =
      declare_parameter<std::string>("source." + name + ".camera_info_topic", "");
    src.has_depth = declare_parameter<bool>("source." + name + ".has_depth", false);
    src.min_confidence = declare_parameter<double>("source." + name + ".min_confidence", 0.5);
    src.min_range = declare_parameter<double>("source." + name + ".min_range", 0.3);
    src.max_range = declare_parameter<double>("source." + name + ".max_range", 8.0);
    src.person_height =
      declare_parameter<double>("source." + name + ".person_height", 1.70);

    if (src.detections_topic.empty() || src.camera_info_topic.empty()) {
      RCLCPP_ERROR(
        get_logger(), "source '%s' is missing detections_topic or camera_info_topic; skipping",
        name.c_str());
      continue;
    }
    sources_.push_back(std::move(src));
  }

  // Subscriptions are created after the vector is fully sized so the index
  // captured by each callback stays valid (push_back would reallocate).
  for (size_t i = 0; i < sources_.size(); ++i) {
    auto & src = sources_[i];

    src.info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
      src.camera_info_topic, rclcpp::SensorDataQoS(),
      [this, i](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        sources_[i].info = msg;
      });

    src.det_sub = create_subscription<dec_interfaces::msg::PersonDetection>(
      src.detections_topic, 10,
      [this, i](const dec_interfaces::msg::PersonDetection::SharedPtr msg) {
        detectionCallback(msg, i);
      });

    RCLCPP_INFO(
      get_logger(), "  source '%s': %s (depth: %s, trusted %.2f-%.2f m)",
      src.name.c_str(), src.detections_topic.c_str(),
      src.has_depth ? "yes" : "no", src.min_range, src.max_range);
  }
}

bool PeopleTracker::projectDetection(
  const CameraSource & src, double u, double v, double box_h,
  double depth, const rclcpp::Time & stamp, Observation & out) const
{
  if (!src.info) {
    return false;
  }
  const auto & K = src.info->k;
  const double fx = K[0], fy = K[4], cx = K[2], cy = K[5];
  if (fx <= 0.0 || fy <= 0.0) {
    return false;
  }

  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_->lookupTransform(
      tracking_frame_, src.info->header.frame_id, stamp,
      tf2::durationFromSec(tf_timeout_));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_DEBUG(
      get_logger(), "TF %s <- %s at detection stamp unavailable: %s",
      tracking_frame_.c_str(), src.info->header.frame_id.c_str(), ex.what());
    return false;
  }

  const Eigen::Quaterniond q(
    tf.transform.rotation.w, tf.transform.rotation.x,
    tf.transform.rotation.y, tf.transform.rotation.z);
  const Eigen::Matrix3d R = q.normalized().toRotationMatrix();
  const Eigen::Vector3d O(
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);

  Eigen::Vector3d point;
  bool have_point = false;
  double range_stddev = head_plane_range_stddev_;

  // ---- 1. Depth, when the camera has it --------------------------------
  // The best option where available. The centroid of a person box lands on
  // torso or legs, which is solid geometry, and the RealSense's depth quality
  // is best exactly where this camera is trusted.
  if (src.has_depth && std::isfinite(depth) && depth > 0.05) {
    const Eigen::Vector3d p_opt((u - cx) / fx * depth, (v - cy) / fy * depth, depth);
    point = R * p_opt + O;
    point.z() = ground_z_;   // people stand on the floor; drop the height estimate
    range_stddev = depth_range_stddev_;
    have_point = true;
  }

  const double img_h = static_cast<double>(src.info->height);
  const double v_feet = v + box_h / 2.0;
  const bool feet_visible = !(img_h > 0.0 && v_feet > img_h - border_margin_px_);

  // ---- 2. Ground plane through the feet --------------------------------
  // No depth needed, and it puts the estimate where the person actually
  // stands. Requires the box bottom to be clear of the image border: if the
  // feet are cropped the ray lands well beyond the person.
  if (!have_point && feet_visible) {
    const Eigen::Vector3d ray_opt((u - cx) / fx, (v_feet - cy) / fy, 1.0);
    const Eigen::Vector3d D = R * ray_opt;
    if (D.z() < -1e-6) {
      const double t = (ground_z_ - O.z()) / D.z();
      if (t > 0.0) {
        point = O + t * D;
        range_stddev = feet_range_stddev_;
        have_point = true;
      }
    }
  }

  // ---- 3. Head plane through the top of the box ------------------------
  // The close-range case for Pepper's front camera. That camera sits at
  // ~1.15 m and, with the head level, its lower frame edge does not reach the
  // floor until ~2.82 m -- and overt_attention's default pitch tilts it UP,
  // pushing that further out. So for exactly the close proximities this camera
  // is meant to cover, the feet are never in frame and method 2 cannot run.
  //
  // Intersecting the ray through the box TOP with z = person_height works
  // there, but the range it yields is WEAK: the camera is only ~0.55 m below a
  // standing head, so ordinary height variation (1.55-1.90 m) swings the range
  // by ~20%. That is not a reason to discard the detection -- the BEARING is
  // still excellent -- so this is admitted with a deliberately large
  // range_stddev and a small cross_stddev, i.e. as a near-bearing-only
  // measurement. The filter then sharpens direction and leaves distance to
  // depth-bearing sources and the motion model.
  if (!have_point) {
    const double v_head = v - box_h / 2.0;
    if (!(img_h > 0.0 && v_head < border_margin_px_)) {
      const Eigen::Vector3d ray_opt((u - cx) / fx, (v_head - cy) / fy, 1.0);
      const Eigen::Vector3d D = R * ray_opt;
      const double plane_z = ground_z_ + src.person_height;
      const double denom = D.z();
      // Ray must RISE toward the head plane: the camera is below head height.
      if (denom > 1e-6) {
        const double t = (plane_z - O.z()) / denom;
        if (t > 0.0) {
          point = O + t * D;
          point.z() = ground_z_;
          range_stddev = head_plane_range_stddev_;
          have_point = true;
        }
      }
    }
  }

  if (!have_point) {
    return false;
  }

  const double dx = point.x() - O.x();
  const double dy = point.y() - O.y();
  const double range = std::hypot(dx, dy);

  // Per-source trust band. This is the close/long division of labour between
  // the two cameras, and it is enforced HERE rather than downstream so an
  // out-of-band detection never enters association at all.
  if (range < src.min_range || range > src.max_range) {
    return false;
  }

  out.bearing = std::atan2(dy, dx);
  // Deprojection error grows with range: one pixel subtends more ground the
  // further out the target is.
  out.range_stddev = range_stddev * std::max(1.0, range);
  out.cross_stddev = cross_stddev_ * std::max(1.0, range);
  out.range = range;

  out.x = point.x();
  out.y = point.y();
  out.stamp = stamp;
  out.source = src.name;
  return true;
}

void PeopleTracker::detectionCallback(
  const dec_interfaces::msg::PersonDetection::SharedPtr msg, size_t source_index)
{
  const auto & src = sources_[source_index];

  // PersonDetection has no header, so the only usable timestamp is the one the
  // CameraInfo carries. Offboard detection makes this matter: using now() here
  // would look up TF at arrival time and place people where they were, plus a
  // whole WiFi round trip.
  const rclcpp::Time stamp = src.info ? rclcpp::Time(src.info->header.stamp) : now();

  const size_t n = msg->centroids.size();
  for (size_t i = 0; i < n; ++i) {
    if (i < msg->confidences.size() && msg->confidences[i] < src.min_confidence) {
      continue;
    }
    const double w = i < msg->width.size() ? msg->width[i] : 0.0;
    const double h = i < msg->height.size() ? msg->height[i] : 0.0;
    if (w <= 0.0 || h <= 0.0) {
      continue;
    }

    Observation obs;
    if (projectDetection(
        src, msg->centroids[i].x, msg->centroids[i].y, h,
        msg->centroids[i].z, stamp, obs))
    {
      pending_.push_back(obs);
    }
  }
}

void PeopleTracker::associateAndUpdate(std::vector<Observation> & observations)
{
  // Greedy nearest-neighbour in metres, closest pair first. At the handful of
  // people a tour guide meets this is equivalent to Hungarian assignment and
  // far easier to reason about when it misbehaves.
  std::vector<bool> obs_used(observations.size(), false);

  struct Pair
  {
    double d;
    size_t obs;
    size_t track;
  };
  std::vector<Pair> pairs;

  for (size_t oi = 0; oi < observations.size(); ++oi) {
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
      // Compare against the track predicted to the observation's own stamp,
      // not its last state, or a late measurement looks like a jump.
      const Eigen::Vector4d s = tracks_[ti]->stateAt(observations[oi].stamp);
      const double d = std::hypot(observations[oi].x - s(0), observations[oi].y - s(1));
      if (d <= association_distance_) {
        pairs.push_back({d, oi, ti});
      }
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair & a, const Pair & b) {return a.d < b.d;});

  std::vector<bool> track_used(tracks_.size(), false);
  for (const auto & p : pairs) {
    if (obs_used[p.obs] || track_used[p.track]) {
      continue;
    }
    tracks_[p.track]->predict(observations[p.obs].stamp, process_noise_accel_);
    tracks_[p.track]->update(observations[p.obs]);
    obs_used[p.obs] = true;
    track_used[p.track] = true;
  }

  for (size_t oi = 0; oi < observations.size(); ++oi) {
    if (!obs_used[oi]) {
      tracks_.push_back(
        std::make_unique<Track>(next_id_++, observations[oi], initial_velocity_variance_));
    }
  }

  for (auto & t : tracks_) {
    if (!t->confirmed() && t->hits() >= confirm_hits_) {
      t->setConfirmed(true);
    }
  }
}

void PeopleTracker::cycle()
{
  std::vector<Observation> observations;
  observations.swap(pending_);

  // Oldest first, so each track's filter advances monotonically.
  std::sort(
    observations.begin(), observations.end(),
    [](const Observation & a, const Observation & b) {return a.stamp < b.stamp;});

  associateAndUpdate(observations);

  const rclcpp::Time t_now = now();
  tracks_.erase(
    std::remove_if(
      tracks_.begin(), tracks_.end(),
      [&](const std::unique_ptr<Track> & t) {
        const double age = (t_now - t->lastUpdate()).seconds();
        // A track that has sprinted off is a deprojection artefact, not a
        // person -- drop it rather than let it drive a phantom social force.
        return age > max_track_age_ || t->speed() > max_speed_;
      }),
    tracks_.end());

  publish();
}

void PeopleTracker::publish()
{
  const rclcpp::Time t_now = now();

  // Where is the robot? Needed only for the audience/bystander split; if TF is
  // unavailable everyone is treated as a bystander, which is the safe default.
  bool have_robot = false;
  double rx = 0.0, ry = 0.0, ryaw = 0.0;
  try {
    const auto tf = tf_buffer_->lookupTransform(
      tracking_frame_, robot_frame_, tf2::TimePointZero);
    rx = tf.transform.translation.x;
    ry = tf.transform.translation.y;
    ryaw = tf2::getYaw(tf.transform.rotation);
    have_robot = true;
  } catch (const tf2::TransformException &) {
    have_robot = false;
  }

  social_nav_msgs::msg::Pedestrians all;
  all.header.stamp = t_now;
  all.header.frame_id = tracking_frame_;
  social_nav_msgs::msg::Pedestrians bystanders = all;

  visualization_msgs::msg::MarkerArray markers;
  visualization_msgs::msg::Marker clear;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(clear);

  int marker_id = 0;
  for (const auto & t : tracks_) {
    if (!t->confirmed()) {
      continue;
    }
    // Predict to now: the filter's last update is as old as the pipeline is
    // slow, and offboard detection makes that hundreds of milliseconds.
    const Eigen::Vector4d s = t->stateAt(t_now);

    social_nav_msgs::msg::Pedestrian ped;
    ped.identifier = std::to_string(t->id());
    ped.pose.x = s(0);
    ped.pose.y = s(1);
    ped.pose.theta = std::atan2(s(3), s(2));
    ped.velocity.x = s(2);
    ped.velocity.y = s(3);
    ped.velocity.theta = 0.0;
    all.pedestrians.push_back(ped);

    // Audience = people following the tour: close, and BEHIND the robot.
    // dec_Tour.xml has Pepper say "follow me", so the group is behind it by
    // design. Vanilla SFM reads that cluster as a large repulsive force and
    // the robot flees its own visitors -- so the audience is excluded from the
    // avoid set while a tour is running.
    bool is_audience = false;
    if (tour_active_ && have_robot) {
      const double dx = s(0) - rx;
      const double dy = s(1) - ry;
      const double dist = std::hypot(dx, dy);
      double bearing = std::atan2(dy, dx) - ryaw;
      while (bearing > M_PI) {bearing -= 2.0 * M_PI;}
      while (bearing < -M_PI) {bearing += 2.0 * M_PI;}
      is_audience = dist <= audience_radius_ && std::fabs(bearing) >= audience_min_bearing_;
    }
    if (!is_audience) {
      bystanders.pedestrians.push_back(ped);
    }

    visualization_msgs::msg::Marker m;
    m.header = all.header;
    m.ns = "pedestrians";
    m.id = marker_id++;
    m.type = visualization_msgs::msg::Marker::CYLINDER;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = s(0);
    m.pose.position.y = s(1);
    m.pose.position.z = ground_z_ + 0.85;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = 0.5;
    m.scale.z = 1.7;
    m.color.a = 0.55f;
    m.color.r = is_audience ? 0.2f : 0.95f;
    m.color.g = is_audience ? 0.8f : 0.35f;
    m.color.b = is_audience ? 0.3f : 0.1f;
    markers.markers.push_back(m);

    visualization_msgs::msg::Marker arrow;
    arrow.header = all.header;
    arrow.ns = "velocity";
    arrow.id = marker_id++;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    geometry_msgs::msg::Point p0, p1;
    p0.x = s(0); p0.y = s(1); p0.z = ground_z_ + 0.1;
    // One second of lookahead -- literally what the SFM critic extrapolates.
    p1.x = s(0) + s(2); p1.y = s(1) + s(3); p1.z = ground_z_ + 0.1;
    arrow.points = {p0, p1};
    arrow.scale.x = 0.06;
    arrow.scale.y = 0.12;
    arrow.color.a = 0.9f;
    arrow.color.r = 1.0f;
    arrow.color.g = 0.9f;
    arrow.color.b = 0.1f;
    markers.markers.push_back(arrow);

    visualization_msgs::msg::Marker label;
    label.header = all.header;
    label.ns = "labels";
    label.id = marker_id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose.position.x = s(0);
    label.pose.position.y = s(1);
    label.pose.position.z = ground_z_ + 1.9;
    label.pose.orientation.w = 1.0;
    label.scale.z = 0.22;
    label.color.a = 1.0f;
    label.color.r = label.color.g = label.color.b = 1.0f;
    label.text = "#" + std::to_string(t->id()) +
      (is_audience ? " (audience)" : "") +
      " " + std::to_string(static_cast<int>(std::hypot(s(2), s(3)) * 100)) + " cm/s";
    markers.markers.push_back(label);
  }

  pub_all_->publish(all);
  pub_bystanders_->publish(bystanders);
  pub_markers_->publish(markers);
}

}  // namespace pepper_social_nav
