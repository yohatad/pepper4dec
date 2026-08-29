// Copyright (C) 2026 Upanzi Network. BSD-3-Clause.
/** Implementation of the social force MPPI critic. */

#include "pepper_social_nav/social_force_critic.hpp"

#include <algorithm>
#include <cmath>

#include "pluginlib/class_list_macros.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace pepper_social_nav
{

void SocialForceCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(topic_, "topic", std::string("/people_tracker/bystanders"));
  getParam(weight_, "cost_weight", 12.0f);
  getParam(power_, "cost_power", 1.0f);
  getParam(force_a_, "force_a", 2.1);
  getParam(force_b_, "force_b", 0.55);
  getParam(radius_sum_, "radius_sum", 0.55);
  getParam(lambda_, "lambda", 0.35);
  getParam(cutoff_, "cutoff", 3.0);
  getParam(message_timeout_, "message_timeout", 1.0);
  getParam(max_extrapolation_, "max_extrapolation", 2.0);
  getParam(consider_footprint_, "consider_footprint", false);

  auto node = parent_.lock();
  sub_ = node->create_subscription<social_nav_msgs::msg::Pedestrians>(
    topic_, rclcpp::SensorDataQoS(),
    std::bind(&SocialForceCritic::pedestriansCallback, this, std::placeholders::_1));
  last_msg_time_ = node->now();

  RCLCPP_INFO(
    logger_,
    "SocialForceCritic: topic %s, A=%.2f B=%.2f r=%.2f lambda=%.2f weight=%.1f",
    topic_.c_str(), force_a_, force_b_, radius_sum_, lambda_, weight_);
}

void SocialForceCritic::pedestriansCallback(
  const social_nav_msgs::msg::Pedestrians::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  pedestrians_ = *msg;
  auto node = parent_.lock();
  last_msg_time_ = node ? node->now() : last_msg_time_;
}

bool SocialForceCritic::currentPedestrians(
  std::vector<CriticPedestrian> & out, double & message_age)
{
  social_nav_msgs::msg::Pedestrians snapshot;
  rclcpp::Time stamp;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot = pedestrians_;
    stamp = last_msg_time_;
  }
  if (snapshot.pedestrians.empty()) {
    return false;
  }

  auto node = parent_.lock();
  if (!node) {
    return false;
  }

  // Offboard perception means the link can drop. Stale people must stop
  // steering the robot -- the lidar-driven collision monitor stays authoritative.
  message_age = (node->now() - stamp).seconds();
  if (message_timeout_ > 0.0 && message_age > message_timeout_) {
    RCLCPP_WARN_THROTTLE(
      logger_, *node->get_clock(), 3000,
      "SocialForceCritic: pedestrians %.2f s stale (timeout %.2f s) -- "
      "scoring without social costs this cycle.", message_age, message_timeout_);
    return false;
  }

  // MPPI rolls trajectories out in the COSTMAP frame, which for the local
  // costmap is odom; the tracker publishes in map. Transform rather than
  // silently score against positions in the wrong frame.
  const std::string costmap_frame = costmap_ros_->getGlobalFrameID();
  const std::string & ped_frame = snapshot.header.frame_id;

  double tx = 0.0, ty = 0.0, ctheta = 1.0, stheta = 0.0;
  if (!ped_frame.empty() && ped_frame != costmap_frame) {
    try {
      const auto tf = costmap_ros_->getTfBuffer()->lookupTransform(
        costmap_frame, ped_frame, tf2::TimePointZero);
      tx = tf.transform.translation.x;
      ty = tf.transform.translation.y;
      const double yaw = tf2::getYaw(tf.transform.rotation);
      ctheta = std::cos(yaw);
      stheta = std::sin(yaw);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        logger_, *node->get_clock(), 3000,
        "SocialForceCritic: no transform %s <- %s (%s)",
        costmap_frame.c_str(), ped_frame.c_str(), ex.what());
      return false;
    }
  }

  out.clear();
  out.reserve(snapshot.pedestrians.size());
  for (const auto & p : snapshot.pedestrians) {
    CriticPedestrian cp;
    cp.x = tx + ctheta * p.pose.x - stheta * p.pose.y;
    cp.y = ty + stheta * p.pose.x + ctheta * p.pose.y;
    cp.vx = ctheta * p.velocity.x - stheta * p.velocity.y;
    cp.vy = stheta * p.velocity.x + ctheta * p.velocity.y;
    cp.heading = std::hypot(cp.vx, cp.vy) > 1e-3
      ? std::atan2(cp.vy, cp.vx)
      : p.pose.theta + std::atan2(stheta, ctheta);
    out.push_back(cp);
  }
  return !out.empty();
}

void SocialForceCritic::score(mppi::CriticData & data)
{
  if (!enabled_) {
    return;
  }

  std::vector<CriticPedestrian> people;
  double message_age = 0.0;
  if (!currentPedestrians(people, message_age)) {
    return;
  }

  const auto & traj_x = data.trajectories.x;
  const auto & traj_y = data.trajectories.y;
  const size_t batch_size = traj_x.shape(0);
  const size_t time_steps = traj_x.shape(1);
  if (batch_size == 0 || time_steps == 0) {
    return;
  }

  const double dt = static_cast<double>(data.model_dt);

  // Pre-extrapolate every pedestrian to every timestep ONCE, instead of inside
  // the batch loop. With ~2000 batches this is the difference between a few
  // hundred position updates and a few million.
  //
  // The extrapolation starts at message_age, not zero: the sample is already
  // that old by the time it reaches us, and offboard detection makes that
  // hundreds of milliseconds rather than a rounding error.
  const size_t n_people = people.size();
  std::vector<double> px(n_people * time_steps);
  std::vector<double> py(n_people * time_steps);

  for (size_t p = 0; p < n_people; ++p) {
    for (size_t k = 0; k < time_steps; ++k) {
      const double t = std::min(message_age + static_cast<double>(k) * dt, max_extrapolation_);
      px[p * time_steps + k] = people[p].x + people[p].vx * t;
      py[p * time_steps + k] = people[p].y + people[p].vy * t;
    }
  }

  const double cutoff_sq = cutoff_ * cutoff_;
  const double inv_b = 1.0 / force_b_;
  const double scale = force_a_ * force_b_;   // A*B, the potential's amplitude

  xt::xtensor<float, 1> repulsion = xt::zeros<float>({batch_size});

  for (size_t i = 0; i < batch_size; ++i) {
    double acc = 0.0;
    for (size_t k = 0; k < time_steps; ++k) {
      const double rx = static_cast<double>(traj_x(i, k));
      const double ry = static_cast<double>(traj_y(i, k));

      for (size_t p = 0; p < n_people; ++p) {
        const double dx = rx - px[p * time_steps + k];
        const double dy = ry - py[p * time_steps + k];
        const double d_sq = dx * dx + dy * dy;

        // Cheap rejection before any exp(). Most sampled points are nowhere
        // near a person, so this is what keeps the critic affordable.
        if (d_sq > cutoff_sq) {
          continue;
        }

        const double d = std::sqrt(d_sq);
        double v = scale * std::exp((radius_sum_ - d) * inv_b);

        // Anisotropy: weight by where the robot sits relative to the
        // pedestrian's own heading. cos(phi) = 1 directly ahead of them.
        if (d > 1e-6) {
          const double cos_phi =
            (std::cos(people[p].heading) * dx + std::sin(people[p].heading) * dy) / d;
          v *= lambda_ + (1.0 - lambda_) * (1.0 + cos_phi) * 0.5;
        }
        acc += v;
      }
    }
    // Mean over the horizon, so the score does not change meaning when
    // time_steps is retuned.
    repulsion(i) = static_cast<float>(acc / static_cast<double>(time_steps));
  }

  if (power_ != 1.0f) {
    data.costs += xt::pow(repulsion * weight_, power_);
  } else {
    data.costs += repulsion * weight_;
  }
}

}  // namespace pepper_social_nav

PLUGINLIB_EXPORT_CLASS(pepper_social_nav::SocialForceCritic, mppi::critics::CriticFunction)
