// Copyright (C) 2026 Upanzi Network. BSD-3-Clause.
/** Implementation of the proxemic costmap layer. */

#include "pepper_social_nav/proxemic_layer.hpp"

#include <algorithm>
#include <cmath>

#include "nav2_costmap_2d/costmap_math.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace pepper_social_nav
{

using nav2_costmap_2d::LETHAL_OBSTACLE;
using nav2_costmap_2d::NO_INFORMATION;

void ProxemicLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"ProxemicLayer: failed to lock node"};
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("topic", rclcpp::ParameterValue(std::string("/people_tracker/bystanders")));
  declareParameter("amplitude", rclcpp::ParameterValue(200.0));
  declareParameter("sigma", rclcpp::ParameterValue(0.55));
  declareParameter("sigma_front", rclcpp::ParameterValue(1.1));
  declareParameter("cutoff", rclcpp::ParameterValue(2.5));
  declareParameter("message_timeout", rclcpp::ParameterValue(1.0));
  declareParameter("use_velocity_scaling", rclcpp::ParameterValue(true));

  node->get_parameter(name_ + "." + "enabled", enabled_);
  node->get_parameter(name_ + "." + "topic", topic_);
  node->get_parameter(name_ + "." + "amplitude", amplitude_);
  node->get_parameter(name_ + "." + "sigma", sigma_);
  node->get_parameter(name_ + "." + "sigma_front", sigma_front_);
  node->get_parameter(name_ + "." + "cutoff", cutoff_);
  node->get_parameter(name_ + "." + "message_timeout", message_timeout_);
  node->get_parameter(name_ + "." + "use_velocity_scaling", use_velocity_scaling_);

  sub_ = node->create_subscription<social_nav_msgs::msg::Pedestrians>(
    topic_, rclcpp::SensorDataQoS(),
    std::bind(&ProxemicLayer::pedestriansCallback, this, std::placeholders::_1));

  last_msg_time_ = node->now();
  current_ = true;

  RCLCPP_INFO(
    logger_, "ProxemicLayer '%s' subscribed to %s (sigma %.2f / front %.2f, cutoff %.2f)",
    name_.c_str(), topic_.c_str(), sigma_, sigma_front_, cutoff_);
}

void ProxemicLayer::pedestriansCallback(
  const social_nav_msgs::msg::Pedestrians::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  pedestrians_ = *msg;
  auto node = node_.lock();
  last_msg_time_ = node ? node->now() : last_msg_time_;
}

void ProxemicLayer::updateBounds(
  double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto & p : pedestrians_.pedestrians) {
    *min_x = std::min(*min_x, p.pose.x - cutoff_);
    *min_y = std::min(*min_y, p.pose.y - cutoff_);
    *max_x = std::max(*max_x, p.pose.x + cutoff_);
    *max_y = std::max(*max_y, p.pose.y + cutoff_);
  }
}

void ProxemicLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto node = node_.lock();
  if (node && message_timeout_ > 0.0) {
    // Perception may be running offboard over WiFi. If the link drops, the last
    // known people must NOT stay painted on the costmap forever -- the robot
    // would swerve around ghosts. Go blind rather than go wrong; the on-robot
    // collision monitor remains the real safety net either way.
    if ((node->now() - last_msg_time_).seconds() > message_timeout_) {
      return;
    }
  }

  if (pedestrians_.pedestrians.empty()) {
    return;
  }

  // The tracker publishes in its tracking frame (map). A costmap rolling in
  // odom would need these transformed; rather than silently paint costs in the
  // wrong place, say so and skip.
  const std::string & costmap_frame = layered_costmap_->getGlobalFrameID();
  if (!pedestrians_.header.frame_id.empty() &&
    pedestrians_.header.frame_id != costmap_frame)
  {
    RCLCPP_WARN_THROTTLE(
      logger_, *node->get_clock(), 5000,
      "ProxemicLayer: pedestrians are in '%s' but this costmap is '%s'. "
      "Set the tracker's tracking_frame to match, or add a transform step.",
      pedestrians_.header.frame_id.c_str(), costmap_frame.c_str());
    return;
  }

  const double resolution = master_grid.getResolution();

  for (const auto & p : pedestrians_.pedestrians) {
    const double speed = std::hypot(p.velocity.x, p.velocity.y);
    const double heading = speed > 1e-3 ? std::atan2(p.velocity.y, p.velocity.x) : p.pose.theta;

    // A walking person needs more room ahead of them than a standing one.
    // Standing still, the field collapses to an isotropic bubble.
    const double front = use_velocity_scaling_
      ? sigma_ + (sigma_front_ - sigma_) * std::min(1.0, speed / 1.2)
      : sigma_front_;

    unsigned int mx0, my0, mx1, my1;
    if (!master_grid.worldToMap(p.pose.x - cutoff_, p.pose.y - cutoff_, mx0, my0)) {
      continue;
    }
    if (!master_grid.worldToMap(p.pose.x + cutoff_, p.pose.y + cutoff_, mx1, my1)) {
      continue;
    }

    const int i0 = std::max(min_i, static_cast<int>(mx0));
    const int j0 = std::max(min_j, static_cast<int>(my0));
    const int i1 = std::min(max_i, static_cast<int>(mx1));
    const int j1 = std::min(max_j, static_cast<int>(my1));

    for (int j = j0; j < j1; ++j) {
      for (int i = i0; i < i1; ++i) {
        double wx, wy;
        master_grid.mapToWorld(static_cast<unsigned int>(i), static_cast<unsigned int>(j), wx, wy);

        const double dx = wx - p.pose.x;
        const double dy = wy - p.pose.y;
        if (std::hypot(dx, dy) > cutoff_) {
          continue;
        }

        // Rotate into the pedestrian's own frame so the asymmetry follows them.
        const double c = std::cos(-heading);
        const double s = std::sin(-heading);
        const double fx = dx * c - dy * s;   // along direction of travel
        const double fy = dx * s + dy * c;   // lateral

        const double sx = fx >= 0.0 ? front : sigma_;
        const double cost = amplitude_ *
          std::exp(-((fx * fx) / (2.0 * sx * sx) + (fy * fy) / (2.0 * sigma_ * sigma_)));

        if (cost < 1.0) {
          continue;
        }

        const unsigned char old_cost = master_grid.getCost(
          static_cast<unsigned int>(i), static_cast<unsigned int>(j));
        if (old_cost == NO_INFORMATION) {
          continue;
        }

        // Never LETHAL. A person is a cost to be preferred against, not a wall:
        // marking them lethal makes the planner declare failure when someone
        // stands in a doorway, and recovery behaviours are worse for everyone
        // than simply waiting or squeezing past politely.
        const auto capped = static_cast<unsigned char>(
          std::min<double>(LETHAL_OBSTACLE - 1, cost));
        master_grid.setCost(
          static_cast<unsigned int>(i), static_cast<unsigned int>(j),
          std::max(old_cost, capped));
      }
    }
    (void)resolution;
  }
}

void ProxemicLayer::reset()
{
  std::lock_guard<std::mutex> lock(mutex_);
  pedestrians_.pedestrians.clear();
}

}  // namespace pepper_social_nav

PLUGINLIB_EXPORT_CLASS(pepper_social_nav::ProxemicLayer, nav2_costmap_2d::Layer)
