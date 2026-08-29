// Copyright (C) 2026 Upanzi Network. BSD-3-Clause.
/**
 * Costmap layer that paints personal space around tracked pedestrians.
 *
 * WHAT THIS IS *NOT*. It does not put people into the costmap -- the L2 lidar
 * already does that through the voxel layer, and does it better. What it adds
 * is the SOCIAL MARGIN around a person: the region that is physically free but
 * that a robot should not drive through. Without it the planner is perfectly
 * happy to shave past a visitor's elbow.
 *
 * WHERE IT EARNS ITS KEEP. Mostly the GLOBAL costmap, where it makes the
 * planner route around a group instead of aiming through it and leaving the
 * controller to sort out the mess. In the local costmap it mainly stops the
 * controller cutting corners around someone.
 *
 * The cost is an asymmetric Gaussian: wider ahead of the person than behind,
 * because walking into someone's path is more of an intrusion than passing
 * behind them. This is the costmap-side mirror of the anisotropy in the SFM
 * critic. Consumes ~/bystanders, not ~/pedestrians: while a tour is running
 * the visitors following Pepper are not obstacles to be routed around.
 */

#ifndef PEPPER_SOCIAL_NAV__PROXEMIC_LAYER_HPP_
#define PEPPER_SOCIAL_NAV__PROXEMIC_LAYER_HPP_

#include <mutex>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "social_nav_msgs/msg/pedestrians.hpp"
#include "tf2_ros/buffer.h"

namespace pepper_social_nav
{

class ProxemicLayer : public nav2_costmap_2d::Layer
{
public:
  ProxemicLayer() = default;

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;

  void reset() override;
  bool isClearable() override {return false;}

private:
  void pedestriansCallback(const social_nav_msgs::msg::Pedestrians::SharedPtr msg);

  rclcpp::Subscription<social_nav_msgs::msg::Pedestrians>::SharedPtr sub_;
  social_nav_msgs::msg::Pedestrians pedestrians_;
  rclcpp::Time last_msg_time_{0, 0, RCL_ROS_TIME};
  std::mutex mutex_;

  std::string topic_;
  double amplitude_{200.0};
  double sigma_{0.55};          ///< lateral / behind spread (m)
  double sigma_front_{1.1};     ///< spread along the direction of travel (m)
  double cutoff_{2.5};          ///< beyond this many metres, contribute nothing
  double message_timeout_{1.0}; ///< stale people are dropped, not frozen in place
  bool use_velocity_scaling_{true};
};

}  // namespace pepper_social_nav

#endif  // PEPPER_SOCIAL_NAV__PROXEMIC_LAYER_HPP_
