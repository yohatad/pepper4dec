// Copyright (C) 2026 Upanzi Network. BSD-3-Clause.
/**
 * MPPI critic scoring trajectories by the social force model's repulsive
 * potential from tracked pedestrians.
 *
 * WHY A POTENTIAL AND NOT A FORCE. MPPI does not integrate forces; it scores
 * sampled trajectories and takes a soft-min. So folding SFM in as an ad-hoc
 * "penalty near people" would throw away the model. Instead this uses the
 * quantity the SFM force is the negative gradient OF. Helbing's pedestrian
 * repulsion
 *
 *     f(d) = A * exp((r - d) / B)
 *
 * integrates to the potential
 *
 *     V(d) = A * B * exp((r - d) / B)
 *
 * so scoring trajectories with V makes the optimum MPPI settles on consistent
 * with the trajectory SFM dynamics would have produced -- while still getting
 * MPPI's constraint handling, obstacle awareness and goal seeking for free.
 * That is the whole reason to embed SFM in the controller rather than bolt a
 * separate SFM velocity node onto the side of Nav2.
 *
 * ANISOTROPY. Moussaid (2009): the repulsion is weighted by
 *     w = lambda + (1 - lambda) * (1 + cos(phi)) / 2
 * with phi the angle between the pedestrian's heading and the direction to the
 * robot. Cutting in front of someone costs more than passing behind them,
 * which is both what people do and what reads as polite from a robot.
 *
 * ANTICIPATION. Plain SFM is purely reactive -- its best-known weakness. Each
 * pedestrian is extrapolated at constant velocity to the time of each
 * trajectory sample, so the robot is scored against where people WILL be. This
 * is the cheap 80% of what a learned trajectory predictor buys, with no model,
 * no GPU and no training data.
 *
 * The critic consumes ~/bystanders, so a tour group following the robot does
 * not repel it (see people_tracker.hpp).
 */

#ifndef PEPPER_SOCIAL_NAV__SOCIAL_FORCE_CRITIC_HPP_
#define PEPPER_SOCIAL_NAV__SOCIAL_FORCE_CRITIC_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "nav2_mppi_controller/critic_function.hpp"
#include "social_nav_msgs/msg/pedestrians.hpp"

namespace pepper_social_nav
{

/** A pedestrian resolved into the costmap frame, ready to extrapolate. */
struct CriticPedestrian
{
  double x{0.0};
  double y{0.0};
  double vx{0.0};
  double vy{0.0};
  double heading{0.0};
};

class SocialForceCritic : public mppi::critics::CriticFunction
{
public:
  void initialize() override;
  void score(mppi::CriticData & data) override;

private:
  void pedestriansCallback(const social_nav_msgs::msg::Pedestrians::SharedPtr msg);

  /** Resolve the latest message into the costmap frame; false if unusable. */
  bool currentPedestrians(std::vector<CriticPedestrian> & out, double & message_age);

  rclcpp::Subscription<social_nav_msgs::msg::Pedestrians>::SharedPtr sub_;
  social_nav_msgs::msg::Pedestrians pedestrians_;
  rclcpp::Time last_msg_time_{0, 0, RCL_ROS_TIME};
  std::mutex mutex_;

  std::string topic_;
  float weight_{12.0f};
  float power_{1.0f};
  double force_a_{2.1};          ///< A: interaction strength (Helbing)
  double force_b_{0.55};         ///< B: interaction range, metres
  double radius_sum_{0.55};      ///< r: robot radius + person radius
  double lambda_{0.35};          ///< anisotropy: 1.0 = isotropic
  double cutoff_{3.0};           ///< ignore people beyond this (metres)
  double message_timeout_{1.0};
  double max_extrapolation_{2.0};///< cap constant-velocity lookahead (seconds)
  bool consider_footprint_{false};
};

}  // namespace pepper_social_nav

#endif  // PEPPER_SOCIAL_NAV__SOCIAL_FORCE_CRITIC_HPP_
