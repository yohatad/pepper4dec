// Copyright (C) 2026 Upanzi Network. BSD-3-Clause.
/** Standalone entry point for the world-frame people tracker. */

#include <memory>

#include "pepper_social_nav/people_tracker.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pepper_social_nav::PeopleTracker>());
  rclcpp::shutdown();
  return 0;
}
