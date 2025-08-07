#ifndef ACTUATORTEST_HPP
#define ACTUATORTEST_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Transform.h>

#include <thread>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <csignal>
#include <memory>

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
using GoalHandle = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;

std::string extractTopic(std::string key);
std::string extractMode();
std::vector<std::string> extractTests(std::string set);
void promptAndExit(int status);
void promptAndContinue();
void signalHandler(int signum);

void moveToPosition(
    rclcpp_action::Client<FollowJointTrajectory>::SharedPtr client,
    const std::vector<std::string>& jointNames,
    double duration,
    const std::string& positionName,
    std::vector<double> positions);

void executeTestsSequentially(
    const std::vector<std::string>& testNames,
    std::shared_ptr<rclcpp::Node> node);

void executeTestsInParallel(
    const std::vector<std::string>& testNames,
    std::shared_ptr<rclcpp::Node> node);

std::vector<std::vector<double>> calculateDuration(
    std::vector<double> homePosition,
    std::vector<double> maxPosition,
    std::vector<double> minPosition,
    std::vector<std::vector<double>> velocity);

void head(std::shared_ptr<rclcpp::Node> node);
void rArm(std::shared_ptr<rclcpp::Node> node);
void lArm(std::shared_ptr<rclcpp::Node> node);
void rHand(std::shared_ptr<rclcpp::Node> node);
void lHand(std::shared_ptr<rclcpp::Node> node);
void leg(std::shared_ptr<rclcpp::Node> node);
void wheels(std::shared_ptr<rclcpp::Node> node);

#endif // ACTUATORTEST_HPP
