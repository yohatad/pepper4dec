#ifndef SENSOR_TEST_HPP
#define SENSOR_TEST_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <fstream>
#include <string>
#include <unordered_map>
#include <ctime>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <functional>
#ifdef PEPPER_ROBOT
#include <naoqi_driver/msg/audio_custom.hpp>
#endif

using TestFunction = std::function<void(std::shared_ptr<rclcpp::Node>)>;
extern bool output;
extern int timeDuration;

void backSonar(std::shared_ptr<rclcpp::Node> node);
void frontSonar(std::shared_ptr<rclcpp::Node> node);
void frontCamera(std::shared_ptr<rclcpp::Node> node);
void bottomCamera(std::shared_ptr<rclcpp::Node> node);
void depthCamera(std::shared_ptr<rclcpp::Node> node);
void stereoCamera(std::shared_ptr<rclcpp::Node> node);
void laserSensor(std::shared_ptr<rclcpp::Node> node);
void jointState(std::shared_ptr<rclcpp::Node> node);
void odom(std::shared_ptr<rclcpp::Node> node);
void imu(std::shared_ptr<rclcpp::Node> node);
void speech(std::shared_ptr<rclcpp::Node> node);
void realsenseRGBCamera(std::shared_ptr<rclcpp::Node> node);
void realsenseDepthCamera(std::shared_ptr<rclcpp::Node> node);

/* Callback functions executed when sensor data arrives */
void backSonarMessageReceived(const sensor_msgs::msg::Range::SharedPtr msg);
void frontSonarMessageReceived(const sensor_msgs::msg::Range::SharedPtr msg);
void frontCameraMessageReceived(const sensor_msgs::msg::Image::ConstSharedPtr msg);
void bottomCameraMessageReceived(const sensor_msgs::msg::Image::ConstSharedPtr msg);
void depthCameraMessageReceived(const sensor_msgs::msg::Image::ConstSharedPtr msg);
void stereoCameraMessageReceived(const sensor_msgs::msg::Image::ConstSharedPtr msg);
void laserSensorMessageReceived(const sensor_msgs::msg::LaserScan::SharedPtr msg);
void jointStateMessageReceived(const sensor_msgs::msg::JointState::SharedPtr msg);
void odomMessageReceived(const nav_msgs::msg::Odometry::SharedPtr msg);
void imuMessageReceived(const sensor_msgs::msg::Imu::SharedPtr msg);
void realsenseRGBCameraMessageReceived(const sensor_msgs::msg::Image::ConstSharedPtr msg);
void realsenseDepthCameraMessageReceived(const sensor_msgs::msg::Image::ConstSharedPtr msg);

#ifdef PEPPER_ROBOT
void microphone(std::shared_ptr<rclcpp::Node> node);
void microphoneMessageReceived(const naoqi_driver::msg::AudioCustom::SharedPtr msg);
#endif

std::vector<std::string> extractTests(std::string key);
std::string extractTopic(std::string key);   
std::string extractMode();
void writeWavHeader(std::ofstream &file, int sampleRate, int numSamples);
void playAndDeleteFile();
void initializeOutputFile(std::ofstream& out_of, const std::string& path);
std::string getOutputFilePath();
std::string getCurrentTime();
void finalizeOutputFile(std::ofstream& out_of, const std::string& path);
void executeTestsSequentially(const std::vector<std::string>& testNames, std::shared_ptr<rclcpp::Node> node);
void executeTestsInParallel(const std::vector<std::string>& testNames, std::shared_ptr<rclcpp::Node> node);
void switchMicrophoneChannel();

void promptAndExit(int err);
void promptAndContinue();

#endif // SENSOR_TEST_HPP
