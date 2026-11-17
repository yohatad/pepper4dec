/* check_frame_contract.cpp
 *
 * A test executable, run by test/test_frame_contract_bag.py against FAST-LIO
 * and lio_odom_bridge on the benchmark bag; it also works against a live
 * system (build/pepper_slam/check_frame_contract).
 *
 * Asserts the LIO frame contract on a running system: odom is backend-native
 * and opaque, while <level_frame> is gravity-aligned and floor-referenced,
 * identical across backends. With calibration-derived levelling these are
 * exact predictions, so this is a real assertion rather than a smoke test.
 *
 *     build/pepper_slam/check_frame_contract
 *     build/pepper_slam/check_frame_contract --ros-args -p level_frame:=map
 *
 * For `duration` seconds it collects the cloud on cloud_topic in level_frame
 * and base_frame's attitude, then checks:
 *   - the levelling offset (level_frame <-> lio_init z) equals the calibration
 *     prediction, base_frame -> lidar_imu_frame z (or expect_offset if set);
 *   - base_frame's mean roll and pitch in level_frame are within tol_rp;
 *   - the floor (median of the lowest 15% of points) is within tol_floor of 0.
 * Prints a report and exits 0 on PASS, 1 on FAIL.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: November 17, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/exceptions.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include "pepper_slam/transform_utils.h"

namespace
{

constexpr double kRadToDeg = 180.0 / M_PI;

// numpy.percentile with its default linear interpolation.
double percentile(std::vector<double> v, double pct)
{
  std::sort(v.begin(), v.end());
  const double pos = pct / 100.0 * static_cast<double>(v.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (pos - static_cast<double>(lo));
}

double median(std::vector<double> v)
{
  return percentile(std::move(v), 50.0);
}

}  // namespace

class CheckFrameContract : public rclcpp::Node
{
public:
  CheckFrameContract()
  : Node("check_frame_contract")
  {
    level_ = declare_parameter<std::string>("level_frame", "odom");
    base_ = declare_parameter<std::string>("base_frame", "base_footprint");
    // Whichever IMU the estimator runs on: camera_imu_optical_frame for
    // l2_rsimu.yaml / l2lidar_rsimu.yaml.
    imu_frame_ = declare_parameter<std::string>("lidar_imu_frame", "camera_imu_optical_frame");
    const auto cloud_topic = declare_parameter<std::string>(
      "cloud_topic", "/cloud_registered_body");
    // 0.0 (default) derives the prediction from TF, base_frame ->
    // lidar_imu_frame z, so it always matches the IMU actually in use.
    expect_ = declare_parameter<double>("expect_offset", 0.0);
    tol_offset_ = declare_parameter<double>("tol_offset", 0.005);  // m
    tol_rp_ = declare_parameter<double>("tol_rp", 3.0);            // deg
    tol_floor_ = declare_parameter<double>("tol_floor", 0.10);     // m
    duration_ = declare_parameter<double>("duration", 25.0);       // s

    buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    listener_ = std::make_unique<tf2_ros::TransformListener>(*buffer_);
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic, 10, [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        onCloud(*msg);
      });
  }

  double duration() const {return duration_;}

  // Print the report; 0 on PASS, 1 on FAIL.
  int finish()
  {
    std::vector<std::string> fails;
    std::printf("\n===== LIO frame contract =====\n");

    double offset = -1.0;
    for (const auto & [from, to] : {std::pair{level_, std::string("lio_init")},
        std::pair{std::string("lio_init"), level_}})
    {
      try {
        offset = std::abs(
          buffer_->lookupTransform(from, to, tf2::TimePointZero).transform.translation.z);
        break;
      } catch (const tf2::TransformException &) {
      }
    }
    if (offset < 0.0) {
      fails.push_back("no transform between " + level_ + " and odom");
    } else {
      if (expect_ == 0.0) {
        try {
          expect_ = buffer_->lookupTransform(base_, imu_frame_, tf2::TimePointZero)
            .transform.translation.z;
        } catch (const tf2::TransformException & ex) {
          fails.push_back(std::string("no ") + base_ + " -> " + imu_frame_ + ": " + ex.what());
        }
      }
      const bool ok = std::abs(offset - expect_) <= tol_offset_;
      std::printf(
        "  leveling offset      %+.4f m   expect %+.4f  [%s]\n", offset, expect_,
        ok ? "OK" : "FAIL");
      if (!ok) {
        fails.push_back("leveling offset != calibration prediction");
      }
    }

    if (!roll_pitch_.empty()) {
      double r = 0.0, p = 0.0;
      for (const auto & [roll, pitch] : roll_pitch_) {
        r += roll;
        p += pitch;
      }
      r /= static_cast<double>(roll_pitch_.size());
      p /= static_cast<double>(roll_pitch_.size());
      const bool ok = std::abs(r) <= tol_rp_ && std::abs(p) <= tol_rp_;
      std::printf(
        "  base roll/pitch      %+.2f / %+.2f deg   (<= %g)  [%s]\n", r, p, tol_rp_,
        ok ? "OK" : "FAIL");
      if (!ok) {
        fails.push_back("base_frame not level in level_frame (drift or bad leveling)");
      }
    } else {
      fails.push_back("no base_frame samples");
    }

    if (!zs_.empty()) {
      const double cut = percentile(zs_, 15.0);
      std::vector<double> low;
      std::copy_if(
        zs_.begin(), zs_.end(), std::back_inserter(low), [cut](double z) {
          return z < cut;
        });
      const double floor = low.empty() ? cut : median(low);
      const bool ok = std::abs(floor) <= tol_floor_;
      std::printf(
        "  floor plane z        %+.3f m   (|z| <= %g)  [%s]\n", floor, tol_floor_,
        ok ? "OK" : "FAIL");
      if (!ok) {
        fails.push_back("floor not at z=0 (level frame is not floor-referenced)");
      }
    } else {
      fails.push_back("no cloud samples -- is the cloud topic publishing?");
    }

    std::string verdict = "PASS";
    if (!fails.empty()) {
      verdict = "FAIL: " + fails.front();
      for (std::size_t i = 1; i < fails.size(); ++i) {
        verdict += "; " + fails[i];
      }
    }
    std::printf("===== %s =====\n\n", verdict.c_str());
    std::fflush(stdout);
    return fails.empty() ? 0 : 1;
  }

private:
  void onCloud(const sensor_msgs::msg::PointCloud2 & msg)
  {
    geometry_msgs::msg::TransformStamped cloud_tf, base_tf;
    try {
      cloud_tf = buffer_->lookupTransform(level_, msg.header.frame_id, tf2::TimePointZero);
      base_tf = buffer_->lookupTransform(level_, base_, tf2::TimePointZero);
    } catch (const tf2::TransformException &) {
      return;
    }
    const tf2::Transform level_cloud = pepper_slam::toTransform(cloud_tf);

    const std::size_t before = zs_.size();
    sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    for (; x != x.end(); ++x, ++y, ++z) {
      if (!std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z)) {
        continue;
      }
      zs_.push_back((level_cloud * tf2::Vector3(*x, *y, *z)).z());
    }
    if (zs_.size() == before) {
      return;
    }
    const auto & b = base_tf.transform.rotation;
    const double roll = std::atan2(2 * (b.w * b.x + b.y * b.z), 1 - 2 * (b.x * b.x + b.y * b.y));
    const double pitch = std::asin(std::clamp(2 * (b.w * b.y - b.z * b.x), -1.0, 1.0));
    roll_pitch_.emplace_back(roll * kRadToDeg, pitch * kRadToDeg);
  }

  std::string level_, base_, imu_frame_;
  double expect_, tol_offset_, tol_rp_, tol_floor_, duration_;
  std::vector<double> zs_;
  std::vector<std::pair<double, double>> roll_pitch_;
  std::unique_ptr<tf2_ros::Buffer> buffer_;
  std::unique_ptr<tf2_ros::TransformListener> listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CheckFrameContract>();
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::duration<double>(node->duration());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
    executor.spin_once(std::chrono::milliseconds(100));
  }
  const int rc = node->finish();
  rclcpp::shutdown();
  return rc;
}
