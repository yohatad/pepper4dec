/* cloud_range_filter.cpp
 *
 * Filters a scan cloud before costmap/octomap use.
 *
 * Applies, in order:
 *   1. Range cut       -- drop the robot's own body (points within
 *                         min_range..max_range).
 *   2. Outlier removal -- drop isolated spike returns (radius outlier filter).
 *   3. Ground plane    -- RANSAC-fit and remove the near-z=0 plane in
 *                         base_footprint (off by default; needs TF).
 *
 * The original topic is untouched; the filtered cloud is republished as
 * plain x/y/z float32. The filters themselves live in cloud_filters.cpp.
 *
 * C++ because it runs on every scan, twice in the Nav2 stacks (the safety
 * stop's /points_safety and the costmaps' /points_costmap). The Python
 * version spent ~9 ms per scan in the ground fit alone, almost all of it
 * per-iteration interpreter overhead.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: October 11, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 * This software is provided 'as-is' for research and educational purposes
 * within the DEC project.
 */

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>

#include "pepper_slam/cloud_filters.h"

using pepper_slam::Point;
using sensor_msgs::msg::PointCloud2;

class CloudRangeFilter : public rclcpp::Node
{
public:
  CloudRangeFilter()
  : Node("cloud_range_filter"), rng_(std::random_device{}())
  {
    const auto in_topic = declare_parameter<std::string>("input_topic", "/cloud_registered_body");
    const auto out_topic = declare_parameter<std::string>(
      "output_topic", "/cloud_registered_body_filtered");
    min_range_ = declare_parameter("min_range", 0.8);   // drop self-hits closer than this
    max_range_ = declare_parameter("max_range", 0.0);   // 0 = no upper limit
    // Radius-outlier removal. OFF by default (0): the sparse per-scan L2 cloud
    // loses real far returns at neighbourly settings. Loose radius if enabled.
    ror_min_neighbors_ = declare_parameter("ror_min_neighbors", 0);   // 0 disables ROR
    ror_radius_ = declare_parameter("ror_radius", 0.5);               // m

    // Ground-plane removal. OFF by default -- only needed when feeding a
    // fixed-height-band consumer (nav2 costmap voxel_layer); octomap's own
    // /cloud_registered_body path already gets adaptive ground handling
    // from octomap_server's filter_ground_plane, so leave this off there.
    remove_ground_ = declare_parameter("remove_ground_plane", false);
    ground_frame_ = declare_parameter<std::string>("ground_frame", "base_footprint");
    // inlier band around fitted plane
    ground_.distance_thresh = declare_parameter("ground_distance_thresh", 0.05);
    // rad, normal vs +Z tolerance
    ground_.angle_thresh = declare_parameter("ground_angle_thresh", 0.15);
    // fitted plane must sit within this of ground_frame z=0
    ground_.z_thresh = declare_parameter("ground_z_thresh", 0.12);
    ground_.iterations = declare_parameter("ground_ransac_iterations", 60);
    // below this, skip filtering (keep all points)
    ground_.min_points = declare_parameter("ground_min_points", 30);

    if (ror_min_neighbors_ > 0 && ror_radius_ <= 0.0) {
      RCLCPP_ERROR(
        get_logger(), "ror_radius must be > 0 (got %.3f); radius outlier removal disabled.",
        ror_radius_);
      ror_min_neighbors_ = 0;
    }

    if (remove_ground_) {
      tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
      // Constructed on this node (not a private one) so a launch-file
      // /tf -> /tf_nav remap reaches it.
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);
    }

    const auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable().durability_volatile();
    // BEST_EFFORT, not RELIABLE: the L2 driver publishes /points and
    // /imu/data BEST_EFFORT (see dec_robot.launch.py's design notes). A
    // RELIABLE subscriber here silently receives nothing -- MEASURED:
    // /points_safety never published a single message with this on
    // RELIABLE, which starves collision_monitor's entire safety-stop
    // layer of input.
    const auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort().durability_volatile();
    pub_ = create_publisher<PointCloud2>(out_topic, pub_qos);
    sub_ = create_subscription<PointCloud2>(
      in_topic, sub_qos, std::bind(&CloudRangeFilter::filter, this, std::placeholders::_1));

    std::ostringstream summary;
    summary << "cloud_range_filter: " << in_topic << " -> " << out_topic << ", keep "
            << min_range_ << " m <= range";
    if (max_range_ > 0.0) {
      summary << " <= " << max_range_ << " m";
    } else {
      summary << " (no max)";
    }
    if (ror_min_neighbors_ > 0) {
      summary << ", ROR >=" << ror_min_neighbors_ << " nbrs in " << ror_radius_ << " m";
    } else {
      summary << ", ROR off";
    }
    if (remove_ground_) {
      summary << ", ground-plane removal vs " << ground_frame_;
    } else {
      summary << ", ground-plane removal off";
    }
    RCLCPP_INFO(get_logger(), "%s.", summary.str().c_str());
  }

private:
  void filter(const PointCloud2::ConstSharedPtr msg)
  {
    std::vector<Point> points;
    if (!readXyz(*msg, points) || points.empty()) {
      return;
    }

    points = pepper_slam::applyMask(
      points, pepper_slam::rangeKeep(points, min_range_, max_range_));
    if (!points.empty()) {
      points = pepper_slam::applyMask(
        points, pepper_slam::radiusOutlierKeep(points, ror_min_neighbors_, ror_radius_));
    }
    if (remove_ground_ && !points.empty()) {
      const std::size_t before = points.size();
      points = pepper_slam::applyMask(points, groundKeep(points, *msg));
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "ground-plane filter: %zu -> %zu pts (%zu dropped as floor)",
        before, points.size(), before - points.size());
    }
    pub_->publish(toCloud(msg->header, points));
  }

  /// Keep-mask dropping the floor, fit in ground_frame. Fails safe: on a
  /// missing TF or an inconclusive fit, keeps every point (never invents
  /// obstacles, but also never silently strips more than we're confident is
  /// floor).
  pepper_slam::KeepMask groundKeep(const std::vector<Point> & points, const PointCloud2 & msg)
  {
    if (points.size() < 3) {
      return pepper_slam::KeepMask(points.size(), 1);
    }
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = tf_buffer_->lookupTransform(
        ground_frame_, msg.header.frame_id, rclcpp::Time(msg.header.stamp));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "TF %s <- %s unavailable, skipping ground-plane removal for this scan: %s",
        ground_frame_.c_str(), msg.header.frame_id.c_str(), ex.what());
      return pepper_slam::KeepMask(points.size(), 1);
    }

    const auto & r = tf.transform.rotation;
    const auto & t = tf.transform.translation;
    tf2::Quaternion q(r.x, r.y, r.z, r.w);
    if (q.length2() < 1e-12) {
      q = tf2::Quaternion::getIdentity();         // degenerate quaternion
    }
    const tf2::Matrix3x3 rot(q);
    const tf2::Vector3 shift(t.x, t.y, t.z);
    std::vector<Point> in_ground;
    in_ground.reserve(points.size());
    for (const Point & p : points) {
      const tf2::Vector3 g = rot * tf2::Vector3(p.x, p.y, p.z) + shift;
      in_ground.push_back({g.x(), g.y(), g.z()});
    }
    return pepper_slam::groundPlaneKeep(in_ground, ground_, rng_);
  }

  /// The x/y/z of every point without a NaN coordinate. False (and an error)
  /// when the cloud has no float32 x/y/z fields.
  bool readXyz(const PointCloud2 & msg, std::vector<Point> & points)
  {
    for (const char * name : {"x", "y", "z"}) {
      bool ok = false;
      for (const auto & f : msg.fields) {
        ok = ok || (f.name == name && f.datatype == sensor_msgs::msg::PointField::FLOAT32);
      }
      if (!ok) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "%s has no float32 '%s' field; dropping the cloud.",
          sub_->get_topic_name(), name);
        return false;
      }
    }
    const std::size_t n = static_cast<std::size_t>(msg.width) * msg.height;
    points.reserve(n);
    sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
    for (std::size_t i = 0; i < n; ++i, ++x, ++y, ++z) {
      if (!std::isnan(*x) && !std::isnan(*y) && !std::isnan(*z)) {
        points.push_back({*x, *y, *z});
      }
    }
    return true;
  }

  static std::unique_ptr<PointCloud2> toCloud(
    const std_msgs::msg::Header & header, const std::vector<Point> & points)
  {
    auto out = std::make_unique<PointCloud2>();
    out->header = header;
    sensor_msgs::PointCloud2Modifier modifier(*out);
    // Three named fields, not "xyz": that shorthand pads each point to 16 bytes.
    modifier.setPointCloud2Fields(
      3, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32);
    modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x(*out, "x"), y(*out, "y"), z(*out, "z");
    for (const Point & p : points) {
      *x = static_cast<float>(p.x);
      *y = static_cast<float>(p.y);
      *z = static_cast<float>(p.z);
      ++x;
      ++y;
      ++z;
    }
    return out;
  }

  double min_range_;
  double max_range_;
  int ror_min_neighbors_;
  double ror_radius_;
  bool remove_ground_;
  std::string ground_frame_;
  pepper_slam::GroundPlaneParams ground_;
  std::mt19937 rng_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CloudRangeFilter>());
  rclcpp::shutdown();
  return 0;
}
