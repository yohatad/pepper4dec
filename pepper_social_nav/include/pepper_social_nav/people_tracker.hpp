// Copyright (C) 2026 Upanzi Network. BSD-3-Clause.
/**
 * World-frame multi-camera people tracker.
 *
 * Turns person_detection's PIXEL detections (PersonDetection.msg carries u, v
 * and an optional depth) into metric, tracked, world-frame pedestrians with
 * velocity -- the input a social force model actually needs.
 *
 * WHY WORLD-FRAME. person_detection associates with ByteTrack in IMAGE space.
 * overt_attention pans Pepper's head continuously, so a fast pan moves every
 * bounding box far enough that IoU matching fails and the track is reborn with
 * a fresh ID and ZERO velocity -- exactly when the robot is closest to someone.
 * Associating in metres instead makes head motion a non-event: TF factors the
 * rotation out, so the person's world position is continuous through the pan.
 * This is also why no appearance/ReID model is needed here (see README).
 *
 * MULTI-CAMERA. Sources are configured as a list, and association happens in
 * the tracking frame, so two cameras with completely different mounts fuse for
 * free. Two are expected on Pepper:
 *   - RealSense D435, body-mounted at z=0.308 m with the optical axis pitched
 *     DOWN 2.31 deg (computed from pepper_slam's sensor_rig.xacro). With the
 *     42.5 deg colour FOV it sees up to 0.308 + 0.343*range metres of height:
 *     waist at 2 m, chest at 3 m, first full body at ~4.5 m. In the 1-3 m
 *     social zone it sees LEGS AND TORSO ONLY. Has depth.
 *   - Pepper's front camera (kTopCamera, /camera/front/image_raw), head-mounted
 *     at ~1.2 m and panning with HeadYaw/HeadPitch. Frames whole people at
 *     social range, which the RealSense cannot. Has NO depth.
 *
 * RANGING. Because one camera has no depth and the other frames people
 * partially, the primary range estimate is a GROUND-PLANE INTERSECTION of the
 * ray through the bounding box's bottom-centre pixel (the feet) with z =
 * ground_z. That works identically for both cameras, needs no depth, and puts
 * the estimate where the person actually stands. Depth, when present and
 * plausible, refines it. A box whose bottom edge touches the image border is
 * rejected: the feet are out of frame, so the ray does not hit the floor where
 * the person is.
 *
 * LATENCY. Detection may run offboard on a laptop over WiFi (see
 * perception_offboard.launch.py), so measurements arrive 100-300 ms late. The
 * filter is updated at the MEASUREMENT stamp, never at arrival time, and the
 * published state is predicted forward to now. Without that the whole pipeline
 * reports where people were, not where they are.
 */

#ifndef PEPPER_SOCIAL_NAV__PEOPLE_TRACKER_HPP_
#define PEPPER_SOCIAL_NAV__PEOPLE_TRACKER_HPP_

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "std_msgs/msg/bool.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

#include "dec_interfaces/msg/person_detection.hpp"
#include "social_nav_msgs/msg/pedestrians.hpp"

namespace pepper_social_nav
{

/**
 * One ground-plane observation in the tracking frame, with an ANISOTROPIC
 * uncertainty.
 *
 * A camera measures bearing far better than it measures range, and how much
 * better depends on which ranging method produced the point. Collapsing that
 * into one isotropic sigma throws away the useful half: a front-camera
 * detection whose range is a guess still pins the direction to about a degree,
 * and fusing it as if both axes were equally bad drags a good track sideways.
 *
 * So the covariance is built in the sensor's own frame -- `range_stddev` along
 * the line of sight, `cross_stddev` across it -- and rotated into the tracking
 * frame by `bearing`.
 */
struct Observation
{
  double x{0.0};
  double y{0.0};
  double range{0.0};          ///< range from the camera to the point
  double bearing{0.0};        ///< camera->point direction in the tracking frame
  double range_stddev{0.2};   ///< uncertainty ALONG the line of sight
  double cross_stddev{0.05};  ///< uncertainty ACROSS it
  rclcpp::Time stamp;
  std::string source;
};

/** Per-camera configuration and runtime state. */
struct CameraSource
{
  std::string name;
  std::string detections_topic;
  std::string camera_info_topic;
  bool has_depth{false};
  double min_confidence{0.5};

  /**
   * Range band this camera is TRUSTED over. The two cameras are deliberately
   * given different bands rather than both running wide open:
   *
   *   front      Close work. VGA 640x480 through a 55.2x44.3 deg lens at
   *              ~1.15 m -- it frames a whole person at conversational range,
   *              which the RealSense physically cannot. Falls off at distance
   *              because VGA runs out of pixels on a small target.
   *   realsense  Long work. Higher resolution and better optics, and its
   *              framing only improves with range (full body past ~4.5 m),
   *              which is exactly the opposite of the front camera.
   *
   * Overlap in the middle is intended: that band is where a handover happens,
   * and world-frame association makes it a continuation of one track rather
   * than a new one.
   */
  double min_range{0.3};
  double max_range{8.0};

  /**
   * Assumed standing height, for the head-plane fallback below. Only used when
   * the feet are out of frame and there is no depth.
   */
  double person_height{1.70};

  sensor_msgs::msg::CameraInfo::SharedPtr info;
  rclcpp::Subscription<dec_interfaces::msg::PersonDetection>::SharedPtr det_sub;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub;
};

/**
 * Constant-velocity Kalman track, state [x, y, vx, vy] in the tracking frame.
 *
 * Constant velocity is deliberate. It is a famously strong short-horizon
 * pedestrian predictor, it costs nothing, and it gives the SFM critic the
 * lookahead that plain reactive SFM lacks. A learned predictor only pays off
 * once the tracks feeding it are clean, which is a later problem than this one.
 */
class Track
{
public:
  Track(int id, const Observation & obs, double initial_velocity_variance);

  void predict(const rclcpp::Time & to, double process_noise_accel);
  void update(const Observation & obs);

  /** State predicted to `to` WITHOUT mutating the filter (for publishing). */
  Eigen::Vector4d stateAt(const rclcpp::Time & to) const;

  int id() const {return id_;}
  const rclcpp::Time & lastUpdate() const {return last_update_;}
  int hits() const {return hits_;}
  bool confirmed() const {return confirmed_;}
  void setConfirmed(bool c) {confirmed_ = c;}
  double x() const {return x_(0);}
  double y() const {return x_(1);}
  double speed() const {return std::hypot(x_(2), x_(3));}

private:
  int id_;
  Eigen::Vector4d x_;
  Eigen::Matrix4d P_;
  rclcpp::Time last_update_;
  int hits_{1};
  bool confirmed_{false};
};

class PeopleTracker : public rclcpp::Node
{
public:
  explicit PeopleTracker(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void declareParameters();
  void setupSources();
  void detectionCallback(const dec_interfaces::msg::PersonDetection::SharedPtr msg, size_t source_index);
  void cycle();

  /**
   * Project one detection to a ground-plane point in the tracking frame.
   * Returns false when the geometry is unusable (feet out of frame, ray not
   * descending, range implausible, TF unavailable at the detection stamp).
   */
  bool projectDetection(
    const CameraSource & src, double u, double v, double box_h,
    double depth, const rclcpp::Time & stamp, Observation & out) const;

  void associateAndUpdate(std::vector<Observation> & observations);
  void publish();

  // --- parameters -----------------------------------------------------------
  std::string tracking_frame_;
  std::string robot_frame_;
  double ground_z_{0.0};
  double association_distance_{0.9};
  double process_noise_accel_{0.6};
  // Baseline sensor noise, scaled per-observation by range and by which
  // ranging method actually produced the point.
  double depth_range_stddev_{0.10};
  double feet_range_stddev_{0.18};
  double head_plane_range_stddev_{0.55};
  double cross_stddev_{0.04};
  double initial_velocity_variance_{1.0};
  double max_track_age_{1.5};
  int confirm_hits_{3};
  double max_speed_{2.5};
  double tf_timeout_{0.15};
  double publish_rate_{15.0};
  double border_margin_px_{6.0};

  // audience vs bystander
  double audience_radius_{3.0};
  double audience_min_bearing_{1.919862};   ///< rad from robot heading (110 deg)
  bool tour_active_{false};

  std::vector<CameraSource> sources_;
  std::vector<std::unique_ptr<Track>> tracks_;
  std::vector<Observation> pending_;
  int next_id_{1};

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<social_nav_msgs::msg::Pedestrians>::SharedPtr pub_all_;
  rclcpp::Publisher<social_nav_msgs::msg::Pedestrians>::SharedPtr pub_bystanders_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_tour_active_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace pepper_social_nav

#endif  // PEPPER_SOCIAL_NAV__PEOPLE_TRACKER_HPP_
