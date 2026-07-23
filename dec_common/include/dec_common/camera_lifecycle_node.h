/* camera_lifecycle_node.h
 *
 * Shared lifecycle-node base for the camera-driven perception nodes
 * (face_detection, person_detection). Owns the camera plumbing that both
 * nodes previously duplicated: topic resolution from pepper_topics.yaml,
 * synchronized color/depth and RGB-only subscriptions (raw and compressed),
 * depth decoding, the debug visualization pipeline, the image-timeout
 * monitor, and depth-in-region lookup.
 *
 * The two nodes' deliberate behavioral differences are captured in
 * CameraNodeBehavior rather than papered over:
 *   - median_depth:         face uses the median depth in a region,
 *                           person uses the mean.
 *   - always_publish_debug: face always publishes debug images; person
 *                           gates both imshow and publishing behind
 *                           verbose_mode + DISPLAY.
 *   - quit_on_q:            person's imshow window quits the node on 'q'.
 *
 * Unifications (previously inconsistent between the two copies):
 *   - The color/depth resolution check runs on the freshly decoded pair
 *     (face's ordering; person checked the previous frame pair).
 *   - The depth frame shown by the visualization timer is snapshotted
 *     under frame_mutex_ in updateLatestFrame (face's pattern; person read
 *     depth_image_ cross-thread, which was a latent data race).
 *   - camera type "video" (RealSense topics) is accepted by both.
 *
 * This base deliberately does NOT override the lifecycle callbacks —
 * derived nodes keep their own configure/activate orchestration and call
 * the protected helpers/members here.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 18, 2026
 * Version: v1.0
 *
 * Copyright (C) 2025 Carnegie Mellon University Africa
 */

#ifndef DEC_COMMON_CAMERA_LIFECYCLE_NODE_H
#define DEC_COMMON_CAMERA_LIFECYCLE_NODE_H

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/opencv.hpp>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace dec_common {

struct CameraNodeBehavior {
    // Package whose share/<pkg>/data/pepper_topics.yaml resolves camera topics.
    std::string topics_package;
    // Prefix for the imshow debug windows, e.g. "Person Detection".
    std::string debug_window_prefix;
    // getDepthInRegion statistic: median (true) or mean (false).
    bool median_depth = false;
    // Publish debug images unconditionally (true) or only when
    // verbose_mode + DISPLAY are set (false).
    bool always_publish_debug = false;
    // 'q' in the imshow window shuts the node down.
    bool quit_on_q = false;
};

class CameraLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public:
    using CallbackReturn =
        rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

protected:
    CameraLifecycleNode(const std::string& node_name, CameraNodeBehavior behavior);

    // Called by the camera callbacks once a frame pair is decoded; derived
    // classes run detection here and call updateLatestFrame() with the
    // annotated frame.
    virtual void processImages() = 0;

    // ── Debug visualization ─────────────────────────────────────────────────
    void updateLatestFrame(const cv::Mat& frame);
    void visualizationCallback();

    // ── Camera topic resolution ─────────────────────────────────────────────
    std::pair<std::string, std::string> getTopicNames();
    std::optional<std::string> extractTopic(const std::string& image_topic_key);

    // ── Subscriptions (called from derived on_activate) ─────────────────────
    bool createCameraSubscriptions();

    // ── Frame callbacks ─────────────────────────────────────────────────────
    void synchronizedCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data,
                              const sensor_msgs::msg::Image::ConstSharedPtr& depth_data);
    void synchronizedCallbackCompressed(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& color_data,
                                        const sensor_msgs::msg::CompressedImage::ConstSharedPtr& depth_data);
    void rgbOnlyCallback(const sensor_msgs::msg::Image::ConstSharedPtr& color_data);
    void rgbOnlyCallbackCompressed(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& color_data);

    std::optional<cv::Mat> processDepthImageMsg(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    std::optional<cv::Mat> processDepthCompressedMsg(const sensor_msgs::msg::CompressedImage::ConstSharedPtr& msg);

    // ── Monitoring / helpers ────────────────────────────────────────────────
    void startTimeoutMonitor();
    void checkTimeout();
    bool checkCameraResolution(const cv::Mat& color_image, const cv::Mat& depth_image) const;
    std::optional<cv::Mat> makeDepthVis(const cv::Mat& depth) const;
    std::optional<float> getDepthInRegion(double centroid_x, double centroid_y, double box_width,
                                          double box_height, double region_scale = 0.1) const;
    cv::Scalar generateDarkColor();

    // ── Shared state ────────────────────────────────────────────────────────
    CameraNodeBehavior behavior_;
    std::string node_name_;

    cv::Mat color_image_;
    cv::Mat depth_image_;

    bool use_compressed_ = false;
    std::string camera_type_ = "realsense";
    bool verbose_mode_ = true;
    double image_timeout_ = 2.0;

    rclcpp::Time timer_start_;
    std::optional<double> last_image_time_;

    std::mutex frame_mutex_;
    cv::Mat latest_frame_;
    cv::Mat latest_depth_;

    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr depth_debug_pub_;

    rclcpp::TimerBase::SharedPtr vis_timer_;
    rclcpp::TimerBase::SharedPtr timeout_timer_;

    // Synchronized (uncompressed) subscription pair.
    using ApproxSync = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image>;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>> color_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image, rclcpp_lifecycle::LifecycleNode>> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<ApproxSync>> sync_;

    // Synchronized (compressed) subscription pair.
    using ApproxSyncCompressed = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::CompressedImage, sensor_msgs::msg::CompressedImage>;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CompressedImage, rclcpp_lifecycle::LifecycleNode>>
        color_sub_compressed_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::CompressedImage, rclcpp_lifecycle::LifecycleNode>>
        depth_sub_compressed_;
    std::shared_ptr<message_filters::Synchronizer<ApproxSyncCompressed>> sync_compressed_;

    // Pepper RGB-only subscription (plain, not message_filters-wrapped).
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_sub_plain_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr color_sub_plain_compressed_;
};

}  // namespace dec_common

#endif  // DEC_COMMON_CAMERA_LIFECYCLE_NODE_H
