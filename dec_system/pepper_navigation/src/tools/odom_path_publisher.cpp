/* odom_path_publisher.cpp
 *
 * Subscribes to /pepper_odom and accumulates a nav_msgs/Path for visual
 * odometry-quality inspection in RViz, along with start/end markers and a
 * deviation line back to the start pose. A reset service clears the
 * accumulated path to begin a new closed-loop test run.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 05, 2026
 * Version: v1.0 - C++ port of odom_path_publisher.py
 */

#include <cmath>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_srvs/srv/empty.hpp>

class OdomPathPublisher : public rclcpp::Node {
public:
    OdomPathPublisher() : Node("odom_path_publisher") {
        min_distance_ = declare_parameter("min_distance", 0.02);
        min_angle_ = declare_parameter("min_angle", 0.05);

        path_.header.frame_id = "pepper_odom";

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/pepper_odom", 10, std::bind(&OdomPathPublisher::odomCallback, this, std::placeholders::_1));
        path_pub_ = create_publisher<nav_msgs::msg::Path>("/odom_path", 10);
        marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/odom_markers", 10);
        reset_srv_ = create_service<std_srvs::srv::Empty>(
            "reset_odom_path",
            std::bind(&OdomPathPublisher::resetCallback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(get_logger(),
            "Odometry path publisher started.\n"
            "  Path topic  : /odom_path\n"
            "  Markers     : /odom_markers  (green sphere = START)\n"
            "  Reset service: ros2 service call /reset_odom_path std_srvs/srv/Empty");
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        geometry_msgs::msg::PoseStamped ps;
        ps.header = msg->header;
        ps.header.frame_id = "pepper_odom";
        ps.pose = msg->pose.pose;

        if (path_.poses.empty()) {
            start_pose_ = ps;
            has_start_pose_ = true;
            path_.poses.push_back(ps);
            publishStartMarkers(ps);
            publishPath(msg->header.stamp);
            return;
        }

        const auto& last = path_.poses.back().pose;
        const double dx = ps.pose.position.x - last.position.x;
        const double dy = ps.pose.position.y - last.position.y;
        const double dist = std::hypot(dx, dy);
        const double dyaw = std::fabs(yaw(ps.pose.orientation) - yaw(last.orientation));

        if (dist < min_distance_ && dyaw < min_angle_) {
            return;
        }

        path_.poses.push_back(ps);
        pose_count_++;
        publishPath(msg->header.stamp);

        // Update end marker and log deviation every 50 new poses
        if (pose_count_ % 50 == 0) {
            publishEndMarker(ps);
            logDeviation(ps);
        }
    }

    void publishPath(const builtin_interfaces::msg::Time& stamp) {
        path_.header.stamp = stamp;
        path_pub_->publish(path_);
    }

    visualization_msgs::msg::Marker makeMarker(
        const geometry_msgs::msg::PoseStamped& ps, int mid, int32_t mtype,
        double sx, double sy, double sz, double r, double g, double b) {
        visualization_msgs::msg::Marker m;
        m.header = ps.header;
        m.ns = "odom_start";
        m.id = mid;
        m.type = mtype;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose = ps.pose;
        m.scale.x = sx;
        m.scale.y = sy;
        m.scale.z = sz;
        m.color.r = r;
        m.color.g = g;
        m.color.b = b;
        m.color.a = 1.0;
        m.lifetime.sec = 0;  // persistent until reset
        m.lifetime.nanosec = 0;
        return m;
    }

    void publishStartMarkers(const geometry_msgs::msg::PoseStamped& ps) {
        // Green sphere at start position
        auto m = makeMarker(ps, 0, visualization_msgs::msg::Marker::SPHERE, 0.2, 0.2, 0.2, 0.0, 1.0, 0.0);
        marker_pub_->publish(m);

        // "START" text label above the sphere
        auto t = makeMarker(ps, 1, visualization_msgs::msg::Marker::TEXT_VIEW_FACING, 0.0, 0.0, 0.15, 1.0, 1.0, 1.0);
        t.pose.position.z += 0.35;
        t.text = "START";
        marker_pub_->publish(t);
    }

    void publishEndMarker(const geometry_msgs::msg::PoseStamped& ps) {
        // Yellow sphere showing current position relative to start
        auto m = makeMarker(ps, 2, visualization_msgs::msg::Marker::SPHERE, 0.15, 0.15, 0.15, 1.0, 1.0, 0.0);
        marker_pub_->publish(m);

        // Deviation line from start to current position
        if (!has_start_pose_) {
            return;
        }
        visualization_msgs::msg::Marker line;
        line.header = ps.header;
        line.ns = "odom_deviation";
        line.id = 3;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.scale.x = 0.03;
        line.color.r = 1.0;
        line.color.g = 0.4;
        line.color.b = 0.0;
        line.color.a = 0.9;
        line.lifetime.sec = 0;
        line.lifetime.nanosec = 0;

        geometry_msgs::msg::Point p0;
        p0.x = start_pose_.pose.position.x;
        p0.y = start_pose_.pose.position.y;
        p0.z = 0.0;
        geometry_msgs::msg::Point p1;
        p1.x = ps.pose.position.x;
        p1.y = ps.pose.position.y;
        p1.z = 0.0;
        line.points = {p0, p1};
        marker_pub_->publish(line);
    }

    void logDeviation(const geometry_msgs::msg::PoseStamped& ps) {
        if (!has_start_pose_) {
            return;
        }
        const double sx = start_pose_.pose.position.x;
        const double sy = start_pose_.pose.position.y;
        const double cx = ps.pose.position.x;
        const double cy = ps.pose.position.y;
        const double dev = std::hypot(cx - sx, cy - sy);
        RCLCPP_INFO(get_logger(), "Deviation from start: %.4f m  |  pos=(%.3f, %.3f)  |  path points: %zu",
            dev, cx, cy, path_.poses.size());
    }

    void resetCallback(
        const std::shared_ptr<std_srvs::srv::Empty::Request>,
        std::shared_ptr<std_srvs::srv::Empty::Response>) {
        path_.poses.clear();
        has_start_pose_ = false;
        pose_count_ = 0;
        RCLCPP_INFO(get_logger(), "Path reset - waiting for next /pepper_odom message.");
    }

    static double yaw(const geometry_msgs::msg::Quaternion& q) {
        return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }

    double min_distance_ = 0.02;
    double min_angle_ = 0.05;
    nav_msgs::msg::Path path_;
    geometry_msgs::msg::PoseStamped start_pose_;
    bool has_start_pose_ = false;
    int pose_count_ = 0;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr reset_srv_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OdomPathPublisher>());
    rclcpp::shutdown();
    return 0;
}
