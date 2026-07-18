/* pepper_odom_anchor_application.cpp
 *
 * Entry point for the pepper_odom_anchor lifecycle node. Spins the node
 * single-threaded; the RobotLocalization class itself is implemented in
 * pepper_odom_anchor_implementation.cpp.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 05, 2026
 * Version: v1.0
 */

#include "pepper_odom_anchor/pepper_odom_anchor_interface.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotLocalization>();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
