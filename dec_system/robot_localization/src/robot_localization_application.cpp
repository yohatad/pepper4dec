/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: July 05, 2026
Version: v1.0 - C++ port of robot_localization.py's main()
*/

#include "robot_localization/robot_localization_interface.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RobotLocalization>();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
