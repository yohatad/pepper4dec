/*
Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: July 05, 2026
Version: v1.0 - C++ port of gesture_execution_application.py
*/

#include "gesture_execution/gesture_execution_interface.h"

#include <iostream>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::cout << "gesture_execution v1.0\n"
              << "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY." << std::endl;

    auto node = std::make_shared<GestureExecutionSystem>();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
