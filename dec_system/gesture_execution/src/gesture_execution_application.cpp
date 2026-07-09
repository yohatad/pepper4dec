/* gesture_execution_application.cpp
 *
 * Entry point for the GestureExecutionSystem lifecycle node. Spins the
 * node single-threaded; the class itself is implemented in
 * gesture_execution_implementation.cpp.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: July 05, 2026
 * Version: v1.0
 */

#include "gesture_execution/gesture_execution_interface.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    RCLCPP_INFO(rclcpp::get_logger("gesture_execution"), "gesture_execution v1.0 — "
        "This program comes with ABSOLUTELY NO WARRANTY.");

    auto node = std::make_shared<GestureExecutionSystem>();
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
