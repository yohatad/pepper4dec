/* overt_attention_application.cpp
 *
 * Entry point for the OvertAttentionNode lifecycle node (Pepper's overt
 * attention controller). Constructs the node and spins it; the node class
 * itself is implemented in overt_attention_implementation.cpp.
 *
 * Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
 * Email: yohatad123@gmail.com
 * Date: June 12, 2026
 * Version: v1.0 - C++ port of overt_attention_unified_attention.py
 */

#include "overt_attention/overt_attention_interface.h"

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<OvertAttentionNode>();
        rclcpp::spin(node->get_node_base_interface());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("overt_attention"), "Exception: %s", e.what());
    }
    rclcpp::shutdown();
    return 0;
}
