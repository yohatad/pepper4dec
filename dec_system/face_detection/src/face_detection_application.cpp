/* face_detection_application.cpp
 *
 * Entry point for the SixDrepNet face and mutual gaze detection lifecycle
 * node. Loads configuration, spins the node, and cleans up (closing any
 * debug windows) on shutdown; the node classes themselves are implemented in
 * face_detection_implementation.cpp.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 06, 2026
 * Version: v1.0
 */

#include "face_detection/face_detection_interface.h"

#include <iostream>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::cout << "face_detection v1.0\n"
              << "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY." << std::endl;

    FaceDetectionConfig config = loadConfiguration();

    auto node = std::make_shared<SixDrepNet>(config);
    rclcpp::spin(node->get_node_base_interface());
    node->cleanup();
    rclcpp::shutdown();
    return 0;
}
