/* person_detection_application.cpp
 *
 * Entry point for the Yolov11Node person detection lifecycle node. Loads
 * configuration, spins the node, and cleans up on shutdown; the node classes
 * themselves are implemented in person_detection_implementation.cpp.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 06, 2026
 * Version: v1.0
 */

#include "person_detection/person_detection_interface.h"

#include <iostream>

namespace {
constexpr const char* kBanner = R"(
================================================================================
                        Person Detection v1.0
================================================================================
  - YOLOv11 person detection with ByteTrack multi-object tracking
  - Configurable target classes via person_detection_configuration.yaml
  - Supported classes: person, car, bottle, chair, and 76 more COCO classes

  This program comes with ABSOLUTELY NO WARRANTY.
================================================================================
)";
}  // namespace

int main(int argc, char** argv) {
    std::cout << kBanner << std::endl;

    PersonDetectionConfig config = loadConfiguration();

    rclcpp::init(argc, argv);
    auto node = std::make_shared<Yolov11Node>(config);
    rclcpp::spin(node->get_node_base_interface());
    rclcpp::shutdown();
    return 0;
}
