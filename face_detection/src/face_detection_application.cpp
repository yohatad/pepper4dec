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

#include "dec_common/node_runner.h"

int main(int argc, char** argv) {
    return dec_common::runNode<SixDrepNet>(
        argc, argv,
        {"face_detection v1.0 — This program comes with ABSOLUTELY NO WARRANTY.", "face_detection"},
        nullptr,
        [](SixDrepNet& node) { node.cleanup(); });
}
