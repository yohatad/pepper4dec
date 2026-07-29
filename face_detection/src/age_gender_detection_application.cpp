/* age_gender_detection_application.cpp
 *
 * Entry point for the MiVOLO age/gender estimation lifecycle node. Loads
 * configuration, spins the node, and cleans up (stopping the estimation
 * worker thread) on shutdown; the node classes themselves are implemented
 * in age_gender_detection_implementation.cpp.
 *
 * Author: Yohannes Tadesse Haile
 * Affiliation: Carnegie Mellon University Africa
 * Date: Jul 29, 2026
 * Version: v1.0
 */

#include "face_detection/age_gender_detection_interface.h"

#include "dec_common/node_runner.h"

int main(int argc, char** argv) {
    return dec_common::runNode<AgeGenderDetectionNode>(
        argc, argv,
        {"age_gender_detection v1.0 — This program comes with ABSOLUTELY NO WARRANTY.", "age_gender_detection"},
        nullptr,
        [](AgeGenderDetectionNode& node) { node.cleanup(); });
}
