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

#include "dec_common/node_runner.h"

int main(int argc, char** argv) {
    return dec_common::runNode<GestureExecutionSystem>(
        argc, argv,
        {"gesture_execution v1.0 — This program comes with ABSOLUTELY NO WARRANTY.", "gesture_execution"});
}
