#!/usr/bin/env python3

""" gesture_execution_application.py

Entry point for the GestureExecutionSystem lifecycle node.
Running this node starts an action server that drives Pepper through deictic,
iconic, bowing, and nodding gestures.

This node loads gesture descriptors and topic mappings from YAML, computes the
joint trajectories needed to perform a requested gesture (using inverse
kinematics for deictic pointing), and streams the resulting joint-angle
trajectory to the robot. While a gesture is executing, it periodically reports
elapsed time back to the action client so callers can stay synchronised with
the robot's motion. Deictic gestures also publish RViz markers showing the
target point, shoulder position, and pointing vector.

Subscribers:
    /joint_states (sensor_msgs/JointState)
        Current joint positions, used to track the robot's arm/head/leg state.
    /robot_localization/pose (geometry_msgs/Pose2D)
        Current robot pose in the world frame, used to compute pointing direction.

Publishers:
    /joint_angles_trajectory (naoqi_bridge_msgs/JointAnglesTrajectory)
        Joint angle trajectories sent to the robot to perform gestures.
    /gesture_execution/visualization (visualization_msgs/Marker)
        Markers visualizing deictic gesture targets, shoulder, and pointing arrow.

Actions:
    /gesture_execution (dec_interfaces/action/Gesture)
        Executes a named or typed gesture (deictic, iconic, bow, nod) with
        feedback on elapsed time and a success/failure result.

Parameters (loaded from gesture_execution_configuration.yaml):
    gestureDescriptors (string, default: "gesture.yaml")
    robotTopics (string, default: "pepperTopics.yaml")
    verboseMode (bool, default: true)

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: April 18, 2025
Version: v1.0
"""

import rclpy
from .gesture_execution_implementation import GestureExecutionSystem

SOFTWARE_VERSION = "v1.0"

def main(args=None):
    """
    Main function to run the gesture execution system.
    """
    rclpy.init(args=args)
    
    node_name = "gesture_execution"
    
    # Construct the copyright message
    copyright_message = (
        f"{node_name} {SOFTWARE_VERSION}\n"
        "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY."
    )
    
    print(copyright_message)

    gesture_system = None  # Initialize to None
    
    try:
        gesture_system = GestureExecutionSystem()
        
        # Spin to handle service calls
        rclpy.spin(gesture_system)
        
    except KeyboardInterrupt:
        print("\nShutting down gesture execution system...")
    except Exception as e:
        print(f"Error running gesture execution system: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        if gesture_system is not None:
            gesture_system.destroy_node()
        
        if rclpy.ok():  # Only shutdown if not already shut down
            rclpy.shutdown()

if __name__ == '__main__':
    main()