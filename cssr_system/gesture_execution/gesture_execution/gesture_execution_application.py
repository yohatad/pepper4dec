#!/usr/bin/env python3

"""
gesture_execution_application.py
ROS2 Node for Gesture Execution.

Implements gesture execution logic.
Configuration is loaded from gesture_execution_configuration.yaml.

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohanneh@andrew.cmu.edu
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

    try:
        gesture_system = GestureExecutionSystem()
        gesture_system.get_logger().info("Gesture Execution System started - waiting for service calls")
        
        # Spin to handle service calls
        rclpy.spin(gesture_system)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running gesture execution system: {e}")
    finally:
        try:
            gesture_system.destroy_node()
        except:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()