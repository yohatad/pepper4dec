#!/home/yoha/face_detection_env/bin/python3

"""
face_detection_application.py
ROS2 Node for Face and Mutual Gaze Detection and Localization.

Implements face detection using SixDrepNet (YOLO + SixDrepNet).
Configuration is loaded from face_detection_configuration.yaml.

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohanneh@andrew.cmu.edu
Date: April 18, 2025
Version: v1.0
"""

import sys
import rclpy
from .face_detection_implementation import SixDrepNet, load_configuration

SOFTWARE_VERSION = "v1.0"

def main():
    """
    Main function to run the face detection system.
    """
    
    rclpy.init()
    node_name = "face_detection"

    copyright_message = (
        f"{node_name} {SOFTWARE_VERSION}\n"
        "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY."
    )

    print(copyright_message)

    # Load configuration
    config = load_configuration()

    try:
        node = SixDrepNet(config)

        # Spin until shutdown
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutdown requested by user")

    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)
        
    finally:
        if node is not None:
            try:
                node.cleanup()
                node.destroy_node()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
