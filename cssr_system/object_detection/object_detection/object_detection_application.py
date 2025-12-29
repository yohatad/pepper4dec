#!/home/yoha/face_detection_env/bin/python3

"""
object_detection_application.py
ROS2 Node for Object Detection and Localization.

Implements object detection using YOLOv8.
Configuration is loaded from object_detection_configuration.yaml.

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohanneh@andrew.cmu.edu
Date: December 07, 2025
Version: v1.0
"""

import sys
import rclpy
from .object_detection_implementation import YOLOv8, load_configuration


BANNER = """objectDetection v1.0
This program comes with ABSOLUTELY NO WARRANTY.
"""

def main():
    print(BANNER)

    # Load configuration
    config = load_configuration()

    rclpy.init()

    node = None
    try:
        # Create YOLOv8 object detection node
        node = YOLOv8(config)

        # Setup subscriptions
        if not node.subscribe_topics():
            node.get_logger().error("Failed to setup subscribers")
            sys.exit(1)

        # Start monitoring for timeouts
        node.start_timeout_monitor()

        # Verify camera resolution compatibility (except Pepper)
        if (node.depth_image is not None and not node.check_camera_resolution(node.color_image, node.depth_image) and node.camera_type != "pepper"):
            node.get_logger().error("Color and depth camera resolutions don't match")
            sys.exit(1)

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
