#!/home/yoha/face_detection_env/bin/python3

"""
object_detection_application.py
ROS2 Node for Object Detection and Localization.

Implements object detection using YOLOv11 with ByteTrack tracking (bytetracker package).
Configuration is loaded from object_detection_configuration.yaml.

Supports configurable object classes to track (e.g., person, car, bottle, etc.)

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohanneh@andrew.cmu.edu
Date: December 07, 2025
Version: v2.0
"""

import sys
import rclpy
from .object_detection_implementation import YOLOv11, load_configuration

BANNER = """
================================================================================
                        Object Detection v1.0
================================================================================
  - YOLOv11 object detection with ByteTrack multi-object tracking
  - Configurable target classes via object_detection_configuration.yaml
  - Supported classes: person, car, bottle, chair, and 76 more COCO classes
  
  This program comes with ABSOLUTELY NO WARRANTY.
================================================================================
"""


def main():
    print(BANNER)
    
    # Load configuration
    config = load_configuration()
    
    # Print configuration summary
    print(f"Configuration:")
    print(f"  Camera: {config.get('camera', 'realsense')}")
    print(f"  Target classes: {config.get('targetClasses', ['person'])}")
    print(f"  Confidence threshold: {config.get('confidenceThreshold', 0.5)}")
    print(f"  Track threshold: {config.get('trackThreshold', 0.45)}")
    print(f"  Track buffer: {config.get('trackBuffer', 30)} frames")
    print(f"  Match threshold: {config.get('matchThreshold', 0.8)}")
    print(f"  Verbose mode: {config.get('verboseMode', True)}")
    print()

    rclpy.init()

    node = None
    try:
        # Create YOLOv11 object detection node with ByteTrack
        node = YOLOv11(config)

        # Spin until shutdown
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutdown requested by user")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
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
