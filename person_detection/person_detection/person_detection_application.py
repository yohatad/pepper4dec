#!/home/yoha/face_detection_env/bin/python3

""" person_detection_application.py

Entry point for the YOLOv11 person detection lifecycle node.
Running this node detects and tracks people (or other configurable COCO
classes) from a synchronized color/depth camera stream and publishes their
positions and depth.

This node loads its configuration from person_detection_configuration.yaml,
then constructs a YOLOv11 lifecycle node and spins it until shutdown. On
configure, the node loads the ONNX detection model and initializes the
ByteTrack tracker. On activate, it subscribes to the color and depth image
topics for the configured camera (RealSense, Pepper, or video), synchronizes
them, runs detection and tracking on each frame, and publishes the results
along with debug visualizations.

Subscribers:
    /camera/color/image_raw_custom (sensor_msgs/Image or CompressedImage)
        Color image stream used for object detection (topic resolved from
        pepper_topics.yaml based on the configured camera type).
    /camera/aligned_depth_to_color/image_raw_custom (sensor_msgs/Image or CompressedImage)
        Depth image stream used to estimate distance to detected objects
        (topic resolved from pepper_topics.yaml based on the configured camera type).

Publishers:
    /person_detection/data (dec_interfaces/PersonDetection)
        Tracked object detections: track IDs, class names/IDs, confidences,
        centroids (with depth), widths, and heights.
    /person_detection/debug (sensor_msgs/Image)
        Annotated color image showing tracked bounding boxes, labels, and depth.
    /person_detection/depth_debug (sensor_msgs/Image)
        Colorized visualization of the raw depth image.

Parameters (loaded from person_detection_configuration.yaml):
    camera (str, default: "realsense")
    useCompressed (bool, default: false)
    imageTimeout (float, default: 2.0)
    verboseMode (bool, default: false)
    confidenceThreshold (float, default: 0.6)
    targetClasses (list, default: ["person"])
    trackThreshold (float, default: 0.45)
    trackBuffer (int, default: 30)
    matchThreshold (float, default: 0.8)
    frameRate (int, default: 30)

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: December 07, 2025
Version: v1.0
"""

import sys
import rclpy
from .person_detection_implementation import YOLOv11, load_configuration

BANNER = """
================================================================================
                        Person Detection v1.0
================================================================================
  - YOLOv11 person detection with ByteTrack multi-object tracking
  - Configurable target classes via person_detection_configuration.yaml
  - Supported classes: person, car, bottle, chair, and 76 more COCO classes
  
  This program comes with ABSOLUTELY NO WARRANTY.
================================================================================
"""


def main():
    print(BANNER)
    
    # Load configuration
    config = load_configuration()
    
    rclpy.init()

    node = None
    try:
        # Create YOLOv11 person detection node with ByteTrack
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
