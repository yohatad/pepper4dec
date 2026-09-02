#!/home/yoha/face_detection_env/bin/python3

""" face_detection_application.py

Entry point for the SixDrepNet face and mutual gaze detection lifecycle node.
Running this node detects faces in the camera stream, estimates head pose, and
publishes mutual gaze and depth information for each tracked face.

On startup, the node loads its configuration, then constructs and spins a
`SixDrepNet` lifecycle node. Once configured and activated, it subscribes to
synchronized color/depth image topics (and, if enabled, person detection
results), runs YOLO face detection followed by SixDrepNet head-pose estimation
on each frame, matches faces to tracked persons, and publishes face detection
results along with debug visualization images.

Subscribers:
    <camera color topic> (sensor_msgs/Image or CompressedImage)
        Synchronized RGB camera frames (topic resolved from camera config; e.g. /camera/color/image_raw_custom)
    <camera depth topic> (sensor_msgs/Image or CompressedImage)
        Synchronized depth camera frames (topic resolved from camera config)
    /person_detection/data (dec_interfaces/PersonDetection)
        Tracked person detections used to constrain and match faces (only if requirePersonDetection is true)

Publishers:
    /face_detection/data (dec_interfaces/FaceDetection)
        Per-frame face tracking results: face IDs, centroids, sizes, and mutual gaze flags
    /face_detection/debug (sensor_msgs/Image)
        Debug visualization of the color frame with face boxes and head-pose axes
    /face_detection/depth_debug (sensor_msgs/Image)
        Colorized depth visualization for debugging

Parameters (loaded from face_detection_configuration.yaml):
    algorithm (str, default: "sixdrep")
    useCompressed (bool, default: False)
    camera (str, default: "realsense")
    verboseMode (bool, default: True)
    imageTimeout (float, default: 2.0)
    sixdrepnetConfidence (float, default: 0.65)
    sixdrepnetHeadposeAngle (int, default: 10)
    requirePersonDetection (bool, default: True)
    personDetectionTimeout (float, default: 0.5)
    prioritizeFaceDepth (bool, default: True)

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: April 18, 2025
Version: v1.0
"""

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
    config = load_configuration()
    node = None

    try:
        node = SixDrepNet(config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
