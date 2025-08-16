#!/usr/bin/env python3

"""
face_detection_application.py ROS2 Node for Face and Mutual Gaze Detection and Localization.

This program comes with ABSOLUTELY NO WARRANTY.
"""

"""
face_detection_application.py   ROS2 node to run the face and mutual gaze detection algorithm.

The face detection is implemented using ROS2 image topics that could be configured to be the intel realsense camera or pepper robot
camera. It uses OpenCV to visualize the detected faces and gaze direction. The gaze direction is calculated using face
mesh landmarks which uses Google's MediaPipe library. The media pipe utilizes CPU for face detection and gaze direction.
The SixDrepNet uses YOLOONNX for face detection and SixDrepNet for gaze direction. The SixDrepNet utilizes GPU for faster
inference and better performance. This code contains the main function that initializes the face detection node and 
starts the face detection algorithm. The face detection algorithm can be either MediaPipe Face Detection or SixDrepNet 
that can be configured from the configuration file. It is also responsible for detecting the head pose estimation of the 
detected face. It subscribes to the intel realsense camera or pepper robot camera topics for the RGB and depth images.
It publishes one topic: /faceDetection/data that contains the face label ID, the centroid of the face, 
mutual gaze direction. 

Libraries
    - cv2
    - mediapipe
    - numpy
    - rclpy
    - ament_index_python
    - os
    - onnxruntime
    - multiprocessing
    - yaml
    - random
    - math (cos, sin, pi)
    - sensor_msgs.msg (Image, CompressedImage)
    - cv_bridge (CvBridge, CvBridgeError)
    - message_filters (ApproximateTimeSynchronizer, Subscriber)
    - geometry_msgs.msg (Point)
    - typing (Tuple, List)
    - cssr_system.msg (face_detection_msg_file)
    - face_detection_tracking (Sort, CentroidTracker)
    
Parameters
    Launch File Parameters:
        ros2 launch cssr_system face_detection_launch_robot.launch.py camera:=realsense
            camera: Camera type or video file (realsense or pepper or video)
            pepper_robot_ip: Pepper robot IP address (e.g 172.29.111.230 or 172.29.111.240)
            network_interface: Network interface for Pepper robot connection

Configuration File Parameters
    Key                                                     Value
    algorithm                                               sixdrep
    useCompressed                                           true
    mpFacedetConfidence                                     0.5
    mpHeadposeAngle                                         5
    centroidMaxDistance                                     15
    centroidMaxDisappeared                                  100
    sixdrepnet_confidence                                   0.65
    sixdrepnetHeadposeAngle                                 10
    sortMaxDisappeared                                      30
    sortMinHits                                             20
    sortIouThreshold                                        0.3
    verboseMode                                             true

Subscribed Topics
    Topic Name                                              Message Type
    /camera/color/image_raw                                 sensor_msgs/Image
    /camera/color/image_raw/compressed                      sensor_msgs/CompressedImage                 
    /camera/aligned_depth_to_color/image_raw                sensor_msgs/Image
    /camera/aligned_depth_to_color/image_raw/compressed     sensor_msgs/CompressedImage
    /naoqi_driver/camera/front/image_raw                    sensor_msgs/Image
    /naoqi_driver/camera/depth/image_raw                    sensor_msgs/Image

Published Topics
    Topic Name                                              Message Type
    /faceDetection/data                                     face_detection/face_detection_msg_file.msg

Input Data Files
    - pepperTopics.dat: Data file for Pepper robot camera topics

Model Files
    - face_detection_YOLO.onnx: YOLOONNX model for face detection
    - face_detection_sixdrepnet360.onnx: SixDrepNet model for gaze direction

Output Data Files
    None

Configuration File
    face_detection_configuration.yaml

Example of instantiation of the module
    ros2 launch cssr_system face_detection_launch_robot.launch.py camera:=realsense

    # Activate the python environment
    source ~/workspace/pepper_rob_ws/cssr4africa_face_person_detection_env/bin/activate

    (In a new terminal)
    ros2 run cssr_system face_detection_node.py

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohanneh@andrew.cmu.edu
Date: April 18, 2025
Version: v1.0
"""

import sys
import rclpy
from rclpy.node import Node
import os
import sys
from cssr_system.face_detection.face_detection_implementation import MediaPipe, SixDrepNet
BANNER = """faceDetection v1.0
This program comes with ABSOLUTELY NO WARRANTY.
"""

ALGORITHMS = {
    "mediapipe": MediaPipe,
    "sixdrep": SixDrepNet,
}

def main():
    rclpy.init()
    node = rclpy.create_node("faceDetection")
    node.get_logger().info(BANNER)

    # Declare and read algorithm param (default = sixdrep)
    node.declare_parameter("algorithm", "sixdrep")
    algo = node.get_parameter("algorithm").get_parameter_value().string_value.lower()

    if algo not in ALGORITHMS:
        node.get_logger().error(f"Invalid algorithm '{algo}'. Choose from {list(ALGORITHMS)}.")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    node.destroy_node()
    algo_node = ALGORITHMS[algo]()  # instantiate the chosen node

    try:
        if hasattr(algo_node, "spin"):
            algo_node.spin()
        else:
            rclpy.spin(algo_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Always try to clean up gracefully
        if hasattr(algo_node, "cleanup"):
            try:
                algo_node.cleanup()
            except Exception:
                pass
        algo_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
