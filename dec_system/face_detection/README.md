<div align="center">
<h1>Face and Mutual Gaze Detection and Localization</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Face and Mutual Gaze Detection and Localization** package detects multiple faces and evaluates their mutual gaze in real-time by subscribing to image topics. It publishes an array of detected faces and their mutual gaze status to the `/face_detection/data` topic. Each entry includes the label ID, centroid coordinates, bounding box dimensions, and mutual gaze status.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **SixDrepNet Algorithm**: State-of-the-art face detection and head pose estimation
- **Person Detection Integration**: Uses YOLO-based person detection to locate faces
- **Mutual Gaze Detection**: Evaluates engagement based on head pose angles
- **Real-time Processing**: Processes synchronized RGB-D camera streams
- **Multi-camera Support**: RealSense and Pepper camera support
- **ROS2 Bag Compatible**: Optional camera launch for use with recorded data

## ✅ Prerequisites
- **ROS2 Humble** or newer
- **CUDA-capable GPU** (recommended for optimal performance; falls back to CPU automatically)
- **Intel RealSense camera** (if using RealSense) with USB 3.0 connection

## 🛠️ Installation

### Package Installation

```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace (pulls in the dec_interfaces/dec_common dependencies automatically)
cd ~/ros2_ws
colcon build --packages-up-to face_detection person_detection
source install/setup.bash
```

### Model Files

Place the required ONNX model files in `models/`:
- `face_detection_goldYOLO.onnx` - face detector weights
- `face_detection_sixdrepnet360.onnx` - head-pose estimator weights

## 🔧 Configuration

Configuration is managed via ROS2 parameters, loaded from `config/face_detection_configuration.yaml`
(`ros2 param get/set /face_detection <name>` also works at runtime):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_compressed` | Use compressed ROS image topics | `false` |
| `camera` | Camera type to use (`realsense`, `pepper`, or `video`) | `pepper` |
| `sixdrepnet_confidence` | Confidence threshold for face detection | `0.90` |
| `sixdrepnet_headpose_angle` | Head pose angle threshold in degrees | `10.0` |
| `image_timeout` | Timeout for shutting down after video ends (s) | `2.0` |
| `verbose_mode` | Enable visualization and detailed logging | `false` |
| `require_person_detection` | Require person detection before running face detection | `true` |
| `person_detection_timeout` | Timeout for person detection messages (s) | `0.5` |
| `prioritize_face_depth` | Prefer face-region depth over person-region depth when both are available | `true` |

## 🚀 Running the Node

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch with default configuration (RealSense camera)
ros2 launch face_detection face_detection_launch_robot.launch.py

# Using ROS2 bag data (disable camera launch)
ros2 launch face_detection face_detection_launch_robot.launch.py launch_camera:=false
```

### Manual Node Execution

```bash
# Start Camera Driver (if not using bags)
ros2 run realsense2_camera realsense2_camera_node \
  --ros-args \
  -p rgb_camera.color_profile:=640x480x15 \
  -p depth_module.depth_profile:=640x480x15 \
  -p align_depth.enable:=true \
  -p enable_sync:=true

# Start Person Detection Node
ros2 run person_detection person_detection

# Start Face Detection Node
ros2 run face_detection face_detection
```

Both this and the launch file above start the node unconfigured. Transition it
manually with `ros2 lifecycle set /face_detection configure` then
`... activate`, or launch the whole stack via `dec_launch`'s
`dec_system.launch.py`, which drives these transitions automatically through
`nav2_lifecycle_manager`.

## 🖥️ ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | Color image from camera |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Depth image |
| `/person_detection/data` | `dec_interfaces/msg/PersonDetection` | Person detection results for face localization |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/face_detection/data` | `dec_interfaces/msg/FaceDetection` | Detected faces with mutual gaze status |
| `/face_detection/debug` | `sensor_msgs/Image` | Debug RGB image with face detection overlays |
| `/face_detection/depth_debug` | `sensor_msgs/Image` | Debug colorized depth image |

## 📨 Message Structure

### `/face_detection/data` (`dec_interfaces/msg/FaceDetection`)

| Field | Type | Description |
|-------|------|-------------|
| `face_label_id[]` | string[] | Array of face IDs |
| `centroids[]` | `geometry_msgs/Point[]` | Array of centroid coordinates |
| `width[]` | float[] | Array of bounding box widths |
| `height[]` | float[] | Array of bounding box heights |
| `mutual_gaze[]` | bool[] | Array of mutual gaze status |

## 📁 Package Structure

```
face_detection/
├── config/
│   └── face_detection_configuration.yaml           # ROS2 parameters
├── data/
│   └── pepper_topics.yaml                          # topic name overrides
├── launch/
│   └── face_detection_launch_robot.launch.py
├── models/
│   ├── face_detection_goldYOLO.onnx                # face detector weights
│   └── face_detection_sixdrepnet360.onnx           # head-pose estimator weights
├── include/face_detection/
│   ├── byte_tracker.h
│   └── face_detection_interface.h                  # node/class declarations
├── src/
│   ├── byte_tracker.cpp
│   ├── face_detection_application.cpp              # node entry point (main)
│   └── face_detection_implementation.cpp           # face detection + gaze estimation
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🏗️ Architecture

The face detection system consists of three main components:

1. **Camera Driver**: Provides synchronized RGB-D image streams
2. **Person Detection Node**: Detects persons in the scene using YOLO
3. **Face Detection Node**:
   - Receives person detections from person detection
   - Performs face detection within person bounding boxes
   - Estimates head pose using SixDrepNet
   - Determines mutual gaze based on head pose angles
   - Publishes face detection results

## 🧪 Testing

```bash
# Check node is running
ros2 node list

# Monitor face detection output
ros2 topic echo /face_detection/data

# Verify topics
ros2 topic list
```

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.