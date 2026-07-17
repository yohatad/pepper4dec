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
- **Python 3.10** or compatible version
- **CUDA-capable GPU** (recommended for optimal performance)
- **Intel RealSense camera** (if using RealSense) with USB 3.0 connection

## 🛠️ Installation

### Package Installation

```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select face_detection person_detection
source install/setup.bash
```

### Python Dependencies

```bash
# Install PyTorch with CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install package requirements
pip install -r ~/ros2_ws/src/pepper4dec/face_detection/requirements.txt
```

## 🔧 Configuration

Configuration is managed via `config/face_detection_configuration.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `useCompressed` | Use compressed ROS image topics | `False` |
| `camera` | Camera type to use | `realsense` |
| `sixdrepnetConfidence` | Confidence threshold for face detection | `0.90` |
| `sixdrepnetHeadposeAngle` | Head pose angle threshold in degrees | `10` |
| `imageTimeout` | Timeout for shutting down after video ends (s) | `2.0` |
| `verboseMode` | Enable visualization and detailed logging | `False` |
| `requirePersonDetection` | Require person detection before running face detection | `True` |
| `objectDetectionTimeout` | Timeout for person detection messages (s) | `0.5` |

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
source ~/face_detection_env/bin/activate
ros2 run person_detection person_detection

# Start Face Detection Node
source ~/face_detection_env/bin/activate
ros2 run face_detection face_detection
```

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

## Package Structure

```
face_detection/
├── config/
│   └── face_detection_configuration.yaml
├── data/
│   └── pepper_topics.yaml
├── launch/
│   └── face_detection_launch_robot.launch.py
├── models/
│   ├── face_detection_goldYOLO.onnx
│   └── face_detection_sixdrepnet360.onnx
├── resource/
│   └── face_detection
├── face_detection/
│   ├── __init__.py
│   ├── age_gender_detection.py
│   ├── face_detection_application.py
│   └── face_detection_implementation.py
├── package.xml
├── setup.py
├── setup.cfg
├── requirements.txt
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