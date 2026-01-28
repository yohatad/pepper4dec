<div align="center">
<h1> Face and Mutual Gaze Detection and Localization (ROS2) </h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Face and Mutual Gaze Detection and Localization** package is a **ROS2** package designed to detect multiple faces and evaluate their **mutual gaze** in real-time by subscribing to image topics. It publishes an array of detected faces and their mutual gaze status to the **/faceDetection/data** topic. Each entry in the published data includes the **label ID** of the detected face, the **centroid** coordinates representing the center point of each face, and a boolean value indicating **mutual gaze** status as either **True** or **False**, the **width** and **height** of the bounding box.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **SixDrepNet Algorithm**: State-of-the-art face detection and head pose estimation
- **Object Detection Integration**: Uses YOLO-based person detection to locate faces within person bounding boxes
- **Real-time Processing**: Processes synchronized RGB-D camera streams
- **Configurable**: Configuration via YAML file
- **Multi-camera Support**: RealSense and Pepper camera support
- **ROS2 Bag Compatible**: Optional camera launch for use with recorded data

# 🛠️ Installation 

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **CUDA-capable GPU** (recommended for optimal performance)
- **Intel RealSense camera** (if using RealSense) with USB 3.0 connection

## Package Installation

1. **Clone and Build the Workspace**
```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone <repository-url>

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select face_detection object_detection
source install/setup.bash
```

2. **Set Up Python Virtual Environment**
```bash
# Create virtual environment
python3.10 -m venv ~/face_detection_env

# Activate the virtual environment
source ~/face_detection_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

3. **Install Python Dependencies**
```bash
# Install PyTorch with CUDA support (recommended)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install package requirements
pip install -r ~/ros2_ws/src/cssr4africa/cssr_system/face_detection/face_detection_requirements_x86.txt
```

4. **Download Model Files**
Ensure the required ONNX model files are in the `models/` directory:
- `face_detection_goldYOLO.onnx` - Face detection model
- `face_detection_sixdrepnet360.onnx` - Head pose estimation model

# 🔧 Configuration Parameters
The configuration is managed via `config/face_detection_configuration.yaml`:

| Parameter                   | Description                                                      | Range/Values            | Default Value |
|-----------------------------|------------------------------------------------------------------|-------------------------|---------------|
| `algorithm`                 | Algorithm selected for face detection                            | `sixdrep`               | `sixdrep`     |
| `useCompressed`             | Use compressed ROS image topics                                  | `True`, `False`         | `False`       |
| `camera`                    | Camera type to use                                               | `realsense`, `pepper`   | `realsense`   |
| `sixdrepnetConfidence`      | Confidence threshold for face detection (SixDRepNet)             | `[0.0 - 1.0]`           | `0.80`        |
| `sixdrepnetHeadposeAngle`   | Head pose angle threshold in degrees (SixDRepNet)                | Positive integer        | `10`          |
| `imageTimeout`              | Timeout (seconds) for shutting down the node after video ends    | Float (seconds)         | `2.0`         |
| `verboseMode`               | Enable visualization using OpenCV windows and detailed logging   | `True`, `False`         | `True`        |

> **Note:**  
> Enabling **`verboseMode`** (`True`) will activate real-time visualization via OpenCV windows. 

# 🚀 Running the Node

## Launch All Components
The launch file starts the camera driver, object detection node, and face detection node:

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch with default configuration (RealSense camera)
ros2 launch face_detection face_detection_launch_robot.launch.py
```

## Launch Arguments
The launch file supports the following arguments:

| Argument        | Description                                      | Default | Example                    |
|-----------------|--------------------------------------------------|---------|----------------------------|
| `launch_camera` | Whether to launch the camera driver              | `true`  | `launch_camera:=false`     |

### Example: Using ROS2 Bag Data
When using pre-recorded ROS2 bag data, disable the camera launch:

```bash
ros2 launch face_detection face_detection_launch_robot.launch.py launch_camera:=false
```

## Manual Node Execution
You can also run nodes individually:

1. **Start Camera Driver** (if not using bags):
```bash
ros2 run realsense2_camera realsense2_camera_node \
  --ros-args \
  -p rgb_camera.color_profile:=640x480x15 \
  -p depth_module.depth_profile:=640x480x15 \
  -p align_depth.enable:=true \
  -p enable_sync:=true
```

2. **Start Object Detection Node**:
```bash
# Activate Python environment first
source ~/face_detection_env/bin/activate

# Run object detection
ros2 run object_detection object_detection
```

3. **Start Face Detection Node**:
```bash
# Activate Python environment
source ~/face_detection_env/bin/activate

# Run face detection
ros2 run face_detection face_detection
```

# 🖥️ Output
The node publishes detected faces and their corresponding data to the `/faceDetection/data` topic. When running in verbose mode, it displays OpenCV-annotated color and depth images for visualization.

## Topic Structure
- **Published Topic**: `/faceDetection/data` (`cssr_interfaces/msg/FaceDetection`)
  - `face_label_id[]`: Array of face IDs (strings)
  - `centroids[]`: Array of centroid coordinates (geometry_msgs/Point)
  - `width[]`: Array of bounding box widths (float)
  - `height[]`: Array of bounding box heights (float)
  - `mutual_gaze[]`: Array of mutual gaze status (boolean)

- **Subscribed Topics**:
  - Color image topic (depends on camera configuration)
  - Depth image topic (depends on camera configuration)
  - `/objectDetection/data`: Object detection results for person localization

## Verification
To verify the node is publishing data:

```bash
# Monitor face detection output
ros2 topic echo /faceDetection/data

# Check node status
ros2 node list
ros2 topic list
```

# 🏗️ Architecture
The face detection system consists of three main components:

1. **Camera Driver**: Provides synchronized RGB-D image streams
2. **Object Detection Node**: Detects persons in the scene using YOLO
3. **Face Detection Node**: 
   - Receives person detections from object detection
   - Performs face detection within person bounding boxes
   - Estimates head pose using SixDrepNet
   - Determines mutual gaze based on head pose angles
   - Publishes face detection results

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohanneh@andrew.cmu.edu">yohanneh@andrew.cmu.edu</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2023 Upanzi Network   

2026-01-10
