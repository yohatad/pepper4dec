<div align="center">
<h1> Object Detection and Tracking </h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Object Detection and Tracking** package is a **ROS2** package designed to detect and track multiple objects (including persons) in real-time by subscribing to image topics. It publishes an array of detected objects with their bounding boxes, labels, and tracking IDs to the **/objectDetection/data** topic. Each entry in the published data includes the **label** of the detected object, the **centroid** coordinates, the **width** and **height** of the bounding box, and a unique **track_id** for tracking over time.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **YOLO-based Detection**: Uses state-of-the-art YOLO models for object detection
- **ByteTrack Tracking**: Multi-object tracking with ByteTrack algorithm
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
colcon build --packages-select object_detection
source install/setup.bash
```

2. **Set Up Python Virtual Environment**
```bash
# Create virtual environment
python3.10 -m venv ~/object_detection_env

# Activate the virtual environment
source ~/object_detection_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

3. **Install Python Dependencies**
```bash
# Install PyTorch with CUDA support (recommended)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install package requirements
pip install -r ~/ros2_ws/src/dec4africa/dec_system/object_detection/requirements.txt
```

4. **Download Model Files**
Ensure the required ONNX model files are in the `models/` directory:
- `yolov8n.onnx` - YOLOv8 detection model (or other YOLO variant)

# 🔧 Configuration Parameters
The configuration is managed via `config/object_detection_configuration.yaml`:

| Parameter                   | Description                                                      | Range/Values            | Default Value |
|-----------------------------|------------------------------------------------------------------|-------------------------|---------------|
| `camera`                    | Camera type to use                                               | `realsense`, `pepper`   | `realsense`   |
| `useCompressed`             | Use compressed ROS image topics                                  | `True`, `False`         | `False`       |
| `confidenceThreshold`       | Confidence threshold for object detection                        | `[0.0 - 1.0]`           | `0.7`         |
| `targetClasses`             | List of target classes to detect (or `all`)                      | List of class names     | `all`         |
| `trackThreshold`            | Confidence threshold for tracking (ByteTrack)                    | `[0.0 - 1.0]`           | `0.45`        |
| `trackBuffer`               | Number of frames to keep lost tracks before removing             | Positive integer        | `30`          |
| `matchThreshold`            | IoU threshold for matching detections to tracks                  | `[0.0 - 1.0]`           | `0.8`         |
| `frameRate`                 | Expected frame rate of the video stream (fps)                    | Positive integer        | `30`          |
| `imageTimeout`              | Timeout (seconds) for shutting down the node after video ends    | Float (seconds)         | `2.0`         |
| `verboseMode`               | Enable visualization using OpenCV windows and detailed logging   | `True`, `False`         | `False`       |

> **Note:**  
> Enabling **`verboseMode`** (`True`) will activate real-time visualization via OpenCV windows. 

# 🚀 Running the Node

## Launch All Components
The launch file starts the camera driver and object detection node:

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch with default configuration (RealSense camera)
ros2 launch object_detection object_detection_launch_robot.launch.py
```

## Launch Arguments
The launch file supports the following arguments:

| Argument        | Description                                      | Default | Example                    |
|-----------------|--------------------------------------------------|---------|----------------------------|
| `launch_camera` | Whether to launch the camera driver              | `true`  | `launch_camera:=false`     |

### Example: Using ROS2 Bag Data
When using pre-recorded ROS2 bag data, disable the camera launch:

```bash
ros2 launch object_detection object_detection_launch_robot.launch.py launch_camera:=false
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
source ~/object_detection_env/bin/activate

# Run object detection
ros2 run object_detection object_detection
```

# 🖥️ Output
The node publishes detected objects and their corresponding data to the `/objectDetection/data` topic. When running in verbose mode, it displays OpenCV-annotated color and depth images for visualization.

## Topic Structure
- **Published Topic**: `/objectDetection/data` (`dec_interfaces/msg/ObjectDetection`)
  - `label[]`: Array of object labels (strings)
  - `centroids[]`: Array of centroid coordinates (geometry_msgs/Point)
  - `width[]`: Array of bounding box widths (float)
  - `height[]`: Array of bounding box heights (float)
  - `track_id[]`: Array of tracking IDs (integer)

- **Subscribed Topics**:
  - Color image topic (depends on camera configuration)
  - Depth image topic (depends on camera configuration)

## Verification
To verify the node is publishing data:

```bash
# Monitor object detection output
ros2 topic echo /objectDetection/data

# Check node status
ros2 node list
ros2 topic list
```

# 🏗️ Architecture
The object detection system consists of two main components:

1. **Camera Driver**: Provides synchronized RGB-D image streams
2. **Object Detection Node**: 
   - Receives image streams from the camera
   - Performs object detection using YOLO model
   - Tracks objects across frames using ByteTrack algorithm
   - Publishes object detection results

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a><br>
<!-- - Visit: <a href="http://www.dec4africa.org">www.dec4africa.org</a> -->

# 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
