<div align="center">
<h1>Person Detection and Tracking</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Person Detection and Tracking** package is a ROS2 package designed to detect and track multiple persons in real-time by subscribing to image topics. It publishes an array of detected persons with their bounding boxes, labels, and tracking IDs to the `/person_detection/data` topic. Each entry includes the label, centroid coordinates, bounding box dimensions, and a unique tracking ID for maintaining identity across frames.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **YOLO-based Detection**: Uses state-of-the-art YOLO models for person detection
- **ByteTrack Tracking**: Multi-person tracking with ByteTrack algorithm
- **Real-time Processing**: Processes synchronized RGB-D camera streams
- **Configurable**: Configuration via YAML file
- **Multi-camera Support**: RealSense and Pepper camera support
- **ROS2 Bag Compatible**: Optional camera launch for use with recorded data

## ✅ Prerequisites
- **ROS2 Humble** or newer
- **CUDA-capable GPU** (recommended for optimal performance; falls back to CPU automatically)
- **Intel RealSense camera** (if using RealSense) with USB 3.0 connection

## 🛠️ Installation

### Package Installation

```bash
cd ~/ros2_ws
colcon build --packages-up-to person_detection
source install/setup.bash
```

### Model Files

Download the required ONNX model files to the `models/` directory:
- `person_detection_yolov11m.onnx` - YOLO11m detection model (or other YOLO variant)

## 🔧 Configuration

Configuration is managed via ROS2 parameters, loaded from `config/person_detection_configuration.yaml`
(`ros2 param get/set /person_detection <name>` also works at runtime):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `camera` | Camera type to use (`realsense`, `pepper`, or `video`) | `pepper` |
| `use_compressed` | Use compressed ROS image topics | `false` |
| `confidence_threshold` | Confidence threshold for person detection | `0.6` |
| `target_classes` | List of target classes to detect (or `all`) | `[person]` |
| `track_threshold` | Confidence threshold for tracking (ByteTrack) | `0.45` |
| `track_buffer` | Number of frames to keep lost tracks before removing | `30` |
| `match_threshold` | IoU threshold for matching detections to tracks | `0.8` |
| `frame_rate` | Expected frame rate of the video stream (fps) | `30` |
| `image_timeout` | Timeout for shutting down after video ends (s) | `2.0` |
| `verbose_mode` | Enable visualization and detailed logging | `false` |

> **Note:** Enabling `verbose_mode` (`true`) activates real-time visualization via OpenCV windows.

## 🚀 Running

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch the person_detection node
ros2 launch person_detection person_detection.launch.py
```

> This launch starts **only** the person_detection node. It expects the camera
> images to already be published — from a shared camera brought up by the
> `overt_attention` system (`attention_system.launch.py`), a standalone camera
> driver, or a `ros2 bag`.
>
> The exact topics it subscribes to depend on the `camera` parameter and are
> resolved from [`data/pepper_topics.yaml`](data/pepper_topics.yaml):
>
> | `camera` | RGB topic | Depth topic |
> |----------|-----------|-------------|
> | `pepper` (default) | `PepperFrontCamera` → `/pepper/front/image_raw` | depth disabled for Pepper |
> | `realsense` / `video` | `RealSenseCameraRGB` → `/camera/color/image_raw_custom` | `RealSenseCameraDepth` → `/camera/aligned_depth_to_color/image_raw_custom` |
>
> Edit `pepper_topics.yaml` to point these at whatever your camera source
> actually publishes.

`ros2 run person_detection person_detection` runs the node without the launch
file. Either way it starts unconfigured: run
`ros2 lifecycle set /person_detection configure`, then `activate`, or use
`dec_launch`'s `dec_system.launch.py`, which does it for you.

## 🖥️ ROS Interface

### Subscribed Topics

Topic names are resolved from [`data/pepper_topics.yaml`](data/pepper_topics.yaml)
according to the `camera` parameter (see the table under *Running the Node*).

| Topic (key in `pepper_topics.yaml`) | Type | Description |
|-------|------|-------------|
| RGB (`pepper`: `/pepper/front/image_raw`, `realsense`: `/camera/color/image_raw_custom`) | `sensor_msgs/Image` | Color image from camera |
| Depth (`realsense` only: `/camera/aligned_depth_to_color/image_raw_custom`) | `sensor_msgs/Image` | Depth image (disabled for Pepper) |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/person_detection/data` | `dec_interfaces/msg/PersonDetection` | Detected persons with tracking IDs |
| `/person_detection/debug` | `sensor_msgs/Image` | Debug RGB image with detection overlays |
| `/person_detection/depth_debug` | `sensor_msgs/Image` | Debug colorized depth image |

## 📨 Message Structure

### `/person_detection/data` (`dec_interfaces/msg/PersonDetection`)

| Field | Type | Description |
|-------|------|-------------|
| `person_label_id[]` | string[] | Array of unique tracking IDs |
| `class_names[]` | string[] | Array of class names (always `person`) |
| `class_ids[]` | int32[] | Array of COCO class IDs (always `0`) |
| `confidences[]` | float32[] | Array of detection confidence scores |
| `centroids[]` | `geometry_msgs/Point[]` | Array of centroid coordinates (z = depth in meters) |
| `width[]` | float32[] | Array of bounding box widths |
| `height[]` | float32[] | Array of bounding box heights |

## 📁 Package Structure

```
person_detection/
├── config/
│   └── person_detection_configuration.yaml         # ROS2 parameters
├── data/
│   └── pepper_topics.yaml                          # topic name overrides
├── launch/
│   └── person_detection.launch.py
├── models/
│   └── person_detection_yolov11m.onnx              # YOLOv11m detector weights
├── include/person_detection/
│   └── person_detection_interface.h                # node/class declarations
├── src/
│   ├── person_detection_application.cpp            # node entry point (main)
│   └── person_detection_implementation.cpp         # YOLO inference + ByteTrack
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🏗️ Architecture

The person detection system consists of two main components:

1. **Camera Driver**: Provides synchronized RGB-D image streams
2. **Person Detection Node**:
   - Receives image streams from the camera
   - Performs person detection using YOLO model
   - Tracks persons across frames using ByteTrack algorithm
   - Publishes person detection results

Tracking uses the shared ByteTrack tracker from `dec_common`; the
[`dec_common` README](../dec_common/README.md#bytetrack-tracker) explains the algorithm.

## 🧪 Testing

```bash
cd ~/ros2_ws
colcon test --packages-select person_detection
colcon test-result --verbose
```

Runs unit tests for `getClassIndices()`, plus a bag-replay regression test that runs the real node on 12 recorded camera frames.

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 🧠 Pretrained Models
This package's own code is BSD-3-Clause, but the bundled YOLO11m detector
weights (`person_detection_yolov11m.onnx`) are **AGPL-3.0** upstream, not
BSD — see [MODELS.md](../MODELS.md) at the repo root for full attribution
and licensing details on every model used across pepper4dec.

## 📜 License
Copyright (C) 2025 Carnegie Mellon University Africa
Licensed under the BSD-3-Clause License. See individual package licenses for details.
