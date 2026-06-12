<div align="center">
<h1>Overt Visual Attention System</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Overt Visual Attention System** package implements a unified visual attention controller for robot heads. It integrates multiple attention cues including face detection with engagement awareness, bottom-up visual saliency, and optional audio localization to generate natural, human-like head movements. The system prioritizes engaged faces, then detected faces, and finally saliency peaks with inhibition of return (IOR) to prevent repetitive scanning.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Multi-modal Attention**: Integrates face detection, visual saliency, and audio cues
- **Engagement Awareness**: Prioritizes faces with mutual gaze (engaged attention)
- **Boolean Map Saliency (BMS)**: Computes bottom-up visual attention
- **Inhibition of Return (IOR)**: Prevents repetitive scanning of the same locations
- **Real-time Processing**: Processes multiple attention cues simultaneously
- **Visualization**: Real-time visualization of attention targets and saliency peaks

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **OpenCV** and **NumPy** for image processing
- **Face Detection Node**: Requires the `face_detection` package
- **Camera System**: RGB camera (RealSense, Pepper camera, or similar)

## Installation

### Package Installation

```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select overt_attention dec_interfaces
source install/setup.bash
```

### Python Dependencies

```bash
pip install opencv-python numpy scipy
```

## Configuration

Configuration is managed via `config/overt_attention_configuration.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_compressed` | Use compressed ROS image topics | `False` |
| `publish_map` | Publish saliency map visualization | `True` |
| `engaged_priority_bonus` | Priority multiplier for engaged faces | `2.0` |
| `face_timeout` | Time after losing faces before switching to saliency (s) | `2.0` |
| `enable_ior` | Enable Inhibition of Return | `True` |
| `ior_half_life` | Half-life for IOR decay (seconds) | `3.0` |
| `face_yaw_lim` | Head yaw joint limit for face tracking (radians) | `1.8` |
| `face_pitch_up` | Head pitch up limit for face tracking (radians) | `0.4` |
| `face_pitch_dn` | Head pitch down limit for face tracking (radians) | `-0.7` |
| `saliency_yaw_lim` | Head yaw joint limit for saliency (radians) | `1.2` |
| `saliency_pitch_up` | Head pitch up limit for saliency (radians) | `0.3` |
| `saliency_pitch_dn` | Head pitch down limit for saliency (radians) | `-0.3` |

Topic names (face detection, saliency, camera, joint angles, target angles) are configured separately via `data/pepper_topics.yaml`.

## Running the Node

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch the complete attention system
ros2 launch overt_attention attention_system.launch.py

# Custom configuration
ros2 launch overt_attention attention_system.launch.py \
  params_file:=/path/to/custom_config.yaml \
  enable_viz:=true
```

### Manual Node Execution

```bash
# Start Saliency Node
ros2 run overt_attention overt_attention_saliency \
  --ros-args \
  -p use_compressed:=false \
  -p publish_map:=true

# Start Unified Attention Controller
ros2 run overt_attention overt_attention_unified_attention \
  --ros-args \
  -p engaged_priority_bonus:=2.0 \
  -p face_timeout:=2.0

# Start Visualization Node
ros2 run overt_attention overt_attention_visualization \
  --ros-args \
  -p publish_overlay:=true \
  -p show_metrics:=true
```

## ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/face_detection/data` | `dec_interfaces/msg/FaceDetection` | Face detection messages |
| `/camera/color/image_raw_custom` | `sensor_msgs/Image` | RGB image from camera |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |
| `/joint_states` | `sensor_msgs/JointState` | Current head position |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_angles` | `naoqi_bridge_msgs/msg/JointAnglesWithSpeed` | Head joint commands |
| `/overt_attention/target_angles` | `geometry_msgs/msg/Vector3` | Current attention target (yaw, pitch, score) |
| `/overt_attention/saliency_peak` | `std_msgs/msg/Float32MultiArray` | Detected saliency peaks |
| `/overt_attention/visualization` | `sensor_msgs/Image` | Annotated visualization overlay |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/overt_attention/set_enabled` | `std_srvs/SetBool` | Enable/disable the attention system |

## Attention Prioritization Logic

1. **Engaged Faces**: Faces with mutual gaze receive highest priority (2x bonus)
2. **Detected Faces**: Other faces scored by distance from center, depth, and continuity
3. **Saliency Peaks**: When no recent faces, switches to saliency with IOR cooldown
4. **Inhibition of Return**: Recently visited locations are suppressed to encourage exploration

## Package Structure

```
overt_attention/
├── config/
│   └── overt_attention_configuration.yaml
├── data/
│   └── pepper_topics.yaml
├── launch/
│   ├── attention_system.launch.py
│   └── realsense_camera.launch.py
├── include/
│   └── overt_attention/
│       └── overt_attention_interface.h
├── src/
│   ├── overt_attention_utilities.cpp
│   ├── overt_attention_saliency.cpp
│   ├── overt_attention_unified_attention.cpp
│   └── overt_attention_visualization.cpp
├── CMakeLists.txt
├── package.xml
└── README.md
```

## Architecture

The overt attention system consists of three main nodes:

1. **Saliency Node**: Computes bottom-up visual attention using Boolean Map Saliency (BMS)
2. **Unified Attention Controller**: Priority-based attention selection with IOR
3. **Visualization Node**: Creates real-time visualization overlay

## Testing

```bash
# Check node is running
ros2 node list

# Monitor attention targets
ros2 topic echo /overt_attention/target_angles

# Monitor head commands
ros2 topic echo /joint_angles

# Enable/disable attention system
ros2 service call /overt_attention/set_enabled std_srvs/SetBool "{data: false}"
ros2 service call /overt_attention/set_enabled std_srvs/SetBool "{data: true}"
```

## Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.