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
| `image_topic_base` | Base topic for RGB images | `/camera/color/image_raw` |
| `publish_map` | Publish saliency map visualization | `True` |
| `face_topic` | Topic for face detection messages | `/faceDetection/data` |
| `saliency_topic` | Topic for saliency peak messages | `/attn/saliency_peak` |
| `head_command_topic` | Topic for head joint commands | `/joint_angles` |
| `engaged_priority_bonus` | Priority multiplier for engaged faces | `2.0` |
| `face_timeout` | Time after losing faces before switching to saliency (s) | `2.0` |
| `enable_ior` | Enable Inhibition of Return | `True` |
| `ior_half_life` | Half-life for IOR decay (seconds) | `3.0` |
| `yaw_lim` | Head yaw joint limit (radians) | `1.8` |
| `pitch_up` | Head pitch up limit (radians) | `0.4` |
| `pitch_dn` | Head pitch down limit (radians) | `-0.7` |

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
ros2 run overt_attention saliency_node \
  --ros-args \
  -p use_compressed:=false \
  -p image_topic_base:="/camera/color/image_raw"

# Start Unified Attention Controller
ros2 run overt_attention unified_attention_node \
  --ros-args \
  -p face_topic:="/faceDetection/data" \
  -p saliency_topic:="/attn/saliency_peak" \
  -p head_command_topic:="/joint_angles"

# Start Visualization Node
ros2 run overt_attention visualization_node \
  --ros-args \
  -p publish_overlay:=true \
  -p show_metrics:=true
```

## ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/faceDetection/data` | `dec_interfaces/msg/FaceDetection` | Face detection messages |
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB image from camera |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |
| `/joint_states` | `sensor_msgs/JointState` | Current head position |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_angles` | `naoqi_bridge_msgs/msg/JointAnglesWithSpeed` | Head joint commands |
| `/attn/target_angles` | `geometry_msgs/msg/Vector3` | Current attention target (yaw, pitch, score) |
| `/attn/saliency_peak` | `std_msgs/msg/Float32MultiArray` | Detected saliency peaks |
| `/attn/visualization` | `sensor_msgs/Image` | Annotated visualization overlay |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/attn/set_enabled` | `std_srvs/SetBool` | Enable/disable the attention system |

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
├── resource/
│   └── overt_attention
├── overt_attention/
│   ├── __init__.py
│   ├── overt_attention_application.py
│   ├── overt_attention_implementation.py
│   ├── saliency_node.py
│   ├── unified_attention_node.py
│   └── visualization_node.py
├── package.xml
├── setup.py
├── setup.cfg
├── requirements.txt
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
ros2 topic echo /attn/target_angles

# Monitor head commands
ros2 topic echo /joint_angles

# Enable/disable attention system
ros2 service call /attn/set_enabled std_srvs/SetBool "{data: false}"
ros2 service call /attn/set_enabled std_srvs/SetBool "{data: true}"
```

## Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.