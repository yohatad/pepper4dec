<div align="center">
<h1>Overt Visual Attention System</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Overt Visual Attention System** package implements a unified visual attention controller for robot heads. It integrates multiple attention cues including face detection with engagement awareness, bottom-up visual saliency, and optional audio localization to generate natural, human-like head movements. The system prioritizes engaged faces, then detected faces, and finally saliency peaks with inhibition of return (IOR) to prevent repetitive scanning.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Multi-modal Attention**: Integrates face detection, visual saliency, and audio cues
- **Engagement Awareness**: Prioritizes faces with mutual gaze (engaged attention)
- **Boolean Map Saliency (BMS)**: Computes bottom-up visual attention
- **Inhibition of Return (IOR)**: Prevents repetitive scanning of the same locations
- **Real-time Processing**: Processes multiple attention cues simultaneously
- **Visualization**: Real-time visualization of attention targets and saliency peaks

## ✅ Prerequisites
- **ROS2 Humble** or newer
- **OpenCV** (C++, via `find_package(OpenCV)`) and **yaml-cpp**
- **Face Detection Node**: Requires the `face_detection` package
- **Camera System**: RGB(-D) camera (RealSense, Pepper camera, or similar)

## 🛠️ Installation

### Package Installation

```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace
cd ~/ros2_ws
colcon build --packages-up-to overt_attention
source install/setup.bash
```

## 🔧 Configuration

Configuration is managed via ROS2 parameters, loaded from `config/overt_attention_configuration.yaml`
(`ros2 param get/set <node> <name>` also works at runtime). The package has three nodes, each with
its own parameter set; a shared `/**` block applies `use_compressed`/`camera_type` to all three.

**Shared (`/**`, all nodes)**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_compressed` | Use compressed ROS image topics | `false` |
| `camera_type` | Camera to use: `"pepper"` or `"realsense"` | `pepper` |

**`saliency_node`**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `publish_map` | Publish saliency map visualization | `true` |
| `down_w` / `down_h` | Downsampled width/height used for saliency computation (px) | `160` / `120` |
| `use_depth_weighting` | Weight saliency by depth proximity | `true` |
| `depth_min_m` / `depth_max_m` | Depth range considered for weighting (m) | `0.3` / `10.0` |
| `depth_weight_min` | Minimum depth weight applied outside range | `0.2` |
| `min_peak` | Minimum saliency score to count as a peak | `0.5` |
| `overlay_alpha` | Blend alpha for the saliency overlay | `0.4` |
| `num_peaks` | Max number of saliency peaks to report | `5` |
| `peak_min_distance_px` | Minimum pixel distance between reported peaks | `50` |
| `process_hz` | Saliency computation rate (Hz) | `1.0` |

**`unified_attention_node`**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_enabled` | Whether attention is active on startup | `true` |
| `move_to_default_on_disable` | Move head to default pose when disabled | `true` |
| `default_yaw` / `default_pitch` | Default head pose (radians) | `0.0` / `-0.2` |
| `default_move_speed` | ALMotion speed used moving to default pose | `0.1` |
| `face_yaw_lim` | Head yaw joint limit for face tracking (radians) | `1.8` |
| `face_pitch_up` / `face_pitch_dn` | Head pitch limits for face tracking (radians) | `0.4` / `-0.7` |
| `saliency_yaw_lim` | Head yaw joint limit for saliency (radians) | `1.2` |
| `saliency_pitch_up` / `saliency_pitch_dn` | Head pitch limits for saliency (radians) | `0.3` / `-0.3` |
| `face_timeout` | Time after losing faces before switching to saliency (s) | `2.0` |
| `engaged_priority_bonus` | Priority multiplier for engaged (mutual-gaze) faces | `2.0` |
| `face_switch_cooldown` | Minimum time between switching tracked faces (s) | `1.0` |
| `same_face_threshold_deg` | Angular threshold to treat a face as "the same" target | `8.0` |
| `prefer_closer_faces` | Prefer nearer faces when scoring candidates | `true` |
| `max_face_distance` | Max face depth considered for tracking (m) | `5.0` |
| `min_angular_change_deg` | Dead-zone: skip a head command if already this close (deg) | `2.0` |
| `target_smoothing_alpha` | EMA weight for new target (0=frozen, 1=raw) | `0.4` |
| `saliency_min_score` | Minimum saliency score to consider switching to it | `0.30` |
| `saliency_min_cooldown` | Minimum dwell before switching saliency targets (s) | `1.5` |
| `saliency_max_dwell` | Max time to stay on one saliency target (s) | `3.0` |
| `switch_score_ratio` | Required score ratio to switch saliency targets | `1.4` |
| `same_target_threshold_deg` | Angular threshold to treat a saliency peak as "the same" | `5.0` |
| `enable_ior` | Enable Inhibition of Return | `true` |
| `ior_max_suppression` | Max suppression strength for a visited location | `0.9` |
| `ior_half_life` | Half-life for IOR decay (s) | `3.0` |
| `ior_radius_deg` | Angular radius of an IOR-suppressed region (deg) | `15.0` |
| `ior_cleanup_threshold` | Suppression value below which an IOR entry is dropped | `0.05` |
| `ior_max_locations` | Max number of IOR-suppressed locations tracked | `20` |

**`attention_visualization`**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `publish_overlay` | Publish the annotated visualization image | `true` |
| `publish_markers` | Publish RViz markers | `true` |
| `show_metrics` | Draw metrics text on the overlay | `true` |
| `show_face_ids` | Draw face IDs on the overlay | `true` |
| `show_depth` | Draw per-face depth on the overlay | `true` |
| `show_engagement` | Draw engagement/mutual-gaze status on the overlay | `true` |

Topic names (face detection, saliency, camera, joint angles, target angles) are configured separately
via `data/pepper_topics.yaml`, keyed by `camera_type` (`pepper` vs `realsense`).

## 🚀 Running the Node

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

## 🖥️ ROS Interface

### Subscribed Topics

Camera/depth/camera-info topics are resolved from `data/pepper_topics.yaml` based on the
`camera_type` parameter. With the default `camera_type: "pepper"`:

| Topic | Type | Description |
|-------|------|-------------|
| `/face_detection/data` | `dec_interfaces/msg/FaceDetection` | Face detection messages (`unified_attention_node`) |
| `/pepper/front/image_raw` | `sensor_msgs/Image` | RGB image from camera (`saliency_node`, `attention_visualization`) |
| `/naoqi_driver/camera/depth/image_raw` | `sensor_msgs/Image` | Depth image (`saliency_node`, if `use_depth_weighting`) |
| `/pepper/front/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics (`attention_visualization`) |
| `/joint_states` | `sensor_msgs/JointState` | Current head position (`unified_attention_node`) |

With `camera_type: "realsense"`, these resolve instead to `/camera/color/image_raw_custom`,
`/camera/aligned_depth_to_color/image_raw_custom`, and `/camera/color/camera_info`. When
`use_compressed: true`, the image/depth subscriptions switch to `sensor_msgs/CompressedImage`.

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

## 📁 Package Structure

```
overt_attention/
├── config/
│   └── overt_attention_configuration.yaml    # ROS2 parameters
├── data/
│   └── pepper_topics.yaml                    # topic name overrides
├── launch/
│   ├── attention_system.launch.py
│   └── realsense_camera.launch.py
├── include/
│   └── overt_attention/
│       └── overt_attention_interface.h       # shared class/struct declarations
├── src/
│   ├── overt_attention_utilities.cpp         # shared helpers
│   ├── overt_attention_saliency.cpp          # saliency node (BMS)
│   ├── overt_attention_unified_attention.cpp # attention controller node
│   └── overt_attention_visualization.cpp     # visualization node
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🏗️ Architecture

The overt attention system consists of three main nodes:

1. **Saliency Node**: Computes bottom-up visual attention using Boolean Map Saliency (BMS)
2. **Unified Attention Controller**: Priority-based attention selection with IOR
3. **Visualization Node**: Creates real-time visualization overlay

## 🧪 Testing

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

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.