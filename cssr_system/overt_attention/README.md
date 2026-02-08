<div align="center">
<h1> Overt Visual Attention System (ROS2) </h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Overt Visual Attention System** package is a **ROS2** package designed to implement a unified visual attention controller for robot heads. It integrates multiple attention cues including **face detection with engagement awareness**, **bottom-up visual saliency**, and optional **audio localization** to generate natural, human-like head movements. The system prioritizes engaged faces, then detected faces, and finally saliency peaks with inhibition of return (IOR) to prevent repetitive scanning of the same locations.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Multi-modal Attention**: Integrates face detection, visual saliency, and audio cues
- **Engagement Awareness**: Prioritizes faces with mutual gaze (engaged attention)
- **Boolean Map Saliency (BMS)**: Computes bottom-up visual attention using state-of-the-art saliency detection
- **Inhibition of Return (IOR)**: Prevents repetitive scanning of the same locations
- **Real-time Processing**: Processes multiple attention cues simultaneously
- **Configurable**: Extensive parameter tuning via YAML configuration
- **Visualization**: Real-time visualization of attention targets, faces, and saliency peaks
- **Robot Agnostic**: Works with any robot with head pan-tilt capabilities (Pepper, NAO, etc.)

# 🛠️ Installation 

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **OpenCV** and **NumPy** for image processing
- **Face Detection Node**: Requires the `face_detection` package for face input
- **Camera System**: RGB camera (RealSense, Pepper camera, or similar)
- **Robot Head Control**: NAOqi bridge or similar head control interface

## Package Installation

1. **Clone and Build the Workspace**
```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone <repository-url>

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select overt_attention cssr_interfaces
source install/setup.bash
```

2. **Install Python Dependencies**
```bash
# Install required Python packages
pip install opencv-python numpy scipy
```

3. **Ensure Required Packages are Built**
The attention system depends on:
- `cssr_interfaces`: Custom message definitions
- `face_detection`: For face detection input (optional but recommended)
- `naoqi_bridge_msgs`: For Pepper/NAO robot control (if using Pepper)

```bash
# Build all required packages
cd ~/ros2_ws
colcon build --packages-select cssr_interfaces overt_attention
source install/setup.bash
```

# 🔧 Configuration Parameters
The configuration is managed via `config/overt_attention_configuration.yaml`. The file contains parameters for three main nodes:

## Shared Parameters
| Parameter                   | Description                                                      | Range/Values            | Default Value |
|-----------------------------|------------------------------------------------------------------|-------------------------|---------------|
| `use_compressed`            | Use compressed ROS image topics                                  | `True`, `False`         | `False`       |

## Saliency Node Parameters
| Parameter                   | Description                                                      | Range/Values            | Default Value |
|-----------------------------|------------------------------------------------------------------|-------------------------|---------------|
| `image_topic_base`          | Base topic for RGB images                                        | String                  | `/camera/color/image_raw` |
| `publish_map`               | Publish saliency map visualization                               | `True`, `False`         | `True`        |
| `down_w`, `down_h`          | Downsampled image dimensions for processing                      | Integers                | `160`, `120`  |
| `min_peak`                  | Minimum saliency score to consider as a peak                     | `[0.0 - 1.0]`           | `0.5`         |
| `num_peaks`                 | Maximum number of saliency peaks to publish                      | Integer                 | `5`           |
| `peak_min_distance_px`      | Minimum distance between peaks (in downsampled pixels)           | Integer                 | `100`         |

## Unified Attention Node Parameters
| Parameter                   | Description                                                      | Range/Values            | Default Value |
|-----------------------------|------------------------------------------------------------------|-------------------------|---------------|
| `face_topic`                | Topic for face detection messages                                | String                  | `/faceDetection/data` |
| `saliency_topic`            | Topic for saliency peak messages                                 | String                  | `/attn/saliency_peak` |
| `camera_info_topic`         | Topic for camera intrinsics                                      | String                  | `/camera/color/camera_info` |
| `head_command_topic`        | Topic for head joint commands                                    | String                  | `/joint_angles` |
| `engaged_priority_bonus`    | Priority multiplier for engaged faces                            | Float                   | `2.0`         |
| `face_timeout`              | Time after losing faces before switching to saliency (seconds)   | Float                   | `2.0`         |
| `saliency_min_score`        | Minimum saliency score to consider                               | `[0.0 - 1.0]`           | `0.30`        |
| `enable_ior`                | Enable Inhibition of Return                                      | `True`, `False`         | `True`        |
| `ior_half_life`             | Half-life for IOR decay (seconds)                                | Float                   | `3.0`         |
| `ior_radius_deg`            | Angular radius for IOR suppression (degrees)                     | Float                   | `15.0`        |
| `yaw_lim`                   | Head yaw joint limit (radians)                                   | Float                   | `1.8`         |
| `pitch_up`, `pitch_dn`      | Head pitch up/down limits (radians)                              | Float                   | `0.4`, `-0.7` |

## Visualization Node Parameters
| Parameter                   | Description                                                      | Range/Values            | Default Value |
|-----------------------------|------------------------------------------------------------------|-------------------------|---------------|
| `publish_overlay`           | Publish visualization overlay image                              | `True`, `False`         | `True`        |
| `publish_markers`           | Publish ROS markers for visualization                            | `True`, `False`         | `True`        |
| `show_metrics`              | Show performance metrics on visualization                        | `True`, `False`         | `True`        |

# 🚀 Running the Node

## Launch All Components
The launch file starts all three attention system nodes:

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch the complete attention system
ros2 launch overt_attention attention_system.launch.py
```

## Launch Arguments
The launch file supports the following arguments:

| Argument        | Description                                      | Default | Example                    |
|-----------------|--------------------------------------------------|---------|----------------------------|
| `params_file`   | Path to parameters YAML file                     | Auto-detected | `params_file:=/path/to/custom.yaml` |
| `enable_viz`    | Enable visualization node                        | `true`  | `enable_viz:=false`        |

### Example: Custom Configuration
```bash
ros2 launch overt_attention attention_system.launch.py \
  params_file:=/path/to/custom_config.yaml \
  enable_viz:=true
```

## Manual Node Execution
You can also run nodes individually:

1. **Start Saliency Node**:
```bash
ros2 run overt_attention saliency_node \
  --ros-args \
  -p use_compressed:=false \
  -p image_topic_base:="/camera/color/image_raw"
```

2. **Start Unified Attention Controller**:
```bash
ros2 run overt_attention unified_attention_node \
  --ros-args \
  -p face_topic:="/faceDetection/data" \
  -p saliency_topic:="/attn/saliency_peak" \
  -p head_command_topic:="/joint_angles"
```

3. **Start Visualization Node**:
```bash
ros2 run overt_attention visualization_node \
  --ros-args \
  -p publish_overlay:=true \
  -p show_metrics:=true
```

## Required Input Topics
The attention system requires the following input topics:
- **Face Detection**: `/faceDetection/data` (`cssr_interfaces/msg/FaceDetection`)
- **Camera Images**: `/camera/color/image_raw` (`sensor_msgs/msg/Image`)
- **Camera Info**: `/camera/color/camera_info` (`sensor_msgs/msg/CameraInfo`)
- **Joint States**: `/joint_states` (`sensor_msgs/msg/JointState`) - for current head position

## Optional Input Topics
- **Audio Azimuth**: `/audio/azimuth_rad` (`std_msgs/msg/Float32`) - for audio-driven attention
- **Depth Images**: `/camera/aligned_depth_to_color/image_raw` - for depth-based filtering

# 🖥️ Output
The system publishes attention commands and visualization data to multiple topics.

## Output Topics
- **Head Commands**: `/joint_angles` (`naoqi_bridge_msgs/msg/JointAnglesWithSpeed`) - Head joint commands
- **Attention Targets**: `/attn/target_angles` (`geometry_msgs/msg/Vector3`) - Current attention target (yaw, pitch, score)
- **Saliency Peaks**: `/attn/saliency_peak` (`std_msgs/msg/Float32MultiArray`) - Detected saliency peaks [u1, v1, score1, u2, v2, score2, ...]
- **Visualization**: `/attn/visualization` (`sensor_msgs/msg/Image`) - Annotated visualization overlay
- **Saliency Map**: `/attn/saliency_map/compressed` (`sensor_msgs/msg/CompressedImage`) - Saliency heatmap visualization

## Service
- **Enable/Disable**: `/attn/set_enabled` (`std_srvs/SetBool`) - Service to enable/disable the attention system

## Verification
To verify the system is working:

```bash
# Monitor attention targets
ros2 topic echo /attn/target_angles

# Monitor head commands
ros2 topic echo /joint_angles

# Check all nodes are running
ros2 node list

# Check all topics
ros2 topic list | grep attn

# Enable/disable attention system
ros2 service call /attn/set_enabled std_srvs/SetBool "{data: false}"
ros2 service call /attn/set_enabled std_srvs/SetBool "{data: true}"
```

# 🏗️ Architecture
The overt attention system consists of three main nodes:

1. **Saliency Node**: 
   - Computes bottom-up visual attention using Boolean Map Saliency (BMS)
   - Publishes multiple saliency peaks with scores
   - Optional depth-based weighting for proximity bias
   - Configurable downsampling for performance

2. **Unified Attention Controller**:
   - **Priority 1**: Engaged faces (mutual gaze) with bonus scoring
   - **Priority 2**: Detected faces with depth and center bias
   - **Priority 3**: Saliency peaks with cooldown and Inhibition of Return (IOR)
   - Implements face switching hysteresis to prevent rapid target changes
   - Applies joint limits and smooth motion constraints
   - Supports enable/disable service with default position return

3. **Visualization Node**:
   - Creates real-time visualization overlay showing:
     - Face bounding boxes with tracking IDs and engagement status
     - Saliency peaks with scores
     - Current attention target
     - Performance metrics
   - Publishes annotated image for monitoring

## Attention Prioritization Logic
1. **Engaged Faces**: Faces with mutual gaze receive highest priority (2x bonus)
2. **Detected Faces**: Other faces scored by distance from center, depth, and continuity
3. **Saliency Peaks**: When no recent faces, switches to saliency with IOR cooldown
4. **Inhibition of Return**: Recently visited locations are suppressed to encourage exploration

# 💡 Usage Examples

## Basic Usage with Pepper Robot
```bash
# Start face detection system
ros2 launch face_detection face_detection_launch_robot.launch.py

# Start attention system
ros2 launch overt_attention attention_system.launch.py

# Enable attention system (if not auto-started)
ros2 service call /attn/set_enabled std_srvs/SetBool "{data: true}"
```

## Testing with Recorded Data
```bash
# Play ROS bag with camera data
ros2 bag play recorded_data.db3

# Launch attention system without camera
ros2 launch overt_attention attention_system.launch.py

# Monitor visualization
ros2 run rqt_image_view rqt_image_view /attn/visualization
```

## Custom Configuration for Different Robots
Create a custom YAML configuration file:
```yaml
unified_attention_node:
  ros__parameters:
    yaw_lim: 2.0  # Increased yaw limit
    pitch_up: 0.5
    pitch_dn: -0.5
    head_command_topic: "/custom_head_commands"
```

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohanneh@andrew.cmu.edu">yohanneh@andrew.cmu.edu</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2023 Upanzi Network   

2026-02-08
