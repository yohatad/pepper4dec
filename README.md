<div align="center">

# Pepper4DEC: Autonomous Humanoid Guide Platform

<img src="images/upanzi-logo.svg" alt="Upanzi Logo" width="800px">

[![CI](https://github.com/yohatad/pepper4dec/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yohatad/pepper4dec/actions/workflows/ci.yml)
![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-blue)
![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-green)

</div>

## 📋 Overview

Pepper4DEC is a ROS 2 Humble stack that lets a SoftBank Pepper humanoid work autonomously as a guide in a public space. The robot navigates between points of interest on a Nav2 stack with interchangeable localization backends, detects and tracks visitors, engages whoever makes eye contact, holds LLM-backed dialogue about what it is showing, and drives speech, gaze and gestures from a single behavior tree, with no operator in the loop.

It is built as a configurable platform rather than a one-off demo. Everything that defines a deployment is data, not code:

- **The mission** - a BehaviorTree.CPP XML selected by parameter (`dec_Tour.xml`, `lab_tour.xml`, ...)
- **The venue** - an environment knowledge base describing locations and exhibits (`decEnvironmentKnowledgeBase.yaml`, `labEnvironmentKnowledgeBase.yaml`)
- **The dialogue domain** - a JSON knowledge base and system prompt behind the RAG conversation manager
- **The cultural register** - `cultureKnowledgeBase.yaml`, governing greetings, gestures and engagement norms

Repointing the robot at a new venue, or at a different front-of-house role such as a receptionist or information desk, is a matter of supplying new data files. The reference deployment runs guided tours of the Upanzi Digital Experience Center (DEC).

## 🖥️ Hardware Setup

<div align="center">
<table>
<tr>
<td align="center" width="50%">
<img src="images/Full_pepper_image.jpg" alt="Pepper with the sensor collar and onboard compute" width="290px"><br>
<em>Pepper carrying the sensor collar and the onboard compute board at the waist</em>
</td>
<td align="center" width="50%">
<img src="images/Lidar_and_RealSense.jpg" alt="Unitree L2 lidar and Intel RealSense on the sensor collar" width="290px"><br>
<em>The 3D-printed collar: Unitree L2 lidar above the Intel RealSense, both facing forward</em>
</td>
</tr>
</table>
</div>

## 🏗️ System Architecture

<div align="center">
<img src="images/System_arch.png" alt="System Architecture" width="1200px">
</div>

The system is built on **ROS2 (Humble)** and follows a modular architecture with specialized packages handling different aspects of robot behavior and perception. Most nodes are **managed lifecycle nodes**, sequenced `unconfigured → active` by a `nav2_lifecycle_manager` instance in `dec_launch`.

### **Core Control Packages**
- **`behavior_controller`** - Mission orchestrator built on **BehaviorTree.CPP v4** + **BehaviorTree.ROS2**. Loads a mission as an XML behavior tree (`data/dec_Tour.xml`, `data/asr_cm_tts_pipeline.xml`, …) and dispatches ROS2 actions to speech, gesture, navigation, conversation, and face-detection nodes; publishes BT state for live Groot2 visualization
- **`animate_behavior`** - Idle/social body animation at 30 Hz with exponential smoothing, plus a synchronized face-LED cascade
- **`conversation_manager`** - RAG dialogue manager (ChromaDB vector store + any OpenAI-compatible LLM) exposed as a ROS2 action, with conversation memory and NAOqi prosody-tagged output
- **`gesture_execution`** - Deictic (pointing), iconic, bowing and nodding gestures with Bézier interpolation, IK, and joint-limit validation
- **`speech_event`** - Whisper ASR with Silero VAD, a post-VAD/pre-ASR noise-reduction pipeline, and optional SRP-PHAT sound-source localization on Pepper's 4-mic array
- **`text_to_speech`** - Streaming TTS across five backends (naoqi_ros, kokoro_local/pepper, elevenlabs_local/pepper) with sentence queueing and automatic mic muting

### **Perception & Attention Packages**
- **`face_detection`** - Real-time face detection, head pose estimation, and mutual gaze detection (SixDRepNet), plus age/gender estimation (MiVOLO) for persons exhibiting mutual gaze — two lifecycle nodes, `face_detection` and `age_gender_detection`, in one package
- **`person_detection`** - YOLO-based person detection and ByteTrack multi-person tracking for scene understanding
- **`overt_attention`** - Unified head-attention controller: engaged faces → detected faces → Boolean Map Saliency peaks, with inhibition of return

### **Navigation & Localization**
- **`pepper_slam`** - 3D mapping and odometry backends: FAST-LIO / Point-LIO lidar-inertial mapping and odometry on the Unitree L2, and RTAB-Map (RGB-D). Launch files and parameters only; the SLAM backends themselves are upstream packages
- **`pepper_navigation`** - Nav2 stack (path planning, obstacle avoidance, keepout zones, collision-monitor safety layer) localizing by default with FAST-LIO against a prior 3D map (`fastlio_localization`). The localization backend is a launch-time profile behind shared costmaps and tuning, so alternatives can be swapped in and compared directly

Localization-only deployments get their `map → base_footprint` pose (`/localization/pose`) from **`fast_lio`**'s `fastlio_localization` node; `gesture_execution` consumes that pose for pointing IK.

### **Infrastructure & Utilities**
- **`dec_launch`** - System launch files, lifecycle sequencing, and startup configurations
- **`dec_interfaces`** - Custom ROS2 message, service, and action definitions
- **`dec_common`** - Shared C++ utilities: the camera lifecycle node base class, the ByteTrack multi-object tracker, and ROS2 parameter-loading helpers

## 🚀 Quick Start

### Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10+**
- **Pepper Robot** (or simulation environment)
- **Intel RealSense Camera** (for perception modules)
- **Unitree L2 Lidar** + the `l2lidar_node` driver (for LIO odometry and lidar-based localization)
- **NVIDIA GPU (CUDA)** - optional; accelerates the ONNX-based perception nodes (`face_detection`, `age_gender_detection`, `person_detection`), which fall back to CPU automatically if unavailable

### Installation

1. **Clone this repository and its sibling dependencies**

Several dependencies are not on the ROS index and must be cloned alongside this
repo (this is the same list the CI workflow installs — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

```bash
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Pepper / NAOqi
git clone https://github.com/yohatad/naoqi_bridge_msgs2.git naoqi_bridge_msgs
git clone https://github.com/yohatad/naoqi_driver2.git
git clone -b ros2 https://github.com/ros-naoqi/libqi.git naoqi_libqi
git clone -b ros2 https://github.com/ros-naoqi/libqicore.git naoqi_libqicore
git clone https://github.com/ros-naoqi/nao_meshes2.git nao_meshes
git clone https://github.com/ros-naoqi/pepper_meshes2.git pepper_meshes

# Behavior trees
git clone https://github.com/BehaviorTree/BehaviorTree.CPP.git
git clone https://github.com/BehaviorTree/BehaviorTree.ROS2.git
```

The lidar stack additionally needs the L2 driver (`l2lidar_node`, publishing
`/points` + `/imu/data`) and the LIO packages referenced by
`pepper_slam`/`pepper_navigation`: `fast_lio`, `point_lio`, and optionally
`fastlio_lc_pgo`. These are only required for the lidar-based
navigation profiles.

2. **Install the remaining dependencies and build**

```bash
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

3. **Set Up Python Environments**

Most perception/actuation packages (`animate_behavior`, `behavior_controller`, `face_detection`, `gesture_execution`, `overt_attention`, `person_detection`) are C++ and need no Python environment. The three Python packages (`conversation_manager`, `speech_event`, `text_to_speech`) each expect their own dedicated virtual environment under `~/ros2_ws/.venvs/` — see each package's own README for the exact venv name and `pip install -r requirements.txt` it expects. Pinned lockfiles used by the Docker image live in [`docker/requirements/`](docker/requirements/).

4. **Download Model Files**
   - Place required ONNX model files in their respective `models/` directories (`face_detection/models/`, `person_detection/models/`, `speech_event/models/`)
   - `models/` directories are **gitignored** — no weights are committed
   - Provenance and licensing for every pretrained model is recorded in [MODELS.md](MODELS.md); note that several are GPL-3.0/AGPL-3.0

## 🚀 Running the Tour System

### Basic Launch
```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch the complete system (requires all dependencies and robot hardware)
ros2 launch dec_launch dec_system.launch.py

# Pick a different Nav2 localization profile
ros2 launch dec_launch dec_system.launch.py nav_profile:=rtabmap_loc

# Perception + behavior only, without the Nav2 stack
# (fastlio_localization still comes up standalone, so /localization/pose exists)
ros2 launch dec_launch dec_system.launch.py enable_navigation:=false
```

Navigation defaults to the `fastloc` profile: FAST-LIO localizing against a prior 3D map. Other localization backends are selectable with `nav_profile:=`; see [pepper_navigation/README.md](pepper_navigation/README.md) for the full list and their trade-offs.

### Component-Based Launch
For development and testing, individual components can be launched:

1. **Launch Perception System** (shared camera + person/face detection + attention)
```bash
ros2 launch overt_attention attention_system.launch.py
```

2. **Launch Behavior Controller**
```bash
ros2 launch behavior_controller behavior_controller.launch.py
```

3. **Launch LIO odometry** (FAST-LIO or Point-LIO on the Unitree L2)
```bash
ros2 launch pepper_slam fastlio_odometry.launch.py
ros2 launch pepper_slam pointlio_odometry.launch.py
```

4. **Launch Nav2** (fastloc profile; other profiles are listed in [pepper_navigation/README.md](pepper_navigation/README.md))
```bash
ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py
```

5. **Replay a recorded bag** (static TF for sensors, no robot required)
```bash
ros2 launch dec_launch bag_static_tf.launch.py
```

### Configuration
Each package contains configuration files in their `config/` directories:
- `behavior_controller/config/behavior_controller_configuration.yaml` - Mission parameters and active scenario
- `face_detection/config/face_detection_configuration.yaml` - Perception settings
- `pepper_slam/config/` and `pepper_navigation/config/` - SLAM, EKF, costmap and Nav2 tuning
- Gesture, attention, speech and TTS parameters in their respective package configs

## 🧪 Testing

Every push to `main` or `devel` builds the workspace and runs the full suite in CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): 458 tests across the 14 packages, plus `ament_flake8`, `ament_pep257` and `ament_copyright` linting.

```bash
cd ~/ros2_ws
colcon test --packages-select dec_common person_detection   # or any subset
colcon test-result --verbose
```

Coverage by tier:

- **Unit, C++ (gtest)** - ByteTrack multi-object tracking, Pepper arm kinematics and joint-limit clamping, the animate_behavior motion math, behavior-tree utilities and knowledge-base validation, Boolean Map Saliency, age/gender temporal smoothing, camera-topic and pixel-to-angle helpers, COCO class filtering
- **Unit, Python (pytest)** - the LIO divergence guard, the map-leveling frame contract, sound-localization geometry, the speech denoiser DSP, LLM response parsing, TTS audio helpers, and a drift check on the Nav2 parameter sections shared across localization profiles
- **Integration** - a bag-replay regression test that runs the real YOLOv11 + ByteTrack person-detection node against recorded camera frames and asserts on the published detections. Frames are fed one at a time, so the result does not depend on machine speed

Tests that need an optional dependency (ONNX model weights, `librosa`, `pyroomacoustics`, ChromaDB) skip cleanly when it is absent, so the suite passes on a bare CI runner and exercises everything on a full workstation.

## 🐳 Docker

A CUDA-capable image and a Compose file are provided; the Dockerfile builds a
separate venv per Python node (see its header for the rationale).

```bash
cp .env.example .env             # DISPLAY, ROS_DOMAIN_ID, API keys
docker compose build             # requires a loaded SSH agent for the private fork
docker compose up pepper4dec
docker compose --profile viz up  # adds rviz2
docker compose --profile dev up  # live-mounts the source over the image
```

## 📊 Package Details

### **Face Detection System**
- **Algorithms**: SixDRepNet for head pose estimation, MiVOLO for age/gender estimation
- **Features**: Multi-face detection, mutual gaze evaluation; age/gender estimation is triggered for tracked persons exhibiting mutual gaze within range, with temporal smoothing across repeated estimates
- **Input**: RGB-D streams from RealSense or Pepper cameras
- **Output**: `/face_detection/data` (face centroids, dimensions, gaze status) from the `face_detection` node; `/face_detection/age_gender_results` (per-person age/gender JSON) from the separate `age_gender_detection` node
- **Performance**: Real-time processing with GPU acceleration support

### **Behavior Controller**
- **Function**: Mission orchestrator, built on BehaviorTree.CPP v4 / BehaviorTree.ROS2
- **Input**: XML behavior trees in `behavior_controller/data/`, plus culture and environment knowledge bases (YAML)
- **Coordination**: Dispatches ROS2 actions to speech, TTS, gesture, navigation, conversation and face-detection nodes
- **Adaptation**: Intent-aware routing — the `ConversationManager` BT node exposes `intent`/`confidence` output ports the tree branches on
- **Tooling**: Publishes BT state for live Groot2 visualization

### **Navigation System**
- **Mapping** (`pepper_slam`): FAST-LIO / Point-LIO 3D lidar-inertial mapping and odometry on the Unitree L2, or RTAB-Map (RGB-D)
- **Localization** (`pepper_navigation`): FAST-LIO against a prior 3D map via `fastlio_localization` by default; other backends are selectable as launch-time profiles sharing the same costmaps and tuning (see [pepper_navigation/README.md](pepper_navigation/README.md))
- **Odometry**: wheel odometry (`/pepper_odom` from `naoqi_driver2`), LIO odometry
- **Path Planning**: Nav2 with 3D voxel costmaps consuming the L2's 360° `PointCloud2` directly, no flattening step
- **Safety**: An independent collision monitor gates every velocity command straight off the lidar, bypassing the costmaps
- **Integration**: Full coordination with the behavior controller

## 📚 Documentation

Detailed documentation is available:
- **Package-specific READMEs** in each package directory
- **Configuration guides** in config directories
- **Model provenance and licensing**: [MODELS.md](MODELS.md)
- **API documentation**: `ros2 interface show dec_interfaces/`
<!-- - **Deliverable reports**: [DEC4Africa Deliverables](https://dec4africa.github.io/deliverables/) -->

## 🎓 Background

Developed at Carnegie Mellon University Africa as a spin-off of the **Culturally Sensitive Social Robotics for Africa (CSSR4Africa)** project. The DEC deployment replaces repetitive, human-led walkthroughs of the center's Digital Public Infrastructure demo (biometric enrollment with MOSIP, financial transactions with MIFOS, and subsidy validation with UPMS) and doubles as a testbed for culturally aware human-robot interaction: visitor interactions are logged to support research on cross-cultural behavior modeling and adaptive dialogue management in public spaces.

## ❓ Support

For issues or questions:
- **Contact**: 
  - [yohatad123@gmail.com](mailto:yohatad123@gmail.com)
## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details. Third-party pretrained model weights are **not** covered by this license — see [MODELS.md](MODELS.md).
