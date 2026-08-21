<div align="center">

# Pepper Robot Tour – Digital Experience Center (DEC)

<img src="images/upanzi-logo.svg" alt="Upanzi Logo" width="800px">

</div>

## 📋 Overview

This repository contains the complete software stack for an autonomous Pepper robot-led tour at the Upanzi Digital Experience Center (DEC), developed as a spin-off of the **Culturally Sensitive Social Robotics for Africa (CSSR4Africa)** project.

The system replaces repetitive, human-led walkthroughs with a fully automated, interactive tour, where Pepper role-plays as a digital guide and engages visitors across multiple Digital Public Infrastructure (DPI) booths. The robot coordinates speech, gestures, dialogue, and task sequencing to guide visitors through the fictional Upanzi Republic, illustrating the end-to-end lifecycle of digital identity and service delivery—ranging from biometric enrollment (MOSIP) to financial transactions (MIFOS) and subsidy validation (UPMS).

Beyond automation, the project serves as a real-world testbed for **culturally aware human-robot interaction**. Pepper adapts its dialogue, gestures, and engagement strategies to local contexts and languages, while visitor interactions are logged to support research on cross-cultural behavior modeling and adaptive dialogue management in public spaces.

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
- **`pepper_slam`** - Mapping and odometry backends: RTAB-Map (RGB-D), SLAM Toolbox (2D lidar), and FAST-LIO / Point-LIO lidar-inertial odometry via the Unitree L2, plus an optional `robot_localization` EKF that fuses leveled LIO odometry with wheel odometry. Launch files and parameters only — the SLAM backends themselves are upstream packages
- **`pepper_navigation`** - Nav2 stack (path planning, obstacle avoidance, keepout zones, collision-monitor safety layer) with **three interchangeable localization profiles** — AMCL, RTAB-Map, or prior-map 3D ICP — behind identical costmaps and tuning

Localization-only deployments get their fused `map → base_footprint` pose (`/localization/pose`) from the sibling **`lio_localization`** package's `transform_fusion` node; `gesture_execution` consumes that pose for pointing IK.

### **Infrastructure & Utilities**
- **`dec_launch`** - System launch files, lifecycle sequencing, and startup configurations
- **`dec_interfaces`** - Custom ROS2 message, service, and action definitions
- **`dec_common`** - Shared C++ utilities: the camera lifecycle node base class, the ByteTrack multi-object tracker, and ROS2 parameter-loading helpers

## 🖥️ Hardware Diagram

<div align="center">
<img src="images/Hardware Diagram.png" alt="Hardware Diagram" width="1200px">
</div>

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
`pepper_slam`/`pepper_navigation`: `fast_lio`, `point_lio`, `lio_localization`,
and optionally `fastlio_lc_pgo`. These are only required for the lidar-based
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
# (lio_localization still comes up standalone, so /localization/pose exists)
ros2 launch dec_launch dec_system.launch.py enable_navigation:=false
```

| `nav_profile` | Nav2 bringup | Localization |
|---|---|---|
| `fastlio_loc` *(default)* | `pepper_nav2_fastlio_loc.launch.py` | FAST-LIO + prior-map 3D ICP (`lio_localization`) |
| `rtabmap_loc` | `pepper_nav2_rtabmap_loc.launch.py` | RTAB-Map localization mode against a `.db` |
| `amcl` | `pepper_nav2_amcl.launch.py` | AMCL over a flattened `/scan`, on FAST-LIO odom |
| `legacy` | `pepper_navigation.launch.py` | AMCL on raw wheel odometry; publishes no `/localization/pose` |

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

3. **Launch SLAM Toolbox (2D Mapping)**
```bash
ros2 launch pepper_slam slam_toolbox.launch.py
```

4. **Launch LIO odometry** (FAST-LIO or Point-LIO on the Unitree L2)
```bash
ros2 launch pepper_slam fastlio_odometry.launch.py
ros2 launch pepper_slam pointlio_odometry.launch.py
```

5. **Launch Nav2 with the localization profile of your choice**
```bash
ros2 launch pepper_navigation pepper_nav2_fastlio_loc.launch.py   # prior-map 3D ICP
ros2 launch pepper_navigation pepper_nav2_rtabmap_loc.launch.py   # RTAB-Map
ros2 launch pepper_navigation pepper_nav2_amcl.launch.py          # 2D AMCL
```

6. **Replay a recorded bag** (static TF for sensors, no robot required)
```bash
ros2 launch dec_launch bag_static_tf.launch.py
```

### Configuration
Each package contains configuration files in their `config/` directories:
- `behavior_controller/config/behavior_controller_configuration.yaml` - Mission parameters and active scenario
- `face_detection/config/face_detection_configuration.yaml` - Perception settings
- `pepper_slam/config/` and `pepper_navigation/config/` - SLAM, EKF, costmap and Nav2 tuning
- Gesture, attention, speech and TTS parameters in their respective package configs

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

## 🧪 Testing

Every package carries a test tier, run through `colcon`:

```bash
cd ~/ros2_ws
colcon test --packages-up-to dec_launch
colcon test-result --verbose
```

The suite covers three kinds of test:

- **Linters** — `ament_lint_auto` across all packages; `ament_flake8`/`ament_pep257`/`ament_copyright` on the Python packages, `xmllint` on the BT mission XML
- **Unit tests** — gtest on the pure-logic C++ helpers (gesture kinematics, behavior-controller utilities, ByteTracker, Boolean Map Saliency, animate-behavior motion math, age/gender temporal smoothing, YOLO class indices); pytest on the Python helpers (response parsing, audio helpers, denoiser, localization geometry)
- **Launch tests** — lifecycle bring-up and bag replay. The bag-replay test self-skips when the ONNX weights are absent, which is the case in CI.

`text_to_speech/manual_tests/` holds scripts that need real hardware and are **not** part of `colcon test`.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the full clone → `rosdep` → `colcon build` → `colcon test` chain on `ros:humble-perception-jammy` for every push and PR to `main` and `cpp`.

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
- **Mapping** (`pepper_slam`): RTAB-Map (RGB-D) or SLAM Toolbox (2D lidar); FAST-LIO / Point-LIO for lidar-inertial odometry on the Unitree L2
- **Localization** (`pepper_navigation`): three interchangeable profiles — AMCL, RTAB-Map, or prior-map 3D ICP via `lio_localization` — sharing identical costmaps and tuning so they can be compared directly
- **Odometry**: wheel odometry (`/pepper_odom` from `naoqi_driver2`), LIO odometry, or the two fused by the `robot_localization` EKF (`pepper_slam ekf_fusion.launch.py`)
- **Path Planning**: Nav2 with 3D voxel costmaps consuming the L2's 360° `PointCloud2` directly, no flattening step
- **Safety**: An independent collision monitor gates every velocity command straight off the lidar, bypassing the costmaps
- **Integration**: Full coordination with the behavior controller

## 🔧 Development

### Adding New Features
1. Create new package in the repo root directory
2. Follow ROS2 package structure conventions
3. Define interfaces in `dec_interfaces` if needed
4. Add a `test/` tier and wire it into `CMakeLists.txt`/`setup.py` so CI picks it up
5. Update `dec_launch` launch files and `dec_launch/package.xml` exec depends

### Code Style
- Follow ROS2 C++ and Python style guides
- Use descriptive variable names with underscores
- Headers use `#pragma once`; production code logs through ROS2 logging, never `print()`
- Update package README.md files

## 📚 Documentation

Detailed documentation is available:
- **Package-specific READMEs** in each package directory
- **Configuration guides** in config directories
- **Model provenance and licensing**: [MODELS.md](MODELS.md)
- **API documentation**: `ros2 interface show dec_interfaces/`
<!-- - **Deliverable reports**: [DEC4Africa Deliverables](https://dec4africa.github.io/deliverables/) -->


## ❓ Support

For issues or questions:
- **Contact**: 
  - [yohatad123@gmail.com](mailto:yohatad123@gmail.com)
## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details. Third-party pretrained model weights are **not** covered by this license — see [MODELS.md](MODELS.md).
