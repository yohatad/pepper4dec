<div align="center">

# Pepper Robot Tour – Digital Experience Center (DEC)

<img src="dec_system/upanzi-logo.svg" alt="Upanzi Logo" width="800px">

</div>

## 📋 Overview

This repository contains the complete software stack for an autonomous Pepper robot-led tour at the Upanzi Digital Experience Center (DEC), developed as a spin-off of the **Culturally Sensitive Social Robotics for Africa (CSSR4Africa)** project.

The system replaces repetitive, human-led walkthroughs with a fully automated, interactive tour, where Pepper role-plays as a digital guide and engages visitors across multiple Digital Public Infrastructure (DPI) booths. The robot coordinates speech, gestures, dialogue, and task sequencing to guide visitors through the fictional Upanzi Republic, illustrating the end-to-end lifecycle of digital identity and service delivery—ranging from biometric enrollment (MOSIP) to financial transactions (MIFOS) and subsidy validation (UPMS).

Beyond automation, the project serves as a real-world testbed for **culturally aware human-robot interaction**. Pepper adapts its dialogue, gestures, and engagement strategies to local contexts and languages, while visitor interactions are logged to support research on cross-cultural behavior modeling and adaptive dialogue management in public spaces.

## 🏗️ System Architecture

<div align="center">
<img src="dec_system/System_arch.png" alt="System Architecture" width="1200px">
</div>

The system is built on **ROS2 (Humble)** and follows a modular architecture with specialized packages handling different aspects of robot behavior and perception:

### **Core Control Packages**
- **`behavior_controller`** - Central mission interpreter that orchestrates tour execution, translating mission specifications into executable robot commands
- **`animate_behavior`** - Manages animated behavior sequences for expressive robot performance
- **`conversation_manager`** - Manages dialogue flow and visitor interaction sequences
- **`gesture_execution`** - Controls Pepper's arm and body movements for expressive gesturing
- **`speech_event`** - Handles speech recognition with post-VAD noise reduction and optional sound-source localization
- **`text_to_speech`** - Converts text to speech with language adaptation

### **Perception & Attention Packages**
- **`face_detection`** - Real-time face detection, head pose estimation, and mutual gaze detection using SixDrepNet algorithm
- **`person_detection`** - YOLO-based person detection for scene understanding
- **`overt_attention`** - Controls robot's attention mechanism based on visitor presence

### **Navigation & Localization**
- **`pepper_navigation`** - RTAB-Map/SLAM Toolbox for mapping and Nav2 for navigation
- **`pepper_odom_anchor`** - Anchors Pepper's relative wheel odometry to an absolute starting pose

### **Infrastructure & Utilities**
- **`dec_launch`** - System launch files and startup configurations
- **`dec_interfaces`** - Custom ROS2 message and service definitions
- **`dec_common`** - Shared header-only C++ utilities for `dec_system` nodes

## 🖥️ Hardware Diagram

<div align="center">
<img src="dec_system/Hardware Diagram.png" alt="Hardware Diagram" width="1200px">
</div>

## 🚀 Quick Start

### Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10+**
- **Pepper Robot** (or simulation environment)
- **Intel RealSense Camera** (for perception modules)
- **Unitree L2 Lidar** (for localization and Navigation)

### Installation

1. **Clone and Build the Workspace**
```bash
# Clone the repository
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build all packages
cd ~/ros2_ws
colcon build
source install/setup.bash
```

2. **Set Up Python Environments**

Most perception/actuation packages (`animate_behavior`, `behavior_controller`, `face_detection`, `gesture_execution`, `overt_attention`, `person_detection`, `pepper_odom_anchor`) are C++ and need no Python environment. The remaining Python packages (`conversation_manager`, `speech_event`, `text_to_speech`) each expect their own dedicated virtual environment — see each package's own README for the exact venv name and `pip install -r requirements.txt` it expects.

3. **Download Model Files**
   - Place required ONNX model files in their respective `models/` directories
   - Ensure face detection models are in `dec_system/face_detection/models/`

## 🚀 Running the Tour System

### Basic Launch
```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch the complete system (requires all dependencies and robot hardware)
ros2 launch dec_launch dec_system.launch.py
```

### Component-Based Launch
For development and testing, individual components can be launched:

1. **Launch Perception System**
```bash
ros2 launch face_detection face_detection_launch_robot.launch.py
```

2. **Launch Behavior Controller**
```bash
ros2 run behavior_controller behavior_controller
```

3. **Launch SLAM Toolbox (2D Mapping)**
```bash
ros2 launch pepper_navigation slam_toolbox.launch.py
```

### Configuration
Each package contains configuration files in their `config/` directories:
- `behavior_controller/config/behavior_controller_configuration.yaml` - Mission parameters
- `face_detection/config/face_detection_configuration.yaml` - Perception settings
- Navigation and gesture parameters in respective package configs

## 📊 Package Details

### **Face Detection System**
- **Algorithm**: SixDrepNet for head pose estimation
- **Features**: Multi-face detection, mutual gaze evaluation, age/gender estimation
- **Input**: RGB-D streams from RealSense or Pepper cameras
- **Output**: `/face_detection/data` topic with face centroids, gaze status, demographics
- **Performance**: Real-time processing with GPU acceleration support

### **Behavior Controller**
- **Function**: Mission interpreter and system orchestrator
- **Input**: XML mission specifications
- **Coordination**: Manages 7+ ROS services and monitors multiple topics
- **Adaptation**: Culturally-aware behavior selection based on context

### **Navigation System**
- **Localization**: Adaptive Monte Carlo Localization (AMCL), RTAB-MAP, or the `robot_localization` EKF node
- **Mapping**: Uses pre-built maps (`dec_system/map.pgm`)
- **Path Planning**: Nav2 stack for safe navigation
- **Integration**: Full coordination with behavior controller

## 🔧 Development

### Adding New Features
1. Create new package in `dec_system/` directory
2. Follow ROS2 package structure conventions
3. Define interfaces in `dec_interfaces` if needed
4. Update `dec_launch` launch files

### Code Style
- Follow ROS2 C++ and Python style guides
- Use descriptive variable names with underscores
- Update package README.md files

## 📚 Documentation

Detailed documentation is available:
- **Package-specific READMEs** in each package directory
- **Configuration guides** in config directories
- **API documentation**: `ros2 interface show dec_interfaces/`
<!-- - **Deliverable reports**: [DEC4Africa Deliverables](https://dec4africa.github.io/deliverables/) -->


## ❓ Support

For issues or questions:
- **Contact**: 
  - [yohatad123@gmail.com](mailto:yohatad123@gmail.com)
## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
