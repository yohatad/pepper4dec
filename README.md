<div align="center">

# Pepper Robot Tour – Digital Experience Center (DEC)

<img src="dec_system/upanzi-logo.svg" alt="Upanzi Logo" width="800px">

</div>

## 📋 Overview

This repository contains the complete software stack for an autonomous Pepper robot-led tour at the Upanzi Digital Experience Center (DEC), developed as a spin-off of the **Culturally Sensitive Social Robotics for Africa (CSSR4Africa)** project.

The system replaces repetitive, human-led walkthroughs with a fully automated, interactive tour, where Pepper role-plays as a digital guide and engages visitors across multiple Digital Public Infrastructure (DPI) booths. The robot coordinates speech, gestures, dialogue, and task sequencing to guide visitors through the fictional Upanzi Republic, illustrating the end-to-end lifecycle of digital identity and service delivery—ranging from biometric enrollment (MOSIP) to financial transactions (MIFOS) and subsidy validation (UPMS).

Beyond automation, the project serves as a real-world testbed for **culturally aware human-robot interaction**. Pepper adapts its dialogue, gestures, and engagement strategies to local contexts and languages, while visitor interactions are logged to support research on cross-cultural behavior modeling and adaptive dialogue management in public spaces.

## 🏗️ System Architecture

The system is built on **ROS2 (Humble)** and follows a modular architecture with specialized packages handling different aspects of robot behavior and perception:

### **Core Control Packages**
- **`behaviorController`** - Central mission interpreter that orchestrates tour execution, translating mission specifications into executable robot commands
- **`animate_behavior`** - Manages animated behavior sequences for expressive robot performance
- **`conversation_manager`** - Manages dialogue flow and visitor interaction sequences
- **`gesture_execution`** - Controls Pepper's arm and body movements for expressive gesturing
- **`speech_event`** - Handles speech recognition and processing
- **`text_to_speech`** - Converts text to speech with language adaptation

### **Perception & Attention Packages**
- **`face_detection`** - Real-time face detection, head pose estimation, and mutual gaze detection using SixDrepNet algorithm
- **`person_detection`** - YOLO-based person detection for scene understanding
- **`overt_attention`** - Controls robot's attention mechanism based on visitor presence

### **Navigation & Localization**
- **`pepper_navmap`** - For navigation, RTAB-Map for SLAM and Nav2 for Navigation.

### **Infrastructure & Utilities**
- **`dec_bringup`** - System launch files and startup configurations
- **`dec_interfaces`** - Custom ROS2 message and service definitions
- **`dec_launch`** - Custom launch configurations

## 🖥️ Hardware Diagram

<div align="center">
<img src="dec_system/Hardware Diagram_white_lower.png" alt="Hardware Diagram" width="900px">
</div>

## 🚀 Quick Start

### Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10+**
- **Pepper Robot** (or simulation environment)
- **Intel RealSense Camera** (for perception modules)

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

2. **Set Up Python Environment**
```bash
# Create virtual environment (recommended)
python3.10 -m venv ~/dec_virtual_envs
source ~/dec_virtual_envs/bin/activate

# Install Python dependencies
pip install -r ros2_ws/src/pepper4dec/dec_system/face_detection/requirements.txt
```

3. **Download Model Files**
   - Place required ONNX model files in their respective `models/` directories
   - Ensure face detection models are in `dec_system/face_detection/models/`

## 🎯 Running the Tour System

### Basic Launch
```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch the complete system
ros2 launch dec_bringup complete_system.launch.py
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

3. **Launch Navigation**
```bash
ros2 launch pepper_navigation navigation.launch.py
```

### Configuration
Each package contains configuration files in their `config/` directories:
- `behavior_controller/config/behaviorControllerConfiguration.ini` - Mission parameters
- `face_detection/config/face_detection_configuration.yaml` - Perception settings
- Navigation and gesture parameters in respective package configs

## 📊 Package Details

### **Face Detection System**
- **Algorithm**: SixDrepNet for head pose estimation
- **Features**: Multi-face detection, mutual gaze evaluation, age/gender estimation
- **Input**: RGB-D streams from RealSense or Pepper cameras
- **Output**: `/faceDetection/data` topic with face centroids, gaze status, demographics
- **Performance**: Real-time processing with GPU acceleration support

### **Behavior Controller**
- **Function**: Mission interpreter and system orchestrator
- **Input**: XML mission specifications
- **Coordination**: Manages 7+ ROS services and monitors multiple topics
- **Adaptation**: Culturally-aware behavior selection based on context

### **Navigation System**
- **Localization**: Adaptive Monte Carlo Localization (AMCL) or RTAB-MAP
- **Mapping**: Uses pre-built maps (`dec_system/map.pgm`)
- **Path Planning**: Dynamic Window Approach (DWA) for safe navigation
- **Integration**: Full coordination with behavior controller

## 🔧 Development

### Adding New Features
1. Create new package in `dec_system/` directory
2. Follow ROS2 package structure conventions
3. Define interfaces in `dec_interfaces` if needed
4. Update `dec_bringup` launch files

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


## 🆘 Support

For issues or questions:
- **Contact**: 
  - [yohatad123@gmail.com](mailto:yohatad123@gmail.com)
  - [muhammed]

# 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
