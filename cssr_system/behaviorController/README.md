<div align="center">
  <h1>Behavior Controller</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Behavior Controller** package is a **ROS2** package that orchestrates robot behavior using [BehaviorTree.CPP](https://www.behaviortree.dev/) and [BehaviorTree.ROS2](https://github.com/BehaviorTree/BehaviorTree.ROS2). It loads a robot mission defined as an XML behavior tree and executes it by dispatching ROS2 action calls to speech, gesture, navigation, and face detection nodes. On startup it loads a YAML configuration file and two knowledge bases — a culture knowledge base (utility phrases) and an environment knowledge base (locations and tour stops).

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **BehaviorTree.CPP v4**: Mission logic is expressed as composable XML behavior trees
- **BehaviorTree.ROS2**: Each action wraps a ROS2 action server or topic subscription as a BT leaf node
- **Multi-language Support**: Speech output in English and Kinyarwanda
- **Configurable**: Configuration via YAML file
- **Knowledge Base Driven**: Loads location data and cultural phrases from YAML knowledge bases
- **Groot2 Compatible**: Publishes BT state for live visualization in Groot2

# 🛠️ Installation

## Prerequisites
- **ROS2 Humble** or newer
- **BehaviorTree.CPP v4** (`behaviortree_cpp`)
- **BehaviorTree.ROS2** (`behaviortree_ros2`)
- **Nav2** (`nav2_msgs`)
- **naoqi_bridge_msgs**
- **cssr_interfaces** (custom messages, services, and actions)
- **yaml-cpp**

## Package Installation

1. **Clone and Build the Workspace**
```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/cssr4africa/cssr4africa.git

# Build the package
cd ~/ros2_ws
colcon build --packages-select behavior_controller
source install/setup.bash
```

# 🔧 Configuration Parameters

The configuration is managed via `config/behaviorControllerConfiguration.yaml`:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `scenario_specification` | Name of the XML behavior tree file (without extension) in the `data/` folder | `lab_tour` |
| `culture_knowledge_base` | YAML file containing utility phrases for supported languages | `cultureKnowledgeBase.yaml` |
| `environment_knowledge_base` | YAML file describing locations and tour stops for the current environment | `labEnvironmentKnowledgeBase.yaml` |
| `language` | Language used for speech output (`English` or `Kinyarwanda`) | `English` |
| `verbose_mode` | Enable detailed diagnostic logging to the terminal | `false` |

> **Note:**
> Enabling **`verbose_mode`** (`true`) will print action goals, feedback, and results for every BT node tick to the terminal.

# 🚀 Running the Node

## Steps

1. **Source the workspace**:
```bash
cd ~/ros2_ws && source install/setup.bash
```

2. **Launch the robot**:
```bash
ros2 launch cssr_system cssrSystemLaunchRobot.launch.py \
  robot_ip:=<robot_ip> \
  roscore_ip:=<roscore_ip> \
  network_interface:=<network_interface>
```

<div style="background-color: #1e1e1e; padding: 15px; border-radius: 4px; border: 1px solid #404040; margin: 10px 0;">
<span style="color: #ff3333; font-weight: bold;">NOTE: </span>
<span style="color: #cccccc;">Ensure that <code>robot_ip</code>, <code>roscore_ip</code>, and <code>network_interface</code> are correctly set for your robot and network configuration.</span>
</div>

3. **Run the behavior controller node** (new terminal):
```bash
cd ~/ros2_ws && source install/setup.bash
ros2 run behavior_controller behaviorController
```

## Required Action Servers and Topics

<div style="background-color: #1e1e1e; padding: 15px; border-radius: 4px; border: 1px solid #404040; margin: 10px 0;">
<span style="color: #ff3333; font-weight: bold;">NOTE: </span>
<span style="color: #cccccc;">The following <strong>ROS2 action servers</strong> must be available before the behavior controller starts executing:

- `AnimateBehavior` — `cssr_interfaces::action::AnimateBehavior`
- `Gesture` — `cssr_interfaces::action::Gesture`
- `SpeechWithFeedback` — `naoqi_bridge_msgs::action::SpeechWithFeedback`
- `SpeechRecognition` — `cssr_interfaces::action::SpeechRecognition`
- `ConversationManager` — `cssr_interfaces::action::ConversationManager`
- `/navigate_to_pose` — `nav2_msgs::action::NavigateToPose`

The following <strong>topic</strong> must also be available:

- `/faceDetection/data` — `cssr_interfaces::msg::FaceDetection`
</span>
</div>

# 🖥️ Output

The node logs startup information, configuration values, and (when `verbose_mode` is `true`) per-tick BT node status to the ROS2 logger. No topics are published by this node — it acts as the mission executor that calls other nodes.

## Behavior Tree Scenarios

XML behavior tree files are placed in the `data/` folder. The active scenario is selected via `scenario_specification` in the configuration file.

| File | Description |
|------|-------------|
| `lab_tour.xml` | Guided tour of the lab environment |
| `dec_Tour.xml` | Guided tour of the DEC environment |
| `asr_cm_tts_pipeline.xml` | Speech recognition → conversation manager → TTS pipeline |

## Verification

```bash
# Confirm the node is running
ros2 node list

# Check active topics
ros2 topic list
```

# 🏗️ Architecture

The behavior controller consists of three main components:

1. **ConfigManager**: Singleton that loads and exposes all configuration parameters from the YAML file at startup.
2. **KnowledgeManager**: Singleton that validates and loads the culture knowledge base (utility phrases per language) and the environment knowledge base (location poses, gesture targets, tour sequence, pre/post messages).
3. **BehaviorTree execution engine**: Registers the following BT leaf nodes and ticks the loaded tree:

   | BT Node | Type | ROS2 Interface |
   |---------|------|----------------|
   | `AnimateBehavior` | Action | `cssr_interfaces::action::AnimateBehavior` |
   | `Gesture` | Action | `cssr_interfaces::action::Gesture` |
   | `SpeechWithFeedback` | Action | `naoqi_bridge_msgs::action::SpeechWithFeedback` |
   | `SpeechRecognition` | Action | `cssr_interfaces::action::SpeechRecognition` |
   | `ConversationManager` | Action | `cssr_interfaces::action::ConversationManager` |
   | `Navigate` | Action | `nav2_msgs::action::NavigateToPose` (`/navigate_to_pose`) |
   | `CheckFaceDetected` | Topic subscriber | `/faceDetection/data` (`cssr_interfaces::msg::FaceDetection`) |

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2023 Upanzi Network

2026-02-09
