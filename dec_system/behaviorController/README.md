<div align="center">
  <h1>Behavior Controller</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Behavior Controller** package orchestrates robot behavior on the Pepper robot using [BehaviorTree.CPP v4](https://www.behaviortree.dev/) and [BehaviorTree.ROS2](https://github.com/BehaviorTree/BehaviorTree.ROS2). It loads a mission defined as an XML behavior tree and executes it by dispatching ROS2 action calls to speech, gesture, navigation, conversation, and face-detection nodes. At startup it loads a YAML configuration file and two knowledge bases — a culture knowledge base (utility phrases) and an environment knowledge base (locations and tour stops).

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **BehaviorTree.CPP v4**: Mission logic expressed as composable XML behavior trees
- **Intent-aware routing**: The `ConversationManager` BT node now exposes `intent` and `confidence` output ports; the `asr_cm_tts_pipeline.xml` tree branches on intent to drive speech, navigation, gesture, or stop behavior accordingly
- **Streaming TTS**: Sentences stream to `/tts/input` while the LLM generates; the `TTS` BT node blocks until playback is complete
- **Multi-language Support**: Speech output in English and Kinyarwanda
- **Configurable**: Configuration via YAML file; active scenario selectable at runtime
- **Knowledge Base Driven**: Loads location data and cultural phrases from YAML knowledge bases
- **Groot2 Compatible**: Publishes BT state for live visualization in Groot2

# 🛠️ Installation

## Prerequisites
- **ROS2 Humble** or newer
- **BehaviorTree.CPP v4** (`behaviortree_cpp`)
- **BehaviorTree.ROS2** (`behaviortree_ros2`)
- **Nav2** (`nav2_msgs`)
- **naoqi_bridge_msgs**
- **dec_interfaces** (custom messages, services, and actions)
- **yaml-cpp**

## Package Installation

```bash
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

cd ~/ros2_ws
colcon build --packages-select dec_interfaces behavior_controller
source install/setup.bash
```

# 🔧 Configuration Parameters

Managed via `config/behaviorControllerConfiguration.yaml`:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `scenario_specification` | XML behavior tree file name (without `.xml`) in `data/` | `lab_tour` |
| `culture_knowledge_base` | YAML file with utility phrases per language | `cultureKnowledgeBase.yaml` |
| `environment_knowledge_base` | YAML file with locations, tour stops, and gesture targets | `labEnvironmentKnowledgeBase.yaml` |
| `language` | Speech output language (`English` or `Kinyarwanda`) | `English` |
| `verbose_mode` | Print per-tick action goals, feedback, and results | `false` |

# 🚀 Running the Node

1. **Source the workspace**:
```bash
source ~/ros2_ws/install/setup.bash
```

2. **Launch the robot** (if using physical Pepper):
```bash
ros2 launch dec_system decSystemLaunchRobot.launch.py \
  robot_ip:=<robot_ip> \
  roscore_ip:=<roscore_ip> \
  network_interface:=<network_interface>
```

3. **Run the behavior controller**:
```bash
ros2 run behavior_controller behaviorController
```

## Required Action Servers and Topics

The following must be available before the behavior controller starts executing:

| Interface | Type | Package |
|-----------|------|---------|
| `/tts` | `dec_interfaces::action::TTS` | `text_to_speech` |
| `/conversation_manager` | `dec_interfaces::action::ConversationManager` | `conversation_manager` |
| `/speech_recognition_action` | `dec_interfaces::action::SpeechRecognition` | `speech_event` |
| `/naoqi_driver/speech_with_feedback` | `naoqi_bridge_msgs::action::SpeechWithFeedback` | `naoqi_bridge` |
| `animate_behavior` | `dec_interfaces::action::AnimateBehavior` | `animate_behavior` |
| `/gesture` | `dec_interfaces::action::Gesture` | `gesture` |
| `/navigate_to_pose` | `nav2_msgs::action::NavigateToPose` | `nav2` |
| `/faceDetection/data` | `dec_interfaces::msg::FaceDetection` (topic) | `face_detection` |

# 🖥️ BT Nodes Reference

All nodes registered in the factory:

| BT Node | ROS2 Interface | Key Ports |
|---------|---------------|-----------|
| `TTS` | `dec_interfaces::action::TTS` (`/tts`) | `text` in; `status`, `message` out |
| `ConversationManager` | `dec_interfaces::action::ConversationManager` (`/conversation_manager`) | `prompt` in; `response`, `intent`, `confidence`, `status` out |
| `SpeechRecognition` | `dec_interfaces::action::SpeechRecognition` | `wait` in; `transcription`, `status` out |
| `SpeechWithFeedback` | `naoqi_bridge_msgs::action::SpeechWithFeedback` | `say` in; `started`, `bookmark`, `current_word` out |
| `AnimateBehavior` | `dec_interfaces::action::AnimateBehavior` | `behavior_type`, `selected_range`, `duration_seconds` in |
| `StopAnimateBehavior` | `std_srvs::srv::Trigger` (`animate_behavior/stop`) | — |
| `Gesture` | `dec_interfaces::action::Gesture` | `gesture_type`, `location_x/y/z` in |
| `Navigate` | `nav2_msgs::action::NavigateToPose` (`/navigate_to_pose`) | `goal_x`, `goal_y`, `goal_theta` in |
| `SetOvertAttention` | `std_srvs::srv::SetBool` (`/attn/set_enabled`) | `enabled` in |
| `SetSpeechListening` | `std_srvs::srv::SetBool` (`/speech_event/set_enabled`) | `enabled` in |
| `CheckFaceDetected` | `/faceDetection/data` topic | `face_count`, `mutual_gaze` out |
| `IsVisitorDiscovered` | `/faceDetection/data` topic | `timeout` in |
| `IsMutualGazeDiscovered` | `/faceDetection/data` topic | `timeout` in |
| `ListenForSpeech` | `/speech_event/text` topic | `transcription` out |
| `GetVisitorResponse` | `/speech_event/text` topic | `timeout` in; `visitor_response` out |
| `IsVisitorResponseYes` | `dec_interfaces::action::ConversationManager` | `visitor_response` in |
| `CheckBlackboard` | — (pure blackboard) | `key`, `expected` in |
| `SetBlackboardValue` | — (pure blackboard) | `key`, `value` in |
| `LogEvent` | — | `message`, `level` in |

# 🏗️ Architecture

## Components

1. **ConfigManager** — Singleton that loads and exposes all YAML configuration at startup.
2. **KnowledgeManager** — Singleton that loads the culture knowledge base (utility phrases per language) and the environment knowledge base (location poses, gesture targets, tour sequence).
3. **BehaviorTree execution engine** — Registers all BT leaf nodes, loads the XML scenario, and ticks the tree at the executor rate.

## Intent-Driven Pipeline (`asr_cm_tts_pipeline.xml`)

The conversation pipeline uses `ConversationManager`'s `intent` output to route behavior without a second LLM call:

```
SpeechRecognition
      │
ConversationManager  ──► {intent}, {confidence}, {llm_response}
      │
      └─► CheckBlackboard(intent == "STOP")
               └─► StopAnimateBehavior
          CheckBlackboard(intent == "NAVIGATION_REQUEST")
               └─► Parallel: TTS + Navigate
          CheckBlackboard(intent == "ASK_EXHIBIT_QUESTION")
               └─► Parallel: TTS + Gesture
          <default>
               └─► TTS only
```

**Intent values** produced by the LLM system prompt:

| Intent | Triggered behavior |
|---|---|
| `ASK_EXHIBIT_QUESTION` | Speak answer + point gesture at exhibit |
| `ASK_TOUR_META` | Speak answer only |
| `NAVIGATION_REQUEST` | Speak answer + navigate in parallel |
| `SOCIAL_SMALL_TALK` | Speak answer only |
| `OFF_TOPIC` | Speak polite apology only |
| `STOP` | Stop animation immediately, no speech |
| `AFFIRMATIVE` | Speak "yes" (parent subtree handles) |
| `NEGATIVE` | Speak "no" (parent subtree handles) |

## Behavior Tree Scenarios

XML files are in the `data/` folder. The active scenario is set via `scenario_specification`.

| File | Description |
|------|-------------|
| `lab_tour.xml` | Guided tour of the lab environment |
| `dec_Tour.xml` | Guided tour of the DEC environment |
| `asr_cm_tts_pipeline.xml` | Intent-routed ASR → ConversationManager → TTS pipeline |

## Package Structure

```
behaviorController/
├── config/
│   └── behaviorControllerConfiguration.yaml
├── data/
│   ├── lab_tour.xml
│   ├── dec_Tour.xml
│   ├── asr_cm_tts_pipeline.xml
│   ├── cultureKnowledgeBase.yaml
│   └── labEnvironmentKnowledgeBase.yaml
├── behaviorController/
│   ├── __init__.py
│   ├── behaviorController_application.py
│   ├── behaviorController_implementation.py
│   ├── configManager.py
│   └── knowledgeManager.py
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## Testing

```bash
# Check node is running
ros2 node list

# Verify BT is ticking
ros2 topic list

# Monitor BT state (Groot2)
ros2 run groot2_gui groot2_gui
```

## Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

# 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
