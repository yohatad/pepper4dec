<div align="center">
<h1>DEC Interfaces</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **DEC Interfaces** package defines the custom ROS2 messages, services and actions the pepper4dec nodes use to talk to each other. It contains only interface definitions; the generated C++ and Python code is what the other packages build against.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Actions** for every long-running robot behavior: speech, conversation, speech recognition, gestures and idle animation
- **Messages** for the per-frame face and person detections
- **Serialization test**: every type is round-tripped through the CDR serializer, so a broken definition fails in CI instead of on the robot

## ✅ Prerequisites
- **ROS2 Humble** or newer, with `rosidl_default_generators`

## 🛠️ Installation

### Package Installation

```bash
cd ~/ros2_ws
colcon build --packages-select dec_interfaces
source install/setup.bash
```

Rebuild it, then every package that uses it, after changing any `.msg`, `.srv` or `.action` file.

## 🚀 Running

Nothing to run. To inspect a definition:

```bash
ros2 interface list | grep dec_interfaces
ros2 interface show dec_interfaces/action/TTS
```

## 🖥️ ROS Interface

### Actions

| Action | Server | Goal | Result |
|---|---|---|---|
| `AnimateBehavior` | `/animate_behavior` (`animate_behavior`) | `behavior_type`, `selected_range`, `duration_seconds` | `success`, `message`, `total_duration` |
| `ConversationManager` | `/conversation_manager` (`conversation_manager`) | `prompt` | `success`, `response`, `intent`, `confidence` |
| `Gesture` | `/gesture_execution` (`gesture_execution`) | `gesture_type`, `gesture_name`, `gesture_duration`, `bow_nod_angle`, `location_x/y/z` | `success`, `message`, `actual_duration_seconds` |
| `SpeechRecognition` | `/speech_recognition` (`speech_event`) | `wait` | `transcription` |
| `TTS` | `/text_to_speech` (`text_to_speech`) | `text` | `success`, `message` |

Each package README describes its action's feedback and field meanings in full.

### Messages

| Message | Published on | Contents |
|---|---|---|
| `FaceDetection` | `/face_detection/data` (`face_detection`) | parallel arrays per face: track ID, centroid (pixels, depth in m), box size, mutual gaze |
| `PersonDetection` | `/person_detection/data` (`person_detection`) | parallel arrays per person: track ID, class, confidence, centroid (pixels, depth in m), box size |

### Services

| Service | Server | Purpose |
|---|---|---|
| `GetDepthROI` | `get_depth_roi` (`dec_launch`'s `depth_roi_service.py`) | depth statistics for one ROI, a list of points, or a list of ROIs |

## 📁 Package Structure

```
dec_interfaces/
├── action/
│   ├── AnimateBehavior.action
│   ├── ConversationManager.action
│   ├── Gesture.action
│   ├── SpeechRecognition.action
│   └── TTS.action
├── msg/
│   ├── FaceDetection.msg
│   └── PersonDetection.msg
├── srv/
│   └── GetDepthROI.srv
├── test/
│   └── test_serialization.cpp    # CDR round-trip of every type
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🧪 Testing

```bash
cd ~/ros2_ws
colcon test --packages-select dec_interfaces
colcon test-result --verbose
```

`test_serialization` fills every type with non-default values (negative numbers, extremes, non-ASCII text, empty and long arrays), serializes it, deserializes it into a fresh object and compares every field.

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2025 Carnegie Mellon University Africa
Licensed under the BSD-3-Clause License. See individual package licenses for details.
