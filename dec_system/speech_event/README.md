<div align="center">
<h1>Speech Event Recognition and Localization</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Speech Event Recognition and Localization** package provides real-time speech recognition using Whisper ASR, voice activity detection (VAD) with Silero VAD (ONNX), and optional sound-source localization. It processes the robot's microphone audio (48 kHz, `naoqi_bridge_msgs/AudioBuffer`), detects speech segments, transcribes them with a low-latency Whisper model, and publishes the recognized text. The module can also estimate the direction-of-arrival of a sound using SRP-PHAT beamforming on Pepper's 4-microphone array.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Whisper ASR**: State-of-the-art speech recognition with low latency
- **Silero VAD**: Voice activity detection with two-threshold hysteresis
- **Sound Localization**: SRP-PHAT beamforming for azimuth estimation
- **Streaming Support**: Audio processing with configurable modes
- **Action Server Interface**: Synchronous transcription requests with feedback
- **Multi-microphone Array**: 4-channel audio processing for localization

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **CUDA-capable GPU** (optional but recommended for Whisper acceleration)
- **Intel RealSense camera** (for localization)

## Installation

### Package Installation

```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select speech_event
source install/setup.bash
```

### Python Dependencies

```bash
# Install PyTorch with CUDA support (recommended)
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages
pip install -r ~/ros2_ws/src/pepper4dec/dec_system/speech_event/requirements.txt
```

## Configuration

Configuration is managed via `config/speech_event_configuration.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sample_rate` | Target sample rate for VAD/ASR (Hz) | `16000` |
| `input_sample_rate` | Robot's native microphone sample rate (Hz) | `48000` |
| `device` | PyTorch device for Whisper inference | `cuda` |
| `language` | Language code for ASR (ISO 639-1) | `en` |
| `speech_threshold` | VAD probability threshold for speech start | `0.7` |
| `neg_threshold` | VAD probability threshold for silence | `0.35` |
| `min_silence_duration_ms` | Minimum silence duration to end speech (ms) | `300` |
| `max_speech_duration_s` | Maximum speech segment duration (s) | `10.0` |
| `action_server` | Enable action server mode | `true` |

## Running the Node

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Run the speech event node (action server mode by default)
ros2 run speech_event speech_event

# Run in standalone (topic-only) mode
ros2 run speech_event speech_event --ros-args -p action_server:=false
```

### Optional Nodes

```bash
# Audio recorder (for debugging/data collection)
ros2 run speech_event speech_event_recorder

# Sound-source localization (requires 4-mic array)
ros2 run speech_event speech_event_localization
```

## ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/audio` | `naoqi_bridge_msgs/AudioBuffer` | Raw multi-channel audio from robot microphone (48 kHz, 4 channels) |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/speech_event/vad_speech_prob` | `std_msgs/Float32` | Real-time VAD probability (0–1) |
| `/speech_event/text` | `std_msgs/String` | Recognized speech text (standalone mode only) |

### Action Servers

| Action | Type | Description |
|--------|------|-------------|
| `/speech_recognition_action` | `dec_interfaces/action/SpeechRecognition` | Synchronous transcription requests |

## Action Interface

**Action Type:** `dec_interfaces/action/SpeechRecognition`

### Goal

| Field | Type | Description |
|-------|------|-------------|
| `wait` | float64 | Maximum seconds to wait for speech onset |

### Result

| Field | Type | Description |
|-------|------|-------------|
| `transcription` | string | The transcribed speech text |

### Feedback

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "waiting", "speech", "transcribing" |

## Operating Modes

### Action Server Mode (default)
Audio processing is gated — audio is only processed while an active goal is being handled.

### Standalone Mode (`action_server: false`)
The node continuously listens and processes audio independent of any action client.

## Sound Source Localization

The `speech_event_localization` node provides real-time sound source localization using SRP-PHAT beamforming.

### Microphone Array Geometry (Pepper)

| Microphone | X (m) | Y (m) | Z (m) |
|------------|-------|-------|-------|
| Rear Left | -0.0267 | +0.0343 | 0.2066 |
| Rear Right | -0.0267 | -0.0343 | 0.2066 |
| Front Left | +0.0313 | +0.0343 | 0.2066 |
| Front Right | +0.0313 | -0.0343 | 0.2066 |

### Localization Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/sound_localization/direction` | `geometry_msgs/Vector3Stamped` | 2D unit direction vector |
| `/sound_localization/azimuth` | `std_msgs/Float32` | Azimuth angle in degrees |
| `/sound_localization/confidence` | `std_msgs/Float32` | Confidence score (0–1) |

## Package Structure

```
speech_event/
├── config/
│   └── speech_event_configuration.yaml
├── data/
│   └── pepper_topics.yaml
├── models/
│   └── silero_vad.onnx
├── resource/
│   └── speech_event
├── speech_event/
│   ├── __init__.py
│   ├── speech_event_application.py
│   ├── speech_event_implementation.py
│   ├── speech_event_localization.py
│   └── speech_event_recorder.py
├── package.xml
├── setup.py
├── setup.cfg
├── requirements.txt
└── README.md
```

## Architecture

1. **Audio Input**: Receives multi-channel audio from robot microphone
2. **VAD Processing**: Silero VAD detects speech segments with hysteresis
3. **Transcription**: Whisper ASR transcribes detected speech segments
4. **Feedback**: Streams status updates during processing
5. **Result**: Returns transcription text as action result

## Testing

```bash
# Check node is running
ros2 node list

# Verify action server is available
ros2 action list

# Send a test transcription request
ros2 action send_goal /speech_recognition_action dec_interfaces/action/SpeechRecognition \
  "{wait: 5.0}"

# Monitor VAD probabilities
ros2 topic echo /speech_event/vad_speech_prob

# Monitor transcribed text (standalone mode)
ros2 topic echo /speech_event/text
```

## Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.