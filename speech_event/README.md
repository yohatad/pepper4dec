<div align="center">
<h1>Speech Event Recognition and Localization</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Speech Event Recognition and Localization** package provides real-time speech recognition using Whisper ASR, voice activity detection (VAD) with Silero VAD (ONNX), and optional sound-source localization. It processes the robot's microphone audio (48 kHz, `naoqi_bridge_msgs/AudioBuffer`), detects speech segments, applies a noise reduction pipeline post-VAD and pre-ASR, transcribes them with a low-latency Whisper model, and publishes the recognized text. The module can also estimate the direction-of-arrival of a sound using SRP-PHAT beamforming on Pepper's 4-microphone array.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Whisper ASR**: State-of-the-art speech recognition with low latency
- **Silero VAD**: Voice activity detection with two-threshold hysteresis
- **Noise Reduction**: Post-VAD, pre-ASR pipeline combining bandpass filtering, harmonic notch filters for fan hum, and a Wiener filter with online noise estimation
- **Sound Localization**: SRP-PHAT beamforming for azimuth estimation
- **Action Server Interface**: Synchronous transcription requests with feedback
- **Multi-microphone Array**: 4-channel audio processing for localization

## ✅ Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **CUDA-capable GPU** (optional but recommended for Whisper acceleration)
- **Intel RealSense camera** (for localization)

## 🛠️ Installation

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
pip install -r ~/ros2_ws/src/pepper4dec/speech_event/requirements.txt
```

## 🔧 Configuration

Configuration is managed via `config/speech_event_configuration.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `microphone_topic` | ROS topic publishing `naoqi_bridge_msgs/AudioBuffer` | `/naoqi_driver/audio` |
| `sample_rate` | Target sample rate for VAD/ASR (Hz) | `16000` |
| `input_sample_rate` | Robot's native microphone sample rate (Hz) | `48000` |
| `device` | PyTorch device for Whisper inference | `cuda` |
| `compute_type` | Whisper computation precision (`float16`/`float32`) | `float16` |
| `language` | Language code for ASR (ISO 639-1) | `en` |
| `whisper_model_id` | HuggingFace model ID or local path for the Whisper model | `deepdml/faster-whisper-large-v3-turbo-ct2` |
| `speech_threshold` | VAD probability threshold for speech start | `0.7` |
| `neg_threshold` | VAD probability threshold for silence | `0.35` |
| `min_silence_duration_ms` | Minimum continuous silence to end a speech segment (ms) | `800` |
| `min_speech_duration` | Minimum segment duration to submit for transcription (s) | `0.3` |
| `max_speech_duration_s` | Maximum speech segment duration (s) | `10.0` |
| `pre_speech_buffer_ms` | Lookback audio prepended before speech onset (ms) | `200` |
| `intensity_threshold` | RMS amplitude gate; quiet audio is skipped before VAD | `0.001` |
| `transcription_timeout_s` | Timeout waiting for a transcription result (s) | `5.0` |
| `action_server` | Enable action server mode (`false` = continuous, publishes to `/speech_event/text`) | `true` |
| `vad_always_active` | Keep VAD running (publishing `/speech_event/vad_speech_prob`) even with no active goal, enabling barge-in monitoring during TTS playback | `true` |
| `noise_cleaning_enabled` | Enable post-VAD noise reduction before ASR | `false` |
| `noise_profile_path` | Path to a `.npy` mean-magnitude-spectrum file recorded at 16 kHz; leave empty for online-only estimation | `"data/noise_profile.npy"` |
| `noise_alpha` | Wiener filter aggressiveness (0.0–1.0); higher = more suppression, more distortion risk | `0.5` |

## 🚀 Running the Node

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

## 🖥️ ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/naoqi_driver/audio` | `naoqi_bridge_msgs/AudioBuffer` | Raw multi-channel audio from robot microphone (48 kHz, 4 channels) |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/speech_event/vad_speech_prob` | `std_msgs/Float32` | Real-time VAD probability (0–1) |
| `/speech_event/text` | `std_msgs/String` | Recognized speech text (standalone mode only) |

### Action Servers

| Action | Type | Description |
|--------|------|-------------|
| `/speech_recognition` | `dec_interfaces/action/SpeechRecognition` | Synchronous transcription requests |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/speech_event/set_enabled` | `std_srvs/SetBool` | Enable or disable audio processing (e.g. mute mic during TTS) |

## 🔌 Action Interface

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

## 📁 Package Structure

```
speech_event/
├── config/
│   └── speech_event_configuration.yaml       # ROS2 parameters
├── data/
│   ├── noise_profile.npy                     # optional precomputed noise spectrum
│   └── pepper_topics.yaml                    # topic name overrides
├── models/
│   └── silero_vad.onnx                       # Silero VAD weights
├── scripts/
│   ├── speech_event                          # venv launcher for `ros2 run`
│   ├── speech_event_localization
│   └── speech_event_recorder
├── resource/
│   └── speech_event
├── speech_event/
│   ├── __init__.py
│   ├── speech_event_application.py           # node entry point, lifecycle node
│   ├── speech_event_denoiser.py              # post-VAD, pre-ASR noise reduction
│   ├── speech_event_implementation.py        # VAD + Whisper transcription
│   ├── speech_event_localization.py          # SRP-PHAT sound-source localization node
│   └── speech_event_recorder.py              # standalone audio recorder (debugging)
├── package.xml
├── setup.py
├── setup.cfg
├── requirements.txt
└── README.md
```

## 🏗️ Architecture

```mermaid
flowchart TD
    A["/naoqi_driver/audio"] --> B["Resample to 16 kHz mono"]
    B --> C{"Silero VAD"}
    C -- "per-frame probability" --> D(["/speech_event/vad_speech_prob"])
    C -- "speech segment detected" --> E["Speech Segment Buffer"]
    E --> F["SpeechDenoiser"]
    F --> G["Whisper ASR"]
    G --> H{"action_server mode?"}
    H -- "true (default): gated,\nonly while a goal is active" --> I(["/speech_recognition\naction result"])
    H -- "false: standalone,\nalways listening" --> J(["/speech_event/text"])

    A --> K["speech_event_localization node\n(SRP-PHAT beamforming)"]
    K --> L(["/sound_localization/direction"])
    K --> M(["/sound_localization/azimuth"])
    K --> N(["/sound_localization/confidence"])
```

`SpeechDenoiser`'s post-VAD, pre-ASR cleaning pipeline:
- Bandpass filter (80 Hz – 7500 Hz)
- Harmonic IIR notch filters targeting fan hum fundamental and harmonics
- Wiener filter with online minimum-statistics noise estimation
- Spectral floor and median smoothing to suppress musical noise
- RMS normalisation to preserve Whisper's internal level thresholds

### Node Lifecycle

`SpeechRecognitionNode` is a `LifecycleNode`; `dec_launch`'s `nav2_lifecycle_manager` drives it through these transitions on startup:

```mermaid
stateDiagram-v2
    [*] --> Unconfigured

    Unconfigured --> Inactive: configure
    Inactive --> Active: activate
    Active --> Inactive: deactivate
    Inactive --> Unconfigured: cleanup

    Unconfigured --> Finalized: shutdown
    Inactive --> Finalized: shutdown
    Active --> Finalized: shutdown
    Finalized --> [*]
```

| Transition | What happens |
|---|---|
| `configure` | Load Silero VAD + Whisper models, construct `SpeechDenoiser`, create publishers, `set_enabled` service, and action server |
| `activate` | Activate publishers, subscribe to `/naoqi_driver/audio` |
| `deactivate` | Destroy the audio subscription, deactivate publishers |
| `cleanup` | Destroy publishers/service/action server, release Whisper/Silero/denoiser instances |
| `shutdown` | Log shutdown and exit (reachable from any state) |

## 🧪 Testing

```bash
# Check node is running
ros2 node list

# Verify action server is available
ros2 action list

# Send a test transcription request
ros2 action send_goal /speech_recognition dec_interfaces/action/SpeechRecognition \
  "{wait: 5.0}"

# Monitor VAD probabilities
ros2 topic echo /speech_event/vad_speech_prob

# Monitor transcribed text (standalone mode)
ros2 topic echo /speech_event/text
```

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.