<div align="center">
<h1>Speech Event Recognition and Localization</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Speech Event Recognition and Localization** package is a ROS2 package that provides real-time speech recognition using Whisper ASR, voice activity detection (VAD) with Silero VAD (ONNX), and optional sound-source localization. It processes the robot's microphone audio (48 kHz, `naoqi_bridge_msgs/AudioBuffer`), detects speech segments, transcribes them with a low-latency Whisper model, and publishes the recognized text. The module can also estimate the direction-of-arrival of a sound using SRP-PHAT beamforming on Pepper's 4-microphone array.

# 📄 Documentation
The main documentation for this deliverable is found in the CSSR4Africa project deliverables. For technical details about the audio processing pipeline, refer to the source code and inline comments.

# 🛠️ Installation

Install the required software components to instantiate and set up the development environment for controlling the Pepper robot. Use the [CSSR4Africa Software Installation Manual](https://cssr4africa.github.io/deliverables/CSSR4Africa_Deliverable_D3.3.pdf).

## Prerequisites
- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+ (tested with Python 3.10.12)
- CUDA-capable GPU (optional but recommended for Whisper acceleration)
  - CUDA 12.1 or compatible version
  - Check your CUDA version: `nvcc --version` or `nvidia-smi`

## ROS2 Dependencies
The following ROS2 packages are required (listed in `package.xml`):
- `rclpy`, `rclcpp`, `rclcpp_action`
- `std_msgs`, `geometry_msgs`, `sensor_msgs`
- `naoqi_bridge_msgs` — for `AudioBuffer` microphone messages
- `dec_interfaces` — for the `SpeechRecognition` action definition
- `ament_index_python`

## Python Environment Setup

```sh
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python virtual environment tools
sudo apt install python3-venv python3-pip -y

# Create a virtual environment (adjust location as needed)
cd ~
python3 -m venv sound  # or any preferred name

# Activate the virtual environment
source ~/sound/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (REQUIRED - install before requirements.txt)
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# For CPU-only systems (alternative to the above):
# pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python packages
pip install -r ~/ros2_ws/src/cssr4africa/cssr_system/speech_event/requirements.txt
```

## Building the ROS2 Package

```sh
cd ~/ros2_ws
source /opt/ros/humble/setup.bash  # or your ROS2 distribution
colcon build --packages-select speech_event
source install/setup.bash
```

# 🔧 Configuration Parameters

The node is configured via ROS2 parameters (can be set via YAML, launch files, or command line). Key parameters are:

## Speech Recognition Node (`speech_recognition`)

| Parameter                   | Description                                                                        | Type/Range                   | Default Value                                  |
|-----------------------------|------------------------------------------------------------------------------------|------------------------------|------------------------------------------------|
| `sample_rate`               | Target sample rate for VAD/ASR (Hz)                                               | int (16000 recommended)      | `16000`                                        |
| `input_sample_rate`         | Robot's native microphone sample rate (Hz)                                        | int                          | `48000`                                        |
| `device`                    | PyTorch device for Whisper inference                                              | string: `"cuda"` or `"cpu"`  | `"cuda"`                                       |
| `compute_type`              | Whisper computation precision                                                     | string: `"float16"`/`"float32"` | `"float16"`                                 |
| `language`                  | Language code for ASR (ISO 639-1)                                                 | string: `"en"`, `"fr"`, etc. | `"en"`                                         |
| `whisper_model_id`          | Hugging Face model ID or local path                                               | string                       | `"deepdml/faster-whisper-large-v3-turbo-ct2"` |
| `speech_threshold`          | VAD probability above which speech is considered started                          | float (0.0–1.0)              | `0.7`                                          |
| `neg_threshold`             | VAD probability below which silence is counted                                    | float (0.0–1.0)              | `0.35`                                         |
| `min_silence_duration_ms`   | Minimum continuous silence duration to end a speech segment (ms)                 | int (milliseconds)           | `300` (YAML default: `1000`)                  |
| `max_speech_duration_s`     | Maximum allowed speech segment duration (seconds)                                 | float (seconds)              | `10.0`                                         |
| `min_speech_duration`       | Minimum speech duration required to submit for transcription (seconds)           | float (seconds)              | `0.3`                                          |
| `pre_speech_buffer_ms`      | Audio to prepend before speech onset (lookback ring buffer, ms)                  | int (milliseconds)           | `200`                                          |
| `intensity_threshold`       | RMS intensity gate — audio quieter than this is skipped before VAD               | float (RMS amplitude)        | `0.001`                                        |
| `microphone_topic`          | ROS topic for raw audio input (`naoqi_bridge_msgs/AudioBuffer`)                  | string (topic name)          | `"/audio"`                                     |
| `action_server`             | Enable the ROS2 action server interface. When `true`, audio is only processed while a goal is active. When `false`, audio is processed continuously (standalone mode). | bool | `true` |

> **VAD Two-Threshold Hysteresis:**
> Speech starts when probability ≥ `speech_threshold`. Silence is accumulated when probability < `neg_threshold`. The speech segment ends after `min_silence_duration_ms` of continuous silence or when `max_speech_duration_s` is reached.
>
> **Note on `min_silence_duration_ms`:** The code default is 300 ms, but the supplied `config/speech_event_configuration.yaml` sets this to 1000 ms. When loading parameters from the YAML file, 1000 ms takes effect.

## Setting Parameters

You can set parameters via:

1. **Launch file**:
```python
Node(
    package='speech_event',
    executable='speech_event',
    parameters=[{
        'device': 'cuda',
        'language': 'en',
        'speech_threshold': 0.7,
        'action_server': True
    }]
)
```

2. **Command line**:
```bash
ros2 run speech_event speech_event --ros-args -p device:=cuda -p language:=en -p action_server:=false
```

3. **YAML file** (see `config/speech_event_configuration.yaml`):
```yaml
speech_recognition:
  ros__parameters:
    # Audio Processing
    sample_rate: 16000
    input_sample_rate: 48000
    microphone_topic: "/audio"

    # Whisper ASR Configuration
    device: "cuda"
    compute_type: "float16"
    language: "en"
    whisper_model_id: "deepdml/faster-whisper-large-v3-turbo-ct2"

    # VAD Parameters
    speech_threshold: 0.7
    neg_threshold: 0.35
    min_silence_duration_ms: 1000
    min_speech_duration: 0.3
    max_speech_duration_s: 10.0
    pre_speech_buffer_ms: 200

    # Audio Gate
    intensity_threshold: 0.001
```
Then load with:
```bash
ros2 run speech_event speech_event --ros-args --params-file ~/ros2_ws/src/cssr4africa/cssr_system/speech_event/config/speech_event_configuration.yaml
```

# 🔄 Operating Modes

The speech recognition node supports two modes of operation controlled by the `action_server` parameter:

## Action Server Mode (default: `action_server: true`)
In this mode, the audio callback is **gated** — audio is only processed while an active goal is being handled by the action server. This prevents unnecessary CPU/GPU usage when the robot is not explicitly listening.

- Audio processing starts when a client sends a goal to `/speech_recognition_action`
- The node waits up to `goal.wait` seconds for speech to begin
- Once speech is detected, it records until silence ends the segment or `max_speech_duration_s` is reached
- The transcript is returned as the action result

## Standalone Mode (`action_server: false`)
In this mode, the node continuously listens and processes audio independent of any action client.

- Audio processing runs continuously
- Transcribed text is published to `/speech_event/text` after each speech segment
- Useful for passive speech monitoring or integration without an action-based flow

# 🚀 Running the Node

Source the workspace in the first terminal:
```bash
cd ~/ros2_ws && source install/setup.bash
```

## 1️⃣ Launch the robot (if not already running):
```bash
ros2 launch cssr_system pepper_bringup.launch.py robot_ip:=<robot_ip>
```

## 2️⃣ Run the Speech Event Recognition node:

In a new terminal, activate the Python environment:
```bash
# Activate the python environment (adjust path to your virtual environment)
source ~/sound/bin/activate
```

```bash
# Run the main speech_event node (action server mode by default)
ros2 run speech_event speech_event

# Run in standalone (topic-only) mode
ros2 run speech_event speech_event --ros-args -p action_server:=false
```

## 3️⃣ Optional: Run the recorder or localization nodes:
```bash
# Audio recorder (for debugging/data collection)
ros2 run speech_event speech_event_recorder

# Sound-source localization (requires 4-mic array)
ros2 run speech_event speech_event_localization
```

# 🖥️ Subscribed Topics

| Topic             | Type                                 | Description                                          |
|-------------------|--------------------------------------|------------------------------------------------------|
| `/audio`          | `naoqi_bridge_msgs/AudioBuffer`      | Raw multi-channel audio from the robot microphone (48 kHz, 4 channels). The `microphone_topic` parameter controls the actual topic name. |

# 🖥️ Published Topics and Actions

| Topic / Action                  | Type                                          | Description                                                                                 |
|---------------------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------|
| `/speech_event/vad_speech_prob` | `std_msgs/Float32`                            | Real-time VAD probability (0–1) for each 512-sample chunk.                                 |
| `/speech_event/text`            | `std_msgs/String`                             | Recognized speech text (published after each segment ends, in standalone mode only).       |
| `/speech_recognition_action`    | `dec_interfaces/action/SpeechRecognition`    | ROS2 action server for synchronous transcription requests (action server mode).            |

You can monitor the output with:
```bash
# Listen to VAD probabilities
ros2 topic echo /speech_event/vad_speech_prob

# Listen to transcribed text (standalone mode)
ros2 topic echo /speech_event/text
```

# 🎯 Action Server Interface

The node provides an action server at `/speech_recognition_action` for synchronous speech recognition. The goal's `wait` field specifies how long (in seconds) to wait for speech to **begin**. Once speech is detected, recording continues until the speaker stops (VAD-triggered silence) or `max_speech_duration_s` is reached.

## Goal Fields
| Field  | Type    | Description                                     |
|--------|---------|-------------------------------------------------|
| `wait` | float64 | Maximum seconds to wait for speech onset        |

## Result Fields
| Field           | Type   | Description                            |
|-----------------|--------|----------------------------------------|
| `transcription` | string | The transcribed speech text            |

## Feedback States
| Status          | Description                                                  |
|-----------------|--------------------------------------------------------------|
| `"waiting"`     | Waiting for speech to begin within the `wait` timeout        |
| `"speech"`      | Speech onset detected; recording in progress                 |
| `"transcribing"`| Speech segment finalized; Whisper ASR is running             |

## Action Goal Lifecycle
1. Client sends goal with `wait` seconds
2. Node enters `"waiting"` state — audio callback gates on
3. On speech onset: node enters `"speech"` state
4. On silence end or max duration: node enters `"transcribing"` state
5. Transcript is returned in the result; node returns to idle

**Concurrent goals are rejected** — if a goal is already being processed, the new goal is aborted immediately.

## Example Action Client (Python)
```python
import rclpy
from rclpy.action import ActionClient
from dec_interfaces.action import SpeechRecognition

# Create client
client = ActionClient(node, SpeechRecognition, '/speech_recognition_action')
goal_msg = SpeechRecognition.Goal()
goal_msg.wait = 5.0  # wait up to 5 seconds for speech to start
future = client.send_goal_async(goal_msg)
```

# 🎯 Sound Source Localization

The `speech_event_localization` node provides real-time sound source localization using **SRP-PHAT** (Steered Response Power with Phase Transform) beamforming via `pyroomacoustics`. It estimates the **azimuth-only direction** of sound sources in a 2D horizontal plane, designed for Pepper's 4-microphone planar configuration.

## Microphone Array Geometry (Pepper)

The localization node uses Pepper's fixed microphone positions (meters, in the Head frame):

| Microphone   | X (m)   | Y (m)   | Z (m)   |
|--------------|---------|---------|---------|
| Rear Left    | -0.0267 | +0.0343 | 0.2066  |
| Rear Right   | -0.0267 | -0.0343 | 0.2066  |
| Front Left   | +0.0313 | +0.0343 | 0.2066  |
| Front Right  | +0.0313 | -0.0343 | 0.2066  |

All microphones are at the same height, enabling azimuth-only (horizontal plane) localization.

## Localization Parameters

| Parameter                     | Description                                                        | Type/Range             | Default Value |
|-------------------------------|--------------------------------------------------------------------|------------------------|---------------|
| `sample_rate`                 | Audio sample rate (Hz)                                            | int                    | `48000`       |
| `microphone_topic`            | ROS topic for raw audio input                                     | string (topic name)    | `"/audio"`    |
| `speed_of_sound`              | Speed of sound in air (m/s)                                       | float                  | `343.0`       |
| `nfft`                        | FFT size for STFT (power of 2 recommended)                        | int                    | `1024`        |
| `angular_resolution`          | Number of azimuth scan angles (evenly spaced over 360°)           | int                    | `36`          |
| `freq_range_min`              | Minimum frequency for SRP-PHAT localization (Hz)                  | int                    | `500`         |
| `freq_range_max`              | Maximum frequency for SRP-PHAT localization (Hz)                  | int                    | `2500`        |
| `num_chunks_for_localization` | Minimum number of audio chunks to accumulate per localization run | int                    | `6`           |
| `update_rate_hz`              | Localization update frequency (Hz)                                | float                  | `2.0`         |
| `confidence_threshold`        | Minimum confidence score to publish results                       | float (0.0–1.0)        | `0.15`        |
| `intensity_threshold`         | RMS threshold to ignore quiet audio                               | float (RMS amplitude)  | `0.001`       |
| `enable_smoothing`            | Enable circular-mean temporal smoothing of azimuth estimates      | bool                   | `true`        |
| `smoothing_window`            | Number of past estimates to average when smoothing is enabled     | int                    | `5`           |

## Localization Published Topics

| Topic                                  | Type                              | Description                                                                                          |
|----------------------------------------|-----------------------------------|------------------------------------------------------------------------------------------------------|
| `/sound_localization/direction`        | `geometry_msgs/Vector3Stamped`    | 2D unit direction vector (x, y, z=0) in the `Head` frame                                            |
| `/sound_localization/azimuth`          | `std_msgs/Float32`                | Azimuth angle in degrees (0° = Front, 90° = Front-Left, 180° = Rear, 270° = Right)                  |
| `/sound_localization/confidence`       | `std_msgs/Float32`                | Confidence score (0–1) based on spatial spectrum contrast (peak-to-mean ratio)                      |
| `/sound_localization/source_pose`      | `geometry_msgs/PoseStamped`       | 3D pose representing source direction at 1 meter distance, in the `Head` frame                      |
| `/sound_localization/visualization`    | `visualization_msgs/Marker`       | RViz arrow marker showing sound direction (green = high confidence, red = low confidence)           |

## Usage Example

```bash
# Run with custom parameters
ros2 run speech_event speech_event_localization --ros-args \
  -p angular_resolution:=72 \
  -p freq_range_min:=300 \
  -p freq_range_max:=3000 \
  -p confidence_threshold:=0.2

# Monitor localization results
ros2 topic echo /sound_localization/azimuth
ros2 topic echo /sound_localization/confidence

# Visualize in RViz2
rviz2
# Add > Marker > Topic: /sound_localization/visualization
```

> **Note:** The localization node requires a **4-microphone planar array**. It performs azimuth-only localization (horizontal plane) and cannot estimate elevation. For best results, ensure the microphone array is level and the sound source is in the horizontal plane. Setting `freq_range_max` too high relative to microphone spacing may cause spatial aliasing — a warning is logged at startup if this is detected.

# 🎙️ Audio Recorder Node

The `speech_event_recorder` node records raw audio from the robot's microphone to WAV file(s). It is primarily used for debugging and data collection.

## Recorder Parameters

| Parameter        | Description                                                  | Type   | Default              |
|------------------|--------------------------------------------------------------|--------|----------------------|
| `mic_topic`      | ROS topic for raw audio input                               | string | `"/audio"`           |
| `output_base`    | Output file path prefix (without extension)                 | string | `"./pepper_audio"`   |
| `max_seconds`    | Stop recording after N seconds (0 = run until Ctrl-C)      | int    | `0`                  |
| `split_channels` | Also save a separate mono WAV per channel                   | bool   | `false`              |

The node writes a multi-channel WAV file named `<output_base>_<freq>Hz_<N>ch.wav`. If `split_channels` is `true`, it additionally writes one mono WAV per channel (e.g., `<output_base>_front_left_48000Hz.wav`).

## Recorder Usage

```bash
# Record to default path (./pepper_audio_48000Hz_4ch.wav)
ros2 run speech_event speech_event_recorder

# Record for 10 seconds with split per-channel WAVs
ros2 run speech_event speech_event_recorder --ros-args \
  -p output_base:=/tmp/my_recording \
  -p max_seconds:=10 \
  -p split_channels:=true
```

# 📁 Package Structure
```
speech_event/
├── config/
│   └── speech_event_configuration.yaml    # ROS2 parameters for speech_recognition node
├── data/
│   └── pepper_topics.yaml                 # Microphone topic mapping (used for reference)
├── models/
│   └── silero_vad.onnx                    # Pre-trained Silero VAD ONNX model
├── resource/
│   └── speech_event                       # Package marker for ament
├── speech_event/
│   ├── __init__.py
│   ├── speech_event_application.py        # Entry point: initializes and spins SpeechRecognitionNode
│   ├── speech_event_implementation.py     # Core VAD+ASR implementation (SpeechRecognitionNode)
│   ├── speech_event_localization.py       # SRP-PHAT sound source localization (SoundLocalizationNode)
│   └── speech_event_recorder.py           # Audio recorder utility (AudioRecorderNode)
├── package.xml                            # ROS2 package manifest
├── setup.py                               # Python package setup and entry points
├── setup.cfg                              # Setup configuration
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## ROS2 Node Names

| Executable                   | ROS2 Node Name       | Entry Point                                         |
|------------------------------|----------------------|-----------------------------------------------------|
| `speech_event`               | `speech_recognition` | `speech_event.speech_event_application:main`        |
| `speech_event_localization`  | `sound_localization` | `speech_event.speech_event_localization:main`       |
| `speech_event_recorder`      | `audio_recorder`     | `speech_event.speech_event_recorder:main`           |

# 🔍 Debugging Tips

## Speech Recognition
- Check that the audio topic is publishing data: `ros2 topic echo /audio --once`
- Verify the VAD probability stream: `ros2 topic echo /speech_event/vad_speech_prob`
- Listen to transcribed text (standalone mode): `ros2 topic echo /speech_event/text`
- If Whisper is slow, ensure CUDA is available: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Enable debug logging for detailed output: `ros2 run speech_event speech_event --ros-args --log-level debug`
- If running in action server mode, audio is only processed while a goal is active — ensure a client is sending goals
- If VAD never fires, try lowering `speech_threshold` or `intensity_threshold`
- If too much background noise is transcribed, raise `speech_threshold` or `intensity_threshold`

## Sound Localization
- Verify localization is running: `ros2 topic echo /sound_localization/azimuth`
- Check confidence values: `ros2 topic echo /sound_localization/confidence`
- If no results are published, try lowering `confidence_threshold` or `intensity_threshold`
- If directions are incorrect, verify the microphone array geometry in `speech_event_localization.py` matches your hardware
- Visualize results in RViz2: Add a Marker subscriber for `/sound_localization/visualization`
- Check for spatial aliasing warnings in startup logs (occurs if `freq_range_max` is too high for the mic spacing)

## Audio Recorder
- Verify the correct topic is used: the default is `/audio`, not `/pepper_robot/audio`
- Check WAV output: `file pepper_audio_48000Hz_4ch.wav && sox --i pepper_audio_48000Hz_4ch.wav`

# 💡 Support

For issues or questions:
- Create an issue on the CSSR4Africa GitHub repository
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2023 CSSR4Africa Consortium
Funded by African Engineering and Technology Network (Afretec)
Inclusive Digital Transformation Research Grant Programme

Last Updated: February 2026
