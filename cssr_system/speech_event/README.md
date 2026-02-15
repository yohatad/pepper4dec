<div align="center">
<h1>Speech Event Recognition and Localization</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Speech Event Recognition and Localization** package is a ROS2 node that provides real‑time speech recognition using Whisper ASR, voice activity detection with Silero VAD, and optional sound‑source localization. It processes the robot's microphone audio (48 kHz), detects speech segments, transcribes them with a low‑latency Whisper model, and publishes the recognized text. The module can also estimate the direction‑of‑arrival of a sound using beamforming on a 4‑microphone array.

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
pip install -r ~/ros2_ws/src/cssr4africa/cssr_system/speech_event/speech_event_requirements.txt
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

| Parameter                   | Description                                                     | Type/Range                 | Default Value |
|-----------------------------|-----------------------------------------------------------------|----------------------------|---------------|
| `sample_rate`               | Target sample rate for VAD/ASR (Hz)                            | int (16000 recommended)    | `16000`       |
| `input_sample_rate`         | Robot's native microphone rate (Hz)                            | int                        | `48000`       |
| `device`                    | PyTorch device for Whisper                                     | string: "cuda" or "cpu"    | `"cuda"`      |
| `compute_type`              | Whisper computation precision                                  | string: "float16"/"float32"| `"float16"`   |
| `language`                  | Language code for ASR (ISO 639-1)                              | string: "en", "fr", etc.   | `"en"`        |
| `whisper_model_id`          | Hugging Face model ID or path                                  | string                     | `"deepdml/faster-whisper-large-v3-turbo-ct2"` |
| `speech_threshold`          | VAD probability above which speech starts                      | float (0.0–1.0)            | `0.7`         |
| `neg_threshold`             | VAD probability below which silence is counted                 | float (0.0–1.0)            | `0.35`        |
| `min_silence_duration_ms`   | Minimum silence duration to end segment (ms)                   | int (milliseconds)         | `300`         |
| `max_speech_duration_s`     | Maximum allowed speech duration (seconds)                      | float (seconds)            | `10.0`        |
| `min_speech_duration`       | Minimum speech duration to keep (seconds)                      | float (seconds)            | `0.3`         |
| `pre_speech_buffer_ms`      | Audio to prepend before speech onset (ms)                      | int (milliseconds)         | `200`         |
| `intensity_threshold`       | RMS intensity gate to ignore quiet audio                       | float (RMS amplitude)      | `0.001`       |
| `microphone_topic`          | ROS topic for raw audio input                                  | string (topic name)        | `"/audio"`    |

> **Note:**
> The node uses a two‑threshold VAD system: speech starts when probability ≥ `speech_threshold`, and silence is counted when probability < `neg_threshold`. The segment ends after `min_silence_duration_ms` of continuous silence.

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
        'speech_threshold': 0.7
    }]
)
```

2. **Command line**:
```bash
ros2 run speech_event speech_event --ros-args -p device:=cuda -p language:=en
```

3. **YAML file** (see `config/speech_event_configuration.yaml`):
```yaml
speech_recognition:
  ros__parameters:
    sample_rate: 16000
    device: "cuda"
    language: "en"
    speech_threshold: 0.7
```
Then load with:
```bash
ros2 run speech_event speech_event --ros-args --params-file ~/ros2_ws/src/cssr4africa/cssr_system/speech_event/config/speech_event_configuration.yaml
```

# 🚀 Running the Node

**Run the `speech_event` node from the `speech_event` package:**

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
# Run the main speech_event node
ros2 run speech_event speech_event
```

## 3️⃣ Optional: Run the recorder or localization nodes:
```bash
# Audio recorder (for debugging)
ros2 run speech_event speech_event_recorder

# Sound‑source localization (requires 4‑mic array)
ros2 run speech_event speech_event_localization
```

# 🎯 Sound Source Localization

The `speech_event_localization` node provides real-time sound source localization using **SRP-PHAT** (Steered Response Power with Phase Transform) beamforming. It estimates the **azimuth-only direction** of sound sources in a 2D horizontal plane, ideal for planar microphone arrays like Pepper's 4-mic configuration.

## Localization Parameters

| Parameter                   | Description                                                     | Type/Range                 | Default Value |
|-----------------------------|-----------------------------------------------------------------|----------------------------|---------------|
| `sample_rate`               | Audio sample rate (Hz)                                         | int                        | `48000`       |
| `microphone_topic`          | ROS topic for raw audio input                                  | string (topic name)        | `"/audio"`    |
| `speed_of_sound`            | Speed of sound in air (m/s)                                    | float                      | `343.0`       |
| `nfft`                      | FFT size for STFT (power of 2 recommended)                     | int                        | `1024`        |
| `angular_resolution`        | Number of azimuth angles to test                               | int                        | `36`          |
| `freq_range_min`            | Minimum frequency for localization (Hz)                        | int                        | `500`         |
| `freq_range_max`            | Maximum frequency for localization (Hz)                        | int                        | `2500`        |
| `num_chunks_for_localization` | Number of audio chunks to accumulate                         | int                        | `6`           |
| `update_rate_hz`            | Localization update frequency (Hz)                             | float                      | `2.0`         |
| `confidence_threshold`      | Minimum confidence to publish results                          | float (0.0–1.0)            | `0.15`        |
| `intensity_threshold`       | RMS threshold to ignore quiet audio                            | float (RMS amplitude)      | `0.001`       |
| `enable_smoothing`          | Enable temporal smoothing of direction estimates               | bool                       | `true`        |
| `smoothing_window`          | Number of past estimates to average (if smoothing enabled)     | int                        | `5`           |

## Localization Topics

| Topic                                  | Type                              | Description                                                                 |
|----------------------------------------|-----------------------------------|-----------------------------------------------------------------------------|
| `/sound_localization/direction`        | `geometry_msgs/Vector3Stamped`    | 2D unit direction vector (x, y, z=0) in Head frame                         |
| `/sound_localization/azimuth`          | `std_msgs/Float32`                | Azimuth angle in degrees (0° = front, 90° = left, 180° = rear, 270° = right) |
| `/sound_localization/confidence`       | `std_msgs/Float32`                | Confidence score (0–1) based on spatial spectrum contrast                  |
| `/sound_localization/source_pose`      | `geometry_msgs/PoseStamped`       | 3D pose representing source direction at 1 meter distance                  |
| `/sound_localization/visualization`    | `visualization_msgs/Marker`       | RViz arrow marker showing sound direction (color = confidence)             |

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

# Visualize in RViz
rviz2
# Add > Marker > Topic: /sound_localization/visualization
```

> **Note:** The localization node requires a **4-microphone planar array**. It performs azimuth-only localization (horizontal plane) and cannot estimate elevation. For best results, ensure the microphone array is level and the sound source is in the horizontal plane.

# 🖥️ Output Topics and Actions
The node publishes the following data:

| Topic/Action                    | Type                              | Description                                                                 |
|---------------------------------|-----------------------------------|-----------------------------------------------------------------------------|
| `/speech_event/vad_speech_prob` | `std_msgs/Float32`                | Real‑time VAD probability (0–1) for each 512‑sample chunk.                 |
| `/speech_event/text`            | `std_msgs/String`                 | Recognized speech text (published after each segment ends).                |
| `/speech_recognition_action`    | `cssr_interfaces/action/SpeechRecognition` | ROS2 action server for synchronous transcription requests.                |

You can monitor the output with:
```bash
# Listen to VAD probabilities
ros2 topic echo /speech_event/vad_speech_prob

# Listen to transcribed text
ros2 topic echo /speech_event/text
```

# 🎯 Action Server Interface
The node provides an action server `/speech_recognition_action` for synchronous speech recognition. A client can send a goal with a `wait` duration (seconds) and receive the transcribed text as a result.

Example action client call (Python):
```python
import rclpy
from rclpy.action import ActionClient
from cssr_interfaces.action import SpeechRecognition

# Create client
client = ActionClient(node, SpeechRecognition, '/speech_recognition_action')
goal_msg = SpeechRecognition.Goal()
goal_msg.wait = 5.0  # listen for up to 5 seconds
future = client.send_goal_async(goal_msg)
```

# 📁 Package Structure
```
speech_event/
├── config/
│   └── speech_event_configuration.yaml    # ROS2 parameters configuration
├── data/
│   └── pepper_topics.yaml                 # Topic mappings
├── models/
│   └── silero_vad.onnx                    # Pre‑trained Silero VAD model
├── resource/
│   └── speech_event                       # Package marker for ament
├── speech_event/
│   ├── __init__.py
│   ├── speech_event_application.py        # Main node entry point
│   ├── speech_event_implementation.py     # Core VAD+ASR implementation
│   ├── speech_event_localization.py       # Sound source localization
│   └── speech_event_recorder.py           # Audio recorder utility
├── package.xml                            # ROS2 package manifest
├── setup.py                               # Python package setup
├── setup.cfg                              # Setup configuration
├── speech_event_requirements.txt          # Python dependencies
└── README.md                              # This file
```

# 🔍 Debugging Tips

## Speech Recognition
- Check that the audio topic (`/audio`) is publishing data: `ros2 topic echo /audio --once`
- Verify the VAD probability stream: `ros2 topic echo /speech_event/vad_speech_prob`
- Listen to transcribed text: `ros2 topic echo /speech_event/text`
- If Whisper is slow, ensure CUDA is available: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Monitor ROS logs for detailed output: `ros2 run speech_event speech_event --ros-args --log-level debug`

## Sound Localization
- Verify localization is running: `ros2 topic echo /sound_localization/azimuth`
- Check confidence values: `ros2 topic echo /sound_localization/confidence`
- If no detections occur, try lowering `confidence_threshold` or `intensity_threshold`
- If directions are incorrect, verify microphone array geometry in the code matches your hardware
- Visualize results in RViz: Add Marker subscriber to `/sound_localization/visualization`
- Check for frequency aliasing warnings in logs (occurs if `freq_max` is too high for mic spacing)

# 💡 Support

For issues or questions:
- Create an issue on the CSSR4Africa GitHub repository
- Contact: <a href="mailto:yohanneh@andrew.cmu.edu">yohanneh@andrew.cmu.edu</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2023 CSSR4Africa Consortium
Funded by African Engineering and Technology Network (Afretec)
Inclusive Digital Transformation Research Grant Programme

Last Updated: February 2026

