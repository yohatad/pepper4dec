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
- Python 3.8+
- CUDA-capable GPU (optional but recommended for Whisper acceleration)

## Python Environment Setup

```sh
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python virtual environment tools
sudo apt install python3.8-venv python3-pip -y

# Create a virtual environment
cd $HOME/workspace/pepper_rob_ws/src/cssr4africa_virtual_envs/
python3.8 -m venv cssr4africa_speech_event_env

# Activate the virtual environment
source cssr4africa_speech_event_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required Python packages
pip install -r ~/workspace/pepper_rob_ws/src/cssr4africa/cssr_system/speech_event/speech_event_requirements.txt

# Additional Whisper dependencies (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install faster-whisper
```

## Building the ROS2 Package

```sh
cd $HOME/workspace/pepper_rob_ws
source /opt/ros/humble/setup.bash  # or your ROS2 distribution
colcon build --packages-select speech_event
source install/setup.bash
```

# 🔧 Configuration Parameters  
The node is configured via a YAML file (`config/speech_event_configuration.yaml`). Key parameters are:

| Parameter                   | Description                                                     | Range/Values               | Default Value |
|-----------------------------|-----------------------------------------------------------------|----------------------------|---------------|
| `sample_rate`               | Target sample rate for VAD/ASR (Hz)                            | 16000, 22050, 44100        | `16000`       |
| `input_sample_rate`         | Pepper's native microphone rate (Hz)                           | 48000                      | `48000`       |
| `device`                    | PyTorch device for Whisper                                     | `"cuda"`, `"cpu"`          | `"cuda"`      |
| `compute_type`              | Whisper computation precision                                  | `"float16"`, `"float32"`   | `"float16"`   |
| `language`                  | Language code for ASR                                          | `"en"`, `"fr"`, etc.       | `"en"`        |
| `whisper_model_id`          | Hugging Face model ID                                          | string                     | `"deepdml/faster-whisper-large-v3-turbo-ct2"` |
| `speech_threshold`          | VAD probability above which speech starts                      | 0.0–1.0                    | `0.7`         |
| `neg_threshold`             | VAD probability below which silence is counted                 | 0.0–1.0                    | `0.35`        |
| `min_silence_duration_ms`   | Minimum silence duration to end a segment (ms)                 | positive integer           | `300`         |
| `max_speech_duration_s`     | Maximum allowed speech duration (seconds)                      | positive float             | `10.0`        |
| `min_speech_duration`       | Minimum speech duration to keep (seconds)                      | positive float             | `0.3`         |
| `pre_speech_buffer_ms`      | Audio to prepend before speech onset (ms)                      | positive integer           | `200`         |
| `intensity_threshold`       | RMS intensity gate to ignore quiet audio                       | positive float             | `0.001`       |
| `microphone_topic`          | ROS topic for raw audio                                        | string                     | `"/audio"`    |
| `useBeamforming`            | Enable beamforming for localization                            | `true`, `false`            | `true`        |
| `whisperModelSize`          | Whisper model size (legacy)                                    | `"tiny"`, `"base"`, etc.   | `"medium"`    |
| `asrBufferDuration`         | Duration of ASR buffer (seconds)                               | positive float             | `3.0`         |

> **Note:**  
> The node uses a two‑threshold VAD system: speech starts when probability ≥ `speech_threshold`, and silence is counted when probability < `neg_threshold`. The segment ends after `min_silence_duration_ms` of continuous silence.

# 🚀 Running the Node

**Run the `speech_event` node from the `speech_event` package:**

Source the workspace in the first terminal:
```bash
cd $HOME/workspace/pepper_rob_ws && source install/setup.bash
```

## 1️⃣ Launch the robot (if not already running):
```bash
ros2 launch cssr_system pepper_bringup.launch.py robot_ip:=<robot_ip>
```

## 2️⃣ Run the Speech Event Recognition node:

In a new terminal, activate the Python environment:
```bash
# Activate the python environment
source $HOME/workspace/pepper_rob_ws/src/cssr4africa_virtual_envs/cssr4africa_speech_event_env/bin/activate
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
│   └── speech_event_configuration.yaml    # Configuration parameters
├── data/
│   └── pepper_topics.yaml                 # Topic mappings
├── models/
│   └── silero_vad.onnx                    Pre‑trained Silero VAD model
├── speech_event/
│   ├── __init__.py
│   ├── speech_event_application.py        # Main node entry point
│   ├── speech_event_implementation.py     # Core VAD+ASR implementation
│   ├── speech_event_localization.py       # Beamforming localization
│   └── speech_event_recorder.py           # Audio recorder utility
├── package.xml
├── setup.py
├── setup.cfg
└── speech_event_requirements.txt
```

# 🔍 Debugging Tips
- Enable `verboseMode: true` in the configuration for detailed terminal output.
- Check that the audio topic (`/audio`) is publishing data: `ros2 topic echo /audio --once`
- Verify the VAD probability stream: `ros2 topic echo /speech_event/vad_speech_prob`
- If Whisper is slow, ensure CUDA is available: `python3 -c "import torch; print(torch.cuda.is_available())"`
- For localization, ensure the robot has a 4‑microphone array and that `useBeamforming` is `true`.

# 💡 Support

For issues or questions:
- Create an issue on the CSSR4Africa GitHub repository
- Contact: <a href="mailto:yohanneh@andrew.cmu.edu">yohanneh@andrew.cmu.edu</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2023 CSSR4Africa Consortium  
Funded by African Engineering and Technology Network (Afretec)  
Inclusive Digital Transformation Research Grant Programme

2025-11-08

