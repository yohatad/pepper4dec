<div align="center">
<h1> Text-to-Speech (TTS)</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Text-to-Speech (TTS)** package is a **ROS2** package designed to synthesize and play speech on the Pepper robot. It receives sentences from `/conversation_manager/response_stream` and speaks them as they arrive — enabling Pepper to start talking before the LLM has finished generating the full response.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Three Backend Options**: naoqi_ros, kokoro_local, kokoro_pepper
- **Sentence Queue**: Background playback thread ensures strict ordering
- **Barge-in Detection**: User speech during playback stops Pepper immediately
- **Microphone Muting**: Automatic mic control during playback
- **ROS2 Action Server**: `/tts` action for programmatic TTS calls

# 🛠️ Installation 

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **Pepper Robot** (for naoqi_ros and kokoro_pepper backends)

## Package Installation

1. **Clone and Build the Workspace**
```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone <repository-url>

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select text_to_speech
source install/setup.bash
```

2. **Set Up Python Virtual Environment**
```bash
# Create virtual environment
python3.10 -m venv ~/tts_virtual_env

# Activate the virtual environment
source ~/tts_virtual_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

3. **Install Python Dependencies**
```bash
# Install package requirements
pip install -r ~/ros2_ws/src/pepper4dec/dec_system/text_to_speech/requirements.txt
```

# 🔧 Configuration Parameters
The configuration is managed via `config/text_to_speech_configuration.yaml`:

| Parameter | Description | Range/Values | Default |
|-----------|-------------|--------------|---------|
| `engine` | Synthesis backend | `naoqi_ros`, `kokoro_local`, `kokoro_pepper` | `naoqi_ros` |
| `playback_method` | Playback method for kokoro_pepper | `file`, `stream` | `file` |
| `naoqi_speech_topic` | ROS2 topic for naoqi_bridge | string | `/speech` |
| `chars_per_second` | Estimated speaking rate | float | `12.0` |
| `speech_padding_s` | Extra seconds to wait after speech | float | `0.5` |
| `voice` | Kokoro-82M voice name | `af_bella`, `af_heart`, etc. | `af_bella` |
| `sample_rate` | Output sample rate (Hz) | integer | `24000` |
| `output_device` | sounddevice output device index | integer (-1 = system default) | `-1` |
| `barge_in_threshold` | VAD probability to trigger barge-in | `[0.0 - 1.0]` | `0.85` |
| `barge_in_chunks` | Consecutive chunks above threshold required | integer | `3` |

> **Note:**  
> The **kokoro_local** backend requires `sounddevice` package. The **kokoro_pepper** backend requires `kokoro` and `soundfile` packages.

# 🚀 Running the Node

## Launch All Components
```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch with default configuration
ros2 launch text_to_speech text_to_speech.launch.py
```

## Manual Node Execution
```bash
# Activate Python environment
source ~/tts_virtual_env/bin/activate

# Run TTS node
ros2 run text_to_speech text_to_speech
```

### Using the Action Server
```bash
ros2 action send_goal /tts dec_interfaces/action/TTS "{text: 'Hello, how can I help you?'}"
```

# 🖥️ Output
The node publishes speaking status and handles audio playback.

## Topic Structure
- **Subscriptions**:
  - `/conversation_manager/response_stream` (std_msgs/String)
  - `/speech_event/vad_speech_prob` (std_msgs/Float32)

- **Publishers**:
  - `/text_to_speech/speaking` (std_msgs/Bool)

- **Service Clients**:
  - `/speech_event/set_enabled` (std_srvs/SetBool)
  - `/naoqi_driver/load_audio_file` (naoqi_bridge_msgs/srv/LoadAudioFile)
  - `/naoqi_driver/unload_audio_file` (naoqi_bridge_msgs/srv/UnloadAudioFile)
  - `/naoqi_driver/send_audio_buffer` (naoqi_bridge_msgs/srv/SendAudioBuffer)

- **Action Clients**:
  - `/naoqi_driver/play_audio` (naoqi_bridge_msgs/action/PlayAudio)

- **Action Servers**:
  - `/tts` (dec_interfaces/action/TTS)

## Verification
To verify the node is publishing data:

```bash
# Monitor speaking output
ros2 topic echo /text_to_speech/speaking

# Check node status
ros2 node list
ros2 topic list
```

# 🏗️ Architecture
The TTS system consists of three main components:

1. **Sentence Receiver**: Subscribes to `/conversation_manager/response_stream` and `/tts` action
2. **Queue Manager**: Thread-safe queue that orders sentences for playback
3. **Playback Backend**: 
   - **naoqi_ros**: Publishes to naoqi_bridge topic
   - **kokoro_local**: Synthesizes locally, plays via sounddevice
   - **kokoro_pepper**: Synthesizes locally, sends audio to robot via naoqi_driver

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a><br>

# 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.

