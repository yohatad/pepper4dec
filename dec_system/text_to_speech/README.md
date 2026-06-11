<div align="center">
<h1>Text-to-Speech (TTS)</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Text-to-Speech (TTS)** package is a ROS2 package designed to synthesize and play speech on the Pepper robot. It receives text sentences on `/text_to_speech/input` and speaks them as they arrive — enabling Pepper to start talking before the LLM has finished generating the full response. The package supports multiple synthesis backends including naoqi_ros, Kokoro-82M, and ElevenLabs with both streaming and file-based playback methods.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Five Backend Options**: naoqi_ros, kokoro_local, kokoro_pepper, elevenlabs_local, elevenlabs_pepper
- **Two Playback Methods**: stream (PCM chunks via ALAudioDevice) and file (SCP + ALAudioPlayer action)
- **ElevenLabs Streaming**: Audio starts playing within ~200 ms of the first API chunk
- **Sentence Queue**: Background playback thread ensures strict ordering
- **Barge-in Detection**: User speech during playback stops Pepper immediately
- **Microphone Muting**: Automatic mic control during playback
- **ROS2 Action Server**: `/text_to_speech` action for programmatic TTS calls with completion feedback

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **Pepper Robot** (for naoqi_ros, kokoro_pepper, elevenlabs_pepper backends)
- **espeak-ng** (for Kokoro phonemiser)

## Installation

### Package Installation

```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select text_to_speech
source install/setup.bash
```

### Python Dependencies

```bash
# Core (all backends)
pip install kokoro soundfile sounddevice scipy

# ElevenLabs backend only
pip install elevenlabs

# espeak-ng for Kokoro phonemiser
sudo apt-get install espeak-ng
```

## Configuration

Configuration is managed via `config/text_to_speech_configuration.yaml`.

### Backend Selection

| `engine` value | Synthesis | Playback |
|----------------|-----------|----------|
| `naoqi_ros` | Pepper on-board ALTextToSpeech | Pepper speakers (no extra deps) |
| `kokoro_local` | Kokoro-82M (local GPU/CPU) | Laptop speakers via sounddevice |
| `kokoro_pepper` | Kokoro-82M (local GPU/CPU) | Pepper speakers via naoqi_driver |
| `elevenlabs_local` | ElevenLabs API (streaming) | Laptop speakers via sounddevice |
| `elevenlabs_pepper` | ElevenLabs API (streaming) | Pepper speakers via naoqi_driver |

### Playback Method (pepper backends only)

| `playback_method` | How it works | Requirements |
|-------------------|--------------|--------------|
| `stream` | Raw PCM chunks → ALAudioDevice.sendRemoteBufferToOutput (no SCP) | None |
| `file` | SCP WAV to robot → ALAudioPlayer.loadFile → play_audio action | Passwordless SSH key to robot |

### All Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `engine` | Synthesis + playback backend | `naoqi_ros` |
| `playback_method` | `stream` or `file` (pepper backends only) | `stream` |
| `naoqi_speech_topic` | ROS topic for naoqi_bridge (naoqi_ros only) | `/speech` |
| `chars_per_second` | Estimated speaking rate for duration estimation | `12.0` |
| `speech_padding_s` | Extra wait after estimated speech end | `0.5` |
| `voice` | Kokoro-82M voice name (af_bella, af_heart, …) | `af_bella` |
| `sample_rate` | Synthesis sample rate in Hz | `24000` |
| `output_device` | sounddevice output device index (-1 = system default) | `-1` |
| `stream_volume` | PCM amplitude multiplier for stream mode | `1.0` |
| `elevenlabs_api_key` | ElevenLabs API key | `""` |
| `elevenlabs_voice_id` | ElevenLabs voice ID | `Rachel` |
| `elevenlabs_model` | ElevenLabs model ID | `eleven_turbo_v2_5` |
| `barge_in_threshold` | VAD probability to trigger barge-in | `0.85` |
| `barge_in_chunks` | Consecutive VAD chunks above threshold required | `3` |

## Running the Node

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Run the TTS node
ros2 run text_to_speech text_to_speech
```

### Sending Text

```bash
# Generic input topic (any source)
ros2 topic pub --once /text_to_speech/input std_msgs/String 'data: "Hello, I am Pepper."'

# Programmatic call via action server (blocks until speech is complete)
ros2 action send_goal /text_to_speech dec_interfaces/action/TTS "{text: 'Hello, how can I help you?'}"
```

## ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/text_to_speech/input` | `std_msgs/String` | Text to speak — accepts sentences from any source |
| `/speech_event/vad_speech_prob` | `std_msgs/Float32` | VAD probability used for barge-in detection |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/text_to_speech/speaking` | `std_msgs/Bool` | `True` while Pepper is speaking |
| `/speech` | `std_msgs/String` | Text forwarded to NAOqi TTS (naoqi_ros backend only) |

### Action Servers

| Action | Type | Description |
|--------|------|-------------|
| `/text_to_speech` | `dec_interfaces/action/TTS` | Speak text and block until complete |

### Service Clients

| Service | Type | Used when |
|---------|------|----------|
| `/speech_event/set_enabled` | `std_srvs/SetBool` | Mute/unmute mic during playback |
| `/naoqi_driver/load_audio_file` | `naoqi_bridge_msgs/srv/LoadAudioFile` | `file` playback mode |
| `/naoqi_driver/unload_audio_file` | `naoqi_bridge_msgs/srv/UnloadAudioFile` | `file` playback mode |
| `/naoqi_driver/send_audio_buffer` | `naoqi_bridge_msgs/srv/SendAudioBuffer` | `stream` playback mode |

### Action Clients

| Action | Type | Used when |
|--------|------|----------|
| `/naoqi_driver/play_audio` | `naoqi_bridge_msgs/action/PlayAudio` | `file` playback mode |

## Action Interface

**Action Type:** `dec_interfaces/action/TTS`

### Goal

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Text to speak |

### Result

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether speech completed successfully |
| `message` | string | Status message |

## Architecture

```
/text_to_speech/input  ──┐
              ├──► Sentence Queue ──► speak_sentence()
/text_to_speech action ──┘                          │
                                        ├── naoqi_ros     → publish /speech
                                        ├── kokoro_local  → sounddevice
                                        ├── kokoro_pepper ─┐
                                        └── elevenlabs_*  ─┤
                                                           ├─ stream → send_audio_buffer → ALAudioDevice
                                                           └─ file   → SCP + play_audio → ALAudioPlayer
```

1. **Input**: Text arrives via `/text_to_speech/input` topic or `/text_to_speech` action goal
2. **Queue**: Sentences are enqueued and drained in order by a background thread
3. **Synthesis**: Kokoro-82M (local GPU/CPU) or ElevenLabs API
4. **Playback**: Stream mode sends aligned PCM chunks directly to the robot; file mode SCPs a WAV file and plays via ALAudioPlayer
5. **Barge-in**: VAD probability on `/speech_event/vad_speech_prob` interrupts playback when the user starts speaking

## Testing

```bash
# Check node is running
ros2 node list

# Verify action server is available
ros2 action list

# Send a test message
ros2 topic pub --once /text_to_speech/input std_msgs/String 'data: "Hello, I am Pepper."'

# Test via action server
ros2 action send_goal /text_to_speech dec_interfaces/action/TTS "{text: 'Hello, how can I help you?'}"
```

### Testing Audio Playback Directly

```bash
cd ~/ros2_ws/src/pepper4dec/dec_system/text_to_speech

# Stream mode (robot speakers)
/home/yoha/tts_kokoro/bin/python3 tests/test_play_audio.py "Hello." --method stream

# File mode (robot speakers, requires SSH key)
/home/yoha/tts_kokoro/bin/python3 tests/test_play_audio.py "Hello." --method file

# Local speakers only
/home/yoha/tts_kokoro/bin/python3 tests/test_play_audio.py "Hello." --local
```

## Package Structure

```
text_to_speech/
├── config/
│   └── text_to_speech_configuration.yaml
├── data/
│   └── pepper_topics.yaml
├── resource/
│   └── text_to_speech
├── tests/
│   └── test_play_audio.py
├── text_to_speech/
│   ├── __init__.py
│   ├── text_to_speech_application.py
│   └── text_to_speech_implementation.py
├── package.xml
├── setup.py
├── setup.cfg
├── requirements.txt
└── README.md
```

## Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
