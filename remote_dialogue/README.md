<div align="center">
<h1>Remote Dialogue</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Remote Dialogue Package** connects Pepper to the remote `/ws/pepper` voice server (API v1.1). The server runs speech recognition, voice activity detection, turn-taking, retrieval, the language model and speech synthesis on a GPU server, in several languages (English and Kinyarwanda tested; Amharic, Swahili and French supported). This node only moves audio and messages between the robot and the server, and exposes each conversation turn as a ROS action for the behavior tree.

It replaces the local `SpeechRecognition → ConversationManager → SpeechWithFeedback` sequence with one `RemoteDialogueTurn`. The local pipeline is untouched and stays available.

## ✨ Key Features
- **Follows the API's audio contract**: one microphone channel streamed continuously at 16 kHz mono in 20 ms frames, silence included; no voice detection on the robot
- **Half duplex**: zeros instead of live audio whenever no turn is listening, while the robot speaks, and for 300 ms after
- **Exact end of playback**: replies are paced to Pepper's speaker in short pieces, and `playback_done` is timed from the samples actually sent
- **Self-reconnecting session**: reconnects after a drop with `"greet": false`, as the API asks
- **Language switching**: `set_language` service, confirmed by the server
- **Token kept out of ROS**: read from an environment variable, never a parameter, never logged

## ✅ Prerequisites
- **ROS2 Humble**, **Python 3.10**
- **naoqi_driver** with audio on (`ros2 launch dec_launch naoqi_driver.launch.py`), which publishes `/audio` and serves `/naoqi_driver/send_audio_buffer`
- The server's address and token, from the server team. The server runs in a GPU job of about two hours; a new job may come with a new address and token.

## 🛠️ Installation

```bash
cd ~/ros2_ws
colcon build --packages-up-to remote_dialogue
source install/setup.bash
```

The node runs in its own venv, `~/ros2_ws/.venvs/remote_dialogue` (`$VENV_REMOTE_DIALOGUE` in `venv_map.sh`):

```bash
python3 -m venv ~/ros2_ws/.venvs/remote_dialogue
~/ros2_ws/.venvs/remote_dialogue/bin/pip install -r ~/ros2_ws/src/pepper4dec/remote_dialogue/requirements.txt
```

## 🔧 Configuration
Parameters are in `config/remote_dialogue_configuration.yaml`. Two settings come from the environment instead:

| Variable | Meaning |
|---|---|
| `REMOTE_DIALOGUE_TOKEN` | The server's shared secret. Required. Kept out of ROS parameters, which anyone on the ROS graph can read |
| `REMOTE_DIALOGUE_URL` | `ws://<laptop-ip>:<port>/ws/pepper`. Overrides `server_url`, since the address changes with each GPU job |

| Parameter | Description | Default |
|---|---|---|
| `server_url` | Server URL without `?token=` | `""` |
| `language` | `eng`, `kin`, `amh`, `swh`, `fra` | `eng` |
| `output_sample_rate` | Reply audio rate; 48000 is Pepper's speaker rate | `48000` |
| `greet_on_connect` | Speak the welcome line on first connect | `false` |
| `microphone_topic` | naoqi microphone topic | `/audio` |
| `microphone_channel` | AudioBuffer channel sent: 0 front-left, 2 front-right, 3 rear-left, 5 rear-right | `0` |
| `mic_guard_s` | Keep sending zeros this long after the robot's speech ends | `0.3` |
| `send_audio_buffer_service` | Speaker service | `/naoqi_driver/send_audio_buffer` |
| `playback_piece_s` | Length of each piece sent to the speaker (max ~0.34 s) | `0.2` |
| `playback_lead_s` | Send the next piece when Pepper has at most this much left | `0.15` |
| `playback_tail_s` | Added to the computed playback end before `playback_done` | `0.15` |
| `turn_timeout_s` | Longest a whole turn may take | `90.0` |
| `language_timeout_s` | Wait for the server to confirm `set_language` | `5.0` |
| `reconnect_interval_s` | Pause between reconnect attempts | `3.0` |
| `verbose_mode` | Log every server message | `false` |

## 🚀 Running

```bash
export REMOTE_DIALOGUE_TOKEN=<token>
export REMOTE_DIALOGUE_URL=ws://<laptop-ip>:<port>/ws/pepper

# Terminal 1: the robot
ros2 launch dec_launch naoqi_driver.launch.py nao_ip:=<PEPPER_IP>

# Terminal 2: remote_dialogue + behavior_controller (remote_dialogue_pipeline.xml)
ros2 launch dec_launch remote_dialogue_pipeline.launch.py
ros2 launch dec_launch remote_dialogue_pipeline.launch.py language:=kin
```

Do not run `speech_event` at the same time: the server does the recognition.

The node alone, without the behavior tree:

```bash
ros2 launch remote_dialogue remote_dialogue.launch.py verbose_mode:=true
ros2 lifecycle set /remote_dialogue configure
ros2 lifecycle set /remote_dialogue activate

# A typed question (no microphone): Pepper speaks the answer
ros2 action send_goal /remote_dialogue dec_interfaces/action/RemoteDialogueTurn \
  "{text: 'Can you tell me about the Digital Experience Center?'}"

# A spoken turn: wait up to 20 s for the visitor
ros2 action send_goal /remote_dialogue dec_interfaces/action/RemoteDialogueTurn "{wait: 20.0}"

ros2 service call /remote_dialogue/set_language dec_interfaces/srv/SetDialogueLanguage "{language: kin}"
ros2 service call /remote_dialogue/reset dec_interfaces/srv/ResetDialogue "{greet: true}"
```

## 🖥️ ROS Interface

### Action Server

#### `/remote_dialogue` (`dec_interfaces/action/RemoteDialogueTurn`)
One conversation turn. The node opens the microphone gate, waits for the visitor to start speaking, plays the reply and returns once Pepper has finished speaking it. With `text` set, the question is sent as `text_input` and the microphone is not used. One turn at a time; a second goal is rejected.

| Goal | Type | Description |
|---|---|---|
| `wait` | float32 | Seconds to wait for the visitor to start speaking |
| `text` | string | Optional typed question |

| Result | Type | Description |
|---|---|---|
| `success` | bool | A reply was received and played to the end |
| `status` | string | `completed`, `no_speech`, `not_connected`, `disconnected`, `timeout`, `cancelled`, `busy` |
| `user_text` | string | What the server heard (or the typed `text`) |
| `assistant_text` | string | The reply, all sentences joined |
| `language` | string | Session language when the turn ended |
| `turn_id` | int32 | Server turn number |

Feedback: `state` = `waiting` / `listening` / `thinking` / `speaking`.

A turn that ends without a reply is aborted with the result filled in, so the behavior tree's `RemoteDialogueTurn` node still reports `status`.

### Services

| Service | Type | Description |
|---|---|---|
| `/remote_dialogue/set_language` | `dec_interfaces/srv/SetDialogueLanguage` | Switch language; succeeds once the server confirms |
| `/remote_dialogue/reset` | `dec_interfaces/srv/ResetDialogue` | New visitor: clear the conversation, optionally `greet` and change `language` |
| `/remote_dialogue/set_listening` | `std_srvs/srv/SetBool` | Keep the microphone gate open outside turns |

### Topics

| Topic | Type | Description |
|---|---|---|
| `/audio` (sub) | `naoqi_bridge_msgs/AudioBuffer` | Pepper's microphones |
| `/remote_dialogue/state` | `std_msgs/String` | Server state: `listening`, `thinking`, `speaking`, `idle` |
| `/remote_dialogue/transcript` | `dec_interfaces/msg/DialogueTranscript` | `user` and `assistant` lines with `turn_id` |
| `/remote_dialogue/language` | `std_msgs/String` (transient local) | Current language |
| `/remote_dialogue/connected` | `std_msgs/Bool` (transient local) | Server session ready |

### Service Clients

| Service | Type | Description |
|---|---|---|
| `/naoqi_driver/send_audio_buffer` | `naoqi_bridge_msgs/srv/SendAudioBuffer` | Reply audio, 48 kHz stereo, one piece per call |

## 🏗️ Architecture

```
/audio (48 kHz, 4 mics) ─► pick channel ─► 16 kHz ─► 20 ms frames ─► gate ─► server
                                                                      │
                                                    live only while a turn listens,
                                                    zeros otherwise and while Pepper speaks
server ─► reply PCM ─► queue ─► 48 kHz stereo pieces, paced ─► send_audio_buffer
                                    │
                         end of playback = samples sent ─► playback_done
server ─► state / transcript / language ─► ROS topics, action feedback and result
```

Three threads: the ROS executor (microphone callback, services, action), the WebSocket session (its own asyncio loop), and playback. The protocol and timing logic is in modules that do not touch ROS, so they are tested directly.

## 📁 Package Structure

```
remote_dialogue/
├── config/remote_dialogue_configuration.yaml
├── launch/remote_dialogue.launch.py
├── scripts/remote_dialogue                      # venv launcher for `ros2 run`
├── remote_dialogue/
│   ├── remote_dialogue_application.py           # entry point, ROS interface docs
│   ├── remote_dialogue_implementation.py        # lifecycle node
│   ├── remote_dialogue_session.py               # WebSocket client, reconnects
│   ├── remote_dialogue_logic.py                 # playback pacing, mic gate, turn outcome
│   └── remote_dialogue_audio.py                 # format conversion
├── test/
│   ├── fake_voice_server.py                     # stand-in for /ws/pepper
│   ├── test_audio.py, test_logic.py             # pure unit tests
│   ├── test_session.py                          # client against the fake server
│   ├── test_node.py                             # node + fake Pepper + fake server
│   └── test_flake8.py, test_pep257.py
├── requirements.txt
├── package.xml, setup.py, setup.cfg
└── README.md
```

## 🧪 Testing

```bash
cd ~/ros2_ws
colcon test --packages-select remote_dialogue
colcon test-result --verbose
```

No robot or real server is needed. `fake_voice_server.py` implements the documented protocol, with an energy detector in place of a real VAD. `test_node.py` runs the node against it and against a fake Pepper. The fake Pepper publishes `/audio` at real-time rate, serves `send_audio_buffer`, and echoes its own speaker into its microphones. The tests check that:
- no live audio reaches the server while a reply plays;
- `playback_done` arrives only after the last sample has played;
- typed and spoken turns, language changes, resets and reconnects all work.

## ⚠️ Not Yet Handled
These are left for the server team, once the pipeline runs on the robot:
- **No intent:** the server returns no intent (navigation, stop, yes/no), and it refuses movement requests.
- **No scripted lines:** the server can't speak exact text, such as the tour's fixed lines in Kinyarwanda or Amharic.
- **No exhibit context:** there's no way to tell the server which exhibit the robot is at.
- **No interrupting:** a visitor can't cut in while the robot speaks.

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>
