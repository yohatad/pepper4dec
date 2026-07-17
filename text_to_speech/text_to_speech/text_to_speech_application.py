#!/home/yoha/tts_virtual_env/bin/python3

""" text_to_speech_application.py

Entry point for the TextToSpeechNode lifecycle node.
Running this node lets Pepper convert streamed text into spoken audio with
low latency.

The node pulls sentences from the /text_to_speech/input topic (typically fed
by an LLM response stream) and speaks them as they arrive, so Pepper can
start talking before the full response has finished generating. Sentences
are queued and drained sequentially by a background playback thread, which
ensures strict ordering even while new sentences keep arriving.

Synthesis/playback is handled by one of several interchangeable backends,
selected via the 'engine' config key: 'naoqi_ros' (Pepper's on-board
ALTextToSpeech via naoqi_bridge), 'kokoro_local' / 'kokoro_pepper' (Kokoro-82M
synthesised locally and played on the laptop or on Pepper's speakers), and
'elevenlabs_local' / 'elevenlabs_pepper' (ElevenLabs streaming TTS played
locally or on Pepper). The node also exposes a /text_to_speech action server
so other nodes can request ad-hoc TTS playback and wait for completion.

Subscribers:
    /text_to_speech/input (std_msgs/String)
        Sentences to speak, enqueued and played in order as they arrive.

Publishers:
    /text_to_speech/speaking (std_msgs/Bool)
        True while Pepper is actively speaking, false otherwise.
    <naoqi_speech_topic> (std_msgs/String)
        Plain-text sentences for naoqi_bridge to speak (naoqi_ros engine only).

Services:
    /speech_event/set_enabled (std_srvs/SetBool)
        Client used to mute/unmute the microphone during playback.
    /naoqi_driver/load_audio_file (naoqi_bridge_msgs/srv/LoadAudioFile)
        Client used to upload synthesised WAV audio to the robot (Pepper backends).
    /naoqi_driver/unload_audio_file (naoqi_bridge_msgs/srv/UnloadAudioFile)
        Client used to free an audio file on the robot after playback.
    /naoqi_driver/send_audio_buffer (naoqi_bridge_msgs/srv/SendAudioBuffer)
        Client used to stream raw PCM audio buffers to the robot (stream playback method).

Actions:
    /naoqi_driver/play_audio (naoqi_bridge_msgs/action/PlayAudio)
        Client used to play an uploaded audio file on Pepper's speakers.
    /text_to_speech (dec_interfaces/action/TTS)
        Server that accepts text, queues it for speech, and reports completion.

Parameters (loaded from text_to_speech_configuration.yaml):
    engine (str, default: "naoqi_ros")
    naoqi_speech_topic (str, default: "/speech")
    chars_per_second (float, default: 12.0)
    speech_padding_s (float, default: 0.5)
    voice (str, default: "af_bella")
    sample_rate (int, default: 24000)
    output_device (int, default: -1)
    playback_method (str, default: "stream")
    stream_volume (float, default: 1.0)
    elevenlabs_api_key (str, default: "")
    elevenlabs_voice_id (str, default: "21m00Tcm4TlvDq8ikWAM")
    elevenlabs_model (str, default: "eleven_turbo_v2_5")
    elevenlabs_stability (float, default: 0.5)
    elevenlabs_similarity_boost (float, default: 0.75)
    elevenlabs_style (float, default: 0.0)
    elevenlabs_speed (float, default: 1.0)

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: April 2025
Version: v1.0
"""

import queue
import sys
import threading
import time

import rclpy
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn

from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from dec_interfaces.action import TTS
from naoqi_bridge_msgs.action import PlayAudio
from naoqi_bridge_msgs.srv import LoadAudioFile, UnloadAudioFile, SendAudioBuffer

from .text_to_speech_implementation import (
    AudioPlayer,
    audio_to_wav_bytes,
    collect_and_resample,
    estimate_duration,
    iter_robot_chunks,
    load_configuration,
    prepare_stream_audio,
    resample_chunks,
    split_into_sentences,
    synthesize_kokoro,
    stream_elevenlabs,
)

SOFTWARE_VERSION = "v1.0"


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class TextToSpeechNode(LifecycleNode):
    """Lifecycle node that converts text to speech and plays it on Pepper's speakers."""

    def __init__(self, config: dict):
        super().__init__("text_to_speech")
        # Store config — heavy initialisation deferred to on_configure
        self.config    = config
        self.node_name = self.get_name()
        self.engine    = config["engine"]

    # ── Lifecycle callbacks ─────────────────────────────────────────────────────

    def on_configure(self, _state) -> TransitionCallbackReturn:
        """Initialize the TTS backend and create publishers, service clients, and the action server."""
        self.get_logger().info(
            f"{self.node_name}: configuring — engine={self.engine}"
        )

        # Internal state
        self.sentence_queue: queue.Queue = queue.Queue()
        self.is_speaking      = False
        self.stop_requested   = False
        self.state_lock       = threading.Lock()
        self.shutdown         = False
        self.audio_player     = None
        self.play_goal_handle = None
        self.play_done_event  = threading.Event()

        # Per-backend initialisation
        if self.engine in ("kokoro_local", "kokoro_pepper"):
            self.warmup_kokoro()

        if self.engine in ("kokoro_local", "elevenlabs_local"):
            self.audio_player = AudioPlayer(
                sample_rate=self.config["sample_rate"],
                output_device=self.config["output_device"],
            )

        # Managed publishers
        self.speaking_pub = self.create_lifecycle_publisher(Bool, "/text_to_speech/speaking", 10)

        if self.engine == "naoqi_ros":
            self.naoqi_pub = self.create_lifecycle_publisher(
                String, self.config["naoqi_speech_topic"], 10
            )
            self.get_logger().info(
                f"{self.node_name}: naoqi_ros backend — "
                f"publishing to '{self.config['naoqi_speech_topic']}'"
            )

        # Service clients
        self.mic_client = self.create_client(SetBool, "/speech_event/set_enabled")

        if self.engine in ("kokoro_pepper", "elevenlabs_pepper"):
            self.load_client   = self.create_client(LoadAudioFile, "/naoqi_driver/load_audio_file")
            self.unload_client = self.create_client(UnloadAudioFile, "/naoqi_driver/unload_audio_file")
            self._play_client  = ActionClient(self, PlayAudio, "/naoqi_driver/play_audio")
            self._send_buffer_client = self.create_client(SendAudioBuffer, "/naoqi_driver/send_audio_buffer")

        # Action server — callback groups for concurrent operation
        self._stream_cb_group = MutuallyExclusiveCallbackGroup()
        self._action_cb_group = MutuallyExclusiveCallbackGroup()

        self._tts_action = ActionServer(
            self, TTS, "/text_to_speech", self.execute_tts_action,
            callback_group=self._action_cb_group,
        )

        self.get_logger().info(f"{self.node_name}: configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, _state) -> TransitionCallbackReturn:
        """Activate publishers, subscribe to the text input topic, and start the playback thread."""
        super().on_activate(_state)

        self._sub_tts_input = self.create_subscription(
            String, "/text_to_speech/input", self.stream_sentence_callback, 10,
            callback_group=self._stream_cb_group,
        )

        self.shutdown = False
        self._playback_thread = threading.Thread(
            target=self.playback_loop, name="tts_playback", daemon=True
        )
        self._playback_thread.start()

        self.get_logger().info(f"{self.node_name}: activated — ready to speak")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, _state) -> TransitionCallbackReturn:
        """Stop the playback thread, drain the queue, and destroy subscriptions."""
        self.cleanup()   # stops thread, drains queue

        self.destroy_subscription(self._sub_tts_input)

        super().on_deactivate(_state)
        self.get_logger().info(f"{self.node_name}: deactivated")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, _state) -> TransitionCallbackReturn:
        """Destroy publishers, service clients, the action server, and release the audio player."""
        self.destroy_lifecycle_publisher(self.speaking_pub)
        if self.engine == "naoqi_ros" and hasattr(self, "naoqi_pub"):
            self.destroy_lifecycle_publisher(self.naoqi_pub)

        self._tts_action.destroy()
        self.destroy_client(self.mic_client)

        if self.engine in ("kokoro_pepper", "elevenlabs_pepper"):
            self.destroy_client(self.load_client)
            self.destroy_client(self.unload_client)
            self.destroy_client(self._send_buffer_client)
            self._play_client.destroy()

        if self.audio_player is not None:
            self.audio_player = None

        self.get_logger().info(f"{self.node_name}: cleaned up")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, _state) -> TransitionCallbackReturn:
        """Log the shutdown transition before the node is destroyed."""
        self.get_logger().info(f"{self.node_name}: shutting down")
        return TransitionCallbackReturn.SUCCESS

    # ── Kokoro warm-up ──────────────────────────────────────────────────────────

    def warmup_kokoro(self):
        """Synthesise a short dummy phrase to pre-load the Kokoro model."""
        self.get_logger().info(f"{self.node_name}: warming up Kokoro model…")
        try:
            synthesize_kokoro("Hello.", self.config["voice"], self.config["sample_rate"])
            self.get_logger().info(f"{self.node_name}: Kokoro warm-up complete.")
        except Exception as e:
            self.get_logger().warn(
                f"{self.node_name}: Kokoro warm-up failed (non-critical): {e}"
            )

    # ── Stream sentence callback ─────────────────────────────────────────────────

    def stream_sentence_callback(self, msg: String):
        """Enqueue each sentence published by conversation_manager."""
        sentence = msg.data.strip()
        if sentence:
            self.sentence_queue.put(sentence)

    # ── TTS action server ───────────────────────────────────────────────────────

    def execute_tts_action(self, goal_handle):
        """Speak the text in the TTS action goal and return when done."""
        text = (goal_handle.request.text or "").strip()
        result = TTS.Result()

        if not text:
            result.success = False
            result.message = "Empty text"
            goal_handle.abort()
            return result

        feedback_msg = TTS.Feedback()
        feedback_msg.status = "queuing"
        goal_handle.publish_feedback(feedback_msg)

        done_event = threading.Event()

        def sentinel():
            done_event.set()

        for sentence in split_into_sentences(text):
            self.sentence_queue.put(sentence)
        self.sentence_queue.put(sentinel)

        feedback_msg.status = "speaking"
        goal_handle.publish_feedback(feedback_msg)

        done_event.wait(timeout=60.0)

        result.success = True
        result.message = "OK"
        goal_handle.succeed()
        return result

    # ── Background playback loop ─────────────────────────────────────────────────

    def playback_loop(self):
        """
        Drain the sentence queue sequentially.

        Queue items are either:
          str       — sentence to speak
          callable  — sentinel signalling that a /text_to_speech action goal is done
        """
        while not self.shutdown:
            try:
                item = self.sentence_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:  # shutdown sentinel
                break

            if callable(item):
                try:
                    item()
                except Exception:
                    pass
                continue

            self.speak_sentence(item)

    def speak_sentence(self, sentence: str):
        """Dispatch a single sentence to the active backend."""
        with self.state_lock:
            self.stop_requested = False
            self.is_speaking = True

        self.publish_speaking(True)

        try:
            if self.engine == "naoqi_ros":
                self.speak_naoqi(sentence)
            elif self.engine == "kokoro_local":
                self.speak_kokoro_local(sentence)
            elif self.engine == "kokoro_pepper":
                self.speak_kokoro_pepper(sentence)
            elif self.engine == "elevenlabs_local":
                self.speak_elevenlabs_local(sentence)
            elif self.engine == "elevenlabs_pepper":
                self.speak_elevenlabs_pepper(sentence)
            else:
                self.get_logger().error(
                    f"{self.node_name}: unknown engine '{self.engine}'"
                )
        except Exception as e:
            self.get_logger().error(
                f"{self.node_name}: playback error — '{sentence[:40]}': {e}"
            )
        finally:
            with self.state_lock:
                if self.sentence_queue.empty():
                    self.is_speaking = False
                    self.publish_speaking(False)

    # ── naoqi_ros backend ────────────────────────────────────────────────────────

    def speak_naoqi(self, sentence: str):
        """
        Publish text to the NAOqi speech topic and wait for estimated duration.

        Mic stays enabled — Pepper's hardware echo cancellation prevents
        feedback.
        """
        msg = String()
        msg.data = sentence
        self.naoqi_pub.publish(msg)
        self.get_logger().info(f"{self.node_name}: [naoqi_ros] speaking: '{sentence}'")

        duration = estimate_duration(
            sentence,
            self.config["chars_per_second"],
            self.config["speech_padding_s"],
        )
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            time.sleep(0.05)
            with self.state_lock:
                if self.stop_requested:
                    return

    # ── kokoro_local backend ─────────────────────────────────────────────────────

    def speak_kokoro_local(self, sentence: str):
        """Synthesise with Kokoro and play on the local machine's speakers."""
        self.get_logger().info(
            f"{self.node_name}: [kokoro_local] synthesising: '{sentence}'"
        )
        try:
            audio = synthesize_kokoro(
                sentence, self.config["voice"], self.config["sample_rate"]
            )
        except RuntimeError as e:
            self.get_logger().error(str(e))
            return

        self.set_mic_enabled(False)
        try:
            duration_s = len(audio) / self.config["sample_rate"]
            self.get_logger().info(
                f"{self.node_name}: [kokoro_local] playing {duration_s:.2f}s"
            )
            completed = self.audio_player.play(audio)
            if not completed:
                self.get_logger().info(
                    f"{self.node_name}: [kokoro_local] playback interrupted"
                )
        finally:
            self.set_mic_enabled(True)

    # ── kokoro_pepper backend ────────────────────────────────────────────────────

    def speak_kokoro_pepper(self, sentence: str):
        """
        Synthesise with Kokoro and play through Pepper's speakers via naoqi_driver.

        Supports two playback methods (configured via playback_method):
          "file"   - load_audio_file + play_audio action (default, supports long audio)
          "stream" - send_audio_buffer service (low-latency, max ~170ms per call)
        """
        playback_method = self.config.get("playback_method", "file")
        
        self.get_logger().info(
            f"{self.node_name}: [kokoro_pepper] synthesising: '{sentence}' "
            f"(method={playback_method})"
        )

        # 1. Synthesise
        try:
            audio = synthesize_kokoro(
                sentence, self.config["voice"], self.config["sample_rate"]
            )
        except RuntimeError as e:
            self.get_logger().error(str(e))
            return

        # 2. Encode to WAV bytes in memory
        wav_bytes = audio_to_wav_bytes(audio, self.config["sample_rate"])

        # 3. Play based on configured method
        if playback_method == "stream":
            self.play_via_stream(wav_bytes)
        else:
            self.play_via_file(wav_bytes)

    # ── elevenlabs_local backend ─────────────────────────────────────────────────

    def speak_elevenlabs_local(self, sentence: str):
        """
        Stream from ElevenLabs and play on the local machine's speakers.
        Audio starts as soon as the first API chunk arrives (~200 ms TTFA).
        """
        self.get_logger().info(
            f"{self.node_name}: [elevenlabs_local] streaming: '{sentence}'"
        )
        self.set_mic_enabled(False)
        try:
            gen, api_rate = stream_elevenlabs(
                sentence,
                voice_id=self.config["elevenlabs_voice_id"],
                api_key=self.config["elevenlabs_api_key"],
                model_id=self.config.get("elevenlabs_model", "eleven_turbo_v2_5"),
                sample_rate=self.config["sample_rate"],
                stability=self.config.get("elevenlabs_stability", 0.5),
                similarity_boost=self.config.get("elevenlabs_similarity_boost", 0.75),
                style=self.config.get("elevenlabs_style", 0.0),
                speed=self.config.get("elevenlabs_speed", 1.0),
            )
            completed = self.audio_player.play_chunks(
                resample_chunks(gen, api_rate, self.config["sample_rate"])
            )
            if not completed:
                self.get_logger().info(
                    f"{self.node_name}: [elevenlabs_local] playback interrupted"
                )
        except RuntimeError as e:
            self.get_logger().error(str(e))
        finally:
            self.set_mic_enabled(True)

    # ── elevenlabs_pepper backend ────────────────────────────────────────────────

    def speak_elevenlabs_pepper(self, sentence: str):
        """
        Stream from ElevenLabs and play through Pepper's speakers.

        stream mode: Chunks are accumulated at api_rate until a full 16 384-frame
                     robot chunk is ready, then resampled to 48 kHz and sent —
                     avoiding click artifacts from resampling tiny API chunks.
        file mode:   Collects full audio then uses SCP + ALAudioPlayer (fallback).
        """
        playback_method = self.config.get("playback_method", "stream")
        self.get_logger().info(
            f"{self.node_name}: [elevenlabs_pepper] streaming: '{sentence}' "
            f"(method={playback_method})"
        )

        try:
            gen, api_rate = stream_elevenlabs(
                sentence,
                voice_id=self.config["elevenlabs_voice_id"],
                api_key=self.config["elevenlabs_api_key"],
                model_id=self.config.get("elevenlabs_model", "eleven_turbo_v2_5"),
                sample_rate=self.config["sample_rate"],
                stability=self.config.get("elevenlabs_stability", 0.5),
                similarity_boost=self.config.get("elevenlabs_similarity_boost", 0.75),
                style=self.config.get("elevenlabs_style", 0.0),
                speed=self.config.get("elevenlabs_speed", 1.0),
            )
        except RuntimeError as e:
            self.get_logger().error(str(e))
            return

        if playback_method == "stream":
            self.stream_elevenlabs_to_robot(gen, api_rate)
        else:
            # file mode: collect full audio → WAV bytes → play via action
            audio = collect_and_resample(gen, api_rate, self.config["sample_rate"])
            if len(audio) == 0:
                return
            self.play_via_file(audio_to_wav_bytes(audio, self.config["sample_rate"]))

    def stream_elevenlabs_to_robot(self, gen, api_rate: int):
        """
        Accumulate streaming float32 chunks into aligned robot buffers,
        resample to 48 kHz stereo, and send via send_audio_buffer.
        """
        volume = float(self.config.get("stream_volume", 1.0))
        self.set_mic_enabled(False)
        try:
            for audio_list, wait_time in iter_robot_chunks(gen, api_rate, volume):
                with self.state_lock:
                    if self.stop_requested:
                        return
                if not self.send_audio_buffer(audio_list):
                    return
                time.sleep(wait_time)
        finally:
            self.set_mic_enabled(True)

    def play_via_file(self, wav_bytes: bytes):
        """
        Play audio via load_audio_file + play_audio action.

        Sends raw WAV bytes to naoqi_driver, which SCPs them to the robot and
        calls ALAudioPlayer.loadFile internally.
        """
        file_id = self.load_audio_file_from_bytes(wav_bytes)
        if file_id < 0:
            return

        self.play_done_event.clear()
        self.play_goal_handle = None

        goal = PlayAudio.Goal()
        goal.file_id = file_id
        goal.volume  = 0.85
        goal.pan     = 0.0
        goal.loop    = False

        send_future = self._play_client.send_goal_async(goal)
        send_future.add_done_callback(self.on_play_goal_response)
        self.play_done_event.wait(timeout=60.0)

        self.unload_audio_file(file_id)

    def play_via_stream(self, wav_bytes: bytes):
        """Play audio via send_audio_buffer service (low-latency streaming)."""
        volume = float(self.config.get("stream_volume", 1.0))
        self.set_mic_enabled(False)
        try:
            for audio_list, wait_time in prepare_stream_audio(wav_bytes, volume):
                with self.state_lock:
                    if self.stop_requested:
                        break
                if not self.send_audio_buffer(audio_list):
                    self.get_logger().error(f"{self.node_name}: send_audio_buffer failed")
                    break
                time.sleep(wait_time)
        finally:
            self.set_mic_enabled(True)

    def send_audio_buffer(self, audio_data: list) -> bool:
        """Send raw audio buffer to naoqi_driver's send_audio_buffer service."""
        if not self._send_buffer_client.service_is_ready():
            self.get_logger().warn(
                f"{self.node_name}: send_audio_buffer service not ready"
            )
            return False

        req = SendAudioBuffer.Request()
        req.audio_data = audio_data
        future = self._send_buffer_client.call_async(req)

        # Wait without spinning — the MultiThreadedExecutor resolves the future.
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=2.0):
            self.get_logger().error(f"{self.node_name}: send_audio_buffer timed out")
            return False

        res = future.result()
        if not res.success:
            self.get_logger().warn(
                f"{self.node_name}: send_audio_buffer failed: {res.message}"
            )
            return False

        return True

    def on_play_goal_response(self, future):
        """Called when the play_audio server accepts or rejects the goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"{self.node_name}: play_audio goal rejected")
            self.play_done_event.set()
            return

        self.play_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_play_result)

    def on_play_result(self, future):
        """Called when play_audio finishes (completed, cancelled, or aborted)."""
        try:
            result = future.result().result
            self.get_logger().info(
                f"{self.node_name}: play_audio done — "
                f"status={result.status}, played={result.played_secs:.1f}s"
            )
        except Exception as e:
            self.get_logger().warn(f"{self.node_name}: play_audio result error: {e}")
        finally:
            self.play_goal_handle = None
            self.play_done_event.set()

    # ── Audio-player service helpers ─────────────────────────────────────────────

    def load_audio_file_from_bytes(self, wav_bytes: bytes) -> int:
        """
        Send raw WAV bytes to naoqi_driver's load_audio_file service.

        naoqi_driver writes the bytes to the robot via ALFileManager and calls
        ALAudioPlayer.loadFile() internally.  Returns the file_id, or -1 on failure.
        """
        return self.call_load_service(remote_path="", audio_data=list(wav_bytes))

    def load_audio_file_from_path(self, remote_path: str) -> int:
        """Call load_audio_file with a path already on the robot. Returns file_id or -1."""
        return self.call_load_service(remote_path=remote_path, audio_data=[])

    def call_load_service(self, remote_path: str, audio_data: list) -> int:
        """Shared implementation for load_audio_file service calls."""
        if not self.load_client.service_is_ready():
            self.get_logger().warn(
                f"{self.node_name}: load_audio_file service not ready"
            )
            return -1

        req = LoadAudioFile.Request()
        req.remote_path = remote_path
        req.audio_data  = audio_data
        future = self.load_client.call_async(req)

        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        if not done.wait(timeout=30.0):  # SCP can take a few seconds
            self.get_logger().error(f"{self.node_name}: load_audio_file timed out")
            return -1

        res = future.result()
        if not res.success:
            self.get_logger().error(
                f"{self.node_name}: load_audio_file failed: {res.message}"
            )
            return -1

        return res.file_id

    def unload_audio_file(self, file_id: int) -> None:
        """Call unload_audio_file service (fire-and-forget)."""
        if not self.unload_client.service_is_ready():
            return
        req = UnloadAudioFile.Request()
        req.file_id = file_id
        self.unload_client.call_async(req)

    # ── Mic mute helpers ─────────────────────────────────────────────────────────

    def set_mic_enabled(self, enabled: bool):
        """Call /speech_event/set_enabled (best-effort, non-blocking)."""
        if not self.mic_client.service_is_ready():
            return
        req = SetBool.Request()
        req.data = enabled
        self.mic_client.call_async(req)

    # ── Speaking state publisher ──────────────────────────────────────────────────

    def publish_speaking(self, speaking: bool):
        if self.shutdown:
            return
        try:
            msg = Bool()
            msg.data = speaking
            self.speaking_pub.publish(msg)
        except Exception:
            pass

    # ── Cleanup ───────────────────────────────────────────────────────────────────

    def cleanup(self):
        """Stop playback, drain the queue, and join the playback thread."""
        try:
            self.get_logger().info(f"{self.node_name}: shutting down…")
        except Exception:
            print("[text_to_speech] shutting down…")
        self.shutdown = True

        # Signal any in-progress playback loop to stop
        with self.state_lock:
            self.stop_requested = True

        # Stop local audio player if active
        if self.engine == "kokoro_local" and self.audio_player is not None:
            self.audio_player.stop()

        # Cancel in-flight play_audio action goal
        if self.engine == "kokoro_pepper" and self.play_goal_handle is not None:
            self.play_goal_handle.cancel_goal_async()

        # Unblock the playback thread (None is the exit sentinel)
        self.sentence_queue.put(None)

        # Wait up to 2 s for the playback thread to exit cleanly
        self._playback_thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        config = load_configuration()
        node = TextToSpeechNode(config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()