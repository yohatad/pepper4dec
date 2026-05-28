"""
text_to_speech_implementation.py Implementation code for Text-to-Speech synthesis 
and local audio playback.

Three synthesis / playback backends are supported (configured via
text_to_speech_configuration.yaml):

  naoqi_ros      — Publishes plain text to a ROS2 topic consumed by
                   naoqi_bridge.  Pepper's on-board ALTextToSpeech handles
                   synthesis and plays through the robot's built-in speakers.
                   No extra Python dependencies.  Recommended for robot
                   deployments.

  kokoro_local   — Kokoro-82M neural TTS synthesised on this machine, played
                   through sounddevice.  Zero network calls, ~100 ms TTFA.
                   Useful for laptop testing without the robot.
                   Requires:  pip install kokoro soundfile sounddevice

  kokoro_pepper  — Kokoro-82M synthesis on this machine.  Raw WAV bytes are
                   sent to naoqi_driver via the load_audio_file service (no
                   SCP/SSH).  naoqi_driver writes the bytes to the robot via
                   ALFileManager and plays through Pepper's speakers via the
                   play_audio action.
                   Requires:  pip install kokoro soundfile

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Date: April 2025
Version: v1.0
"""

import io
import math
import os
import queue
import re
import threading
from typing import Generator, Iterator, List, Optional, Tuple

import numpy as np
import rclpy
import rclpy.logging
from rclpy.action import ActionServer, ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool

from dec_interfaces.action import TTS
from naoqi_bridge_msgs.action import PlayAudio
from naoqi_bridge_msgs.srv import LoadAudioFile, UnloadAudioFile, SendAudioBuffer

logger = rclpy.logging.get_logger("text_to_speech")

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list:
    """Split *text* into individual sentences at .!? boundaries."""
    parts = SENTENCE_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Duration estimation  (naoqi_ros backend)
# ---------------------------------------------------------------------------

def estimate_duration(text: str, chars_per_second: float, speech_padding_s: float) -> float:
    """Estimate how long Pepper will take to speak *text* (seconds)."""
    return max(1.0, len(text) / chars_per_second) + speech_padding_s


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_configuration() -> dict:
    """
    Load configuration from the default YAML file location.

    Returns:
        dict: Configuration with defaults applied for any missing keys.
    """
    from ament_index_python.packages import get_package_share_directory
    import yaml

    config = {
        # Backend selection
        'engine': 'naoqi_ros',

        # naoqi_ros backend
        'naoqi_speech_topic': '/speech',
        'chars_per_second': 12.0,
        'speech_padding_s': 0.5,

        # kokoro backends (kokoro_local / kokoro_pepper)
        'voice': 'af_bella',
        'sample_rate': 24000,
        'output_device': -1,          # kokoro_local only; -1 = system default

        # kokoro_pepper playback method: "file" or "stream"
        #   "file"   - naoqi_driver SCPs WAV to robot, plays via ALAudioPlayer (full feedback)
        #   "stream" - send_audio_buffer / ALAudioDevice (no SCP needed)
        'playback_method': 'stream',

        # Stream volume multiplier (kokoro_pepper / elevenlabs_pepper, stream mode)
        # 1.0 = unity gain; 2.0 = double amplitude (clips at ±32767)
        'stream_volume': 1.0,

        # Barge-in detection (all backends)
        'barge_in_threshold': 0.85,
        'barge_in_chunks': 3,

        # ElevenLabs backend
        'elevenlabs_api_key': '',
        'elevenlabs_voice_id': '21m00Tcm4TlvDq8ikWAM',  # Rachel
        'elevenlabs_model': 'eleven_turbo_v2_5',
        'elevenlabs_stability': 0.5,
        'elevenlabs_similarity_boost': 0.75,
        'elevenlabs_style': 0.0,
        'elevenlabs_speed': 1.0,
    }

    try:
        package_path = get_package_share_directory("text_to_speech")
        config_file = os.path.join(
            package_path, "config", "text_to_speech_configuration.yaml"
        )
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                file_config = yaml.safe_load(f) or {}
            config.update(file_config)
            logger.info(f"Loaded TTS configuration from {config_file}")
        else:
            logger.warn(
                f"Configuration file not found at {config_file}, using defaults"
            )
    except Exception as e:
        logger.error(f"Error reading configuration: {e} — using defaults")

    # Allow overriding the ElevenLabs key via environment variable
    env_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if env_key:
        config["elevenlabs_api_key"] = env_key

    return config


# ---------------------------------------------------------------------------
# Kokoro-82M synthesis
# ---------------------------------------------------------------------------

kokoro_pipeline = None
kokoro_pipeline_lock = threading.Lock()


def getkokoro_pipeline():
    """Return (and lazily load) the shared KPipeline instance."""
    global kokoro_pipeline
    if kokoro_pipeline is not None:
        return kokoro_pipeline
    with kokoro_pipeline_lock:
        if kokoro_pipeline is None:
            try:
                from kokoro import KPipeline  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Kokoro-82M is not installed.\n"
                    "  pip install kokoro soundfile sounddevice\n"
                    "  sudo apt-get install espeak-ng"
                ) from exc
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Kokoro-82M model on {device} (first call only)…")
            try:
                kokoro_pipeline = KPipeline(lang_code="a", device=device)
            except Exception as exc:
                if device == "cuda" and (
                    "cuda" in str(exc).lower() or "cudnn" in str(exc).lower()
                ):
                    logger.warn(f"CUDA failed ({exc}); falling back to CPU.")
                    kokoro_pipeline = KPipeline(lang_code="a", device="cpu")
                else:
                    raise
            logger.info(f"Kokoro-82M loaded on {device}.")
    return kokoro_pipeline


def synthesize_kokoro(text: str, voice: str, sample_rate: int) -> np.ndarray:
    """
    Synthesise *text* with Kokoro-82M.

    Returns a float32 numpy array at *sample_rate* Hz.
    The model is loaded once and cached for the process lifetime.
    """
    pipeline = getkokoro_pipeline()

    chunks = []
    for _, _, audio_chunk in pipeline(text, voice=voice, speed=1.0):
        if audio_chunk is not None:
            chunks.append(np.asarray(audio_chunk, dtype=np.float32))

    if not chunks:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(chunks)

    kokoro_native_rate = 24_000
    if kokoro_native_rate != sample_rate:
        from scipy.signal import resample_poly
        gcd = math.gcd(kokoro_native_rate, sample_rate)
        audio = resample_poly(
            audio, sample_rate // gcd, kokoro_native_rate // gcd
        ).astype(np.float32)

    return audio


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode a float32 numpy array as PCM-16 WAV and return the raw bytes."""
    import soundfile as sf  # type: ignore
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Robot audio helpers  (kokoro_pepper / elevenlabs_pepper backends)
# ---------------------------------------------------------------------------

ROBOT_RATE = 48_000
ROBOT_CHUNK_FRAMES = 16_384  # driver hard cap


def prepare_stream_audio(
    wav_bytes: bytes,
    stream_volume: float = 1.0,
) -> Iterator[Tuple[List[int], float]]:
    """
    Decode WAV bytes, resample to ROBOT_RATE stereo int16, and yield
    ``(audio_list, wait_time)`` pairs ready for send_audio_buffer.

    ``audio_list`` is a ``List[int]`` of raw bytes (as unsigned ints).
    ``wait_time``  is how long to sleep before sending the next chunk.
    """
    import soundfile as sf  # type: ignore
    from scipy.signal import resample_poly

    audio, src_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    if src_rate != ROBOT_RATE:
        _g = math.gcd(src_rate, ROBOT_RATE)
        audio = resample_poly(audio, ROBOT_RATE // _g, src_rate // _g).astype(np.float32)

    stereo = (
        np.column_stack([audio, audio]).flatten() * 32767 * stream_volume
    ).clip(-32768, 32767).astype(np.int16)

    max_chunk = ROBOT_CHUNK_FRAMES * 2          # stereo int16 samples per send
    bytes_per_second = ROBOT_RATE * 2 * 2       # 48 kHz × 2 ch × 2 B

    offset = 0
    while offset < len(stereo):
        chunk = stereo[offset : offset + max_chunk]
        wait_time = max(0.01, chunk.nbytes / bytes_per_second - 0.01)
        yield list(chunk.tobytes()), wait_time
        offset += max_chunk


def iter_robot_chunks(
    gen,
    api_rate: int,
    stream_volume: float = 1.0,
) -> Iterator[Tuple[List[int], float]]:
    """
    Accumulate streaming float32 chunks (at *api_rate*), resample to
    ROBOT_RATE stereo int16 on aligned ROBOT_CHUNK_FRAMES boundaries, and
    yield ``(audio_list, wait_time)`` pairs ready for send_audio_buffer.

    Resampling on aligned boundaries avoids click artifacts that occur when
    each tiny API chunk is resampled independently.
    """
    from scipy.signal import resample_poly

    _g = math.gcd(api_rate, ROBOT_RATE)
    src_frames_per_chunk = ROBOT_CHUNK_FRAMES * api_rate // ROBOT_RATE
    up, down = ROBOT_RATE // _g, api_rate // _g

    def _flush(src_buf: np.ndarray) -> Tuple[List[int], float]:
        resampled = resample_poly(src_buf, up, down).astype(np.float32)
        stereo = (
            np.column_stack([resampled, resampled]).flatten() * 32767 * stream_volume
        ).clip(-32768, 32767).astype(np.int16)
        duration = len(stereo) // 2 / ROBOT_RATE
        return list(stereo.tobytes()), max(0.01, duration - 0.01)

    buf = np.zeros(0, dtype=np.float32)
    for chunk in gen:
        buf = np.concatenate([buf, chunk])
        while len(buf) >= src_frames_per_chunk:
            yield _flush(buf[:src_frames_per_chunk])
            buf = buf[src_frames_per_chunk:]

    if len(buf) > 0:
        yield _flush(buf)


def collect_and_resample(gen, src_rate: int, target_rate: int) -> np.ndarray:
    """Drain *gen*, concatenate chunks, and resample to *target_rate*."""
    from scipy.signal import resample_poly

    chunks = list(gen)
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(chunks)
    if src_rate != target_rate:
        _g = math.gcd(src_rate, target_rate)
        audio = resample_poly(
            audio, target_rate // _g, src_rate // _g
        ).astype(np.float32)

    return audio


def resample_chunks(gen, src_rate: int, target_rate: int) -> Generator[np.ndarray, None, None]:
    """Yield resampled float32 chunks from *gen* (no-op if rates match)."""
    if src_rate == target_rate:
        yield from gen
        return

    from scipy.signal import resample_poly

    _g = math.gcd(src_rate, target_rate)
    _up, _down = target_rate // _g, src_rate // _g
    for chunk in gen:
        yield resample_poly(chunk, _up, _down).astype(np.float32)


# ---------------------------------------------------------------------------
# ElevenLabs synthesis  (streaming + batch)
# ---------------------------------------------------------------------------

# Supported PCM output rates by ElevenLabs API
ELEVENLABS_PCM_RATES = {16000, 22050, 24000, 44100}


def elevenlabs_client(api_key: str):
    try:
        from elevenlabs.client import ElevenLabs  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ElevenLabs SDK is not installed.\n"
            "  pip install elevenlabs"
        ) from exc
    return ElevenLabs(api_key=api_key)


def stream_elevenlabs(
    text: str,
    voice_id: str,
    api_key: str,
    model_id: str = "eleven_turbo_v2_5",
    sample_rate: int = 24000,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    speed: float = 1.0,
):
    """
    Stream audio from ElevenLabs as an iterator of float32 numpy chunks.

    Returns ``(generator, api_rate)``.  Each yielded chunk is a float32 array
    at *api_rate* Hz (the nearest PCM rate the API supports to *sample_rate*).
    The caller is responsible for resampling if api_rate ≠ the target rate.

    Audio starts arriving within ~200 ms (TTFA = first API chunk).

    Requires:  pip install elevenlabs
    """
    from elevenlabs import VoiceSettings
    api_rate = min(ELEVENLABS_PCM_RATES, key=lambda r: abs(r - sample_rate))
    api_rate = min(ELEVENLABS_PCM_RATES, key=lambda r: abs(r - sample_rate))

    def _gen():
        client = elevenlabs_client(api_key)
        logger.info(
            f"ElevenLabs stream: voice={voice_id}, model={model_id}, "
            f"format=pcm_{api_rate}, speed={speed}, stability={stability}"
        )
        for raw in client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            output_format=f"pcm_{api_rate}",
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                speed=speed,
            ),
        ):
            if raw:
                yield np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    return _gen(), api_rate


def synthesize_elevenlabs(
    text: str,
    voice_id: str,
    api_key: str,
    sample_rate: int,
    model_id: str = "eleven_turbo_v2_5",
) -> np.ndarray:
    """
    Batch-synthesise *text* via the ElevenLabs API (collects full audio).

    Returns a float32 numpy array at *sample_rate* Hz.
    For lower latency use stream_elevenlabs() instead.

    Requires:  pip install elevenlabs
    """
    gen, api_rate = stream_elevenlabs(
        text, voice_id, api_key, model_id=model_id, sample_rate=sample_rate
    )
    chunks = list(gen)
    if not chunks:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(chunks)

    if api_rate != sample_rate:
        from scipy.signal import resample_poly
        g = math.gcd(api_rate, sample_rate)
        audio = resample_poly(
            audio, sample_rate // g, api_rate // g
        ).astype(np.float32)

    return audio


# ---------------------------------------------------------------------------
# Local audio playback via sounddevice  (kokoro_local backend)
# ---------------------------------------------------------------------------

class AudioPlayer:
    """
    Plays a float32 audio array through sounddevice in 50 ms chunks
    so that stop() interrupts within ~50 ms from any thread.
    """

    CHUNK_DURATION_S = 0.05

    def __init__(self, sample_rate: int, output_device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.output_device = (
            None if (output_device is None or output_device < 0) else output_device
        )
        self._stop_event = threading.Event()

    def play(self, audio: np.ndarray) -> bool:
        """
        Play *audio* synchronously.

        Returns True on natural completion, False if interrupted by stop().
        """
        if len(audio) == 0:
            return True

        self._stop_event.clear()
        chunk_size = int(self.sample_rate * self.CHUNK_DURATION_S)

        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed.  Run:  pip install sounddevice"
            ) from exc

        device_kw = {} if self.output_device is None else {"device": self.output_device}
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                **device_kw,
            ) as stream:
                pos = 0
                while pos < len(audio):
                    if self._stop_event.is_set():
                        return False
                    chunk = audio[pos : pos + chunk_size].reshape(-1, 1)
                    stream.write(chunk)
                    pos += chunk_size
        except Exception as exc:
            logger.error(f"AudioPlayer error: {exc}")
            return False

        return True

    def play_chunks(self, chunks) -> bool:
        """
        Stream an iterable of float32 arrays through one OutputStream.

        Audio starts playing as soon as the first chunk arrives.
        Returns True on natural completion, False if interrupted by stop().
        """
        self._stop_event.clear()

        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed.  Run:  pip install sounddevice"
            ) from exc

        device_kw = {} if self.output_device is None else {"device": self.output_device}
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                **device_kw,
            ) as stream:
                for chunk in chunks:
                    if self._stop_event.is_set():
                        return False
                    if len(chunk) > 0:
                        stream.write(chunk.reshape(-1, 1))
        except Exception as exc:
            logger.error(f"AudioPlayer.play_chunks error: {exc}")
            return False

        return True

    def stop(self):
        """Interrupt playback at the next chunk boundary (~50 ms)."""
        self._stop_event.set()


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class TextToSpeechNode(Node):
    """
    ROS2 node that drives Pepper's voice.

    Sentences arrive either via:
      (a) /conversation_manager/response_stream  — streaming from LLM
      (b) /tts action goal                       — programmatic calls

    Both routes enqueue sentences into a thread-safe queue that is drained by
    a single background playback thread, ensuring strict ordering and clean
    barge-in handling.
    """

    def __init__(self, config: dict):
        super().__init__("text_to_speech")

        self.config = config
        self.node_name = self.get_name()
        self.engine = config["engine"]

        self.get_logger().info(
            f"{self.node_name}: starting — engine={self.engine}, "
            f"barge_in_threshold={config['barge_in_threshold']}"
        )

        # ── Internal state ────────────────────────────────────────────────────
        self.sentence_queue: queue.Queue = queue.Queue()
        self.is_speaking = False
        self.barge_in_counter = 0
        self.barge_in_triggered = False
        self.state_lock = threading.Lock()

        # ── Per-backend initialisation ────────────────────────────────────────
        self.audio_player = None
        self.play_goal_handle = None
        self.play_done_event = threading.Event()

        if self.engine in ("kokoro_local", "kokoro_pepper"):
            self.warmup_kokoro()

        if self.engine == "kokoro_local":
            self.audio_player = AudioPlayer(
                sample_rate=config["sample_rate"],
                output_device=config["output_device"],
            )

        # ── Publishers ────────────────────────────────────────────────────────
        self.speaking_pub = self.create_publisher(Bool, "/text_to_speech/speaking", 10)

        if self.engine == "naoqi_ros":
            self.naoqi_pub = self.create_publisher(
                String, config["naoqi_speech_topic"], 10
            )
            self.get_logger().info(
                f"{self.node_name}: naoqi_ros backend — "
                f"publishing to '{config['naoqi_speech_topic']}'"
            )

        # ── Service clients ───────────────────────────────────────────────────
        self.mic_client = self.create_client(SetBool, "/speech_event/set_enabled")

        if self.engine == "kokoro_pepper":
            self._load_client = self.create_client(
                LoadAudioFile, "/naoqi_driver/load_audio_file"
            )
            self._unload_client = self.create_client(
                UnloadAudioFile, "/naoqi_driver/unload_audio_file"
            )
            self._play_client = ActionClient(
                self, PlayAudio, "/naoqi_driver/play_audio"
            )
            # Streaming service (for playback_method="stream")
            self._send_buffer_client = self.create_client(
                SendAudioBuffer, "/naoqi_driver/send_audio_buffer"
            )

        # ── Subscriptions ─────────────────────────────────────────────────────
        stream_cb_group = MutuallyExclusiveCallbackGroup()
        vad_cb_group    = MutuallyExclusiveCallbackGroup()

        self.create_subscription(String, "/conversation_manager/response_stream", self.stream_sentence_callback,
            10, callback_group=stream_cb_group,)
        
        self.create_subscription(Float32, "/speech_event/vad_speech_prob", self.vad_prob_callback,
            10,callback_group=vad_cb_group,)

        # ── TTS action server ─────────────────────────────────────────────────
        action_cb_group = MutuallyExclusiveCallbackGroup()
        self._tts_action = ActionServer(self, TTS, "/tts", self.execute_tts_action,
            callback_group=action_cb_group,)

        # ── Background playback thread ─────────────────────────────────────────
        self._playback_thread = threading.Thread(
            target=self.playback_loop, name="tts_playback", daemon=True
        )
        self._playback_thread.start()

        self.get_logger().info(f"{self.node_name}: ready.")

    # ── Kokoro warm-up ──────────────────────────────────────────────────────────

    def warmup_kokoro(self):
        """Synthesise a short dummy phrase to pre-load the Kokoro model."""
        self.get_logger().info(f"{self.node_name}: warming up Kokoro model…")
        try:
            synthesize_kokoro("Hello.", self.config["voice"], self.config["sample_rate"])
            self.get_logger().info(f"{self.node_name}: Kokoro warm-up complete.")
        except Exception as e:
            self.get_logger().warn(
                f"{self.node_name}: Kokoro warm-up failed (non-critical): {e}")

    # ── Stream sentence callback ─────────────────────────────────────────────────

    def stream_sentence_callback(self, msg: String):
        """Enqueue each sentence published by conversation_manager."""
        sentence = msg.data.strip()
        if sentence:
            self.sentence_queue.put(sentence)

    # ── VAD / barge-in ──────────────────────────────────────────────────────────

    def vad_prob_callback(self, msg: Float32):
        """Detect user speech during playback and trigger barge-in."""
        with self.state_lock:
            if not self.is_speaking:
                self.barge_in_counter = 0
                return

            if msg.data >= self.config["barge_in_threshold"]:
                self.barge_in_counter += 1
            else:
                self.barge_in_counter = 0

            if self.barge_in_counter >= self.config["barge_in_chunks"]:
                if not self.barge_in_triggered:
                    self.barge_in_triggered = True
                    self.get_logger().info(
                        f"{self.node_name}: barge-in detected "
                        f"(VAD={msg.data:.2f} × {self.barge_in_counter}). "
                        "Interrupting playback."
                    )
                    self.interrupt_playback()
                    while not self.sentence_queue.empty():
                        try:
                            self.sentence_queue.get_nowait()
                        except queue.Empty:
                            break

    def interrupt_playback(self):
        """Stop whichever backend is currently playing."""
        if self.engine == "kokoro_local" and self.audio_player is not None:
            self.audio_player.stop()
        elif self.engine == "kokoro_pepper" and self.play_goal_handle is not None:
            self.play_goal_handle.cancel_goal_async()

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
          callable  — sentinel signalling that a /tts action goal is done
        """
        while True:
            try:
                item = self.sentence_queue.get(timeout=0.1)
            except queue.Empty:
                continue

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
            self.barge_in_triggered = False
            self.barge_in_counter = 0
            self.is_speaking = True

        self.publish_speaking(True)

        try:
            if self.engine == "naoqi_ros":
                self.speak_naoqi(sentence)
            elif self.engine == "kokoro_local":
                self.speak_kokoro_local(sentence)
            elif self.engine == "kokoro_pepper":
                self.speak_kokoro_pepper(sentence)
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
        feedback.  Barge-in via VAD still works.
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
                if self.barge_in_triggered:
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

    def play_via_file(self, wav_bytes: bytes):
        """Play audio via load_audio_file + play_audio action."""
        # Load on Pepper by sending bytes through naoqi_driver
        file_id = self.load_audio_file_from_bytes(wav_bytes)
        if file_id < 0:
            return

        # Play via action
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

        # Unload
        self.unload_audio_file(file_id)

    def play_via_stream(self, wav_bytes: bytes):
        """Play audio via send_audio_buffer service (low-latency streaming)."""
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        
        # Robot expects 48000 Hz stereo 16-bit PCM
        ROBOT_SAMPLE_RATE = 48000
        
        # Read audio from WAV bytes (mono)
        audio, kokoro_rate = sf.read(io.BytesIO(wav_bytes), dtype='float32')
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Take first channel if stereo
        
        # Resample from Kokoro rate (24000) to robot rate (48000)
        if kokoro_rate != ROBOT_SAMPLE_RATE:
            gcd = np.gcd(kokoro_rate, ROBOT_SAMPLE_RATE)
            audio = resample_poly(
                audio, ROBOT_SAMPLE_RATE // gcd, kokoro_rate // gcd
            ).astype(np.float32)
            sample_rate = ROBOT_SAMPLE_RATE
        else:
            sample_rate = kokoro_rate
        
        # Convert to stereo by duplicating mono channel
        stereo_audio = np.column_stack([audio, audio]).flatten().astype(np.int16)
        
        # Send in chunks (max 16384 frames = 32768 bytes for stereo)
        # Each frame = 2 channels * 2 bytes = 4 bytes
        max_chunk_size = 16384 * 4  # 32768 bytes
        
        # Calculate duration of each chunk in seconds
        # sample_rate * channels * bytes_per_sample = bytes_per_second
        bytes_per_second = sample_rate * 2 * 2
        
        self.set_mic_enabled(False)
        try:
            offset = 0
            while offset < len(stereo_audio):
                # Check for barge-in
                with self.state_lock:
                    if self.barge_in_triggered:
                        break
                
                chunk = stereo_audio[offset:offset + max_chunk_size]
                if len(chunk) == 0:
                    break
                
                # Convert int16 to bytes (little-endian)
                audio_bytes = chunk.tobytes()
                # Convert to list of uint8
                audio_list = list(audio_bytes)
                    
                # Send chunk to robot
                if not self.send_audio_buffer(audio_list):
                    self.get_logger().error(
                        f"{self.node_name}: send_audio_buffer failed at offset {offset}"
                    )
                    break
                
                # Calculate how long this chunk will play
                chunk_duration = len(chunk) / bytes_per_second
                
                # Wait for the chunk to finish playing before sending the next
                # Subtract a small margin (10ms) to keep audio flowing smoothly
                wait_time = max(0.01, chunk_duration - 0.01)
                time.sleep(wait_time)
                
                offset += max_chunk_size
                
            duration = len(stereo_audio) / bytes_per_second
            self.get_logger().info(
                f"{self.node_name}: [stream] played {duration:.2f}s of audio"
            )
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

        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            return future.result().success if future.done() else False
        except Exception as e:
            self.get_logger().error(f"send_audio_buffer call failed: {e}")
            return False

    # ── Audio file loading / unloading (kokoro_pepper) ─────────────────────────

    def load_audio_file_from_bytes(self, wav_bytes: bytes) -> int:
        """Load WAV bytes onto Pepper via naoqi_driver and return file_id."""
        if not self._load_client.service_is_ready():
            self.get_logger().warn(
                f"{self.node_name}: load_audio_file service not ready")
            return -1

        req = LoadAudioFile.Request()
        req.filename = "/tmp/pepper_tts.wav"
        req.data = list(wav_bytes)

        try:
            future = self._load_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            result = future.result()
            return result.file_id if result and result.success else -1
        except Exception as e:
            self.get_logger().error(f"load_audio_file failed: {e}")
            return -1

    def unload_audio_file(self, file_id: int):
        """Unload audio file from Pepper via naoqi_driver."""
        if file_id < 0:
            return
        if not self._unload_client.service_is_ready():
            return

        req = UnloadAudioFile.Request()
        req.file_id = file_id

        try:
            future = self._unload_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        except Exception:
            pass

    def on_play_goal_response(self, future):
        """Callback for play_audio action goal response."""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn(
                    f"{self.node_name}: play_audio goal rejected"
                )
                self.play_done_event.set()
                return

            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.on_play_goal_done)
        except Exception as e:
            self.get_logger().error(f"play_audio goal failed: {e}")
            self.play_done_event.set()

    def on_play_goal_done(self, future):
        """Callback for play_audio action completion."""
        self.play_done_event.set()

    # ── Mic control ────────────────────────────────────────────────────────────

    def set_mic_enabled(self, enabled: bool):
        """Call /speech_event/set_enabled (best-effort, non-blocking)."""
        if not self.mic_client.service_is_ready():
            return
        req = SetBool.Request()
        req.data = enabled
        self.mic_client.call_async(req)

    # ── Speaking state publisher ──────────────────────────────────────────────────

    def publish_speaking(self, speaking: bool):
        msg = Bool()
        msg.data = speaking
        self.speaking_pub.publish(msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────────

    def cleanup(self):
        """Cancel any in-flight play_audio goal and wake the playback thread."""
        self.get_logger().info(f"{self.node_name}: shutting down…")
        self.sentence_queue.put(None)
        if self.engine == "kokoro_pepper" and self.play_goal_handle is not None:
            self.play_goal_handle.cancel_goal_async()


