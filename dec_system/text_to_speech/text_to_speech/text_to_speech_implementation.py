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
Version: v3.0
"""

import math
import os
import re
import threading
from typing import Optional, Dict

import numpy as np
import rclpy.logging

logger = rclpy.logging.get_logger("text_to_speech")

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list:
    """Split *text* into individual sentences at .!? boundaries."""
    parts = _SENTENCE_BOUNDARY.split(text.strip())
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
        #   "file"   - load_audio_file + play_audio action (default, supports long audio)
        #   "stream" - send_audio_buffer service (low-latency, max ~170ms per call)
        'playback_method': 'file',

        # Barge-in detection (all backends)
        'barge_in_threshold': 0.85,
        'barge_in_chunks': 3,
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

    return config


# ---------------------------------------------------------------------------
# Kokoro-82M synthesis
# ---------------------------------------------------------------------------

_kokoro_pipeline = None
_kokoro_pipeline_lock = threading.Lock()


def _get_kokoro_pipeline():
    """Return (and lazily load) the shared KPipeline instance."""
    global _kokoro_pipeline
    if _kokoro_pipeline is not None:
        return _kokoro_pipeline
    with _kokoro_pipeline_lock:
        if _kokoro_pipeline is None:
            try:
                from kokoro import KPipeline  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Kokoro-82M is not installed.\n"
                    "  pip install kokoro soundfile sounddevice\n"
                    "  sudo apt-get install espeak-ng"
                ) from exc
            logger.info("Loading Kokoro-82M model (first call only)…")
            _kokoro_pipeline = KPipeline(lang_code="a")
            logger.info("Kokoro-82M loaded.")
    return _kokoro_pipeline


def synthesize_kokoro(text: str, voice: str, sample_rate: int) -> np.ndarray:
    """
    Synthesise *text* with Kokoro-82M.

    Returns a float32 numpy array at *sample_rate* Hz.
    The model is loaded once and cached for the process lifetime.
    """
    pipeline = _get_kokoro_pipeline()

    chunks = []
    for audio_chunk, _, _ in pipeline(text, voice=voice, speed=1.0):
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
    import io
    import soundfile as sf  # type: ignore
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


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

    def stop(self):
        """Interrupt playback at the next chunk boundary (~50 ms)."""
        self._stop_event.set()


