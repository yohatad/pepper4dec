#!/usr/bin/env python3
"""
test_play_audio.py
Standalone test for the kokoro_pepper audio playback path.

Exercises three functions from text_to_speech_implementation:
  synthesize_kokoro  — Kokoro-82M TTS synthesis
  audio_to_wav_bytes — float32 array → PCM-16 WAV bytes
  AudioPlayer        — local sounddevice playback (sanity check only)

Then drives the full kokoro_pepper pipeline in one of two modes:

  file   (default):
    /naoqi_driver/load_audio_file  (service) — send WAV bytes to naoqi_driver,
                                               which SCPs them to the robot and
                                               calls ALAudioPlayer.loadFile()
    /naoqi_driver/play_audio       (action)  — play via ALAudioPlayer with feedback
    /naoqi_driver/unload_audio_file (service) — release the file

  stream:
    /naoqi_driver/send_audio_buffer (service) — stream raw PCM chunks directly
                                                to ALAudioDevice (no SCP needed)

Usage:
  # From your ROS2 workspace (source install/setup.bash first):
  python3 tests/test_play_audio.py
  python3 tests/test_play_audio.py "Good morning, I am Pepper."
  python3 tests/test_play_audio.py "Hello." --local        # laptop speakers only
  python3 tests/test_play_audio.py "Hello." --method stream # streaming mode
  python3 tests/test_play_audio.py "Hello." --voice af_heart --rate 22050

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from naoqi_bridge_msgs.action import PlayAudio
from naoqi_bridge_msgs.srv import LoadAudioFile, UnloadAudioFile, SendAudioBuffer

# ── Pull helpers from the package under test ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_to_speech.text_to_speech_implementation import (
    AudioPlayer,
    audio_to_wav_bytes,
    synthesize_kokoro,
)

# ---------------------------------------------------------------------------
# Default config (mirrors text_to_speech_configuration.yaml)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "voice": "af_bella",
    "sample_rate": 24000,
    "output_device": -1,
}

DEFAULT_TEXT = "Hello, I am Pepper. The play audio test is working correctly."

# Robot audio settings for streaming
ROBOT_SAMPLE_RATE = 48000
STREAM_CHUNK_FRAMES = 16384          # max frames per send_audio_buffer call
STREAM_CHUNK_BYTES  = STREAM_CHUNK_FRAMES * 4  # stereo 16-bit = 4 bytes/frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separator(label: str = "") -> None:
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print("─" * width)


# ---------------------------------------------------------------------------
# Step 1 – synthesis + local playback
# ---------------------------------------------------------------------------

def test_synthesis_and_local_playback(text: str, config: dict) -> np.ndarray:
    """
    Synthesise *text* with Kokoro and play it locally via sounddevice.

    Returns the raw float32 audio array so it can be reused for the robot
    playback test without synthesising twice.
    """
    _separator("1 · Kokoro synthesis")
    print(f"  text   : {text!r}")
    print(f"  voice  : {config['voice']}")
    print(f"  rate   : {config['sample_rate']} Hz")

    t0 = time.monotonic()
    audio = synthesize_kokoro(text, config["voice"], config["sample_rate"])
    elapsed = time.monotonic() - t0

    duration_s = len(audio) / config["sample_rate"]
    print(f"  samples: {len(audio)} ({duration_s:.2f}s audio in {elapsed:.2f}s)")
    assert len(audio) > 0, "synthesize_kokoro returned empty audio"

    _separator("2 · Local playback (sounddevice)")
    player = AudioPlayer(
        sample_rate=config["sample_rate"],
        output_device=config["output_device"],
    )
    completed = player.play(audio)
    print(f"  completed: {completed}")
    assert completed, "AudioPlayer.play() was interrupted unexpectedly"

    return audio


# ---------------------------------------------------------------------------
# Step 2a – file mode: load_audio_file + play_audio action
# ---------------------------------------------------------------------------

class FilePlayTester(Node):
    """
    Tests the 'file' playback path:
      load_audio_file (bytes) → naoqi_driver SCPs to robot → ALAudioPlayer.loadFile
      play_audio action        → ALAudioPlayer.play with feedback
      unload_audio_file        → ALAudioPlayer.unloadFile
    """

    SERVICE_TIMEOUT_S = 10.0
    LOAD_TIMEOUT_S    = 30.0   # SCP can take a few seconds
    PLAY_TIMEOUT_S    = 60.0

    def __init__(self):
        super().__init__("test_play_audio_file")
        self._load_client   = self.create_client(LoadAudioFile,   "/naoqi_driver/load_audio_file")
        self._unload_client = self.create_client(UnloadAudioFile, "/naoqi_driver/unload_audio_file")
        self._play_client   = ActionClient(self, PlayAudio,       "/naoqi_driver/play_audio")

    def wait_for_services(self) -> bool:
        self.get_logger().info("Waiting for naoqi_driver services…")
        for client, name in [
            (self._load_client,   "/naoqi_driver/load_audio_file"),
            (self._unload_client, "/naoqi_driver/unload_audio_file"),
        ]:
            if not client.wait_for_service(timeout_sec=self.SERVICE_TIMEOUT_S):
                self.get_logger().error(f"Service {name} not available after {self.SERVICE_TIMEOUT_S}s")
                return False

        if not self._play_client.wait_for_server(timeout_sec=self.SERVICE_TIMEOUT_S):
            self.get_logger().error(
                f"/naoqi_driver/play_audio action server not available after {self.SERVICE_TIMEOUT_S}s"
            )
            return False

        self.get_logger().info("All services ready.")
        return True

    def load_audio(self, wav_bytes: bytes) -> int:
        """Send WAV bytes to naoqi_driver (driver SCPs to robot). Returns file_id or -1."""
        req = LoadAudioFile.Request()
        req.remote_path = ""
        req.audio_data  = list(wav_bytes)

        self.get_logger().info(f"Sending {len(wav_bytes):,} bytes to load_audio_file…")
        future = self._load_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.LOAD_TIMEOUT_S)

        if not future.done():
            self.get_logger().error("load_audio_file timed out (SCP may have failed)")
            return -1

        res = future.result()
        if not res.success:
            self.get_logger().error(f"load_audio_file failed: {res.message}")
            return -1

        self.get_logger().info(f"Audio loaded — file_id={res.file_id}")
        return res.file_id

    def play_audio(self, file_id: int) -> bool:
        """Send play_audio goal and block until it completes. Returns True on success."""
        goal = PlayAudio.Goal()
        goal.file_id = file_id
        goal.volume  = 0.85
        goal.pan     = 0.0
        goal.loop    = False

        self.get_logger().info(f"Sending play_audio goal (file_id={file_id})…")
        send_future = self._play_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)

        if not send_future.done():
            self.get_logger().error("send_goal_async timed out")
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("play_audio goal was rejected")
            return False

        self.get_logger().info("Goal accepted — waiting for result…")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.PLAY_TIMEOUT_S)

        if not result_future.done():
            self.get_logger().error("play_audio result timed out")
            return False

        result = result_future.result().result
        self.get_logger().info(
            f"play_audio done — status={result.status}, played={result.played_secs:.1f}s"
        )
        return result.success

    def unload_audio(self, file_id: int) -> None:
        req = UnloadAudioFile.Request()
        req.file_id = file_id
        self._unload_client.call_async(req)
        rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info(f"Unloaded file_id={file_id}")

    def run(self, wav_bytes: bytes) -> bool:
        """Load → play → unload. Returns True on success."""
        _separator("3 · file mode: load_audio_file + play_audio action")

        if not self.wait_for_services():
            return False

        file_id = self.load_audio(wav_bytes)
        if file_id < 0:
            return False

        success = self.play_audio(file_id)
        self.unload_audio(file_id)
        return success


# ---------------------------------------------------------------------------
# Step 2b – stream mode: send_audio_buffer chunks
# ---------------------------------------------------------------------------

class StreamPlayTester(Node):
    """
    Tests the 'stream' playback path:
      Resample to 48 kHz stereo → send PCM chunks via send_audio_buffer service
      (ALAudioDevice.sendRemoteBufferToOutput — no SCP, no file_id)
    """

    SERVICE_TIMEOUT_S = 10.0

    def __init__(self):
        super().__init__("test_play_audio_stream")
        self._buffer_client = self.create_client(SendAudioBuffer, "/naoqi_driver/send_audio_buffer")

    def wait_for_services(self) -> bool:
        self.get_logger().info("Waiting for send_audio_buffer service…")
        if not self._buffer_client.wait_for_service(timeout_sec=self.SERVICE_TIMEOUT_S):
            self.get_logger().error(
                f"/naoqi_driver/send_audio_buffer not available after {self.SERVICE_TIMEOUT_S}s"
            )
            return False
        self.get_logger().info("Service ready.")
        return True

    def _resample_to_robot(self, wav_bytes: bytes) -> np.ndarray:
        """Read WAV, resample to 48 kHz stereo int16."""
        import soundfile as sf
        from scipy.signal import resample_poly

        audio, src_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        if src_rate != ROBOT_SAMPLE_RATE:
            from math import gcd
            g = gcd(src_rate, ROBOT_SAMPLE_RATE)
            audio = resample_poly(audio, ROBOT_SAMPLE_RATE // g, src_rate // g).astype(np.float32)

        # Mono → stereo, float32 → int16
        stereo = np.column_stack([audio, audio])
        return (stereo * 32767).clip(-32768, 32767).astype(np.int16).flatten()

    def _send_chunk(self, chunk: np.ndarray) -> bool:
        req = SendAudioBuffer.Request()
        req.audio_data = chunk.tobytes()
        future = self._buffer_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done():
            self.get_logger().error("send_audio_buffer timed out")
            return False
        res = future.result()
        if not res.success:
            self.get_logger().warn(f"send_audio_buffer failed: {res.message}")
        return res.success

    def run(self, wav_bytes: bytes) -> bool:
        """Resample → stream chunks to robot. Returns True on success."""
        _separator("3 · stream mode: send_audio_buffer chunks")

        if not self.wait_for_services():
            return False

        self.get_logger().info("Resampling to 48 kHz stereo…")
        pcm = self._resample_to_robot(wav_bytes)

        total_frames    = len(pcm) // 2          # stereo → frames
        duration_s      = total_frames / ROBOT_SAMPLE_RATE
        bytes_per_sec   = ROBOT_SAMPLE_RATE * 4  # 2ch * 2B
        self.get_logger().info(
            f"Streaming {len(pcm) * 2:,} bytes ({duration_s:.2f}s) in "
            f"{(len(pcm) * 2 + STREAM_CHUNK_BYTES - 1) // STREAM_CHUNK_BYTES} chunks…"
        )

        offset  = 0
        success = True
        while offset < len(pcm):
            chunk = pcm[offset: offset + STREAM_CHUNK_FRAMES * 2]  # *2 for stereo samples
            if not self._send_chunk(chunk):
                success = False
                break

            chunk_duration = (len(chunk) // 2) / ROBOT_SAMPLE_RATE
            time.sleep(max(0.01, chunk_duration - 0.01))
            offset += STREAM_CHUNK_FRAMES * 2

        if success:
            self.get_logger().info(f"Stream complete ({duration_s:.2f}s)")
        return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the kokoro_pepper audio playback pipeline"
    )
    parser.add_argument(
        "text",
        nargs="?",
        default=DEFAULT_TEXT,
        help="Text to synthesise and play (default: built-in test phrase)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run synthesis + local sounddevice playback only (skip robot steps)",
    )
    parser.add_argument(
        "--method",
        choices=["file", "stream"],
        default="file",
        help="Robot playback method: 'file' (SCP + action, default) or 'stream' (PCM chunks)",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_CONFIG["voice"],
        help=f"Kokoro voice (default: {DEFAULT_CONFIG['voice']})",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_CONFIG["sample_rate"],
        help=f"Sample rate in Hz (default: {DEFAULT_CONFIG['sample_rate']})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "voice": args.voice,
        "sample_rate": args.rate,
        "output_device": DEFAULT_CONFIG["output_device"],
    }

    _separator("text_to_speech play_audio test")
    print(f"  text  : {args.text!r}")
    print(f"  local : {args.local}")
    print(f"  method: {args.method}")

    # ── Step 1: synthesis + local playback ───────────────────────────────────
    try:
        audio = test_synthesis_and_local_playback(args.text, config)
    except Exception as exc:
        print(f"\n[FAIL] Synthesis / local playback: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.local:
        _separator("PASS (local only)")
        sys.exit(0)

    # ── Step 2: robot playback ───────────────────────────────────────────────
    _separator("Encoding WAV bytes")
    wav_bytes = audio_to_wav_bytes(audio, config["sample_rate"])
    print(f"  WAV size: {len(wav_bytes):,} bytes")

    rclpy.init()
    if args.method == "stream":
        node = StreamPlayTester()
    else:
        node = FilePlayTester()

    try:
        ok = node.run(wav_bytes)
    except KeyboardInterrupt:
        ok = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if ok:
        _separator("PASS")
        sys.exit(0)
    else:
        _separator("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
