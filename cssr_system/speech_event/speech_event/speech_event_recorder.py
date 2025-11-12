#!/usr/bin/env python3
"""
ROS2 Audio Recorder for naoqi_bridge_msgs/AudioBuffer

- Subscribes to a topic publishing naoqi_bridge_msgs/msg/AudioBuffer
- Writes interleaved PCM to a multichannel .wav
- Optional: also writes one mono .wav per channel

Params (ros2 params):
  mic_topic     (string) : topic name (default: "/pepper_robot/audio")
  output_base   (string) : file path prefix without extension (default: "./pepper_audio")
  max_seconds   (int)    : stop after N seconds (0 = run until Ctrl-C)
  split_channels(bool)   : also save per-channel WAVs (default: false)
"""

import os
import wave
import signal
import numpy as np
import rclpy
from rclpy.node import Node
from naoqi_bridge_msgs.msg import AudioBuffer

CHANNEL_LABELS = {
    AudioBuffer.CHANNEL_FRONT_LEFT:   "front_left",
    AudioBuffer.CHANNEL_FRONT_CENTER: "front_center",
    AudioBuffer.CHANNEL_FRONT_RIGHT:  "front_right",
    AudioBuffer.CHANNEL_REAR_LEFT:    "rear_left",
    AudioBuffer.CHANNEL_REAR_CENTER:  "rear_center",
    AudioBuffer.CHANNEL_REAR_RIGHT:   "rear_right",
    AudioBuffer.CHANNEL_SURROUND_LEFT:  "surround_left",
    AudioBuffer.CHANNEL_SURROUND_RIGHT: "surround_right",
    AudioBuffer.CHANNEL_SUBWOOFER:    "subwoofer",
    AudioBuffer.CHANNEL_LFE:          "lfe",
}

class AudioRecorderNode(Node):
    def __init__(self):
        super().__init__('audio_recorder')

        # Declare params
        self.declare_parameter('mic_topic', '/pepper_robot/audio')
        self.declare_parameter('output_base', './pepper_audio')
        self.declare_parameter('max_seconds', 0)
        self.declare_parameter('split_channels', False)

        self.mic_topic = self.get_parameter('mic_topic').get_parameter_value().string_value
        self.output_base = self.get_parameter('output_base').get_parameter_value().string_value
        self.max_seconds = int(self.get_parameter('max_seconds').get_parameter_value().integer_value)
        self.split_channels = bool(self.get_parameter('split_channels').get_parameter_value().bool_value)

        # State
        self.wave_main = None           # wave.Wave_write handle (multi-channel)
        self.wave_split = []            # list[wave.Wave_write] for mono files
        self.freq = None                # sample rate
        self.channels = None            # number of channels
        self.frames_written = 0         # frames (per channel) written
        self.shutting_down = False

        # Subscribe
        self.sub = self.create_subscription(
            AudioBuffer, self.mic_topic, self.on_audio, 10
        )
        self.get_logger().info(f"Recording from: {self.mic_topic}")
        self.get_logger().info(f"Saving to base: {self.output_base}  split_channels={self.split_channels}  max_seconds={self.max_seconds or '∞'}")

        # SIGINT-friendly
        signal.signal(signal.SIGINT, self._sigint)

    # ---------- helpers ----------

    def _open_main(self, freq: int, channels: int, channel_map):
        """Open the multichannel WAV file."""
        base = self.output_base
        path = f"{base}_{freq}Hz_{channels}ch.wav"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        wf = wave.open(path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(2)   # int16
        wf.setframerate(freq)
        self.get_logger().info(f"Opened main WAV: {path}")

        # Log channel order
        labels = [CHANNEL_LABELS.get(c, f"ch{c}") for c in channel_map]
        self.get_logger().info(f"Channel order (from channel_map): {labels}")
        return wf

    def _open_splits(self, freq: int, channel_map):
        """Open per-channel WAVs if requested."""
        split_handles = []
        for i, ch_enum in enumerate(channel_map):
            label = CHANNEL_LABELS.get(ch_enum, f"ch{ch_enum}")
            path = f"{self.output_base}_{label}_{freq}Hz.wav"
            wf = wave.open(path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(freq)
            split_handles.append(wf)
            self.get_logger().info(f"Opened split WAV: {path}")
        return split_handles

    def _close_all(self):
        if self.wave_main is not None:
            try: self.wave_main.close()
            except Exception: pass
            self.wave_main = None

        for wf in self.wave_split:
            try: wf.close()
            except Exception: pass
        self.wave_split = []

    def _sigint(self, *_):
        self.get_logger().info("Ctrl-C received. Stopping recording...")
        self.shutting_down = True
        self._close_all()
        rclpy.shutdown()

    # ---------- callback ----------
    def on_audio(self, msg: AudioBuffer):
        if self.shutting_down:
            return

        try:
            # Metadata
            freq_in = int(msg.frequency)
            channel_map = list(msg.channel_map)
            channels = len(channel_map)
            data_i16 = np.asarray(msg.data, dtype=np.int16)
            if channels == 0 or data_i16.size == 0:
                return

            # Lazy open on first packet
            if self.wave_main is None:
                self.freq = freq_in
                self.channels = channels
                self.wave_main = self._open_main(freq_in, channels, channel_map)
                if self.split_channels:
                    self.wave_split = self._open_splits(freq_in, channel_map)

            # Warn if stream format changes mid-flight
            if freq_in != self.freq or channels != self.channels:
                self.get_logger().warning(
                    f"Stream format changed: {self.freq}Hz/{self.channels}ch -> {freq_in}Hz/{channels}ch"
                )

            # Frames = samples per channel
            num_frames = data_i16.size // channels
            if num_frames <= 0:
                return

            # Write interleaved bytes directly to main WAV
            self.wave_main.writeframes(data_i16[:num_frames*channels].tobytes())

            # Optionally split per channel
            if self.split_channels:
                frames = data_i16[:num_frames*channels].reshape(num_frames, channels)
                for i, wf in enumerate(self.wave_split):
                    wf.writeframes(frames[:, i].astype(np.int16).tobytes())

            # Count frames for duration
            self.frames_written += num_frames
            if self.max_seconds > 0:
                seconds = self.frames_written / float(self.freq)
                if seconds >= self.max_seconds:
                    self.get_logger().info(f"Reached {self.max_seconds}s. Stopping.")
                    self._close_all()
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Recorder error: {e}")

def main():
    rclpy.init()
    node = AudioRecorderNode()
    try:
        rclpy.spin(node)
    finally:
        node._close_all()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
