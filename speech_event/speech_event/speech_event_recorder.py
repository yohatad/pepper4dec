#!/usr/bin/env python3

"""speech_event_recorder.py

Entry point for the audio_recorder node.
Running this module starts AudioRecorderNode, which records microphone audio
to WAV files for offline inspection or debugging.

The node subscribes to a topic publishing naoqi_bridge_msgs/msg/AudioBuffer and
writes the interleaved PCM data to a multichannel .wav file. It can optionally
also write one mono .wav file per channel. Recording stops, and the node exits,
after `max_seconds`, on Ctrl-C, or when the file reaches the 4 GB WAV size
limit (about 3 hours at 48 kHz x 4 channels, which is 1.4 GB an hour).

Files are named <output_base>_<YYYYmmdd-HHMMSS>_<Hz>Hz_<N>ch.wav, so a new
run never overwrites an earlier recording. A relative output_base is placed
under $ROS_HOME/speech_event (ROS_HOME defaults to ~/.ros), never inside the
installed package; "~" and absolute paths are used as given.

Subscribers:
    <mic_topic> (naoqi_bridge_msgs/AudioBuffer)
        Raw multi-channel microphone audio (topic name set by the
        `mic_topic` parameter).

Parameters (loaded from speech_event_configuration.yaml):
    mic_topic (string, default: "/audio")
    output_base (string, default: "pepper_audio")
    max_seconds (int, default: 3600; 0 = until Ctrl-C or the WAV size limit)
    split_channels (bool, default: false)

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: April 27, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os
import time
import wave
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from naoqi_bridge_msgs.msg import AudioBuffer

SAMPLE_WIDTH = 2  # bytes per sample (int16)

# A WAV header stores the RIFF chunk size, 36 + the data size, in 32 bits, so
# the audio data can be at most this many bytes.
WAV_MAX_DATA_BYTES = 0xFFFFFFFF - 36

CHANNEL_LABELS = {
    AudioBuffer.CHANNEL_FRONT_LEFT: "front_left",
    AudioBuffer.CHANNEL_FRONT_CENTER: "front_center",
    AudioBuffer.CHANNEL_FRONT_RIGHT: "front_right",
    AudioBuffer.CHANNEL_REAR_LEFT: "rear_left",
    AudioBuffer.CHANNEL_REAR_CENTER: "rear_center",
    AudioBuffer.CHANNEL_REAR_RIGHT: "rear_right",
    AudioBuffer.CHANNEL_SURROUND_LEFT: "surround_left",
    AudioBuffer.CHANNEL_SURROUND_RIGHT: "surround_right",
    AudioBuffer.CHANNEL_SUBWOOFER: "subwoofer",
    AudioBuffer.CHANNEL_LFE: "lfe",
}


def resolve_output_base(raw, ros_home=None):
    """Return the absolute path that output file names start with.

    Args:
        raw (str): The output_base parameter.
        ros_home (str | None): Base for relative paths; None means
            $ROS_HOME, or ~/.ros when that is unset.

    Returns:
        str: "~" expanded and absolute paths unchanged; a relative path
        placed under <ros_home>/speech_event.
    """
    path = os.path.expanduser(raw)
    if os.path.isabs(path):
        return path
    if ros_home is None:
        ros_home = os.environ.get('ROS_HOME') or os.path.join('~', '.ros')
    return os.path.join(os.path.expanduser(ros_home), 'speech_event', path)


def frame_limit(freq, channels, max_seconds):
    """Return how many frames (samples per channel) to record.

    Args:
        freq (int): Sample rate in Hz.
        channels (int): Channels in the multichannel file.
        max_seconds (int): Time limit; 0 means none.

    Returns:
        int: max_seconds worth of frames, but never more than fits in one
        WAV file.
    """
    wav_cap = WAV_MAX_DATA_BYTES // (channels * SAMPLE_WIDTH)
    if max_seconds > 0:
        return min(max_seconds * freq, wav_cap)
    return wav_cap


class AudioRecorderNode(Node):
    """Records microphone AudioBuffer messages to multichannel and per-channel WAV files."""

    def __init__(self):
        """Declare and read parameters, open the WAV files, and subscribe to
        the microphone topic.

        This is a secondary (non-orchestrated) helper node, so it uses a plain
        rclpy Node rather than the lifecycle pattern used by the package's
        primary node.
        """
        super().__init__('audio_recorder')

        # Declare params
        self.declare_parameter('mic_topic', '/audio')
        self.declare_parameter('output_base', 'pepper_audio')
        self.declare_parameter('max_seconds', 3600)
        self.declare_parameter('split_channels', False)

        self.mic_topic = self.get_parameter('mic_topic').get_parameter_value().string_value
        raw_output_base = self.get_parameter('output_base').get_parameter_value().string_value
        self.max_seconds = int(self.get_parameter(
            'max_seconds').get_parameter_value().integer_value)
        self.split_channels = bool(self.get_parameter(
            'split_channels').get_parameter_value().bool_value)

        # Outside the installed package, and stamped with the start time so a
        # new run does not overwrite the last recording.
        self.output_base = (f"{resolve_output_base(raw_output_base)}_"
                            f"{time.strftime('%Y%m%d-%H%M%S')}")

        # State
        self.wave_main = None           # wave.Wave_write handle (multi-channel)
        self.wave_split = []            # list[wave.Wave_write] for mono files
        self.freq = None                # sample rate
        self.channels = None            # number of channels
        self.frames_written = 0         # frames (per channel) written
        self.frame_limit = None         # set from the first message
        self.shutting_down = False
        self.done = False               # limit reached; main() exits

        # Subscribe
        self.sub = self.create_subscription(AudioBuffer, self.mic_topic, self.on_audio, 10)
        self.get_logger().info(f"Recording from: {self.mic_topic}")
        self.get_logger().info(
            f"Saving to base: {self.output_base}  split_channels={self.split_channels}  "
            f"max_seconds={self.max_seconds or 'until Ctrl-C (or 4 GB)'}")

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
        """Close every open WAV handle, ignoring already-closed files."""
        if self.wave_main is not None:
            try:
                self.wave_main.close()
            except Exception:
                pass
            self.wave_main = None

        for wf in self.wave_split:
            try:
                wf.close()
            except Exception:
                pass
        self.wave_split = []

    def _sigint(self, *_):
        """Stop recording and close the WAV files on Ctrl-C."""
        self.get_logger().info("Ctrl-C received. Stopping recording...")
        self.shutting_down = True
        self._close_all()

    # ---------- callback ----------
    def on_audio(self, msg: AudioBuffer):
        """Append one audio buffer to the WAV files.

        Args:
            msg (naoqi_bridge_msgs/AudioBuffer): Raw multi-channel microphone
                audio; the first message also fixes the sample rate and
                channel layout of the output files.
        """
        if self.shutting_down or self.done:
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
                self.frame_limit = frame_limit(freq_in, channels, self.max_seconds)
                self.wave_main = self._open_main(freq_in, channels, channel_map)
                if self.split_channels:
                    self.wave_split = self._open_splits(freq_in, channel_map)

            # Warn if stream format changes mid-flight
            if freq_in != self.freq or channels != self.channels:
                self.get_logger().warning(
                    f"Stream format changed: {self.freq}Hz/{self.channels}ch -> "
                    f"{freq_in}Hz/{channels}ch"
                )

            # Frames = samples per channel
            # Never past the limit: the last buffer is cut to fit.
            num_frames = min(data_i16.size // channels,
                             self.frame_limit - self.frames_written)
            if num_frames <= 0:
                return

            # Write interleaved bytes directly to main WAV
            self.wave_main.writeframes(data_i16[:num_frames * channels].tobytes())

            # Optionally split per channel
            if self.split_channels:
                frames = data_i16[:num_frames * channels].reshape(num_frames, channels)
                for i, wf in enumerate(self.wave_split):
                    wf.writeframes(frames[:, i].astype(np.int16).tobytes())

            self.frames_written += num_frames
            if self.frames_written >= self.frame_limit:
                self.get_logger().info(
                    f"Recorded {self.frames_written / float(self.freq):.0f} s, the "
                    f"limit (max_seconds or the 4 GB WAV size). Stopping.")
                self._close_all()
                self.done = True

        except Exception as e:
            self.get_logger().error(f"Recorder error: {e}")


def main():
    """Entry point for the audio_recorder ROS2 node.

    Initializes rclpy, instantiates the AudioRecorderNode, and spins it until
    Ctrl-C or until the node reaches its recording limit, closing the WAV files
    on the way out.
    """
    rclpy.init()
    node = AudioRecorderNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("Ctrl-C received. Stopping recording...")
    finally:
        node._close_all()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
