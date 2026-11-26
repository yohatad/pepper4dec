"""remote_dialogue_audio.py

Audio format conversion between Pepper and the /ws/pepper voice server.

    Pepper /audio          48 kHz, 16-bit, all microphones interleaved
    server microphone in   16 kHz, 16-bit little-endian, mono, raw PCM
    server speech out      16/24/48 kHz (as configured), 16-bit LE, mono
    Pepper speaker         48 kHz, 16-bit, stereo interleaved, at most
                           16384 frames per send_audio_buffer call

None of this module touches ROS, so it is unit-tested directly.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 26, 2026
Version: v1.0

Copyright (C) 2026 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import math

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi, resample_poly

SERVER_INPUT_RATE = 16000
ROBOT_OUTPUT_RATE = 48000
# naoqi_driver's send_audio_buffer hard cap (ALAudioDevice.sendRemoteBufferToOutput)
ROBOT_MAX_FRAMES = 16384
FRAME_BYTES = 640  # 20 ms at 16 kHz, 16-bit mono: the API's suggested chunk


class MicrophoneConverter:
    """Turn naoqi AudioBuffer contents into 16 kHz mono int16 bytes.

    Decimation is stateful (one FIR filter carried across calls), so
    consecutive buffers join without the edge clicks a per-buffer resample
    would add.
    """

    def __init__(self, channel=0, output_rate=SERVER_INPUT_RATE):
        """Pick `channel` (an AudioBuffer CHANNEL_* value) from each buffer."""
        self.channel = int(channel)
        self.output_rate = int(output_rate)
        self._input_rate = None
        self._factor = 1
        self._taps = None
        self._zi = None
        self._phase = 0

    def _setup(self, input_rate):
        """(Re)build the anti-alias filter for a new input rate."""
        if input_rate % self.output_rate != 0:
            raise ValueError(
                f'input rate {input_rate} Hz is not a multiple of {self.output_rate} Hz')
        self._input_rate = input_rate
        self._factor = input_rate // self.output_rate
        self._phase = 0
        if self._factor == 1:
            self._taps = None
            self._zi = None
            return
        # Cut-off a little below the output Nyquist frequency.
        self._taps = firwin(97, 0.45 * self.output_rate, fs=input_rate)
        self._zi = lfilter_zi(self._taps, 1.0) * 0.0

    def convert(self, data, channel_map, frequency):
        """Return one channel of an AudioBuffer as 16 kHz int16 little-endian bytes.

        Args:
            data: interleaved int16 samples (sequence or array)
            channel_map: AudioBuffer.channel_map (one entry per channel)
            frequency: AudioBuffer.frequency in Hz

        Returns:
            bytes, possibly empty.
        """
        channels = len(channel_map)
        samples = np.asarray(data, dtype=np.int16)
        if channels == 0 or samples.size < channels:
            return b''
        frequency = int(frequency)
        if frequency != self._input_rate:
            self._setup(frequency)

        frames = samples[:samples.size - samples.size % channels].reshape(-1, channels)
        channel_map = list(channel_map)
        index = channel_map.index(self.channel) if self.channel in channel_map else 0
        mono = frames[:, index].astype(np.float64)

        if self._factor > 1:
            filtered, self._zi = lfilter(self._taps, 1.0, mono, zi=self._zi)
            # Keep the decimation phase continuous across buffers.
            start = (-self._phase) % self._factor
            mono = filtered[start::self._factor]
            self._phase = (self._phase + filtered.size) % self._factor

        return np.clip(np.round(mono), -32768, 32767).astype('<i2').tobytes()


class FrameChunker:
    """Re-cut a byte stream into fixed-size frames."""

    def __init__(self, frame_bytes=FRAME_BYTES):
        """Emit frames of exactly `frame_bytes` bytes."""
        self.frame_bytes = int(frame_bytes)
        self._pending = bytearray()

    def push(self, data):
        """Add bytes and return the list of complete frames now available."""
        self._pending.extend(data)
        count = len(self._pending) // self.frame_bytes
        frames = [bytes(self._pending[i * self.frame_bytes:(i + 1) * self.frame_bytes])
                  for i in range(count)]
        del self._pending[:count * self.frame_bytes]
        return frames

    def clear(self):
        """Drop any partial frame."""
        self._pending.clear()


def mono_to_robot_stereo(pcm, input_rate):
    """Convert mono int16 LE bytes at `input_rate` to 48 kHz stereo int16 bytes.

    Args:
        pcm: mono 16-bit little-endian PCM bytes (an even number of bytes)
        input_rate: sample rate of `pcm` in Hz

    Returns:
        Interleaved stereo bytes for send_audio_buffer.
    """
    mono = np.frombuffer(pcm[:len(pcm) - len(pcm) % 2], dtype='<i2')
    if mono.size == 0:
        return b''
    if input_rate != ROBOT_OUTPUT_RATE:
        gcd = math.gcd(int(input_rate), ROBOT_OUTPUT_RATE)
        mono = resample_poly(mono.astype(np.float64),
                             ROBOT_OUTPUT_RATE // gcd, int(input_rate) // gcd)
        mono = np.clip(np.round(mono), -32768, 32767).astype('<i2')
    return np.repeat(mono, 2).astype('<i2').tobytes()


def pcm_duration(num_bytes, rate, channels=1):
    """Seconds of audio in `num_bytes` of 16-bit PCM."""
    return num_bytes / (2.0 * channels * rate)
