"""Tests for remote_dialogue_audio: Pepper <-> server audio formats."""

import numpy as np
import pytest

from remote_dialogue.remote_dialogue_audio import (FRAME_BYTES, FrameChunker,
                                                   MicrophoneConverter,
                                                   mono_to_robot_stereo, pcm_duration,
                                                   ROBOT_OUTPUT_RATE)

# naoqi_driver's channel_map for Pepper's four microphones
PEPPER_MAP = [0, 2, 3, 5]


def pepper_buffer(signals):
    """Interleave per-channel float signals into an AudioBuffer-style int16 list."""
    frames = np.stack(signals, axis=1)
    return np.round(frames).astype(np.int16).reshape(-1)


def sine(freq, seconds, rate, amplitude=10000.0):
    t = np.arange(int(seconds * rate)) / rate
    return amplitude * np.sin(2 * np.pi * freq * t)


def dominant_frequency(samples, rate):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    return np.fft.rfftfreq(samples.size, 1.0 / rate)[np.argmax(spectrum)]


def test_converter_selects_channel_and_decimates_to_16k():
    rate = 48000
    left = sine(1000, 1.0, rate)
    others = [sine(3000, 1.0, rate) for _ in range(3)]
    data = pepper_buffer([left] + others)

    out = np.frombuffer(MicrophoneConverter(channel=0).convert(data, PEPPER_MAP, rate),
                        dtype='<i2')

    assert out.size == 16000
    assert dominant_frequency(out.astype(float), 16000) == pytest.approx(1000, abs=5)
    # Amplitude survives the anti-alias filter (passband gain ~1).
    assert np.max(np.abs(out[200:])) == pytest.approx(10000, rel=0.05)


def test_converter_picks_requested_channel():
    rate = 48000
    signals = [sine(f, 0.5, rate) for f in (500, 1500, 2500, 3500)]
    data = pepper_buffer(signals)
    out = np.frombuffer(MicrophoneConverter(channel=3).convert(data, PEPPER_MAP, rate),
                        dtype='<i2')
    assert dominant_frequency(out.astype(float), 16000) == pytest.approx(2500, abs=10)


def test_converter_is_continuous_across_buffers():
    """Converting in naoqi-sized pieces equals converting all at once."""
    rate = 48000
    signal = sine(700, 1.0, rate)
    data = pepper_buffer([signal] * 4)
    whole = MicrophoneConverter().convert(data, PEPPER_MAP, rate)

    chunked = MicrophoneConverter()
    pieces = []
    step = 4 * 4096 + 4 * 7   # uneven size: decimation phase must carry over
    for start in range(0, data.size, step):
        pieces.append(chunked.convert(data[start:start + step], PEPPER_MAP, rate))
    assert b''.join(pieces) == whole


def test_converter_rejects_rate_that_is_not_a_multiple():
    with pytest.raises(ValueError):
        MicrophoneConverter().convert(np.zeros(400, np.int16), PEPPER_MAP, 44100)


def test_converter_passes_16k_through_and_handles_empty():
    converter = MicrophoneConverter()
    data = pepper_buffer([np.full(160, 123.0)] * 4)
    out = np.frombuffer(converter.convert(data, PEPPER_MAP, 16000), dtype='<i2')
    assert out.tolist() == [123] * 160
    assert converter.convert([], PEPPER_MAP, 48000) == b''
    assert converter.convert([1, 2, 3], [], 48000) == b''


def test_chunker_emits_exact_frames_and_keeps_the_rest():
    chunker = FrameChunker()
    assert chunker.push(b'\x01' * 1000) == [b'\x01' * FRAME_BYTES]
    frames = chunker.push(b'\x02' * 1000)
    assert len(frames) == 2 and all(len(f) == FRAME_BYTES for f in frames)
    assert frames[0][:360] == b'\x01' * 360 and frames[0][360:] == b'\x02' * 280
    chunker.clear()
    assert chunker.push(b'\x03' * 639) == []


@pytest.mark.parametrize('rate', [16000, 24000, 48000])
def test_mono_to_robot_stereo(rate):
    mono = (sine(440, 0.5, rate)).astype('<i2').tobytes()
    stereo = np.frombuffer(mono_to_robot_stereo(mono, rate), dtype='<i2').reshape(-1, 2)
    assert stereo.shape[0] == int(0.5 * ROBOT_OUTPUT_RATE)
    assert np.array_equal(stereo[:, 0], stereo[:, 1])
    assert dominant_frequency(stereo[:, 0].astype(float), ROBOT_OUTPUT_RATE) == \
        pytest.approx(440, abs=3)
    if rate == ROBOT_OUTPUT_RATE:
        assert np.array_equal(stereo[:, 0], np.frombuffer(mono, dtype='<i2'))


def test_mono_to_robot_stereo_odd_and_empty():
    assert mono_to_robot_stereo(b'', 48000) == b''
    assert mono_to_robot_stereo(b'\x01', 48000) == b''
    assert len(mono_to_robot_stereo(b'\x01\x00\x02', 48000)) == 4


def test_pcm_duration():
    assert pcm_duration(96000, 48000) == pytest.approx(1.0)
    assert pcm_duration(192000, 48000, channels=2) == pytest.approx(1.0)
