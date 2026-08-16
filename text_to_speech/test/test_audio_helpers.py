#!/usr/bin/env python3
"""
Unit tests for the pure text and audio helpers in
text_to_speech_implementation.py: sentence splitting, speech-duration
estimation, and the resampling/chunking path that feeds Pepper's audio buffer.

No ROS graph, no TTS model, no audio device, no network. Synthesis backends
(Kokoro, ElevenLabs) and the WAV codec path are deliberately out of scope —
they need model weights, API keys, or libsndfile.

Run via: colcon test --packages-select text_to_speech

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: Aug 16, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import numpy as np
import pytest

from text_to_speech import text_to_speech_implementation as tts


# ─────────────────────────────────────────────────────────────────────────────
# split_into_sentences
# ─────────────────────────────────────────────────────────────────────────────

def test_splits_on_each_terminator():
    assert tts.split_into_sentences('One. Two! Three?') == ['One.', 'Two!', 'Three?']


def test_single_sentence_without_terminator():
    assert tts.split_into_sentences('No terminator here') == ['No terminator here']


def test_strips_surrounding_whitespace():
    assert tts.split_into_sentences('  Hello.   World.  ') == ['Hello.', 'World.']


def test_empty_and_whitespace_only_input():
    assert tts.split_into_sentences('') == []
    assert tts.split_into_sentences('    ') == []
    assert tts.split_into_sentences('\n\t ') == []


def test_no_empty_fragments_from_repeated_terminators():
    """Repeated terminators must not yield empty strings that speak as silence."""
    for sentence in tts.split_into_sentences('Wait... go!'):
        assert sentence.strip() == sentence
        assert sentence


def test_splits_on_newline_boundaries_after_terminator():
    """The boundary is any whitespace run, so newlines split too."""
    assert tts.split_into_sentences('First.\nSecond.') == ['First.', 'Second.']


def test_does_not_split_a_decimal_number():
    """A period with no following whitespace is not a boundary — splitting
    "3.5" would make Pepper say "three" and "five" as separate sentences.
    """
    assert tts.split_into_sentences('It costs 3.5 dollars.') == \
        ['It costs 3.5 dollars.']


def test_preserves_naoqi_tags_inside_sentences():
    r"""Prosody tags must ride along with their sentence, not be stripped."""
    out = tts.split_into_sentences(r'\rspd=85\Hello there. Second one.')
    assert out == [r'\rspd=85\Hello there.', 'Second one.']


def test_rejoining_preserves_all_words():
    """Property: splitting must not drop content."""
    text = 'Alpha beta. Gamma delta! Epsilon zeta?'
    assert ' '.join(tts.split_into_sentences(text)).split() == text.split()


# ─────────────────────────────────────────────────────────────────────────────
# estimate_duration
# ─────────────────────────────────────────────────────────────────────────────

def test_duration_scales_with_length():
    # 100 chars at 10 chars/s = 10 s, plus 0.5 s padding.
    assert tts.estimate_duration('x' * 100, 10.0, 0.5) == pytest.approx(10.5)


def test_short_text_is_floored_at_one_second_plus_padding():
    """The floor keeps the caller from racing ahead of a very short utterance."""
    assert tts.estimate_duration('hi', 10.0, 0.5) == pytest.approx(1.5)
    assert tts.estimate_duration('', 10.0, 0.5) == pytest.approx(1.5)


def test_duration_is_monotonic_in_length():
    previous = 0.0
    for n in (0, 5, 20, 100, 500, 2000):
        current = tts.estimate_duration('x' * n, 15.0, 0.3)
        assert current >= previous, f'regressed at n={n}'
        previous = current


def test_padding_is_additive():
    base = tts.estimate_duration('x' * 100, 10.0, 0.0)
    assert tts.estimate_duration('x' * 100, 10.0, 2.0) == pytest.approx(base + 2.0)


def test_faster_rate_never_takes_longer():
    slow = tts.estimate_duration('x' * 300, 5.0, 0.0)
    fast = tts.estimate_duration('x' * 300, 25.0, 0.0)
    assert fast <= slow


# ─────────────────────────────────────────────────────────────────────────────
# Robot audio constants
# ─────────────────────────────────────────────────────────────────────────────

def test_robot_audio_constants():
    """Both are hard contracts with the NAOqi audio driver — ROBOT_CHUNK_FRAMES
    is the documented per-send cap, and exceeding it drops audio silently.
    """
    assert tts.ROBOT_RATE == 48_000
    assert tts.ROBOT_CHUNK_FRAMES == 16_384


# ─────────────────────────────────────────────────────────────────────────────
# resample_chunks
# ─────────────────────────────────────────────────────────────────────────────

def test_resample_chunks_is_a_passthrough_at_matching_rates():
    """Equal rates must skip scipy entirely and hand back the same arrays."""
    chunks = [np.ones(10, dtype=np.float32), np.zeros(5, dtype=np.float32)]
    out = list(tts.resample_chunks(iter(chunks), 24_000, 24_000))
    assert len(out) == 2
    assert out[0] is chunks[0]
    assert out[1] is chunks[1]


def test_resample_chunks_changes_length_by_the_rate_ratio():
    chunk = np.zeros(2400, dtype=np.float32)
    out = list(tts.resample_chunks(iter([chunk]), 24_000, 48_000))
    assert len(out) == 1
    assert out[0].dtype == np.float32
    assert len(out[0]) == pytest.approx(4800, rel=0.02)


def test_resample_chunks_on_empty_generator():
    assert list(tts.resample_chunks(iter([]), 24_000, 48_000)) == []


# ─────────────────────────────────────────────────────────────────────────────
# collect_and_resample
# ─────────────────────────────────────────────────────────────────────────────

def test_collect_and_resample_concatenates_then_resamples():
    chunks = [np.zeros(1200, dtype=np.float32) for _ in range(2)]
    out = tts.collect_and_resample(iter(chunks), 24_000, 48_000)
    assert out.dtype == np.float32
    assert len(out) == pytest.approx(4800, rel=0.02)


def test_collect_and_resample_on_empty_generator():
    out = tts.collect_and_resample(iter([]), 24_000, 48_000)
    assert isinstance(out, np.ndarray)
    assert len(out) == 0
    assert out.dtype == np.float32


def test_collect_and_resample_is_a_plain_concat_at_matching_rates():
    chunks = [np.arange(4, dtype=np.float32), np.arange(3, dtype=np.float32)]
    out = tts.collect_and_resample(iter(chunks), 16_000, 16_000)
    np.testing.assert_array_equal(out, np.concatenate(chunks))


# ─────────────────────────────────────────────────────────────────────────────
# iter_robot_chunks
# ─────────────────────────────────────────────────────────────────────────────

def _decode(audio_list):
    """Turn the List[int] byte payload back into interleaved stereo int16."""
    return np.frombuffer(bytes(audio_list), dtype=np.int16)


def test_robot_chunks_are_stereo_int16_within_the_driver_cap():
    src_rate = 24_000
    # Three full aligned chunks' worth of source audio.
    src_frames = tts.ROBOT_CHUNK_FRAMES * src_rate // tts.ROBOT_RATE
    gen = iter([np.zeros(src_frames * 3, dtype=np.float32)])

    emitted = list(tts.iter_robot_chunks(gen, src_rate))
    assert emitted, 'expected at least one chunk'

    for audio_list, wait_time in emitted:
        samples = _decode(audio_list)
        # Stereo interleaving means an even sample count...
        assert len(samples) % 2 == 0
        # ...and the driver cap is per-channel frames.
        assert len(samples) // 2 <= tts.ROBOT_CHUNK_FRAMES
        assert wait_time >= 0.01


def test_robot_chunks_duplicate_the_mono_signal_across_both_channels():
    src_rate = tts.ROBOT_RATE  # no resampling, so samples map 1:1
    signal = np.full(tts.ROBOT_CHUNK_FRAMES // 2, 0.5, dtype=np.float32)
    audio_list, _ = next(tts.iter_robot_chunks(iter([signal]), src_rate))

    samples = _decode(audio_list)
    left, right = samples[0::2], samples[1::2]
    np.testing.assert_array_equal(left, right)
    # 0.5 * 32767 lands on 16383 after truncation to int16.
    assert left[0] == pytest.approx(16383, abs=1)


def test_robot_chunks_apply_stream_volume():
    src_rate = tts.ROBOT_RATE
    signal = np.full(1024, 0.5, dtype=np.float32)

    loud, _ = next(tts.iter_robot_chunks(iter([signal]), src_rate, stream_volume=1.0))
    quiet, _ = next(tts.iter_robot_chunks(iter([signal]), src_rate, stream_volume=0.5))

    assert abs(int(_decode(quiet)[0])) < abs(int(_decode(loud)[0]))


def test_robot_chunks_clip_instead_of_wrapping():
    """A volume that overdrives the signal must saturate at the int16 rail.
    Without the clip this wraps to a large negative value — a loud pop.
    """
    signal = np.ones(1024, dtype=np.float32)
    audio_list, _ = next(
        tts.iter_robot_chunks(iter([signal]), tts.ROBOT_RATE, stream_volume=4.0))

    samples = _decode(audio_list)
    assert samples.max() == 32767
    assert samples.min() >= 0, 'clipping wrapped to negative'


def test_robot_chunks_flush_a_short_tail():
    """Audio shorter than one aligned chunk must still be emitted, not held."""
    emitted = list(tts.iter_robot_chunks(
        iter([np.zeros(256, dtype=np.float32)]), tts.ROBOT_RATE))
    assert len(emitted) == 1
    assert len(_decode(emitted[0][0])) == 512  # 256 frames, stereo


def test_robot_chunks_reassemble_the_whole_stream():
    """Property: chunking must be lossless — many small input chunks must
    produce the same total frame count as one big one.
    """
    src_rate = tts.ROBOT_RATE
    total_frames = tts.ROBOT_CHUNK_FRAMES * 2 + 500

    one_shot = list(tts.iter_robot_chunks(
        iter([np.zeros(total_frames, dtype=np.float32)]), src_rate))
    dribbled = list(tts.iter_robot_chunks(
        iter([np.zeros(100, dtype=np.float32)] * (total_frames // 100)), src_rate))

    frames_of = lambda chunks: sum(len(_decode(c)) // 2 for c, _ in chunks)  # noqa: E731
    assert frames_of(one_shot) == total_frames
    assert frames_of(dribbled) == (total_frames // 100) * 100


def test_robot_chunks_on_empty_generator():
    assert list(tts.iter_robot_chunks(iter([]), 24_000)) == []
