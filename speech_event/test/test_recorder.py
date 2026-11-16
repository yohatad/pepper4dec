#!/usr/bin/env python3
"""
Unit tests for the audio_recorder's output path and recording limit
(speech_event_recorder.py): where the WAV files go, and how many frames it
records before it stops.

Pure functions only: no ROS graph, no microphone.

Run via: colcon test --packages-select speech_event

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: November 16, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import os

from speech_event.speech_event_recorder import (
    SAMPLE_WIDTH, WAV_MAX_DATA_BYTES, frame_limit, resolve_output_base)


def test_relative_output_base_goes_under_ros_home():
    assert (resolve_output_base('pepper_audio', ros_home='/data/ros')
            == '/data/ros/speech_event/pepper_audio')


def test_relative_output_base_defaults_to_dot_ros(monkeypatch):
    monkeypatch.delenv('ROS_HOME', raising=False)
    assert resolve_output_base('pepper_audio') == os.path.join(
        os.path.expanduser('~'), '.ros', 'speech_event', 'pepper_audio')


def test_relative_output_base_follows_ros_home_variable(monkeypatch):
    monkeypatch.setenv('ROS_HOME', '/tmp/rh')
    assert resolve_output_base('a/b') == '/tmp/rh/speech_event/a/b'


def test_absolute_and_home_paths_are_kept():
    assert resolve_output_base('/srv/rec/x', ros_home='/data/ros') == '/srv/rec/x'
    assert resolve_output_base('~/rec/x', ros_home='/data/ros') == os.path.join(
        os.path.expanduser('~'), 'rec', 'x')


def test_output_base_is_never_inside_the_package_share():
    # The old default resolved against share/speech_event and filled install/.
    assert 'share' not in resolve_output_base('data/pepper_audio',
                                              ros_home='/data/ros').split(os.sep)


def test_max_seconds_limits_the_frames():
    assert frame_limit(48000, 4, 3600) == 3600 * 48000


def test_wav_size_caps_an_unlimited_recording():
    frames = frame_limit(48000, 4, 0)
    assert frames * 4 * SAMPLE_WIDTH <= WAV_MAX_DATA_BYTES
    assert (frames + 1) * 4 * SAMPLE_WIDTH > WAV_MAX_DATA_BYTES
    # About 3.1 hours of Pepper's 48 kHz x 4 channel stream.
    assert 3.0 < frames / 48000 / 3600 < 3.2


def test_wav_size_caps_a_too_long_max_seconds():
    assert frame_limit(48000, 4, 24 * 3600) == frame_limit(48000, 4, 0)
