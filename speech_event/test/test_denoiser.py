#!/usr/bin/env python3
"""
Unit tests for the signal-processing helpers in speech_event_denoiser.py:
the speech bandpass, fan-fundamental detection from a noise profile, and the
harmonic notch cascade.

Pure DSP on synthetic signals — no ROS graph, no microphone, no recorded audio.
Assertions are on measurable filter behaviour (passband retained, stopband
attenuated, notch centred), not on exact sample values.

The module imports librosa at import time, so this file skips when that isn't
installed (same convention as person_detection's bag-replay test skipping on a
missing ONNX model).

Run via: colcon test --packages-select speech_event

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: Aug 16, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import numpy as np
import pytest

# Deliberately NOT pytest.importorskip: under pytest 6.2.5 (the version ament
# ships) a module-level Skipped raised during collection aborts collection for
# the whole session, which silently disables the sibling flake8/pep257 tests.
# A pytestmark skipif collects normally and skips at run time instead.
try:
    from speech_event import speech_event_denoiser as denoiser
except ImportError:  # pragma: no cover - exercised only where deps are absent
    denoiser = None

pytestmark = pytest.mark.skipif(
    denoiser is None,
    reason='speech_event deps (librosa) not installed')

SR = 16_000


def tone(freq_hz, duration_s=0.5, sr=SR, amplitude=1.0):
    """A unit-amplitude sine at *freq_hz*."""
    t = np.arange(int(duration_s * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float64)


def rms(signal):
    return float(np.sqrt(np.mean(np.square(signal))))


def settled(signal, sr=SR):
    """Drop the first 100 ms so IIR start-up transients don't skew the RMS."""
    return signal[int(0.1 * sr):]


# ─────────────────────────────────────────────────────────────────────────────
# butter_bandpass
# ─────────────────────────────────────────────────────────────────────────────

def test_bandpass_coefficients_have_the_requested_order():
    """A band-pass of order N has 2N+1 coefficients per side."""
    for order in (2, 3, 5):
        b, a = denoiser.butter_bandpass(80, 7500, SR, order=order)
        assert len(b) == 2 * order + 1
        assert len(a) == 2 * order + 1


def test_bandpass_filter_is_stable():
    """All poles strictly inside the unit circle — an unstable filter turns a
    quiet room into a diverging scream on the ASR input.
    """
    _, a = denoiser.butter_bandpass(80, 7500, SR)
    assert np.all(np.abs(np.roots(a)) < 1.0)


def test_bandpass_is_normalized_by_sample_rate():
    """The cutoffs are normalised against Nyquist, so the same Hz values at a
    different rate must give different coefficients.
    """
    b_16k, _ = denoiser.butter_bandpass(80, 7500, 16_000)
    b_48k, _ = denoiser.butter_bandpass(80, 7500, 48_000)
    assert not np.allclose(b_16k, b_48k)


# ─────────────────────────────────────────────────────────────────────────────
# apply_bandpass
# ─────────────────────────────────────────────────────────────────────────────

def test_bandpass_preserves_speech_band_content():
    """1 kHz sits in the middle of the speech band and must survive."""
    signal = tone(1000)
    filtered = denoiser.apply_bandpass(signal, SR)
    assert rms(settled(filtered)) > 0.5 * rms(settled(signal))


def test_bandpass_attenuates_low_frequency_rumble():
    """20 Hz is well below the 80 Hz corner — handling noise and HVAC rumble."""
    signal = tone(20)
    filtered = denoiser.apply_bandpass(signal, SR)
    assert rms(settled(filtered)) < 0.2 * rms(settled(signal))


def test_bandpass_removes_dc_offset():
    """A constant offset must be filtered away, not carried into the ASR."""
    signal = np.ones(SR, dtype=np.float64)
    filtered = denoiser.apply_bandpass(signal, SR)
    assert abs(float(np.mean(settled(filtered)))) < 0.05


def test_bandpass_attenuates_above_the_upper_corner():
    """7900 Hz is above the 7500 Hz corner and near Nyquist."""
    signal = tone(7900)
    filtered = denoiser.apply_bandpass(signal, SR)
    assert rms(settled(filtered)) < rms(settled(signal))


def test_bandpass_preserves_length_and_is_finite():
    signal = tone(500)
    filtered = denoiser.apply_bandpass(signal, SR)
    assert len(filtered) == len(signal)
    assert np.all(np.isfinite(filtered))


def test_bandpass_is_linear():
    """Property: scaling the input scales the output by the same factor."""
    signal = tone(1000)
    once = denoiser.apply_bandpass(signal, SR)
    twice = denoiser.apply_bandpass(2.0 * signal, SR)
    np.testing.assert_allclose(twice, 2.0 * once, rtol=1e-9, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# SpeechDenoiser.detect_fundamental
# ─────────────────────────────────────────────────────────────────────────────

def make_denoiser(n_fft=512, sr=SR):
    """A denoiser with no profile on disk — construction stays offline."""
    return denoiser.SpeechDenoiser(noise_profile_path=None, sr=sr, n_fft=n_fft)


def test_detect_fundamental_finds_the_injected_peak():
    """A synthetic profile with one strong bin must resolve to that bin's
    frequency — this is how the fan's whine is located before notching.
    """
    d = make_denoiser()
    freqs = np.fft.rfftfreq(d.n_fft, d=1.0 / d.sr)
    profile = np.full(freqs.shape, 0.01)

    target_bin = int(np.argmin(np.abs(freqs - 250.0)))
    profile[target_bin] = 10.0

    assert d.detect_fundamental(profile) == pytest.approx(freqs[target_bin])


def test_detect_fundamental_ignores_peaks_outside_the_search_band():
    """A huge peak at 4 kHz (speech energy) must not be mistaken for the fan
    fundamental — only 50-500 Hz is searched.
    """
    d = make_denoiser()
    freqs = np.fft.rfftfreq(d.n_fft, d=1.0 / d.sr)
    profile = np.full(freqs.shape, 0.01)

    in_band = int(np.argmin(np.abs(freqs - 125.0)))
    out_of_band = int(np.argmin(np.abs(freqs - 4000.0)))
    profile[in_band] = 5.0
    profile[out_of_band] = 500.0  # much larger, but out of range

    assert d.detect_fundamental(profile) == pytest.approx(freqs[in_band])


def test_detect_fundamental_result_is_inside_the_search_band():
    """Property: whatever the profile shape, the answer stays in range."""
    d = make_denoiser()
    rng = np.random.default_rng(0)
    freqs = np.fft.rfftfreq(d.n_fft, d=1.0 / d.sr)
    for _ in range(20):
        fundamental = d.detect_fundamental(rng.random(freqs.shape))
        assert 50.0 <= fundamental <= 500.0


def test_detect_fundamental_honours_a_custom_band():
    d = make_denoiser()
    freqs = np.fft.rfftfreq(d.n_fft, d=1.0 / d.sr)
    profile = np.full(freqs.shape, 0.01)
    profile[int(np.argmin(np.abs(freqs - 100.0)))] = 5.0
    profile[int(np.argmin(np.abs(freqs - 300.0)))] = 9.0

    # Narrowing the band excludes the taller 300 Hz peak.
    assert d.detect_fundamental(profile, min_hz=50, max_hz=150) == \
        pytest.approx(freqs[int(np.argmin(np.abs(freqs - 100.0)))])


# ─────────────────────────────────────────────────────────────────────────────
# SpeechDenoiser.apply_notch_filters
# ─────────────────────────────────────────────────────────────────────────────

def test_notch_suppresses_the_fundamental():
    d = make_denoiser()
    signal = tone(200)
    notched = d.apply_notch_filters(signal, fundamental=200.0)
    assert rms(settled(notched)) < 0.2 * rms(settled(signal))


def test_notch_suppresses_harmonics():
    """The cascade targets k*f0, so the 3rd harmonic must drop too — fan noise
    is harmonically rich, and notching only f0 leaves most of it in place.
    """
    d = make_denoiser()
    signal = tone(600)  # 3rd harmonic of 200 Hz
    notched = d.apply_notch_filters(signal, fundamental=200.0)
    assert rms(settled(notched)) < 0.3 * rms(settled(signal))


def test_notch_leaves_frequencies_between_harmonics_alone():
    """A narrow notch (Q=30) must not gut the speech sitting between harmonics."""
    d = make_denoiser()
    signal = tone(1500)  # not a multiple of 200
    notched = d.apply_notch_filters(signal, fundamental=200.0)
    assert rms(settled(notched)) > 0.8 * rms(settled(signal))


def test_notch_stops_at_nyquist():
    """Harmonics at or above Nyquist must break the loop rather than be passed
    to iirnotch, which rejects a normalised frequency >= 1.
    """
    d = make_denoiser()
    signal = tone(1000)
    # 6 harmonics of 3000 Hz would reach 18 kHz, well past the 8 kHz Nyquist.
    notched = d.apply_notch_filters(signal, fundamental=3000.0, n_harmonics=6)
    assert len(notched) == len(signal)
    assert np.all(np.isfinite(notched))


def test_notch_does_not_mutate_its_input():
    """The implementation copies before filtering; callers reuse the buffer."""
    d = make_denoiser()
    signal = tone(200)
    original = signal.copy()
    d.apply_notch_filters(signal, fundamental=200.0)
    np.testing.assert_array_equal(signal, original)


def test_notch_preserves_length_and_is_finite():
    d = make_denoiser()
    signal = tone(200)
    notched = d.apply_notch_filters(signal, fundamental=200.0)
    assert len(notched) == len(signal)
    assert np.all(np.isfinite(notched))
