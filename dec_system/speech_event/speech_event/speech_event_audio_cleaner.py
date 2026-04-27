"""
speech_event_audio_cleaner.py
Post-VAD, pre-ASR noise reduction for speech segments.

Applies a pipeline of bandpass filtering, harmonic notch filters for fan hum,
and a Wiener filter with online noise estimation to clean speech segments
collected by the VAD before they are passed to Whisper for transcription.

Author: Yohannes Tadesse Haile
Date: April 2026
Version: v1.0

This program comes with ABSOLUTELY NO WARRANTY.
"""

import os
import numpy as np
import scipy.ndimage
import librosa
from scipy.signal import butter, lfilter, iirnotch


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a


def apply_bandpass(data, fs, lowcut=80, highcut=7500):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)


class AudioCleaner:
    """
    Noise reduction for VAD-extracted speech segments at a fixed sample rate.

    Instantiate once at node startup (loads the noise profile and detects the
    fan hum fundamental).  Call clean() on each speech segment before Whisper.

    Parameters
    ----------
    noise_profile_path : str or None
        Path to a .npy file containing the mean magnitude spectrum of the noise
        floor, recorded at `sr` Hz with n_fft bins.  If None or the file does
        not exist, only the online (minimum-statistics) estimator is used.
        NOTE: the profile must be recorded at `sr` (16 kHz for this pipeline) —
        the 48 kHz fan_noise_profile.npy from the standalone test script is not
        compatible.
    sr : int
        Sample rate of the audio that will be passed to clean() (default 16000).
    n_fft : int
        FFT size for STFT (default 512).
    hop_length : int
        STFT hop length (default 256).
    alpha : float
        Wiener filter aggressiveness.  Higher = more suppression.
        Start at 0.5; lower toward 0.3 if speech is being over-suppressed.
    spectral_floor_scale : float
        Floor applied to each bin as a fraction of the noise estimate, to avoid
        complete silence in noise-only bins (default 0.02).
    smoothing_size : tuple
        Kernel size for the median filter applied to the magnitude spectrogram
        along the frequency axis (default (5, 1)).
    logger : rclpy logger or None
        If provided, info/warning messages are sent through ROS logging.
    """

    def __init__(self, noise_profile_path=None, sr=16000, n_fft=512, hop_length=256,
                 alpha=0.5, spectral_floor_scale=0.02, smoothing_size=(5, 1), logger=None):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        self.spectral_floor_scale = spectral_floor_scale
        self.smoothing_size = smoothing_size
        self._log = logger

        self.static_noise_profile = None  # shape (n_fft//2+1,) after load
        self.fundamental_hz = None

        if noise_profile_path:
            self.load_profile(noise_profile_path)

    # ── Initialisation helpers ────────────────────────────────────────────────

    def load_profile(self, path):
        if not os.path.exists(path):
            self.log(
                f"AudioCleaner: noise profile not found at '{path}' — "
                "using online estimation only"
            )
            return

        profile = np.load(path)
        expected_bins = self.n_fft // 2 + 1

        if profile.shape[0] != expected_bins:
            self.log(
                f"AudioCleaner: profile has {profile.shape[0]} bins but "
                f"n_fft={self.n_fft} expects {expected_bins} — skipping static "
                f"profile (record one at {self.sr} Hz with n_fft={self.n_fft})"
            )
            return

        self.static_noise_profile = profile
        self.fundamental_hz = self.detect_fundamental(profile)
        self.log(
            f"AudioCleaner: loaded profile from '{path}' | "
            f"fan fundamental={self.fundamental_hz:.1f} Hz | sr={self.sr} Hz"
        )

    def detect_fundamental(self, profile_1d, min_hz=50, max_hz=500):
        """Return the frequency of the strongest spectral peak between min_hz and max_hz."""
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr)
        mask = (freqs >= min_hz) & (freqs <= max_hz)
        peak_bin = np.argmax(profile_1d[mask])
        return freqs[mask][peak_bin]

    # ── Per-segment processing ────────────────────────────────────────────────

    def apply_notch_filters(self, signal, fundamental, n_harmonics=6, quality=30):
        """Apply narrow IIR notch filters at the fundamental and its harmonics."""
        out = signal.copy()
        for k in range(1, n_harmonics + 1):
            freq = fundamental * k
            if freq >= self.sr / 2:
                break
            b, a = iirnotch(freq, quality, self.sr)
            out = lfilter(b, a, out)
        return out

    def clean(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to a float32 speech segment at self.sr.

        The output is RMS-matched to the input so Whisper's internal
        no-speech threshold sees the same signal level as before cleaning.

        Parameters
        ----------
        audio : np.ndarray, float32, shape (N,)
            Raw speech segment from the VAD buffer.

        Returns
        -------
        np.ndarray
            Cleaned float32 audio clipped to [-1, 1].
        """
        if len(audio) == 0:
            return audio

        orig_rms = float(np.sqrt(np.mean(audio ** 2)))

        # 1. Bandpass — pass 80 Hz–7500 Hz, leave headroom below Nyquist
        highcut = min(7500, int(self.sr / 2) - 100)
        audio = apply_bandpass(audio, self.sr, lowcut=80, highcut=highcut)

        # 2. Notch filters for fan hum harmonics (only if profile was loaded)
        if self.fundamental_hz is not None:
            audio = self.apply_notch_filters(audio, self.fundamental_hz)

        # 3. STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        signal_mag = np.abs(stft)
        signal_phase = np.angle(stft)

        # 4. Noise estimate: rolling minimum (~0.8 s window) blended with static profile
        online_noise = scipy.ndimage.minimum_filter1d(signal_mag, size=30, axis=1)
        if self.static_noise_profile is not None:
            noise_profile = np.maximum(online_noise, self.static_noise_profile[:, None])
        else:
            noise_profile = online_noise

        # 5. Wiener filter — per-bin gain in [0, 1] based on local SNR
        snr = signal_mag / (self.alpha * noise_profile + 1e-10)
        wiener_gain = snr / (snr + 1)
        clean_mag = wiener_gain * signal_mag

        # 6. Spectral floor — prevent complete silence in noise-only bins
        clean_mag = np.maximum(clean_mag, self.spectral_floor_scale * noise_profile)

        # 7. Frequency-axis smoothing to reduce residual musical noise
        clean_mag = scipy.ndimage.median_filter(clean_mag, size=self.smoothing_size)

        # 8. Reconstruct — preserve original length to avoid timestamp drift in Whisper
        clean_stft = clean_mag * np.exp(1j * signal_phase)
        clean_audio = librosa.istft(clean_stft, hop_length=self.hop_length, length=len(audio))

        # 9. RMS-match output to input so Whisper's level thresholds are undisturbed
        clean_rms = float(np.sqrt(np.mean(clean_audio ** 2)))
        if clean_rms > 1e-8 and orig_rms > 1e-8:
            clean_audio = clean_audio * (orig_rms / clean_rms)

        return np.clip(clean_audio.astype(np.float32), -1.0, 1.0)

    def log(self, msg):
        if self._log:
            self._log.info(msg)
        else:
            print(msg)
