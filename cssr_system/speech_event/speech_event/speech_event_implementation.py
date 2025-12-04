"""
speech_event_implementation.py
Implementation of 4-microphone speech localization with beamforming and Whisper ASR

Author: Yohannes Tadesse Haile
Date: November 8, 2025
Version: v1.0

This program comes with ABSOLUTELY NO WARRANTY.
"""


import math
import time
import numpy as np
import torch
import rclpy
import os
import onnxruntime
import threading
import noisereduce as nr
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Vector3Stamped
from naoqi_bridge_msgs.msg import AudioBuffer
from faster_whisper import WhisperModel
from scipy.signal import resample_poly


# =============================================================================
# Silero VAD ONNX Wrapper
# =============================================================================
class OnnxWrapper:
    def __init__(self, path, force_onnx_cpu=False):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and 'CPUExecutionProvider' in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=opts)
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        self.sample_rate = 16000
        self.reset_states()

    def reset_states(self, batch_size=1):
        self._state = torch.zeros((2, batch_size, 128)).float()
        self._context = torch.zeros(batch_size, 64)
        self._last_batch_size = batch_size

    def __call__(self, x: np.ndarray) -> float:
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        if batch_size != self._last_batch_size:
            self.reset_states(batch_size)
        
        x_with_context = torch.cat([self._context, x], dim=1)
        
        ort_inputs = {
            'input': x_with_context.numpy(),
            'state': self._state.numpy(),
            'sr': np.array(self.sample_rate, dtype='int64')
        }
        ort_outs = self.session.run(None, ort_inputs)
        out, state = ort_outs
        
        self._state = torch.from_numpy(state)
        self._context = x_with_context[:, -64:]
        self._last_batch_size = batch_size
        
        return float(out.squeeze())


# =============================================================================
# Main Speech Recognition Node
# =============================================================================
class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__("speech_recognition")

        # =====================================================
        # Parameters
        # =====================================================
        # Audio
        self.declare_parameter("input_sample_rate", 48000)  # Pepper's native rate
        self.declare_parameter("vad_sample_rate", 16000)    # Silero/Whisper rate
        self.declare_parameter("device", "cuda")
        self.declare_parameter("compute_type", "float16")
        self.declare_parameter("language", "en")
        self.declare_parameter("whisper_model_id", "deepdml/faster-whisper-large-v3-turbo-ct2")
        
        # VAD thresholds
        self.declare_parameter("speech_threshold", 0.7)
        self.declare_parameter("neg_threshold", 0.35)
        self.declare_parameter("min_silence_duration_ms", 300)
        self.declare_parameter("max_speech_duration_s", 10.0)
        self.declare_parameter("min_speech_duration", 0.3)
        self.declare_parameter("pre_speech_buffer_ms", 200)
        
        # Localization & beamforming
        self.declare_parameter("use_beamforming", True)
        self.declare_parameter("use_localization", True)
        self.declare_parameter("speed_of_sound", 343.0)
        
        # Noise reduction
        self.declare_parameter("use_noise_reduction", True)
        self.declare_parameter("noise_context_duration", 2.0)
        self.declare_parameter("nr_stationary", True)
        self.declare_parameter("nr_prop_decrease", 0.9)
        
        # Intensity gate
        self.declare_parameter("intensity_threshold", 0.004)
        
        # Topics
        self.declare_parameter("microphone_topic", "/audio")

        # Load parameters
        self.input_sample_rate = int(self.get_parameter("input_sample_rate").value)
        self.vad_sample_rate = int(self.get_parameter("vad_sample_rate").value)
        self.device = self.get_parameter("device").value
        self.compute_type = self.get_parameter("compute_type").value
        self.language = self.get_parameter("language").value
        self.whisper_model_id = self.get_parameter("whisper_model_id").value
        
        self.speech_threshold = float(self.get_parameter("speech_threshold").value)
        self.neg_threshold = float(self.get_parameter("neg_threshold").value)
        self.min_silence_duration_ms = int(self.get_parameter("min_silence_duration_ms").value)
        self.max_speech_duration_s = float(self.get_parameter("max_speech_duration_s").value)
        self.min_speech_duration = float(self.get_parameter("min_speech_duration").value)
        self.pre_speech_buffer_ms = int(self.get_parameter("pre_speech_buffer_ms").value)
        
        self.use_beamforming = bool(self.get_parameter("use_beamforming").value)
        self.use_localization = bool(self.get_parameter("use_localization").value)
        self.speed_of_sound = float(self.get_parameter("speed_of_sound").value)
        
        self.use_noise_reduction = bool(self.get_parameter("use_noise_reduction").value)
        self.noise_context_duration = float(self.get_parameter("noise_context_duration").value)
        self.nr_stationary = bool(self.get_parameter("nr_stationary").value)
        self.nr_prop_decrease = float(self.get_parameter("nr_prop_decrease").value)
        
        self.intensity_threshold = float(self.get_parameter("intensity_threshold").value)
        self.microphone_topic = self.get_parameter("microphone_topic").value

        # =====================================================
        # Microphone array geometry (Pepper head frame, meters)
        # Order: [back_left, back_right, front_left, front_right]
        # =====================================================
        self.mic_positions = np.array([
            [-0.0267,  0.0343, 0.2066],  # Back-left  (RL)
            [-0.0267, -0.0343, 0.2066],  # Back-right (RR)
            [ 0.0313,  0.0343, 0.2066],  # Front-left (FL)
            [ 0.0313, -0.0343, 0.2066]   # Front-right(FR)
        ])

        # =====================================================
        # Load models
        # =====================================================
        package_path = get_package_share_directory('speech_event')
        silero_path = os.path.join(package_path, 'models', 'silero_vad.onnx')

        self.silero_model = OnnxWrapper(silero_path)
        self.get_logger().info(f"Silero VAD loaded for {self.vad_sample_rate}Hz")

        self.get_logger().info("Loading Whisper model...")
        self.whisper = WhisperModel(self.whisper_model_id, device=self.device, compute_type=self.compute_type)
        self.get_logger().info("Whisper model loaded.")
        
        self._warmup_whisper()

        # =====================================================
        # VAD chunk processing (at 16kHz)
        # =====================================================
        self.vad_chunk_size = 512  # Silero requirement
        self.vad_pending_buffer = np.zeros(0, dtype=np.float32)
        
        # Pre-speech ring buffer (16kHz)
        self.pre_speech_samples = int((self.pre_speech_buffer_ms / 1000.0) * self.vad_sample_rate)
        self.pre_speech_ring = deque(maxlen=self.pre_speech_samples)
        
        # =====================================================
        # Speech collection (16kHz for Whisper)
        # =====================================================
        self.speech_buffer_16k = []
        self.speech_active = False
        self.speech_start_time = None
        
        # Silence tracking
        self.silence_chunks = 0
        self.min_silence_chunks = int((self.min_silence_duration_ms / 1000.0) * self.vad_sample_rate / self.vad_chunk_size)
        self.max_speech_chunks = int(self.max_speech_duration_s * self.vad_sample_rate / self.vad_chunk_size)
        self.speech_chunk_count = 0

        # =====================================================
        # Noise reduction context (48kHz - before resampling)
        # =====================================================
        self.noise_context_size = int(self.input_sample_rate * self.noise_context_duration)
        self.noise_context_window = np.zeros(self.noise_context_size, dtype=np.float32)

        # =====================================================
        # Localization state
        # =====================================================
        self.current_azimuth = 0.0
        self.current_elevation = 0.0
        self.direction_lock = threading.Lock()

        # =====================================================
        # Async transcription
        # =====================================================
        self.transcription_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self.transcription_lock = threading.Lock()
        self.is_transcribing = False

        # =====================================================
        # Publishers
        # =====================================================
        self.vad_prob_pub = self.create_publisher(Float32, "/vad/speech_prob", 10)
        self.asr_pub = self.create_publisher(String, "/asr/text", 10)
        self.direction_pub = self.create_publisher(Vector3Stamped, "/speech/sound_direction", 10)
        self.azimuth_pub = self.create_publisher(Float32, "/speech/azimuth", 10)

        # Subscriber
        self.audio_sub = self.create_subscription(AudioBuffer, self.microphone_topic, self.audio_callback, 10)

        self.get_logger().info("speech_recognition ready.")
        self.get_logger().info(f"VAD: thresh={self.speech_threshold}/{self.neg_threshold}, "
                              f"silence={self.min_silence_duration_ms}ms, max={self.max_speech_duration_s}s")
        self.get_logger().info(f"Localization: {self.use_localization}, Beamforming: {self.use_beamforming}, "
                              f"Noise reduction: {self.use_noise_reduction}")

    def _warmup_whisper(self):
        """Warm up Whisper to avoid first-inference latency."""
        self.get_logger().info("Warming up Whisper...")
        dummy_audio = np.zeros(self.vad_sample_rate, dtype=np.float32)
        try:
            segments, _ = self.whisper.transcribe(
                dummy_audio,
                language=self.language,
                task="transcribe",
                beam_size=1,
                vad_filter=False
            )
            _ = list(segments)
            self.get_logger().info("Whisper warmup complete.")
        except Exception as e:
            self.get_logger().warning(f"Whisper warmup failed (non-critical): {e}")

    # =========================================================================
    # Audio Parsing
    # =========================================================================
    def parse_audio_buffer(self, msg):
        """
        Parse AudioBuffer to 4 channels at native 48kHz.
        Returns [back_left, back_right, front_left, front_right] or None.
        """
        try:
            freq_in = int(msg.frequency)
            channel_map = list(msg.channel_map)
            channels = len(channel_map)
            data = np.asarray(msg.data, dtype=np.int16)

            if channels == 0 or data.size == 0:
                return None

            num_frames = data.size // channels
            if num_frames <= 0:
                return None
            
            frames = data[:num_frames * channels].reshape(num_frames, channels).astype(np.float32) / 32767.0

            def get_chan(enum_val, fallback=None):
                if enum_val in channel_map:
                    idx = channel_map.index(enum_val)
                    return frames[:, idx]
                return np.copy(fallback) if fallback is not None else np.zeros(num_frames, dtype=np.float32)

            FL = get_chan(AudioBuffer.CHANNEL_FRONT_LEFT)
            FR = get_chan(AudioBuffer.CHANNEL_FRONT_RIGHT)
            RL = get_chan(AudioBuffer.CHANNEL_REAR_LEFT, FL)
            RR = get_chan(AudioBuffer.CHANNEL_REAR_RIGHT, FR)

            # Order: [back_left, back_right, front_left, front_right]
            mic_signals = [RL, RR, FL, FR]

            # Resample if input isn't 48kHz (shouldn't happen with Pepper)
            if freq_in != self.input_sample_rate:
                gcd = math.gcd(freq_in, self.input_sample_rate)
                up = self.input_sample_rate // gcd
                down = freq_in // gcd
                mic_signals = [resample_poly(sig, up, down).astype(np.float32) for sig in mic_signals]

            return mic_signals

        except Exception as e:
            self.get_logger().error(f"AudioBuffer parse error: {e}")
            return None

    def resample_48k_to_16k(self, audio_48k: np.ndarray) -> np.ndarray:
        """Resample 48kHz audio to 16kHz for VAD/Whisper."""
        # 48000 / 16000 = 3, so down=3, up=1
        return resample_poly(audio_48k, 1, 3).astype(np.float32)

    # =========================================================================
    # Intensity Gate
    # =========================================================================
    def is_intense_enough(self, signal: np.ndarray) -> bool:
        """Check if signal exceeds RMS intensity threshold."""
        rms = float(np.sqrt(np.mean(signal ** 2)))
        return rms >= self.intensity_threshold

    # =========================================================================
    # Noise Reduction (operates at 48kHz)
    # =========================================================================
    def apply_noise_reduction(self, signal_48k: np.ndarray) -> np.ndarray:
        """Apply noise reduction using rolling context window at 48kHz."""
        if not self.use_noise_reduction:
            return signal_48k

        try:
            block_size = len(signal_48k)
            if block_size <= 0:
                return signal_48k

            # Roll context and append new block
            self.noise_context_window = np.roll(self.noise_context_window, -block_size)
            self.noise_context_window[-block_size:] = signal_48k

            reduced_context = nr.reduce_noise(
                y=self.noise_context_window,
                sr=self.input_sample_rate,
                stationary=self.nr_stationary,
                prop_decrease=self.nr_prop_decrease
            )

            return reduced_context[-block_size:].astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Noise reduction error: {e}")
            return signal_48k

    # =========================================================================
    # GCC-PHAT Localization (operates at 48kHz for precision)
    # =========================================================================
    def gcc_phat(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """GCC-PHAT for time delay estimation."""
        try:
            n = sig1.shape[0] + sig2.shape[0]

            SIG1 = np.fft.rfft(sig1, n=n)
            SIG2 = np.fft.rfft(sig2, n=n)

            R = SIG1 * np.conj(SIG2)
            R /= (np.abs(R) + 1e-10)

            cc = np.fft.irfft(R, n=n)

            max_shift = int(n / 2)
            cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
            shift = int(np.argmax(np.abs(cc)) - max_shift)

            return shift / float(self.input_sample_rate)
        except Exception as e:
            self.get_logger().error(f"GCC-PHAT error: {e}")
            return 0.0

    def localize_4mic(self, mic_signals: list) -> tuple:
        """
        4-mic sound source localization using TDOA at 48kHz.
        Returns (azimuth, elevation, confidence).
        """
        try:
            num_mics = 4
            tdoa_matrix = np.zeros((num_mics, num_mics), dtype=np.float32)

            for i in range(num_mics):
                for j in range(i + 1, num_mics):
                    tdoa = self.gcc_phat(mic_signals[i], mic_signals[j])
                    tdoa_matrix[i, j] = tdoa
                    tdoa_matrix[j, i] = -tdoa

            azimuth, elevation = self._estimate_direction_from_tdoa(tdoa_matrix)

            # Confidence from TDOA consistency
            tdoa_variance = float(np.var(tdoa_matrix[np.triu_indices(num_mics, k=1)]))
            confidence = 1.0 / (1.0 + tdoa_variance * 1e6)

            return azimuth, elevation, confidence

        except Exception as e:
            self.get_logger().error(f"4-mic localization error: {e}")
            return 0.0, 0.0, 0.0

    def _estimate_direction_from_tdoa(self, tdoa_matrix: np.ndarray) -> tuple:
        """Estimate direction from TDOA using least squares optimization."""
        def cost_function(angles):
            azimuth, elevation = angles

            source_vec = np.array([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation)
            ])

            predicted_tdoa = np.zeros((4, 4), dtype=np.float32)
            for i in range(4):
                for j in range(i + 1, 4):
                    dist_i = float(np.dot(self.mic_positions[i], source_vec))
                    dist_j = float(np.dot(self.mic_positions[j], source_vec))
                    predicted_tdoa[i, j] = (dist_j - dist_i) / self.speed_of_sound
                    predicted_tdoa[j, i] = -predicted_tdoa[i, j]

            return float(np.sum((tdoa_matrix - predicted_tdoa) ** 2))

        result = minimize(
            cost_function,
            [0.0, 0.0],
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi), (-np.pi/4, np.pi/4)]
        )

        return float(result.x[0]), float(result.x[1])

    # =========================================================================
    # Beamforming (operates at 48kHz)
    # =========================================================================
    def apply_beamforming(self, mic_signals: list, azimuth: float, elevation: float) -> np.ndarray:
        """Delay-and-sum beamforming at 48kHz."""
        if not self.use_beamforming:
            return mic_signals[2]  # Default to front-left

        try:
            target_direction = np.array([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation)
            ])

            delays = np.zeros(4, dtype=np.float32)
            for i in range(4):
                distance = float(np.dot(self.mic_positions[i], target_direction))
                delays[i] = distance / self.speed_of_sound

            delays -= delays.min()
            delay_samples = (delays * self.input_sample_rate).astype(int)

            signal_length = min(len(sig) for sig in mic_signals)
            beamformed = np.zeros(signal_length, dtype=np.float32)

            for i in range(4):
                delay = int(delay_samples[i])
                if delay >= signal_length:
                    continue
                shifted = np.zeros(signal_length, dtype=np.float32)
                shifted[delay:] = mic_signals[i][:signal_length - delay]
                beamformed += shifted

            beamformed /= 4.0
            return np.clip(beamformed, -1.0, 1.0)

        except Exception as e:
            self.get_logger().error(f"Beamforming error: {e}")
            return mic_signals[2]

    def publish_direction(self, azimuth: float, elevation: float, confidence: float):
        """Publish sound source direction."""
        stamp = self.get_clock().now().to_msg()

        direction_msg = Vector3Stamped()
        direction_msg.header.stamp = stamp
        direction_msg.header.frame_id = 'Head'
        direction_msg.vector.x = float(np.cos(elevation) * np.cos(azimuth))
        direction_msg.vector.y = float(np.cos(elevation) * np.sin(azimuth))
        direction_msg.vector.z = float(np.sin(elevation))
        self.direction_pub.publish(direction_msg)

        azimuth_msg = Float32()
        azimuth_msg.data = float(azimuth)
        self.azimuth_pub.publish(azimuth_msg)

    # =========================================================================
    # Audio Callback
    # =========================================================================
    def audio_callback(self, msg: AudioBuffer):
        """
        Main audio processing pipeline:
        1. Parse 4-channel 48kHz audio
        2. Intensity gate
        3. Localization at 48kHz
        4. Beamforming at 48kHz
        5. Noise reduction at 48kHz
        6. Resample to 16kHz
        7. VAD + speech collection
        """
        # Parse multi-channel audio (48kHz)
        mic_signals_48k = self.parse_audio_buffer(msg)
        if mic_signals_48k is None:
            return

        # Use front-left for intensity check
        if not self.is_intense_enough(mic_signals_48k[2]):
            return

        # =====================================================
        # Localization at 48kHz (full precision)
        # =====================================================
        if self.use_localization:
            azimuth, elevation, confidence = self.localize_4mic(mic_signals_48k)
            
            with self.direction_lock:
                self.current_azimuth = azimuth
                self.current_elevation = elevation
            
            self.publish_direction(azimuth, elevation, confidence)
        else:
            azimuth, elevation = 0.0, 0.0

        # =====================================================
        # Beamforming at 48kHz
        # =====================================================
        if self.use_beamforming and self.use_localization:
            audio_48k = self.apply_beamforming(mic_signals_48k, azimuth, elevation)
        else:
            audio_48k = mic_signals_48k[2]  # Front-left

        # =====================================================
        # Noise reduction at 48kHz
        # =====================================================
        audio_48k_clean = self.apply_noise_reduction(audio_48k)

        # =====================================================
        # Resample to 16kHz for VAD/Whisper
        # =====================================================
        audio_16k = self.resample_48k_to_16k(audio_48k_clean)

        # Update pre-speech ring buffer
        for sample in audio_16k:
            self.pre_speech_ring.append(sample)

        # Accumulate for chunk-aligned VAD
        self.vad_pending_buffer = np.concatenate([self.vad_pending_buffer, audio_16k])

        # Process complete 512-sample chunks
        while len(self.vad_pending_buffer) >= self.vad_chunk_size:
            vad_chunk = self.vad_pending_buffer[:self.vad_chunk_size]
            self.vad_pending_buffer = self.vad_pending_buffer[self.vad_chunk_size:]
            self._process_vad_chunk(vad_chunk)

    # =========================================================================
    # VAD State Machine
    # =========================================================================
    def _process_vad_chunk(self, vad_chunk: np.ndarray):
        """Process 512-sample VAD chunk through state machine."""
        speech_prob = self._run_silero_vad(vad_chunk)

        prob_msg = Float32()
        prob_msg.data = float(speech_prob)
        self.vad_prob_pub.publish(prob_msg)

        vad_is_speech = speech_prob >= self.speech_threshold
        vad_is_silence = speech_prob < self.neg_threshold

        if not self.speech_active and vad_is_speech:
            # START: Include pre-speech buffer
            self.speech_active = True
            self.speech_start_time = time.time()
            self.silence_chunks = 0
            self.speech_chunk_count = 1

            pre_speech_audio = np.array(list(self.pre_speech_ring), dtype=np.float32)
            self.speech_buffer_16k = [pre_speech_audio, vad_chunk.copy()]

            self.get_logger().info(f"VAD: speech START (prob={speech_prob:.3f}, "
                                  f"pre-buffer={len(pre_speech_audio)} samples)")

        elif self.speech_active and vad_is_speech:
            self.speech_buffer_16k.append(vad_chunk.copy())
            self.silence_chunks = 0
            self.speech_chunk_count += 1

            if self.speech_chunk_count >= self.max_speech_chunks:
                self._finalize_speech(speech_prob, reason="max_duration")

        elif self.speech_active and vad_is_silence:
            self.speech_buffer_16k.append(vad_chunk.copy())
            self.silence_chunks += 1
            self.speech_chunk_count += 1

            if self.silence_chunks >= self.min_silence_chunks:
                self._finalize_speech(speech_prob, reason="silence")
            elif self.speech_chunk_count >= self.max_speech_chunks:
                self._finalize_speech(speech_prob, reason="max_duration")

        elif self.speech_active:
            self.speech_buffer_16k.append(vad_chunk.copy())
            self.speech_chunk_count += 1

            if self.speech_chunk_count >= self.max_speech_chunks:
                self._finalize_speech(speech_prob, reason="max_duration")

    def _run_silero_vad(self, audio_chunk: np.ndarray) -> float:
        """Run Silero VAD on 512-sample chunk."""
        try:
            if len(audio_chunk) != 512:
                return 0.0
            return self.silero_model(audio_chunk)
        except Exception as e:
            self.get_logger().error(f"Silero VAD error: {e}")
            return 0.0

    def _reset_vad_state(self):
        """Reset VAD internal state."""
        self.silero_model.reset_states()

    # =========================================================================
    # Speech Finalization & Transcription
    # =========================================================================
    def _finalize_speech(self, speech_prob: float, reason: str):
        """Finalize speech segment and trigger async transcription."""
        self.speech_active = False

        if reason == "silence":
            silence_duration_s = self.silence_chunks * self.vad_chunk_size / self.vad_sample_rate
            self.get_logger().info(f"VAD: speech END (prob={speech_prob:.3f}, silence={silence_duration_s:.2f}s)")
        else:
            self.get_logger().info(f"VAD: speech END (max duration {self.max_speech_duration_s}s reached)")

        self._reset_vad_state()

        if not self.speech_buffer_16k:
            self._reset_speech_state()
            return

        speech_audio = np.concatenate(self.speech_buffer_16k).astype(np.float32)
        duration_s = len(speech_audio) / self.vad_sample_rate

        if duration_s < self.min_speech_duration:
            self.get_logger().info(f"Ignoring short segment ({duration_s:.2f}s)")
            self._reset_speech_state()
            return

        with self.transcription_lock:
            if self.is_transcribing:
                self.get_logger().warning("Previous transcription still running, queuing...")
            self.is_transcribing = True

        self.transcription_executor.submit(self._transcribe_async, speech_audio, duration_s)
        self._reset_speech_state()

    def _reset_speech_state(self):
        """Reset speech collection state."""
        self.speech_buffer_16k = []
        self.silence_chunks = 0
        self.speech_chunk_count = 0
        self.speech_start_time = None

    def _transcribe_async(self, audio: np.ndarray, duration_s: float):
        """Async Whisper transcription."""
        try:
            self.get_logger().info(f"[Async] Running Whisper on {duration_s:.2f}s segment...")
            start_time = time.time()

            segments, info = self.whisper.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                beam_size=1,
                vad_filter=False
            )

            text = " ".join([seg.text.strip() for seg in segments]).strip()
            elapsed = time.time() - start_time
            rtf = elapsed / info.duration if info.duration > 0 else 0.0

            self.get_logger().info(f"[Async] ASR done. elapsed={elapsed:.3f}s, RTF={rtf:.3f}")
            self.get_logger().info(f"[Async] Transcript: '{text}'")

            if text:
                out = String()
                out.data = text
                self.asr_pub.publish(out)

        except Exception as e:
            self.get_logger().error(f"[Async] Transcription error: {e}")
        finally:
            with self.transcription_lock:
                self.is_transcribing = False

    def destroy_node(self):
        """Clean shutdown."""
        self.get_logger().info("Shutting down...")
        self.transcription_executor.shutdown(wait=True)
        super().destroy_node()