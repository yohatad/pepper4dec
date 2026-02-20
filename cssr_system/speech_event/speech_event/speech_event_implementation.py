"""
speech_event_implementation.py
Implementation of speech recognition with Whisper ASR.

Author: Yohannes Tadesse Haile
Date: November 8, 2025
Version: v1.0

This program comes with ABSOLUTELY NO WARRANTY.
"""

import math
import time
import numpy as np
import torch
import os
import onnxruntime
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from cssr_interfaces.action import SpeechRecognition

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float32, String
from naoqi_bridge_msgs.msg import AudioBuffer
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

class OnnxWrapper():
    def __init__(self, path, force_onnx_cpu=False, logger=None):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 4
        opts.intra_op_num_threads = 4

        available_providers = onnxruntime.get_available_providers()

        if force_onnx_cpu:
            # Explicitly force CPU
            self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=opts)
            self.device = 'CPU (forced)'
        else:
            # Try to use GPU providers in priority order
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif 'TensorrtExecutionProvider' in available_providers:
                providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            self.session = onnxruntime.InferenceSession(path, providers=providers, sess_options=opts)

            # Determine which provider was actually used
            provider = self.session.get_providers()[0]
            if 'CUDA' in provider:
                self.device = 'GPU (CUDA)'
            elif 'TensorRT' in provider or 'Tensorrt' in provider:
                self.device = 'GPU (TensorRT)'
            elif 'CPU' in provider:
                self.device = 'CPU'
            else:
                self.device = provider

        if logger:
            logger.info(f"Silero VAD using: {self.device}")
            logger.info(f"Available ONNX providers: {available_providers}")
            logger.info(f"Active ONNX provider: {self.session.get_providers()[0]}")

        # Always work with 16kHz (since we resample from 48kHz)
        self.sample_rate = 16000
        self._lock = threading.Lock()
        self.reset_states()

    def reset_states(self, batch_size=1):
        with self._lock:
            self._state = torch.zeros((2, batch_size, 128)).float()
            self._context = torch.zeros(batch_size, 64)  # context_size = 64 for 16kHz
            self._last_batch_size = batch_size

    def __call__(self, x: np.ndarray) -> float:
        """
        Process a single 512-sample chunk at 16kHz.

        Args:
            x: np.ndarray of shape (512,) at 16kHz

        Returns:
            float: Speech probability [0, 1]
        """
        with self._lock:
            # Convert to torch tensor
            if not torch.is_tensor(x):
                x = torch.from_numpy(x).float()

            # Ensure 2D: (batch_size, samples)
            if x.dim() == 1:
                x = x.unsqueeze(0)

            batch_size = x.shape[0]

            # Reset states if batch size changed
            if batch_size != self._last_batch_size:
                # Note: reset_states also acquires lock, but Python's threading.Lock is reentrant for same thread
                # Actually, we need to avoid nested lock - let's call the internal reset directly
                self._state = torch.zeros((2, batch_size, 128)).float()
                self._context = torch.zeros(batch_size, 64)
                self._last_batch_size = batch_size

            # Prepend context (64 samples for 16kHz)
            x_with_context = torch.cat([self._context, x], dim=1)

            # Run ONNX inference
            ort_inputs = {
                'input': x_with_context.numpy(),
                'state': self._state.numpy(),
                'sr': np.array(self.sample_rate, dtype='int64')
            }
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs

            # Update state and context for next call
            self._state = torch.from_numpy(state)
            self._context = x_with_context[:, -64:]  # Keep last 64 samples
            self._last_batch_size = batch_size

            # Return speech probability as scalar
            return float(out.squeeze())

class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__("speech_recognition")

        # Parameters
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("input_sample_rate", 48000)  # Pepper's native rate
        self.declare_parameter("device", "cuda")
        self.declare_parameter("compute_type", "float16")
        self.declare_parameter("language", "en")
        self.declare_parameter("whisper_model_id", "deepdml/faster-whisper-large-v3-turbo-ct2")

        self.declare_parameter("speech_threshold", 0.7)
        self.declare_parameter("neg_threshold", 0.35)
        self.declare_parameter("min_silence_duration_ms", 300)
        self.declare_parameter("max_speech_duration_s", 10.0)
        self.declare_parameter("min_speech_duration", 0.3)
        self.declare_parameter("pre_speech_buffer_ms", 200)
        self.declare_parameter("intensity_threshold", 0.001)

        self.declare_parameter("microphone_topic", "/audio")
        self.declare_parameter("action_server", True)

        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.input_sample_rate = int(self.get_parameter("input_sample_rate").value)
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
        self.intensity_threshold = float(self.get_parameter("intensity_threshold").value)

        self.microphone_topic = self.get_parameter("microphone_topic").value
        self.action_server_enabled = bool(self.get_parameter("action_server").value)

        # =====================================================
        # Validate parameters
        # =====================================================
        if self.sample_rate <= 0:
            self.get_logger().error(f"Invalid sample_rate: {self.sample_rate}")
            raise ValueError("sample_rate must be positive")

        if self.input_sample_rate <= 0:
            self.get_logger().error(f"Invalid input_sample_rate: {self.input_sample_rate}")
            raise ValueError("input_sample_rate must be positive")

        if not (0.0 <= self.speech_threshold <= 1.0):
            self.get_logger().error(f"Invalid speech_threshold: {self.speech_threshold}")
            raise ValueError("speech_threshold must be between 0.0 and 1.0")

        if not (0.0 <= self.neg_threshold <= 1.0):
            self.get_logger().error(f"Invalid neg_threshold: {self.neg_threshold}")
            raise ValueError("neg_threshold must be between 0.0 and 1.0")

        if self.min_silence_duration_ms < 0:
            self.get_logger().error(f"Invalid min_silence_duration_ms: {self.min_silence_duration_ms}")
            raise ValueError("min_silence_duration_ms must be non-negative")

        if self.max_speech_duration_s <= 0:
            self.get_logger().error(f"Invalid max_speech_duration_s: {self.max_speech_duration_s}")
            raise ValueError("max_speech_duration_s must be positive")

        if self.min_speech_duration < 0:
            self.get_logger().error(f"Invalid min_speech_duration: {self.min_speech_duration}")
            raise ValueError("min_speech_duration must be non-negative")

        if self.pre_speech_buffer_ms < 0:
            self.get_logger().error(f"Invalid pre_speech_buffer_ms: {self.pre_speech_buffer_ms}")
            raise ValueError("pre_speech_buffer_ms must be non-negative")

        if self.intensity_threshold < 0:
            self.get_logger().error(f"Invalid intensity_threshold: {self.intensity_threshold}")
            raise ValueError("intensity_threshold must be non-negative")

        self.get_logger().info("All parameters validated successfully")

        package_path = get_package_share_directory('speech_event')
        silero_path = os.path.join(package_path, 'models', 'silero_vad.onnx')

        # Load Silero VAD with logger
        self.silero_model = OnnxWrapper(silero_path, logger=self.get_logger())
        self.get_logger().info(f"Silero VAD loaded for {self.sample_rate}Hz")

        # Load Whisper with optimizations
        self.get_logger().info(f"Loading Whisper model on device: {self.device}...")
        self.whisper = WhisperModel(
            self.whisper_model_id,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=1,  # Reduce overhead for short segments
            cpu_threads=4   # For CPU fallback
        )

        # Check if Whisper is actually using GPU
        if self.device == "cuda":
            if torch.cuda.is_available():
                self.get_logger().info(f"Whisper using GPU: {torch.cuda.get_device_name(0)}")
                self.get_logger().info(f"CUDA version: {torch.version.cuda}")
            else:
                self.get_logger().warning("CUDA requested but not available! Whisper will use CPU")
        else:
            self.get_logger().info(f"Whisper using CPU (device parameter: {self.device})")

        self.get_logger().info("Whisper model loaded.")

        # Warmup Whisper to avoid first-inference latency
        self.warmup_whisper()

        # =====================================================
        # Proper chunk-aligned VAD processing
        # =====================================================
        # Input: 4096 samples @ 48kHz -> 1365 samples @ 16kHz (after resampling)
        # VAD needs: 512 samples per chunk
        # We'll accumulate resampled audio and process in 512-sample chunks

        self.vad_chunk_size = 512  # Silero VAD requirement
        self.vad_pending_buffer = np.zeros(0, dtype=np.float32)  # Accumulates until we have 512 samples

        # =====================================================
        # Pre-speech lookback buffer (ring buffer)
        # =====================================================
        # Keep last N ms of audio to prepend when speech starts
        self.pre_speech_samples = int((self.pre_speech_buffer_ms / 1000.0) * self.sample_rate)
        self.pre_speech_ring = deque(maxlen=self.pre_speech_samples)

        # =====================================================
        # Speech collection buffers
        # =====================================================
        self.speech_buffer = []
        self.speech_active = False
        self.speech_start_time = None

        # Silence tracking (chunk-based)
        self.silence_chunks = 0
        self.min_silence_chunks = int((self.min_silence_duration_ms / 1000.0) * self.sample_rate / self.vad_chunk_size)

        # Max speech duration in chunks (set from max_speech_duration_s parameter)
        self.max_speech_chunks = int(self.max_speech_duration_s * self.sample_rate / self.vad_chunk_size)
        self.speech_chunk_count = 0

        # =====================================================
        # Transcription thread pool
        # =====================================================
        self.transcription_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self.transcription_lock = threading.Lock()
        self.is_transcribing = False
        self.transcribed_text = ""

        # =====================================================
        # Action server state (all guarded by action_server_lock)
        # =====================================================
        self.action_server_lock = threading.Lock()
        # action_started: True while a goal is active; gates audio_callback and
        # concurrent goal rejection.  Always read/written under action_server_lock.
        self.action_started = False
        self.action_goal_complete = threading.Event()
        self.speech_detected_event = threading.Event()
        self._current_goal_handle = None
        self.asr_action_server = None

        # Separate callback groups so the blocking action callback and the
        # audio subscription can run concurrently under MultiThreadedExecutor.
        self._action_cb_group = MutuallyExclusiveCallbackGroup()
        self._audio_cb_group = MutuallyExclusiveCallbackGroup()

        if self.action_server_enabled:
            self.initialize_action_server()

        # Publishers
        self.vad_prob_pub = self.create_publisher(Float32, "/speech_event/vad_speech_prob", 10)
        self.asr_pub = self.create_publisher(String, "/speech_event/text", 10)

        # Subscriber
        self.audio_sub = self.create_subscription(
            AudioBuffer, self.microphone_topic, self.audio_callback, 10,
            callback_group=self._audio_cb_group,
        )

        self.get_logger().info("speech_recognition ready.")
        self.get_logger().info(f"VAD config: speech_thresh={self.speech_threshold}, neg_thresh={self.neg_threshold}, "
                              f"min_silence={self.min_silence_duration_ms}ms ({self.min_silence_chunks} chunks), "
                              f"min_speech={self.min_speech_duration}s, max_speech={self.max_speech_duration_s}s")
        self.get_logger().info(f"Pre-speech buffer: {self.pre_speech_buffer_ms}ms ({self.pre_speech_samples} samples)")
        self.get_logger().info(f"Action server: {'enabled' if self.action_server_enabled else 'disabled'}")

    def warmup_whisper(self):
        """Run a dummy transcription to warm up CUDA kernels."""
        self.get_logger().info("Warming up Whisper...")
        dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence
        try:
            segments, _ = self.whisper.transcribe(
                dummy_audio,
                language=self.language,
                task="transcribe",
                beam_size=1,
                vad_filter=False
            )
            # Consume the generator
            _ = list(segments)
            self.get_logger().info("Whisper warmup complete.")
        except Exception as e:
            self.get_logger().warning(f"Whisper warmup failed (non-critical): {e}")

    # =========================================================================
    # Action Server
    # =========================================================================
    def initialize_action_server(self):
        """Initialize ROS2 Action Server for speech recognition."""
        self.asr_action_server = ActionServer(
            self,
            SpeechRecognition,
            '/speech_recognition_action',
            self.execute_asr_action_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._action_cb_group,
        )

    def cancel_callback(self, goal_handle):
        """Accept cancellation requests immediately."""
        self.get_logger().info(f"Cancel request received for goal {goal_handle.goal_id}.")
        return CancelResponse.ACCEPT

    def execute_asr_action_callback(self, goal_handle):
        """Handle ASR action goal.

        goal.wait defines how long (seconds) to wait for speech to BEGIN.
        Once speech is detected, recording continues until the speaker stops
        (VAD silence) or max_speech_duration_s from the config is reached.
        """
        # --- Reject concurrent goals ---
        with self.action_server_lock:
            if self.action_started:
                self.get_logger().warning("Rejecting goal: a goal is already being processed.")
                goal_handle.abort()
                return SpeechRecognition.Result()

            self.action_started = True
            self._current_goal_handle = goal_handle

        self.speech_detected_event.clear()
        self.action_goal_complete.clear()

        self.get_logger().info(
            f"ASR Action Goal received. Waiting up to {goal_handle.request.wait:.1f}s for speech to start."
        )
        self.publish_feedback("waiting")

        def cleanup():
            with self.action_server_lock:
                self.action_started = False
                self._current_goal_handle = None
            self.speech_detected_event.clear()
            self.action_goal_complete.clear()

        # ---- Phase 1: wait for speech onset ----
        onset_deadline = time.monotonic() + goal_handle.request.wait
        speech_started = False

        while time.monotonic() < onset_deadline:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal cancelled while waiting for speech.")
                cleanup()
                goal_handle.canceled()
                return SpeechRecognition.Result()

            if self.speech_detected_event.wait(timeout=0.1):
                speech_started = True
                break

        if not speech_started:
            self.get_logger().warning(
                f"No speech detected within {goal_handle.request.wait:.1f}s. Aborting.")
            cleanup()
            goal_handle.abort()
            return SpeechRecognition.Result()

        self.get_logger().info("Speech detected. Waiting for speaker to finish...")

        # ---- Phase 2: wait for transcription to complete ----
        transcription_timeout = self.max_speech_duration_s
        transcription_deadline = time.monotonic() + transcription_timeout
        completed = False

        while time.monotonic() < transcription_deadline:
            if goal_handle.is_cancel_requested:
                self.get_logger().info("Goal cancelled during transcription.")
                cleanup()
                goal_handle.canceled()
                return SpeechRecognition.Result()

            if self.action_goal_complete.wait(timeout=0.1):
                completed = True
                break

        # --- Collect result and clean up ---
        result = SpeechRecognition.Result()
        result.transcription = self.transcribed_text
        cleanup()

        if completed:
            self.get_logger().info(f"ASR Action Goal completed. Transcript: '{result.transcription}'")
            goal_handle.succeed()
        else:
            self.get_logger().warning(
                f"Timed out waiting for transcription after {transcription_timeout:.1f}s."
            )
            goal_handle.abort()

        return result

    def publish_feedback(self, status: str):
        """Publish action feedback. status: 'waiting', 'speech', 'transcribing'."""
        with self.action_server_lock:
            gh = self._current_goal_handle
        if gh is None:
            return
        feedback = SpeechRecognition.Feedback()
        feedback.status = status
        gh.publish_feedback(feedback)

    # =========================================================================
    # Audio Parsing
    # =========================================================================
    def parse_audio_buffer(self, msg):
        """
        Convert naoqi_bridge_msgs/AudioBuffer to single-channel audio at 48kHz.

        Returns:
            tuple: (audio_48k, freq_in) or (None, None) on failure
        """
        try:
            freq_in = int(msg.frequency)
            channel_map = list(msg.channel_map)
            channels = len(channel_map)
            data = np.asarray(msg.data, dtype=np.int16)

            if channels == 0 or data.size == 0:
                return None, None

            # De-interleave
            num_frames = data.size // channels
            if num_frames <= 0:
                return None, None

            frames = data[:num_frames * channels].reshape(num_frames, channels).astype(np.float32) / 32767.0

            # Extract front-left channel (primary microphone)
            def get_chan(enum_val, fallback=None):
                if enum_val in channel_map:
                    idx = channel_map.index(enum_val)
                    return frames[:, idx]
                return np.copy(fallback) if fallback is not None else np.zeros(num_frames, dtype=np.float32)

            FL = get_chan(AudioBuffer.CHANNEL_FRONT_LEFT)

            return FL, freq_in

        except Exception as e:
            self.get_logger().error(f"AudioBuffer parse error: {e}")
            return None, None

    def resample_to_16k(self, audio_48k: np.ndarray, freq_in: int) -> np.ndarray:
        """Resample audio to 16kHz for VAD/Whisper."""
        if freq_in == self.sample_rate:
            return audio_48k

        try:
            gcd = math.gcd(freq_in, self.sample_rate)
            up = self.sample_rate // gcd
            down = freq_in // gcd
            return resample_poly(audio_48k, up, down).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Resample failed ({freq_in}->{self.sample_rate}): {e}")
            return audio_48k

    # =========================================================================
    # Intensity Gate
    # =========================================================================
    def is_intense_enough(self, signal: np.ndarray) -> bool:
        """Check if signal exceeds RMS intensity threshold."""
        rms = float(np.sqrt(np.mean(signal ** 2)))
        return rms >= self.intensity_threshold

    # =========================================================================
    # Audio Callback
    # =========================================================================
    def audio_callback(self, msg: AudioBuffer):
        """
        Process incoming audio:
        1. Parse single-channel audio at 48kHz
        2. Resample to 16kHz
        3. Update pre-speech ring buffer (always)
        4. Intensity gate (bypassed during active speech collection)
        5. VAD + speech collection
        """
        # Action server guard: skip processing when no goal is active
        if self.action_server_enabled:
            with self.action_server_lock:
                active = self.action_started
            if not active:
                return

        # Parse single-channel audio (keep at 48kHz)
        audio_48k, freq_in = self.parse_audio_buffer(msg)
        if audio_48k is None:
            return

        # Resample to 16kHz
        resampled_audio = self.resample_to_16k(audio_48k, freq_in)

        # =====================================================
        # Always update pre-speech ring buffer.
        # This must happen before the intensity gate so that the
        # lookback window contains uninterrupted audio history.
        # =====================================================
        for sample in resampled_audio:
            self.pre_speech_ring.append(sample)

        # =====================================================
        # Intensity gate — only applied when NOT actively
        # collecting speech, to avoid dropping mid-speech frames.
        # =====================================================
        if not self.speech_active and not self.is_intense_enough(audio_48k):
            return

        # =====================================================
        # Accumulate for chunk-aligned VAD processing
        # =====================================================
        self.vad_pending_buffer = np.concatenate([self.vad_pending_buffer, resampled_audio])

        # Process all complete 512-sample chunks
        while len(self.vad_pending_buffer) >= self.vad_chunk_size:
            vad_chunk = self.vad_pending_buffer[:self.vad_chunk_size]
            self.vad_pending_buffer = self.vad_pending_buffer[self.vad_chunk_size:]
            self.process_vad_chunk(vad_chunk)

    # =========================================================================
    # VAD State Machine (unified)
    # =========================================================================
    def process_vad_chunk(self, vad_chunk: np.ndarray):
        """
        Process a single 512-sample VAD chunk through the state machine.
        Works for both action server mode and standalone topic mode.
        """
        speech_prob = self.run_silero_vad(vad_chunk)

        prob_msg = Float32()
        prob_msg.data = float(speech_prob)
        self.vad_prob_pub.publish(prob_msg)

        # Two-threshold hysteresis
        vad_is_speech = speech_prob >= self.speech_threshold
        vad_is_silence = speech_prob < self.neg_threshold

        with self.action_server_lock:
            max_chunks = self.max_speech_chunks

        if not self.speech_active and vad_is_speech:
            # ---- START ----
            self.speech_active = True
            self.speech_start_time = time.time()
            self.silence_chunks = 0
            self.speech_chunk_count = 1

            pre_speech_audio = np.array(list(self.pre_speech_ring), dtype=np.float32)
            self.speech_buffer = [pre_speech_audio, vad_chunk.copy()]

            self.get_logger().info(f"VAD: speech START (prob={speech_prob:.3f}, "
                                  f"pre-buffer={len(pre_speech_audio)} samples)")
            self.speech_detected_event.set()
            self.publish_feedback("speech")

        elif self.speech_active and vad_is_speech:
            # ---- CONTINUE ----
            self.speech_buffer.append(vad_chunk.copy())
            self.silence_chunks = 0
            self.speech_chunk_count += 1
            if self.speech_chunk_count >= max_chunks:
                self.finalize_speech(speech_prob, reason="max_duration")

        elif self.speech_active and vad_is_silence:
            # ---- POSSIBLE END ----
            self.speech_buffer.append(vad_chunk.copy())
            self.silence_chunks += 1
            self.speech_chunk_count += 1
            if self.silence_chunks >= self.min_silence_chunks:
                self.finalize_speech(speech_prob, reason="silence")
            elif self.speech_chunk_count >= max_chunks:
                self.finalize_speech(speech_prob, reason="max_duration")

        elif self.speech_active:
            # ---- IN BETWEEN THRESHOLDS ----
            self.speech_buffer.append(vad_chunk.copy())
            self.speech_chunk_count += 1
            if self.speech_chunk_count >= max_chunks:
                self.finalize_speech(speech_prob, reason="max_duration")

    def finalize_speech(self, speech_prob: float, reason: str):
        """
        Finalize a speech segment and submit transcription to the thread pool.
        Non-blocking: the audio callback returns immediately after this call.
        Result delivery differs by mode:
          - Action server mode: transcribe_worker sets action_goal_complete.
          - Standalone mode:    transcribe_worker publishes to /speech_event/text.
        """
        self.speech_active = False

        if reason == "silence":
            silence_duration_s = self.silence_chunks * self.vad_chunk_size / self.sample_rate
            self.get_logger().info(f"VAD: speech END (prob={speech_prob:.3f}, "
                                  f"silence={silence_duration_s:.2f}s)")
        else:
            self.get_logger().info(f"VAD: speech END (max duration {self.max_speech_duration_s}s reached)")

        self.reset_vad_state()

        if not self.speech_buffer:
            self.get_logger().warning("Empty speech buffer at finalization")
            self.reset_speech_state()
            return

        speech_audio = np.concatenate(self.speech_buffer).astype(np.float32)
        duration_s = len(speech_audio) / self.sample_rate

        if duration_s < self.min_speech_duration:
            self.get_logger().info(f"Ignoring short segment ({duration_s:.2f}s)")
            self.reset_speech_state()
            return

        # Capture action mode flag before resetting state
        with self.action_server_lock:
            in_action_mode = self.action_started

        with self.transcription_lock:
            if self.is_transcribing:
                self.get_logger().warning("Previous transcription still running, skipping...")
                self.reset_speech_state()
                return
            self.is_transcribing = True

        try:
            self.publish_feedback("transcribing")
            self.transcription_executor.submit(
                self.transcribe_worker, speech_audio, duration_s, in_action_mode
            )
        except Exception as e:
            self.get_logger().error(f"Failed to submit transcription task: {e}")
            with self.transcription_lock:
                self.is_transcribing = False

        self.reset_speech_state()

    def reset_speech_state(self):
        """Reset speech collection state."""
        self.speech_buffer = []
        self.silence_chunks = 0
        self.speech_chunk_count = 0
        self.speech_start_time = None

    # =========================================================================
    # Transcription (runs in thread pool, never blocks audio callback)
    # =========================================================================
    def transcribe_worker(self, audio: np.ndarray, duration_s: float, in_action_mode: bool):
        """
        Thread-pool worker: transcribes audio then either signals the waiting
        action callback or publishes the result to the ASR topic.
        """
        try:
            text = self.do_transcribe(audio, duration_s)

            if in_action_mode:
                self.transcribed_text = text
                self.action_goal_complete.set()
            else:
                if text:
                    out = String()
                    out.data = text
                    self.asr_pub.publish(out)

        except Exception as e:
            self.get_logger().error(f"Transcription error: {e}")
            if in_action_mode:
                # Unblock the action callback even on failure so it does not
                # hang until timeout; result will be whatever was last stored.
                self.action_goal_complete.set()
        finally:
            with self.transcription_lock:
                self.is_transcribing = False

    def do_transcribe(self, audio: np.ndarray, duration_s: float) -> str:
        """Run Whisper on a speech segment and return the transcript."""
        self.get_logger().info(f"Running Whisper on {duration_s:.2f}s segment...")
        start_time = time.time()

        segments, info = self.whisper.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6
        )

        text = " ".join([seg.text.strip() for seg in segments]).strip()
        elapsed = time.time() - start_time
        rtf = elapsed / info.duration if info.duration > 0 else 0.0

        self.get_logger().info(f"ASR done. elapsed={elapsed:.3f}s, RTF={rtf:.3f}")
        self.get_logger().info(f"Transcript: '{text}'")

        return text

    # =========================================================================
    # VAD helpers
    # =========================================================================
    def run_silero_vad(self, audio_chunk: np.ndarray) -> float:
        """Run Silero VAD on 512-sample chunk."""
        try:
            if len(audio_chunk) != 512:
                self.get_logger().warning(f"VAD frame size mismatch: {len(audio_chunk)}")
                return 0.0

            speech_prob = self.silero_model(audio_chunk)
            return speech_prob

        except Exception as e:
            self.get_logger().error(f"Silero VAD error: {e}")
            return 0.0

    def reset_vad_state(self):
        """Reset VAD state."""
        self.silero_model.reset_states()

    def destroy_node(self):
        """Clean shutdown."""
        self.get_logger().info("Shutting down transcription executor...")
        self.transcription_executor.shutdown(wait=True)
        super().destroy_node()
