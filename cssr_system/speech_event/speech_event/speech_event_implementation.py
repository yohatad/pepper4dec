"""
speech_event_implementation.py
Implementation of 4-microphone sound localization with beamforming and Whisper ASR

Author: Yohannes Tadesse Haile
Date: November 8, 2025
Version: v2.0

Copyright (C) 2023 CSSR4Africa Consortium

This program comes with ABSOLUTELY NO WARRANTY.
"""

import math
import os
import yaml
import time
import rclpy
import queue
import threading
import webrtcvad
import numpy as np
import noisereduce as nr
from faster_whisper import WhisperModel
from datetime import datetime
from rclpy.node import Node
from naoqi_bridge_msgs.msg import AudioBuffer
from std_msgs.msg import Float32, String
from geometry_msgs.msg import Vector3Stamped
from threading import Lock
from ament_index_python.packages import get_package_share_directory
from collections import deque

class SpeechRecognitionNode(Node):
    """
    SpeechRecognitionNode performs 4-microphone sound localization using GCC-PHAT,
    applies beamforming to enhance audio from the detected direction, and uses
    faster-whisper with streaming for real-time automatic speech recognition.
    """

    def __init__(self):
        """Initialize the SpeechRecognitionNode with 4-mic localization and streaming ASR."""
        super().__init__('speech_recognition')

        # Load configuration
        self.config = self.read_yaml_config('speech_event', 'speech_event_configuration.yaml')

        # Audio parameters
        self.sample_rate = 48000
        self.speed_of_sound = 343.0
        self.verbose_mode = bool(self.config.get('verboseMode', False))

        # Microphone array geometry (4 mics in head frame, meters)
        # Order used throughout: [back_left, back_right, front_left, front_right]
        self.mic_positions = np.array([
            [-0.0267,  0.0343, 0.2066],  # Back-left  (RL)
            [-0.0267, -0.0343, 0.2066],  # Back-right (RR)
            [ 0.0313,  0.0343, 0.2066],  # Front-left (FL)
            [ 0.0313, -0.0343, 0.2066]   # Front-right(FR)
        ])

        # Localization buffers
        self.localization_buffer_size = int(self.config.get('localizationBufferSize', 8192))
        self.mic_buffers = [np.zeros(self.localization_buffer_size, dtype=np.float32) for _ in range(4)]
        self.accumulated_samples = 0
        self.lock = Lock()

        # VAD parameters (WebRTC VAD supports 8/16/32/48 kHz; frames 10/20/30 ms)
        self.vad_aggressiveness = int(self.config.get('vadAggressiveness', 2))
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        self.vad_frame_duration = 0.02  # 20ms
        self.vad_frame_size = int(self.sample_rate * self.vad_frame_duration)

        # Intensity threshold
        self.intensity_threshold = float(self.config.get('intensityThreshold', 3.9e-3))

        # Noise reduction parameters
        self.use_noise_reduction = bool(self.config.get('useNoiseReduction', True))
        self.context_duration = float(self.config.get('contextDuration', 2.0))
        self.context_size = int(self.sample_rate * self.context_duration)
        self.context_window = np.zeros(self.context_size, dtype=np.float32)
        self.nr_stationary = bool(self.config.get('stationary', True))
        self.nr_prop_decrease = float(self.config.get('propDecrease', 0.9))

        # faster-whisper ASR parameters
        self.whisper_model_size = str(self.config.get('whisperModelSize', 'tiny'))  # Use tiny for streaming
        self.whisper_language = str(self.config.get('whisperLanguage', 'en'))
        self.whisper_device = str(self.config.get('whisperDevice', 'auto'))
        self.whisper_compute_type = str(self.config.get('whisperComputeType', 'auto'))
        self.beam_size = int(self.config.get('beamSize', 1))  # Lower beam size for streaming
        
        # Streaming ASR parameters
        self.streaming_chunk_duration = float(self.config.get('streamingChunkDuration', 1.0))  # seconds
        self.streaming_chunk_size = int(16000 * self.streaming_chunk_duration)  # 16kHz for Whisper
        self.overlap_duration = float(self.config.get('overlapDuration', 0.5))  # seconds
        self.overlap_size = int(16000 * self.overlap_duration)
        self.min_speech_duration = float(self.config.get('minSpeechDuration', 0.3))  # seconds
        self.silence_threshold = float(self.config.get('silenceThreshold', 0.5))  # seconds before finalizing
        
        # Streaming buffers
        self.streaming_buffer = deque(maxlen=int(16000 * 10))  # Max 10 seconds
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_duration = 0.0
        self.last_speech_time = time.time()
        
        # Partial transcript tracking
        self.current_transcript = ""
        self.finalized_transcript = ""
        self.streaming_lock = Lock()

        # Determine device and compute type automatically if set to 'auto'
        if self.whisper_device == 'auto':
            try:
                import torch
                self.whisper_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.whisper_device = 'cpu'
            self.get_logger().info(f"Auto-detected device: {self.whisper_device}")
        
        if self.whisper_compute_type == 'auto':
            if self.whisper_device == 'cuda':
                self.whisper_compute_type = 'float16'
            else:
                self.whisper_compute_type = 'int8'
            self.get_logger().info(f"Auto-selected compute type: {self.whisper_compute_type}")

        # Load faster-whisper model
        self.get_logger().info(f"Loading faster-whisper model: {self.whisper_model_size} on {self.whisper_device}")
        try:
            self.whisper_model = WhisperModel(
                self.whisper_model_size,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
                num_workers=1
            )
            self.get_logger().info("faster-whisper model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load faster-whisper model: {e}")
            raise

        # Current sound direction (for beamforming)
        self.current_azimuth = 0.0
        self.current_elevation = 0.0
        self.direction_lock = Lock()

        # Beamforming parameters
        self.use_beamforming = bool(self.config.get('useBeamforming', True))

        # Audio timeout monitoring
        self.last_audio_time = self.get_clock().now()
        self.audio_timeout = float(self.config.get('audioTimeout', 5))
        self.received_first_audio = False

        # Get microphone topic
        microphone_topic = self.extract_topics('Microphone')
        if not microphone_topic:
            self.get_logger().error("Microphone topic not found in topic file.")
            raise ValueError("Missing microphone topic configuration.")

        # ROS2 Subscribers and Publishers
        self.audio_sub = self.create_subscription(
            AudioBuffer,
            microphone_topic,
            self.audio_callback,
            10
        )
        self.get_logger().info(f"Subscribed to {microphone_topic}")

        # Publishers
        self.direction_pub = self.create_publisher(Vector3Stamped, '/speech/sound_direction', 10)
        self.azimuth_pub = self.create_publisher(Float32, '/speech/azimuth', 10)
        self.transcript_pub = self.create_publisher(String, '/speech/transcript', 10)
        self.partial_transcript_pub = self.create_publisher(String, '/speech/partial_transcript', 10)
        self.confidence_pub = self.create_publisher(Float32, '/speech/confidence', 10)

        # Start monitoring threads
        self.start_timeout_monitor()
        self.start_streaming_asr_thread()

        self.get_logger().info("Speech Recognition Node initialized with streaming ASR")

    # ------------------------ I/O + config helpers ------------------------

    def read_yaml_config(self, package_name, config_file):
        """Read YAML configuration file."""
        try:
            package_path = get_package_share_directory(package_name)
            config_path = os.path.join(package_path, 'config', config_file)
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    data = yaml.safe_load(file)
                    return data if data is not None else {}
            else:
                self.get_logger().warning(f"Configuration file not found: {config_path}, using defaults")
                return {}
        except Exception as e:
            self.get_logger().error(f"Error reading YAML config: {e}")
            return {}

    def extract_topics(self, topic_key):
        """Extract topic name from topics YAML file."""
        try:
            package_path = get_package_share_directory('speech_event')
            config_path = os.path.join(package_path, 'data', 'pepper_topics.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    topics_data = yaml.safe_load(file)
                    return topics_data.get(topic_key)
            else:
                self.get_logger().error(f"Topics file not found: {config_path}")
        except Exception as e:
            self.get_logger().error(f"Error reading topics file: {e}")
        return None

    def start_timeout_monitor(self):
        """Monitor for audio stream timeouts."""
        def monitor():
            while rclpy.ok():
                if self.received_first_audio:
                    time_since_last = (self.get_clock().now() - self.last_audio_time).nanoseconds / 1e9
                    if time_since_last > self.audio_timeout:
                        self.get_logger().warning(f"No audio for {self.audio_timeout}s. Shutting down.")
                        rclpy.shutdown()
                        break
                time.sleep(1.0)
        threading.Thread(target=monitor, daemon=True).start()

    # ------------------------ Audio parsing + preprocessing ------------------------

    def _parse_audio_buffer(self, msg):
        """
        Convert naoqi_bridge_msgs/AudioBuffer to 4 float32 channel arrays
        ordered as [back_left, back_right, front_left, front_right].
        Resamples to self.sample_rate if needed.

        Returns:
            list[np.ndarray] or None on failure
        """
        try:
            # Metadata
            freq_in = int(msg.frequency)
            channel_map = list(msg.channel_map)
            channels = len(channel_map)
            data = np.asarray(msg.data, dtype=np.int16)

            if channels == 0 or data.size == 0:
                self.get_logger().warning(
                    f"AudioBuffer empty: channels={channels}, data_len={data.size}"
                )
                return None

            # De-interleave
            num_frames = data.size // channels
            if num_frames <= 0:
                self.get_logger().warning("AudioBuffer has no complete frames.")
                return None
            frames = data[:num_frames * channels].reshape(num_frames, channels).astype(np.float32) / 32767.0

            # Map enums -> indices using channel_map
            def get_chan(enum_val, fallback=None):
                if enum_val in channel_map:
                    idx = channel_map.index(enum_val)
                    return frames[:, idx]
                return np.copy(fallback) if fallback is not None else np.zeros(num_frames, dtype=np.float32)

            # Use constants from the message class
            FL = get_chan(AudioBuffer.CHANNEL_FRONT_LEFT)
            FR = get_chan(AudioBuffer.CHANNEL_FRONT_RIGHT)
            RL = get_chan(AudioBuffer.CHANNEL_REAR_LEFT,  FL)
            RR = get_chan(AudioBuffer.CHANNEL_REAR_RIGHT, FR)

            # Reorder to match mic_positions: [back_left, back_right, front_left, front_right]
            mic_signals = [RL, RR, FL, FR]

            # Resample if needed
            if freq_in != self.sample_rate:
                try:
                    from scipy.signal import resample_poly
                    gcd = math.gcd(freq_in, self.sample_rate)
                    up = self.sample_rate // gcd
                    down = freq_in // gcd
                    mic_signals = [resample_poly(sig, up, down).astype(np.float32) for sig in mic_signals]
                except Exception as e:
                    self.get_logger().warning(f"Resample (polyphase) failed ({freq_in}->{self.sample_rate}): {e}")
                    try:
                        from scipy import signal as scipy_signal
                        tgt_len = int(len(mic_signals[0]) * (self.sample_rate / float(freq_in)))
                        mic_signals = [scipy_signal.resample(sig, tgt_len).astype(np.float32) for sig in mic_signals]
                    except Exception as e2:
                        self.get_logger().error(f"Resample fallback failed: {e2}")
            return mic_signals

        except Exception as e:
            self.get_logger().error(f"AudioBuffer parse error: {e}")
            return None

    def voice_detected(self, audio_frame):
        """Use VAD to detect speech presence."""
        try:
            speech_frames = 0
            total_frames = 0
            
            for start in range(0, len(audio_frame) - self.vad_frame_size + 1, self.vad_frame_size):
                frame = audio_frame[start:start + self.vad_frame_size]
                frame_bytes = (frame * 32767).astype(np.int16).tobytes()
                if self.vad.is_speech(frame_bytes, self.sample_rate):
                    speech_frames += 1
                total_frames += 1
            
            # Return True if at least 30% of frames contain speech
            return total_frames > 0 and (speech_frames / total_frames) > 0.3
        except Exception as e:
            self.get_logger().warning(f"VAD error: {e}")
            return False

    def is_intense_enough(self, signal_data):
        """Check if signal exceeds intensity threshold (RMS)."""
        intensity = float(np.sqrt(np.mean(signal_data ** 2)))
        return intensity >= self.intensity_threshold

    def apply_noise_reduction(self, signal):
        """Apply noise reduction using rolling context window."""
        if not self.use_noise_reduction:
            return signal

        try:
            block_size = len(signal)
            if block_size <= 0:
                return signal

            # Roll and append current block into context
            if block_size > self.context_size:
                self.context_size = block_size
                self.context_window = np.zeros(self.context_size, dtype=np.float32)

            self.context_window = np.roll(self.context_window, -block_size)
            self.context_window[-block_size:] = signal

            reduced_context = nr.reduce_noise(
                y=self.context_window,
                sr=self.sample_rate,
                stationary=self.nr_stationary,
                prop_decrease=self.nr_prop_decrease
            )

            return reduced_context[-block_size:]
        except Exception as e:
            self.get_logger().error(f"Noise reduction error: {e}")
            return signal

    # ------------------------ Localization + beamforming ------------------------

    def gcc_phat(self, sig1, sig2):
        """GCC-PHAT algorithm for time delay estimation."""
        try:
            n = sig1.shape[0] + sig2.shape[0]

            # FFT
            SIG1 = np.fft.rfft(sig1, n=n)
            SIG2 = np.fft.rfft(sig2, n=n)

            # Cross-correlation in frequency domain
            R = SIG1 * np.conj(SIG2)

            # Phase transform
            R /= (np.abs(R) + 1e-10)

            # IFFT
            cc = np.fft.irfft(R, n=n)

            # Find peak
            max_shift = int(n / 2)
            cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
            shift = int(np.argmax(np.abs(cc)) - max_shift)

            return shift / float(self.sample_rate)
        except Exception as e:
            self.get_logger().error(f"GCC-PHAT error: {e}")
            return 0.0

    def localize_4mic(self, mic_signals):
        """Perform 4-microphone sound source localization using TDOA."""
        try:
            num_mics = 4
            tdoa_matrix = np.zeros((num_mics, num_mics), dtype=np.float32)

            for i in range(num_mics):
                for j in range(i + 1, num_mics):
                    tdoa = self.gcc_phat(mic_signals[i], mic_signals[j])
                    tdoa_matrix[i, j] = tdoa
                    tdoa_matrix[j, i] = -tdoa

            azimuth, elevation = self.estimate_direction_from_tdoa(tdoa_matrix)

            # Confidence from TDOA consistency
            tdoa_variance = float(np.var(tdoa_matrix[np.triu_indices(num_mics, k=1)]))
            confidence = 1.0 / (1.0 + tdoa_variance * 1e6)

            return azimuth, elevation, confidence

        except Exception as e:
            self.get_logger().error(f"4-mic localization error: {e}")
            return 0.0, 0.0, 0.0

    def estimate_direction_from_tdoa(self, tdoa_matrix):
        """Estimate sound direction from TDOA measurements using least squares."""
        from scipy.optimize import minimize

        def cost_function(angles):
            azimuth, elevation = angles

            # Source direction unit vector
            source_vec = np.array([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation)
            ])

            # Predicted TDOAs
            predicted_tdoa = np.zeros((4, 4), dtype=np.float32)
            for i in range(4):
                for j in range(i + 1, 4):
                    dist_i = float(np.dot(self.mic_positions[i], source_vec))
                    dist_j = float(np.dot(self.mic_positions[j], source_vec))
                    predicted_tdoa[i, j] = (dist_j - dist_i) / self.speed_of_sound
                    predicted_tdoa[j, i] = -predicted_tdoa[i, j]

            # MSE
            error = float(np.sum((tdoa_matrix - predicted_tdoa) ** 2))
            return error

        # Initial guess: front
        initial_guess = [0.0, 0.0]

        result = minimize(
            cost_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=[(-np.pi, np.pi), (-np.pi/4, np.pi/4)]
        )

        return float(result.x[0]), float(result.x[1])

    def apply_beamforming(self, mic_signals, azimuth, elevation):
        """Apply delay-and-sum beamforming to enhance audio from target direction."""
        if not self.use_beamforming:
            return mic_signals[2]

        try:
            # Target direction unit vector
            target_direction = np.array([
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation)
            ])

            # Calculate per-mic delays
            delays = np.zeros(4, dtype=np.float32)
            for i in range(4):
                distance = float(np.dot(self.mic_positions[i], target_direction))
                delays[i] = distance / self.speed_of_sound

            # Normalize delays
            delays -= delays.min()
            delay_samples = (delays * self.sample_rate).astype(int)

            # Apply delays and sum
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
            beamformed = np.clip(beamformed, -1.0, 1.0)

            return beamformed

        except Exception as e:
            self.get_logger().error(f"Beamforming error: {e}")
            return mic_signals[2]

    # ------------------------ Streaming ASR ------------------------

    def start_streaming_asr_thread(self):
        """Start background thread for streaming ASR processing."""
        def streaming_worker():
            while rclpy.ok():
                try:
                    self.process_streaming_asr()
                    time.sleep(0.1)  # Check every 100ms
                except Exception as e:
                    self.get_logger().error(f"Streaming ASR thread error: {e}")
        
        threading.Thread(target=streaming_worker, daemon=True).start()
        self.get_logger().info("Streaming ASR thread started")

    def process_streaming_asr(self):
        """Process audio buffer for streaming transcription."""
        with self.streaming_lock:
            buffer_length = len(self.streaming_buffer)
            
            if buffer_length < self.streaming_chunk_size:
                return
            
            # Get chunk with overlap
            chunk_start = max(0, buffer_length - self.streaming_chunk_size - self.overlap_size)
            audio_chunk = np.array(list(self.streaming_buffer)[chunk_start:])
            
            # Check if we have speech
            has_speech = self.voice_detected(audio_chunk)
            current_time = time.time()
            
            if has_speech:
                self.is_speaking = True
                self.last_speech_time = current_time
                self.silence_duration = 0.0
                
                # Add to speech buffer
                if len(self.speech_buffer) == 0:
                    self.speech_buffer = audio_chunk.tolist()
                else:
                    # Avoid duplicates from overlap
                    new_samples = audio_chunk[self.overlap_size:]
                    self.speech_buffer.extend(new_samples.tolist())
                
                # Transcribe if we have enough speech
                if len(self.speech_buffer) >= self.streaming_chunk_size:
                    self.transcribe_partial()
            
            elif self.is_speaking:
                # Track silence
                self.silence_duration = current_time - self.last_speech_time
                
                # Finalize if silence threshold exceeded
                if self.silence_duration >= self.silence_threshold:
                    if len(self.speech_buffer) >= int(16000 * self.min_speech_duration):
                        self.finalize_transcript()
                    
                    # Reset
                    self.is_speaking = False
                    self.speech_buffer = []
                    self.current_transcript = ""

    def transcribe_partial(self):
        """Transcribe current speech buffer (partial result)."""
        try:
            if len(self.speech_buffer) == 0:
                return
            
            audio_array = np.array(self.speech_buffer, dtype=np.float32)
            
            # Transcribe
            segments, info = self.whisper_model.transcribe(
                audio_array,
                language=self.whisper_language,
                beam_size=self.beam_size,
                vad_filter=False,  # We already did VAD
                task='transcribe',
                condition_on_previous_text=True,
                initial_prompt=self.finalized_transcript  # Use context
            )
            
            # Collect text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)
            
            partial_text = ' '.join(text_parts).strip()
            
            if partial_text and partial_text != self.current_transcript:
                self.current_transcript = partial_text
                
                # Publish partial transcript
                msg = String()
                msg.data = partial_text
                self.partial_transcript_pub.publish(msg)
                
                if self.verbose_mode:
                    self.get_logger().info(f"Partial: {partial_text}")
        
        except Exception as e:
            self.get_logger().error(f"Partial transcription error: {e}")

    def finalize_transcript(self):
        """Finalize and publish complete transcript."""
        try:
            if len(self.speech_buffer) == 0:
                return
            
            audio_array = np.array(self.speech_buffer, dtype=np.float32)
            
            # Final transcription with higher quality settings
            segments, info = self.whisper_model.transcribe(
                audio_array,
                language=self.whisper_language,
                beam_size=5,  # Higher for final
                vad_filter=False,
                task='transcribe',
                condition_on_previous_text=True,
                initial_prompt=self.finalized_transcript
            )
            
            # Collect segments
            text_parts = []
            confidences = []
            
            for segment in segments:
                text_parts.append(segment.text)
                confidence = np.exp(segment.avg_logprob)
                confidences.append(confidence)
            
            final_text = ' '.join(text_parts).strip()
            
            if final_text:
                self.get_logger().info(f"Final: {final_text}")
                
                # Update finalized transcript for context
                self.finalized_transcript = (self.finalized_transcript + " " + final_text).strip()
                
                # Keep only last 100 words for context
                words = self.finalized_transcript.split()
                if len(words) > 100:
                    self.finalized_transcript = ' '.join(words[-100:])
                
                # Publish final transcript
                msg = String()
                msg.data = final_text
                self.transcript_pub.publish(msg)
                
                # Publish confidence
                avg_confidence = float(np.mean(confidences)) if confidences else 0.5
                conf_msg = Float32()
                conf_msg.data = avg_confidence
                self.confidence_pub.publish(conf_msg)
                
                if self.verbose_mode:
                    self.get_logger().info(f"Confidence: {avg_confidence:.2f}")
        
        except Exception as e:
            self.get_logger().error(f"Final transcription error: {e}")

    # ------------------------ Audio callback ------------------------

    def audio_callback(self, msg):
        """Process incoming AudioBuffer message."""
        try:
            self.last_audio_time = self.get_clock().now()

            if not self.received_first_audio:
                self.received_first_audio = True
                self.get_logger().info("First audio received")

            # Parse AudioBuffer -> 4 channels
            mic_signals = self._parse_audio_buffer(msg)
            if mic_signals is None:
                return

            back_left, back_right, front_left, front_right = mic_signals

            # Intensity gate
            if not self.is_intense_enough(front_left):
                return

            # Update localization buffers
            with self.lock:
                data_length = int(len(front_left))
                if self.accumulated_samples + data_length <= self.localization_buffer_size:
                    start = self.accumulated_samples
                    end = start + data_length
                    self.mic_buffers[0][start:end] = back_left[:data_length]
                    self.mic_buffers[1][start:end] = back_right[:data_length]
                    self.mic_buffers[2][start:end] = front_left[:data_length]
                    self.mic_buffers[3][start:end] = front_right[:data_length]
                    self.accumulated_samples += data_length
                else:
                    remaining = self.localization_buffer_size - self.accumulated_samples
                    if remaining > 0:
                        self.mic_buffers[0][self.accumulated_samples:] = back_left[:remaining]
                        self.mic_buffers[1][self.accumulated_samples:] = back_right[:remaining]
                        self.mic_buffers[2][self.accumulated_samples:] = front_left[:remaining]
                        self.mic_buffers[3][self.accumulated_samples:] = front_right[:remaining]
                        self.accumulated_samples = self.localization_buffer_size

            # Perform localization when buffer is full
            if self.accumulated_samples >= self.localization_buffer_size:
                azimuth, elevation, confidence = self.localize_4mic(self.mic_buffers)

                # Update current direction
                with self.direction_lock:
                    self.current_azimuth = azimuth
                    self.current_elevation = elevation

                # Publish direction
                self.publish_direction(azimuth, elevation, confidence)

                # Beamforming
                beamformed_audio = self.apply_beamforming(self.mic_buffers, azimuth, elevation)

                # Noise reduction
                beamformed_clean = self.apply_noise_reduction(beamformed_audio)

                # Resample to 16kHz for Whisper and add to streaming buffer
                audio_16k = self.resample_to_16k(beamformed_clean)
                with self.streaming_lock:
                    self.streaming_buffer.extend(audio_16k.tolist())

                # Reset localization buffers
                with self.lock:
                    self.mic_buffers = [np.zeros(self.localization_buffer_size, dtype=np.float32) for _ in range(4)]
                    self.accumulated_samples = 0

        except Exception as e:
            self.get_logger().error(f"Audio callback error: {e}")

    def resample_to_16k(self, audio_signal):
        """Resample audio to 16kHz for Whisper."""
        try:
            if self.sample_rate == 16000:
                return audio_signal
            
            from scipy.signal import resample_poly
            gcd = math.gcd(self.sample_rate, 16000)
            up = 16000 // gcd
            down = self.sample_rate // gcd
            return resample_poly(audio_signal, up, down).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"Resampling error: {e}")
            return audio_signal

    def publish_direction(self, azimuth, elevation, confidence):
        """Publish sound source direction."""
        try:
            stamp = self.get_clock().now().to_msg()

            # Publish direction as unit vector
            direction_msg = Vector3Stamped()
            direction_msg.header.stamp = stamp
            direction_msg.header.frame_id = 'pepper_head'
            direction_msg.vector.x = float(np.cos(elevation) * np.cos(azimuth))
            direction_msg.vector.y = float(np.cos(elevation) * np.sin(azimuth))
            direction_msg.vector.z = float(np.sin(elevation))
            self.direction_pub.publish(direction_msg)

            # Publish azimuth
            azimuth_msg = Float32()
            azimuth_msg.data = float(azimuth)
            self.azimuth_pub.publish(azimuth_msg)

            if self.verbose_mode:
                self.get_logger().info(
                    f"Direction: azimuth={np.degrees(azimuth):.1f}°, "
                    f"elevation={np.degrees(elevation):.1f}°, "
                    f"confidence={confidence:.2f}"
                )

        except Exception as e:
            self.get_logger().error(f"Direction publishing error: {e}")

    def destroy_node(self):
        """Cleanup on shutdown."""
        self.get_logger().info("Shutting down Speech Recognition Node")
        super().destroy_node()