import math
import time
import numpy as np
import torch
import rclpy
import os
import onnxruntime

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from naoqi_bridge_msgs.msg import AudioBuffer
from faster_whisper import WhisperModel
from scipy import signal as scipy_signal
from scipy.signal import resample_poly

class OnnxWrapper():
    def __init__(self, path, force_onnx_cpu=False):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        if force_onnx_cpu and 'CPUExecutionProvider' in onnxruntime.get_available_providers():
            self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=opts)
        else:
            self.session = onnxruntime.InferenceSession(path, sess_options=opts)

        # Always work with 16kHz (since we resample from 48kHz)
        self.sample_rate = 16000
        self.reset_states()

    def reset_states(self, batch_size=1):
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
        # Convert to torch tensor
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        
        # Ensure 2D: (batch_size, samples)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Reset states if batch size changed
        if batch_size != self._last_batch_size:
            self.reset_states(batch_size)
        
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
        self.declare_parameter("device", "cuda")
        self.declare_parameter("compute_type", "float16")
        self.declare_parameter("language", "en")
        self.declare_parameter("whisper_model_id", "deepdml/faster-whisper-large-v3-turbo-ct2")
        
        self.declare_parameter("speech_threshold", 0.7)
        self.declare_parameter("neg_threshold", 0.35)
        self.declare_parameter("min_silence_duration_ms", 300)
        self.declare_parameter("max_vad_window_s", 2.0)
        self.declare_parameter("min_speech_duration", 0.3)
        
        self.declare_parameter("vad_channel_index", 0)
        self.declare_parameter("microphone_topic", "/audio")

        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.device = self.get_parameter("device").value
        self.compute_type = self.get_parameter("compute_type").value
        self.language = self.get_parameter("language").value
        self.whisper_model_id = self.get_parameter("whisper_model_id").value
        
        self.speech_threshold = float(self.get_parameter("speech_threshold").value)
        self.neg_threshold = float(self.get_parameter("neg_threshold").value)
        self.min_silence_duration_ms = int(self.get_parameter("min_silence_duration_ms").value)
        self.max_vad_window_s = float(self.get_parameter("max_vad_window_s").value)
        self.min_speech_duration = float(self.get_parameter("min_speech_duration").value)
        
        self.vad_channel_index = int(self.get_parameter("vad_channel_index").value)
        self.microphone_topic = self.get_parameter("microphone_topic").value

        package_path = get_package_share_directory('speech_event')
        silero_path = os.path.join(package_path, 'models', 'silero_vad.onnx')

        # Load Silero VAD
        self.silero_model = OnnxWrapper(silero_path)
        self.get_logger().info(f"Silero VAD loaded for {self.sample_rate}Hz")

        # Load Whisper
        self.get_logger().info("Loading Whisper model...")
        self.whisper = WhisperModel(self.whisper_model_id, device=self.device, compute_type=self.compute_type)
        self.get_logger().info("Whisper model loaded.")

        # Buffers
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.speech_buffer = []
        self.speech_active = False
        
        # Silence tracking
        self.silence_chunks = 0  # Count consecutive silent chunks
        self.min_silence_chunks = int((self.min_silence_duration_ms / 1000.0) * self.sample_rate / 512)  # 512 samples per chunk

        # Publishers
        self.vad_prob_pub = self.create_publisher(Float32, "/vad/speech_prob", 10)
        self.asr_pub = self.create_publisher(String, "/asr/text", 10)

        # Subscriber
        self.audio_sub = self.create_subscription(AudioBuffer, self.microphone_topic, self.audio_callback, 10)

        self.get_logger().info("speech_recognition ready.")
        self.get_logger().info(f"VAD config: speech_thresh={self.speech_threshold}, neg_thresh={self.neg_threshold}, "
                              f"min_silence={self.min_silence_duration_ms}ms ({self.min_silence_chunks} chunks), "
                              f"min_speech={self.min_speech_duration}s")

    def parse_audio_buffer(self, msg):
        """
        Convert naoqi_bridge_msgs/AudioBuffer to 4 float32 channel arrays.
        Resamples from 48kHz to 16kHz.
        
        Returns:
            list[np.ndarray] or None on failure
        """
        try:
            freq_in = int(msg.frequency)
            channel_map = list(msg.channel_map)
            channels = len(channel_map)
            data = np.asarray(msg.data, dtype=np.int16)

            if channels == 0 or data.size == 0:
                return None

            # De-interleave
            num_frames = data.size // channels
            if num_frames <= 0:
                return None
            
            frames = data[:num_frames * channels].reshape(num_frames, channels).astype(np.float32) / 32767.0

            # Extract channels
            def get_chan(enum_val, fallback=None):
                if enum_val in channel_map:
                    idx = channel_map.index(enum_val)
                    return frames[:, idx]
                return np.copy(fallback) if fallback is not None else np.zeros(num_frames, dtype=np.float32)

            FL = get_chan(AudioBuffer.CHANNEL_FRONT_LEFT)
            FR = get_chan(AudioBuffer.CHANNEL_FRONT_RIGHT)
            RL = get_chan(AudioBuffer.CHANNEL_REAR_LEFT, FL)
            RR = get_chan(AudioBuffer.CHANNEL_REAR_RIGHT, FR)

            mic_signals = [RL, RR, FL, FR]

            # Resample from 48kHz to 16kHz
            if freq_in != self.sample_rate:
                try:
                    gcd = math.gcd(freq_in, self.sample_rate)
                    up = self.sample_rate // gcd
                    down = freq_in // gcd
                    mic_signals = [resample_poly(sig, up, down).astype(np.float32) for sig in mic_signals]
                except Exception as e:
                    self.get_logger().error(f"Resample failed ({freq_in}->{self.sample_rate}): {e}")
                    return None
                
            return mic_signals

        except Exception as e:
            self.get_logger().error(f"AudioBuffer parse error: {e}")
            return None

    def audio_callback(self, msg: AudioBuffer):
        """Process incoming audio with VAD and speech collection."""
        # Parse multi-channel audio
        mic_signals = self.parse_audio_buffer(msg)
        if mic_signals is None:
            return
        
        # Select channel for VAD
        selected_channel = mic_signals[self.vad_channel_index]
        
        # Append to rolling buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, selected_channel])
        
        # Keep only last max_vad_window_s seconds
        max_samples = int(self.max_vad_window_s * self.sample_rate)
        if len(self.audio_buffer) > max_samples:
            self.audio_buffer = self.audio_buffer[-max_samples:]
        
        # Need at least 512 samples for VAD
        if len(self.audio_buffer) < 512:
            return
        
        # Run VAD on last 512 samples
        vad_frame = self.audio_buffer[-512:]
        speech_prob = self.run_silero_vad(vad_frame)
        
        # Publish probability
        prob_msg = Float32()
        prob_msg.data = float(speech_prob)
        self.vad_prob_pub.publish(prob_msg)
        
        # Two-threshold system
        vad_is_speech = speech_prob >= self.speech_threshold
        vad_is_silence = speech_prob < self.neg_threshold
        
        # State machine with chunk-based silence tracking
        if not self.speech_active and vad_is_speech:
            # START: Speech detected
            self.speech_active = True
            self.speech_buffer = [selected_channel.copy()]
            self.silence_chunks = 0
            self.get_logger().info(f"VAD: speech START (prob={speech_prob:.3f})")
        
        elif self.speech_active and vad_is_speech:
            # CONTINUE: Still speaking - reset silence counter
            self.speech_buffer.append(selected_channel.copy())
            self.silence_chunks = 0
        
        elif self.speech_active and vad_is_silence:
            # POSSIBLE END: Below negative threshold
            self.speech_buffer.append(selected_channel.copy())
            self.silence_chunks += 1
            
            # Check if silence duration exceeded threshold
            if self.silence_chunks >= self.min_silence_chunks:
                # END: Silence long enough
                self.speech_active = False
                silence_duration_s = self.silence_chunks * 512 / self.sample_rate
                self.get_logger().info(f"VAD: speech END (prob={speech_prob:.3f}, silence={silence_duration_s:.2f}s)")
                
                # Reset VAD state
                self.reset_vad_state()
                
                # Concatenate speech
                speech_audio = np.concatenate(self.speech_buffer).astype(np.float32)
                duration_s = len(speech_audio) / self.sample_rate
                
                # Filter short segments
                if duration_s < self.min_speech_duration:
                    self.get_logger().info(f"Ignoring short segment ({duration_s:.2f}s)")
                    self.speech_buffer = []
                    self.silence_chunks = 0
                    return
                
                # Transcribe
                self.transcribe_segment(speech_audio, duration_s)
                self.speech_buffer = []
                self.silence_chunks = 0
        
        elif self.speech_active:
            # In between thresholds - continue collecting, don't increment silence counter
            self.speech_buffer.append(selected_channel.copy())

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

    def transcribe_segment(self, audio: np.ndarray, duration_s: float):
        """Transcribe speech segment with Whisper."""
        self.get_logger().info(f"Running Whisper on {duration_s:.2f}s segment...")
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

        self.get_logger().info(f"ASR done. elapsed={elapsed:.3f}s, RTF={rtf:.3f}")
        self.get_logger().info(f"Transcript: '{text}'")

        # Only publish non-empty transcripts
        if text:
            out = String()
            out.data = text
            self.asr_pub.publish(out)