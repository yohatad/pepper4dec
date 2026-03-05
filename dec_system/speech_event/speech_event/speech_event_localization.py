"""
sound_localization_node.py
Dedicated sound source localization using pyroomacoustics SRP-PHAT
2D azimuth-only localization for planar microphone arrays

Author: Yohannes Tadesse Haile
Date: January 2026
Version: v1.0 - Azimuth-only

This program comes with ABSOLUTELY NO WARRANTY.
"""

import numpy as np
import pyroomacoustics as pra
import threading
import time
from collections import deque

from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from naoqi_bridge_msgs.msg import AudioBuffer
from visualization_msgs.msg import Marker
from scipy import signal

class SoundLocalizationNode(Node):
    def __init__(self):
        super().__init__("sound_localization")

        # =====================================================
        # Parameters
        # =====================================================
        self.declare_parameter("sample_rate", 48000)
        self.declare_parameter("microphone_topic", "/audio")
        self.declare_parameter("speed_of_sound", 343.0)
        self.declare_parameter("nfft", 1024)
        self.declare_parameter("angular_resolution", 36)
        self.declare_parameter("freq_range_min", 500)
        self.declare_parameter("freq_range_max", 2500)
        self.declare_parameter("num_chunks_for_localization", 6)
        self.declare_parameter("update_rate_hz", 2.0)
        self.declare_parameter("confidence_threshold", 0.15)
        self.declare_parameter("intensity_threshold", 0.001)
        self.declare_parameter("enable_smoothing", True)
        self.declare_parameter("smoothing_window", 5)

        # Get parameters
        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.microphone_topic = self.get_parameter("microphone_topic").value
        self.speed_of_sound = float(self.get_parameter("speed_of_sound").value)
        self.nfft = int(self.get_parameter("nfft").value)
        self.angular_resolution = int(self.get_parameter("angular_resolution").value)
        self.freq_min = int(self.get_parameter("freq_range_min").value)
        self.freq_max = int(self.get_parameter("freq_range_max").value)
        self.num_chunks = int(self.get_parameter("num_chunks_for_localization").value)
        self.update_rate = float(self.get_parameter("update_rate_hz").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.intensity_threshold = float(self.get_parameter("intensity_threshold").value)
        self.enable_smoothing = bool(self.get_parameter("enable_smoothing").value)
        self.smoothing_window = int(self.get_parameter("smoothing_window").value)

        # =====================================================
        # Parameter Validation
        # =====================================================
        if self.sample_rate <= 0:
            self.get_logger().error(f"Invalid sample_rate: {self.sample_rate}")
            raise ValueError("sample_rate must be positive")

        if self.nfft <= 0:
            self.get_logger().error(f"Invalid nfft: {self.nfft}")
            raise ValueError("nfft must be positive")

        if self.nfft & (self.nfft - 1) != 0:
            self.get_logger().warning(
                f"nfft ({self.nfft}) is not a power of 2. This may reduce FFT efficiency."
            )

        if self.angular_resolution <= 0:
            self.get_logger().error(f"Invalid angular_resolution: {self.angular_resolution}")
            raise ValueError("angular_resolution must be positive")

        if self.freq_min < 0 or self.freq_max < 0:
            self.get_logger().error(f"Invalid frequency range: [{self.freq_min}, {self.freq_max}]")
            raise ValueError("Frequency range values must be non-negative")

        if self.freq_min >= self.freq_max:
            self.get_logger().error(
                f"Invalid frequency range: freq_min ({self.freq_min}) >= freq_max ({self.freq_max})"
            )
            raise ValueError("freq_min must be less than freq_max")

        if self.freq_max > self.sample_rate / 2:
            self.get_logger().error(
                f"freq_max ({self.freq_max}) exceeds Nyquist frequency ({self.sample_rate / 2})"
            )
            raise ValueError("freq_max must be less than or equal to sample_rate/2")

        if self.num_chunks <= 0:
            self.get_logger().error(f"Invalid num_chunks_for_localization: {self.num_chunks}")
            raise ValueError("num_chunks_for_localization must be positive")

        if self.update_rate <= 0:
            self.get_logger().error(f"Invalid update_rate_hz: {self.update_rate}")
            raise ValueError("update_rate_hz must be positive")

        if not 0.0 <= self.confidence_threshold <= 1.0:
            self.get_logger().error(f"Invalid confidence_threshold: {self.confidence_threshold}")
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")

        if self.intensity_threshold < 0:
            self.get_logger().error(f"Invalid intensity_threshold: {self.intensity_threshold}")
            raise ValueError("intensity_threshold must be non-negative")

        if self.smoothing_window <= 0:
            self.get_logger().error(f"Invalid smoothing_window: {self.smoothing_window}")
            raise ValueError("smoothing_window must be positive")

        if self.speed_of_sound <= 0:
            self.get_logger().error(f"Invalid speed_of_sound: {self.speed_of_sound}")
            raise ValueError("speed_of_sound must be positive")

        self.get_logger().info("All parameters validated successfully")

        # =====================================================
        # Microphone Array Geometry (Pepper, in meters)
        # =====================================================
        self.mic_positions = np.array([
            [-0.0267,  0.0343, 0.2066],  # Rear Left (RL)
            [-0.0267, -0.0343, 0.2066],  # Rear Right (RR)
            [ 0.0313,  0.0343, 0.2066],  # Front Left (FL)
            [ 0.0313, -0.0343, 0.2066]   # Front Right (FR)
        ]).T

        # Calculate array characteristics
        mic_spacing = np.linalg.norm(self.mic_positions[:, 0] - self.mic_positions[:, 2])
        max_freq_unambiguous = self.speed_of_sound / (2 * mic_spacing)
        
        self.get_logger().info(f"Microphone array characteristics:")
        self.get_logger().info(f"  Planar array (all mics at same height - azimuth-only localization)")
        self.get_logger().info(f"  Spacing: {mic_spacing*100:.2f} cm")
        self.get_logger().info(f"  Max unambiguous frequency: {max_freq_unambiguous:.0f} Hz")
        if self.freq_max > max_freq_unambiguous:
            self.get_logger().warning(
                f"  WARNING: freq_max ({self.freq_max} Hz) exceeds safe limit! "
                f"May cause spatial aliasing."
            )

        # =====================================================
        # Audio Buffer Configuration
        # =====================================================
        self.chunk_size = 4096  # Pepper sends 4096 samples per chunk (~85ms at 48kHz)
        
        # Ensure enough samples for robust STFT
        min_required_samples = 8 * self.nfft
        min_chunks_needed = int(np.ceil(min_required_samples / self.chunk_size))
        actual_num_chunks = max(self.num_chunks, min_chunks_needed)
        
        self.window_samples = actual_num_chunks * self.chunk_size
        self.window_duration_ms = (self.window_samples / self.sample_rate) * 1000
        
        self.get_logger().info(f"Audio buffer configuration:")
        self.get_logger().info(f"  Chunk size: {self.chunk_size} samples (~85ms)")
        self.get_logger().info(f"  NFFT: {self.nfft}, minimum required: {min_required_samples} samples")
        self.get_logger().info(f"  Using {actual_num_chunks} chunks")
        self.get_logger().info(f"  Window size: {self.window_samples} samples ({self.window_duration_ms:.1f}ms)")
        
        # Buffer stores more than needed for continuous processing
        self.audio_buffer = [deque(maxlen=self.window_samples * 2) for _ in range(4)]
        self.buffer_lock = threading.Lock()

        # =====================================================
        # Localization State
        # =====================================================
        self.last_localization_time = 0.0
        self.min_localization_interval = 1.0 / self.update_rate
        
        # Smoothing buffer (azimuth only)
        self.azimuth_history = deque(maxlen=self.smoothing_window)
        
        # Current direction
        self.current_azimuth = 0.0
        self.current_confidence = 0.0
        self.direction_lock = threading.Lock()

        # =====================================================
        # Initialize SRP-PHAT DOA Estimator
        # =====================================================
        azimuth_grid = np.linspace(0, 2*np.pi, self.angular_resolution, endpoint=False)
        colatitude_grid = np.array([np.pi/2])  # 2D horizontal plane only
        
        self.doa = pra.doa.SRP(
            L=self.mic_positions,
            fs=self.sample_rate,
            nfft=self.nfft,
            c=self.speed_of_sound,
            num_src=1,
            mode='far',
            azimuth=azimuth_grid,
            colatitude=colatitude_grid
        )
        
        self.get_logger().info(f"SRP-PHAT localization initialized:")
        self.get_logger().info(f"  Sample rate: {self.sample_rate} Hz")
        self.get_logger().info(f"  NFFT: {self.nfft}")
        self.get_logger().info(f"  Angular resolution: {self.angular_resolution} angles ({360/self.angular_resolution:.1f}° per step)")
        self.get_logger().info(f"  Frequency range: {self.freq_min}-{self.freq_max} Hz")
        self.get_logger().info(f"  Update rate: {self.update_rate} Hz")
        self.get_logger().info(f"  Confidence threshold: {self.confidence_threshold}")

        # =====================================================
        # Publishers
        # =====================================================
        self.direction_pub = self.create_publisher(Vector3Stamped, "/sound_localization/direction", 10)
        self.azimuth_pub = self.create_publisher(Float32, "/sound_localization/azimuth", 10)
        self.confidence_pub = self.create_publisher(Float32, "/sound_localization/confidence", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/sound_localization/source_pose", 10)
        self.marker_pub = self.create_publisher(Marker, "/sound_localization/visualization", 10)

        # =====================================================
        # Subscriber
        # =====================================================
        self.audio_sub = self.create_subscription(AudioBuffer, self.microphone_topic, 
            self.audio_callback, 10)

        self.get_logger().info("Sound localization node ready.")

    def parse_audio_buffer(self, msg):
        """Parse AudioBuffer message to 4-channel array."""
        try:
            freq_in = int(msg.frequency)
            channel_map = list(msg.channel_map)
            channels = len(channel_map)
            data = np.asarray(msg.data, dtype=np.int16)

            if channels == 0 or data.size == 0:
                return None, None

            num_frames = data.size // channels
            if num_frames <= 0:
                return None, None
            
            # Convert to float32 in range [-1, 1]
            frames = data[:num_frames * channels].reshape(num_frames, channels).astype(np.float32) / 32767.0

            def get_chan(enum_val, fallback=None):
                if enum_val in channel_map:
                    idx = channel_map.index(enum_val)
                    return frames[:, idx]
                return np.copy(fallback) if fallback is not None else np.zeros(num_frames, dtype=np.float32)

            # Extract channels in order: RL, RR, FL, FR
            FL = get_chan(AudioBuffer.CHANNEL_FRONT_LEFT)
            FR = get_chan(AudioBuffer.CHANNEL_FRONT_RIGHT)
            RL = get_chan(AudioBuffer.CHANNEL_REAR_LEFT, FL)
            RR = get_chan(AudioBuffer.CHANNEL_REAR_RIGHT, FR)

            mic_signals = [RL, RR, FL, FR]
            return mic_signals, freq_in

        except Exception as e:
            self.get_logger().error(f"AudioBuffer parse error: {e}")
            return None, None

    def check_intensity(self, mic_signals: list) -> bool:
        """Check if audio has sufficient energy for localization."""
        # Use front-left microphone for intensity check
        rms = float(np.sqrt(np.mean(mic_signals[2] ** 2)))
        return rms >= self.intensity_threshold

    def audio_callback(self, msg: AudioBuffer):
        """Process incoming audio chunks and update localization buffer."""
        mic_signals, freq_in = self.parse_audio_buffer(msg)
        if mic_signals is None:
            return

        if not self.check_intensity(mic_signals):
            return

        # Accumulate samples in buffer
        with self.buffer_lock:
            for i, signal_data in enumerate(mic_signals):
                self.audio_buffer[i].extend(signal_data)

        # Check if we should perform localization
        current_time = time.time()
        time_since_last = current_time - self.last_localization_time
        
        chunks_accumulated = len(self.audio_buffer[0]) >= self.window_samples
        time_ready = time_since_last >= self.min_localization_interval
        
        if chunks_accumulated and time_ready:
            self.perform_localization()
            self.last_localization_time = current_time

    def perform_localization(self):
        """Perform SRP-PHAT sound source localization on accumulated audio."""
        with self.buffer_lock:
            if len(self.audio_buffer[0]) < self.window_samples:
                return
            
            # Extract exactly window_samples from the end of buffer
            multichannel_audio = np.array([
                list(self.audio_buffer[i])[-self.window_samples:] for i in range(4)
            ], dtype=np.float32)

        # Verify shape
        expected_shape = (4, self.window_samples)
        if multichannel_audio.shape != expected_shape:
            self.get_logger().error(
                f"Audio shape mismatch: got {multichannel_audio.shape}, expected {expected_shape}"
            )
            return

        try:
            start_time = time.time()
            
            # Compute STFT for each microphone channel
            stft_data = []
            for ch in range(4):
                f, t, Zxx = signal.stft(
                    multichannel_audio[ch],
                    fs=self.sample_rate,
                    window='hann',
                    nperseg=self.nfft,
                    noverlap=self.nfft // 2,
                    nfft=self.nfft,
                    boundary=None,
                    padded=False
                )
                stft_data.append(Zxx)
            
            # Stack into shape (n_mics, n_freqs, n_frames)
            X = np.array(stft_data, dtype=np.complex128)
            
            # Verify STFT shape
            if X.shape[1] != self.nfft // 2 + 1:
                self.get_logger().error(
                    f"STFT frequency bins mismatch: got {X.shape[1]}, expected {self.nfft // 2 + 1}"
                )
                return
            
            # Run SRP-PHAT localization
            self.doa.locate_sources(
                X,
                num_src=1,
                freq_range=[self.freq_min, self.freq_max]
            )
            
            if len(self.doa.azimuth_recon) > 0:
                azimuth_rad = self.doa.azimuth_recon[0]

                # Calculate confidence from spatial spectrum
                spatial_spectrum = self.doa.grid.values
                max_power = np.max(spatial_spectrum)
                mean_power = np.mean(spatial_spectrum)
                confidence = (max_power - mean_power) / (mean_power + 1e-6)
                confidence = min(confidence, 1.0)

                elapsed = time.time() - start_time

                # Convert to degrees
                azimuth_deg = np.degrees(azimuth_rad) % 360

                # Apply smoothing if enabled
                if self.enable_smoothing:
                    self.azimuth_history.append(azimuth_deg)
                    azimuth_deg = self.circular_mean(list(self.azimuth_history))

                # Update state
                with self.direction_lock:
                    self.current_azimuth = azimuth_deg
                    self.current_confidence = confidence

                # Publish if confidence exceeds threshold
                if confidence >= self.confidence_threshold:
                    self.publish_results(azimuth_deg, confidence)

                    self.get_logger().info(
                        f"Sound at {azimuth_deg:.1f}° ({self.get_direction_name(azimuth_deg)}), "
                        f"confidence={confidence:.3f}, processing={elapsed*1000:.1f}ms"
                    )
                else:
                    self.get_logger().debug(
                        f"Low confidence: {confidence:.3f} < {self.confidence_threshold}"
                    )
            else:
                self.get_logger().debug("No source detected")

        except Exception as e:
            self.get_logger().error(f"Localization error: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def circular_mean(self, angles_deg: list) -> float:
        """Compute circular mean of angles (handles 0°/360° wraparound)."""
        angles_rad = np.radians(angles_deg)
        sin_sum = np.sum(np.sin(angles_rad))
        cos_sum = np.sum(np.cos(angles_rad))
        mean_rad = np.arctan2(sin_sum, cos_sum)
        return float(np.degrees(mean_rad) % 360)

    def publish_results(self, azimuth_deg: float, confidence: float):
        """Publish localization results (azimuth-only in Head frame)."""
        stamp = self.get_clock().now().to_msg()

        azimuth_rad = np.radians(azimuth_deg)

        # Calculate 2D direction vector (in horizontal plane of Head frame)
        direction_x = np.cos(azimuth_rad)
        direction_y = np.sin(azimuth_rad)
        direction_z = 0.0  # Horizontal plane only

        # Publish direction vector
        direction_msg = Vector3Stamped()
        direction_msg.header.stamp = stamp
        direction_msg.header.frame_id = 'Head'
        direction_msg.vector.x = float(direction_x)
        direction_msg.vector.y = float(direction_y)
        direction_msg.vector.z = float(direction_z)
        self.direction_pub.publish(direction_msg)

        # Publish azimuth
        azimuth_msg = Float32()
        azimuth_msg.data = float(azimuth_deg)
        self.azimuth_pub.publish(azimuth_msg)

        # Publish confidence
        confidence_msg = Float32()
        confidence_msg.data = float(confidence)
        self.confidence_pub.publish(confidence_msg)

        # Publish pose (unit direction vector at 1 meter)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'Head'
        pose_msg.pose.position.x = float(direction_x)
        pose_msg.pose.position.y = float(direction_y)
        pose_msg.pose.position.z = float(direction_z)
        pose_msg.pose.orientation.w = 1.0
        self.pose_pub.publish(pose_msg)

        # Publish visualization marker
        self.publish_marker(direction_x, direction_y, direction_z, confidence, stamp)

    def publish_marker(self, dx: float, dy: float, dz: float, confidence: float, stamp):
        """Publish RViz marker for sound direction visualization."""
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = 'Head'
        marker.ns = 'sound_direction'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        from geometry_msgs.msg import Point
        
        # Arrow from origin to direction
        p1 = Point()
        p1.x, p1.y, p1.z = 0.0, 0.0, 0.0
        marker.points.append(p1)
        
        p2 = Point()
        p2.x, p2.y, p2.z = float(dx), float(dy), float(dz)
        marker.points.append(p2)

        # Color based on confidence (red = low, green = high)
        marker.color.r = 1.0 - confidence
        marker.color.g = confidence
        marker.color.b = 0.0
        marker.color.a = 0.8

        marker.scale.x = 0.05  # Shaft diameter
        marker.scale.y = 0.1   # Head diameter
        marker.scale.z = 0.1   # Head length

        marker.lifetime.nanosec = 500000000  # 0.5 seconds

        self.marker_pub.publish(marker)

    def get_direction_name(self, azimuth_deg: float) -> str:
        """Convert azimuth to human-readable direction name."""
        angle = azimuth_deg % 360
        
        directions = [
            (0, 22.5, "Front"),
            (22.5, 67.5, "Front-Left"),
            (67.5, 112.5, "Left"),
            (112.5, 157.5, "Rear-Left"),
            (157.5, 202.5, "Rear"),
            (202.5, 247.5, "Rear-Right"),
            (247.5, 292.5, "Right"),
            (292.5, 337.5, "Front-Right"),
            (337.5, 360, "Front"),
        ]
        
        for start, end, name in directions:
            if start <= angle < end:
                return name
        return "Front"

    def get_current_direction(self):
        """Get current direction (thread-safe). Returns (azimuth, confidence)."""
        with self.direction_lock:
            return self.current_azimuth, self.current_confidence


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    
    node = SoundLocalizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()