"""
sound_detection_implementation.py Implementation code for running the sound detection and localization algorithm

Author: Yohannes Tadesse Haile
Date: April 13, 2025
Version: v1.0

Copyright (C) 2023 CSSR4Africa Consortium

This project is funded by the African Engineering and Technology Network (Afretec)
Inclusive Digital Transformation Research Grant Programme.

Website: www.cssr4africa.org

This program comes with ABSOLUTELY NO WARRANTY.
"""

import math
import os
import yaml
import rclpy
import std_msgs.msg
import webrtcvad
import numpy as np
import threading
import noisereduce as nr 
import soundfile as sf 
from datetime import datetime
from rclpy.node import Node
from naoqi_bridge_msgs.msg import AudioBuffer
from threading import Lock
from std_msgs.msg import Float32MultiArray, Float32
from ament_index_python.packages import get_package_share_directory

class SoundDetectionNode(Node):
    """
    SoundDetectionNode processes audio data from a microphone topic, applies VAD to determine if speech is present,
    applies bandpass filtering and spectral subtraction on the left channel, and localizes the sound source by computing
    the interaural time difference (ITD) via GCC-PHAT.
    """
    def __init__(self):
        """
        Initialize the SoundDetectionNode.
        Sets up ROS subscribers, publishers, and loads configuration parameters.
        """
        super().__init__('sound_detection')
        
        # Load configuration from YAML file
        self.config = self.read_yaml_config('cssr_system', 'sound_detection_configuration.yaml')
        
        # Set parameters from config
        self.frequency_sample = 48000
        self.speed_of_sound = 343.0
        self.distance_between_ears = self.config.get('distanceBetweenEars', 0.07)
        self.intensity_threshold = self.config.get('intensityThreshold', 3.9e-3)
        self.verbose_mode = self.config.get('verboseMode', False)

        # Initialize parameter for noise reduction filter 
        self.noise_type = self.config.get('stationary', True)
        self.prop_decrease = self.config.get('propDecrease', 0.9)
        
        # Buffer for localization (2 channels)
        self.localization_buffer_size = self.config.get('localizationBufferSize', 8192)
        self.frontleft_buffer = np.zeros(self.localization_buffer_size, dtype=np.float32)
        self.frontright_buffer = np.zeros(self.localization_buffer_size, dtype=np.float32)
        self.accumulated_samples = 0

        # Initialize VAD with configurable aggressiveness mode
        self.vad_aggressiveness = self.config.get('vadAggressiveness', 1)
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        self.vad_frame_duration = 0.02  # 20 ms (WebRTC VAD requires specific frame durations)
        self.vad_frame_size = int(self.frequency_sample * self.vad_frame_duration)

        # Initialize RMS parameters
        self.target_rms = self.config.get('targetRMS', 0.2)

        # Initialize timeout parameters
        self.last_audio_time = self.get_clock().now()
        self.audio_timeout = self.config.get('audioTimeout', 2)  # Default 2 seconds timeout
        self.received_first_audio = False  # Flag to track if we've received any audio yet

        # Initialize noise reduction parameters
        self.context_duration = self.config.get('contextDuration', 2.0)  # Context window duration in seconds
        self.context_size = int(self.frequency_sample * self.context_duration)
        self.left_context_window = np.zeros(self.context_size, dtype=np.float32) 
        self.use_noise_reduction = self.config.get('useNoiseReduction', True)
        
        if self.use_noise_reduction and self.verbose_mode:
            self.get_logger().info(f"Noise reduction enabled for left channel with {self.context_duration}s context window")

        # Retrieve the microphone topic from the YAML file
        microphone_topic = self.extract_topics('Microphone')
        if not microphone_topic:
            self.get_logger().error("Microphone topic not found in topic file.")
            raise ValueError("Missing microphone topic configuration.")

        # Initialize thread lock for shared resources
        self.lock = Lock()
        
        # Timer for periodic status message
        self.last_status_time = self.get_clock().now()

        # Set up ROS subscribers and publishers
        self.audio_sub = self.create_subscription(AudioBuffer, microphone_topic, self.audio_callback,10)
        self.get_logger().info(f"Subscribed to {microphone_topic}")
        
        self.signal_pub = self.create_publisher(Float32MultiArray, '/soundDetection/signal', 10)
        self.direction_pub = self.create_publisher(Float32, '/soundDetection/direction', 10)

        # Start timeout monitor
        self.start_timeout_monitor()

    def start_timeout_monitor(self):
        """
        Start a background thread to monitor for audio timeouts.
        Shuts down the node if no audio is received within the timeout period,
        but only after at least one audio message has been received.
        """
        def monitor():
            while rclpy.ok():
                # Only check for timeouts if we've received at least one audio message
                if self.received_first_audio:
                    time_since_last = (self.get_clock().now() - self.last_audio_time).nanoseconds / 1e9
                    if time_since_last > self.audio_timeout:
                        self.get_logger().warn(f"No audio received for {self.audio_timeout} seconds. Shutting down.")
                        rclpy.shutdown()
                        break
                threading.Event().wait(1.0)  # Wait 1 second

        threading.Thread(target=monitor, daemon=True).start()
        if self.verbose_mode:
            self.get_logger().info(f"Audio timeout monitor started (timeout: {self.audio_timeout}s)")

    def read_yaml_config(self, package_name, config_file):
        """
        Read and parse a YAML configuration file from the specified ROS package.
        
        Args:
            package_name (str): Name of the ROS package containing the config file
            config_file (str): Name of the YAML configuration file
            
        Returns:
            dict: Configuration data from YAML file, or empty dict if file not found
        """
        try:
            package_path = get_package_share_directory(package_name)
            config_path = os.path.join(package_path, 'config', config_file)
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    data = yaml.safe_load(file)
                    return data if data is not None else {}
            else:
                self.get_logger().error(f"Configuration file not found at {config_path}")
                return {}
                
        except Exception as e:
            self.get_logger().error(f"Error reading YAML configuration file: {e}")
            return {}

    def extract_topics(self, topic_key):
        """
        Extract the topic name for a given key from the topics YAML file.
        
        Args:
            topic_key (str): Key to search for in the topics file
            
        Returns:
            str or None: The topic name if found, None otherwise
        """
        try:
            package_path = get_package_share_directory('cssr_system')
            config_path = os.path.join(package_path, 'config', 'pepper_topics.yaml')
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    topics_data = yaml.safe_load(file)
                    return topics_data.get(topic_key)
            else:
                self.get_logger().error(f"Topics YAML file not found at {config_path}")
        except Exception as e:
            self.get_logger().error(f"Error reading topics YAML file: {e}")
        return None
    
    def normalize_rms(self, audio_data, target_rms=None, min_rms=1e-10):
        """
        Apply RMS normalization to audio data.
        
        Args:
            audio_data (np.ndarray): Audio data to normalize
            target_rms (float): Target RMS value (typically 0.1-0.3)
            min_rms (float): Minimum RMS value to avoid division by zero
            
        Returns:
            np.ndarray: Normalized audio data
        """
        if target_rms is None:
            target_rms = self.target_rms
            
        # Calculate current RMS value
        rms_current = np.sqrt(np.mean(audio_data**2))
        
        # Skip normalization if RMS is too low (silent)
        if rms_current < min_rms:
            if self.verbose_mode:
                self.get_logger().info(f"Audio too quiet for normalization (RMS: {rms_current:.6f})")
            return audio_data
        
        # Calculate scaling factor
        scaling_factor = target_rms / rms_current
        
        # Apply normalization
        normalized_data = audio_data * scaling_factor
        
        # Clip to prevent overflow
        normalized_data = np.clip(normalized_data, -1.0, 1.0)
        
        if self.verbose_mode:
            self.get_logger().info(f"Applied RMS normalization - Before RMS: {rms_current:.4f}, After RMS: {target_rms:.4f}, Factor: {scaling_factor:.4f}")
        
        return normalized_data
    
    def apply_noise_reduction(self, current_block):
        """
        Apply noise reduction to an audio block using the rolling context window approach.
        Only processes the left channel.
        
        Args:
            current_block (np.ndarray): New audio block to process
            
        Returns:
            np.ndarray: Noise-reduced audio block
        """
        try:
            # Skip if noise reduction is disabled
            if not self.use_noise_reduction:
                return current_block
                
            # Update context window: shift old data left and add new block at the end
            block_size_actual = len(current_block)
            self.left_context_window = np.roll(self.left_context_window, -block_size_actual)
            self.left_context_window[-block_size_actual:] = current_block
            
            # Apply stationary noise reduction to the context window
            reduced_context = nr.reduce_noise(
                y=self.left_context_window,
                sr=self.frequency_sample,
                stationary=self.noise_type,
                prop_decrease=self.prop_decrease,
            )
            
            # Extract only the most recent block from the processed context
            processed_block = reduced_context[-block_size_actual:]
            
            return processed_block
            
        except Exception as e:
            self.get_logger().error(f"Error in noise reduction: {e}")
            return current_block  # Return original block on error
    
    def voice_detected(self, audio_frame):
        """
        Use Voice Activity Detection (VAD) to determine if voice is present.
        
        Args:
            audio_frame (np.ndarray): Audio frame to analyze
            
        Returns:
            bool: True if voice is detected, False otherwise
        """
        try:
            # Process the audio in VAD frame-sized chunks
            for start in range(0, len(audio_frame) - self.vad_frame_size + 1, self.vad_frame_size):
                frame = audio_frame[start:start + self.vad_frame_size]
                
                # Convert to int16 bytes for WebRTC VAD
                frame_bytes = (frame * 32767).astype(np.int16).tobytes()
                
                # Check if this frame contains speech
                if self.vad.is_speech(frame_bytes, self.frequency_sample):
                    return True
            return False
        except Exception as e:
            self.get_logger().warn(f"Error in VAD processing: {e}")
            return False

    def audio_callback(self, msg):
        """
        Process incoming audio data from the microphone.
        
        Args:
            msg (SoundDetectionMicrophoneMsgFile): The audio data message
        """
        try:
            self.last_audio_time = self.get_clock().now()
    
            # If this is the first audio message, log it and set the flag
            if not self.received_first_audio:
                self.received_first_audio = True
                if self.verbose_mode:
                    self.get_logger().info("First audio data received, timeout monitoring active")
            
            # Print a status message every 10 seconds
            current_time = self.get_clock().now()
            if (current_time - self.last_status_time).nanoseconds / 1e9 >= 10:
                self.get_logger().info("running.")
                self.last_status_time = current_time
                
            # Process audio data
            sigIn_frontLeft, sigIn_frontRight = self.process_audio_data(msg)

            # Check intensity threshold
            if not self.is_intense_enough(sigIn_frontLeft):
                return
                       
            # Apply noise reduction only to the left channel
            sigIn_frontLeft_clean = self.apply_noise_reduction(sigIn_frontLeft)
            
            # Check for voice activity in the left channel (using noise-reduced signal for better detection)
            self.speech_detected = self.voice_detected(sigIn_frontLeft_clean)

            # If no speech detected, we can skip further processing
            if not self.speech_detected:
                return
            
            # Publish the noise-reduced left channel signal
            self.publish_signal(sigIn_frontLeft_clean)

            # Update localization buffers with RAW signals (not noise-reduced)
            with self.lock:
                self.update_buffers(sigIn_frontLeft, sigIn_frontRight)

            # Localization processing
            if self.accumulated_samples >= self.localization_buffer_size:
                
                # Perform localization
                self.localize(self.frontleft_buffer, self.frontright_buffer)
                
                # Reset buffers for next batch
                with self.lock:
                    self.frontleft_buffer = np.zeros(self.localization_buffer_size, dtype=np.float32)
                    self.frontright_buffer = np.zeros(self.localization_buffer_size, dtype=np.float32)
                    self.accumulated_samples = 0

        except Exception as e:
            self.get_logger().error(f"Error in audio_callback: {e}")

    def process_audio_data(self, msg):
        """
        Extract and normalize audio data from the message.
        
        Args:
            msg (SoundDetectionMicrophoneMsgFile): The audio data message
            
        Returns:
            tuple: (left_channel, right_channel) as normalized float32 arrays
        """
        try:
            # Convert int16 data to float32 and normalize to [-1.0, 1.0]
            sigIn_frontLeft = np.array(msg.front_left, dtype=np.float32) / 32767.0
            sigIn_frontRight = np.array(msg.front_right, dtype=np.float32) / 32767.0
            return sigIn_frontLeft, sigIn_frontRight
        except Exception as e:
            self.get_logger().error(f"Error processing audio data: {e}")
            return (np.zeros(self.localization_buffer_size, dtype=np.float32),
                    np.zeros(self.localization_buffer_size, dtype=np.float32))

    def is_intense_enough(self, signal_data):
        """
        Check if the signal intensity exceeds the threshold.
        
        Args:
            signal_data (np.ndarray): The audio signal data
            
        Returns:
            bool: True if signal is intense enough, False otherwise
        """
        # Calculate root mean square (RMS) intensity
        intensity = np.sqrt(np.mean(signal_data ** 2))
        return intensity >= self.intensity_threshold

    def update_buffers(self, sigIn_frontLeft, sigIn_frontRight):
        """
        Update the internal buffers with new audio data.
        
        Args:
            sigIn_frontLeft (np.ndarray): Left channel audio data
            sigIn_frontRight (np.ndarray): Right channel audio data
        """
        data_length = len(sigIn_frontLeft)
        if self.accumulated_samples + data_length <= self.localization_buffer_size:
            # There's room for all the new data
            start_index = self.accumulated_samples
            end_index = start_index + data_length
            self.frontleft_buffer[start_index:end_index] = sigIn_frontLeft
            self.frontright_buffer[start_index:end_index] = sigIn_frontRight
            self.accumulated_samples += data_length
        else:
            # Only part of the new data will fit
            remaining = self.localization_buffer_size - self.accumulated_samples
            if remaining > 0:
                self.frontleft_buffer[self.accumulated_samples:] = sigIn_frontLeft[:remaining]
                self.frontright_buffer[self.accumulated_samples:] = sigIn_frontRight[:remaining]
                self.accumulated_samples = self.localization_buffer_size

    def localize(self, sigIn_frontLeft, sigIn_frontRight):
        """
        Localize the sound source using the GCC-PHAT algorithm.
        
        Args:
            sigIn_frontLeft (np.ndarray): Left channel audio data
            sigIn_frontRight (np.ndarray): Right channel audio data
        """
        try:
            # Calculate Interaural Time Difference (ITD)
            itd = self.gcc_phat(sigIn_frontLeft, sigIn_frontRight, self.frequency_sample)
            
            # Convert ITD to angle
            angle = self.calculate_angle(itd)
            
            # Publish the calculated angle
            self.publish_angle(angle)
        except Exception as e:
            self.get_logger().warn(f"Error in localization: {e}")

    def gcc_phat(self, sig, ref_sig, fs, max_tau=None, interp=16):
        """
        Implement the GCC-PHAT algorithm for time delay estimation.
        
        Args:
            sig (np.ndarray): Signal from first channel
            ref_sig (np.ndarray): Signal from reference channel
            fs (int): Sampling frequency
            max_tau (float, optional): Maximum delay to consider
            interp (int, optional): Interpolation factor
            
        Returns:
            float: Estimated time delay in seconds
        """
        try:
            # Compute FFT length
            n = sig.shape[0] + ref_sig.shape[0]
            
            # Compute FFTs
            SIG = np.fft.rfft(sig, n=n)
            REFSIG = np.fft.rfft(ref_sig, n=n)
            
            # Compute cross-correlation in frequency domain
            R = SIG * np.conj(REFSIG)
            
            # Apply phase transform (PHAT)
            R /= (np.abs(R) + 1e-10)
            
            # Compute inverse FFT to get time-domain cross-correlation
            cc = np.fft.irfft(R, n=n)
            
            # Find maximum correlation
            max_shift = int(n / 2)
            if max_tau:
                max_shift = min(int(fs * max_tau), max_shift)
            
            # Concatenate the end and beginning of cc to align the shifts properly
            cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
            
            # Find the shift that gives maximum correlation
            shift = np.argmax(np.abs(cc)) - max_shift
            
            # Convert shift to time
            return shift / float(fs)
        except Exception as e:
            self.get_logger().error(f"Error in GCC-PHAT: {e}")
            return 0.0

    def calculate_angle(self, itd):
        """
        Calculate the sound source angle from the ITD.
        
        Args:
            itd (float): Interaural Time Difference in seconds
            
        Returns:
            float: Sound source angle in degrees
        """
        try:
            # Calculate sine of the angle
            z = itd * (self.speed_of_sound / self.distance_between_ears)
            
            # Clamp value to valid range for arcsin
            z = max(-1.0, min(1.0, z))
            
            # Calculate angle in degrees
            angle = math.asin(z) * (180.0 / math.pi)

            # If the angle is not in [-67, 67], skip it 
            if angle < -67 or angle > 67:
                return None

            return angle
        except ValueError as e:
            self.get_logger().warn(f"Invalid ITD for angle calculation: {e}")
            return None

    def publish_angle(self, angle):
        """
        Publish the calculated angle to the direction topic.
        
        Args:
            angle (float): Sound source angle in degrees
        """
        if angle is None:
            return
        angle_msg = Float32()
        angle_msg.data = angle
        self.direction_pub.publish(angle_msg)

    def publish_signal(self, signal_data):
        """
        Publish the processed signal to the signal topic.
        
        Args:
            signal_data (np.ndarray): Processed audio signal
        """
        signal_msg = Float32MultiArray()
        signal_msg.data = signal_data.tolist()
        self.signal_pub.publish(signal_msg)
        
    def destroy_node(self):
        """
        Handle cleanup when the node is shutting down.
        """
        super().destroy_node()