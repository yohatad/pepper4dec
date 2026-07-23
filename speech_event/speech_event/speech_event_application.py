#!/home/yoha/sound/bin/python3

""" speech_event_application.py

Entry point for the speech_recognition lifecycle node.
Running this module starts SpeechRecognitionNode and spins it on a
multi-threaded executor until shutdown.

The speech_recognition node performs voice-activity detection (Silero VAD) and
speech-to-text transcription (Whisper) on audio from the robot's microphone.
Depending on the `action_server` parameter, it either runs continuously and
publishes transcripts to a topic, or gates audio processing behind an ASR
action server that returns a transcript per goal.

Subscribers:
    /naoqi_driver/audio (naoqi_bridge_msgs/AudioBuffer)
        Raw multi-channel microphone audio (topic name set by the
        `microphone_topic` parameter).

Publishers:
    /speech_event/vad_speech_prob (std_msgs/Float32)
        Per-chunk Silero VAD speech probability.
    /speech_event/text (std_msgs/String)
        Transcribed speech segments (standalone/non-action-server mode).

Services:
    /speech_event/set_enabled (std_srvs/SetBool)
        Enable or disable audio processing (e.g. mute mic during TTS).

Actions:
    /speech_recognition (dec_interfaces/SpeechRecognition)
        Waits for speech onset and returns the transcription once the
        speaker finishes (only created when `action_server` is true).

Parameters (loaded from speech_event_configuration.yaml):
    microphone_topic (string, default: "/naoqi_driver/audio")
    sample_rate (int, default: 16000)
    input_sample_rate (int, default: 48000)
    device (string, default: "cuda")
    compute_type (string, default: "float16")
    language (string, default: "en")
    whisper_model_id (string, default: "deepdml/faster-whisper-large-v3-turbo-ct2")
    speech_threshold (double, default: 0.7)
    neg_threshold (double, default: 0.35)
    min_silence_duration_ms (int, default: 300)
    max_speech_duration_s (double, default: 10.0)
    min_speech_duration (double, default: 0.3)
    pre_speech_buffer_ms (int, default: 200)
    intensity_threshold (double, default: 0.001)
    transcription_timeout_s (double, default: 5.0)
    action_server (bool, default: true)
    vad_always_active (bool, default: false)
    noise_cleaning_enabled (bool, default: true)
    noise_profile_path (string, default: "")
    noise_alpha (double, default: 0.5)

Lifecycle:
    configure  -> load Silero VAD and Whisper models, create the denoiser,
                  set up buffers, and create publishers/service/action server
    activate   -> activate publishers and subscribe to the microphone topic
    deactivate -> unsubscribe from the microphone topic and deactivate publishers
    cleanup    -> destroy publishers, service, action server, and release models

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 8, 2025
Version: v1.0
"""

import sys
import rclpy
from rclpy.executors import MultiThreadedExecutor
from .speech_event_implementation import SpeechRecognitionNode

def main(args=None):
    """Initialize and run the speech event node."""
    rclpy_inited = False
    try:
        rclpy.init(args=args)
        rclpy_inited = True

        node_name = "speechEvent"
        software_version = "v1.0"

        node = SpeechRecognitionNode()
        node.get_logger().info(
            f"{node_name} {software_version}\n"
            "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY."
        )
        node.get_logger().info(f"{node_name}: startup.")

        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()

    except KeyboardInterrupt:
        # Graceful exit on Ctrl-C
        pass

    except Exception as e:
        # If node exists, use ROS logging; else print to stderr
        try:
            node.get_logger().error(f"Unhandled exception: {e}")
        except Exception:
            print(f"[speech_event] Unhandled exception: {e}", file=sys.stderr)

    finally:
        if rclpy_inited:
            try:
                rclpy.shutdown()
            except Exception:
                pass

if __name__ == "__main__":
    main()
