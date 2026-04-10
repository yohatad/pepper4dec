#!/home/yoha/tts_virtual_env/bin/python3

"""
text_to_speech_application.py
ROS2 node for Text-to-Speech synthesis and playback on the Pepper robot.

Receives sentences from /conversation_manager/response_stream and speaks them
as they arrive — enabling Pepper to start talking before the LLM has finished
generating the full response.

Backends (set via config key 'engine'):
  naoqi_ros      — Pepper's on-board ALTextToSpeech via naoqi_bridge.
  kokoro_local   — Kokoro-82M synthesised locally, played on the laptop speakers.
                   Useful for development / testing without the robot.
  kokoro_pepper  — Kokoro-82M synthesised locally.  Raw WAV bytes are sent to
                   naoqi_driver via the load_audio_file service (no SCP/SSH).
                   naoqi_driver writes the bytes to the robot via ALFileManager
                   and plays them through Pepper's speakers via play_audio action.

Features:
  - Sentence queue with a background playback thread.
  - Barge-in detection: user speech during playback stops Pepper immediately.
  - Microphone muting during playback (naoqi_ros / kokoro_local backends).
  - /tts action server for programmatic TTS calls.

ROS Subscriptions:
  /conversation_manager/response_stream (std_msgs/String)
  /speech_event/vad_speech_prob         (std_msgs/Float32)

ROS Service clients:
  /speech_event/set_enabled             (std_srvs/SetBool)
  /naoqi_driver/load_audio_file         (naoqi_bridge_msgs/srv/LoadAudioFile)
  /naoqi_driver/unload_audio_file       (naoqi_bridge_msgs/srv/UnloadAudioFile)

ROS Action clients:
  /naoqi_driver/play_audio              (naoqi_bridge_msgs/action/PlayAudio)

ROS Action server:
  /tts                                  (dec_interfaces/action/TTS)

ROS Publishers:
  /text_to_speech/speaking              (std_msgs/Bool)

Author: Yohannes Tadesse Haile, Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: April 2025
Version: v3.0
"""

import rclpy
from .text_to_speech_implementation import TextToSpeechNode, load_configuration

SOFTWARE_VERSION = "v1.0"


def main():
    """
    Main function to run the text-to-speech system.
    """
    rclpy.init()
    node_name = "text_to_speech"

    copyright_message = (
        f"{node_name} {SOFTWARE_VERSION}\n"
        "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY."
    )

    print(copyright_message)
    config = load_configuration()
    node = None

    try:
        node = TextToSpeechNode(config)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.cleanup()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()