"""remote_dialogue_application.py

Entry point for the remote_dialogue lifecycle node.
Running this module starts RemoteDialogueNode and spins it on a
multi-threaded executor until shutdown.

The remote_dialogue node connects Pepper to the /ws/pepper voice server
(PEPPER_API.md, v1.1), which does speech recognition, turn-taking, the
language model and speech synthesis. The node streams one microphone channel
to the server at 16 kHz, plays the spoken replies on Pepper's speaker, and
exposes each conversation turn as an action for the behavior tree.

Subscribers:
    /audio (naoqi_bridge_msgs/AudioBuffer)
        Pepper's microphones (topic set by `microphone_topic`).

Publishers:
    /remote_dialogue/state (std_msgs/String)
        Server state: "listening" | "thinking" | "speaking" | "idle".
    /remote_dialogue/transcript (dec_interfaces/DialogueTranscript)
        What the server heard ("user") and each reply sentence ("assistant").
    /remote_dialogue/language (std_msgs/String, transient local)
        Current session language.
    /remote_dialogue/connected (std_msgs/Bool, transient local)
        True while the server session is ready.

Service clients:
    /naoqi_driver/send_audio_buffer (naoqi_bridge_msgs/SendAudioBuffer)
        Reply audio, 48 kHz stereo (name set by `send_audio_buffer_service`).

Services:
    /remote_dialogue/set_language (dec_interfaces/SetDialogueLanguage)
        Switch language; returns once the server confirms.
    /remote_dialogue/reset (dec_interfaces/ResetDialogue)
        New visitor: clear the conversation, optionally greet.
    /remote_dialogue/set_listening (std_srvs/SetBool)
        Keep the microphone gate open outside action goals.

Actions:
    /remote_dialogue (dec_interfaces/RemoteDialogueTurn)
        One conversation turn: listen (or send `text`), play the reply,
        return what was said.

Environment:
    REMOTE_DIALOGUE_TOKEN   the server's shared secret (required; never a
                            ROS parameter, which anyone on the graph can read)
    REMOTE_DIALOGUE_URL     overrides `server_url` (the address changes with
                            each GPU job)

Parameters (code defaults; overridden by remote_dialogue_configuration.yaml):
    server_url (string, default: "")
    language (string, default: "eng")
    output_sample_rate (int, default: 48000)
    greet_on_connect (bool, default: false)
    microphone_topic (string, default: "/audio")
    microphone_channel (int, default: 0 = CHANNEL_FRONT_LEFT)
    send_audio_buffer_service (string, default: "/naoqi_driver/send_audio_buffer")
    playback_piece_s (double, default: 0.2)
    playback_lead_s (double, default: 0.15)
    playback_tail_s (double, default: 0.15)
    mic_guard_s (double, default: 0.3)
    turn_timeout_s (double, default: 90.0)
    language_timeout_s (double, default: 5.0)
    reconnect_interval_s (double, default: 3.0)
    verbose_mode (bool, default: false)

Lifecycle:
    configure  -> validate parameters and the token, create publishers,
                  services, the action server and the speaker client
    activate   -> connect to the server (reconnecting on drops), start the
                  playback thread, subscribe to the microphone
    deactivate -> unsubscribe, stop playback, close the session
    cleanup    -> destroy the ROS interface
    shutdown   -> close the session

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 26, 2026
Version: v1.0

Copyright (C) 2026 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import signal

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from .remote_dialogue_implementation import RemoteDialogueNode


def main(args=None):
    """Initialize and run the remote dialogue node."""
    rclpy.init(args=args)
    node = RemoteDialogueNode()
    node.get_logger().info('remote_dialogue v1.0: startup.')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # A second SIGINT during cleanup would interrupt the thread joins.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node._shutdown_runtime()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
