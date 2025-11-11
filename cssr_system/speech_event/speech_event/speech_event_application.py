#!/home/yoha/sound/bin/python3

"""
speech_event_applications.py
Implementation of 4-microphone sound localization with beamforming and Whisper ASR

Author: Yohannes Tadesse Haile
Date: November 8, 2025
Version: v2.0

Copyright (C) 2023 CSSR4Africa Consortium

This program comes with ABSOLUTELY NO WARRANTY.
"""

import sys
import rclpy
from rclpy.node import Node

def main(args=None):
    """Initialize and run the speech event node."""
    rclpy_inited = False
    try:
        rclpy.init(args=args)
        rclpy_inited = True

        # Import AFTER init for cleaner error reporting
        from .speech_event_implementation import SpeechRecognitionNode

        node_name = "speechEvent"
        software_version = "v1.0"

        node = SpeechRecognitionNode()
        node.get_logger().info(
            f"{node_name} {software_version}\n"
            "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY."
        )
        node.get_logger().info(f"{node_name}: startup.")

        rclpy.spin(node)

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
