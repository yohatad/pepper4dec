#!/usr/bin/env python3

import sys
from sound_detection_implementation import main

def main(args=None):
    """
    Main function to initialize and run the sound detection node.
    """
    rclpy.init(args=args)
    
    # Define the node name and software version
    node_name = "soundDetection"
    software_version = " v1.0"  # Replace with the actual software version

    # Construct the copyright message
    copyright_message = (
        f"{node_name}  {software_version}\n"
        "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY."
    )
    
    try:
        node = SoundDetectionNode()
        
        # Print the messages using ROS logging
        node.get_logger().info(copyright_message)
        node.get_logger().info(f"{node_name}: startup.")
        
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()