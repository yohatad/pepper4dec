#!/usr/bin/env python3

import rclpy
from .overt_attention_implementation import OvertAttentionSystem

SOFTWARE_VERSION = "v1.0"

def main(args=None):
    """
    Main function to initialize and run the overt attention system.
    """
    rclpy.init(args=args)
    
    node_name = "overtAttention"
    
    # Construct the copyright message
    copyright_message = (
        f"{node_name} {SOFTWARE_VERSION}\n"
        "\t\t\t    This program comes with ABSOLUTELY NO WARRANTY."
    )
    
    try:
        node = OvertAttentionSystem()
        
        # Print startup messages using ROS logging
        node.get_logger().info(copyright_message)
        node.get_logger().info(f"{node_name}: startup.")
        
        # Create control loop timer (20 Hz)
        timer = node.create_timer(0.05, node.run_once)
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            if 'timer' in locals():
                node.destroy_timer(timer)
            if 'node' in locals():
                node.destroy_node()
        except:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()