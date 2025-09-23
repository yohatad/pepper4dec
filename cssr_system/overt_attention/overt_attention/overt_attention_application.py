#!/usr/bin/env python3

import rclpy
import time
import cv2
import math
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
        attention_system = OvertAttentionSystem()
        
        # Create timer for main control loop
        timer_period = 0.05  # 20 Hz for more responsive behavior
        timer = attention_system.create_timer(timer_period, attention_system.run_once)
        
        # Log social control mode
        social_mode = attention_system.behaviors.social_attention_system.control_mode
        attention_system.get_logger().info(f"Social attention mode: {social_mode.value}")
        
        # Log movement filtering status
        movement_config = attention_system.behaviors.social_attention_system.smooth_controller.config
        attention_system.get_logger().info(f"Movement filtering - Min threshold: {math.degrees(movement_config.MIN_MOVEMENT_THRESHOLD):.1f}°, "
                                         f"Stability time: {movement_config.STABILITY_TIME}s, "
                                         f"Movement timeout: {movement_config.MOVEMENT_TIMEOUT}s")
        
        rclpy.spin(attention_system)
        
    except KeyboardInterrupt:
        pass
    finally:
        if 'attention_system' in locals():
            attention_system.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()