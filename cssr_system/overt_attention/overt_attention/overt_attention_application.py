#!/usr/bin/env python3
import rclpy
from .overt_attention_implementation import OvertAttentionSystem

SOFTWARE_VERSION = "v1.0"

def main(args=None):
    rclpy.init(args=args)
    node = OvertAttentionSystem()
    # 20 Hz timer for control loop
    timer = node.create_timer(0.05, node.run_once)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_timer(timer)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()