#!/usr/bin/env python3
import rclpy
from overt_attention_system import OvertAttentionSystem  # adjust import path as needed

SOFTWARE_VERSION = "v1.0"

def main(args=None):
    rclpy.init(args=args)
    node = OvertAttentionSystem()

    node.get_logger().info(
        f"{node.get_name()}: {SOFTWARE_VERSION}\n"
        "This program comes with ABSOLUTELY NO WARRANTY."
    )

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
