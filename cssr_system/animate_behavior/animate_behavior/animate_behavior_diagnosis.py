#!/usr/bin/env python3
"""
Diagnostic script to check animate_behavior system
"""

import rclpy
from rclpy.node import Node
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import time

class AnimateDiagnostic(Node):
    def __init__(self):
        super().__init__('animate_diagnostic')
        
        self.joint_cmd_count = 0
        self.vel_cmd_count = 0
        self.joint_state_count = 0
        self.report_done = False
        self.shutdown_done = False
        
        print("\n" + "="*60)
        print("ANIMATE BEHAVIOR DIAGNOSTICS")
        print("="*60)
        
        # Listen to what animate_behavior publishes
        self.joint_sub = self.create_subscription(
            JointAnglesWithSpeed,
            '/joint_angles',
            self.joint_callback,
            10
        )
        
        self.vel_sub = self.create_subscription(Twist, '/cmd_vel', self.vel_callback, 10)
        
        # Listen to robot feedback
        self.state_sub = self.create_subscription(JointState, '/joint_states', self.state_callback, 10)
        
        print("\n✓ Subscribed to topics")
        print("  - /joint_angles (commands FROM animate_behavior)")
        print("  - /cmd_vel (velocity FROM animate_behavior)")
        print("  - /joint_states (feedback FROM robot)")
        
        print("\nWaiting 5 seconds to check for messages...")
        print("(Make sure animate_behavior action is running)")
        print("="*60 + "\n")
        
        # Check after 5 seconds
        self.report_timer = self.create_timer(5.0, self.report_callback)
    
    def joint_callback(self, msg):
        self.joint_cmd_count += 1
        print(f"\n[{self.joint_cmd_count}] JOINT COMMAND received:")
        print(f"  Joints: {msg.joint_names}")
        print(f"  Angles: {[f'{a:.2f}' for a in msg.joint_angles]}")
        print(f"  Speed: {msg.speed}")
        print(f"  Relative: {msg.relative}")
    
    def vel_callback(self, msg):
        self.vel_cmd_count += 1
        print(f"\n[{self.vel_cmd_count}] VELOCITY COMMAND received:")
        print(f"  Linear: x={msg.linear.x:.3f}, y={msg.linear.y:.3f}, z={msg.linear.z:.3f}")
        print(f"  Angular: x={msg.angular.x:.3f}, y={msg.angular.y:.3f}, z={msg.angular.z:.3f}")
    
    def state_callback(self, msg):
        if self.joint_state_count == 0:
            print(f"\n✓ JOINT STATES received from robot")
            print(f"  Total joints: {len(msg.name)}")
            print(f"  Sample joints: {msg.name[:5]}")
        self.joint_state_count += 1
    
    def report_callback(self):
        if self.report_done:
            return
        self.report_done = True
        self.report_timer.cancel()
        
        self.report()
        
        # Schedule shutdown
        self.shutdown_timer = self.create_timer(30.0, self.shutdown_callback)
    
    def shutdown_callback(self):
        if self.shutdown_done:
            return
        self.shutdown_done = True
        self.shutdown_timer.cancel()
        self.shutdown()
    
    def report(self):
        print("\n" + "="*60)
        print("DIAGNOSTIC REPORT")
        print("="*60)
        print(f"Joint commands received: {self.joint_cmd_count}")
        print(f"Velocity commands received: {self.vel_cmd_count}")
        print(f"Joint states from robot: {self.joint_state_count}")
        
        print("\n--- ANALYSIS ---")
        
        if self.joint_cmd_count == 0 and self.vel_cmd_count == 0:
            print("❌ NO COMMANDS detected from animate_behavior")
            print("\nPossible issues:")
            print("  1. animate_behavior node not running")
            print("  2. No action goal sent to animate_behavior")
            print("  3. Action execution not started")
            print("\nTo fix:")
            print("  - Check: ros2 node list | grep animate")
            print("  - Send goal: ros2 action send_goal /animate_behavior ...")
        
        elif self.joint_state_count == 0:
            print("❌ NO FEEDBACK from robot (no /joint_states)")
            print("\nPossible issues:")
            print("  1. naoqi_driver not running")
            print("  2. Robot not connected")
            print("  3. Network issues")
            print("\nTo fix:")
            print("  - Check: ros2 node list | grep naoqi")
            print("  - Check: ros2 topic echo /joint_states (should show data)")
            print("  - Restart naoqi_driver_py node")
        
        else:
            print("✓ Commands being sent AND robot feedback received")
            print("\nIf robot still not moving, check:")
            print("  1. Is naoqi_driver_py actually controlling the robot?")
            print("     Test with: ros2 topic pub /joint_angles ...")
            print("  2. Are there error messages in naoqi_driver logs?")
            print("  3. Is the robot in stiffness mode? (needs ALMotion.setStiffnesses)")
        
        print("="*60)
        print("\nKeeping diagnostic running for 30 more seconds...")
        print("Send action goals and watch for messages above")
    
    def shutdown(self):
        print("\n✓ Diagnostic complete - you can Ctrl+C now")
        # Don't call rclpy.shutdown() - let user interrupt

def main():
    rclpy.init()
    node = AnimateDiagnostic()
    rclpy.spin(node)

if __name__ == '__main__':
    main()