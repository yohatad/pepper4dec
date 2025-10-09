#!/usr/bin/env python3
"""
Simple Tour Guide Simulator
Simulates visitor interactions for testing
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
import time
import sys


class SimpleTourSimulator(Node):
    """Simple simulator for tour guide testing"""
    
    def __init__(self):
        super().__init__('simple_tour_simulator')
        
        # Publishers
        self.visitor_pub = self.create_publisher(Bool, '/perception/visitor_detected', 10)
        self.gaze_pub = self.create_publisher(Bool, '/perception/mutual_gaze', 10)
        self.response_pub = self.create_publisher(String, '/visitor/speech_response', 10)
        self.button_pub = self.create_publisher(String, '/visitor/button_press', 10)
        
        # Subscribers to monitor robot
        self.create_subscription(String, '/robot/text_to_speech', self.tts_callback, 10)
        self.create_subscription(String, '/robot/attention_mode', self.attention_callback, 10)
        self.create_subscription(String, '/robot/perform_gesture', self.gesture_callback, 10)
        self.create_subscription(String, '/robot/exhibit_description', self.exhibit_callback, 10)
        
        self.get_logger().info('='*80)
        self.get_logger().info('Simple Tour Simulator Started')
        self.get_logger().info('='*80)
        
    def tts_callback(self, msg):
        """Monitor robot speech"""
        self.get_logger().info(f'🗣️  Robot: {msg.data}')
    
    def attention_callback(self, msg):
        """Monitor attention mode"""
        self.get_logger().info(f'👀 Attention: {msg.data}')
    
    def gesture_callback(self, msg):
        """Monitor gestures"""
        self.get_logger().info(f'👋 Gesture: {msg.data}')
    
    def exhibit_callback(self, msg):
        """Monitor exhibit descriptions"""
        self.get_logger().info(f'📖 {msg.data}')
    
    def detect_visitor(self):
        """Simulate visitor detection"""
        msg = Bool()
        msg.data = True
        self.visitor_pub.publish(msg)
        self.get_logger().info('✅ Visitor detected!')
    
    def establish_gaze(self):
        """Simulate mutual gaze"""
        msg = Bool()
        msg.data = True
        self.gaze_pub.publish(msg)
        self.get_logger().info('👁️  Mutual gaze established!')
    
    def visitor_says_yes(self):
        """Visitor accepts tour"""
        msg = String()
        msg.data = 'yes'
        self.response_pub.publish(msg)
        self.get_logger().info('✅ Visitor: "Yes!"')
    
    def visitor_says_no(self):
        """Visitor declines tour"""
        msg = String()
        msg.data = 'no'
        self.response_pub.publish(msg)
        self.get_logger().info('❌ Visitor: "No, thank you."')
    
    def run_auto_tour(self):
        """Automatically simulate a full tour"""
        self.get_logger().info('='*80)
        self.get_logger().info('🤖 AUTO MODE: Simulating full tour')
        self.get_logger().info('='*80)
        
        # Wait for system to start
        time.sleep(2)
        
        # Detect visitor
        self.get_logger().info('Step 1: Detecting visitor...')
        self.detect_visitor()
        time.sleep(2)
        
        # Establish gaze
        self.get_logger().info('Step 2: Establishing mutual gaze...')
        self.establish_gaze()
        time.sleep(3)
        
        # Accept tour
        self.get_logger().info('Step 3: Visitor accepts tour...')
        self.visitor_says_yes()
        time.sleep(2)
        
        # Visit 4 exhibits
        for i in range(4):
            self.get_logger().info(f'Step {4+i}: Exhibit {i+1} - navigation...')
            time.sleep(5)  # Navigation time
            
            self.get_logger().info(f'  Establishing gaze at exhibit {i+1}...')
            self.establish_gaze()
            time.sleep(8)  # Description time
        
        self.get_logger().info('='*80)
        self.get_logger().info('🎉 AUTO MODE COMPLETE!')
        self.get_logger().info('='*80)


def print_help():
    """Print help message"""
    print('='*80)
    print('Simple Tour Simulator')
    print('='*80)
    print('Usage:')
    print('  python3 simple_tour_simulator.py              # Interactive mode')
    print('  python3 simple_tour_simulator.py auto         # Auto mode')
    print('  python3 simple_tour_simulator.py decline      # Test rejection')
    print('')
    print('Interactive commands:')
    print('  d - Detect visitor')
    print('  g - Establish gaze')
    print('  y - Say yes')
    print('  n - Say no')
    print('  a - Auto full tour')
    print('  h - Help')
    print('  q - Quit')
    print('='*80)


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    simulator = SimpleTourSimulator()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'auto':
            # Run auto mode
            simulator.run_auto_tour()
            
        elif mode == 'decline':
            # Test visitor declining
            simulator.get_logger().info('Testing visitor declining tour...')
            time.sleep(2)
            simulator.detect_visitor()
            time.sleep(2)
            simulator.establish_gaze()
            time.sleep(2)
            simulator.visitor_says_no()
            simulator.get_logger().info('Decline test complete')
            
        elif mode == 'help' or mode == '-h' or mode == '--help':
            print_help()
            return
        
        # Spin for a bit to see results
        try:
            rclpy.spin(simulator)
        except KeyboardInterrupt:
            pass
    else:
        # Interactive mode
        print_help()
        print('\nEntering interactive mode...\n')
        
        import threading
        
        def input_loop():
            """Handle user input"""
            while rclpy.ok():
                try:
                    cmd = input('Command: ').strip().lower()
                    
                    if cmd == 'd':
                        simulator.detect_visitor()
                    elif cmd == 'g':
                        simulator.establish_gaze()
                    elif cmd == 'y':
                        simulator.visitor_says_yes()
                    elif cmd == 'n':
                        simulator.visitor_says_no()
                    elif cmd == 'a':
                        threading.Thread(target=simulator.run_auto_tour, daemon=True).start()
                    elif cmd == 'h':
                        print_help()
                    elif cmd == 'q':
                        rclpy.shutdown()
                        break
                    else:
                        print(f'Unknown command: {cmd}')
                except (EOFError, KeyboardInterrupt):
                    rclpy.shutdown()
                    break
        
        input_thread = threading.Thread(target=input_loop, daemon=True)
        input_thread.start()
        
        try:
            rclpy.spin(simulator)
        except KeyboardInterrupt:
            pass
    
    simulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()