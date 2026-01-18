#!/usr/bin/env python3
"""
Head Reactivity Test for Pepper Robot
Tests how fast the head controller responds to random points in an image.
Measures latency from command to actual head movement.
"""

import rclpy
from rclpy.node import Node
import time
import random
import math
import numpy as np
from collections import deque
import statistics
import cv2

from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import CameraInfo, JointState, Image
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge

# ============ Helper Functions ============
def pixel_to_angles(u, v, fx, fy, cx, cy):
    """Convert pixel coordinates to head yaw/pitch angles (simple pinhole model)."""
    x = (u - cx) / fx
    y = (v - cy) / fy
    yaw = math.atan2(x, 1.0)
    pitch = math.atan2(y, 1.0)
    return yaw, pitch

def clamp_angles(yaw, pitch, yaw_limits=(-1.8, 1.8), pitch_limits=(-0.7, 0.4)):
    """Clamp angles to Pepper's head joint limits."""
    yaw_clamped = max(yaw_limits[0], min(yaw_limits[1], yaw))
    pitch_clamped = max(pitch_limits[0], min(pitch_limits[1], pitch))
    return yaw_clamped, pitch_clamped

# ============ Main Test Node ============
class HeadReactivityTest(Node):
    def __init__(self):
        super().__init__("head_reactivity_test")
        
        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Parameters
        self.declare_parameter("test_duration", 60.0)  # seconds
        self.declare_parameter("point_interval", 2.0)  # seconds between points
        self.declare_parameter("settle_threshold", 0.2)  # rad, threshold to consider settled
        self.declare_parameter("max_response_time", 5.0)  # seconds, timeout for movement
        self.declare_parameter("image_width", 640)  # camera image width (fallback)
        self.declare_parameter("image_height", 480)  # camera image height (fallback)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("head_command_topic", "/joint_angles")
        self.declare_parameter("image_topic", "/camera/color/image_raw_custom")  # Image topic to subscribe
        self.declare_parameter("use_image_dimensions", True)  # Whether to get dimensions from image topic
        self.declare_parameter("visualize", True)  # Whether to visualize selected points
        
        # State variables
        self.camera_info = None
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.command_time = None
        self.awaiting_settle = False
        self.response_times = []
        
        # Image state
        self.latest_image = None
        self.image_dimensions = None  # (width, height)
        self.image_received = False
        
        # Define QoS profile for camera topics (typically use BEST_EFFORT)
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers with QoS
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self.joint_state_callback,
            10  # joint_states typically use default QoS
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.camera_info_callback,
            camera_qos  # Use BEST_EFFORT for camera topics
        )
        
        # Image subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.image_callback,
            camera_qos  # Use BEST_EFFORT for camera topics
        )
        
        # Publisher for head commands
        self.head_pub = self.create_publisher(
            JointAnglesWithSpeed,
            self.get_parameter("head_command_topic").value,
            10
        )
        
        # Timer for test control
        self.test_start_time = None
        self.next_point_time = None
        self.test_active = False
        
        # Statistics
        self.response_time_window = deque(maxlen=100)  # rolling window for recent times
        
        # Visualization
        self.visualization_window = None
        self.last_selected_point = None
        
        # Log startup
        self.get_logger().info("Head Reactivity Test Node initialized")
        self.get_logger().info("Waiting for camera info, image, and joint states...")
        self.get_logger().info(f"Subscribed to image topic: {self.get_parameter('image_topic').value}")

    def image_callback(self, msg):
        """Process incoming image to get dimensions and optionally visualize."""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Update image dimensions if not set or if using image dimensions
            if self.get_parameter("use_image_dimensions").value or self.image_dimensions is None:
                height, width = cv_image.shape[:2]
                self.image_dimensions = (width, height)
                self.image_received = True
                self.latest_image = cv_image
            
            # Visualize if enabled
            if self.get_parameter("visualize").value and self.latest_image is not None:
                self.visualize_point(self.latest_image)
            
            # Check if we can start the test
            if not self.test_active:
                self.check_test_start()
                
        except Exception as e:
            self.get_logger().warn(f"Failed to process image: {e}")

    def visualize_point(self, image):
        """Draw the last selected point on the image for visualization."""
        if self.last_selected_point is not None:
            u, v = self.last_selected_point
            # Draw a red circle at the selected point
            cv2.circle(image, (int(u), int(v)), 10, (0, 0, 255), -1)
            # Draw crosshair
            cv2.line(image, (int(u)-15, int(v)), (int(u)+15, int(v)), (0, 255, 0), 2)
            cv2.line(image, (int(u), int(v)-15), (int(u), int(v)+15), (0, 255, 0), 2)
            # Add text
            cv2.putText(image, f"Target: ({int(u)}, {int(v)})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show the image
        cv2.imshow("Head Reactivity Test - Selected Point", image)
        cv2.waitKey(1)  # Required for OpenCV to update window

    def camera_info_callback(self, msg):
        """Store camera intrinsics when received."""
        if self.camera_info is None:
            self.camera_info = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5]
            }
            self.get_logger().info(f"Camera info received: fx={self.camera_info['fx']}, fy={self.camera_info['fy']}")
            
            # Start test if we have both camera info and joint states
            self.check_test_start()

    def joint_state_callback(self, msg):
        """Update current head position and check if we've reached target."""
        try:
            # Find HeadYaw and HeadPitch indices
            yaw_idx = msg.name.index('HeadYaw')
            pitch_idx = msg.name.index('HeadPitch')
            
            self.current_yaw = msg.position[yaw_idx]
            self.current_pitch = msg.position[pitch_idx]
            
            # If we're waiting for settle, check if we've reached target
            if self.awaiting_settle and self.command_time is not None:
                yaw_error = abs(self.current_yaw - self.target_yaw)
                pitch_error = abs(self.current_pitch - self.target_pitch)
                
                if yaw_error < self.get_parameter("settle_threshold").value and \
                   pitch_error < self.get_parameter("settle_threshold").value:
                    
                    response_time = time.time() - self.command_time
                    self.response_times.append(response_time)
                    self.response_time_window.append(response_time)
                    
                    self.get_logger().info(
                        f"Target reached in {response_time:.3f}s. "
                        f"Current stats: mean={np.mean(self.response_times):.3f}s, "
                        f"std={np.std(self.response_times):.3f}s"
                    )
                    
                    self.awaiting_settle = False
                    self.command_time = None
                    
                # Check for timeout
                elif time.time() - self.command_time > self.get_parameter("max_response_time").value:
                    self.get_logger().warn(f"Timeout waiting for head movement (>{self.get_parameter('max_response_time').value}s)")
                    self.awaiting_settle = False
                    self.command_time = None
            
            # Check if we can start the test
            if not self.test_active:
                self.check_test_start()
                
        except (ValueError, IndexError):
            # HeadYaw or HeadPitch not in joint states yet
            pass

    def check_test_start(self):
        """Start test if we have camera info and joint states (and image if required)."""
        # Check if we have camera info
        if self.camera_info is None:
            return
            
        # Check if we have joint states
        if self.current_yaw is None:
            return
            
        # Check if we need image dimensions and have them
        if self.get_parameter("use_image_dimensions").value and not self.image_received:
            return
            
        if not self.test_active:
            self.test_active = True
            self.test_start_time = time.time()
            self.next_point_time = time.time()
            test_duration = self.get_parameter("test_duration").value
            self.get_logger().info(
                f"Test started! Will run for {test_duration} seconds. "
                f"Point interval: {self.get_parameter('point_interval').value}s"
            )

    def publish_head_command(self, yaw, pitch):
        """Publish head command to look at target angles."""
        msg = JointAnglesWithSpeed()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = ['HeadYaw', 'HeadPitch']
        msg.joint_angles = [float(yaw), float(pitch)]
        msg.speed = 0.08  # Moderate speed
        msg.relative = False
        
        self.head_pub.publish(msg)
        self.command_time = time.time()
        self.awaiting_settle = True
        self.target_yaw = yaw
        self.target_pitch = pitch
        
        self.get_logger().info(f"Commanded head to yaw={math.degrees(yaw):.1f}°, pitch={math.degrees(pitch):.1f}°")

    def get_image_dimensions(self):
        """Get current image dimensions, either from image topic or parameters."""
        if self.get_parameter("use_image_dimensions").value and self.image_dimensions is not None:
            return self.image_dimensions
        else:
            width = self.get_parameter("image_width").value
            height = self.get_parameter("image_height").value
            return (width, height)

    def generate_random_point(self):
        """Generate a random pixel point within image bounds."""
        width, height = self.get_image_dimensions()
        
        # Generate random point with some margin from edges
        margin = 50
        u = random.uniform(margin, width - margin)
        v = random.uniform(margin, height - margin)
        
        # Store for visualization
        self.last_selected_point = (u, v)
        
        return u, v

    def run_test_step(self):
        """Execute one test step: generate point, command head, measure response."""
        if not self.test_active or self.awaiting_settle:
            return
            
        current_time = time.time()
        
        # Check if test duration has elapsed
        if current_time - self.test_start_time > self.get_parameter("test_duration").value:
            self.test_active = False
            self.log_final_statistics()
            self.get_logger().info("Test completed!")
            return
            
        # Check if it's time for next point
        if current_time >= self.next_point_time:
            # Generate random point
            u, v = self.generate_random_point()
            
            # Convert to angles
            yaw, pitch = pixel_to_angles(
                u, v,
                self.camera_info['fx'], self.camera_info['fy'],
                self.camera_info['cx'], self.camera_info['cy']
            )
            
            # Clamp to joint limits
            yaw, pitch = clamp_angles(yaw, pitch)
            
            # Publish command
            self.publish_head_command(yaw, pitch)
            
            # Schedule next point
            self.next_point_time = current_time + self.get_parameter("point_interval").value

    def log_final_statistics(self):
        """Log final statistics when test completes."""
        if not self.response_times:
            self.get_logger().warning("No response times recorded!")
            return
            
        times = np.array(self.response_times)
        self.get_logger().info("=" * 60)
        self.get_logger().info("HEAD REACTIVITY TEST RESULTS")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"Total trials: {len(self.response_times)}")
        self.get_logger().info(f"Mean response time: {np.mean(times):.3f}s")
        self.get_logger().info(f"Std deviation: {np.std(times):.3f}s")
        self.get_logger().info(f"Min response time: {np.min(times):.3f}s")
        self.get_logger().info(f"Max response time: {np.max(times):.3f}s")
        self.get_logger().info(f"Median response time: {np.median(times):.3f}s")
        self.get_logger().info(f"95th percentile: {np.percentile(times, 95):.3f}s")
        self.get_logger().info("=" * 60)
        
        # Save results to file
        try:
            import csv
            filename = f"head_reactivity_results_{int(time.time())}.csv"
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['trial', 'response_time'])
                for i, rt in enumerate(self.response_times):
                    writer.writerow([i, rt])
            self.get_logger().info(f"Results saved to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save results: {e}")

    def destroy_node(self):
        """Cleanup when node is destroyed."""
        if self.test_active:
            self.log_final_statistics()
        # Close OpenCV windows if any
        if self.visualization_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HeadReactivityTest()
    
    # Create a timer for test control (10 Hz)
    timer = node.create_timer(0.1, lambda: node.run_test_step())
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Test interrupted by user")
    finally:
        if rclpy.ok():  # Only log if context is still valid
            node.log_final_statistics()
        node.destroy_node()
        if rclpy.ok():  # Only shutdown if not already shutdown
            rclpy.shutdown()

if __name__ == "__main__":
    main()
