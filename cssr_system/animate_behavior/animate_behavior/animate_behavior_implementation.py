#!/usr/bin/env python3
"""
animate_behavior.py - Fully non-blocking version

The execute_callback returns immediately and everything is handled by timers.
"""

import rclpy
import time
import random
from typing import List
from dataclasses import dataclass
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from sensor_msgs.msg import JointState
from cssr_interfaces.action import AnimateBehavior

@dataclass
class JointDef:
    """Joint definition with limits and home positions"""
    names: List[str]
    min: List[float]
    max: List[float]
    home: List[float]
    factors: List[float]

class AnimateBehaviorServer(Node):
    """Animate behavior action server for Pepper robot"""
    
    def __init__(self):
        super().__init__('animate_behavior')
        
        self.get_logger().info('='*60)
        self.get_logger().info('ANIMATE BEHAVIOR NODE (NON-BLOCKING)')
        self.get_logger().info('='*60)
        
        # Use reentrant callback group
        self.callback_group = ReentrantCallbackGroup()
        
        # Joint definitions
        self.joints = {
            'RArm': JointDef(
                names=['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw'],
                min=[-2.09, -1.56, -2.09, 0.01, -1.82],
                max=[2.09, -0.01, 2.09, 1.56, 1.82],
                home=[1.54, -0.10, 1.70, 0.31, -0.06],
                factors=[0.6, 0.4, 0.6, 0.4, 0.5]
            ),
            'LArm': JointDef(
                names=['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw'],
                min=[-2.09, 0.01, -2.09, -1.56, -1.82],
                max=[2.09, 1.56, 2.09, -0.01, 1.82],
                home=[1.56, 0.10, -1.72, -0.34, 0.07],
                factors=[0.6, 0.4, 0.6, 0.4, 0.5]
            ),
            'RHand': JointDef(
                names=['RHand'], 
                min=[0.0], 
                max=[1.0], 
                home=[0.67], 
                factors=[0.5]
            ),
            'LHand': JointDef(
                names=['LHand'], 
                min=[0.0], 
                max=[1.0], 
                home=[0.67], 
                factors=[0.5]
            ),
            'Leg': JointDef(
                names=['HipPitch', 'HipRoll', 'KneePitch'],
                min=[-1.04, -0.51, -0.51],
                max=[1.04, 0.51, 0.51],
                home=[-0.11, -0.01, 0.03],
                factors=[0.3, 0.2, 0.3]
            )
        }
        
        # Animation state
        self.active = False
        self.behavior = ''
        self.range = 0.5
        self.limbs_to_animate = []
        self.last_gesture = {}
        self.last_rotation = 0.0
        self.gesture_count = 0
        
        # Goal tracking
        self.current_goal_handle = None
        self.start_time = 0.0
        self.duration = 0.0
        
        # Timing intervals
        self.gesture_interval = 3.0  # seconds between gestures
        self.rotation_interval = 5.0  # seconds between rotations
        
        # Publishers
        self.joint_pub = self.create_publisher(
            JointAnglesWithSpeed, 
            '/joint_angles', 
            10
        )
        self.vel_pub = self.create_publisher(
            Twist, 
            '/cmd_vel', 
            10
        )
        
        # Subscriber
        self.joint_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_states_callback, 
            10
        )
        
        # Action server
        self._action_server = ActionServer(
            self, 
            AnimateBehavior, 
            'animate_behavior',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )
        
        # Main animation timer - 10Hz
        self.animation_timer = self.create_timer(
            0.1, 
            self.animation_update,
            callback_group=self.callback_group
        )
        
        # Feedback timer - 2Hz
        self.feedback_timer = self.create_timer(
            0.5,
            self.feedback_update,
            callback_group=self.callback_group
        )
        
        self.get_logger().info('✓ Node ready, timers running at 10Hz and 2Hz')
        self.get_logger().info('✓ Waiting for goals on /animate_behavior')
        self.get_logger().info('='*60)
    
    def joint_states_callback(self, msg: JointState):
        """Receive joint states"""
        pass
    
    def goal_callback(self, goal_request):
        """Validate goal"""
        self.get_logger().info(f'Goal: {goal_request.behavior_type}, '
                              f'range={goal_request.selected_range}, '
                              f'duration={goal_request.duration_seconds}s')
        
        valid_behaviors = ['All', 'body', 'hands', 'rotation']
        if goal_request.behavior_type not in valid_behaviors:
            self.get_logger().error(f'Invalid behavior: {goal_request.behavior_type}')
            return GoalResponse.REJECT
        
        if not 0.0 <= goal_request.selected_range <= 1.0:
            self.get_logger().error(f'Invalid range: {goal_request.selected_range}')
            return GoalResponse.REJECT
        
        self.get_logger().info('✓ Goal accepted')
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Handle cancellation"""
        self.get_logger().info('Cancel requested')
        return CancelResponse.ACCEPT
    
    def execute_callback(self, goal_handle):
        """Execute - returns immediately, timers do the work"""
        self.get_logger().info('='*60)
        self.get_logger().info('EXECUTE CALLBACK STARTED')
        self.get_logger().info('='*60)
        
        # Stop previous animation
        if self.active:
            self.stop_animation()
        
        # Setup new animation
        goal = goal_handle.request
        self.behavior = goal.behavior_type
        self.range = goal.selected_range
        self.duration = goal.duration_seconds
        self.start_time = time.time()
        self.gesture_count = 0
        self.current_goal_handle = goal_handle
        
        # Get limbs
        self.limbs_to_animate = self.get_limbs_for_behavior(self.behavior)
        self.get_logger().info(f'Animating: {", ".join(self.limbs_to_animate) if self.limbs_to_animate else "rotation only"}')
        
        # Initialize timing
        self.last_gesture = {limb: 0.0 for limb in self.limbs_to_animate}
        self.last_rotation = 0.0
        
        # Move to home
        self.get_logger().info('Moving to home...')
        for limb in self.limbs_to_animate:
            self.move_to_home(limb)
        
        # Activate - timers will now do the work
        self.active = True
        
        self.get_logger().info('✓ Animation ACTIVE - returning from execute')
        self.get_logger().info('  Timers will handle animation and completion')
        self.get_logger().info('='*60)
        
        # Return immediately - don't block!
        # The animation_update and feedback_update timers handle everything
        return AnimateBehavior.Result(
            success=True,
            message='Animation started',
            total_duration=0.0
        )
    
    def animation_update(self):
        """Main animation loop - 10Hz"""
        if not self.active:
            return
        
        # Check duration
        if self.duration > 0:
            elapsed = time.time() - self.start_time
            if elapsed >= self.duration:
                self.get_logger().info(f'Duration complete: {elapsed:.1f}s')
                self.complete_goal()
                return
        
        # Check cancellation
        if self.current_goal_handle and self.current_goal_handle.is_cancel_requested:
            self.get_logger().info('Goal cancelled')
            self.cancel_goal()
            return
        
        now = time.time()
        
        # Animate limbs
        for limb in self.limbs_to_animate:
            if limb in self.last_gesture:
                time_since = now - self.last_gesture[limb]
                if time_since >= self.gesture_interval:
                    self.animate_limb(limb)
                    self.last_gesture[limb] = now
                    self.gesture_count += 1
        
        # Animate rotation
        if self.behavior in ['All', 'rotation']:
            time_since_rot = now - self.last_rotation
            if time_since_rot >= self.rotation_interval:
                self.animate_rotation()
                self.last_rotation = now
    
    def feedback_update(self):
        """Publish feedback - 2Hz"""
        if not self.active or not self.current_goal_handle:
            return
        
        elapsed = time.time() - self.start_time
        
        feedback = AnimateBehavior.Feedback()
        feedback.current_limb = self.behavior
        feedback.gestures_completed = self.gesture_count
        feedback.elapsed_time = elapsed
        feedback.is_running = self.active
        
        self.current_goal_handle.publish_feedback(feedback)
        
        self.get_logger().info(f'[{elapsed:.1f}s] Gestures: {self.gesture_count}')
    
    def complete_goal(self):
        """Complete the current goal"""
        if not self.current_goal_handle:
            return
        
        elapsed = time.time() - self.start_time
        
        self.stop_animation()
        self.current_goal_handle.succeed()
        
        result = AnimateBehavior.Result(
            success=True,
            message=f'Completed {self.gesture_count} gestures',
            total_duration=elapsed
        )
        
        self.get_logger().info('='*60)
        self.get_logger().info(f'✓ Complete: {result.message} in {elapsed:.1f}s')
        self.get_logger().info('='*60)
        
        self.current_goal_handle = None
    
    def cancel_goal(self):
        """Cancel the current goal"""
        if not self.current_goal_handle:
            return
        
        elapsed = time.time() - self.start_time
        
        self.stop_animation()
        self.current_goal_handle.canceled()
        
        self.get_logger().info(f'✓ Cancelled after {elapsed:.1f}s')
        
        self.current_goal_handle = None
    
    def animate_limb(self, limb: str):
        """Animate one limb"""
        if limb not in self.joints:
            return
        
        joint_def = self.joints[limb]
        target_angles = []
        
        for i in range(len(joint_def.names)):
            center = joint_def.home[i]
            full_range = joint_def.max[i] - joint_def.min[i]
            allowed_range = full_range * joint_def.factors[i] * self.range / 2.0
            
            min_angle = max(center - allowed_range, joint_def.min[i])
            max_angle = min(center + allowed_range, joint_def.max[i])
            
            target = random.uniform(min_angle, max_angle)
            target_angles.append(target)
        
        self.send_joint_command(joint_def.names, target_angles, speed=0.2)
        self.get_logger().info(f'  → {limb}')
    
    def animate_rotation(self):
        """Rotate base"""
        angular_vel = random.uniform(-0.3 * self.range, 0.3 * self.range)
        
        msg = Twist()
        msg.angular.z = angular_vel
        self.vel_pub.publish(msg)
        
        self.get_logger().info(f'  → Rotation: {angular_vel:.2f} rad/s')
        
        # Stop after half interval
        def stop_rotation():
            if self.active:
                msg = Twist()
                self.vel_pub.publish(msg)
        
        self.create_timer(
            self.rotation_interval / 2,
            stop_rotation,
            callback_group=self.callback_group
        )
    
    def send_joint_command(self, joint_names: List[str], angles: List[float], speed: float):
        """Send joint command"""
        msg = JointAnglesWithSpeed()
        msg.joint_names = joint_names
        msg.joint_angles = angles
        msg.speed = speed
        msg.relative = 0
        
        self.joint_pub.publish(msg)
    
    def get_limbs_for_behavior(self, behavior: str) -> List[str]:
        """Get limbs to animate"""
        limbs = []
        
        if behavior in ['All', 'body']:
            limbs.extend(['RArm', 'LArm', 'Leg'])
        
        if behavior in ['All', 'hands']:
            limbs.extend(['RHand', 'LHand'])
        
        return limbs
    
    def move_to_home(self, limb: str):
        """Move to home position"""
        if limb not in self.joints:
            return
        
        joint_def = self.joints[limb]
        self.send_joint_command(joint_def.names, joint_def.home, speed=0.2)
    
    def stop_animation(self):
        """Stop animation"""
        self.get_logger().info('Stopping...')
        
        self.active = False
        
        # Return to home
        for limb in self.limbs_to_animate:
            self.move_to_home(limb)
        
        # Stop rotation
        msg = Twist()
        self.vel_pub.publish(msg)
        
        self.limbs_to_animate = []
        self.last_gesture = {}

def main(args=None):
    rclpy.init(args=args)
    
    node = AnimateBehaviorServer()
    
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()