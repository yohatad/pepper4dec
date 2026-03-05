#!/usr/bin/env python3
"""
animate_behavior.py - Fully non-blocking version

The execute_callback returns immediately and everything is handled by timers.
"""

import rclpy
import time
import random
import threading
from typing import List, Dict
from dataclasses import dataclass
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from sensor_msgs.msg import JointState
from cssr_interfaces.action import AnimateBehavior
from std_srvs.srv import Trigger

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
                home=[-0.10, -0.01, 0.03],
                factors=[0.2, 0.2, 0.1]
            )
        }
        
        # Animation state
        self.active = False
        self.behavior = ''
        self.range = 0.2
        self.limbs_to_animate = []
        self.last_gesture = {}
        self.last_rotation = 0.0
        self.gesture_count = 0

        # Goal tracking
        self.current_goal_handle = None
        self.start_time = 0.0
        self.duration = 0.0
        self.goal_complete_event = None

        # Smooth motion state
        self.current_positions: Dict[str, float] = {}  # Current joint positions
        self.target_positions: Dict[str, float] = {}   # Target positions for smooth interpolation
        self.joint_states_received = False

        # Timing intervals
        self.gesture_interval = random.uniform(2.5, 4.0)  # Variable timing for naturalness
        self.rotation_interval = 5.0  # seconds between rotations

        # Motion parameters
        self.update_rate = 30.0  # Hz - high frequency for smooth motion
        self.smoothing_factor = 0.15  # Exponential smoothing (0.1-0.2 good range)
        self.motion_speed = 0.08  # Speed parameter for ALMotion
        
        # Publishers
        self.joint_pub = self.create_publisher(JointAnglesWithSpeed, '/joint_angles', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber
        self.joint_sub = self.create_subscription(JointState, '/joint_states', 
            self.joint_states_callback, 10)
        
        # Action server
        self._action_server = ActionServer(self,
            AnimateBehavior,
            'animate_behavior',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group
        )

        # Service for explicit stop control from BT
        self.stop_service = self.create_service(Trigger,
            'animate_behavior/stop',
            self.stop_service_callback,
            callback_group=self.callback_group
        )
        
        # Main animation timer - 30Hz for smooth motion
        self.animation_timer = self.create_timer(1.0 / self.update_rate,
            self.animation_update,
            callback_group=self.callback_group)
        
        # Feedback timer - 2Hz
        self.feedback_timer = self.create_timer(0.5, self.feedback_update,
            callback_group=self.callback_group
        )
        
        self.get_logger().info(f'✓ Node ready, animation at {self.update_rate}Hz, feedback at 2Hz')
        self.get_logger().info('✓ Waiting for goals on /animate_behavior')
        self.get_logger().info('✓ Stop service available at /animate_behavior/stop')
        self.get_logger().info('='*60)
    
    def joint_states_callback(self, msg: JointState):
        """Receive joint states - track current positions for smooth motion"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]

        if not self.joint_states_received:
            self.joint_states_received = True
            self.get_logger().info(f'✓ Joint states received: {len(self.current_positions)} joints')
    
    def goal_callback(self, goal_request):
        """Validate goal"""
        self.get_logger().info(f'Goal: {goal_request.behavior_type}, '
                              f'range={goal_request.selected_range}, '
                              f'duration={goal_request.duration_seconds}s')

        valid_behaviors = ['All', 'body', 'hands', 'arms', 'rotation', 'home']
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

    def stop_service_callback(self, request, response):
        """Service callback for explicit stop from BT"""
        if self.active:
            self.get_logger().info('Stop service called - cancelling animation')
            # Cancel the current goal if active
            if self.current_goal_handle:
                self.cancel_goal()
            else:
                # If no goal but still active, just stop
                self.stop_animation()
            response.success = True
            response.message = 'Animation stopped successfully'
        else:
            self.get_logger().info('Stop service called but no animation active')
            response.success = True
            response.message = 'No animation was running'
        return response
    
    def execute_callback(self, goal_handle):
        """Execute - waits for timers to complete the animation"""
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

        # Create event to wait for completion
        self.goal_complete_event = threading.Event()

        # Check if this is a "home" command - special behavior to return to home
        if self.behavior == 'home':
            self.get_logger().info('Home position requested - returning all limbs to home')
            # Get all limbs that could be animated
            all_limbs = ['RArm', 'LArm', 'RHand', 'LHand', 'Leg']
            for limb in all_limbs:
                self.move_to_home(limb)

            # Stop any rotation
            msg = Twist()
            self.vel_pub.publish(msg)

            # Return immediately with success
            result = AnimateBehavior.Result()
            result.success = True
            result.message = 'Returned to home position'
            result.total_duration = 0.0
            goal_handle.succeed()
            return result

        # Normal animation behaviors
        # Get limbs
        self.limbs_to_animate = self.get_limbs_for_behavior(self.behavior)
        self.get_logger().info(f'Animating: {", ".join(self.limbs_to_animate) if self.limbs_to_animate else "rotation only"}')

        # Initialize timing with random offsets for natural staggered motion
        self.last_gesture = {}
        for limb in self.limbs_to_animate:
            # Stagger the start times slightly for more natural motion
            self.last_gesture[limb] = -random.uniform(0, 2.0)
        self.last_rotation = 0.0

        # Initialize target positions to home
        self.get_logger().info('Initializing targets to home positions...')
        for limb in self.limbs_to_animate:
            self.initialize_limb_targets(limb)

        # Activate - timers will now do the work
        self.active = True

        self.get_logger().info('✓ Animation ACTIVE - waiting for completion')
        self.get_logger().info('  Timers will handle animation and completion')
        self.get_logger().info('='*60)

        # Wait for the goal to complete (timer will set this event)
        self.goal_complete_event.wait()

        # Return the result (set by complete_goal or cancel_goal)
        return self.goal_result
    
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

        # Generate new target positions periodically
        for limb in self.limbs_to_animate:
            if limb in self.last_gesture:
                time_since = now - self.last_gesture[limb]
                if time_since >= self.gesture_interval:
                    # Generate new random target for this limb
                    self.set_new_limb_target(limb)
                    self.last_gesture[limb] = now
                    # Vary the next interval for natural timing
                    self.gesture_interval = random.uniform(2.5, 4.5)
                    self.gesture_count += 1

        # Smoothly move towards all target positions (30Hz updates)
        self.update_smooth_motion()

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

        self.goal_result = AnimateBehavior.Result(
            success=True,
            message=f'Completed {self.gesture_count} gestures',
            total_duration=elapsed
        )

        self.get_logger().info('='*60)
        self.get_logger().info(f'✓ Complete: {self.goal_result.message} in {elapsed:.1f}s')
        self.get_logger().info('='*60)

        # Mark goal as succeeded BEFORE signaling the event
        self.current_goal_handle.succeed()
        self.current_goal_handle = None

        # Signal execute_callback to return
        if self.goal_complete_event:
            self.goal_complete_event.set()
    
    def cancel_goal(self):
        """Cancel the current goal"""
        if not self.current_goal_handle:
            return

        elapsed = time.time() - self.start_time

        self.stop_animation()

        self.goal_result = AnimateBehavior.Result(
            success=False,
            message=f'Cancelled after {elapsed:.1f}s',
            total_duration=elapsed
        )

        self.get_logger().info(f'✓ Cancelled after {elapsed:.1f}s')

        # Mark goal as canceled BEFORE signaling the event
        self.current_goal_handle.canceled()
        self.current_goal_handle = None

        # Signal execute_callback to return
        if self.goal_complete_event:
            self.goal_complete_event.set()
    
    def initialize_limb_targets(self, limb: str):
        """Initialize target positions for a limb to home position"""
        if limb not in self.joints:
            return

        joint_def = self.joints[limb]
        for i, joint_name in enumerate(joint_def.names):
            self.target_positions[joint_name] = joint_def.home[i]
            # Initialize current position if not yet received from joint_states
            if joint_name not in self.current_positions:
                self.current_positions[joint_name] = joint_def.home[i]

    def set_new_limb_target(self, limb: str):
        """Generate new random target position for a limb"""
        if limb not in self.joints:
            return

        joint_def = self.joints[limb]

        for i in range(len(joint_def.names)):
            joint_name = joint_def.names[i]
            center = joint_def.home[i]
            full_range = joint_def.max[i] - joint_def.min[i]
            allowed_range = full_range * joint_def.factors[i] * self.range / 2.0

            min_angle = max(center - allowed_range, joint_def.min[i])
            max_angle = min(center + allowed_range, joint_def.max[i])

            # Generate new target with some bias towards center for natural return motion
            if random.random() < 0.3:  # 30% chance to return towards home
                target = center + random.uniform(-allowed_range * 0.3, allowed_range * 0.3)
            else:
                target = random.uniform(min_angle, max_angle)

            self.target_positions[joint_name] = target

        self.get_logger().info(f'  → New target: {limb}')

    def update_smooth_motion(self):
        """Update all joints smoothly towards their targets - called at 30Hz"""
        if not self.target_positions:
            return

        # Collect all joint updates
        joints_to_update = {}

        for joint_name, target in self.target_positions.items():
            current = self.current_positions.get(joint_name, target)

            # Exponential smoothing: move fraction of distance towards target
            error = target - current
            new_position = current + error * self.smoothing_factor

            # Update tracked position
            self.current_positions[joint_name] = new_position
            joints_to_update[joint_name] = new_position

        # Group by limb and send commands
        self.send_smooth_updates(joints_to_update)

    def send_smooth_updates(self, joint_updates: Dict[str, float]):
        """Send smooth position updates grouped by limb"""
        # Group joints by limb for efficient commands
        limb_updates = {}

        for limb_name, joint_def in self.joints.items():
            limb_joints = []
            limb_positions = []

            for joint_name in joint_def.names:
                if joint_name in joint_updates:
                    limb_joints.append(joint_name)
                    limb_positions.append(joint_updates[joint_name])

            if limb_joints:
                limb_updates[limb_name] = (limb_joints, limb_positions)

        # Send all updates
        for limb_name, (joints, positions) in limb_updates.items():
            if joints:
                self.send_joint_command(joints, positions, speed=self.motion_speed)
    
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

        if behavior in ['All', 'hands', 'arms']:
            limbs.extend(['RHand', 'LHand'])

        if behavior == 'arms':
            limbs.extend(['RArm', 'LArm'])

        return limbs
    
    def move_to_home(self, limb: str):
        """Move to home position"""
        if limb not in self.joints:
            return
        
        joint_def = self.joints[limb]
        self.send_joint_command(joint_def.names, joint_def.home, speed=0.1)
    
    def stop_animation(self):
        """Stop animation"""
        self.get_logger().info('Stopping...')

        self.active = False

        # Set targets to home for smooth return
        for limb in self.limbs_to_animate:
            joint_def = self.joints[limb]
            for i, joint_name in enumerate(joint_def.names):
                self.target_positions[joint_name] = joint_def.home[i]

        # Send immediate home command
        for limb in self.limbs_to_animate:
            self.move_to_home(limb)

        # Stop rotation
        msg = Twist()
        self.vel_pub.publish(msg)

        self.limbs_to_animate = []
        self.last_gesture = {}
        self.target_positions.clear()

    def cleanup(self):
        """Cleanup method for safe shutdown"""
        if self.active:
            self.active = False

            # Stop rotation without logging
            msg = Twist()
            try:
                self.vel_pub.publish(msg)
            except Exception:
                pass

            # Signal any waiting goal to complete
            if self.goal_complete_event:
                self.goal_complete_event.set()
