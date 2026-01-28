#!/usr/bin/env python3
"""
animate_behavior.py

Simplified action server with extensive debug output.
"""

import rclpy
import time
import random
from typing import List
from dataclasses import dataclass
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from geometry_msgs.msg import Twist
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from sensor_msgs.msg import JointState
from cssr_interfaces.action import AnimateBehavior

@dataclass
class JointDef:
    """Simple joint definition"""
    names: List[str]
    min: List[float]
    max: List[float]
    home: List[float]
    factors: List[float]

class AnimateBehaviorSimple(Node):
    """Simplified animate behavior with debug output"""
    
    def __init__(self):
        super().__init__('animate_behavior')
        
        self.get_logger().info('='*60)
        self.get_logger().info('INITIALIZING ANIMATE BEHAVIOR NODE')
        self.get_logger().info('='*60)
        
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
        
        self.get_logger().info(f'Loaded {len(self.joints)} limb definitions')
        for limb_name, joint_def in self.joints.items():
            self.get_logger().info(f'  {limb_name}: {len(joint_def.names)} joints')
        
        # State
        self.active = False
        self.behavior = ''
        self.range = 0.5
        self.last_gesture = {}
        self.last_rotation = 0.0
        self.gesture_count = 0
        
        # Timing
        self.gesture_interval = 0.2  # seconds
        self.rotation_interval = 2.0  # seconds
        
        # Joint states from robot
        self.current_joint_states = {}
        self.joint_states_received = False
        
        # Publishers
        self.get_logger().info('Creating publishers...')
        self.joint_pub = self.create_publisher(JointAnglesWithSpeed, '/joint_angles', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info(f'  Joint publisher: /joint_angles')
        self.get_logger().info(f'  Velocity publisher: /cmd_vel')
        
        # Subscriber
        self.get_logger().info('Creating subscriber...')
        self.joint_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_states_callback, 
            10
        )
        self.get_logger().info(f'  Joint states subscriber: /joint_states')
        
        # Action server
        self.get_logger().info('Creating action server...')
        self._action_server = ActionServer(
            self, 
            AnimateBehavior, 
            'animate_behavior',
            execute_callback=self.execute,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )
        self.get_logger().info(f'  Action server: /animate_behavior')
        
        # Main timer
        self.get_logger().info('Creating update timer at 20Hz...')
        self.timer = self.create_timer(0.05, self.update)
        
        self.get_logger().info('='*60)
        self.get_logger().info('NODE READY - Waiting for action goals')
        self.get_logger().info('Send a goal with:')
        self.get_logger().info('  ros2 action send_goal /animate_behavior \\')
        self.get_logger().info('    animate_behavior_interfaces/action/AnimateBehavior \\')
        self.get_logger().info('    "{behavior_type: \'All\', selected_range: 0.5, duration_seconds: 10}" \\')
        self.get_logger().info('    --feedback')
        self.get_logger().info('='*60)
    
    def joint_states_callback(self, msg: JointState):
        """Receive joint states from robot"""
        if not self.joint_states_received:
            self.get_logger().info(f'✓ Received joint states: {len(msg.name)} joints')
            self.joint_states_received = True
        
        for name, position in zip(msg.name, msg.position):
            self.current_joint_states[name] = position
    
    def goal_callback(self, goal_request):
        """Handle incoming goal"""
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('GOAL RECEIVED')
        self.get_logger().info('='*60)
        self.get_logger().info(f'  behavior_type: {goal_request.behavior_type}')
        self.get_logger().info(f'  selected_range: {goal_request.selected_range}')
        self.get_logger().info(f'  duration_seconds: {goal_request.duration_seconds}')
        
        # Validate
        valid_behaviors = ['All', 'body', 'hands', 'rotation']
        if goal_request.behavior_type not in valid_behaviors:
            self.get_logger().error(f'❌ Invalid behavior type: {goal_request.behavior_type}')
            self.get_logger().error(f'   Valid types: {valid_behaviors}')
            return GoalResponse.REJECT
        
        if not 0.0 <= goal_request.selected_range <= 1.0:
            self.get_logger().error(f'❌ Invalid range: {goal_request.selected_range}')
            self.get_logger().error(f'   Must be between 0.0 and 1.0')
            return GoalResponse.REJECT
        
        self.get_logger().info('✓ Goal validated and ACCEPTED')
        self.get_logger().info('='*60)
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Handle cancel request"""
        self.get_logger().warn('')
        self.get_logger().warn('⚠ CANCEL REQUESTED')
        return CancelResponse.ACCEPT
    
    def execute(self, goal_handle):
        """Execute action"""
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('STARTING EXECUTION')
        self.get_logger().info('='*60)
        
        # Stop previous
        if self.active:
            self.get_logger().warn('⚠ Stopping previous execution')
            self.stop()
        
        # Initialize
        goal = goal_handle.request
        self.behavior = goal.behavior_type
        self.range = goal.selected_range
        self.active = True
        self.gesture_count = 0
        start_time = time.time()
        
        # Get limbs to animate
        limbs = self.get_limbs(self.behavior)
        self.get_logger().info(f'Limbs to animate: {limbs}')
        
        # Reset timers
        self.last_gesture = {}
        for limb in limbs:
            self.last_gesture[limb] = 0.0
        self.last_rotation = 0.0
        
        self.get_logger().info(f'Initialized {len(self.last_gesture)} limb timers')
        
        # Move to home
        self.get_logger().info('Moving to home positions...')
        self.move_home(limbs)
        self.get_logger().info('✓ Home positions sent')
        
        self.get_logger().info('='*60)
        self.get_logger().info('EXECUTION ACTIVE - Timer will handle updates')
        self.get_logger().info('='*60)
        
        # Monitor loop
        rate = self.create_rate(10)
        feedback_counter = 0
        
        while rclpy.ok() and self.active:
            # Check cancel
            if goal_handle.is_cancel_requested:
                self.get_logger().warn('')
                self.get_logger().warn('❌ Goal cancelled by client')
                self.stop()
                goal_handle.canceled()
                return AnimateBehavior.Result(
                    success=False,
                    message='Cancelled by client',
                    total_duration=time.time() - start_time
                )
            
            # Check duration
            elapsed = time.time() - start_time
            if goal.duration_seconds > 0 and elapsed >= goal.duration_seconds:
                self.get_logger().info('')
                self.get_logger().info(f'✓ Duration reached: {elapsed:.1f}s')
                break
            
            # Publish feedback
            feedback_counter += 1
            if feedback_counter >= 10:  # Every second (10 * 0.1s)
                feedback = AnimateBehavior.Feedback()
                feedback.current_limb = self.behavior
                feedback.gestures_completed = self.gesture_count
                feedback.elapsed_time = elapsed
                feedback.is_running = self.active
                goal_handle.publish_feedback(feedback)
                
                self.get_logger().info(
                    f'[{elapsed:.1f}s] Gestures: {self.gesture_count}, Active: {self.active}'
                )
                feedback_counter = 0
            
            rate.sleep()
        
        # Complete
        self.get_logger().info('')
        self.get_logger().info('='*60)
        self.get_logger().info('EXECUTION COMPLETE')
        self.get_logger().info('='*60)
        self.stop()
        goal_handle.succeed()
        
        result = AnimateBehavior.Result(
            success=True,
            message=f'Completed {self.gesture_count} gestures',
            total_duration=time.time() - start_time
        )
        
        self.get_logger().info(f'✓ Success: {result.message}')
        self.get_logger().info(f'✓ Duration: {result.total_duration:.1f}s')
        self.get_logger().info('='*60)
        
        return result
    
    def update(self):
        """Main update loop - 20Hz"""
        if not self.active:
            return
        
        now = time.time()
        
        # Update limbs
        for limb in self.last_gesture.keys():
            time_since = now - self.last_gesture[limb]
            
            if time_since >= self.gesture_interval:
                self.get_logger().info(
                    f'[UPDATE] {limb}: time_since={time_since:.3f}s >= interval={self.gesture_interval}s'
                )
                self.animate_limb(limb)
                self.last_gesture[limb] = now
                self.gesture_count += 1
        
        # Update rotation
        if 'rotation' in self.behavior or self.behavior == 'All':
            time_since_rot = now - self.last_rotation
            
            if time_since_rot >= self.rotation_interval:
                self.get_logger().info(
                    f'[UPDATE] Rotation: time_since={time_since_rot:.3f}s >= interval={self.rotation_interval}s'
                )
                self.animate_rotation()
                self.last_rotation = now
    
    def animate_limb(self, limb: str):
        """Animate one limb"""
        self.get_logger().info(f'')
        self.get_logger().info(f'--- Animating {limb} ---')
        
        joint_def = self.joints[limb]
        target = []
        
        # Generate random position
        for i in range(len(joint_def.names)):
            center = joint_def.home[i]
            full_range = joint_def.max[i] - joint_def.min[i]
            allowed = full_range * joint_def.factors[i] * self.range / 2.0
            
            low = max(center - allowed, joint_def.min[i])
            high = min(center + allowed, joint_def.max[i])
            
            random_val = random.uniform(low, high)
            target.append(random_val)
            
            self.get_logger().info(
                f'  {joint_def.names[i]}: '
                f'home={center:.2f}, '
                f'range=[{low:.2f}, {high:.2f}], '
                f'target={random_val:.2f}'
            )
        
        # Send command
        self.get_logger().info(f'  Publishing to /joint_angles')
        self.send_joints(joint_def.names, target, 0.15)
        self.get_logger().info(f'✓ {limb} command sent')
    
    def animate_rotation(self):
        """Animate rotation"""
        vel = random.uniform(-0.3 * self.range, 0.3 * self.range)
        
        self.get_logger().info('')
        self.get_logger().info('--- Animating Rotation ---')
        self.get_logger().info(f'  Angular velocity: {vel:.3f} rad/s')
        self.get_logger().info(f'  Publishing to /cmd_vel')
        
        msg = Twist()
        msg.angular.z = vel
        self.vel_pub.publish(msg)
        
        self.get_logger().info(f'✓ Rotation command sent')
        
        # Schedule stop
        def stop():
            self.get_logger().info('  Stopping rotation')
            msg = Twist()
            self.vel_pub.publish(msg)
        
        self.create_timer(self.rotation_interval / 2, stop, oneshot=True)
    
    def send_joints(self, names: List[str], angles: List[float], speed: float):
        """Send joint command"""
        msg = JointAnglesWithSpeed()
        msg.joint_names = names
        msg.joint_angles = angles
        msg.speed = speed
        msg.relative = 0
        
        # Debug output
        self.get_logger().info(f'  Message contents:')
        self.get_logger().info(f'    joint_names: {msg.joint_names}')
        self.get_logger().info(f'    joint_angles: {[f"{a:.2f}" for a in msg.joint_angles]}')
        self.get_logger().info(f'    speed: {msg.speed}')
        self.get_logger().info(f'    relative: {msg.relative}')
        
        self.joint_pub.publish(msg)
        
        # Check if anyone is listening
        num_connections = self.joint_pub.get_subscription_count()
        if num_connections == 0:
            self.get_logger().warn(f'  ⚠ WARNING: No subscribers on /joint_angles!')
            self.get_logger().warn(f'  ⚠ Is naoqi_driver running?')
        else:
            self.get_logger().info(f'  ✓ Message published ({num_connections} subscribers)')
    
    def get_limbs(self, behavior: str) -> List[str]:
        """Get limbs for behavior"""
        limbs = []
        
        if behavior in ['All', 'body']:
            limbs.extend(['RArm', 'LArm', 'Leg'])
            self.get_logger().info(f'  Added body limbs: RArm, LArm, Leg')
        
        if behavior in ['All', 'hands']:
            limbs.extend(['RHand', 'LHand'])
            self.get_logger().info(f'  Added hand limbs: RHand, LHand')
        
        if behavior == 'rotation':
            self.get_logger().info(f'  Rotation only (no limbs)')
        
        return limbs
    
    def move_home(self, limbs: List[str]):
        """Move to home positions"""
        for limb in limbs:
            joint_def = self.joints[limb]
            self.get_logger().info(f'  {limb} -> home: {[f"{h:.2f}" for h in joint_def.home]}')
            self.send_joints(joint_def.names, joint_def.home, 0.2)
        
        self.get_logger().info('  Waiting 0.5s for movement...')
        time.sleep(0.5)
    
    def stop(self):
        """Stop all movement"""
        self.get_logger().info('')
        self.get_logger().info('--- STOPPING ---')
        
        self.active = False
        
        # Stop limbs
        for limb in self.last_gesture.keys():
            joint_def = self.joints[limb]
            self.get_logger().info(f'  Stopping {limb}')
            self.send_joints(joint_def.names, joint_def.home, 0.0)
        
        # Stop rotation
        self.get_logger().info(f'  Stopping rotation')
        msg = Twist()
        self.vel_pub.publish(msg)
        
        self.last_gesture = {}
        self.get_logger().info('✓ All movement stopped')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = AnimateBehaviorSimple()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n[INFO] Keyboard interrupt received')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()