"""
animate_behavior_implementation.py Implementation code for running the Animate Behavior ROS2 node.

Author: Yohannes Tadesse Haile
Date: April 27, 2026
Version: v1.0

This program comes with ABSOLUTELY NO WARRANTY.
"""

import rclpy
import time
import random
from typing import List, Dict
from dataclasses import dataclass
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import Twist
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from naoqi_bridge_msgs.action import RunLed
from sensor_msgs.msg import JointState
from dec_interfaces.action import AnimateBehavior
from std_srvs.srv import Trigger
from std_msgs.msg import ColorRGBA


@dataclass
class JointDef:
    names: List[str]
    min: List[float]
    max: List[float]
    home: List[float]
    factors: List[float]


class AnimateBehaviorNode(LifecycleNode):
    CASCADE_LAYERS = [
        (['FaceLedRight1', 'FaceLedLeft1'], 0.00),
        (['FaceLedRight0', 'FaceLedRight2', 'FaceLedLeft0', 'FaceLedLeft2'], 0.09),
        (['FaceLedRight7', 'FaceLedRight3', 'FaceLedLeft7', 'FaceLedLeft3'], 0.18),
        (['FaceLedRight6', 'FaceLedRight4', 'FaceLedLeft6', 'FaceLedLeft4'], 0.36),
        (['FaceLedRight5', 'FaceLedLeft5'], 0.54),
    ]

    def __init__(self):
        super().__init__('animate_behavior')

        # Declare parameters in __init__ so they are settable via launch before configure
        self.declare_parameter('verbose_mode', True)
        self.declare_parameter('led_enabled', True)
        self.declare_parameter('led_white_step', 0.06)
        self.declare_parameter('led_dark_step', 0.04)
        self.declare_parameter('led_fade_duration', 0.10)
        self.declare_parameter('led_white_hold', 2.0)
        self.declare_parameter('led_dark_pause', 0.2)
        self.declare_parameter('gesture_update_rate', 30.0)
        self.declare_parameter('gesture_smoothing_factor', 0.15)
        self.declare_parameter('gesture_motion_speed', 0.08)
        self.declare_parameter('gesture_interval_min', 2.5)
        self.declare_parameter('gesture_interval_max', 4.5)
        self.declare_parameter('gesture_rotation_interval', 5.0)

        # Guard flag: set True in on_activate, False in on_deactivate
        self._lifecycle_active = False

    # ── Lifecycle callbacks ─────────────────────────────────────────────────────

    def on_configure(self, _state) -> TransitionCallbackReturn:
        """Read parameters, build joint definitions, create publishers/servers/timers."""
        node_name = self.get_name()

        self.verbose_mode       = self.get_parameter('verbose_mode').get_parameter_value().bool_value
        self.led_enabled        = self.get_parameter('led_enabled').get_parameter_value().bool_value
        self.led_white_step     = self.get_parameter('led_white_step').get_parameter_value().double_value
        self.led_dark_step      = self.get_parameter('led_dark_step').get_parameter_value().double_value
        self.led_fade_duration  = self.get_parameter('led_fade_duration').get_parameter_value().double_value
        self.led_white_hold     = self.get_parameter('led_white_hold').get_parameter_value().double_value
        self.led_dark_pause     = self.get_parameter('led_dark_pause').get_parameter_value().double_value
        self.gesture_interval_min  = self.get_parameter('gesture_interval_min').get_parameter_value().double_value
        self.gesture_interval_max  = self.get_parameter('gesture_interval_max').get_parameter_value().double_value
        self.rotation_interval  = self.get_parameter('gesture_rotation_interval').get_parameter_value().double_value
        self.update_rate        = self.get_parameter('gesture_update_rate').get_parameter_value().double_value
        self.smoothing_factor   = self.get_parameter('gesture_smoothing_factor').get_parameter_value().double_value
        self.motion_speed       = self.get_parameter('gesture_motion_speed').get_parameter_value().double_value

        if self.verbose_mode:
            self.get_logger().info(f'{node_name}: ANIMATE BEHAVIOR NODE (LIFECYCLE)')

        self.callback_group = ReentrantCallbackGroup()

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
            'RHand': JointDef(names=['RHand'], min=[0.0], max=[1.0], home=[0.67], factors=[0.5]),
            'LHand': JointDef(names=['LHand'], min=[0.0], max=[1.0], home=[0.67], factors=[0.5]),
            'Leg': JointDef(
                names=['HipPitch', 'HipRoll', 'KneePitch'],
                min=[-1.04, -0.51, -0.51],
                max=[1.04, 0.51, 0.51],
                home=[-0.10, -0.01, 0.03],
                factors=[0.2, 0.2, 0.1]
            ),
        }

        # Animation state
        self.active                 = False
        self.behavior               = ''
        self.range                  = 0.2
        self.limbs_to_animate       = []
        self.last_gesture           = {}
        self.last_rotation          = 0.0
        self.gesture_count          = 0
        self.current_goal_handle    = None
        self.start_time             = 0.0
        self.duration               = 0.0
        self.goal_complete_event    = None
        self.current_positions: Dict[str, float] = {}
        self.target_positions: Dict[str, float]  = {}
        self.joint_states_received  = False
        self.gesture_interval       = random.uniform(self.gesture_interval_min, self.gesture_interval_max)

        # Managed publishers — silenced when node is INACTIVE
        self.joint_pub = self.create_lifecycle_publisher(JointAnglesWithSpeed, '/joint_angles', 10)
        self.vel_pub   = self.create_lifecycle_publisher(Twist, '/cmd_vel', 10)

        # Action server — created once; goal_callback guards against inactive state
        self.action_server = ActionServer(
            self, AnimateBehavior, '/animate_behavior',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        # Stop service — always available after configure
        self.stop_service = self.create_service(
            Trigger, '/animate_behavior/stop',
            self.stop_service_callback,
            callback_group=self.callback_group,
        )

        # LED action client
        self.led_client    = ActionClient(self, RunLed, '/naoqi_driver/run_led')
        self.led_active    = False
        self.led_scheduled_timers: List = []

        self.get_logger().info(f'{node_name}: configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        """Activate managed publishers, start subscription and animation timers."""
        super().on_activate(state)
        self._lifecycle_active = True

        # Joint state subscription — only needed while active
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10
        )

        # Animation timers
        self.animation_timer = self.create_timer(
            1.0 / self.update_rate, self.animation_update,
            callback_group=self.callback_group,
        )
        self.feedback_timer = self.create_timer(
            0.5, self.feedback_update,
            callback_group=self.callback_group,
        )

        # Start LED animation if the server is available
        if self.led_enabled:
            if self.led_client.wait_for_server(timeout_sec=5.0):
                self.led_active = True
                self.cascade_wave()
                if self.verbose_mode:
                    self.get_logger().info(f'{self.get_name()}: LED animation enabled')
            else:
                if self.verbose_mode:
                    self.get_logger().warn(f'{self.get_name()}: LED server not available — disabled')
        else:
            if self.verbose_mode:
                self.get_logger().info(f'{self.get_name()}: LED animation disabled in config')

        self.get_logger().info(
            f'{self.get_name()}: activated — animation@{self.update_rate}Hz, '
            f'feedback@2Hz | waiting for goals on /animate_behavior'
        )
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        """Stop animation, cancel timers, destroy subscription, deactivate publishers."""
        self._lifecycle_active = False
        self.stop()

        self.animation_timer.cancel()
        self.destroy_timer(self.animation_timer)
        self.feedback_timer.cancel()
        self.destroy_timer(self.feedback_timer)

        self.destroy_subscription(self.joint_sub)

        super().on_deactivate(state)
        self.get_logger().info(f'{self.get_name()}: deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        self.destroy_lifecycle_publisher(self.joint_pub)
        self.destroy_lifecycle_publisher(self.vel_pub)
        self.action_server.destroy()
        self.destroy_service(self.stop_service)
        self.led_client.destroy()
        self.get_logger().info(f'{self.get_name()}: cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        self.get_logger().info(f'{self.get_name()}: shutting down')
        return TransitionCallbackReturn.SUCCESS

    # ── Subscription callback ────────────────────────────────────────────────────

    def joint_states_callback(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_positions[name] = msg.position[i]
        if not self.joint_states_received:
            self.joint_states_received = True
            if self.verbose_mode:
                self.get_logger().info(
                    f'{self.get_name()}: joint states received: {len(self.current_positions)} joints'
                )

    # ── Action server callbacks ──────────────────────────────────────────────────

    def goal_callback(self, goal_request):
        if not self._lifecycle_active:
            self.get_logger().warn(f'{self.get_name()}: rejecting goal — node is not active')
            return GoalResponse.REJECT
        if self.verbose_mode:
            self.get_logger().info(
                f'{self.get_name()}: goal: {goal_request.behavior_type}, '
                f'range={goal_request.selected_range}, duration={goal_request.duration_seconds}s'
            )
        valid_behaviors = ['All', 'body', 'head', 'arms', 'hands', 'idle']
        if goal_request.behavior_type not in valid_behaviors:
            self.get_logger().error(f'{self.get_name()}: invalid behavior: {goal_request.behavior_type}')
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        if self.verbose_mode:
            self.get_logger().info(f'{self.get_name()}: cancel requested')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.current_goal_handle = goal_handle
        self.start_time = self.get_clock().now()
        self.duration   = goal_handle.request.duration_seconds
        self.behavior   = goal_handle.request.behavior_type
        self.range      = goal_handle.request.selected_range

        if self.verbose_mode:
            self.get_logger().info(
                f'{self.get_name()}: starting behavior: {self.behavior}, range={self.range}'
            )

        self.limbs_to_animate = []
        if self.behavior in ['All', 'body']:
            self.limbs_to_animate.extend(['RArm', 'LArm', 'RHand', 'LHand', 'Leg'])
        if self.behavior in ['All', 'head']:
            self.limbs_to_animate.append('Head')
        if self.behavior in ['All', 'arms']:
            self.limbs_to_animate.extend(['RArm', 'LArm'])
        if self.behavior in ['All', 'hands']:
            self.limbs_to_animate.extend(['RHand', 'LHand'])
        if self.behavior == 'idle':
            self.limbs_to_animate = []

        self.active           = True
        self.gesture_count    = 0
        self.last_gesture     = {}
        self.last_rotation    = 0.0
        self.gesture_interval = random.uniform(self.gesture_interval_min, self.gesture_interval_max)

        goal_handle.execute_async()

    def stop_service_callback(self, request, response):
        if self.verbose_mode:
            self.get_logger().info(f'{self.get_name()}: stop service called')
        self.stop()
        response.success = True
        response.message = 'Animation stopped'
        return response

    def stop(self):
        self.active = False
        msg = Twist()
        self.vel_pub.publish(msg)
        self.stop_leds()
        self.limbs_to_animate = []
        self.last_gesture     = {}
        self.target_positions.clear()

    # ── Timer callbacks ──────────────────────────────────────────────────────────

    def feedback_update(self):
        if self.current_goal_handle is None or not self.active:
            return
        elapsed  = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        feedback = AnimateBehavior.Feedback()
        feedback.elapsed_seconds = elapsed
        self.current_goal_handle.publish_feedback(feedback)

        if self.duration > 0 and elapsed >= self.duration:
            if self.verbose_mode:
                self.get_logger().info(f'{self.get_name()}: duration reached: {self.duration}s')
            self.stop()
            result = AnimateBehavior.Result()
            result.completed             = True
            result.final_elapsed_seconds = elapsed
            self.current_goal_handle.set_succeeded(result)
            self.current_goal_handle = None

    def animation_update(self):
        if not self.active or not self.joint_states_received:
            return

        current_time = time.time()

        if self.limbs_to_animate:
            if current_time - self.last_rotation >= self.rotation_interval:
                self.apply_rotation()
                self.last_rotation = current_time

            if current_time - self.gesture_count >= self.gesture_interval:
                self.apply_gesture()
                self.gesture_count    = current_time
                self.gesture_interval = random.uniform(self.gesture_interval_min, self.gesture_interval_max)

        self.publish_joints()

    # ── Motion helpers ───────────────────────────────────────────────────────────

    def apply_rotation(self):
        twist = Twist()
        twist.angular.z = 0.3
        self.vel_pub.publish(twist)

    def apply_gesture(self):
        for limb in self.limbs_to_animate:
            if limb in self.joints:
                self.random_gesture(limb)

    def random_gesture(self, limb_name: str):
        joint_def = self.joints[limb_name]
        for i, name in enumerate(joint_def.names):
            factor    = joint_def.factors[i]
            rand_val  = random.uniform(-1, 1) * self.range * factor
            mid       = (joint_def.max[i] + joint_def.min[i]) / 2
            target    = mid + rand_val
            target    = max(joint_def.min[i], min(joint_def.max[i], target))
            self.target_positions[name] = target
        self.last_gesture[limb_name] = time.time()

    def publish_joints(self):
        msg = JointAnglesWithSpeed()
        msg.header.stamp = self.get_clock().now().to_msg()
        names  = []
        angles = []
        for name, target in self.target_positions.items():
            current  = self.current_positions.get(name, target)
            smoothed = current + self.smoothing_factor * (target - current)
            self.current_positions[name] = smoothed
            names.append(name)
            angles.append(smoothed)
        msg.joint_names  = names
        msg.joint_angles = angles
        msg.speed        = self.motion_speed
        if names:
            self.joint_pub.publish(msg)

    # ── LED helpers ──────────────────────────────────────────────────────────────

    def make_color(self, r: float, g: float, b: float) -> ColorRGBA:
        c = ColorRGBA()
        c.r = float(r)
        c.g = float(g)
        c.b = float(b)
        c.a = 1.0
        return c

    def send_rgb_fade(self, r: float, g: float, b: float, duration: float, target: str):
        goal          = RunLed.Goal()
        goal.target   = target
        goal.mode     = RunLed.Goal.MODE_RGB_FADE
        goal.color    = self.make_color(r, g, b)
        goal.duration = float(duration)
        self.led_client.send_goal_async(goal)

    def send_off(self, target: str):
        goal        = RunLed.Goal()
        goal.target = target
        goal.mode   = RunLed.Goal.MODE_OFF
        self.led_client.send_goal_async(goal)

    def schedule_led(self, delay_sec: float, callback):
        ref = [None]
        def fire():
            ref[0].cancel()
            if self.led_active:
                callback()
        timer = self.create_timer(delay_sec, fire, callback_group=self.callback_group)
        ref[0] = timer
        self.led_scheduled_timers.append(ref)

    def cascade_wave(self):
        n          = len(self.CASCADE_LAYERS)
        reversed_l = list(reversed(self.CASCADE_LAYERS))

        for i, (names, _) in enumerate(reversed_l):
            t = i * self.led_white_step
            for name in names:
                self.schedule_led(
                    t,
                    lambda nm=name: self.send_rgb_fade(1.0, 1.0, 1.0, self.led_fade_duration, target=nm)
                )

        t_hold_end = (n - 1) * self.led_white_step + self.led_fade_duration + self.led_white_hold
        for i, (names, _) in enumerate(self.CASCADE_LAYERS):
            t = t_hold_end + i * self.led_dark_step
            for name in names:
                self.schedule_led(
                    t,
                    lambda nm=name: self.send_rgb_fade(0.0, 0.0, 0.0, self.led_fade_duration, target=nm)
                )

        last_dark = t_hold_end + (n - 1) * self.led_dark_step
        self.schedule_led(last_dark + self.led_fade_duration + self.led_dark_pause, self.cascade_wave)

    def stop_leds(self):
        self.led_active = False
        for names, _ in self.CASCADE_LAYERS:
            for n in names:
                self.send_off(target=n)
