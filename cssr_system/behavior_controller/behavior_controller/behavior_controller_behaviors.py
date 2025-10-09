#!/usr/bin/env python3
"""
Tour Guide BehaviorTree Nodes
All custom behaviors for the museum tour guide robot
"""

import py_trees
from py_trees.common import Status, Access
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

# =============================================================================
# Helpers
# =============================================================================

_GLOBAL_BB = None
def get_blackboard() -> py_trees.blackboard.Client:
    """Singleton client so every behaviour shares the same blackboard handle."""
    global _GLOBAL_BB
    if _GLOBAL_BB is None:
        _GLOBAL_BB = py_trees.blackboard.Client(name="GlobalBlackboard")
    return _GLOBAL_BB

def ros_time_sec(node: Node) -> float:
    """Return ROS time (seconds, float). Respects /use_sim_time."""
    return node.get_clock().now().nanoseconds / 1e9


# =============================================================================
# Action Behaviours
# =============================================================================

class StartOfTree(py_trees.behaviour.Behaviour):
    """Initialize tour with unique ID"""
    def __init__(self, name="StartOfTree", node: Node=None):
        super().__init__(name)
        self.node = node
        self._bb = None
        self._initialised_once = False

    def setup(self, **kwargs):
        self.logger.info(f"  {self.name} [setup]")
        self._bb = get_blackboard()
        self._bb.register_key(key='tour_id', access=Access.WRITE)

    def update(self):
        tour_id = f"tour_{int(ros_time_sec(self.node)*1e9)}"
        if not self._initialised_once:
            self.node.get_logger().info(f"Starting new tour: {tour_id}")
            self._initialised_once = True
        self._bb.tour_id = tour_id
        return Status.SUCCESS

    def terminate(self, new_status):
        if new_status != Status.RUNNING:
            self._initialised_once = False


class SetAnimateBehavior(py_trees.behaviour.Behaviour):
    """Enable/disable robot animation behaviours"""
    def __init__(self, name="SetAnimateBehavior", node: Node=None, animate_enabled=True):
        super().__init__(name)
        self.node = node
        self.animate_enabled = animate_enabled
        self.pub = None
        self._sent_once = False

    def setup(self, **kwargs):
        self.pub = self.node.create_publisher(Bool, '/robot/animate_behavior', 10)
        self.logger.info(f"  {self.name} [setup]")

    def initialise(self):
        self._sent_once = False

    def update(self):
        if not self._sent_once:
            msg = Bool(data=self.animate_enabled)
            self.pub.publish(msg)
            self._sent_once = True
            self.node.get_logger().debug(f"Set animate behavior: {self.animate_enabled}")
        return Status.SUCCESS


class SetOvertAttentionMode(py_trees.behaviour.Behaviour):
    """Control robot's attention/gaze behaviour"""
    def __init__(self, name="SetOvertAttentionMode", node: Node=None, mode="social"):
        super().__init__(name)
        self.node = node
        self.mode = mode
        self.pub = None
        self._sent_once = False

    def setup(self, **kwargs):
        self.pub = self.node.create_publisher(String, '/robot/attention_mode', 10)
        self.logger.info(f"  {self.name} [setup]")

    def initialise(self):
        self._sent_once = False

    def update(self):
        if not self._sent_once:
            self.pub.publish(String(data=self.mode))
            self._sent_once = True
            self.node.get_logger().debug(f"Set attention mode: {self.mode}")
        return Status.SUCCESS


class PerformIconicGesture(py_trees.behaviour.Behaviour):
    """Perform predefined gestures (wave, welcome, goodbye)"""
    def __init__(self, name="PerformIconicGesture", node: Node=None, gesture_type="wave", duration=2.0):
        super().__init__(name)
        self.node = node
        self.gesture_type = gesture_type
        self.duration = duration
        self.pub = None
        self.start_t = None

    def setup(self, **kwargs):
        self.pub = self.node.create_publisher(String, '/robot/perform_gesture', 10)
        self.logger.info(f"  {self.name} [setup]")

    def initialise(self):
        self.pub.publish(String(data=self.gesture_type))
        self.start_t = ros_time_sec(self.node)
        self.node.get_logger().info(f"Performing gesture: {self.gesture_type}")

    def update(self):
        if self.start_t is None:
            return Status.RUNNING
        if ros_time_sec(self.node) - self.start_t >= self.duration:
            return Status.SUCCESS
        return Status.RUNNING


class PerformDeicticGesture(py_trees.behaviour.Behaviour):
    """Point to a specific location (current exhibit)"""
    def __init__(self, name="PerformDeicticGesture", node: Node=None, duration=2.0):
        super().__init__(name)
        self.node = node
        self.duration = duration
        self.pub = None
        self.start_t = None
        self._bb = None

    def setup(self, **kwargs):
        self.pub = self.node.create_publisher(String, '/robot/point_at', 10)
        self.logger.info(f"  {self.name} [setup]")
        self._bb = get_blackboard()
        self._bb.register_key(key='current_exhibit', access=Access.READ)

    def initialise(self):
        target = getattr(self._bb, "current_exhibit", "exhibit")
        self.pub.publish(String(data=f"pointing_at_{target}"))
        self.start_t = ros_time_sec(self.node)
        self.node.get_logger().info(f"Pointing at: {target}")

    def update(self):
        if self.start_t is None:
            return Status.RUNNING
        if ros_time_sec(self.node) - self.start_t >= self.duration:
            return Status.SUCCESS
        return Status.RUNNING


class SayText(py_trees.behaviour.Behaviour):
    """Text-to-speech action (by id)"""
    def __init__(self, name="SayText", node: Node=None, text_id="", wait_for_completion=True, duration=3.0):
        super().__init__(name)
        self.node = node
        self.text_id = text_id
        self.wait = wait_for_completion
        self.duration = duration
        self.pub = None
        self.start_t = None

    def setup(self, **kwargs):
        self.pub = self.node.create_publisher(String, '/robot/text_to_speech', 10)
        self.logger.info(f"  {self.name} [setup]")

    def initialise(self):
        self.pub.publish(String(data=self.text_id))
        self.start_t = ros_time_sec(self.node)
        self.node.get_logger().info(f"Speaking: {self.text_id}")

    def update(self):
        if not self.wait:
            return Status.SUCCESS
        if self.start_t is None:
            return Status.RUNNING
        if ros_time_sec(self.node) - self.start_t >= self.duration:
            return Status.SUCCESS
        return Status.RUNNING


class Navigate(py_trees.behaviour.Behaviour):
    """Navigation action using Nav2"""
    def __init__(self, name="Navigate", node: Node=None, timeout=60.0):
        super().__init__(name)
        self.node = node
        self.timeout = timeout
        self.client = None
        self.goal_handle = None
        self.result_future = None
        self.start_t = None
        self._bb = None
        self._init_error = False

    def setup(self, **kwargs):
        self.client = ActionClient(self.node, NavigateToPose, 'navigate_to_pose')
        self.logger.info(f"  {self.name} [setup]")
        self._bb = get_blackboard()
        self._bb.register_key(key='exhibit_pose', access=Access.READ)

    def initialise(self):
        self._init_error = False
        self.start_t = ros_time_sec(self.node)
        target_pose = getattr(self._bb, "exhibit_pose", None)
        if target_pose is None:
            self.node.get_logger().error("No target pose in blackboard")
            self._init_error = True
            return

        if not self.client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error("Navigation action server not available")
            self._init_error = True
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose

        self.node.get_logger().info("Sending navigation goal")
        send_fut = self.client.send_goal_async(goal_msg)
        send_fut.add_done_callback(self._goal_response_cb)
        self.goal_handle = None
        self.result_future = None

    def _goal_response_cb(self, future):
        gh = future.result()
        if gh is None or not gh.accepted:
            self.node.get_logger().error("Navigation goal rejected")
            self.goal_handle = None
            self.result_future = None
            return
        self.node.get_logger().info("Navigation goal accepted")
        self.goal_handle = gh
        self.result_future = gh.get_result_async()

    def update(self):
        if self._init_error:
            return Status.FAILURE

        if self.start_t is None:
            return Status.FAILURE

        if ros_time_sec(self.node) - self.start_t > self.timeout:
            self.node.get_logger().error("Navigation timeout")
            if self.goal_handle:
                try:
                    self.goal_handle.cancel_goal_async()
                except Exception:
                    pass
            return Status.FAILURE

        if self.goal_handle is None:
            return Status.RUNNING

        if self.result_future is None or not self.result_future.done():
            return Status.RUNNING

        result_resp = self.result_future.result()  # GetResult.Response
        if result_resp.status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info("Navigation completed (SUCCEEDED)")
            return Status.SUCCESS

        self.node.get_logger().warn(f"Navigation ended with status={result_resp.status}")
        return Status.FAILURE

    def terminate(self, new_status):
        if new_status == Status.INVALID and self.goal_handle:
            self.node.get_logger().info("Navigation cancelled")
            try:
                self.goal_handle.cancel_goal_async()
            except Exception:
                pass


class RetrieveListOfExhibits(py_trees.behaviour.Behaviour):
    """Get available exhibits."""
    def __init__(self, name="RetrieveListOfExhibits", node: Node=None):
        super().__init__(name)
        self.node = node
        self._bb = None

    def setup(self, **kwargs):
        self._bb = get_blackboard()
        self._bb.register_key(key='remaining_exhibits', access=Access.WRITE)

    def update(self):
        exhibits = ["dinosaur_hall", "space_exhibit", "ancient_egypt", "modern_art"]
        self._bb.remaining_exhibits = exhibits.copy()
        self.node.get_logger().info(f"Retrieved {len(exhibits)} exhibits")
        return Status.SUCCESS


class SelectExhibit(py_trees.behaviour.Behaviour):
    """Select next exhibit from the list and set a pose."""
    def __init__(self, name="SelectExhibit", node: Node=None, selection_strategy="sequential"):
        super().__init__(name)
        self.node = node
        self.selection_strategy = selection_strategy
        self._bb = None

    def setup(self, **kwargs):
        self._bb = get_blackboard()
        self._bb.register_key(key='remaining_exhibits', access=Access.WRITE)
        self._bb.register_key(key='current_exhibit', access=Access.WRITE)
        self._bb.register_key(key='exhibit_pose', access=Access.WRITE)

    def update(self):
        remaining = getattr(self._bb, "remaining_exhibits", [])
        if not remaining:
            self.node.get_logger().info("No more exhibits to visit")
            return Status.FAILURE

        # select first (sequential)
        selected = remaining.pop(0)
        self._bb.remaining_exhibits = remaining
        self._bb.current_exhibit = selected

        # mock map poses
        exhibit_poses = {
            "dinosaur_hall": (5.0, 3.0),
            "space_exhibit": (8.0, 5.0),
            "ancient_egypt": (10.0, 2.0),
            "modern_art": (6.0, 8.0)
        }
        x, y = exhibit_poses.get(selected, (5.0, 3.0))

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0

        self._bb.exhibit_pose = pose
        self.node.get_logger().info(f"Selected exhibit: {selected}")
        return Status.SUCCESS


class DescribeExhibitSpeech(py_trees.behaviour.Behaviour):
    """Describe the current exhibit."""
    def __init__(self, name="DescribeExhibitSpeech", node: Node=None, speech_part="full", duration=4.0):
        super().__init__(name)
        self.node = node
        self.speech_part = speech_part
        self.duration = duration
        self.pub = None
        self.start_t = None
        self._bb = None

    def setup(self, **kwargs):
        self.pub = self.node.create_publisher(String, '/robot/exhibit_description', 10)
        self.logger.info(f"  {self.name} [setup]")
        self._bb = get_blackboard()
        self._bb.register_key(key='current_exhibit', access=Access.READ)

    def initialise(self):
        exhibit_id = getattr(self._bb, "current_exhibit", "unknown")
        speech_text = f"Describing {exhibit_id} - {self.speech_part}"
        self.pub.publish(String(data=speech_text))
        self.start_t = ros_time_sec(self.node)
        self.node.get_logger().info(speech_text)

    def update(self):
        if self.start_t is None:
            return Status.RUNNING
        if ros_time_sec(self.node) - self.start_t >= self.duration:
            return Status.SUCCESS
        return Status.RUNNING


class ResetRobotPose(py_trees.behaviour.Behaviour):
    """Reset robot to home position (stub)."""
    def __init__(self, name="ResetRobotPose", node: Node=None):
        super().__init__(name)
        self.node = node

    def update(self):
        self.node.get_logger().info("Resetting robot pose")
        # TODO: call a service / publish to reset pose in your stack
        return Status.SUCCESS


class GetVisitorResponse(py_trees.behaviour.Behaviour):
    """Get yes/no response from visitor via ASR."""
    def __init__(self, name="GetVisitorResponse", node: Node=None, timeout=10.0):
        super().__init__(name)
        self.node = node
        self.timeout = timeout
        self.sub = None
        self.latest = None
        self.start_t = None
        self._bb = None

    def setup(self, **kwargs):
        self.sub = self.node.create_subscription(String, '/visitor/speech_response', self._cb, 10)
        self.logger.info(f"  {self.name} [setup]")
        self._bb = get_blackboard()
        self._bb.register_key(key='visitor_response', access=Access.WRITE)

    def _cb(self, msg: String):
        self.latest = msg.data

    def initialise(self):
        self.latest = None
        self.start_t = ros_time_sec(self.node)
        self.node.get_logger().info("Waiting for visitor response...")

    def update(self):
        if ros_time_sec(self.node) - self.start_t > self.timeout:
            self.node.get_logger().warn("Timeout waiting for visitor response")
            return Status.FAILURE
        if self.latest is not None:
            self._bb.visitor_response = self.latest
            self.node.get_logger().info(f"Got response: {self.latest}")
            return Status.SUCCESS
        return Status.RUNNING


class PressYesNoDialogue(py_trees.behaviour.Behaviour):
    """Wait for yes/no via button press."""
    def __init__(self, name="PressYesNoDialogue", node: Node=None, timeout=15.0):
        super().__init__(name)
        self.node = node
        self.timeout = timeout
        self.sub = None
        self.button = None
        self.start_t = None
        self._bb = None

    def setup(self, **kwargs):
        self.sub = self.node.create_subscription(String, '/visitor/button_press', self._cb, 10)
        self.logger.info(f"  {self.name} [setup]")
        self._bb = get_blackboard()
        self._bb.register_key(key='visitor_response', access=Access.WRITE)

    def _cb(self, msg: String):
        self.button = msg.data

    def initialise(self):
        self.button = None
        self.start_t = ros_time_sec(self.node)
        self.node.get_logger().info("Waiting for button press (yes/no)...")

    def update(self):
        if ros_time_sec(self.node) - self.start_t > self.timeout:
            self.node.get_logger().warn("Timeout waiting for button press")
            return Status.FAILURE
        if self.button is not None:
            self._bb.visitor_response = self.button
            self.node.get_logger().info(f"Button pressed: {self.button}")
            return Status.SUCCESS
        return Status.RUNNING


class SetSpeechEvent(py_trees.behaviour.Behaviour):
    """Enable/disable speech recognition."""
    def __init__(self, name="SetSpeechEvent", node: Node=None, event_enabled=True):
        super().__init__(name)
        self.node = node
        self.event_enabled = event_enabled
        self.pub = None
        self._sent_once = False

    def setup(self, **kwargs):
        self.pub = self.node.create_publisher(Bool, '/asr/enable', 10)
        self.logger.info(f"  {self.name} [setup]")

    def initialise(self):
        self._sent_once = False

    def update(self):
        if not self._sent_once:
            self.pub.publish(Bool(data=self.event_enabled))
            self._sent_once = True
            self.node.get_logger().info(f"Speech recognition: {'enabled' if self.event_enabled else 'disabled'}")
        return Status.SUCCESS


# =============================================================================
# Condition Behaviours
# =============================================================================

class IsVisitorDiscovered(py_trees.behaviour.Behaviour):
    """Check if a visitor has been detected."""
    def __init__(self, name="IsVisitorDiscovered", node: Node=None, timeout=30.0):
        super().__init__(name)
        self.node = node
        self.timeout = timeout
        self.sub = None
        self.detected = False
        self.start_t = None

    def setup(self, **kwargs):
        self.sub = self.node.create_subscription(Bool, '/perception/visitor_detected', self._cb, 10)
        self.logger.info(f"  {self.name} [setup]")

    def _cb(self, msg: Bool):
        self.detected = msg.data

    def initialise(self):
        self.detected = False
        self.start_t = ros_time_sec(self.node)
        self.node.get_logger().info("Checking for visitor...")

    def update(self):
        if ros_time_sec(self.node) - self.start_t > self.timeout:
            self.node.get_logger().warn("Timeout: No visitor detected")
            return Status.FAILURE
        if self.detected:
            self.node.get_logger().info("Visitor discovered!")
            return Status.SUCCESS
        return Status.RUNNING


class IsMutualGazeDiscovered(py_trees.behaviour.Behaviour):
    """Check if mutual gaze is established."""
    def __init__(self, name="IsMutualGazeDiscovered", node: Node=None, timeout=10.0):
        super().__init__(name)
        self.node = node
        self.timeout = timeout
        self.sub = None
        self.mutual = False
        self.start_t = None

    def setup(self, **kwargs):
        self.sub = self.node.create_subscription(Bool, '/perception/mutual_gaze', self._cb, 10)
        self.logger.info(f"  {self.name} [setup]")

    def _cb(self, msg: Bool):
        self.mutual = msg.data

    def initialise(self):
        self.mutual = False
        self.start_t = ros_time_sec(self.node)
        self.node.get_logger().info("Seeking mutual gaze...")

    def update(self):
        if ros_time_sec(self.node) - self.start_t > self.timeout:
            self.node.get_logger().warn("Timeout: Mutual gaze not established")
            return Status.FAILURE
        if self.mutual:
            self.node.get_logger().info("Mutual gaze established!")
            return Status.SUCCESS
        return Status.RUNNING


class IsASREnabled(py_trees.behaviour.Behaviour):
    """Check if Automatic Speech Recognition is enabled (by param)."""
    def __init__(self, name="IsASREnabled", node: Node=None):
        super().__init__(name)
        self.node = node

    def update(self):
        try:
            if self.node.get_parameter('asr_enabled').value:
                self.node.get_logger().debug("ASR is enabled")
                return Status.SUCCESS
        except Exception:
            pass
        self.node.get_logger().debug("ASR is disabled")
        return Status.FAILURE


class IsListWithExhibit(py_trees.behaviour.Behaviour):
    """Check if there are remaining exhibits."""
    def __init__(self, name="IsListWithExhibit", node: Node=None):
        super().__init__(name)
        self.node = node
        self._bb = None

    def setup(self, **kwargs):
        self._bb = get_blackboard()
        self._bb.register_key(key='remaining_exhibits', access=Access.READ)

    def update(self):
        remaining = getattr(self._bb, "remaining_exhibits", [])
        if not remaining:
            self.node.get_logger().info("No more exhibits in list")
            return Status.FAILURE
        self.node.get_logger().debug(f"{len(remaining)} exhibits remaining")
        return Status.SUCCESS


class IsVisitorResponseYes(py_trees.behaviour.Behaviour):
    """Check if visitor responded with 'yes'."""
    def __init__(self, name="IsVisitorResponseYes", node: Node=None):
        super().__init__(name)
        self.node = node
        self._bb = None

    def setup(self, **kwargs):
        self._bb = get_blackboard()
        self._bb.register_key(key='visitor_response', access=Access.READ)

    def update(self):
        response = getattr(self._bb, "visitor_response", "")
        response = response.lower().strip()
        affirmative = {'yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay'}
        if any(word == response or word in response for word in affirmative):
            self.node.get_logger().info("Visitor responded: YES")
            return Status.SUCCESS
        self.node.get_logger().info("Visitor responded: NO")
        return Status.FAILURE
