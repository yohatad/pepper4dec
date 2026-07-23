#!/usr/bin/env python3
"""
Bag-replay regression test for the person_detection pipeline.

Launches the real node (YOLOv11 ONNX + ByteTrack), feeds it recorded camera frames from
the checked-in mini-bag (test/data/person_walk_minibag, 12 frames sampled
from a lab recording with people in view), and asserts on the published
/person_detection/data stream.

Frames are fed one at a time, each waiting for the node to finish inference
before the next is sent. This keeps the test deterministic on slow/CPU-only
machines — real-time playback would silently drop frames whenever inference
is slower than the camera rate, making pass/fail depend on machine speed.

What this catches that unit tests cannot: ONNX model loading/inference glue,
image decoding, the full subscribe -> detect -> track -> publish path, and
message field consistency — against real sensor data, with no robot.

Run via: colcon test --packages-select person_detection

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: Jul 18, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import os
import signal
import time
import unittest
from collections import Counter

import cv2
import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import numpy as np
import pytest
import rclpy
import rosbag2_py
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image

from dec_interfaces.msg import PersonDetection

NODE_NAME = 'personDetection'
CAMERA_TOPIC = '/test_camera/image_raw'
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
BAG_PATH = os.path.join(_TEST_DIR, 'data', 'person_walk_minibag')
# The YOLOv11 model (77MB) is git-ignored, so CI checkouts don't have it —
# skip there rather than fail in on_configure.
MODEL_PATH = os.path.join(_TEST_DIR, '..', 'models',
                          'person_detection_yolov11m.onnx')

# Generous per-frame inference budget: YOLOv11m on a weak CPU needs ~4s.
FRAME_TIMEOUT = 15.0
CONFIGURE_TIMEOUT = 60.0


@pytest.mark.launch_test
def generate_test_description():
    detection_node = launch_ros.actions.Node(
        package='person_detection',
        executable='person_detection',
        name=NODE_NAME,
        output='screen',
        parameters=[{'camera': 'pepper', 'verbose_mode': False}],
        # 'pepper' camera mode subscribes RGB-only to the PepperFrontCamera
        # topic from data/pepper_topics.yaml; remap it to the test topic.
        remappings=[('/pepper/front/image_raw', CAMERA_TOPIC)],
    )
    return launch.LaunchDescription([
        detection_node,
        launch_testing.actions.ReadyToTest(),
    ]), {'detection_node': detection_node}


def load_bag_frames():
    """Read the mini-bag and return decoded bgr8 Image messages."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('', ''))
    frames = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        compressed = deserialize_message(data, CompressedImage)
        bgr = cv2.imdecode(np.frombuffer(compressed.data, np.uint8),
                           cv2.IMREAD_COLOR)
        msg = Image()
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding = 'bgr8'
        msg.step = bgr.shape[1] * 3
        msg.data = bgr.tobytes()
        frames.append(msg)
    return frames


class TestBagReplay(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.probe = rclpy.create_node('bag_replay_probe')

    @classmethod
    def tearDownClass(cls):
        cls.probe.destroy_node()
        rclpy.shutdown()

    def _call(self, srv_type, srv_name, request, timeout):
        client = self.probe.create_client(srv_type, srv_name)
        try:
            self.assertTrue(client.wait_for_service(timeout_sec=timeout),
                            f'service {srv_name} not available')
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.probe, future, timeout_sec=timeout)
            self.assertIsNotNone(future.result(), f'{srv_name} call timed out')
            return future.result()
        finally:
            self.probe.destroy_client(client)

    def _transition(self, transition_id, timeout):
        request = ChangeState.Request()
        request.transition = Transition(id=transition_id)
        result = self._call(ChangeState, f'/{NODE_NAME}/change_state',
                            request, timeout)
        self.assertTrue(result.success)

    def test_replay_produces_stable_person_tracks(self):
        if not os.path.isfile(MODEL_PATH):
            self.skipTest('ONNX model not present (git-ignored, 77MB) — '
                          'run on a machine with models/ populated')
        # ONNX model load happens in on_configure — allow generous time.
        self._transition(Transition.TRANSITION_CONFIGURE, CONFIGURE_TIMEOUT)
        self._transition(Transition.TRANSITION_ACTIVATE, 10.0)
        state = self._call(GetState, f'/{NODE_NAME}/get_state',
                           GetState.Request(), 10.0)
        self.assertEqual(state.current_state.id, State.PRIMARY_STATE_ACTIVE)

        detections = []
        self.probe.create_subscription(
            PersonDetection, '/person_detection/data', detections.append, 10)
        pub = self.probe.create_publisher(Image, CAMERA_TOPIC, 10)

        # Wait for pub/sub matching before the first frame.
        deadline = time.time() + 10.0
        while pub.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(self.probe, timeout_sec=0.1)
        self.assertGreater(pub.get_subscription_count(), 0,
                           'node never subscribed to the test camera topic')

        frames = load_bag_frames()
        self.assertGreaterEqual(len(frames), 10, 'mini-bag missing frames')

        answered = 0
        for frame in frames:
            frame.header.stamp = self.probe.get_clock().now().to_msg()
            seen_before = len(detections)
            # The node subscribes best-effort (SensorDataQoS); large raw
            # images can be dropped wholesale on lossy loopback transports
            # (e.g. WSL2). Republish the same frame every 2s until the node
            # answers — duplicates are harmless to detector and tracker.
            pub.publish(frame)
            deadline = time.time() + FRAME_TIMEOUT
            next_republish = time.time() + 2.0
            while len(detections) == seen_before and time.time() < deadline:
                rclpy.spin_once(self.probe, timeout_sec=0.1)
                if time.time() >= next_republish:
                    pub.publish(frame)
                    next_republish = time.time() + 2.0
            if len(detections) > seen_before:
                answered += 1

        # ── Assertions ──────────────────────────────────────────────────────
        # 1. Most frames must yield a detection message (all frames in the
        #    mini-bag contain at least one person or person-like robot).
        self.assertGreaterEqual(
            answered, int(0.6 * len(frames)),
            f'only {answered}/{len(frames)} frames produced detections')

        # 2. Every message is internally consistent and non-empty.
        for msg in detections:
            n = len(msg.person_label_id)
            self.assertGreater(n, 0)
            self.assertEqual(len(msg.confidences), n)
            self.assertEqual(len(msg.centroids), n)
            self.assertEqual(len(msg.width), n)
            self.assertEqual(len(msg.height), n)
            for conf in msg.confidences:
                self.assertGreaterEqual(conf, 0.0)
                self.assertLessEqual(conf, 1.0)

        # 3. Tracking is stable: at least one track ID persists through most
        #    of the sequence (the scene contains stationary person-like
        #    subjects, so ByteTrack must hold their IDs across frames).
        id_counts = Counter(tid for msg in detections
                            for tid in msg.person_label_id)
        most_common_id, seen = id_counts.most_common(1)[0]
        self.assertGreaterEqual(
            seen, int(0.6 * len(detections)),
            f'no stable track: best ID {most_common_id!r} only in '
            f'{seen}/{len(detections)} messages')


@launch_testing.post_shutdown_test()
class TestProcessOutcome(unittest.TestCase):

    def test_exit_code(self, proc_info, detection_node):
        # In CI (no model -> immediate skip) the harness sends SIGINT within
        # ~0.1s of process start, often before rclcpp's own signal handler is
        # installed, so the process dies to the *default* SIGINT disposition
        # (-signal.SIGINT) instead of exiting 0 via a caught, graceful
        # rclcpp::shutdown(). Both are the harness's normal shutdown path, not
        # a crash.
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, -signal.SIGINT], process=detection_node)
