#!/usr/bin/env python3
"""
Bag-replay test for face_detection on the CSSR4Africa RealSense test bags.

Launches the real node (Gold-YOLO + SixDRepNet ONNX, RGB-only, without
person_detection) and plays each CSSR4Africa face-detection bag to it at the
recorded frame rate (about 15 fps), every frame in order, as the camera would
on the robot. The node publishes one /face_detection/data message per frame it
processes (empty when it sees no face); all of them are collected. Per bag:

  every bag       the node answers at least 80% of the frames, i.e. it keeps
                  up with the camera (needs the GPU build of ONNX Runtime,
                  see dec_common's dec_find_onnxruntime)
  single_face     a face in at least 40% of the answers (the person is often
                  too far away), never more than two
  multiple_faces  two or more faces in at least 10% of the answers
  mutual_gaze     a face in most answers, mutual gaze in at least 10% of them
  lighting_1/2    a face in at least half of the answers

The bags are the ROS1 recordings from
https://huggingface.co/cssr4africa/cssr4africa_unit_tests_data_files,
converted to ROS2 by scripts/fetch_test_bags.sh into
~/ros2_ws/bags/cssr4africa (override with PEPPER4DEC_TEST_BAGS). The test
skips when the bags or the git-ignored models are missing.

Run via: colcon test --packages-select face_detection

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: November 13, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import os
import signal
import time
import unittest

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

from dec_interfaces.msg import FaceDetection

NODE_NAME = 'face_detection'
# Unique per test: colcon runs packages' tests in parallel on one domain.
CAMERA_TOPIC = '/test_camera/face_detection/image_raw'
BAG_TOPIC = '/camera/color/image_raw/compressed'
BAG_ROOT = os.path.join(
    os.environ.get('PEPPER4DEC_TEST_BAGS',
                   os.path.expanduser('~/ros2_ws/bags/cssr4africa')),
    'face_detection')
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = [os.path.join(_TEST_DIR, '..', 'models', m) for m in
          ('face_detection_goldYOLO.onnx', 'face_detection_sixdrepnet360.onnx')]
BAGS = ('single_face', 'multiple_faces', 'mutual_gaze', 'lighting_1',
        'lighting_2')

MIN_ANSWERED = 0.8   # share of frames the node must answer to keep up
TAIL = 1.0           # seconds to collect answers after the last frame
CONFIGURE_TIMEOUT = 60.0


@pytest.mark.launch_test
def generate_test_description():
    node = launch_ros.actions.Node(
        package='face_detection',
        executable='face_detection',
        name=NODE_NAME,
        output='screen',
        parameters=[{'camera': 'pepper', 'require_person_detection': False,
                     'verbose_mode': False}],
        # 'pepper' mode subscribes RGB-only to the PepperFrontCamera topic.
        remappings=[('/pepper/front/image_raw', CAMERA_TOPIC)],
    )
    return launch.LaunchDescription([
        node, launch_testing.actions.ReadyToTest()]), {'node': node}


def read_bag(bag_dir):
    """Return [(bag time in s, serialized CompressedImage)] for the colour topic."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[BAG_TOPIC]))
    frames = []
    while reader.has_next():
        _, data, stamp = reader.read_next()
        frames.append((stamp * 1e-9, data))
    return frames


def to_image(data):
    """Decode a serialized CompressedImage into a bgr8 Image."""
    compressed = deserialize_message(data, CompressedImage)
    bgr = cv2.imdecode(np.frombuffer(compressed.data, np.uint8),
                       cv2.IMREAD_COLOR)
    msg = Image()
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.step = bgr.shape[1] * 3
    msg.data = bgr.tobytes()
    return msg


class TestCssr4AfricaFaceBags(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.probe = rclpy.create_node('face_bag_probe')

    @classmethod
    def tearDownClass(cls):
        cls.probe.destroy_node()
        rclpy.shutdown()

    def _call(self, srv_type, name, request, timeout):
        client = self.probe.create_client(srv_type, name)
        try:
            self.assertTrue(client.wait_for_service(timeout_sec=timeout),
                            f'service {name} not available')
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.probe, future,
                                             timeout_sec=timeout)
            self.assertIsNotNone(future.result(), f'{name} call timed out')
            return future.result()
        finally:
            self.probe.destroy_client(client)

    def _transition(self, transition_id, timeout):
        request = ChangeState.Request()
        request.transition = Transition(id=transition_id)
        self.assertTrue(self._call(ChangeState, f'/{NODE_NAME}/change_state',
                                   request, timeout).success)

    def _play(self, pub, replies, frames):
        """Publish frames at their recorded rate; return the answers."""
        del replies[:]
        start_wall = time.monotonic()
        start_bag = frames[0][0]
        for stamp, data in frames:
            msg = to_image(data)
            due = start_wall + (stamp - start_bag)
            while time.monotonic() < due:
                rclpy.spin_once(self.probe,
                                timeout_sec=max(0.0, due - time.monotonic()))
            msg.header.stamp = self.probe.get_clock().now().to_msg()
            pub.publish(msg)
            rclpy.spin_once(self.probe, timeout_sec=0.0)
        end = time.monotonic() + TAIL
        while time.monotonic() < end:
            rclpy.spin_once(self.probe, timeout_sec=0.05)
        return list(replies)

    def test_cssr4africa_bags(self):
        missing = [m for m in MODELS if not os.path.isfile(m)]
        if missing:
            self.skipTest(f'models not present: {missing}')
        bags = {b: os.path.join(BAG_ROOT, b) for b in BAGS
                if os.path.isfile(os.path.join(BAG_ROOT, b, 'metadata.yaml'))}
        if not bags:
            self.skipTest(f'no bags under {BAG_ROOT}: run '
                          'scripts/fetch_test_bags.sh')

        self._transition(Transition.TRANSITION_CONFIGURE, CONFIGURE_TIMEOUT)
        self._transition(Transition.TRANSITION_ACTIVATE, 10.0)
        state = self._call(GetState, f'/{NODE_NAME}/get_state',
                           GetState.Request(), 10.0)
        self.assertEqual(state.current_state.id, State.PRIMARY_STATE_ACTIVE)

        replies = []
        self.probe.create_subscription(FaceDetection, '/face_detection/data',
                                       replies.append, 100)
        pub = self.probe.create_publisher(Image, CAMERA_TOPIC, 10)
        deadline = time.time() + 10.0
        while pub.get_subscription_count() == 0 and time.time() < deadline:
            rclpy.spin_once(self.probe, timeout_sec=0.1)
        self.assertGreater(pub.get_subscription_count(), 0,
                           'node never subscribed to the test camera topic')

        for bag, path in bags.items():
            with self.subTest(bag=bag):
                frames = read_bag(path)
                answers = self._play(pub, replies, frames)
                for a in answers:
                    n = len(a.face_label_id)
                    self.assertEqual(len(a.centroids), n)
                    self.assertEqual(len(a.mutual_gaze), n)
                n = len(answers)
                counts = [len(a.face_label_id) for a in answers]
                with_face = sum(c > 0 for c in counts) / max(n, 1)
                two_plus = sum(c >= 2 for c in counts) / max(n, 1)
                gaze = sum(any(a.mutual_gaze) for a in answers) / max(n, 1)
                print(f'[{bag}] {n}/{len(frames)} frames answered; face in '
                      f'{with_face:.0%}, 2+ faces in {two_plus:.0%}, mutual '
                      f'gaze in {gaze:.0%}, most faces {max(counts, default=0)}',
                      flush=True)

                # More answers than frames means another publisher fed the node.
                self.assertLessEqual(n, len(frames), 'answers exceed frames')
                self.assertGreaterEqual(
                    n, MIN_ANSWERED * len(frames),
                    f'node answered {n} of {len(frames)} frames: it cannot '
                    'keep up with the camera (is ONNX Runtime on the GPU?)')
                if bag == 'single_face':
                    # The person stands far back for much of the recording,
                    # with the face below the detector's 20 px minimum.
                    self.assertGreaterEqual(with_face, 0.4)
                    self.assertLessEqual(max(counts), 2)
                elif bag == 'multiple_faces':
                    self.assertGreaterEqual(two_plus, 0.1)
                elif bag == 'mutual_gaze':
                    self.assertGreaterEqual(with_face, 0.7)
                    self.assertGreaterEqual(gaze, 0.1)
                else:
                    self.assertGreaterEqual(with_face, 0.5)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_code(self, proc_info, node):
        # A skipped run stops the node before rclcpp installs its SIGINT
        # handler, so the default disposition (-SIGINT) is a clean exit too.
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, -signal.SIGINT], process=node)
