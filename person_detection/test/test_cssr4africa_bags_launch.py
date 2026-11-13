#!/usr/bin/env python3
"""
Bag-replay test for person_detection on the CSSR4Africa RealSense test bags.

Launches the real node (YOLO11 ONNX + ByteTrack, RGB-only) and plays each
CSSR4Africa person-detection bag to it at the recorded frame rate (about
15 fps), every frame in order, as the camera would on the robot. The node
publishes a /person_detection/data message for each processed frame in which
it tracks someone (nothing when it sees no one), so the score is that count
over the frames played. Per bag:

  single_person    a person in at least 90% of the frames, never more than two
  multiple_people  a person in at least 90%, two or more in at least 30%
  lighting_1/2     a person in at least 85% of the frames

The model sees a person in nearly every frame of these bags, so these
thresholds also catch a node that drops frames (it once did: double-sigmoid
scores made each frame take about a second, and spinning ONNX Runtime threads
starved the camera feed).

A machine that cannot run the model at camera rate (ONNX Runtime on the CPU)
answers too few frames and fails these; see dec_common's dec_find_onnxruntime
for the GPU build. Every message is also checked for consistent array sizes.

The bags are the ROS1 recordings from
https://huggingface.co/cssr4africa/cssr4africa_unit_tests_data_files,
converted to ROS2 by scripts/fetch_test_bags.sh into
~/ros2_ws/bags/cssr4africa (override with PEPPER4DEC_TEST_BAGS). The test
skips when the bags or the git-ignored models are missing.

Run via: colcon test --packages-select person_detection

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

from dec_interfaces.msg import PersonDetection

NODE_NAME = 'person_detection'
# Unique per test: colcon runs packages' tests in parallel on one domain.
CAMERA_TOPIC = '/test_camera/person_detection_cssr4africa/image_raw'
BAG_TOPIC = '/camera/color/image_raw/compressed'
BAG_ROOT = os.path.join(
    os.environ.get('PEPPER4DEC_TEST_BAGS',
                   os.path.expanduser('~/ros2_ws/bags/cssr4africa')),
    'person_detection')
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = [os.path.join(_TEST_DIR, '..', 'models',
                       'person_detection_yolov11m.onnx')]
BAGS = ('single_person', 'multiple_people', 'lighting_1', 'lighting_2')

TAIL = 1.0           # seconds to collect answers after the last frame
CONFIGURE_TIMEOUT = 60.0


@pytest.mark.launch_test
def generate_test_description():
    node = launch_ros.actions.Node(
        package='person_detection',
        executable='person_detection',
        name=NODE_NAME,
        output='screen',
        parameters=[{'camera': 'pepper', 'verbose_mode': False}],
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


class TestCssr4AfricaPersonBags(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.probe = rclpy.create_node('person_bag_probe')

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
        self.probe.create_subscription(
            PersonDetection, '/person_detection/data', replies.append, 100)
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
                    n = len(a.person_label_id)
                    self.assertGreater(n, 0)
                    for field in (a.class_ids, a.confidences, a.centroids,
                                  a.width, a.height):
                        self.assertEqual(len(field), n)
                counts = [len(a.person_label_id) for a in answers]
                frames_n = len(frames)
                with_person = len(answers) / frames_n
                two_plus = sum(c >= 2 for c in counts) / frames_n
                print(f'[{bag}] {len(answers)} messages over {frames_n} '
                      f'frames; person in {with_person:.0%}, 2+ people in '
                      f'{two_plus:.0%}, most people {max(counts, default=0)}',
                      flush=True)

                # More answers than frames means another publisher fed the node.
                self.assertLessEqual(len(answers), frames_n,
                                     'answers exceed frames')
                hint = ' (too few answers: is ONNX Runtime on the GPU?)'
                if bag == 'single_person':
                    self.assertGreaterEqual(with_person, 0.9, hint)
                    self.assertLessEqual(max(counts), 2)
                elif bag == 'multiple_people':
                    self.assertGreaterEqual(with_person, 0.9, hint)
                    self.assertGreaterEqual(two_plus, 0.3, hint)
                else:
                    self.assertGreaterEqual(with_person, 0.85, hint)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_code(self, proc_info, node):
        # A skipped run stops the node before rclcpp installs its SIGINT
        # handler, so the default disposition (-SIGINT) is a clean exit too.
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, -signal.SIGINT], process=node)
