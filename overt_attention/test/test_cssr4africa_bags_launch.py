#!/usr/bin/env python3
"""
Bag-replay test for overt_attention on the CSSR4Africa RealSense test bags.

Launches the attention pipeline as on the robot, minus the robot: the real
face_detection node (RealSense mode: compressed colour plus aligned
compressedDepth, without person_detection), the saliency node (colour with
depth weighting) and the attention controller. Each of the five distinct
CSSR4Africa recordings is played at its recorded 15 fps: colour, depth and
colour camera_info, re-stamped to the present. Pepper's head is not in the
recordings, so the test publishes a fixed head pose (yaw 0, pitch 0) on the
joint states; the controller's targets are then camera angles. Per bag:

  depth       faces come with a plausible depth (face_detection reads the
              aligned depth stream)
  saliency    peaks at the configured rate, inside the image, scores within
              [min_peak, 1], strongest first, at most num_peaks
  limits      every head command lies within Pepper's head joint limits
  faces       while faces are visible the target points at one of them
  engagement  with two faces of which one is in mutual gaze, the target is
              usually that one (multiple_faces)
  timeout     after faces disappear nothing is commanded until face_timeout
              has passed; saliency then takes over, within its tighter
              limits (the recordings seldom lose faces that long, so there
              are too few saliency targets to test inhibition of return)

and once, on mutual_gaze: set_enabled false sends the head to its default
pose and stops all commands; set_enabled true resumes them.

The recordings carry no labels of where the robot should look, so targets are
checked against face_detection's output and the controller's own rules.

Bags: https://huggingface.co/cssr4africa/dec_unit_test_data_files, fetched by
scripts/fetch_test_bags.sh into ~/ros2_ws/bags/cssr4africa (override with
PEPPER4DEC_TEST_BAGS). The test skips when the bags or face_detection's
git-ignored models are missing. At camera rate it needs the GPU.

Run via: colcon test --packages-select overt_attention

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: November 14, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import math
import os
import signal
import time
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import pytest
import rclpy
import rosbag2_py
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Vector3
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool

from dec_interfaces.msg import FaceDetection

# Every topic and service is unique to this test: colcon runs the packages'
# tests in parallel on one domain, and face_detection's own bag test listens
# on /face_detection/data.
P = '/test_overt'
COLOR = P + '/color/compressed'
DEPTH = P + '/depth/compressedDepth'
INFO = P + '/color/camera_info'
FACES = P + '/faces'
PEAKS = P + '/saliency_peak'
JOINTS = P + '/joint_states'
HEAD = P + '/joint_angles'
TARGET = P + '/target_angles'
ENABLE = P + '/set_enabled'
FACE_NODE = 'overt_face_detection'
ATTENTION_NODE = 'overt_attention'

BAG_COLOR = '/camera/color/image_raw/compressed'
BAG_DEPTH = '/camera/aligned_depth_to_color/image_raw/compressedDepth'
BAG_INFO = '/camera/color/camera_info'
BAG_ROOT = os.path.join(
    os.environ.get('PEPPER4DEC_TEST_BAGS',
                   os.path.expanduser('~/ros2_ws/bags/cssr4africa')),
    'face_detection')
BAGS = ('single_face', 'multiple_faces', 'mutual_gaze', 'lighting_1',
        'lighting_2')
_FACE_SHARE = get_package_share_directory('face_detection')
MODELS = [os.path.join(_FACE_SHARE, 'models', m) for m in
          ('face_detection_goldYOLO.onnx', 'face_detection_sixdrepnet360.onnx')]
CONFIG = os.path.join(get_package_share_directory('overt_attention'), 'config',
                      'overt_attention_configuration.yaml')

# From config/overt_attention_configuration.yaml and the controller.
FACE_TIMEOUT = 2.0
SAL_YAW, SAL_PITCH_UP, SAL_PITCH_DN = 1.2, 0.3, -0.3
FACE_YAW, FACE_PITCH_UP, FACE_PITCH_DN = 1.8, 0.4, -0.7
DEFAULT_YAW, DEFAULT_PITCH = 0.0, -0.2
MIN_PEAK, NUM_PEAKS, PROCESS_HZ = 0.5, 5, 1.0
EPS = 1e-3
TAIL = 1.0
GAP = 3.0            # idle seconds between bags, longer than FACE_TIMEOUT
JOINT_PERIOD = 0.1   # head joint state rate


@pytest.mark.launch_test
def generate_test_description():
    camera = [('/camera/color/image_raw_custom/compressed', COLOR),
              ('/camera/aligned_depth_to_color/image_raw_custom/compressedDepth',
               DEPTH)]
    face = launch_ros.actions.Node(
        package='face_detection', executable='face_detection', name=FACE_NODE,
        output='screen',
        parameters=[{'camera': 'realsense', 'use_compressed': True,
                     'require_person_detection': False,
                     'verbose_mode': False}],
        remappings=camera + [('/face_detection/data', FACES)])
    saliency = launch_ros.actions.Node(
        package='overt_attention', executable='overt_attention_saliency',
        name='saliency_node', output='screen',
        parameters=[CONFIG, {'camera_type': 'realsense', 'use_compressed': True,
                             'publish_map': False, 'process_hz': PROCESS_HZ}],
        remappings=[('/camera/color/image_raw_custom/compressed', COLOR),
                    ('/camera/aligned_depth_to_color/image_raw_custom'
                     '/compressedDepth', DEPTH),
                    ('/overt_attention/saliency_peak', PEAKS)])
    attention = launch_ros.actions.Node(
        package='overt_attention', executable='overt_attention',
        name=ATTENTION_NODE, output='screen',
        parameters=[CONFIG, {'camera_type': 'realsense'}],
        remappings=[('/face_detection/data', FACES),
                    ('/overt_attention/saliency_peak', PEAKS),
                    ('/camera/color/camera_info', INFO),
                    ('/joint_states', JOINTS),
                    ('/joint_angles', HEAD),
                    ('/overt_attention/target_angles', TARGET),
                    ('/overt_attention/set_enabled', ENABLE)])
    return launch.LaunchDescription([
        face, saliency, attention, launch_testing.actions.ReadyToTest()]), {
            'face': face, 'saliency': saliency, 'attention': attention}


def read_bag(bag_dir):
    """Return [(topic, header stamp in s, message)] for colour, depth and info."""
    types = {BAG_COLOR: CompressedImage, BAG_DEPTH: CompressedImage,
             BAG_INFO: CameraInfo}
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(types)))
    out = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, types[topic])
        out.append((topic, msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    msg))
    out.sort(key=lambda m: m[1])
    return out


def to_angles(u, v, info):
    """pixelToAngles of overt_attention_interface.h, in radians."""
    fx, cx, fy, cy = info.k[0], info.k[2], info.k[4], info.k[5]
    return -math.atan((u - cx) / fx), math.atan((v - cy) / fy)


def stamp(t):
    sec = math.floor(t)
    return Time(sec=int(sec), nanosec=int((t - sec) * 1e9))


class TestCssr4AfricaAttentionBags(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.probe = rclpy.create_node('overt_attention_bag_probe')

    @classmethod
    def tearDownClass(cls):
        cls.probe.destroy_node()
        rclpy.shutdown()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _call(self, srv_type, name, request, timeout):
        client = self.probe.create_client(srv_type, name)
        try:
            self.assertTrue(client.wait_for_service(timeout_sec=timeout),
                            f'service {name} not available')
            future = client.call_async(request)
            end = time.monotonic() + timeout
            while not future.done() and time.monotonic() < end:
                self._spin(0.05)
            self.assertTrue(future.done(), f'{name} call timed out')
            return future.result()
        finally:
            self.probe.destroy_client(client)

    def _bring_up(self, node, configure_timeout):
        for transition, timeout in ((Transition.TRANSITION_CONFIGURE,
                                     configure_timeout),
                                    (Transition.TRANSITION_ACTIVATE, 10.0)):
            request = ChangeState.Request()
            request.transition = Transition(id=transition)
            self.assertTrue(self._call(ChangeState, f'/{node}/change_state',
                                       request, timeout).success)
        state = self._call(GetState, f'/{node}/get_state', GetState.Request(),
                           10.0)
        self.assertEqual(state.current_state.id, State.PRIMARY_STATE_ACTIVE)

    def _spin(self, timeout):
        """Spin the probe, keeping the head joint states flowing."""
        now = time.monotonic()
        if now >= self.next_joint:
            msg = JointState()
            msg.header.stamp = self.probe.get_clock().now().to_msg()
            msg.name = ['HeadYaw', 'HeadPitch']
            msg.position = [0.0, 0.0]
            self.joint_pub.publish(msg)
            self.next_joint = now + JOINT_PERIOD
        rclpy.spin_once(self.probe, timeout_sec=max(
            0.0, min(timeout, self.next_joint - time.monotonic())))

    def _idle(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self._spin(end - time.monotonic())

    def _play(self, messages, at=None):
        """
        Publish a bag at its recorded rate, re-stamped to the present.

        at: optional {seconds into the bag: callable} run during playback.
        Returns the playback start (monotonic) and its duration.
        """
        pubs = {BAG_COLOR: self.color_pub, BAG_DEPTH: self.depth_pub,
                BAG_INFO: self.info_pub}
        actions = sorted((at or {}).items())
        t0 = messages[0][1]
        start = time.monotonic()
        ros_start = self.probe.get_clock().now().nanoseconds * 1e-9
        for topic, t, msg in messages:
            due = start + (t - t0)
            while actions and actions[0][0] <= t - t0:
                actions.pop(0)[1]()
            while time.monotonic() < due:
                self._spin(due - time.monotonic())
            msg.header.stamp = stamp(ros_start + (t - t0))
            pubs[topic].publish(msg)
        for _, action in actions:
            action()
        return start, messages[-1][1] - t0

    def _record(self):
        for log in self.log.values():
            del log[:]

    # ── test ────────────────────────────────────────────────────────────────

    def test_cssr4africa_bags(self):
        missing = [m for m in MODELS if not os.path.isfile(m)]
        if missing:
            self.skipTest(f'face_detection models not installed: {missing}')
        bags = {b: os.path.join(BAG_ROOT, b) for b in BAGS
                if os.path.isfile(os.path.join(BAG_ROOT, b, 'metadata.yaml'))}
        if not bags:
            self.skipTest(f'no bags under {BAG_ROOT}: run '
                          'scripts/fetch_test_bags.sh')

        p = self.probe
        self.color_pub = p.create_publisher(CompressedImage, COLOR, 10)
        self.depth_pub = p.create_publisher(CompressedImage, DEPTH, 10)
        self.info_pub = p.create_publisher(CameraInfo, INFO, 10)
        self.joint_pub = p.create_publisher(JointState, JOINTS, 10)
        self.next_joint = 0.0
        self.log = {'faces': [], 'peaks': [], 'head': [], 'target': []}
        stamp_now = time.monotonic
        p.create_subscription(
            FaceDetection, FACES,
            lambda m: self.log['faces'].append((stamp_now(), m)), 100)
        p.create_subscription(
            Float32MultiArray, PEAKS,
            lambda m: self.log['peaks'].append((stamp_now(), m)), 100)
        p.create_subscription(
            JointAnglesWithSpeed, HEAD,
            lambda m: self.log['head'].append((stamp_now(), m)), 100)
        p.create_subscription(
            Vector3, TARGET,
            lambda m: self.log['target'].append((stamp_now(), m)), 100)

        self._bring_up(FACE_NODE, 60.0)
        self._bring_up(ATTENTION_NODE, 20.0)
        deadline = time.monotonic() + 10.0
        while ((self.color_pub.get_subscription_count() < 2 or
                self.depth_pub.get_subscription_count() < 2) and
               time.monotonic() < deadline):
            self._spin(0.1)
        self.assertGreaterEqual(self.color_pub.get_subscription_count(), 2,
                                'face_detection and saliency must subscribe')
        self._idle(1.0)

        for bag, path in bags.items():
            with self.subTest(bag=bag):
                messages = read_bag(path)
                info = next(m for topic, _, m in messages if topic == BAG_INFO)
                self._record()
                start, duration = self._play(messages)
                self._idle(TAIL)
                self._check_bag(bag, info, start, duration)
                self._idle(GAP)

        if 'mutual_gaze' in bags:
            with self.subTest(check='set_enabled'):
                self._check_enable(read_bag(bags['mutual_gaze']))

    # ── checks ──────────────────────────────────────────────────────────────

    def _check_bag(self, bag, info, start, duration):
        faces = [(t - start, m) for t, m in self.log['faces']]
        peaks = [(t - start, m) for t, m in self.log['peaks']]
        heads = [(t - start, m) for t, m in self.log['head']]
        targets = [(t - start, m) for t, m in self.log['target']]
        seen = [(t, m) for t, m in faces if m.centroids]

        # depth: face_detection read the aligned depth stream.
        depths = [c.z for _, m in seen for c in m.centroids]
        with_depth = sum(0.3 < z < 8.0 for z in depths) / max(len(depths), 1)

        # saliency peaks.
        for _, m in peaks:
            d = list(m.data)
            self.assertEqual(len(d) % 3, 0)
            self.assertLessEqual(len(d) // 3, NUM_PEAKS)
            scores = d[2::3]
            self.assertEqual(scores, sorted(scores, reverse=True))
            for u, v, s in zip(d[0::3], d[1::3], scores):
                self.assertTrue(0 <= u <= info.width and 0 <= v <= info.height,
                                f'peak ({u}, {v}) outside the image')
                self.assertTrue(MIN_PEAK - EPS <= s <= 1 + EPS, f'score {s}')

        # limits.
        for _, m in heads:
            yaw, pitch = m.joint_angles
            self.assertTrue(-FACE_YAW - EPS <= yaw <= FACE_YAW + EPS, yaw)
            self.assertTrue(FACE_PITCH_DN - EPS <= pitch <= FACE_PITCH_UP + EPS,
                            pitch)

        # Classify every target by the faces last seen before it.
        def last_seen(t):
            last = None
            for tf, _ in seen:
                if tf > t:
                    break
                last = tf
            return last

        def latest_faces(t):
            latest = None
            for tf, m in faces:
                if tf > t:
                    break
                latest = m
            return latest

        face_err, engaged_hits, engaged_total = [], 0, 0
        too_early, saliency_targets = [], []
        for t, tgt in targets:
            ls = last_seen(t)
            since = math.inf if ls is None else t - ls
            m = latest_faces(t)
            if m is not None and m.centroids and since < 0.3:
                angles = [to_angles(c.x, c.y, info) for c in m.centroids]
                errs = [math.hypot(tgt.x - a[0], tgt.y - a[1]) for a in angles]
                nearest = min(range(len(errs)), key=errs.__getitem__)
                face_err.append(math.degrees(errs[nearest]))
                if len(m.centroids) >= 2 and sum(m.mutual_gaze) == 1:
                    engaged_total += 1
                    engaged_hits += bool(m.mutual_gaze[nearest])
            elif 0.3 < since < FACE_TIMEOUT - 0.3:
                too_early.append(round(since, 2))
            elif since >= FACE_TIMEOUT - 0.3:
                saliency_targets.append((t, tgt))

        face_err.sort()

        def pct(q):
            return face_err[int(q * (len(face_err) - 1))] if face_err else 0.0

        no_face = sum(1 for i in range(1, len(faces))
                      if not faces[i][1].centroids) / max(len(faces), 1)

        print(f'[{bag}] {duration:.1f} s; face msgs {len(faces)} '
              f'({len(seen)} with faces, no face in {no_face:.0%}); depth '
              f'plausible for {with_depth:.0%} of {len(depths)} faces; '
              f'{len(peaks)} saliency msgs; {len(heads)} head cmds, '
              f'{len(targets)} targets: {len(face_err)} on faces (error '
              f'median {pct(0.5):.1f} deg, p90 {pct(0.9):.1f}), engaged '
              f'chosen {engaged_hits}/{engaged_total}, {len(saliency_targets)} '
              f'saliency, {len(too_early)} inside '
              f'the face timeout {too_early[:5]}', flush=True)

        self.assertGreaterEqual(with_depth, 0.9, 'faces without depth')
        self.assertGreaterEqual(len(face_err), 0.8 * len(targets),
                                'targets not on a visible face')
        self.assertLessEqual(pct(0.5), 2.0, 'target off the face (median)')
        self.assertLessEqual(pct(0.9), 4.0, 'target off the face (p90)')
        if engaged_total >= 20:
            self.assertGreaterEqual(engaged_hits, 0.8 * engaged_total,
                                    'engaged face not preferred')
        if bag == 'multiple_faces':
            self.assertGreaterEqual(engaged_total, 20,
                                    'too few two-face moments to judge')
        self.assertGreaterEqual(
            len(peaks), int(duration * PROCESS_HZ) - 2,
            'saliency node published too few peak messages')
        self.assertEqual(too_early, [],
                         'targets commanded before face_timeout elapsed')
        for _, tgt in saliency_targets:
            self.assertTrue(abs(tgt.x) <= SAL_YAW + EPS and
                            SAL_PITCH_DN - EPS <= tgt.y <= SAL_PITCH_UP + EPS,
                            f'saliency target ({tgt.x}, {tgt.y}) outside limits')

    def _check_enable(self, messages):
        events = {}

        def toggle(value):
            def call():
                req = SetBool.Request()
                req.data = value
                client = self.enable_client
                events[value] = (time.monotonic(), client.call_async(req))
            return call

        self.enable_client = self.probe.create_client(SetBool, ENABLE)
        self.assertTrue(self.enable_client.wait_for_service(timeout_sec=10.0))
        half = (messages[-1][1] - messages[0][1]) / 2
        self._record()
        self._play(messages, at={0.0: toggle(False), half: toggle(True)})
        self._idle(TAIL)
        for value, (_, future) in events.items():
            self.assertTrue(future.done() and future.result().success,
                            f'set_enabled {value} failed')
        off, on = events[False][0], events[True][0]
        heads = self.log['head']
        # A saliency command already under way when the request went in may
        # still arrive; everything after the move to the default pose counts.
        while_off = [m for t, m in heads if off < t < on]
        default = [i for i, m in enumerate(while_off)
                   if abs(m.joint_angles[0] - DEFAULT_YAW) < 1e-3 and
                   abs(m.joint_angles[1] - DEFAULT_PITCH) < 1e-3]
        after_default = while_off[default[0] + 1:] if default else while_off
        after_on = [m for t, m in heads if t > on]
        print(f'[set_enabled] {len(while_off)} head cmds while disabled, '
              f'{len(after_default)} after the default pose; {len(after_on)} '
              'after enabling', flush=True)
        self.assertTrue(default, 'disabling did not send the default pose')
        self.assertEqual(after_default, [], 'commands while disabled')
        self.assertGreater(len(after_on), 0, 'no commands after re-enabling')


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info, face, saliency, attention):
        for proc in (face, saliency, attention):
            launch_testing.asserts.assertExitCodes(
                proc_info, allowable_exit_codes=[0, -signal.SIGINT], process=proc)
