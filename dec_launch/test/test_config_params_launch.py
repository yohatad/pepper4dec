#!/usr/bin/env python3
"""
Every node loads its configuration YAML the way the launch files pass it.

Starts each node with ``parameters=[<its config YAML>]``, as its own launch
file does, then reads every parameter from the YAML back through the running
node's ``get_parameters`` service and compares. Catches:

  - a YAML section keyed by the wrong node name (the node silently runs on
    its defaults),
  - a YAML parameter the node never declares (ROS drops it without a word);
    a ``/**`` (all nodes) parameter only has to be declared by the nodes that
    use it, so it is not required of each one,
  - a value that does not survive loading, and
  - a wrong-typed value (e.g. ``2`` for a double): the node exits at startup,
    so its services never appear and the test fails pointing at its log.

Nested YAML maps are flattened to ROS's dotted names (``a: {b: 1}`` is
parameter ``a.b``), and ``/**`` sections apply to every node in the file.

Nodes whose imports need pip-only libraries (see ``requires``) are skipped,
by name, where those libraries are missing, as they are in CI. Nodes that
declare their parameters in on_configure() are sent the configure
transition first; that transition may then fail for lack of a model file or
a camera, which is fine: the parameters are declared before that point.

Run via: colcon test --packages-select dec_launch

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: September 25, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import importlib.util
import math
import os
import unittest

from ament_index_python.packages import get_package_share_directory
import launch
import launch_ros.actions
import launch_testing.actions
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState
import pytest
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.parameter import parameter_value_to_python
import yaml


# (node name, package, executable, config file in <package>/config,
#  send configure before reading, pip-only modules its imports need)
NODES = [
    ('animate_behavior', 'animate_behavior', 'animate_behavior',
     'animate_behavior_configuration.yaml', False, []),
    ('behavior_controller', 'behavior_controller', 'behavior_controller',
     'behavior_controller_configuration.yaml', False, []),
    ('face_detection', 'face_detection', 'face_detection',
     'face_detection_configuration.yaml', True, []),
    ('age_gender_detection', 'face_detection', 'age_gender_detection',
     'age_gender_detection_configuration.yaml', True, []),
    ('gesture_action_server', 'gesture_execution', 'gesture_execution',
     'gesture_execution_configuration.yaml', False, []),
    ('overt_attention', 'overt_attention', 'overt_attention',
     'overt_attention_configuration.yaml', False, []),
    ('saliency_node', 'overt_attention', 'overt_attention_saliency',
     'overt_attention_configuration.yaml', False, []),
    ('attention_visualization', 'overt_attention', 'overt_attention_visualization',
     'overt_attention_configuration.yaml', False, []),
    ('person_detection', 'person_detection', 'person_detection',
     'person_detection_configuration.yaml', True, []),
    ('speech_recognition', 'speech_event', 'speech_event',
     'speech_event_configuration.yaml', False, ['faster_whisper', 'onnxruntime']),
    ('sound_localization', 'speech_event', 'speech_event_localization',
     'speech_event_configuration.yaml', False, ['pyroomacoustics']),
    ('audio_recorder', 'speech_event', 'speech_event_recorder',
     'speech_event_configuration.yaml', False, []),
    ('text_to_speech', 'text_to_speech', 'text_to_speech',
     'text_to_speech_configuration.yaml', False, []),
    ('conversation_manager', 'conversation_manager', 'conversation_manager',
     'conversation_manager_configuration.yaml', False, ['openai', 'chromadb']),
]

SERVICE_TIMEOUT_S = 30.0
CALL_TIMEOUT_S = 30.0


def config_path(package, filename):
    return os.path.join(get_package_share_directory(package), 'config', filename)


def missing_modules(requires):
    return [m for m in requires if importlib.util.find_spec(m) is None]


def flatten(params, prefix=''):
    """Turn nested YAML maps into ROS's dotted parameter names."""
    flat = {}
    for key, value in params.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            flat.update(flatten(value, name + '.'))
        else:
            flat[name] = value
    return flat


def expected_parameters(path, node_name):
    """Return (params the YAML sets for node_name, names that come only from /**)."""
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    shared = flatten((doc.get('/**') or {}).get('ros__parameters') or {})
    own = {}
    for key in (node_name, '/' + node_name):
        own.update(flatten((doc.get(key) or {}).get('ros__parameters') or {}))
    return {**shared, **own}, set(shared) - set(own)


def same(expected, actual):
    if isinstance(expected, float) or isinstance(actual, float):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-9))
    if isinstance(expected, list):
        return (isinstance(actual, (list, tuple)) and len(expected) == len(actual)
                and all(same(e, a) for e, a in zip(expected, actual)))
    return expected == actual


RUNNABLE = [n for n in NODES if not missing_modules(n[5])]


@pytest.mark.launch_test
def generate_test_description():
    nodes = [
        launch_ros.actions.Node(
            package=package, executable=executable, name=name,
            parameters=[config_path(package, config)], output='screen')
        for name, package, executable, config, _, _ in RUNNABLE
    ]
    return launch.LaunchDescription(nodes + [launch_testing.actions.ReadyToTest()])


class TestConfigParameters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('config_param_checker')

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=CALL_TIMEOUT_S)
        self.assertTrue(future.done(), f'{client.srv_name} did not answer')
        return future.result()

    def check_node(self, name, package, config, configure_first, requires):
        missing = missing_modules(requires)
        if missing:
            self.skipTest(f'{name} needs {", ".join(missing)} (pip-only, not installed)')

        path = config_path(package, config)
        expected, shared_only = expected_parameters(path, name)
        self.assertTrue(expected, f'{config} has no parameters for node "{name}": '
                                  'is its top-level key the node name?')

        getter = self.node.create_client(GetParameters, f'/{name}/get_parameters')
        self.assertTrue(
            getter.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S),
            f'{name} never came up. A wrong-typed YAML value makes a node exit at '
            'startup; see its output above.')

        if configure_first:
            changer = self.node.create_client(ChangeState, f'/{name}/change_state')
            self.assertTrue(changer.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S))
            # The outcome does not matter: the parameters are declared before
            # the step that may fail here (model file, camera).
            self.call(changer, ChangeState.Request(
                transition=Transition(id=Transition.TRANSITION_CONFIGURE)))

        names = sorted(expected)
        values = list(self.call(getter, GetParameters.Request(names=names)).values)
        if len(values) != len(names):
            # rclcpp answers the whole request with an empty list when any one
            # name is undeclared, so ask again one name at a time. (rclpy
            # instead returns PARAMETER_NOT_SET for that name.)
            values = []
            for param in names:
                one = self.call(getter, GetParameters.Request(names=[param])).values
                values.append(one[0] if one else None)
        problems = []
        for param, value in zip(names, values):
            if value is None or value.type == ParameterType.PARAMETER_NOT_SET:
                if param in shared_only:
                    continue  # a /** entry this node does not use
                problems.append(f'  {param}: in the YAML but not declared by the node')
                continue
            actual = parameter_value_to_python(value)
            if not same(expected[param], actual):
                problems.append(f'  {param}: YAML {expected[param]!r}, node has {actual!r}')
        self.assertFalse(problems, f'{name} ({config}):\n' + '\n'.join(problems))


def _make_test(entry):
    name, package, _, config, configure_first, requires = entry

    def test(self):
        self.check_node(name, package, config, configure_first, requires)
    test.__name__ = f'test_{name}'
    test.__doc__ = f'{name} loads {package}/config/{config}'
    return test


for _entry in NODES:
    setattr(TestConfigParameters, f'test_{_entry[0]}', _make_test(_entry))
