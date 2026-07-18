#!/usr/bin/env python3
"""
Launch-level test for the pepper_odom_anchor lifecycle node.

Starts the real executable with the shipped config/localization.yaml,
then drives it through
configure -> activate via the lifecycle services and asserts every state.

This catches what unit tests cannot: undeclared/mistyped parameters, a config
file that no longer matches the code, and lifecycle callbacks that fail
outside a full robot bringup. No hardware or other nodes are required — the
node only creates a subscription and a timer on activation.

Run via: colcon test --packages-select pepper_odom_anchor

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: Jul 18, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import os
import unittest

import launch
import launch_ros.actions
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import pytest
import rclpy
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState

NODE_NAME = 'robot_localization'
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'localization.yaml')


@pytest.mark.launch_test
def generate_test_description():
    anchor_node = launch_ros.actions.Node(
        package='pepper_odom_anchor',
        executable='pepper_odom_anchor',
        name=NODE_NAME,
        output='screen',
        parameters=[CONFIG_PATH],
    )
    return launch.LaunchDescription([
        anchor_node,
        launch_testing.actions.ReadyToTest(),
    ]), {'anchor_node': anchor_node}


class TestLifecycleTransitions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.probe = rclpy.create_node('lifecycle_probe')

    @classmethod
    def tearDownClass(cls):
        cls.probe.destroy_node()
        rclpy.shutdown()

    def _call(self, srv_type, srv_name, request, timeout=10.0):
        client = self.probe.create_client(srv_type, srv_name)
        try:
            self.assertTrue(
                client.wait_for_service(timeout_sec=timeout),
                f'service {srv_name} not available within {timeout}s')
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.probe, future, timeout_sec=timeout)
            self.assertIsNotNone(future.result(), f'call to {srv_name} timed out')
            return future.result()
        finally:
            self.probe.destroy_client(client)

    def _get_state(self):
        result = self._call(GetState, f'/{NODE_NAME}/get_state', GetState.Request())
        return result.current_state.id

    def _transition(self, transition_id):
        request = ChangeState.Request()
        request.transition = Transition(id=transition_id)
        return self._call(ChangeState, f'/{NODE_NAME}/change_state', request).success

    def test_configure_then_activate(self):
        # Fresh node must be unconfigured.
        self.assertEqual(self._get_state(), State.PRIMARY_STATE_UNCONFIGURED)

        # configure: declares/reads all parameters from the shipped YAML —
        # fails here if code and config file have drifted apart.
        self.assertTrue(self._transition(Transition.TRANSITION_CONFIGURE),
                        'on_configure failed with the shipped localization.yaml')
        self.assertEqual(self._get_state(), State.PRIMARY_STATE_INACTIVE)

        # activate: creates the odom subscription and pose timer.
        self.assertTrue(self._transition(Transition.TRANSITION_ACTIVATE),
                        'on_activate failed')
        self.assertEqual(self._get_state(), State.PRIMARY_STATE_ACTIVE)

        # The advertised pose topic must exist once active.
        topics = dict(self.probe.get_topic_names_and_types())
        self.assertIn('/robot_localization/pose', topics)


@launch_testing.post_shutdown_test()
class TestProcessOutcome(unittest.TestCase):

    def test_exit_code(self, proc_info, anchor_node):
        launch_testing.asserts.assertExitCodes(proc_info, process=anchor_node)
