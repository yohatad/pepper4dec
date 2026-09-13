#!/usr/bin/env python3
"""One "I am lost, fix it" service that works whichever localization backend is running.

The Nav2 profiles each recover differently, so an operator switching between
them would otherwise have to remember one procedure per backend:

  fastloc  -> /relocalize (std_srvs/Trigger), re-arms the ScanContext search
  pointloc -> /relocalize as well: the Point-LIO localizer is a port of the
              FAST-LIO one and exposes the same service
  amcl     -> /reinitialize_global_localization (std_srvs/Empty), scatters
              particles for a global re-draw
  rtabmap  -> no forced re-search exists; it relocalizes on its own via loop
              closure, so the only operator action is seeding /initialpose

This exposes /localization_recover (Trigger) in front of all of them and
dispatches to whatever is actually up. For rtabmap it reports honestly that
there is nothing to call rather than pretending a no-op succeeded.
"""
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Empty, Trigger

FASTLOC_SRV = '/relocalize'
AMCL_SRV = '/reinitialize_global_localization'


class LocalizationRecovery(Node):
    def __init__(self):
        super().__init__('localization_recovery')
        # 'auto' resolves from the graph; pin it per launch file when known.
        self.declare_parameter('backend', 'auto')
        self.declare_parameter('service_timeout', 5.0)
        self.backend = self.get_parameter('backend').value
        self.timeout = self.get_parameter('service_timeout').value

        # The service callback blocks on a client call, so the two must be able
        # to run concurrently -- same callback group would deadlock under any
        # executor.
        self.srv_group = MutuallyExclusiveCallbackGroup()
        self.cli_group = ReentrantCallbackGroup()

        self.fastloc_cli = self.create_client(
            Trigger, FASTLOC_SRV, callback_group=self.cli_group)
        self.amcl_cli = self.create_client(
            Empty, AMCL_SRV, callback_group=self.cli_group)

        self.srv = self.create_service(
            Trigger, '/localization_recover', self.recover,
            callback_group=self.srv_group)
        self.get_logger().info(
            f'/localization_recover ready (backend={self.backend})')

    def detect_backend(self):
        """Which localization stack is live, by what other nodes actually SERVE.

        Deliberately not get_service_names_and_types(): that also lists
        services this node merely holds a CLIENT for, so it reported our own
        /relocalize client and every auto-detect resolved to 'fastloc' with
        nothing running at all. Asking per node returns servers only.
        """
        names = set()
        me = (self.get_name(), self.get_namespace())
        for node_name, node_ns in self.get_node_names_and_namespaces():
            if (node_name, node_ns) == me:
                continue
            try:
                names.update(
                    n for n, _ in
                    self.get_service_names_and_types_by_node(node_name, node_ns))
            except Exception:
                # Node vanished between listing and querying -- normal churn.
                continue
        if FASTLOC_SRV in names:
            # fastloc and pointloc both serve /relocalize, so the service alone
            # cannot tell them apart -- disambiguate by who is running.
            live = {n for n, _ in self.get_node_names_and_namespaces()}
            return 'pointloc' if 'point_lio_localization' in live else 'fastloc'
        if AMCL_SRV in names:
            return 'amcl'
        if any(n.startswith('/rtabmap') for n in names):
            return 'rtabmap'
        return None

    def _await(self, future):
        """Wait for a client call without blocking this node's own executor."""
        deadline = time.monotonic() + self.timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        return future.result() if future.done() else None

    def recover(self, request, response):
        backend = self.backend if self.backend != 'auto' else self.detect_backend()

        if backend in ('fastloc', 'pointloc'):
            if not self.fastloc_cli.wait_for_service(timeout_sec=self.timeout):
                response.success = False
                response.message = f'{FASTLOC_SRV} did not appear'
                return response
            result = self._await(self.fastloc_cli.call_async(Trigger.Request()))
            if result is None:
                response.success = False
                response.message = f'{FASTLOC_SRV} timed out'
                return response
            node_name = ('pointlio_localization' if backend == 'pointloc'
                         else 'fastlio_localization')
            response.success = result.success
            response.message = f'{node_name}: {result.message}'

        elif backend == 'amcl':
            if not self.amcl_cli.wait_for_service(timeout_sec=self.timeout):
                response.success = False
                response.message = f'{AMCL_SRV} did not appear'
                return response
            if self._await(self.amcl_cli.call_async(Empty.Request())) is None:
                response.success = False
                response.message = f'{AMCL_SRV} timed out'
                return response
            response.success = True
            response.message = ('amcl: particles re-scattered globally. Drive the '
                                'robot so the filter can converge.')

        elif backend == 'rtabmap':
            # Not a failure of this node -- rtabmap simply has no equivalent
            # primitive, and saying so beats reporting a success that did nothing.
            response.success = False
            response.message = ('rtabmap has no forced re-search: it relocalizes '
                                'from loop closure on its own. Seed it instead '
                                'with RViz 2D Pose Estimate (/initialpose).')

        else:
            response.success = False
            response.message = ('no known localization backend detected '
                                '(looked for /relocalize, '
                                '/reinitialize_global_localization, /rtabmap*)')

        log = self.get_logger().info if response.success else self.get_logger().warn
        log(f'/localization_recover -> {response.message}')
        return response


def main():
    rclpy.init()
    node = LocalizationRecovery()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
