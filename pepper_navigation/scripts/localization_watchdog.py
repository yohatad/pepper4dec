#!/usr/bin/env python3
"""Stop navigating when localization says it is lost.

fastlio_localization reports its own health (/diagnostics, from the periodic
map-overlap check), but nothing downstream consumed it: a wrong-but-confident
lock left Nav2 planning happily against a costmap built on a bad
map -> base_footprint, and the robot kept driving toward a goal computed from
a pose that was not where it was.

This watches that status and, once it has been bad long enough to not be a
transient, cancels the active navigate_to_pose goal -- and keeps cancelling
while still lost, so a goal issued during that window does not slip through.
It does NOT touch cmd_vel: the collision monitor already owns the
safety-critical stop, and two things arbitrating velocity is worse than one.

Deliberately gated on SUSTAINED badness, for the same reason the health check
itself is: overlap dips for a scan or two whenever the robot turns a corner
into unmapped space or someone walks through the scan, and cancelling a goal
every time that happened would make navigation unusable.

Also treats silence as lost (stale_timeout): if fastlio_localization dies, no
diagnostics arrive at all, and a frozen TF is exactly as dangerous as a wrong
one. Only applies once a first status has been seen, so a slow startup is not
mistaken for a crash.
"""
import rclpy
from action_msgs.srv import CancelGoal
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.node import Node
from std_srvs.srv import Trigger


class LocalizationWatchdog(Node):
    def __init__(self):
        super().__init__('localization_watchdog')
        self.declare_parameter('status_name', 'fastlio_localization: pose lock')
        self.declare_parameter('lost_duration', 5.0)
        self.declare_parameter('stale_timeout', 10.0)     # 0 disables
        self.declare_parameter('treat_warn_as_lost', False)
        self.declare_parameter('cancel_goals', True)
        self.declare_parameter('call_recovery', False)
        self.declare_parameter('nav_action', '/navigate_to_pose')
        self.declare_parameter('recovery_service', '/localization_recover')
        self.declare_parameter('check_period', 1.0)
        self.declare_parameter('recancel_period', 3.0)

        g = self.get_parameter
        self.status_name = g('status_name').value
        self.lost_duration = g('lost_duration').value
        self.stale_timeout = g('stale_timeout').value
        self.warn_is_lost = g('treat_warn_as_lost').value
        self.cancel_goals = g('cancel_goals').value
        self.call_recovery = g('call_recovery').value
        self.check_period = g('check_period').value
        self.recancel_period = g('recancel_period').value

        self.cancel_cli = self.create_client(
            CancelGoal, g('nav_action').value + '/_action/cancel_goal')
        self.recover_cli = self.create_client(Trigger, g('recovery_service').value)

        self.sub = self.create_subscription(
            DiagnosticArray, '/diagnostics', self.on_diagnostics, 10)

        self.last_ok = None        # last time the status was healthy
        self.last_seen = None      # last time the status was seen at all
        self.acting = False        # currently treating localization as lost
        self.last_cancel = None

        self.timer = self.create_timer(self.check_period, self.tick)
        self.get_logger().info(
            f'watching "{self.status_name}"; will cancel goals on '
            f'{self.lost_duration:.0f}s of bad health '
            f'(cancel_goals={self.cancel_goals}, call_recovery={self.call_recovery})')

    def on_diagnostics(self, msg):
        for status in msg.status:
            if status.name != self.status_name:
                continue
            now = self.get_clock().now()
            self.last_seen = now
            bad = (status.level == DiagnosticStatus.ERROR or
                   (self.warn_is_lost and status.level == DiagnosticStatus.WARN))
            if not bad:
                self.last_ok = now
            elif self.last_ok is None:
                # First sighting is already bad: start the clock now rather
                # than treating "never been healthy" as infinitely lost.
                self.last_ok = now
            return

    def _elapsed(self, since):
        if since is None:
            return 0.0
        return (self.get_clock().now() - since).nanoseconds / 1e9

    def tick(self):
        if self.last_seen is None:
            return    # nothing has reported yet; not our business to guess

        stale = (self.stale_timeout > 0.0 and
                 self._elapsed(self.last_seen) > self.stale_timeout)
        bad_for = self._elapsed(self.last_ok)
        lost = stale or bad_for >= self.lost_duration

        if not lost:
            if self.acting:
                self.get_logger().info(
                    'localization healthy again; no longer holding navigation')
                self.acting = False
            return

        if not self.acting:
            reason = (f'no localization status for {self._elapsed(self.last_seen):.0f}s'
                      if stale else f'unhealthy for {bad_for:.0f}s')
            self.get_logger().error(
                f'localization LOST ({reason}) -- holding navigation. '
                f'The pose is not trustworthy; goals will keep being cancelled '
                f'until it recovers.')
            self.acting = True
            self.last_cancel = None
            if self.call_recovery:
                self.request_recovery()

        # last_cancel is None means "never cancelled", i.e. due NOW -- not
        # "just cancelled". Reading it through _elapsed() (which returns 0.0
        # for None) made the period gate permanently closed, so the watchdog
        # detected LOST correctly and then never actually cancelled anything.
        due = (self.last_cancel is None or
               self._elapsed(self.last_cancel) >= self.recancel_period)
        if self.cancel_goals and due:
            self.cancel_all_goals()
            self.last_cancel = self.get_clock().now()

    def cancel_all_goals(self):
        if not self.cancel_cli.service_is_ready():
            self.get_logger().warn(
                'nav2 cancel service not available; cannot hold navigation',
                throttle_duration_sec=30.0)
            return
        # Zero goal_id and zero stamp is the action-spec idiom for "cancel
        # everything", so this needs no goal handle of its own.
        self.cancel_cli.call_async(CancelGoal.Request()).add_done_callback(
            self._cancelled)

    def _cancelled(self, future):
        try:
            n = len(future.result().goals_canceling)
        except Exception as exc:
            self.get_logger().warn(f'cancel call failed: {exc}')
            return
        if n:
            self.get_logger().warn(f'cancelled {n} active navigation goal(s)')

    def request_recovery(self):
        if not self.recover_cli.service_is_ready():
            self.get_logger().warn('/localization_recover not available')
            return
        self.recover_cli.call_async(Trigger.Request()).add_done_callback(
            lambda f: self.get_logger().info(f'recovery: {f.result().message}'))


def main():
    rclpy.init()
    node = LocalizationWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
