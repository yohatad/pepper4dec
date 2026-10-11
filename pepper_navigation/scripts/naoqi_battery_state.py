#!/usr/bin/env python3
"""Pepper's battery as sensor_msgs/BatteryState, from naoqi_driver's /diagnostics.

naoqi_driver reports the battery only as diagnostics: the status
"naoqi_driver_battery:Status" carries Percentage, "Charging" (NAOqi's
BatteryPowerPlugged flag: power is connected) and "Fully Charged", and
"naoqi_driver_battery:Current" carries Current.

Two topics come out:

  battery_state: the battery as it is. Current is NAOqi's, times current_sign,
      so that charging is positive as BatteryState wants (check the sign on the
      robot: TEST_PLAN_docking.md).
  docking/battery_state: for opennav_docking's SimpleChargingDock, whose
      isCharging() is just `current > charging_threshold`. Current is +1 A while
      the robot is plugged in and -1 A otherwise. Plugged-in, not current, is
      what docking needs: a full battery draws no current but is still docked.

Subscribes:
    /diagnostics (diagnostic_msgs/DiagnosticArray)

Publishes:
    battery_state, docking/battery_state (sensor_msgs/BatteryState)

Parameters:
    current_sign (default: 1.0)
        Multiplies NAOqi's current so that charging comes out positive.
"""

import math

from diagnostic_msgs.msg import DiagnosticArray
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import BatteryState

STATUS = 'naoqi_driver_battery:Status'
CURRENT = 'naoqi_driver_battery:Current'


def parse_battery(statuses):
    """{'percentage', 'plugged', 'full', 'current'} from DiagnosticStatus list, or None.

    Returns None until the battery status has been seen; current is NaN if
    its status is missing.
    """
    values = {}
    for status in statuses:
        if status.name in (STATUS, CURRENT):
            values.update({kv.key: kv.value for kv in status.values})
    if 'Percentage' not in values:
        return None

    def flag(key):
        return values.get(key, 'False').strip().lower() in ('true', '1')

    try:
        current = float(values['Current'])
    except (KeyError, ValueError):
        current = math.nan
    return {'percentage': float(values['Percentage']) / 100.0, 'plugged': flag('Charging'),
            'full': flag('Fully Charged'), 'current': current}


def battery_states(battery, current_sign=1.0):
    """(battery_state, docking battery_state) messages for a parse_battery() result."""
    state = BatteryState()
    state.present = True
    state.percentage = battery['percentage']
    state.current = battery['current'] * current_sign
    state.voltage = state.charge = state.capacity = state.design_capacity = math.nan
    state.temperature = math.nan
    state.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
    if not battery['plugged']:
        state.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
    elif battery['full']:
        state.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_FULL
    else:
        state.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_CHARGING
    docking = BatteryState()
    docking.present = True
    docking.percentage = state.percentage
    docking.power_supply_status = state.power_supply_status
    docking.current = 1.0 if battery['plugged'] else -1.0
    return state, docking


class NaoqiBatteryState(Node):

    def __init__(self):
        super().__init__('naoqi_battery_state')
        self.current_sign = self.declare_parameter('current_sign', 1.0).value
        self.state_pub = self.create_publisher(BatteryState, 'battery_state', 10)
        self.docking_pub = self.create_publisher(BatteryState, 'docking/battery_state', 10)
        self.create_subscription(DiagnosticArray, '/diagnostics', self.on_diagnostics, 10)

    def on_diagnostics(self, msg):
        battery = parse_battery(msg.status)
        if battery is None:
            return
        state, docking = battery_states(battery, self.current_sign)
        state.header = docking.header = msg.header
        self.state_pub.publish(state)
        self.docking_pub.publish(docking)


def main():
    rclpy.init()
    node = NaoqiBatteryState()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
