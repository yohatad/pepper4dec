#!/bin/bash

# Wait for the camera node to be ready
sleep 3

# Set all QoS to SENSOR_DATA
ros2 param set /camera color_qos SENSOR_DATA
ros2 param set /camera color_info_qos SENSOR_DATA
ros2 param set /camera depth_qos SENSOR_DATA
ros2 param set /camera depth_info_qos SENSOR_DATA
ros2 param set /camera infra1_qos SENSOR_DATA
ros2 param set /camera infra1_info_qos SENSOR_DATA
ros2 param set /camera infra2_qos SENSOR_DATA
ros2 param set /camera infra2_info_qos SENSOR_DATA
ros2 param set /camera accel_qos SENSOR_DATA
ros2 param set /camera accel_info_qos SENSOR_DATA
ros2 param set /camera gyro_qos SENSOR_DATA
ros2 param set /camera gyro_info_qos SENSOR_DATA

# Restart streams to apply QoS
ros2 param set /camera enable_color false
ros2 param set /camera enable_depth false
sleep 1
ros2 param set /camera enable_color true
ros2 param set /camera enable_depth true

echo "QoS parameters set to SENSOR_DATA and streams restarted"