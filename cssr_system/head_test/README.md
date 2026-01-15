# Head Reactivity Test Package

A ROS2 package for testing Pepper robot head reactivity by measuring response time to random points in camera images.

## Overview

This package provides a test node that:
1. Subscribes to camera images and camera info topics
2. Generates random pixel points within the image
3. Converts pixel coordinates to head yaw/pitch angles
4. Commands Pepper's head to look at the selected points
5. Measures the time from command to actual head movement
6. Logs statistics and saves results to CSV files

## Dependencies

- ROS2 (Humble or Foxy)
- `rclpy`, `sensor_msgs`, `std_msgs`, `geometry_msgs`, `naoqi_bridge_msgs`
- `cv_bridge`, `image_transport`
- `opencv-python`, `numpy`

## Installation

1. Build the package:
```bash
cd /home/yoha/ros2_ws
colcon build --packages-select head_test
source install/setup.bash
```

## Usage

### Running the test

Method 1: Using launch file (recommended):
```bash
ros2 launch head_test head_reactivity_test.launch.py
```

Method 2: Direct node execution:
```bash
ros2 run head_test head_reactivity_test
```

### Test Parameters

The test can be configured with the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_duration` | 120.0 | Test duration in seconds |
| `point_interval` | 3.0 | Seconds between random points |
| `settle_threshold` | 0.05 | Angular threshold (rad) to consider head settled |
| `max_response_time` | 5.0 | Timeout for head movement in seconds |
| `image_width` | 640 | Fallback image width (used if `use_image_dimensions=false`) |
| `image_height` | 480 | Fallback image height |
| `joint_state_topic` | `/joint_states` | Topic for Pepper's joint states |
| `camera_info_topic` | `/camera/color/camera_info` | Topic for camera intrinsics |
| `head_command_topic` | `/joint_angles` | Topic to publish head commands |
| `image_topic` | `/camera/color/image_raw` | Topic for camera images |
| `use_image_dimensions` | true | Use actual image dimensions from image topic |
| `visualize` | false | Enable OpenCV visualization of selected points |

### Customizing Parameters

You can override parameters when launching:

```bash
ros2 run head_test head_reactivity_test --ros-args \
  -p test_duration:=60.0 \
  -p point_interval:=2.0 \
  -p visualize:=true
```

## How It Works

1. **Initialization**: The node waits for:
   - Camera info (intrinsics: fx, fy, cx, cy)
   - Joint states (to get current head position)
   - First image (if `use_image_dimensions=true`)

2. **Test Execution**:
   - Generates random pixel point within image bounds
   - Converts pixel to head angles using pinhole camera model
   - Clamps angles to Pepper's joint limits (yaw: ±1.8 rad, pitch: -0.7 to 0.4 rad)
   - Publishes head command to `/joint_angles`
   - Monitors `/joint_states` for head movement
   - Measures time from command to when head reaches target (within threshold)

3. **Statistics**:
   - Real-time logging of response times
   - Final summary with mean, std, min, max, median, 95th percentile
   - Results saved to CSV file: `head_reactivity_results_<timestamp>.csv`

4. **Visualization** (optional):
   - Shows selected points on camera image
   - Displays crosshair and coordinates

## Integration with Existing System

This test is designed to work with the existing Pepper attention system:
- Uses same joint command interface (`/joint_angles` topic)
- Compatible with existing camera topics
- Non-invasive: only publishes head commands, doesn't interfere with other nodes

## Expected Results

The test will output statistics like:
```
HEAD REACTIVITY TEST RESULTS
============================================================
Total trials: 40
Mean response time: 0.873s
Std deviation: 0.123s
Min response time: 0.612s
Max response time: 1.234s
Median response time: 0.845s
95th percentile: 1.102s
============================================================
Results saved to head_reactivity_results_1747315200.csv
```

## Troubleshooting

1. **No camera info received**: Ensure camera is running and publishing to `/camera/color/camera_info`
2. **No joint states**: Check if Pepper's joint states are being published to `/joint_states`
3. **No image received**: Verify camera is publishing to `/camera/color/image_raw`
4. **Head not moving**: Check Pepper's motor power and `/joint_angles` topic subscription

## Files

- `head_test/head_reactivity_test.py` - Main test node
- `launch/head_reactivity_test.launch.py` - Launch file with default parameters
- `package.xml`, `setup.py` - Package configuration
- `README.md` - This documentation

## Testing Without Robot

For simulation/testing without actual Pepper robot:
1. Set `visualize:=true` to see selected points
2. The test will still run but won't measure actual movement
3. Use mock joint states publisher for testing response measurement

## Extending

To modify the test:
1. Change point generation in `generate_random_point()`
2. Adjust response measurement logic in `joint_state_callback()`
3. Add new statistics or visualization features

## License

MIT
