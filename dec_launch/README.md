<div align="center">
<h1>DEC Launch</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **DEC Launch** package holds the top-level launch files for the Pepper4DEC system: the full tour system, the sensor drivers, the NAOqi bridge to the robot, and a few calibration and debugging tools. Each package still owns its own launch file; this package composes them and sequences their lifecycle nodes.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **One-command bringup**: `dec_system.launch.py` starts every package and drives their lifecycle nodes to `active` in dependency order
- **Selectable navigation**: pick the Nav2 localization profile at launch, or run without navigation
- **Sensor layer**: the Unitree L2 lidar and the bottom RealSense, configured for this rig
- **Config check**: a launch test that loads every node with its YAML and reads each parameter back

## ✅ Prerequisites
- **ROS2 Humble** or newer
- Every pepper4dec package built in the same workspace
- **`l2lidar_node`**, **`realsense2_camera`** and **`naoqi_driver2`** for the drivers
- **`nav2_lifecycle_manager`**

## 🛠️ Installation

### Package Installation

```bash
cd ~/ros2_ws
colcon build --packages-up-to dec_launch
source install/setup.bash
```

## 🚀 Running

### Full system

```bash
ros2 launch dec_launch dec_system.launch.py
ros2 launch dec_launch dec_system.launch.py nav_profile:=rtabmap_loc
ros2 launch dec_launch dec_system.launch.py enable_navigation:=false
```

| Argument | Default | Meaning |
|---|---|---|
| `enable_navigation` | `true` | bring up `pepper_navigation`. With `false`, `fastlio_localization` still starts on its own, so `/localization/pose` exists for `gesture_execution` |
| `nav_profile` | `fastloc` | `fastloc`, `pointloc`, `rtabmap_loc` or `amcl`; see the `pepper_navigation` README |

### Sensors and robot

```bash
# L2 lidar + bottom RealSense (drivers only)
ros2 launch dec_launch dec_robot.launch.py

# The NAOqi bridge to Pepper, started separately
ros2 launch dec_launch naoqi_driver.launch.py nao_ip:=<robot ip>
```

`dec_robot.launch.py` takes `enable_lidar` and `enable_camera` (both `true`). It publishes no static TF: every stack that uses the sensors includes `pepper_slam`'s `pepper_sensor_tf.launch.py`. When running the drivers alone (recording a bag, looking at `/points` in RViz), start that yourself:

```bash
ros2 launch pepper_slam pepper_sensor_tf.launch.py
```

`naoqi_driver.launch.py` defaults `nao_ip` to `172.29.111.240` (the CMU-Pepper router). On the PepperNet access point the robot is at `10.42.0.204`.

### Other launch files

| Launch file | Starts |
|---|---|
| `l2lidar.launch.py` | the L2 driver alone (`/points` about 11 Hz, `/imu/data`) |
| `realsense_bottom.launch.py` | the bottom RealSense alone, with its point cloud |
| `asr_cm_pipeline.launch.py` | `speech_event`, `conversation_manager` and `behavior_controller` with their own lifecycle manager |
| `bag_static_tf.launch.py` | the static TF recorded in `slam_august_8_bag`, for replaying it with `--start-offset` (which skips the bag's own `/tf_static`) |

### Tools

```bash
ros2 run dec_launch lidar_depth_calibrator.py   # L2-to-RealSense extrinsic by ICP; point both sensors at a wall corner
ros2 run dec_launch lidar_colorizer.py          # colours /points from the RealSense image, on /points_colored
ros2 run dec_launch depth_roi_service.py        # serves get_depth_roi (dec_interfaces/srv/GetDepthROI)
```

**The L2 publishes `/points` and `/imu/data` BEST_EFFORT.** A subscriber that asks for RELIABLE receives nothing, and only the driver logs a warning. `ros2 bag record` defaults to RELIABLE, so record with `--qos-profile-overrides-path` (see the `pepper_slam` README).

## 🖥️ ROS Interface

### Lifecycle management

`dec_system.launch.py` runs one `nav2_lifecycle_manager` (`lifecycle_manager_dec_system`) that drives these nodes from `unconfigured` to `active`, in this order: `person_detection`, `face_detection`, `overt_attention`, `animate_behavior`, `gesture_action_server`, `speech_recognition`, `text_to_speech`, `conversation_manager`, `behavior_controller`. The Nav2 and localization nodes are managed by their own launch files.

## 📁 Package Structure

```
dec_launch/
├── config/
│   └── realsense_bottom_pointcloud.yaml  # enables the RealSense point cloud filter
├── launch/
│   ├── dec_system.launch.py              # full system (default entry point)
│   ├── dec_robot.launch.py               # L2 + bottom RealSense drivers
│   ├── naoqi_driver.launch.py            # the NAOqi bridge to Pepper
│   ├── l2lidar.launch.py                 # L2 driver
│   ├── realsense_bottom.launch.py        # bottom RealSense driver
│   ├── asr_cm_pipeline.launch.py         # speech -> conversation -> behavior only
│   ├── bag_static_tf.launch.py           # static TF for slam_august_8_bag replays
│   └── reset.sh                          # sets the RealSense topics to SENSOR_DATA QoS
├── scripts/
│   ├── lidar_depth_calibrator.py         # L2-to-RealSense extrinsic by ICP
│   ├── lidar_colorizer.py                # /points coloured by the RealSense image
│   └── depth_roi_service.py              # GetDepthROI service
├── test/
│   └── test_config_params_launch.py      # every node against its config YAML
├── ament_flake8.ini
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🧪 Testing

```bash
cd ~/ros2_ws
colcon test --packages-select dec_launch
colcon test-result --verbose
```

`test_config_params_launch.py` starts each node with its config YAML, as its launch file does, and reads every parameter back through the running node. It fails on a YAML section keyed by the wrong node name, a parameter the node never declares, a value that does not survive loading, or a wrong-typed value. Python nodes whose virtualenv is missing (as in CI) are skipped by name.

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2025 Carnegie Mellon University Africa
Licensed under the BSD-3-Clause License. See individual package licenses for details.
