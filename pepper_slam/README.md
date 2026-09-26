<div align="center">
<h1>Pepper SLAM and Mapping</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

Mapping, odometry and localization bringup for the Pepper sensor rig: a
Unitree L2 lidar (`/points`, `/imu/data`) and a RealSense (RGB, aligned depth,
IMU). The package ships launch files, parameters and a few rclpy helper nodes;
the estimators themselves (FAST-LIO, Point-LIO, RTAB-Map) are upstream
packages launched by name, so nothing here compiles.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **LIO odometry**: FAST-LIO or Point-LIO on the Unitree L2, bridged to a gravity-levelled `odom`
- **Divergence guard**: optionally rejects LIO poses the base cannot physically produce
- **Mapping**: FAST-LIO odometry with RTAB-Map loop closure
- **Prior-map localization**: standalone FAST-LIO and Point-LIO localizers for testing outside Nav2
- **Sensor rig TF**: one static `base_footprint -> l2lidar_frame -> camera` chain shared by every launch file

## ✅ Prerequisites

- **ROS 2 Humble**
- **`fast_lio`** and/or **`point_lio`** built in the same workspace
- **`rtabmap_ros`** for the RTAB-Map mapping and localization paths
- **Unitree L2 lidar** via `l2lidar_node`, and the RealSense via
  `realsense2_camera`. Neither is started by this package:
  `ros2 launch dec_launch dec_robot.launch.py` brings both up

## 🛠️ Installation

```bash
sudo apt install \
  ros-humble-rtabmap-ros \
  ros-humble-imu-filter-madgwick

cd ~/ros2_ws
colcon build --packages-select pepper_slam
source install/setup.bash
```

`fast_lio` and `point_lio` are source packages: build them in the same
workspace first (see Prerequisites).

## 🚀 Running

```bash
source ~/ros2_ws/install/setup.bash
```

### Sensor rig TF

Every launch file below nests `pepper_sensor_tf.launch.py`, which publishes
the static `base_footprint -> l2lidar_frame -> camera` chain. Run it alone
only when using the drivers by themselves (recording a bag, looking at
`/points`):

```bash
ros2 launch pepper_slam pepper_sensor_tf.launch.py
```

| `scope` | Publishes | Use for |
|---|---|---|
| `mount` (default) | the rig mount edges only | the real robot, and bags that recorded `/tf_static` |
| `all` | the mount edges plus the RealSense-internal edges | older bags with an empty `/tf_static` |

On the robot `realsense2_camera` publishes its own internal edges, so `all`
there gives them two publishers and the last one silently wins.

### Odometry

```bash
ros2 launch pepper_slam fastlio_odometry.launch.py      # FAST-LIO
ros2 launch pepper_slam pointlio_odometry.launch.py     # Point-LIO
```

Both start the estimator plus `lio_odom_bridge`, which publishes
`odom -> base_footprint`. Use these, not `fast_lio`'s own `mapping.launch.py`,
which lacks the rig TF and hangs with no error. There is no loop closure: use
the mapping path below for a map worth keeping.

- `bridge_level_frame` (default `true`): publishes a gravity-levelled `odom`
  above the estimator's tilted `lio_init`. Set it `false` whenever RTAB-Map,
  PGO or AMCL owns `odom`'s parent, or `odom` gets two parents.
- `guard_enable` (default `false`): rejects poses implying motion the base
  cannot perform, carries the pose on wheel odometry meanwhile, and publishes
  `/localization/wheel_trust_scale` and `/diagnostics`. Thresholds are in
  `lio_odom_guard.py`. `pointlio_odometry.launch.py` does not forward it; set
  it on the bridge node there.

### Mapping

The validated configuration is FAST-LIO odometry with RTAB-Map loop closure,
run from a bag:

```bash
ros2 launch pepper_slam rtabmap_fastlio_bag.launch.py
ros2 launch pepper_slam rtabmap_fastlio_bag.launch.py rgbd:=true   # colour the map from the depth camera
```

Keep `rgbd:=false` for maps Nav2 will use: with it on, camera points are
folded into the 2D grid too. `rtabmap_base.launch.py` is the underlying
bringup (`localization:=true` to stop mapping, `rviz:=true` for RViz).

### Localization against a prior map

For testing the localizer without Nav2; `pepper_navigation`'s fastloc and
pointloc profiles start these themselves.

```bash
ros2 launch pepper_slam fastlio_localization.launch.py
ros2 launch pepper_slam pointlio_localization.launch.py
```

Both include the rig TF and default `use_sim_time:=false`; pass `true` only
when replaying a bag with `--clock`.

### Recording bags

**Record `/tf_static` with the QoS override, or you will lose it:**

```bash
ros2 bag record -a \
  --qos-profile-overrides-path $(ros2 pkg prefix --share pepper_slam)/config/record_qos.yaml \
  -o my_bag
```

`/tf_static` is published once and latched, and the recorder only catches it
with the override. A bag recorded this way needs no `pepper_sensor_tf` on
replay. `-a` includes 30 Hz colour; name topics explicitly for long sessions.

### Bag replay

The files in `launch/bag_test/` wrap a live launch file with
`use_sim_time:=true` and replay-only settings. Launch them by file name only:
the `bag_test/` prefix fails with "not found".

| Bag entry point | Wraps |
|---|---|
| `fastlio_odometry_bag` / `pointlio_odometry_bag` | `fastlio_odometry` / `pointlio_odometry` |
| `fastlio_lc_bag` / `pointlio_lc_bag` | `fastlio_lc_pgo`'s `fastlio_lc_l2` / `pointlio_lc_l2` |
| `rtabmap_*_bag` | `rtabmap_base` plus a LIO |
| `odom_compare_bag` | FAST-LIO against the bag's wheel odometry, side by side in RViz |

Localization and navigation need no wrapper; they take `use_sim_time`
directly.

```bash
ros2 bag play <bag> --clock --read-ahead-queue-size 2000
```

- **Subscriber QoS:** the sensor topics replay BEST_EFFORT, as recorded.
  Subscribe with `SensorDataQoS`; a RELIABLE subscriber silently gets nothing.
- **Replay `/tf`:** it carries Pepper's body chain, including the head-camera
  frames. Older bags may also carry a wheel-odometry `odom -> base_footprint`
  edge that fights the bridge; check with `view_frames` before trusting one.
- **Rig transforms:** every wrapper defaults `publisher:=none`, which suits
  bags recorded with `record_qos.yaml`. For `slam_recording*` and
  `slam_bench_run*` (no `/tf_static`), pass `publisher:=urdf scope:=all`.
- **IMU:** the wrappers use the RealSense IMU. To try the L2's own, pass both
  `config_file:=l2.yaml` and `lidar_imu_frame:=l2lidar_frame_imu`; with only
  the first, `odom -> base_footprint` never closes.
- **Arguments:** `--show-args` lists every argument in the include tree, far
  more than a file honours; read the "ARGUMENTS THIS FILE HONOURS" block in its
  header instead.
- **Backgrounding the player:** add `--disable-keyboard-controls`, or SIGTTIN
  stops it.

## 🧭 Frames

| Frame | Published by | Notes |
|---|---|---|
| `pepper_odom` | `naoqi_driver2` | wheel odometry; deliberately **not** named `odom` |
| `lio_init` | FAST-LIO / Point-LIO | the estimator's world frame, tilted by the sensor mount |
| `odom` | `lio_odom_bridge` | gravity-levelled parent of `lio_init`; Z-up |
| `map` | RTAB-Map or the PGO bridge | whichever layer owns the loop-closure correction |
| `map` | `fastlio_localization` / `pointlio_localization` | publishes `map -> base_footprint` directly, with no `odom` edge |

Only one node may publish a frame's parent. RTAB-Map anchors on `odom`, not
the tilted `lio_init`. `FRAMES.md` has the full frame contract.

## 📁 Package Structure

```
pepper_slam/
├── config/
│   ├── record_qos.yaml                 # /tf_static QoS override for ros2 bag record
│   └── sensor_tf.yaml                  # rig transform values; provenance in the header
├── launch/
│   ├── pepper_sensor_tf.launch.py      # static rig TF, nested by everything else
│   ├── fastlio_odometry.launch.py      # FAST-LIO odometry + bridge
│   ├── pointlio_odometry.launch.py     # same, for Point-LIO
│   ├── lio_odom_bridge.launch.py       # the bridge alone, for an estimator started elsewhere
│   ├── fastlio_localization.launch.py  # standalone prior-map localization
│   ├── pointlio_localization.launch.py # same, for Point-LIO
│   ├── rtabmap_base.launch.py          # RTAB-Map bringup (vendored upstream; excluded from flake8)
│   ├── view_rig.launch.py              # sensor rig visualization from the URDF
│   └── bag_test/                       # bag-replay wrappers (see Bag replay)
├── rviz/                               # configs for the mapping, replay and rig-view launches
├── scripts/
│   ├── lio_odom_bridge.py              # odom -> base_footprint from the LIO pose; levels lio_init once
│   ├── lio_odom_guard.py               # divergence gate; a module imported by the bridge
│   ├── pgo_map_odom_bridge.py          # map -> odom from a pose-graph-optimized trajectory
│   ├── cloud_range_filter.py           # range / outlier / ground filters; feeds Nav2's /points_safety and /points_costmap
│   ├── cloud_to_odom_frame.py          # re-expresses a cloud in the odom frame
│   ├── check_frame_contract.py         # asserts the LIO frame contract, whichever backend runs
│   ├── odom_compare.py                 # side-by-side odometry comparison
│   └── static_tf_publisher.py          # a whole static TF chain from one node
├── test/                               # pytest suites, resolve_launch.py and its baselines
├── urdf/                               # the L2 + D435i rig xacro, alone and on Pepper's body
├── FRAMES.md                           # the frame contract in full
├── CMakeLists.txt
├── package.xml
└── README.md
```

Saved maps live in `pepper_navigation/`, which uses them; RTAB-Map `.db` files
go to `~/.ros/`.

## 🧪 Testing

```bash
cd ~/ros2_ws
colcon test --packages-select pepper_slam
colcon test-result --verbose
```

Besides the linters, this runs `test_leveling_frame_contract.py` (skipped when
the LIO packages are not built) and `test_lio_odom_guard.py`.

**Refactoring a launch file:** `test/resolve_launch.py` resolves one to the
nodes, parameters and remappings it will start. Compare against the snapshots
in `test/launch_baselines/`, and regenerate those only deliberately:

```bash
SNAP_OUT=after.json python3 test/resolve_launch.py \
    $(ros2 pkg prefix --share pepper_slam)/launch/fastlio_odometry.launch.py
diff <(jq -S . test/launch_baselines/fl_odom.json) <(jq -S . after.json)
```

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2025 Carnegie Mellon University Africa
Licensed under the BSD-3-Clause License. See individual package licenses for details.
