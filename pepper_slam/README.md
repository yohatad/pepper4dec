<div align="center">
<h1>Pepper SLAM and Mapping</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

Mapping, odometry and localization bringup for the Pepper sensor rig: a
Unitree L2 lidar (`/points`, `/imu/data`) and a RealSense (RGB, aligned depth,
IMU). The package ships launch files, parameters and a few rclpy helper nodes;
the estimators themselves (FAST-LIO, Point-LIO) are upstream
packages launched by name, so nothing here compiles.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **LIO odometry**: FAST-LIO or Point-LIO on the Unitree L2, bridged to a gravity-levelled `odom`
- **Divergence guard**: optionally rejects LIO poses the base cannot physically produce
- **Mapping**: FAST-LIO odometry with pose-graph (`fastlio_lc_pgo`) loop closure
- **Prior-map localization**: standalone FAST-LIO and Point-LIO localizers for testing outside Nav2
- **Sensor rig TF**: one static `base_footprint -> l2lidar_frame -> camera` chain shared by every launch file

## ✅ Prerequisites

- **ROS 2 Humble**
- **`fast_lio`** and/or **`point_lio`** built in the same workspace
- **`fastlio_lc_pgo`** built in the same workspace, for mapping with loop
  closure
- **Unitree L2 lidar** via `l2lidar_node`, and the RealSense via
  `realsense2_camera`. Neither is started by this package:
  `ros2 launch dec_launch dec_robot.launch.py` brings both up

## 🛠️ Installation

```bash
cd ~/ros2_ws
colcon build --packages-select pepper_slam
source install/setup.bash
```

`fast_lio`, `point_lio` and `fastlio_lc_pgo` are source packages: build them
in the same workspace first (see Prerequisites).

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
  above the estimator's tilted `lio_init`. Set it `false` whenever PGO or
  AMCL owns `odom`'s parent, or `odom` gets two parents.
- `guard_enable` (default `false`): rejects poses implying motion the base
  cannot perform, carries the pose on wheel odometry meanwhile, and publishes
  `/localization/wheel_trust_scale` and `/diagnostics`. Thresholds are in
  `pepper_slam_configuration.yaml`. `pointlio_odometry.launch.py` does not forward it; set
  it on the bridge node there.

### Mapping

Mapping is FAST-LIO odometry with pose-graph loop closure (radius search,
ICP-verified, optimized in GTSAM), in `fastlio_lc_pgo`. From a bag:

```bash
ros2 launch pepper_slam fastlio_lc_bag.launch.py save_directory:=~/Lidar/run_l2_lc/pgo_output/
```

`save_directory` is required (no default). When the bag finishes, write the
map with `ros2 service call /pgo_batch_optimize std_srvs/srv/Trigger`; see
`fastlio_lc_pgo`'s own README for the full option list. The same run also
builds the 2D occupancy grid on `/projected_map` (octomap).

RTAB-Map was the earlier mapping backend and has been removed: its
appearance-based loop closure needs RTAB-Map's own visual odometry to own
the pose, which is incompatible with using FAST-LIO, and every configuration
tried produced worse maps than the pose-graph path.

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
  --qos-profile-overrides-path $(ros2 pkg prefix --share pepper_slam)/data/record_qos.yaml \
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
- **Rig transforms:** every wrapper defaults `publisher:=urdf scope:=all`
  and republishes the whole rig, so any bag replays, with or without
  `/tf_static`. A bag's own copy sits at t=0, so `--start-offset` skips it.
  Pass `publisher:=none` only to replay a bag's recorded rig, from the start.
- **IMU:** the wrappers use the RealSense IMU, and that is now the only
  option: `l2.yaml`, which drove the estimator from the L2's own IMU, was
  removed after it diverged on `slam_20260823_aligned` (map overlap 1.00 ->
  0.29, health warnings that the estimate was diverging). It also needed
  `lidar_imu_frame:=l2lidar_frame_imu` passed alongside it, or
  `odom -> base_footprint` never closed.
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
| `map` | the PGO bridge or AMCL | whichever layer owns the loop-closure correction |
| `map` | `fastlio_localization` / `pointlio_localization` | publishes `map -> base_footprint` directly, with no `odom` edge |

Only one node may publish a frame's parent. `FRAMES.md` has the full frame
contract.

## ⚙️ Configuration

Every node reads its parameters from `config/pepper_slam_configuration.yaml`,
one section per node name (`lio_odom_bridge`, `pgo_map_odom_bridge`,
`points_safety_filter`, `points_costmap_filter`, `wheel_odom_tf`), like the
other pepper4dec packages' configuration files. The launch files add only `use_sim_time` and
the body frame the bridges must share with the estimator (`lidar_imu_frame`,
read from the estimator's config, so it is never in these files). A launch
argument such as `guard_enable:=true` overrides the file for that launch only;
left empty, the file's value stands.

`data/record_qos.yaml` is the `ros2 bag record` QoS override for `/tf_static`
(see "Recording bags"). The estimators' own configurations are `fast_lio`'s
`config/l2_rsimu.yaml` and `point_lio`'s `config/l2lidar_rsimu.yaml`.

The rig's geometry and its calibration notes live in `urdf/sensor_rig.xacro`,
which `pepper_sensor_tf.launch.py` publishes. Three values elsewhere are
derived from it by hand: the `extrinsic_R` / `extrinsic_T` in both estimator
configs and `sc_lidar_height` in `fastlio_lc_pgo`'s `fastlio_lc_l2.launch.py`.
After a recalibration, update them too; `test_rig_consistency.py` fails until
they agree.

## 📁 Package Structure

```
pepper_slam/
├── config/
│   └── pepper_slam_configuration.yaml  # every node's parameters, one section per node
├── data/
│   └── record_qos.yaml                 # /tf_static QoS override for ros2 bag record
├── launch/
│   ├── pepper_sensor_tf.launch.py      # static rig TF, nested by everything else
│   ├── fastlio_odometry.launch.py      # FAST-LIO odometry + bridge
│   ├── pointlio_odometry.launch.py     # same, for Point-LIO
│   ├── lio_odom_bridge.launch.py       # the bridge alone, for an estimator started elsewhere
│   ├── fastlio_localization.launch.py  # standalone prior-map localization
│   ├── pointlio_localization.launch.py # same, for Point-LIO
│   ├── view_rig.launch.py              # sensor rig visualization from the URDF
│   └── bag_test/                       # bag-replay wrappers (see Bag replay)
├── rviz/                               # configs for the mapping, replay and rig-view launches
├── src/                                # one executable per node
│   ├── cloud_range_filter.cpp          # range / outlier / ground filters; feeds Nav2's /points_safety and /points_costmap, and octomap's grid while mapping
│   ├── lio_odom_bridge.cpp             # odom -> base_footprint from the LIO pose; levels lio_init once
│   ├── pgo_map_odom_bridge.cpp         # map -> odom from a pose-graph-optimized trajectory
│   ├── wheel_odom_tf.cpp               # odom -> base_footprint from /pepper_odom, for the loc stacks
│   └── lib/                            # pepper_slam_lib: ROS-free code the nodes share (headers in include/pepper_slam/)
│       ├── cloud_filters.cpp           # cloud_range_filter's filters
│       ├── lio_odom_guard.cpp          # lio_odom_bridge's divergence gate
│       └── transform_utils.cpp         # pose conversions and levelling shared by the two bridges
├── test/                               # pytest suites, resolve_launch.py and its baselines
├── urdf/                               # the L2 + D435i rig xacro (the calibration), alone and on Pepper's body
├── FRAMES.md                           # the frame contract in full
├── CMakeLists.txt
├── package.xml
└── README.md
```

Saved maps live in `pepper_navigation/`, which uses them.

## 🧪 Testing

```bash
cd ~/ros2_ws
colcon test --packages-select pepper_slam
colcon test-result --verbose
```

Besides the linters, this runs `test_leveling_frame_contract.py` (skipped when
the LIO packages are not built), `test_lio_odom_guard` (the divergence guard),
`test_cloud_filters` (the `cloud_range_filter` filters on synthetic clouds),
`test_rig_consistency.py` (the rig URDF's `scope` split, and the values derived
from it in the estimator configs and `fastlio_lc_pgo`; those cases skip when the
packages are not in the workspace) and `test_frame_contract_bag.py` (the frame contract on the
benchmark bag through FAST-LIO and the bridge; skipped without the bag or
`fast_lio`).

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
