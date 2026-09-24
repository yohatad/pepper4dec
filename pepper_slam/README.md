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

## ✅ Prerequisites

- **ROS 2 Humble**
- **`fast_lio`** and/or **`point_lio`** built in the same workspace
- **`rtabmap_ros`** for the RTAB-Map mapping and localization paths
- **Unitree L2 lidar** via `l2lidar_node`, and the RealSense via
  `realsense2_camera`. Neither is started by this package:
  `ros2 launch dec_launch dec_robot.launch.py` brings both up

## 🚀 Running

```bash
source ~/ros2_ws/install/setup.bash
```

### Sensor rig TF

Every launch file below nests `pepper_sensor_tf.launch.py`, which publishes
the static `base_footprint -> l2lidar_frame -> camera` chain. Run it alone
only when using the drivers by themselves (bag recording, a raw look at
`/points`), since nothing else relates the two sensors then.

```bash
ros2 launch pepper_slam pepper_sensor_tf.launch.py
```

| `scope` | Publishes | Use for |
|---|---|---|
| `mount` (default) | the rig mount edges only | the real robot, and any bag that recorded `/tf_static` |
| `all` | the mount edges plus the RealSense-internal edges | older bags recorded with an empty `/tf_static` |

`mount` is the default because `realsense2_camera` publishes its own internal
extrinsics from the device. Publishing them here too gives those latched
`/tf_static` edges two publishers, and whichever lands last silently wins.

### Odometry

```bash
ros2 launch pepper_slam fastlio_odometry.launch.py      # FAST-LIO
ros2 launch pepper_slam pointlio_odometry.launch.py     # Point-LIO
```

Both start the estimator plus `lio_odom_bridge`, which publishes
`odom -> base_footprint`. Do not run `fast_lio`'s own `mapping.launch.py`
directly on this robot: it does not include the rig TF, so the bridge waits
forever for `base_footprint -> l2lidar_frame_imu` and the stack hangs with no
error.

This is odometry, not SLAM. There is no loop closure, so revisiting a place
after drift lays the same wall down twice; use it for measuring odometry, and
one of the mapping paths below for a map worth keeping.

Two arguments matter:

- `bridge_level_frame` (default `true`): the bridge publishes a one-time
  gravity-leveled `odom` above the estimator's tilted `lio_init` frame. Set it
  `false` whenever a higher layer (RTAB-Map, PGO, AMCL) owns `odom`'s parent,
  or `odom` gets two parents and the TF tree breaks.
- `guard_enable` (default `false`): turns on the divergence guard inside the
  bridge (`lio_odom_guard.py`, imported by `lio_odom_bridge.py`, not a
  separate node). It rejects any pose implying motion the base physically cannot
  perform, carries the pose forward on wheel odometry while rejecting, and
  escalates to a fault after `max_hold_duration`. It publishes
  `/localization/wheel_trust_scale` and a `/diagnostics` status. The script's
  docstring documents the thresholds and the reasoning.

`pointlio_odometry.launch.py` reaches the bridge through `point_lio`'s own
launch file, which does not forward the guard arguments; set them on the node
directly there.

### Mapping

The validated mapping configuration is FAST-LIO odometry with RTAB-Map loop
closure, run from a bag:

```bash
ros2 launch pepper_slam bag_test/rtabmap_fastlio_bag.launch.py
ros2 launch pepper_slam bag_test/rtabmap_fastlio_bag.launch.py rgbd:=true   # colour the map from the depth camera
```

Keep `rgbd:=false` for maps that Nav2 will consume: with it on, camera points
are also folded into the 2D grid. `rtabmap_base.launch.py` is the underlying
RTAB-Map bringup (`localization:=true` to stop mapping, `rviz:=true` for
RViz). Each `bag_test/` launch file's docstring carries its playback command;
`bag_test/README.md` compares them.

### Localization against a prior map

Standalone live entry points, for testing the localizer outside the full Nav2
bringup. In normal use `pepper_navigation`'s `pepper_nav2_fastloc` and
`pepper_nav2_pointloc` profiles start these for you.

```bash
ros2 launch pepper_slam fastlio_localization.launch.py
ros2 launch pepper_slam pointlio_localization.launch.py
```

Both default `use_sim_time:=false` (the upstream launch files default it
`true`, being bag-oriented) and include the rig TF, which the upstream files
do not. Pass `use_sim_time:=true` only when replaying a bag with `--clock`.

### Recording bags

**Record `/tf_static` with the transient-local QoS override, or you will lose
it.**

```bash
ros2 bag record -a \
  --qos-profile-overrides-path $(ros2 pkg prefix pepper_slam)/share/pepper_slam/config/record_qos.yaml \
  -o my_bag
```

`/tf_static` is published once and latched; `ros2 bag record` subscribes
volatile by default and only receives messages sent after it starts, so
without the override, capturing the transforms is a race against your launch
files. A bag recorded with the override needs no `pepper_sensor_tf.launch.py`
on replay. `-a` records everything including 30 Hz colour; name topics
explicitly for long sessions.

## 🧭 Frames

| Frame | Published by | Notes |
|---|---|---|
| `pepper_odom` | `naoqi_driver2` | wheel odometry; deliberately **not** named `odom` |
| `lio_init` | FAST-LIO / Point-LIO | the estimator's own world frame, tilted by the sensor mount; not gravity-aligned |
| `odom` | `lio_odom_bridge` | one-time gravity-leveled parent of `lio_init`; Z-up |
| `map` | RTAB-Map or the PGO bridge | whichever layer owns the loop-closure correction |
| `map` | `fastlio_localization` / `pointlio_localization` | prior-map localization publishes `map -> base_footprint` directly, with no `odom` edge |

Only one node may publish a given frame's parent; that constraint is what
`bridge_level_frame` and the launch-file split above are managing. RTAB-Map
must anchor on `odom`, not `lio_init`: projecting a 2D grid out of a tilted
frame silently produces a wrong ground/obstacle split. `FRAMES.md` has the
full frame contract, and `pepper_navigation/config/README.md` the odometry
naming rules.

## 📁 Package Structure

```
pepper_slam/
├── config/
│   ├── record_qos.yaml                 # /tf_static QoS override for ros2 bag record
│   ├── sensor_tf.yaml                  # rig transform values; provenance in the header
│   └── mapper_params_online_async.yaml # SLAM Toolbox params (retired with it)
├── launch/
│   ├── pepper_sensor_tf.launch.py      # static rig TF, nested by everything else
│   ├── fastlio_odometry.launch.py      # FAST-LIO odometry + bridge (guard_enable, bridge_level_frame)
│   ├── pointlio_odometry.launch.py     # same, for Point-LIO
│   ├── lio_odom_bridge.launch.py       # the bridge alone, for an estimator started elsewhere
│   ├── fastlio_localization.launch.py  # standalone live prior-map localization
│   ├── pointlio_localization.launch.py # same, for Point-LIO
│   ├── rtabmap_base.launch.py          # RTAB-Map bringup (vendored upstream; excluded from flake8)
│   ├── slam_toolbox.launch.py          # 2D SLAM Toolbox; retired, kept for old 2D bags
│   ├── view_rig.launch.py              # sensor rig visualization from the URDF
│   └── bag_test/                       # bag-replay bringups, one per odometry/mapping combination
│       ├── rtabmap_fastlio_bag.launch.py   # the validated mapping configuration
│       ├── rtabmap_fused_bag.launch.py
│       ├── rtabmap_l2_bag.launch.py
│       ├── rtabmap_rgbd_wheel_bag.launch.py
│       ├── fastlio_odometry_bag.launch.py
│       ├── fastlio_lc_bag.launch.py
│       ├── pointlio_odometry_bag.launch.py
│       ├── pointlio_lc_bag.launch.py
│       ├── odom_compare_bag.launch.py
│       └── README.md                   # what each one is for
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
├── test/
│   ├── test_lio_odom_guard.py
│   ├── test_leveling_frame_contract.py
│   ├── resolve_launch.py               # resolves a launch file to its node list
│   ├── launch_baselines/               # expected node lists for the LIO launches
│   └── README.md
├── urdf/
│   ├── sensor_rig.xacro                # the L2 + D435i rig as one xacro macro
│   ├── pepper_sensor_rig.urdf.xacro    # rig only, rooted for the real robot
│   └── pepper_display_with_rig.urdf.xacro # display only: Pepper body + rig
├── FRAMES.md                           # the frame contract in full
├── CMakeLists.txt
├── package.xml
└── README.md
```

Saved 2D grids and the prior-map keyframes live in `pepper_navigation/`,
because Nav2 is their runtime consumer. RTAB-Map `.db` files go to `~/.ros/`
and are not version-controlled.

## 📜 License
Copyright (C) 2025 Carnegie Mellon University Africa
Licensed under the BSD-3-Clause License. See individual package licenses for details.
