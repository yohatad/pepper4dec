<div align="center">
<h1>Pepper SLAM and Mapping</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Pepper SLAM** package builds and localizes against maps of the Pepper
robot's environment. It ships launch files and parameters only — every SLAM
component (RTAB-Map, SLAM Toolbox, FAST-LIO) is an upstream package launched by
name, so this package compiles no nodes of its own.

It was split out of `pepper_navigation` so that mapping and bag-replay
experiments no longer drag in the whole Nav2 dependency tree, and so the
frequently-retuned SLAM configs are isolated from the comparatively stable
navigation params.

## Relationship to `pepper_navigation`

The dependency is **one-way**:

```
pepper_slam  ──produces──>  map frame + /map  ──consumed by──>  pepper_navigation
```

`pepper_navigation` `exec_depend`s on `pepper_slam` (its
`pepper_nav2_bringup.launch.py` includes `rtabmap_realsense.launch.py` in
localization mode). `pepper_slam` never depends on `pepper_navigation`.

Saved maps (`.pgm`/`.yaml`) and keepout masks stay in `pepper_navigation/map/`,
because Nav2's map server and costmap filters are their runtime consumers.
RTAB-Map databases (`.db`) live in `~/.ros/` and are not version-controlled.

## The frame contract

This is the interface between the two packages, and the thing most likely to
bite you:

| Frame | Published by | Notes |
|-------|--------------|-------|
| `pepper_odom` | `naoqi_driver2` | wheel odometry; **not** named `odom` on purpose |
| `odom` | FAST-LIO | IMU-aligned, tilted ~90° on Pepper's mount — **not** gravity-aligned |
| `odom_level` | `lio_map_odom_bridge` | one-time gravity-leveled parent of `odom`; Z-up |
| `map` | RTAB-Map | anchored on `odom_level`, which the 2D occupancy projection requires |

RTAB-Map must anchor on `odom_level`, not `odom` — projecting a 2D occupancy
grid out of a tilted frame silently produces garbage ground/obstacle splits.
See `config/README.md` in `pepper_navigation` for the odometry naming rules.

## 🚀 Running

```bash
source ~/ros2_ws/install/setup.bash
```

### RTAB-Map SLAM (3D mapping with RealSense)

```bash
ros2 launch pepper_slam rtabmap_realsense.launch.py

# Localization only (no new mapping)
ros2 launch pepper_slam rtabmap_realsense.launch.py localization:=true

# With RViz
ros2 launch pepper_slam rtabmap_realsense.launch.py rviz:=true
```

### SLAM Toolbox (2D mapping with LiDAR)

```bash
ros2 launch pepper_slam slam_toolbox.launch.py
```

### Bag-replay experiments

Each of these wraps `rtabmap_realsense.launch.py` (unchanged) with
bag-specific overrides and writes to a throwaway database, so recorded maps are
never at risk. Playback commands are documented in each file's header.

| Launch file | Sensor setup | Odometry source |
|-------------|--------------|-----------------|
| `rtabmap_bag_test.launch.py` | RealSense RGB-D (infra1 + depth) | bag TF (`pepper_odom`) |
| `rtabmap_l2_bag_test.launch.py` | Unitree L2 lidar + IMU, no camera | RTAB-Map `icp_odometry` |
| `rtabmap_fastlio_bag_test.launch.py` | L2 lidar + RGB for loop closure | FAST-LIO (best measured: 0.19 m closure) |

`rtabmap_fastlio_bag_test.launch.py` is the validated mapping configuration —
`pepper_nav2_bringup.launch.py` reuses its exact ICP/grid tuning for
localization.

## 📁 Package Structure

Launch and params only (`ament_cmake`, no compiled targets):

```
pepper_slam/
├── config/
│   └── mapper_params_online_async.yaml   # SLAM Toolbox parameters
├── launch/
│   ├── rtabmap_realsense.launch.py       # vendored from upstream rtabmap_ros; excluded from flake8
│   ├── slam_toolbox.launch.py
│   ├── rtabmap_bag_test.launch.py
│   ├── rtabmap_l2_bag_test.launch.py
│   └── rtabmap_fastlio_bag_test.launch.py
├── rviz/
│   └── rtabmap_fastlio_mapping.rviz
├── ament_flake8.ini
├── package.xml
└── README.md
```

## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
