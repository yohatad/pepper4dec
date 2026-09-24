<div align="center">
<h1>Pepper Navigation</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

Nav2 bringup for the Pepper robot: path planning, obstacle avoidance from 3D
point clouds (the Unitree L2 and the bottom RealSense), an independent
collision-monitor safety layer, and the saved maps those consume. Localization
is a launch-time profile; the default is `fastlio_localization`, which
registers the live scan against a prior 3D map inside FAST-LIO's filter.

## ✅ Prerequisites

- **ROS 2 Humble**
- **`pepper_slam`** built in the same workspace (supplies the sensor-rig static
  TF, and `rtabmap_base.launch.py` for the RTAB-Map profile)
- **`fast_lio`** built in the same workspace (odometry for every profile, and
  the `fastlio_localization` node the default profile runs)
- **Unitree L2 lidar** publishing `/points` and `/imu/data` via `l2lidar_node`.
  No launch file here starts it, and nothing works without it
- **`naoqi_driver2`** on the robot, if it should actually move: it consumes the
  `/cmd_vel` this package produces

## 🛠️ Installation

```bash
sudo apt install \
  ros-humble-nav2-bringup \
  ros-humble-nav2-collision-monitor \
  ros-humble-pointcloud-to-laserscan

cd ~/ros2_ws
colcon build --packages-select pepper_slam pepper_navigation
source install/setup.bash
```

`nav2-collision-monitor` is the safety layer every profile routes `cmd_vel`
through. `pointcloud-to-laserscan` is only needed by the AMCL profile, which
flattens the 3D cloud into the `/scan` AMCL requires.

## 🚀 Running

### 1. Start the sensors

No launch file in this package starts a driver.

```bash
source ~/ros2_ws/install/setup.bash

# L2 lidar + bottom RealSense together (each has an enable_* argument).
# Does not start naoqi_driver.
ros2 launch dec_launch dec_robot.launch.py

# The robot itself, if it should drive
ros2 launch naoqi_driver pepper_bringup.launch.py nao_ip:=<robot-ip>
```

The RealSense is required for the default fastloc profile: it is the **only**
obstacle source of that profile's local costmap (see [Costmaps](#-costmaps)).
It is also required for RTAB-Map, which subscribes to RGB. On pointloc and
AMCL it is a second, dense forward source next to the L2 and the local costmap
still works without it.

### 2. Bring up navigation

The default profile:

```bash
ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py
```

`fastlio_localization` finds its own initial pose with ScanContext, so no
`/initialpose` is needed. It requires two maps of the same environment: the
per-keyframe clouds and poses it registers against (`map_scan_dir`,
`map_pose_file`), and the 2D grid `map_server` publishes for the global
costmap (`map`). All three default to the files shipped in this package, with
one catch:

> `map_scan_dir` defaults to `pcd/sc_pcd_20260823/`. Those clouds are
> gitignored (75 MB) and **are not in a fresh checkout**; copy them in first.
> Without them `fastlio_localization` fails `on_configure`, nothing publishes
> `map -> base_footprint`, and the whole Nav2 bringup sits inactive with no
> other symptom.

Tuning of the localizer itself (overlap thresholds, motion gating, lock
verification) belongs to `fast_lio`; see that package.

Every profile opens RViz with a matching config (`rviz:=false` for headless)
and accepts `use_sim_time:=true` for bag replay.

### Other profiles

All profiles share the same global costmap, controller tuning and safety
layer, so switching profile mostly changes what corrects the odometry drift.
The exception is the local costmap: fastloc feeds it from the RealSense only,
the others from the L2 and the RealSense (see [Costmaps](#-costmaps)).

| Launch file | Localization | Needs | Initial pose |
|---|---|---|---|
| `pepper_nav2_fastloc.launch.py` *(default)* | `fastlio_localization`: prior map inside FAST-LIO's iEKF | keyframe clouds + `pose.json`, plus a 2D grid | automatic (ScanContext) |
| `pepper_nav2_pointloc.launch.py` | `pointlio_localization`: the Point-LIO equivalent | same as fastloc | automatic |
| `pepper_nav2_rtabmap_loc.launch.py` | RTAB-Map localization mode | an RTAB-Map `.db` via `database_path:=`, and RGB | RViz **2D Pose Estimate** |
| `pepper_nav2_amcl.launch.py` | AMCL over a flattened `/scan`, on FAST-LIO odom | a 2D grid via `map:=` | RViz **2D Pose Estimate** |
| `pepper_navigation.launch.py` | legacy: AMCL on raw wheel odometry | a 2D grid; expects a `/scan` the current rig does not publish | RViz **2D Pose Estimate** |

For the AMCL profile, if the particle cloud never tightens, the flattening
height band is the first thing to adjust:

```bash
ros2 launch pepper_navigation pepper_nav2_amcl.launch.py \
    scan_min_height:=0.30 scan_max_height:=1.20
```

### Bag replay

```bash
ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py use_sim_time:=true
ros2 bag play <bag> --clock --topics /points /imu/data
```

Nav2 will localize and build costmaps, but a bag cannot react to `cmd_vel`;
driving to a goal needs the real robot.

### Tuning the costmaps on a bag

`launch/bag_test/costmap_bag.launch.py` runs only the two costmaps (no
controller goal, no robot) plus tools that show why each point is or is not
marked. Its docstring lists the exact `ros2 bag play` flags per bag type.

```bash
# Bag without a recorded pose: FAST-LIO localizes it (the defaults)
ros2 launch pepper_navigation costmap_bag.launch.py

# Bag that already carries map -> base_footprint
ros2 launch pepper_navigation costmap_bag.launch.py localize:=false depth_images:=false

# A/B test against a copy of the params
ros2 launch pepper_navigation costmap_bag.launch.py params_file:=/path/to/copy.yaml
```

| Argument | Default | Meaning |
|---|---|---|
| `params_file` | `config/nav2_params_fastloc.yaml` | Nav2 params the costmaps load |
| `localize` | `true` | Run FAST-LIO localization; `false` if the bag has the pose |
| `depth_images` | `true` | Rebuild the RealSense cloud from depth + colour images (older bags) |
| `sensor_tf` | `urdf` | Sensor rig TF: `urdf`, `yaml`, or `none` if the bag has `/tf_static` |
| `rviz` | `true` | Open `rviz/costmap_bag_test.rviz` |

With `localize:=true`, nothing is drawn until FAST-LIO reports a lock (about
25 s of replay). `config/nav2_params_l2voxel_test.yaml` is a ready-made
`params_file` that puts the L2 back into the local costmap as a voxel layer.

To see what the collision monitor would count, without Nav2 running:

```bash
python3 utils/collision_zone_viewer.py
rviz2 -d utils/collision_zone_viewer.rviz
```

### Watching from a laptop

`rviz_remote.launch.py` starts RViz alone with a config that subscribes to no
point clouds, camera or prior map, for watching a stack that runs on the
robot over WiFi. Its docstring covers the CycloneDDS discovery setup that has
to be right for it to see anything.

## 🎯 Sending goals

In RViz, use the **Nav2 Goal** tool. From the command line:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 1.0}, orientation: {w: 1.0}}}}"
```

`ros2 run pepper_navigation send_goal` sends one fixed goal from
`src/tools/send_goal.cpp`; edit the coordinates there and rebuild.

## 📁 Package Structure

```
pepper_navigation/
├── config/
│   ├── nav2_params_fastloc.yaml        # Nav2 params, fastloc profile (default)
│   ├── nav2_params_pointloc.yaml       # Nav2 params, pointloc profile
│   ├── nav2_params_rtabmap_loc.yaml    # Nav2 params, RTAB-Map profile
│   ├── nav2_params_amcl.yaml           # Nav2 params, AMCL profile
│   ├── nav2_params_l2voxel_test.yaml   # fastloc with the L2 as a local voxel layer; bag tests only
│   ├── nav2_params.yaml                # Nav2 params, legacy wheel-odometry bringup
│   ├── nav2_params_wheel_odom.yaml     # not launched by anything
│   ├── ekf_nav.yaml                    # not launched by anything
│   └── README.md                       # odometry naming rules (pepper_odom vs odom)
├── launch/
│   ├── pepper_nav2_fastloc.launch.py   # Nav2 + fastlio_localization (default)
│   ├── pepper_nav2_pointloc.launch.py  # Nav2 + pointlio_localization
│   ├── pepper_nav2_rtabmap_loc.launch.py # Nav2 + RTAB-Map localization
│   ├── pepper_nav2_amcl.launch.py      # Nav2 + AMCL on FAST-LIO odom
│   ├── pepper_navigation.launch.py     # legacy: AMCL on raw wheel odometry
│   ├── rviz_remote.launch.py           # RViz only, for a laptop watching the robot
│   ├── odom_test.launch.py             # odometry-only bringup with a path publisher
│   └── bag_test/
│       └── costmap_bag.launch.py       # the two costmaps on a recorded bag, for tuning
├── map/
│   ├── pepper_map_lc.yaml / .pgm       # the 2D grid every profile's map_server defaults to
│   ├── keepout_zone.yaml / .pgm        # keepout mask, legacy only; regenerate before reuse
│   └── *.png                           # map preview renders
├── pcd/
│   ├── sc_pose_20260823.json           # per-keyframe poses for fastloc / pointloc
│   ├── pepper_map_lc_poses.txt         # the PGO run's trajectory
│   └── sc_pcd_20260823/                # matching keyframe clouds; gitignored, copy in
├── rviz/
│   ├── costmap_bag_test.rviz           # for costmap_bag.launch.py
│   ├── nav2_fastloc.rviz               # one config per profile ...
│   ├── nav2_amcl.rviz
│   ├── nav2_*_voxel.rviz               # ... voxel-layer variants
│   ├── nav2_fastloc_remote.rviz        # ... and the light remote view
│   ├── nav2_remote_light.rviz
│   └── odometry_test.rviz
├── scripts/
│   ├── wait_for_map_then_start.py      # starts Nav2 once map -> base_footprint exists
│   ├── localization_watchdog.py        # cancels goals while the localizer reports lost
│   ├── localization_recovery.py        # /localization_recover, dispatched per backend
│   ├── tf_nav_relay.py                 # /tf without Pepper's joint tree, on /tf_nav
│   ├── cloud_delay.py                  # bag_test: delays clouds so the pose arrives first
│   ├── costmap_explain.py              # bag_test: colours points by what the costmap does
│   ├── depth_to_cloud.py               # bag_test: RealSense cloud from depth images
│   └── points_to_image.py              # bag_test: camera picture from the cloud
├── utils/
│   └── collision_zone_viewer.py/.rviz  # what the collision monitor counts; run by hand
├── src/tools/
│   ├── send_goal.cpp                   # one-shot goal sender
│   └── odom_path_publisher.cpp         # traversed path for RViz
├── test/
│   └── test_shared_nav2_params.py      # guards the params blocks shared across profiles
├── CMakeLists.txt
├── package.xml
└── README.md
```

Maps live here rather than in `pepper_slam` because Nav2's `map_server` is
their runtime consumer. Mapping itself is `pepper_slam`'s job.

## 🧭 Frames

Who publishes what, per profile. The local costmap rolls in the odometry frame
and the global costmap lives in `map`:

| Profile | `map -> odom` | `odom -> base_footprint` | Local costmap frame |
|---|---|---|---|
| fastloc / pointloc | the localizer publishes `map -> base_footprint` directly; there is no `odom` edge | | `map` |
| rtabmap_loc | `rtabmap` | `lio_odom_bridge` (FAST-LIO) | `odom` |
| amcl | `amcl` | `lio_odom_bridge` (FAST-LIO) | `odom` |
| legacy | `amcl` | `naoqi_driver2` (`pepper_odom`) | `pepper_odom` |

The static sensor chain (`base_footprint -> l2lidar_frame -> cameras`) comes
from `pepper_slam`'s `pepper_sensor_tf.launch.py`, which every profile nests.
**Exactly one node may own `map -> odom`**: the AMCL and RTAB-Map profiles run
FAST-LIO with `bridge_level_frame:=false` so the bridge's own `odom -> odom`
edge does not give `odom` a second parent. `config/README.md` explains why
wheel odometry is named `pepper_odom` and never plain `odom`.

### `/tf_nav`

`naoqi_driver2` publishes Pepper's whole joint tree on `/tf` (about 4000
transforms/s), which saturates the costmaps' TF listener until the local
costmap stops updating. Every profile therefore runs `tf_nav_relay`, which
republishes on `/tf_nav` only the transforms whose parent is `map`, `odom`,
`lio_init` or `base_footprint`. `controller_server`, `planner_server`,
`behavior_server`, `bt_navigator` and `collision_monitor` remap `/tf` to
`/tf_nav`. Nodes that broadcast TF (the localizers, `amcl`, `rtabmap`) are not
remapped, `/tf_static` is untouched, and RViz keeps the full `/tf`.

## 🧱 Costmaps

| Costmap | Profile | Layers | Obstacle sources |
|---|---|---|---|
| Global (`map`) | all | static, voxel, inflation | L2 |
| Local (rolling, 10 x 10 m) | fastloc | voxel, inflation | RealSense only |
| Local | pointloc, amcl, rtabmap_loc | voxel, inflation | L2 and RealSense |

The L2 source is `/points`, except on AMCL, where both costmaps read
`/points_costmap`: the L2 cloud with the near range and the ground plane
removed by a second `cloud_range_filter` (`points_costmap_filter`).

On fastloc the L2 was taken out of the local costmap because it marked
reflections on the floor as obstacles. The local costmap is therefore blind
outside the camera cone (about +/-29 deg, from about 0.35 m). The L2 still
feeds the global costmap and the collision monitor, so a close obstacle
anywhere around the robot still stops it. The L2 layer is kept in
`nav2_params_fastloc.yaml` as `l2_voxel`, commented out of `plugins`, together
with the plugin lines to switch it back on.

The local voxel grid is at most 16 cells tall (Nav2 packs a column into 32
bits), so `z_voxels` above 16 has no effect.

## 🖥️ ROS interface

### Subscribed

| Topic | Type | Notes |
|---|---|---|
| `/points` | `sensor_msgs/PointCloud2` | L2 cloud; 360° obstacle source for the global costmap, and for the local one except on fastloc |
| `/points_costmap` | `sensor_msgs/PointCloud2` | AMCL only: filtered L2 cloud the costmaps read instead of `/points` |
| `/camera/depth/color/points` | `sensor_msgs/PointCloud2` | RealSense depth; local costmap only, and its only source on fastloc |
| `/tf_nav` | `tf2_msgs/TFMessage` | Navigation edges of `/tf`, from `tf_nav_relay`; what the Nav2 nodes listen to |
| `/points_safety` | `sensor_msgs/PointCloud2` | L2 cloud with self-hits removed; the collision monitor's only input |
| `/scan` | `sensor_msgs/LaserScan` | Flattened L2 scan; AMCL profiles only |
| `/map` | `nav_msgs/OccupancyGrid` | From `map_server`, or from RTAB-Map on that profile |
| `/Odometry` | `nav_msgs/Odometry` | FAST-LIO odometry, read by `bt_navigator` |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | RViz 2D Pose Estimate; seeds AMCL and RTAB-Map |

### Published

| Topic | Type | Notes |
|---|---|---|
| `/cmd_vel_raw` | `geometry_msgs/Twist` | Controller output. **Not** what the robot drives on |
| `/cmd_vel` | `geometry_msgs/Twist` | Collision-monitor-gated velocity, consumed by `naoqi_driver2` |
| `/collision_monitor_state` | `nav2_msgs/CollisionMonitorState` | Active safety action: none, slowdown or stop |
| `/polygon_stop`, `/polygon_slowdown` | `geometry_msgs/PolygonStamped` | Safety zones, 0.40 m and 0.80 m |
| `/plan` | `nav_msgs/Path` | Current global path |
| `/global_costmap/costmap`, `/local_costmap/costmap` | `nav_msgs/OccupancyGrid` | Costmaps |
| `/particle_cloud` | `nav2_msgs/ParticleCloud` | AMCL profiles only |
| `/localization/overlap` | `std_msgs/Float32` | fastloc / pointloc only: fraction of the live scan landing on the prior map, 1 Hz |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | fastloc / pointloc only: pose-lock status |

### Services and actions

| Name | Type | Notes |
|---|---|---|
| `/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | Goal execution |
| `/localization_recover` | `std_srvs/Trigger` | "I am lost"; same call on every profile, dispatched to the running backend |
| `/relocalize` | `std_srvs/Trigger` | fastloc / pointloc: re-arm the ScanContext search from scratch |
| `/reinitialize_global_localization` | `std_srvs/Empty` | AMCL: re-scatter the particles |

## 🛡️ Safety layer

Nothing reaches the wheels unvetted:

```
controller_server ─┐
                   ├─> /cmd_vel_raw ──> collision_monitor ──> /cmd_vel ──> naoqi_driver2
behavior_server  ──┘                          ^
                                     /points_safety
```

`collision_monitor` slows to 30 % inside 0.80 m and stops inside 0.40 m,
independently of the costmaps and the planner. It triggers on more than 3
points inside a zone. It reads `/points_safety` rather than `/points`: the L2
cloud passed through `pepper_slam`'s `cloud_range_filter.py`
(`points_safety_filter`), started by every profile.

Two cuts decide which points count, and both are set in the launch files and
param files of every profile:

| Cut | Where | Value | Removes |
|---|---|---|---|
| `min_range` | `points_safety_filter` | 0.22 m from the lidar | the L2's own housing |
| `min_height` / `max_height` | `collision_monitor` | 0.22 m to 1.80 m | floor returns near the robot |

`utils/collision_zone_viewer.py` applies the same two cuts live, so they can be
tuned without running Nav2 (see [Tuning the costmaps on a bag](#tuning-the-costmaps-on-a-bag)).

## 🔁 Losing localization

fastloc and pointloc re-score the live scan against the prior map at 1 Hz,
because a wrong-but-confident lock has no other symptom. Sustained low overlap
re-arms the search automatically, and `localization_watchdog` cancels the
active goal for as long as it lasts. Only sustained badness counts; a single
low reading is normal when turning into unmapped space or when someone walks
through the scan. Pass `watchdog:=false` to monitor without holding
navigation.

```bash
ros2 service call /localization_recover std_srvs/srv/Trigger   # any profile
ros2 topic echo /localization/overlap                          # fastloc health
```

## 🗺️ Saving a new map

With a `pepper_slam` mapping session running:

```bash
ros2 run nav2_map_server map_saver_cli -f ~/maps/my_new_map
ros2 launch pepper_navigation pepper_nav2_amcl.launch.py map:=~/maps/my_new_map.yaml
```

The profiles take an absolute `map:=` path, so a map does not have to live in
the package. With `colcon build --symlink-install`, every file under `config/`,
`launch/`, `rviz/`, `map/` and `pcd/` is installed as a symlink, so edits to
existing files apply without a rebuild. A **new** file (a new map, a new prior
map) still needs one rebuild to be installed.

## 🔧 Troubleshooting

```bash
ros2 lifecycle get /controller_server      # every Nav2 node should be 'active'
ros2 topic hz /points /imu/data            # both must flow before anything else works
ros2 run tf2_ros tf2_echo map base_footprint
ros2 topic echo /collision_monitor_state   # 0 = clear, otherwise slowdown/stop
ros2 topic hz /cmd_vel_raw /cmd_vel        # what the planner asked vs what the robot got
```

| Symptom | Likely cause |
|---|---|
| Nav2 never becomes active, no error | fastloc: keyframe clouds missing, so no lock, so `wait_for_map_then_start` never fires. Check `pcd/sc_pcd_20260823/` exists |
| Costmaps empty, no plan | `/points` not flowing; the lidar driver is not running |
| fastloc: local costmap empty, obstacles ignored | The RealSense is not running: `/camera/depth/color/points` is the local costmap's only source |
| Local costmap stops updating after a while | The Nav2 nodes are on the full `/tf`: check `tf_nav_relay` is running and `ros2 topic hz /tf_nav` flows |
| Robot freezes and never moves | Collision monitor stopping on self-hits: check `/points_safety` exists and its filter is running |
| `map -> odom` jitter or TF warnings | Two publishers of one transform: check `bridge_level_frame` for the profile you are running |
| AMCL particles never tighten | Flattened `/scan` does not match the grid: retune `scan_min_height` / `scan_max_height` |
| Global costmap all unknown | `/map` never arrived: wrong `map:=` path, or `map_server` never activated |

## 💡 Support

- Issues: [pepper4dec on GitHub](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2025 Carnegie Mellon University Africa
Licensed under the BSD-3-Clause License. See individual package licenses for details.
