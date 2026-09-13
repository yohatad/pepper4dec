<div align="center">
<h1>Pepper Navigation</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Pepper Navigation** package provides autonomous navigation for the Pepper
robot: Nav2 path planning, obstacle avoidance, goal execution, keepout zones,
and the saved maps those consume.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Nav2 Stack**: Full autonomous navigation with path planning and obstacle avoidance
- **Three interchangeable localization stacks**: AMCL, a prior map carried
  inside FAST-LIO's filter (`fastlio_localization`), or RTAB-Map — same costmaps
  and tuning behind each, so they can be compared directly
- **3D obstacle avoidance**: costmaps consume the L2's 360° `PointCloud2` directly
  through voxel layers, with no flattening step
- **Independent safety layer**: a collision monitor gates every velocity command
  straight off the lidar, bypassing the costmaps
- **Keepout Zones**: Configurable restricted areas using costmap filter masks
- **Goal Navigation API**: C++ utility for programmatic navigation goal sending
- **Pre-built Maps**: Saved maps and filter masks for localization-only deployments

## ✅ Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **Pepper robot** with ROS2 driver configured (`naoqi_driver2` — consumes the
  `/cmd_vel` this package produces)
- **`pepper_slam`** built in the same workspace (supplies `pepper_sensor_tf.launch.py`
  and, for the RTAB-Map stack, `rtabmap_base.launch.py`)
- **`fast_lio`** built in the same workspace — every current stack uses it for
  `odom → base_footprint`, and its `cloud_range_filter.py` feeds the collision monitor
- **Unitree L2 lidar** (`l2lidar_node`) publishing `/points` + `/imu/data` — no launch
  file here starts it, and nothing works without it

## 🛠️ Installation

### Required ROS2 Packages

```bash
sudo apt install \
  ros-humble-nav2-bringup \
  ros-humble-nav2-map-server \
  ros-humble-nav2-amcl \
  ros-humble-nav2-controller \
  ros-humble-nav2-planner \
  ros-humble-nav2-behaviors \
  ros-humble-nav2-bt-navigator \
  ros-humble-nav2-lifecycle-manager \
  ros-humble-nav2-costmap-2d \
  ros-humble-nav2-collision-monitor \
  ros-humble-pointcloud-to-laserscan
```

`nav2-collision-monitor` is the safety layer every current stack routes
`cmd_vel` through. `pointcloud-to-laserscan` is needed only by
`pepper_nav2_amcl.launch.py`, which flattens the L2's 3D cloud into the
LaserScan AMCL requires — it is *not* pulled in by `nav2-bringup`.

SLAM dependencies (`slam-toolbox`, `rtabmap-ros`, `realsense2-camera`) are
declared by `pepper_slam` — see that package's README.

### Package Installation

```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select pepper_slam pepper_navigation
source install/setup.bash
```

## 🔧 Configuration

### Nav2 Components

| Component | Description |
|-----------|-------------|
| `map_server` | Serves a saved 2D grid as `/map` (AMCL and fastloc stacks; RTAB-Map serves its own) |
| `amcl` | Adaptive Monte Carlo Localization — owns `map → odom` when used |
| `bt_navigator` | Behavior tree-based navigation, serves `/navigate_to_pose` |
| `controller_server` | Local path follower (DWB controller) |
| `planner_server` | Global path planner (NavFn) |
| `behavior_server` | Recovery behaviors (spin / backup / wait) |
| `collision_monitor` | Independent safety layer: gates `cmd_vel_raw → cmd_vel` off the raw lidar |
| `costmap_filter` | Keepout zone filter integration (legacy stack only — see the mask caveat below) |

### Parameter files

Each launch file has its own params file; everything except the localization
block is kept identical across the three current stacks, so a behavioural
difference between them is a *localization* difference and nothing else.

| Params file | Used by | Localization |
|-------------|---------|--------------|
| `nav2_params.yaml` | `pepper_navigation.launch.py` | AMCL on wheel odom (`pepper_odom`), legacy |
| `nav2_params_amcl.yaml` | `pepper_nav2_amcl.launch.py` | AMCL on FAST-LIO odom |
| `nav2_params_fastloc.yaml` | `pepper_nav2_fastloc.launch.py` | `fastlio_localization`: prior map inside the iEKF |
| `nav2_params_rtabmap_loc.yaml` | `pepper_nav2_rtabmap_loc.launch.py` | RTAB-Map localization mode vs a `.db` |
| `nav2_params_wheel_odom.yaml` | not launched by default | wheel-odometry variant, kept for comparison |

`test/test_shared_nav2_params.py` guards the three node blocks
(`behavior_server`, `controller_server`, `planner_server`) that are meant to be
byte-identical across all five, so tuning a gain in four files and forgetting
the fifth is caught rather than discovered later on the robot.

The **keepout filter is not carried into the three current stacks**:
`keepout_zone.yaml`'s mask was authored against an older map's
frame, so it has to be re-generated (`ros2_ws/utils/generate_keepout.py`) against the
current map before that layer can be re-enabled.

## 🚀 Running the Stack

### Pick a localization stack

All three current stacks share the same costmaps, DWB tuning, safety layer and
FAST-LIO dead reckoning. They differ only in what corrects the drift:

| Launch file | Localization | Prior map | Cost |
|-------------|--------------|-----------|------|
| `pepper_nav2_amcl.launch.py` | AMCL particle filter over a flattened `/scan` | 2D grid (`.yaml`/`.pgm`) | Cheapest; 2D only |
| `pepper_nav2_fastloc.launch.py` | `fastlio_localization`, prior map loaded into the iEKF's ikd-Tree | keyframe `pose.json` + clouds **+** a matching 2D grid | Light — no Open3D, no PGO at runtime |
| `pepper_nav2_rtabmap_loc.launch.py` | RTAB-Map localization mode (ICP + appearance) | RTAB-Map `.db` | Heaviest; also needs RGB |

`pepper_navigation.launch.py` is the legacy fourth path — AMCL on naoqi's wheel
odometry (`pepper_odom`) against `map/pepper_map_lc.yaml`, from the 2D-lidar
era. It expects a `/scan` that nothing in the current rig publishes, so start
from `pepper_nav2_amcl.launch.py` instead unless you specifically want it.

### Always start the sensors first

No launch file in this package starts a driver. Bring these up first, in this
order:

```bash
source ~/ros2_ws/install/setup.bash

# Both sensors at once (each has an enable_* argument). It publishes no
# TF of its own -- the stacks below nest the sensor-rig static TF. It does
# NOT start naoqi_driver; launch that separately (step 3 below):
ros2 launch dec_launch dec_robot.launch.py
```

or individually:

```bash
source ~/ros2_ws/install/setup.bash

# 1. L2 lidar -> /points + /imu/data. Everything downstream needs both.
ros2 launch l2lidar_node l2lidar.launch.py

# 2. RealSense. Optional for the AMCL/fastloc stacks (a second, dense forward
#    obstacle source); REQUIRED for the RTAB-Map stack, which subscribes to RGB.
ros2 launch dec_launch realsense_bottom.launch.py

# 3. The robot itself, if it should actually move: naoqi_driver2 consumes the
#    /cmd_vel the collision monitor emits.
ros2 launch naoqi_driver pepper_bringup.launch.py nao_ip:=<robot-ip>
```

Then bring up one of the stacks below. Each opens RViz with the right config
(`rviz:=false` to run headless). AMCL and RTAB-Map expect you to seed the pose
with RViz's **2D Pose Estimate**; fastloc does not — ScanContext finds its own
initial pose, and `/relocalize` re-arms that search if it is ever wrong.

### Option 1: AMCL on FAST-LIO odometry

```bash
ros2 launch pepper_navigation pepper_nav2_amcl.launch.py
```

The `map` argument defaults to the packaged `map/pepper_map_lc.yaml`; pass
`map:=<path to a .yaml>` to use a different grid.

AMCL corrects FAST-LIO's `odom` (**not** `pepper_odom`), and
`pointcloud_to_laserscan` flattens the L2's 360° `/points` into the `/scan` it
needs. Watch the green particle cloud in RViz tighten after you set the pose. If
it never converges, the flattening height band is the first knob — the default
0.20–1.50 m slice may be picking up furniture that isn't in the 2D grid:

```bash
ros2 launch pepper_navigation pepper_nav2_amcl.launch.py \
    scan_min_height:=0.30 scan_max_height:=1.20
```

### Option 2: FAST-LIO + fastlio_localization (prior keyframe map)

The lightest localization stack — the prior map is loaded straight into the
ikd-Tree the iEKF registers against, so the map constrains the estimate at scan
rate with no separate correction node, no Open3D and no PGO at runtime, which
matters on the Jetson CPU budget. ScanContext finds the initial pose, so no
`/initialpose` is needed:

```bash
ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py
```

It needs **both** maps of the same environment: the per-keyframe clouds and
poses that ScanContext and the iEKF register against (`map_scan_dir` +
`map_pose_file`), and the 2D grid `map_server` publishes for the global
costmap's static layer (`map`). If the L2 scan only partly overlaps the prior
map, lower `init_min_overlap` (default `0.70`) to accept a lock at startup, or
`health_min_overlap` (default `0.45`) to stop the post-lock health check
declaring itself lost mid-run.

A lock is accepted standing still. `init_require_motion` defaults to `false`
here, so ScanContext can converge from a parked start — which is what keeps a
stationary bag or an undriveable robot from stalling bringup entirely, since
nothing publishes `map → base_footprint` until the lock and
`wait_for_map_then_start` waits on exactly that.

The cost is that two estimates taken standing still can agree on the same wrong
place without ever being forced apart in space (measured 41 m off in a
corridor). Pass `init_require_motion:=true` to require ~0.5 m of driving
between them if you would rather fail to start than start in the wrong place —
worth doing for unattended deployment, where nobody is watching the first lock.
Either way the post-lock health check and `localization_watchdog` catch a wrong
lock afterwards; the argument only decides whether one is caught *before* it is
used.

> `map_scan_dir` defaults to `pcd/sc_pcd_20260823/` — the keyframe clouds
> `utils/pgo_to_scancontext_map.py` writes from a `fastlio_lc_pgo` run. They are
> gitignored (75 MB) and are **not in a fresh checkout**; copy them in first.
> Without them `fastlio_localization` fails `on_configure` and exits, and
> because nothing then publishes `map → base_footprint`,
> `wait_for_map_then_start` never fires and the whole Nav2 bringup sits inactive
> with no other symptom.

### Option 3: FAST-LIO + RTAB-Map localization (`.db`)

Reuses the exact odometry/appearance/ICP pipeline already tuned for mapping, and
publishes `/map` itself — so there is no `map_server` here:

```bash
ros2 launch pepper_navigation pepper_nav2_rtabmap_loc.launch.py \
    database_path:=~/.ros/rtabmap_fastlio_refined.db
```

### Bag replay

Every stack takes `use_sim_time:=true` for replaying a recording instead of
driving the robot. Nav2 will localize and build costmaps, but a bag cannot react
to `cmd_vel` — driving to a goal needs the real robot.

```bash
ros2 launch pepper_navigation pepper_nav2_amcl.launch.py use_sim_time:=true
ros2 bag play <bag> --clock --topics /points /imu/data
```

### Building a map first

Mapping is `pepper_slam`'s job:

```bash
ros2 launch pepper_slam rtabmap_base.launch.py        # RTAB-Map, RealSense
ros2 launch pepper_slam slam_toolbox.launch.py        # SLAM Toolbox, 2D LiDAR
```

## 🧭 Frames

Who publishes what, per stack. In every case the local costmap rolls in the
odometry frame and the global costmap lives in `map`:

| Stack | `map → odom` | `odom → base_footprint` | Local costmap frame |
|-------|--------------|-------------------------|---------------------|
| `pepper_nav2_amcl` | `amcl` | `lio_odom_bridge` (FAST-LIO) | `odom` |
| `pepper_nav2_fastloc` | `fastlio_localization` (publishes `map → base_footprint`; no `odom` edge) | — | `map` |
| `pepper_nav2_rtabmap_loc` | `rtabmap` (to `odom`) | `lio_odom_bridge` | `odom` |
| `pepper_navigation` (legacy) | `amcl` | `naoqi_driver2` (`pepper_odom`) | `pepper_odom` |

The static sensor chain (`base_footprint → l2lidar_frame → cameras`) comes from
`pepper_slam`'s `pepper_sensor_tf.launch.py`. **Exactly one node may own
`map → odom`** — this is why the AMCL and RTAB-Map stacks run FAST-LIO with
`bridge_level_frame:=false`: the bridge's static `odom → odom` would
otherwise give `odom` a second parent. See `config/README.md` on why wheel odom
is deliberately named `pepper_odom` and never plain `odom`.

## 🖥️ ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/points` | `sensor_msgs/PointCloud2` | L2 lidar cloud — primary 360° obstacle source for both costmaps, consumed directly (no flattening) |
| `/camera/depth/color/points` | `sensor_msgs/PointCloud2` | RealSense depth — complementary dense *forward* source on the local costmap only |
| `/points_safety` | `sensor_msgs/PointCloud2` | Self-hit-filtered L2 cloud (< 0.8 m removed), the collision monitor's only input |
| `/scan` | `sensor_msgs/LaserScan` | Flattened L2 scan — AMCL stacks only |
| `/map` | `nav_msgs/OccupancyGrid` | Occupancy grid, from `map_server` or RTAB-Map (`transient_local`) |
| `/Odometry` | `nav_msgs/Odometry` | FAST-LIO odometry, read by `bt_navigator` for velocity-based BT conditions |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | RViz *2D Pose Estimate* — seeds AMCL / the ICP global localization |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | Transform tree |

Camera RGB topics are subscribed by RTAB-Map (via `pepper_slam`) in that stack only.

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel_raw` | `geometry_msgs/Twist` | Controller/behavior output — **not** what the robot drives on |
| `/cmd_vel` | `geometry_msgs/Twist` | Collision-monitor-gated velocity, consumed by `naoqi_driver2` |
| `/collision_monitor_state` | `nav2_msgs/CollisionMonitorState` | Which safety action is active (none / slowdown / stop) |
| `/polygon_stop`, `/polygon_slowdown` | `geometry_msgs/PolygonStamped` | Safety zone visualization (0.40 m stop, 0.80 m slow) |
| `/particle_cloud` | `nav2_msgs/ParticleCloud` | AMCL particle filter state (AMCL stacks only) |
| `/map` | `nav_msgs/OccupancyGrid` | 2D occupancy grid map |
| `/global_costmap/costmap` | `nav_msgs/OccupancyGrid` | Global costmap |
| `/local_costmap/costmap` | `nav_msgs/OccupancyGrid` | Local costmap |
| `/plan` | `nav_msgs/Path` | Current planned global path |
| `/localization/overlap` | `std_msgs/Float32` | fastloc only: fraction of the live scan landing on the prior map, 1 Hz |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | fastloc only: `fastlio_localization: pose lock` status (OK / WARN / ERROR) |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/localization_recover` | `std_srvs/Trigger` | "I am lost, fix it", same call on every profile — dispatches to whichever backend is running |
| `/relocalize` | `std_srvs/Trigger` | fastloc only: re-arms the ScanContext search from scratch (what `/localization_recover` forwards to) |
| `/reinitialize_global_localization` | `std_srvs/Empty` | AMCL only: re-scatters particles globally |

### Losing localization

`fastlio_localization` re-checks its own lock at 1 Hz by scoring the live scan
against the prior map, because a wrong-but-confident lock has no other symptom —
a self-similar corridor still produces plausible scan matches at the wrong
place. Sustained low overlap re-arms its search automatically, and
`localization_watchdog` cancels the active navigation goal while it lasts, so
the robot stops driving toward a goal computed from a pose it no longer trusts.

Only sustained badness counts: a single low reading is normal when turning a
corner into unmapped space or when someone walks through the scan. Pass
`watchdog:=false` to monitor without ever holding navigation.

```bash
ros2 service call /localization_recover std_srvs/srv/Trigger   # any profile
ros2 topic echo /localization/overlap                          # fastloc health
```

### The velocity chain

Nothing reaches the wheels unvetted:

```
controller_server ─┐
                   ├─> /cmd_vel_raw ──> collision_monitor ──> /cmd_vel ──> naoqi_driver2
behavior_server  ──┘                          ↑
                                     /points_safety (L2, self-hits stripped)
```

`collision_monitor` slows to 30 % inside 0.80 m and hard-stops inside 0.40 m,
independent of the costmaps and the planner. It reads `/points_safety` rather
than `/points` because the low-mounted L2 sees Pepper's own body ~0.3–0.6 m out
and the robot would otherwise freeze on itself.

### Action Servers

| Action | Type | Description |
|--------|------|-------------|
| `/navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | Navigation goal execution |

## Sending Navigation Goals

### Using RViz2
Use the **Nav2 Goal** tool in RViz2 to click a target pose on the map.

### Using the send_goal Utility

```bash
ros2 run pepper_navigation send_goal
```

Edit `src/tools/send_goal.cpp` to change target coordinates (requires a rebuild after editing):

```cpp
goal.pose.pose.position.x = 2.0;   // Target X in map frame (meters)
goal.pose.pose.position.y = 1.0;   // Target Y in map frame (meters)
goal.pose.pose.orientation.w = 1.0; // Orientation (1.0 = facing forward)
```

### Using Nav2 Action CLI

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 1.0}, orientation: {w: 1.0}}}}"
```

## Pre-built Maps

Shipped in this package, so every stack but RTAB-Map comes up on a fresh
checkout with no absolute paths:

| Path | Used by |
|------|---------|
| `map/pepper_map_lc.yaml` (+ `.pgm`) | `map` arg of the AMCL and fastloc stacks, and the legacy stack's hardcoded map (2D grid) |
| `map/keepout_zone.yaml` (+ `.pgm`) | keepout filter mask, legacy stack only |
| `pcd/sc_pose_20260823.json` | `map_pose_file` of the fastloc stack — per-keyframe poses |
| `pcd/pepper_map_lc.pcd`, `pcd/pepper_map_lc_poses.txt` | the PGO run's 3D map product; not a runtime input any more |

Two things are **not** in the package:

| Path | Used by |
|------|---------|
| `pcd/sc_pcd_20260823/` (2735 clouds, 75 MB) | `map_scan_dir` of the fastloc stack — gitignored; copy it alongside a checkout |
| `~/.ros/rtabmap_fastlio_refined.db` | `database_path` arg of the RTAB-Map stack |

The keepout mask was authored against an older map's frame and does **not** line
up with `pepper_map_lc.yaml` — regenerate it with `utils/generate_keepout.py`
before re-enabling that layer.

### Saving a New Map

With a `pepper_slam` mapping session running:

```bash
# Into the package (versioned; needs a rebuild before Nav2 sees it)
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/src/pepper4dec/pepper_navigation/map/my_new_map

# Or alongside the other L2-era maps, and point the launch arg at it
ros2 run nav2_map_server map_saver_cli -f ~/maps/my_new_map
ros2 launch pepper_navigation pepper_nav2_amcl.launch.py map:=<path to your new .yaml>
```

Either works — the current stacks take an absolute `map:=` path, so nothing has
to live inside the package. If you do add it here, rebuild: `install(DIRECTORY
map/ ...)` copies rather than symlinks.

## 📁 Package Structure

Pure C++ (`ament_cmake`), aside from one standalone, non-ROS utility script
(`ros2_ws/utils/generate_keepout.py`):

```
pepper_navigation/
├── config/
│   ├── nav2_params.yaml                      # Nav2 stack parameters (AMCL + static map, wheel odom)
│   ├── nav2_params_amcl.yaml                 # Nav2 params for AMCL on FAST-LIO odom
│   ├── nav2_params_fastloc.yaml              # Nav2 params for the fastlio_localization stack
│   ├── nav2_params_rtabmap_loc.yaml          # Nav2 params for the RTAB-Map localization stack
│   ├── ekf_nav.yaml                          # robot_localization EKF parameters (not yet launched)
│   └── README.md
├── launch/
│   ├── pepper_navigation.launch.py           # Nav2 + AMCL against a static map
│   ├── pepper_nav2_amcl.launch.py            # Nav2 + AMCL on FAST-LIO odom (localization baseline)
│   ├── pepper_nav2_fastloc.launch.py         # Nav2 + fastlio_localization (prior map in the iEKF)
│   ├── pepper_nav2_rtabmap_loc.launch.py     # Nav2 + FAST-LIO + RTAB-Map localization (.db)
│   └── odom_test.launch.py
├── map/
│   ├── pepper_map_lc.yaml        # the 2D grid every current stack defaults to; .pgm alongside
│   ├── keepout_zone.yaml         # keepout filter mask, legacy stack only; .pgm alongside
│   └── *.png                     # map preview renders
├── pcd/                          # fastloc's prior map (poses tracked, clouds gitignored)
│   ├── sc_pose_20260823.json     # per-keyframe poses, read by fastlio_localization
│   └── pepper_map_lc.pcd, pepper_map_lc_poses.txt   # the PGO run's 3D map product
├── scripts/                      # rclpy helper nodes, installed as executables
│   ├── wait_for_map_then_start.py # starts nav2 once the map frame exists
│   ├── localization_recovery.py   # /localization_recover, one entry point per profile
│   └── localization_watchdog.py   # holds navigation while localization is lost
├── src/tools/                    # dev/debug tooling, not the production pipeline
│   ├── send_goal.cpp             # CLI utility to send Nav2 goals
│   └── odom_path_publisher.cpp   # publishes traversed path for RViz2
├── tools/
│   (generate_keepout.py moved to ros2_ws/utils/)
│                                  # (not a ROS2 node - run manually with python3)
├── rviz/
│   ├── nav2_amcl.rviz            # AMCL stack: map, particle cloud, /scan, costmaps, zones
│   ├── nav2_fastloc.rviz         # fastloc stack: map, costmaps, plans, safety zones
│   └── odometry_test.rviz
├── package.xml
└── README.md
```

SLAM launch files, `mapper_params_online_async.yaml` and
`rtabmap_fastlio_mapping.rviz` now live in `pepper_slam/`. Maps stay here
because Nav2's map server and costmap filters are their runtime consumers.

Note: the wheel-odometry covariance node that `ekf_nav.yaml`'s `odom0` expects
(`/pepper_odom_filtered`) lives in a separate top-level package, `pepper_odom_covariance`
(`~/ros2_ws/src/pepper_odom_covariance/`) — not inside this package. It's
infrastructure-tier (reusable, dependency-light), not navigation-specific, so
it sits alongside `naoqi_driver2` rather than inside this repository.

## 🏗️ Architecture

The navigation stack integrates four main subsystems:

1. **Odometry Layer** (upstream, separate package):
   - `naoqi_driver2` publishes raw wheel+IMU odometry on `/pepper_odom`, with a
     flat, non-growing covariance
   - `pepper_odom_covariance` (top-level package, not part of this repository)
     republishes it as `/pepper_odom_filtered` with a covariance that grows
     with distance/rotation traveled - this is what `ekf_nav.yaml`'s `odom0`
     and `nav2_params.yaml`'s `bt_navigator.odom_topic` expect as input

2. **Mapping Layer** (separate package, `pepper_slam`):
   - **RTAB-Map**: RGB-D / lidar SLAM, optionally on FAST-LIO odometry
   - **SLAM Toolbox**: 2D LiDAR SLAM with loop closure
   - Publishes the `map` frame and `/map` that this package consumes

3. **Localization Layer** — three interchangeable implementations, one interface
   (`map → odom` + a `/map` for the static layer):
   - **AMCL** over a flattened `/scan`, on FAST-LIO odometry
   - **`fastlio_localization`**, the prior map registered inside FAST-LIO's iEKF
   - **RTAB-Map** in localization mode against a `.db`, which also publishes `/map`
   - `ekf_nav.yaml` configures `robot_localization`'s `ekf_node` to fuse
     `/pepper_odom_filtered` (and, once wired in, a LIO odometry source) -
     drafted but not yet launched by anything

4. **Navigation Layer (Nav2)**:
   - **Map Server**: Serves occupancy grid and keepout filter mask
   - **Controller Server**: Local trajectory following
   - **Planner Server**: Global path computation
   - **Behavior Server**: Recovery behaviors
   - **BT Navigator**: Behavior tree orchestration
   - **Lifecycle Manager**: Node lifecycle management

5. **Safety Layer**:
   - `cloud_range_filter.py` (from `fast_lio`) strips Pepper's own body out of
     the raw L2 cloud and republishes it as `/points_safety`
   - `collision_monitor` reads that directly — not the costmaps — and gates
     `cmd_vel_raw → cmd_vel`, so it can veto the planner regardless of what the
     costmaps believe. The RTAB-Map stack gives it its own lifecycle manager so a
     planner failure and a safety failure don't take each other's bond down

> **Fixed**: `pepper_navigation.launch.py` used to also launch a
> `static_transform_publisher` publishing a fixed identity `map→odom`
> transform unconditionally, alongside AMCL's own live, scan-corrected one -
> two publishers of the same transform, a real TF conflict and a likely
> source of localization jitter. Removed; AMCL's `nav2_params.yaml` already
> has `set_initial_pose: true` (origin) and `tf_broadcast: true`, so it
> publishes the real transform on its own without it.

## 🧪 Testing

```bash
# Check active nodes, and that every lifecycle node reached 'active'
ros2 node list
ros2 lifecycle get /amcl        # or /map_server, /controller_server, ...

# Sensors alive? Both must be flowing before anything else works.
ros2 topic hz /points
ros2 topic hz /imu/data

# Check the transform tree -- exactly one publisher of map -> odom
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo map base_footprint

# Localization converging? (AMCL stacks)
ros2 topic hz /particle_cloud
ros2 topic echo /amcl_pose --once

# Monitor map / costmap output
ros2 topic echo /map --no-arr

# Is the safety layer intervening? 0 = clear, non-zero = slowdown/stop
ros2 topic echo /collision_monitor_state

# Compare what the planner asked for against what the robot got
ros2 topic hz /cmd_vel_raw /cmd_vel

# Monitor navigation feedback
ros2 action list
```

Common failure modes:

| Symptom | Likely cause |
|---------|--------------|
| Costmaps empty, no plan | `/points` not flowing — the lidar driver isn't running |
| Robot freezes in place, never moves | Collision monitor stopping on self-hits: check `/points_safety` exists and `points_safety_filter` is running |
| `map → odom` jitter or TF warnings | Two publishers of the same transform — check `bridge_level_frame` is set correctly for the stack you are running |
| AMCL particles never tighten | Flattened `/scan` doesn't match the grid — retune `scan_min_height`/`scan_max_height` |
| Global costmap all unknown | `/map` never arrived: wrong `map` path, or `map_server` never activated |

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.