<div align="center">
<h1>Pepper Navigation</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Pepper Navigation** package provides autonomous navigation for the Pepper
robot: Nav2 path planning, obstacle avoidance, goal execution, keepout zones,
and the saved maps those consume.

> **SLAM and mapping live in a separate package: [`pepper_slam`](../pepper_slam/).**
> RTAB-Map, SLAM Toolbox and the FAST-LIO integration were split out so that
> mapping and bag-replay work doesn't pull in the Nav2 dependency tree. This
> package `exec_depend`s on `pepper_slam`; the reverse is never true. See
> `pepper_slam/README.md` for the `pepper_odom → odom → odom_level → map`
> frame contract between them.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Nav2 Stack**: Full autonomous navigation with path planning and obstacle avoidance
- **Keepout Zones**: Configurable restricted areas using costmap filter masks
- **Goal Navigation API**: C++ utility for programmatic navigation goal sending
- **Pre-built Maps**: Saved maps and filter masks for localization-only deployments

## ✅ Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **Pepper robot** with ROS2 driver configured
- **`pepper_slam`** built in the same workspace (supplies the `map` frame and `/map`)

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
  ros-humble-nav2-costmap-2d
```

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
| `amcl` | Adaptive Monte Carlo Localization |
| `bt_navigator` | Behavior tree-based navigation |
| `controller_server` | Local path follower (DWB controller) |
| `planner_server` | Global path planner (NavFn) |
| `behavior_server` | Recovery behaviors |
| `costmap_filter` | Keepout zone filter integration |

## 🚀 Running the Stack

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash
```

### Option 1: Nav2 with a static pre-built map (AMCL)

```bash
ros2 launch pepper_navigation pepper_navigation.launch.py
```

### Option 2: Nav2 on FAST-LIO + RTAB-Map localization

The current best-validated stack. Brings up FAST-LIO and RTAB-Map (in
localization mode, from `pepper_slam`) alongside the Nav2 servers:

```bash
ros2 launch pepper_navigation pepper_nav2_bringup.launch.py

# against a bag instead of the robot
ros2 launch pepper_navigation pepper_nav2_bringup.launch.py use_sim_time:=true
```

### Building a map first

Mapping is `pepper_slam`'s job:

```bash
ros2 launch pepper_slam rtabmap_realsense.launch.py   # RTAB-Map, RealSense
ros2 launch pepper_slam slam_toolbox.launch.py        # SLAM Toolbox, 2D LiDAR
```

## 🖥️ ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/scan` | `sensor_msgs/LaserScan` | 2D LiDAR scan |
| `/points` | `sensor_msgs/PointCloud2` | Lidar cloud, consumed directly by the Nav2 costmaps |
| `/pepper_odom_filtered` | `nav_msgs/Odometry` | Covariance-corrected wheel odometry |
| `/map` | `nav_msgs/OccupancyGrid` | Occupancy grid, from `pepper_slam` or the map server |
| `/tf` | `tf2_msgs/TFMessage` | Transform tree |

Camera topics are subscribed by the SLAM nodes in `pepper_slam`, not here.

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/map` | `nav_msgs/OccupancyGrid` | 2D occupancy grid map |
| `/global_costmap/costmap` | `nav_msgs/OccupancyGrid` | Global costmap |
| `/local_costmap/costmap` | `nav_msgs/OccupancyGrid` | Local costmap |
| `/plan` | `nav_msgs/Path` | Current planned global path |

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

| File | Description |
|------|-------------|
| `rtabmap_march_28.yaml` | Map built with RTAB-Map (default for Nav2) |
| `rtabmap_feb_15.yaml`, `rtabmap_feb_26.yaml` | Earlier RTAB-Map captures |
| `map.yaml` | General-purpose map |
| `my_map.yaml` | Alternative saved map |
| `keepout_zone.yaml` | Keepout zone filter mask |

### Saving a New Map

With a `pepper_slam` mapping session running, save the grid into this package's
`map/` directory (that's where Nav2 expects to find it):

```bash
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/src/pepper4dec/pepper_navigation/map/my_new_map
```

## 📁 Package Structure

Pure C++ (`ament_cmake`), aside from one standalone, non-ROS utility script
(`tools/generate_keepout.py`):

```
pepper_navigation/
├── config/
│   ├── nav2_params.yaml                      # Nav2 stack parameters (AMCL + static map)
│   ├── nav2_params_rtabmap_loc.yaml          # Nav2 params for the RTAB-Map localization stack
│   ├── ekf_nav.yaml.yaml                     # robot_localization EKF parameters (not yet launched)
│   └── README.md
├── launch/
│   ├── pepper_navigation.launch.py           # Nav2 + AMCL against a static map
│   ├── pepper_nav2_bringup.launch.py         # Nav2 + FAST-LIO + RTAB-Map localization
│   └── odom_test.launch.py
├── map/
│   ├── rtabmap_march_28.yaml     # default RTAB-Map map (used by Nav2); .pgm alongside
│   ├── map.yaml, my_map.yaml     # general-purpose saved maps; .pgm alongside
│   ├── rtabmap_feb_15.yaml, rtabmap_feb_26.yaml  # earlier RTAB-Map captures; .pgm alongside
│   ├── keepout_zone.yaml         # keepout filter mask; .pgm alongside
│   └── *.png                     # map preview renders
├── src/tools/                    # dev/debug tooling, not the production pipeline
│   ├── send_goal.cpp             # CLI utility to send Nav2 goals
│   └── odom_path_publisher.cpp   # publishes traversed path for RViz2
├── tools/
│   └── generate_keepout.py       # standalone script: builds a keepout mask from a map
│                                  # (not a ROS2 node - run manually with python3)
├── rviz/
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

3. **Localization Layer**:
   - RTAB-Map provides continuous localization
   - AMCL for static map localization
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

> **Fixed**: `pepper_navigation.launch.py` used to also launch a
> `static_transform_publisher` publishing a fixed identity `map→odom`
> transform unconditionally, alongside AMCL's own live, scan-corrected one -
> two publishers of the same transform, a real TF conflict and a likely
> source of localization jitter. Removed; AMCL's `nav2_params.yaml` already
> has `set_initial_pose: true` (origin) and `tf_broadcast: true`, so it
> publishes the real transform on its own without it.

## 🧪 Testing

```bash
# Check active nodes
ros2 node list

# Monitor map output
ros2 topic echo /map --no-arr

# Check transform tree
ros2 run tf2_tools view_frames

# Monitor navigation feedback
ros2 action list
```

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.