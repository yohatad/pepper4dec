<div align="center">
<h1>Pepper Navigation and Mapping (pepper_navmap)</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **pepper_navmap** package is a **ROS2** metapackage that provides autonomous localization, mapping, and navigation capabilities for the Pepper robot. It integrates **RTAB-Map** for 3D SLAM using an Intel RealSense depth camera, **SLAM Toolbox** for 2D LiDAR-based mapping, and **Nav2** for path planning, obstacle avoidance, and goal navigation. The package also supports keepout zones and provides a utility node for programmatically sending navigation goals.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **RTAB-Map Integration**: 3D SLAM using RealSense RGB-D camera for rich environment mapping
- **SLAM Toolbox Integration**: 2D LiDAR-based online asynchronous SLAM with loop closure
- **Nav2 Stack**: Full autonomous navigation with path planning, behavior trees, and costmaps
- **Keepout Zones**: Configurable restricted areas using costmap filter masks
- **Goal Navigation API**: Python utility (`send_goal.py`) for programmatic navigation goal sending
- **Pre-built Maps**: Includes saved maps for localization-only deployments

# 🛠️ Installation

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **Intel RealSense D-series camera** with USB 3.0 connection (for RTAB-Map)
- **YDLidar or compatible 2D LiDAR** (for SLAM Toolbox)
- **Pepper robot** with ROS2 driver configured

## Required ROS2 Packages

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
  ros-humble-slam-toolbox \
  ros-humble-rtabmap-ros \
  ros-humble-realsense2-camera
```

## Package Installation

1. **Clone and Build the Workspace**
```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone <repository-url>

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select pepper_navigation
source install/setup.bash
```

# 🔧 Configuration

## SLAM Toolbox (`config/mapper_params_online_async.yaml`)

Key parameters tuned for YDLidar on Pepper:

| Parameter                        | Description                                      | Default Value |
|----------------------------------|--------------------------------------------------|---------------|
| `mode`                           | Operation mode (`mapping` or `localization`)     | `mapping`     |
| `scan_topic`                     | LiDAR scan topic                                 | `/scan`       |
| `base_frame`                     | Robot base frame                                 | `base_footprint` |
| `max_laser_range`                | Maximum laser range (m)                          | `12.0`        |
| `resolution`                     | Map resolution (m/cell)                          | `0.05`        |
| `link_scan_maximum_distance`     | Maximum distance for scan matching links (m)     | `10.0`        |
| `loop_search_maximum_distance`   | Maximum distance for loop closure search (m)     | `8.0`         |
| `do_loop_closing`                | Enable loop closure                              | `true`        |
| `minimum_travel_distance`        | Min travel distance before new scan (m)          | `0.2`         |
| `hit_probability`                | Occupancy grid hit probability                   | `0.7`         |

## Nav2 (`config/nav2_params.yaml`)

Key components configured:

| Component              | Description                                                |
|------------------------|------------------------------------------------------------|
| `amcl`                 | Adaptive Monte Carlo Localization (for use with saved maps)|
| `bt_navigator`         | Behavior tree-based navigation with replanning/recovery    |
| `controller_server`    | Local path follower (DWB controller)                       |
| `planner_server`       | Global path planner (NavFn/Smac)                           |
| `behavior_server`      | Recovery behaviors (spin, backup, wait)                    |
| `costmap_filter`       | Keepout zone filter integration                            |

> **Note:**
> AMCL is disabled by default in `pepper_navigation.launch.py` since RTAB-Map provides localization. Enable it only when using a pre-built static map without RTAB-Map.

# 🚀 Running the Stack

## Source the Workspace
```bash
source ~/ros2_ws/install/setup.bash
```

## Option 1: Nav2 Navigation with Pre-built Map
Launches the full Nav2 stack with a pre-built map, keepout zones, and costmap filter:

```bash
ros2 launch pepper_navigation pepper_navigation.launch.py
```

## Option 2: RTAB-Map SLAM (3D Mapping with RealSense)
Launches RTAB-Map with the RealSense camera for 3D SLAM and localization:

```bash
ros2 launch pepper_navigation rtabmap_realsense.launch.py
```

## Option 3: SLAM Toolbox (2D Mapping with LiDAR)
Launches SLAM Toolbox for 2D online mapping:

```bash
ros2 launch pepper_navigation slam_toolbox.launch.py
```

## Launch Arguments

### `pepper_navigation.launch.py`
No additional arguments required. Edit `launch/pepper_navigation.launch.py` to change the map file or parameters path.

### `rtabmap_realsense.launch.py`
Supports multiple configurable arguments:

| Argument          | Description                                         | Default     |
|-------------------|-----------------------------------------------------|-------------|
| `localization`    | Run in localization mode (no new mapping)           | `false`     |
| `use_sim_time`    | Use simulation time                                 | `false`     |
| `rviz`            | Launch RViz visualization                           | `false`     |
| `rtabmapviz`      | Launch RTAB-Map visualization tool                  | `false`     |

### Example: Localization Only (no new mapping)
```bash
ros2 launch pepper_navigation rtabmap_realsense.launch.py localization:=true
```

### Example: With RViz Visualization
```bash
ros2 launch pepper_navigation rtabmap_realsense.launch.py rviz:=true
```

# 🗺️ Maps

Pre-built maps are stored in the `map/` directory:

| File                      | Description                                  |
|---------------------------|----------------------------------------------|
| `rtabmap_feb_15.yaml`     | Map built with RTAB-Map (default for Nav2)   |
| `map.yaml`                | General-purpose map                          |
| `my_map.yaml`             | Alternative saved map                        |
| `keepout_zones.yaml`      | Keepout zone filter mask for Nav2 costmaps   |

### Saving a New Map (SLAM Toolbox)
```bash
ros2 run nav2_map_server map_saver_cli -f ~/ros2_ws/src/pepper4dec/dec_system/pepper_navmap/map/my_new_map
```

# 🧭 Sending Navigation Goals

## Using RViz2
With the Nav2 stack running, use the **Nav2 Goal** tool in RViz2 to click a target pose on the map.

## Using the send_goal Utility
The `send_goal.py` script programmatically sends a navigation goal using the Nav2 Simple Commander API:

```bash
ros2 run pepper_navigation send_goal
```

Edit `pepper_navigation/send_goal.py` to change the target coordinates:

```python
goal_pose.pose.position.x = 2.0   # Target X in map frame (meters)
goal_pose.pose.position.y = 1.0   # Target Y in map frame (meters)
goal_pose.pose.orientation.w = 1.0 # Orientation (1.0 = facing forward)
```

## Using Nav2 Action CLI
```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 1.0}, orientation: {w: 1.0}}}}"
```

# 🖥️ Topics and Interfaces

## Key Published Topics

| Topic                    | Type                              | Description                          |
|--------------------------|-----------------------------------|--------------------------------------|
| `/map`                   | `nav_msgs/OccupancyGrid`          | 2D occupancy grid map                |
| `/global_costmap/costmap`| `nav_msgs/OccupancyGrid`          | Global costmap for path planning     |
| `/local_costmap/costmap` | `nav_msgs/OccupancyGrid`          | Local costmap for obstacle avoidance |
| `/plan`                  | `nav_msgs/Path`                   | Current planned global path          |

## Key Subscribed Topics

| Topic              | Type                        | Description                              |
|--------------------|-----------------------------|------------------------------------------|
| `/scan`            | `sensor_msgs/LaserScan`     | 2D LiDAR scan (SLAM Toolbox / AMCL)     |
| `/odom`            | `nav_msgs/Odometry`         | Robot odometry                           |
| `/tf`              | `tf2_msgs/TFMessage`        | Transform tree                           |
| `/camera/color/image_raw` | `sensor_msgs/Image`  | RGB image (RTAB-Map)                    |
| `/camera/depth/image_raw` | `sensor_msgs/Image`  | Depth image (RTAB-Map)                  |

## Verification
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

# 🏗️ Architecture

The navigation stack integrates three main subsystems:

1. **Mapping Layer** (choose one):
   - **RTAB-Map**: RGB-D SLAM using RealSense — builds 3D point cloud map and 2D occupancy projection
   - **SLAM Toolbox**: 2D LiDAR SLAM with online loop closure — suitable for flat environments

2. **Localization Layer**:
   - RTAB-Map provides continuous localization when used as the SLAM backend
   - AMCL can be used for localization against a static pre-built map

3. **Navigation Layer (Nav2)**:
   - **Map Server**: Serves the occupancy grid and keepout filter mask
   - **Controller Server**: Executes local trajectory following (DWB)
   - **Planner Server**: Computes global paths (NavFn)
   - **Behavior Server**: Handles recovery behaviors
   - **BT Navigator**: Orchestrates navigation via behavior trees
   - **Lifecycle Manager**: Manages startup and shutdown of all Nav2 nodes

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a><br>
- Visit: <a href="http://www.dec4africa.org">www.dec4africa.org</a>

# 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
