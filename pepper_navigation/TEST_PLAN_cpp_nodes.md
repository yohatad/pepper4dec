# Robot test plan: C++ relay, cloud filter and wheel odom TF; Python TF listeners

Temporary. Delete this file once the tests below have passed on the robot.

## What is under test

| Repo | Branch | Commits |
|---|---|---|
| pepper4dec | `feature/auto-docking` | `2359628` tf_nav_relay in C++, `6f41eb7` cloud_range_filter + wheel_odom_tf in C++, `1b87fe3` Python nodes drop /tf after their one-time lookups |
| fastlio_lc_pgo | `main` (uncommitted) | `executable='cloud_range_filter'` in `fastlio_lc_l2.launch.py` and `pointlio_lc_l2.launch.py` |

The three pepper4dec commits sit directly on `devel`'s tip, before any
docking commit, so `devel` can take them on their own. The fastlio_lc_pgo
change must land together with `6f41eb7`: before it, those launches start
`cloud_range_filter.py`, which no longer exists.

## Setup

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select pepper_slam pepper_navigation fastlio_lc_pgo \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
colcon test --packages-select pepper_slam && colcon test-result --test-result-base build/pepper_slam
```

Release matters for the C++ nodes: a plain `colcon build` compiles them
unoptimized (the Dockerfile already passes Release). With `--symlink-install`,
delete any leftover `install/pepper_*/lib/pepper_*/{tf_nav_relay,cloud_range_filter,wheel_odom_tf}.py`
symlinks; they point at files that are gone.

Expected: builds clean, pepper_slam tests pass (86 at the time of writing).

## 1. AMCL stack: relay and both cloud filters

`ros2 launch pepper_navigation pepper_nav2_amcl.launch.py`

| Check | Expected |
|---|---|
| tf_nav_relay log | `relaying /tf -> /tf_nav for parents [base_footprint, lio_init, map, odom], dropping children [base_link]` |
| `ros2 topic echo /tf_nav --field transforms` | Only map / odom / lio_init / base_footprint parents, no joint frames |
| Local costmap over a few minutes of driving | Keeps updating (the freeze the relay fixed does not come back) |
| `ros2 topic hz /points_safety` | Same rate as `/points` |
| Walk into the 0.40 m stop zone (legs, low box) | collision_monitor stops the base, as before |
| points_costmap_filter log, every 2 s | `ground-plane filter: N -> M pts (K dropped as floor)` with K a large share of N |
| points_costmap_filter log | No `TF base_footprint <- l2lidar_frame unavailable` warnings (it now reads /tf_nav + /tf_static) |
| Local costmap while driving | No floor marked as obstacles |
| `top -c` | tf_nav_relay, both cloud_range_filter and lio_odom_bridge each at a few % or less (before: ~10%, ~20%+ and ~11%) |

## 2. fastloc with `odom_frame:=odom`: wheel odom TF and the start waiter

`ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py odom_frame:=odom` (needs naoqi_driver, for `/pepper_odom`)

| Check | Expected |
|---|---|
| wheel_odom_tf log | `/pepper_odom -> TF odom -> base_footprint` |
| `ros2 run tf2_ros tf2_echo odom base_footprint` while driving | Updates at ~50 Hz and follows the wheels |
| wait_for_map_then_start log after the lock | `map is up after Ns; starting nav2`, then `nav2 startup: OK` |
| `top -c` after Nav2 is up | wait_for_map_then_start ~0% (before: ~11% for the whole run) |

## 3. Mapping: fastlio_lc_l2 (and pointlio_lc_l2) with occupancy on

| Check | Expected |
|---|---|
| cloud_range_filter log | `cloud_range_filter: /cloud_registered_body -> /cloud_registered_body_filtered, keep 0.8 m <= range (no max), ROR off, ground-plane removal off.` |
| `ros2 topic hz /cloud_registered_body_filtered` | Same rate as `/cloud_registered_body`; octomap fills as before |
| pgo_map_odom_bridge log | `Published static map -> pgo_init (one-time leveling from calibration, ...)` |
| lio_odom_bridge log | `Published static odom -> lio_init ...`; `tf2_echo lio_init base_footprint` keeps updating |
| `top -c` | Both bridges under ~2% (before: ~11-12% each) |
