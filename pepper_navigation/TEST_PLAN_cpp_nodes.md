# Robot test plan: C++ relay, cloud filter and wheel odom TF; Python TF listeners

Temporary. Delete this file once the tests below have passed on the robot.

## What is under test

| Repo | Branch | Commits |
|---|---|---|
| pepper4dec | `devel` | `2359628` tf_nav_relay in C++, `6f41eb7` cloud_range_filter + wheel_odom_tf in C++, `1b87fe3` Python nodes drop /tf after their one-time lookups, `9b88ae7` the two LIO bridges run OpenBLAS single-threaded |
| fastlio_lc_pgo | `main` | `cf16625` `executable='cloud_range_filter'` in `fastlio_lc_l2.launch.py` and `pointlio_lc_l2.launch.py` |

All of these are merged. The fastlio_lc_pgo commit had to land with `6f41eb7`:
before it, those launches started `cloud_range_filter.py`, which no longer
exists.

## Bag replay results (2026-11-13, laptop, no robot)

Replayed `bags/slam_20260823_aligned` with `--clock` and `use_sim_time:=true`.
The 28-core laptop is not Pepper's CPU, so the CPU figures compare nodes with
each other, not with the robot.

| What | Result |
|---|---|
| Setup: Release build, `colcon test` on pepper_slam | PASS: 92 tests, 0 failures, 5 skipped (cppcheck, which ament skips on Jammy) |
| §3 `fastlio_lc_l2` with occupancy on | PASS for everything except octomap filling, which was not looked at. cloud_range_filter log as below; `/points`, `/cloud_registered_body` and `/cloud_registered_body_filtered` all 11.34 Hz; `map -> pgo_init` published; `lio_init -> base_footprint` updating; cloud_range_filter 0.8% CPU |
| §3 bridge CPU | FAILED at first, then FIXED: lio_odom_bridge 304%, pgo_map_odom_bridge 73% (71% even after the bag stopped). numpy's OpenBLAS worker pool busy-waits between calls. `9b88ae7` sets `OPENBLAS_NUM_THREADS=1` before numpy is imported; both are now ~3% |
| §1 tf_nav_relay, run alone on the bag's `/tf` plus wheel_odom_tf's `odom -> base_footprint` | PASS: start-up log as below; of 84 parent/child pairs on `/tf` (83 joint frames and `base_footprint -> base_link`) only `odom -> base_footprint` was forwarded, all 856 messages in 20 s; 1.4% CPU |
| §2 wheel_odom_tf, run alone on the bag's `/pepper_odom` while driving | PASS: start-up log as below; one `odom -> base_footprint` per `/pepper_odom` message (42.8/s, the bag's odom rate); all 856 transforms identical to the odometry pose at the same stamp, robot moving 0.6 to 2.1 m; 1.2% CPU |

Still needs the robot: the costmap, collision_monitor and Nav2 start-up rows
in §1 and §2, plus a CPU check of every node on Pepper itself.

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
delete any leftover `install/pepper_*/lib/pepper_*/*.py` symlinks that point
at files that are gone (`find -L install -name '*.py' -type l` lists them).

Expected: builds clean, pepper_slam tests pass (92 at the time of writing).

## 1. AMCL stack: relay and both cloud filters

`ros2 launch pepper_navigation pepper_nav2_amcl.launch.py`

| Check | Expected |
|---|---|
| tf_nav_relay log | `relaying /tf -> /tf_nav for parents [base_footprint, lio_init, map, odom], dropping children [base_link]` (bag: PASS) |
| `ros2 topic echo /tf_nav --field transforms` | Only map / odom / lio_init / base_footprint parents, no joint frames (bag: PASS) |
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
| wheel_odom_tf log | `/pepper_odom -> TF odom -> base_footprint` (bag: PASS) |
| `ros2 run tf2_ros tf2_echo odom base_footprint` while driving | Updates at the `/pepper_odom` rate (~50 Hz live, 43 Hz in the bag) and follows the wheels (bag: PASS) |
| wait_for_map_then_start log after the lock | `map is up after Ns; starting nav2`, then `nav2 startup: OK` |
| `top -c` after Nav2 is up | wait_for_map_then_start ~0% (before: ~11% for the whole run) |

## 3. Mapping: fastlio_lc_l2 (and pointlio_lc_l2) with occupancy on

| Check | Expected |
|---|---|
| cloud_range_filter log | `cloud_range_filter: /cloud_registered_body -> /cloud_registered_body_filtered, keep 0.8 m <= range (no max), ROR off, side-body cut off, body-box cut off, ground-plane removal off.` (bag: PASS) |
| `ros2 topic hz /cloud_registered_body_filtered` | Same rate as `/cloud_registered_body`; octomap fills as before (bag: rates PASS, octomap not checked) |
| pgo_map_odom_bridge log | `Published static map -> pgo_init (one-time leveling from calibration, ...)` (bag: PASS) |
| lio_odom_bridge log | `tf2_echo lio_init base_footprint` keeps updating (bag: PASS). There is no `Published static odom -> lio_init` line in this launch: it starts the bridge with leveling off (`a higher layer owns lio_init`), and pgo_map_odom_bridge does the leveling |
| `top -c` | Both bridges ~3% or less (before: ~11-12% each; ~300% / ~73% on a 28-core laptop before `9b88ae7`) |
