# Robot test plan: map -> odom, robot radius 0.25, KISS-ICP profile

Temporary. Delete this file once the tests below have passed on the robot and
the feature branches are merged.

## What is under test

| Repo | Branch | Commits |
|---|---|---|
| pepper4dec | `devel` | `6ac6313` opt-in map -> odom (fastloc/pointloc), `fda18c4` robot_radius 0.25 + geometry test, `0c4a347` KISS-ICP Nav2 profile |
| point_lio | `feature/map-odom-frame` | `57ef9c8` no map -> base_footprint before the lock, `c75dbf4` opt-in map -> odom |
| FAST_LIO | `feature/map-odom-frame` | `54a356f` opt-in map -> odom |
| kiss-icp | unchanged (`045c48b`) | needs building |

Nothing here is merged into `main` / `ros2`. Merge only after the tests pass.

## Setup

```bash
cd ~/ros2_ws/src
git -C pepper4dec switch devel && git -C pepper4dec pull
git -C FAST_LIO fetch && git -C FAST_LIO switch feature/map-odom-frame
git -C point_lio fetch && git -C point_lio switch feature/map-odom-frame
git -C kiss-icp pull
cd .. && colcon build --packages-select fast_lio point_lio kiss_icp pepper_slam pepper_navigation
source install/setup.bash
python3 -m pytest -q src/pepper4dec/pepper_navigation/test src/pepper4dec/pepper_slam/test
```

Expected: everything builds and all tests pass (24 at the time of writing).

Tests 3, 4 and 7 need `naoqi_driver` running, for `/pepper_odom`.

## 1. Point-LIO pre-lock fix

`ros2 launch pepper_navigation pepper_nav2_pointloc.launch.py` (default mode)

| Check | Expected |
|---|---|
| `ros2 run tf2_ros tf2_echo map base_footprint` before "Localized:" in the log | No transform. The robot is not drawn at the map origin |
| `ros2 topic hz /localization/pose` before the lock | Nothing published |
| 2D Pose Estimate in RViz before the lock | `/initialpose accepted` (not "extrinsic not ready"), then it locks |
| After the lock | Transform appears; the robot is in the right place on the map |

## 2. fastloc default mode (regression)

`ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py`

| Check | Expected |
|---|---|
| Normal navigation run | Same as before: `map -> base_footprint` direct, local costmap in `map` |

## 3. fastloc with `odom_frame:=odom`

`ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py odom_frame:=odom`

| Check | Expected |
|---|---|
| Startup log | `Publishing map -> odom (REP-105)`, and wheel_odom_tf logs `/pepper_odom -> TF odom -> base_footprint` |
| `ros2 run tf2_tools view_frames` | `map -> odom -> base_footprint -> l2lidar_frame`. `base_footprint` has one parent; no direct `map -> base_footprint` |
| `tf2_echo map base_footprint` vs `/localization/pose`, while driving | Agree within ~2 cm / 1 deg |
| `tf2_echo map odom`, normal driving | Changes slowly (a few cm/s at most). No "step ... held" warnings |
| 2D Pose Estimate or `/relocalize` mid-run | `map -> odom` jumps; the local costmap and the robot do not jerk; no "held" warning (a seed resets the gate) |
| Stop `naoqi_driver` while idle | Localizer warns `no odom -> base_footprint at the scan stamp`, TF goes stale, Nav2 stops. Recovers when the driver restarts |
| Normal route including a doorway | Goal reached; driving as smooth as default mode or better |

If "held" warnings appear in normal driving, naoqi's host-time stamps on
`/pepper_odom` are the likely cause: try `publish.odom_lookup_timeout: 0.1` and
note the step sizes in the log.

## 4. pointloc with `odom_frame:=odom`

`ros2 launch pepper_navigation pepper_nav2_pointloc.launch.py odom_frame:=odom`

Same checks and expected results as test 3.

## 5. Robot radius 0.25

Run on fastloc, then once on amcl (test 6).

| Check | Expected |
|---|---|
| Costmap startup log | No "inflation radius is smaller than the inscribed radius" warning |
| RViz footprint circle vs the real base | The 0.25 m circle covers Pepper's base on the floor (check physically once) |
| Narrowest doorway on the route | Passable if at least ~0.70 m wide. If the planner refuses, note the width |
| Approach a wall or a person | Slows at 0.80 m, stops at 0.40 m, as before |

## 6. amcl profile at the new radius

`ros2 launch pepper_navigation pepper_nav2_amcl.launch.py map:=<path>`

| Check | Expected |
|---|---|
| Set a pose, send one goal | Localizes and reaches the goal, with a slightly wider margin than before |

rtabmap_loc cannot be tested: no map database yet.

## 7. KISS-ICP profile

`ros2 launch pepper_navigation pepper_nav2_kissicp.launch.py map:=<path>`

| Check | Expected |
|---|---|
| kiss_icp_node log | Base frame `base_footprint`, odom frame `odom`; no repeated "wheel prior unavailable" |
| `view_frames` | `map -> odom` (amcl) `-> base_footprint` (kiss); one parent each |
| `ros2 topic hz /odom_lio` | ~10 Hz |
| Stand still 60 s | `odom -> base_footprint` drifts less than a few cm |
| `tf2_echo odom base_footprint` z / roll / pitch | z within +-5 cm, roll and pitch under 3 deg |
| Set a pose, drive a loop | AMCL converges and stays localized |
| Same route as test 6 | Goal reached; compare smoothness and relocalization with the FAST-LIO amcl stack |

## Known caveats

- Bag replay in odom mode: if the bag's `/tf` carries `pepper_odom -> base_footprint`,
  `base_footprint` gets two parents. Test live first.
- The map -> odom jump gate only logs; it does not yet raise a diagnostic the
  Nav2 watchdog acts on.
- KISS-ICP has no IMU and no pose guard (unlike `lio_odom_guard` on the LIO stacks).

## After the tests

- All pass: merge `feature/map-odom-frame` into `ros2` (FAST_LIO) and `main`
  (point_lio), fast-forward pepper4dec `main` to `devel`, and consider making
  `odom_frame:=odom` the default in the two loc launch files.
- Delete this file.
