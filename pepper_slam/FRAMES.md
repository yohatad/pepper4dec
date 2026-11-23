# Frame contract

One place for what every frame means. Launch files and configs should point
here rather than re-deriving the story.

## TL;DR

| frame | what it is | use it for |
|---|---|---|
| `map` | fixed world reference, **gravity-aligned, floor-referenced** | goals, global costmap, prior maps |
| `odom` | continuous dead reckoning, **gravity-aligned, floor-referenced** | local costmap, collision monitor, AMCL odom frame |
| `lio_init` | the **LIO's own** world frame -- tilted ~90 deg by the sensor mount | nothing. Private to the backend. |
| `pgo_init` | **PGO's own** world frame -- same tilt | nothing. Private to `pgo_node`. |
| `base_footprint` | the robot, on the ground | -- |

**Rule: if a DOWNSTREAM config names `lio_init` or `pgo_init`, it is wrong.**
(The estimator configs necessarily define them via `publish.map_frame` -- that
is the one legitimate mention. Nothing consuming poses should name them.)

## Why the `*_init` frames exist

The L2 is bolted to Pepper's base at ~90 deg: **Z points forward**, and gravity
splits across the IMU's X and Y as `[-0.913, +0.407, +0.004]`. The LIO
initialises its world frame at identity, so that frame inherits the mount
orientation -- its Z axis is roughly *forward*, not up.

The mount is a single rotation of **163.1 deg** about `[-0.701, 0.149, -0.698]`.
Decomposed: `lidar +Z -> base +X` (forward), with the lidar clocked **24.1 deg**
about its own boresight -- that clocking is exactly what splits gravity across
X and Y instead of putting it on one axis.

That frame cannot simply be redefined, because it is the frame each backend
stamps its own outputs in: `/cloud_registered`, `/path`, `/Laser_map` for the
LIO; `/aft_pgo_map`, `/aft_pgo_path` for PGO. Rotating it means rotating all of
them, continuously. So the tilted frame keeps a name that says whose it is
(`lio_init`, `pgo_init`), and the leveled frame takes the standard name.

The two backends also disagree about that frame:

* **FAST-LIO** leaves it at identity, carrying gravity as an estimated S2 state
  (`IMU_Processing.hpp:200`; the alignment line is commented out upstream).
* **Point-LIO** can rotate its world frame to align gravity (`gravity_align`).

`mapping.gravity_align` is therefore **false** in
`point_lio/config/l2lidar_node.yaml`, so both backends define `lio_init`
identically. Turning it back on breaks `level_source: calibration` below.

## The ladder

Mapping (`fastlio_mapping`, `pointlio_mapping`) and AMCL:

```
odom                       leveled, floor-referenced   <- costmaps use this
 └── lio_init            the LIO's tilted native frame
      └── base_footprint   the robot
           ├── base_link        body (Pepper URDF hangs off this)
           └── l2lidar_frame    the lidar
                └── l2lidar_frame_imu the IMU inside it
```

PGO (`fastlio_lc_l2`, `pointlio_lc_l2`):

```
map -> pgo_init -> lio_init -> base_footprint -> ...
```

Localization (`fastlio_localization` / `pointlio_localization`, via
`pepper_nav2_fastloc` / `pepper_nav2_pointloc`), default `odom_frame:=''`:

```
map -> base_footprint -> ...      NO odom, NO lio_init
```

With `odom_frame:=odom` (REP-105, opt-in, see below):

```
map -> odom -> base_footprint -> ...      NO lio_init
```

The default is the exception to everything above. `fastlio_localization` loads the
prior map into the ikd-Tree its iEKF registers against, so after the initial
lock the filter state **is** the map pose -- there is no separate odometry
estimate to name, and no `map -> odom` correction to publish. It broadcasts
`map -> base_footprint` itself (`publish.tf_child_frame`), and neither
`lio_odom_bridge` nor any fusion node runs in that stack; adding one would give
`base_footprint` two parents. The local costmap therefore rolls in `map`, not
`odom` -- see `local_costmap.global_frame` in `nav2_params_fastloc.yaml`.

The cost of that shortcut: every step in the filter's pose (lock handover,
`/initialpose` seed, re-arm) lands directly on `base_footprint`, where the
local costmap and controller see it, and nothing gates it. The overlap health
check only reports a bad lock, after 5 s. If the localizer stalls,
`base_footprint` stops moving too.

`odom_frame:=odom` restores the REP-105 split. The localizer broadcasts
`map -> odom`, composed at the scan stamp so that
`map -> odom -> base_footprint` equals its own pose (planar: x, y, z, yaw).
`wheel_odom_tf` supplies `odom -> base_footprint` from `/pepper_odom`, and
the local costmap moves to `odom`. The edge goes through a jump gate
(`map_odom.hpp` in both localizers): a step over 0.30 m / 10 deg per scan is
held for up to 2 s, then adopted with an ERROR, while a deliberate handover or
seed resets the gate and passes. **Not yet verified on the robot.**

In the other three stacks `base_footprint` is a child of `lio_init` -- that is
the edge the bridge publishes. A lookup of `odom -> base_footprint` traverses
`odom -> lio_init -> base_footprint` and works normally.

| edge | kind | published by | meaning |
|---|---|---|---|
| `l2lidar_frame -> l2lidar_frame_imu` | static | `pepper_sensor_tf` | IMU's position inside the lidar housing (17 mm, no rotation) |
| `base_footprint -> l2lidar_frame` | static | `pepper_sensor_tf` | **the mount calibration** |
| `lio_init -> base_footprint` | dynamic ~11 Hz | `lio_odom_bridge` | **the odometry** -- continuous, drifts, never corrected |
| `map -> lio_init` | dynamic, jumps | AMCL | **the correction** -- discontinuous, does not drift |
| `map -> base_footprint` | dynamic, scan rate | `fastlio_localization`, `pointlio_localization` (default) | pose and correction in one: the map is inside the filter, so there is no separate correction edge |
| `map -> odom` | dynamic, scan rate, gated | `fastlio_localization`, `pointlio_localization` (`odom_frame:=odom`) | the filter pose with wheel odometry composed out |
| `odom -> base_footprint` | dynamic, 50 Hz | `wheel_odom_tf` (loc stacks, `odom_frame:=odom` only) | wheel odometry, planar |
| `odom <-> lio_init` | static, one-time | `lio_odom_bridge` | **the leveling** (0.3139 m with the RealSense IMU, from calibration) |
| `map -> pgo_init` | static, one-time | `pgo_map_odom_bridge` | the leveling, map side |

Measured on the July_22 bag over 35 s: `lio_init -> base_footprint` moved in
320 smooth ~32 mm increments (robot motion); `map -> lio_init` stepped 9 times
by ~110 mm (corrections). Two very different signatures on the same clock --
which is why local consumers read the odom side and global ones read `map`.

## Where the leveling comes from

`level_source: calibration` (default) reads it off the rigid mount:
`R_level = R_base_imu`, floor offset `= t_base_imu.z`: 0.3139 m for the
RealSense IMU (`camera_imu_optical_frame`) every config now uses; it was
0.2571 m for the L2's own IMU (`l2lidar_frame_imu`) before `l2.yaml` was
removed. Exact,
identical on every backend and every run, and available before any odometry
arrives.

`level_source: odometry` is the legacy path -- it snapshots the first odometry
message, which is only equivalent if the robot has not moved by then. It had
not: FAST-LIO measured 0.257 m, Point-LIO 0.232 m for the same rig, because the
backends start publishing at different points in their init. The `R_init` terms
cancel algebraically; only the robot's motion before that first sample survives.

`calibration` REQUIRES the LIO world frame to start at identity rotation --
true for FAST-LIO always, and for Point-LIO only with `gravity_align: false`.

## Parent or child?

`lio_init` can only have one parent, so the leveled frame flips role
depending on who owns it:

| stack | leveling edge |
|---|---|
| `fastlio_mapping`, `pointlio_mapping`, `pepper_nav2_amcl` | `odom -> lio_init` (odom is parent) |
| `fastlio_lc_l2`, `pointlio_lc_l2` (PGO) | `map -> pgo_init`; no `odom` frame here |
| `fastlio_localization`, `pointlio_localization` | none -- no `lio_init`; the prior map is already leveled. Default: `map -> base_footprint` straight out. `odom_frame:=odom`: `map -> odom`, with `odom` planar wheel odometry, floor-referenced by construction |

`pgo_init` exists only during mapping. Both artifacts are written **into** the
leveled frame (octomap builds the grid in `map`; `pgo_node` transforms
`map_batch.pcd` into `map` before saving), so the localization stacks get a
`map` that is already leveled and need no `pgo_init` of their own.

## Why every z number means something

`min_obstacle_height`, `max_obstacle_height`, `origin_z`, `occ_min_z` are all
"metres above the floor" **only** because `odom`/`map` put z=0 on the ground
plane. In a tilted frame a "height" band slices horizontally through the room
and cannot separate floor from ceiling at all.

## Verifying

`colcon test --packages-select pepper_slam` runs this on the benchmark bag
through FAST-LIO and `lio_odom_bridge` (`test/test_frame_contract_bag.py`). The
checker is a test executable, so against a live system, whichever backend runs:

```bash
build/pepper_slam/check_frame_contract
build/pepper_slam/check_frame_contract --ros-args -p level_frame:=map
```

It asserts the leveling offset equals the calibration prediction (`t_base_imu.z`,
read from TF: 0.3139 m by default),
`base_footprint` roll/pitch are small, and the floor plane sits at z ~= 0.

Measured after the rename (2026-08-02), July_22 bag, when the estimators still
ran on the L2 IMU (hence the 0.2571 m offset):

| stack | offset | roll/pitch | floor z | verified live |
|---|---|---|---|---|
| FAST-LIO mapping | 0.2571 | +0.40 / -2.00 | -0.038 | yes |
| Point-LIO mapping | 0.2571 | -0.55 / -3.26 | -0.041 | yes |
| AMCL | 0.2571 | +0.40 / -1.99 | -0.038 | yes |
| PGO (FAST-LIO) | -- | robot upright ~4 deg in `map` | -- | yes |
| FAST-LIO localization † | -- | robot upright 3.69 deg in `map` | -- | yes |
| Point-LIO localization † | -- | robot upright 2.68 deg in `map` | -- | yes |

† Measured against the old `lio_localization` stack, which has since been
removed. The numbers stand as a record of the leveling at that date, but they
are **not** a verification of today's `fastlio_localization`, which reaches
`map` by a different path (no `lio_init`, no `odom`).

**KNOWN ISSUE:** the roll/pitch assertion cannot distinguish leveling error from
LIO attitude drift, so it flags Point-LIO at -3.26 deg against an arbitrary
3.0 deg threshold. Its offset and floor position match the passing stacks
exactly, so this is a threshold artifact, not a defect. The fix is to compare
t=0 against later in the run -- leveling error appears immediately, drift
accumulates -- not to widen the threshold.

## Oddities

* **`aft_mapped`** -- orphan off `lio_init`. Point-LIO broadcasts it
  unconditionally (no `publish_tf` flag, unlike FAST-LIO) and it is left
  unclaimed so it cannot fight the static chain for a parent. Harmless.
* **`pepper_odom`** -- naoqi wheel odometry, on a deliberately **disconnected**
  tree (a second live parent for `base_footprint` would split the tree).
* **`base_footprint` vs `base_link`** -- REP-105 specifies `base_link`;
  `base_footprint` (ground projection) is convention, not spec.

## Static TF

The rig's 10 static transforms live in `urdf/sensor_rig.xacro`, the
calibration source of truth, and `pepper_sensor_tf.launch.py` publishes them
through one `robot_state_publisher` (node `/sensor_rig/pepper_sensor_tf`). With
`scope:=mount`, the default, it leaves out the seven edges below
`camera_camera_link`, which the RealSense driver publishes itself. Edit the
xacro, not the launch file, and update the values derived from it (listed in
the xacro's header); `test/test_rig_consistency.py` checks them.

## History

Frames were renamed (2026-08-02) so the standard name is the useful one:

```
before        after
odom       -> lio_init     (the LIO's native, tilted frame)
odom_level -> odom           (leveled -- the standard name)
map        -> pgo_init      (PGO's native, tilted frame)
map_level  -> map            (leveled -- the standard name)
```

Rationale: `odom` -- the name every ROS tool, tutorial and config expects -- was
the one you must *not* use, while the correct frame had a name nothing
recognised. REP-105 requires `odom` to be *continuous*; it does not require it
to be the LIO's raw output frame. Nav configs are now plain
`global_frame: odom`, and "someone used the raw frame for geometry" -- the bug
class behind a 90-deg-tilted saved map -- is no longer expressible.
