# Frame contract

One place for what every frame means. Launch files and configs should point
here rather than re-deriving the story.

## TL;DR

| frame | what it is | use it for |
|---|---|---|
| `map` | fixed world reference, **gravity-aligned, floor-referenced** | goals, global costmap, prior maps |
| `odom` | continuous dead reckoning, **gravity-aligned, floor-referenced** | local costmap, collision monitor, AMCL/RTAB-Map odom frame |
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

Localization (`fastlio_localization_l2`, `pointlio_localization_l2`):

```
map -> lio_init -> base_footprint -> ...
        └── odom          published as a CHILD (level_frame_as_child:=true)
```

`base_footprint` is a child of `lio_init` in every case -- that is the edge
the bridge publishes. A lookup of `odom -> base_footprint` traverses
`odom -> lio_init -> base_footprint` and works normally.

| edge | kind | published by | meaning |
|---|---|---|---|
| `l2lidar_frame -> l2lidar_frame_imu` | static | `static_tf_publisher` | IMU's position inside the lidar housing (17 mm, no rotation) |
| `base_footprint -> l2lidar_frame` | static | `static_tf_publisher` | **the mount calibration** |
| `lio_init -> base_footprint` | dynamic ~11 Hz | `lio_odom_bridge` | **the odometry** -- continuous, drifts, never corrected |
| `map -> lio_init` | dynamic, jumps | the localizer | **the correction** -- discontinuous, does not drift |
| `odom <-> lio_init` | static, one-time | `lio_odom_bridge` | **the leveling** (0.2571 m, from calibration) |
| `map -> pgo_init` | static, one-time | `pgo_map_odom_bridge` | the leveling, map side |

Measured on the July_22 bag over 35 s: `lio_init -> base_footprint` moved in
320 smooth ~32 mm increments (robot motion); `map -> lio_init` stepped 9 times
by ~110 mm (corrections). Two very different signatures on the same clock --
which is why local consumers read the odom side and global ones read `map`.

## Where the leveling comes from

`level_source: calibration` (default) reads it off the rigid mount:
`R_level = R_base_imu`, floor offset `= t_base_imu.z = 0.2571 m`. Exact,
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
| `fastlio_localization_l2`, `pointlio_localization_l2` | `lio_init -> odom` (child; `level_frame_as_child:=true`) |
| `fastlio_lc_l2`, `pointlio_lc_l2` (PGO) | `map -> pgo_init`; no `odom` frame here |

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

```bash
ros2 run pepper_slam check_frame_contract.py
ros2 run pepper_slam check_frame_contract.py --ros-args -p level_frame:=map
```

Asserts the leveling offset equals the calibration prediction (0.2571 m),
`base_footprint` roll/pitch are small, and the floor plane sits at z ~= 0.

Measured after the rename (2026-08-02), July_22 bag:

| stack | offset | roll/pitch | floor z | verified live |
|---|---|---|---|---|
| FAST-LIO mapping | 0.2571 | +0.40 / -2.00 | -0.038 | yes |
| Point-LIO mapping | 0.2571 | -0.55 / -3.26 | -0.041 | yes |
| AMCL | 0.2571 | +0.40 / -1.99 | -0.038 | yes |
| PGO (FAST-LIO) | -- | robot upright ~4 deg in `map` | -- | yes |
| FAST-LIO localization | -- | robot upright 3.69 deg in `map` | -- | yes |
| Point-LIO localization | -- | robot upright 2.68 deg in `map` | -- | yes |
| RTAB-Map localization | -- | -- | -- | **blocked** |

RTAB-Map localization is **blocked, not skipped**: its default database
`~/.ros/rtabmap_fastlio_refined.db` does not exist, and the only `.db` present
(`rtabmap_fastlio_bag_test.db`, 106 kB, Jul 22) is far too small to be a map of
this environment and predates the floor-datum and frame changes. It needs a
fresh RTAB-Map mapping run before it can be verified. Its configs *were*
renamed correctly (`nav2_params_rtabmap_loc.yaml` `global_frame: odom`,
`odom_frame_id: "odom"`; `pepper_nav2_rtabmap_loc.launch.py` `odom_frame_id`),
and its topic remaps in `rtabmap_base.launch.py` were correctly left as bare
`odom`.

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
  tree (a second live parent for `base_footprint` would split the tree). This
  is why `pepper_odom_relabel.py` exists: with no TF path between them,
  `robot_localization` cannot transform the data itself.
* **`base_footprint` vs `base_link`** -- REP-105 specifies `base_link`;
  `base_footprint` (ground projection) is convention, not spec.
* **bare `odom` is also a TOPIC name** in the rtabmap launch files
  (`("odom", LaunchConfiguration('odom_topic'))`, `default_value='odom'`).
  Never bulk-rename `odom` across this workspace without excluding those --
  a blanket `\bodom\b` substitution silently corrupts the remappings.

## Static TF

The rig's 10 static transforms live in `config/sensor_tf.yaml` and are published
by a single `static_tf_publisher` node. This was previously 10 separate
`static_transform_publisher` processes -- one per edge, each a full ROS node
with its own executor and DDS participant, to publish seven constants. Edit the
YAML, not the launch file. The node rejects a frame given two parents and warns
on a non-unit quaternion.

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
