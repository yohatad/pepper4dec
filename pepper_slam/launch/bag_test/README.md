# bag_test/ — replay entry points

Every launch file in here targets **recorded bags**. The live-robot equivalents
stay one directory up, in `pepper_slam/launch/`.

Each file here is a thin wrapper: it sets `use_sim_time:=true` and whatever else
only applies to replay, then delegates to the real launch file. Nothing is
duplicated — if you change behaviour, change it in the live launch and both
paths follow.

| bag entry point | wraps | live equivalent |
|---|---|---|
| `fastlio_odometry_bag.launch.py` | `pepper_slam/fastlio_odometry.launch.py` | same, `use_sim_time:=false` |
| `pointlio_odometry_bag.launch.py` | `pepper_slam/pointlio_odometry.launch.py` | same |
| `fastlio_lc_bag.launch.py` | `fastlio_lc_pgo/fastlio_lc_l2.launch.py` | same |
| `pointlio_lc_bag.launch.py` | `fastlio_lc_pgo/pointlio_lc_l2.launch.py` | same |
| `fastlio_localization_bag.launch.py` | `lio_localization/fastlio_localization_l2.launch.py` | same |
| `rtabmap_*_bag.launch.py` | `pepper_slam/rtabmap_base.launch.py` (+ a LIO) | build one from `rtabmap_base` |

Navigation follows the same convention but lives in its own package, since the
live entry point does:

| bag entry point | wraps |
|---|---|
| `pepper_navigation/launch/bag_test/pepper_nav2_fastlio_loc_bag.launch.py` | `pepper_navigation/pepper_nav2_fastlio_loc.launch.py` |

```bash
ros2 launch pepper_navigation pepper_nav2_fastlio_loc_bag.launch.py \
    map_pcd:=<run>/map_batch.pcd map:=<run>/grid.yaml \
    keyframe_poses:=<run>/optimized_poses.txt
```

## `--show-args` lists far more than any one file honours

It walks the whole include tree, so `rtabmap_fused_bag.launch.py` advertises
**81** arguments and `fastlio_odometry_bag.launch.py` 15, of which that file
declares 5. Read the "ARGUMENTS THIS FILE HONOURS" block in each header instead.

Two consequences worth knowing:

* An argument declared anywhere below you IS settable from the command line and
  overrides the declared default — that is how `publisher:=none` reaches
  `pepper_sensor_tf.launch.py` through three levels of wrapper.
* But an explicit `launch_arguments={'x': ...}` entry **shadows** a command-line
  `x:=…` for that subtree. The CLI value is silently ignored while `x` still
  appears in `--show-args`. So do not forward an inner argument under an alias
  "for convenience" — it makes the documented name a no-op. Measured 2026-08-24.

## Invocation

`ros2 launch` resolves by BASENAME, searching the share directory recursively --
so the `bag_test/` prefix is not part of the command and in fact fails:

```bash
ros2 launch pepper_slam fastlio_odometry_bag.launch.py     # correct
ros2 launch pepper_slam bag_test/fastlio_odometry_bag.launch.py   # NOT FOUND
```

The subdirectory organises the source tree; it is invisible at the command line.
Basenames therefore have to stay unique across `launch/` and `launch/bag_test/`,
which is why the bag wrappers are `*_bag.launch.py` rather than repeating the
live names.

## The replay command

Every one of these needs the same player invocation. **The QoS overrides are not
optional:**

```bash
ros2 bag play <bag> --clock \
  --qos-profile-overrides-path $(ros2 pkg prefix pepper_slam)/../../config/play_qos.yaml \
  --read-ahead-queue-size 2000
```

or, from the workspace root, `--qos-profile-overrides-path config/play_qos.yaml`.

`/imu/data`, `/camera/imu` and `/points` were all recorded BEST_EFFORT. A
RELIABLE subscriber matches nothing against a BEST_EFFORT publisher, so without
the overrides rmw silently delivers nothing and the estimator waits forever for
IMU init with no error printed. `/points` matters only for FAST-LIVO2, which
subscribes RELIABLE; the others use `SensorDataQoS()` and would match either way.

## Do NOT replay `/tf`

The bag's wheel odometry (`pepper_odom -> base_footprint`) fights
`lio_map_odom_bridge` for `base_footprint`'s parent. `/tf_static` is fine and
mostly redundant — `pepper_sensor_tf` supplies the rig transforms itself.

## Why there is no `scope` argument here

`pepper_sensor_tf`'s scope is derived from `use_sim_time` inside the live launch
files, because both answer the same question: is a RealSense driver running?

- replay → `all`: no driver, so the camera edges come from calibration.
  Without them `camera_imu_optical_frame` — the body frame `l2_rsimu.yaml`
  names — does not resolve and the bridge waits forever.
- robot → `mount`: the driver publishes those edges from the device, and
  `config/sensor_tf.yaml` warns its values *can differ* from the recovered ones.
  Publishing both leaves whichever `/tf_static` arrives last in force.

Setting `use_sim_time:=true` here is therefore enough. Override
`sensor_tf_scope` only for the odd case (a bag that already carries the
driver's camera TF, or replaying with a camera plugged in).

## IMU

All of these default to the **RealSense IMU** (`l2_rsimu.yaml` and friends). The
L2's own gyro cancels rotation about the gravity axis below ~16 deg/s and cost
139 deg of heading over a 744 s run — see `utils/L2_IMU/REPORT.md`. Pass
`config_file:=l2.yaml` to A/B against it; the matching `lidar_imu_frame` is
derived automatically, so you do not have to remember to change it too.
