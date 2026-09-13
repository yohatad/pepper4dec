<div align="center">
<h1>Pepper SLAM and Mapping</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Pepper SLAM** package builds and localizes against maps of the Pepper
robot's environment. It ships launch files and parameters only — RTAB-Map, SLAM
Toolbox and FAST-LIO are all upstream packages launched by name, so nothing here
compiles. The rig carries a RealSense (RGB + aligned depth + IMU) and a Unitree
L2 lidar (`/points` + `/imu/data`); odometry is wheel (`/pepper_odom`), RTAB-Map
`icp_odometry`, or FAST-LIO.

```bash
source ~/ros2_ws/install/setup.bash
```

## Frames

| Frame | Published by | Notes |
|-------|--------------|-------|
| `pepper_odom` | `naoqi_driver2` | wheel odometry; **not** named `odom` on purpose |
| `odom` | FAST-LIO | IMU-aligned, tilted ~90° on Pepper's mount — **not** gravity-aligned |
| `odom` | `lio_odom_bridge` | one-time gravity-leveled parent of `odom`; Z-up |
| `map` | RTAB-Map / PGO | whichever layer owns the loop-closure correction |
| `map` | `fastlio_localization` | prior-map localization: publishes `map -> base_footprint` directly, with no `odom` edge at all (see `FRAMES.md`) |

**Only one node may publish a given frame's parent** — that constraint is what
the FAST-LIO options below are choosing between. RTAB-Map must anchor on
`odom`, not `odom`: projecting a 2D grid out of a tilted frame silently
produces garbage ground/obstacle splits. See `config/README.md` in
`pepper_navigation` for the odometry naming rules.

## FAST-LIO options

All of these start the `fastlio_mapping` node plus `lio_odom_bridge`, via
`fast_lio mapping.launch.py`. **Do not run that upstream file directly on this
robot** — it does not include `pepper_sensor_tf.launch.py`, so the bridge waits
forever for the static `base_footprint -> l2lidar_frame_imu` chain and the stack
hangs with no error. Use the `pepper_slam` wrappers below, which include the TF
and pass `config_file:=l2.yaml` for you (the other profiles in `fast_lio/config/`
are upstream defaults for other lidars). Pass `use_sim_time:=true` for bag replay
— the wrappers already default it true.

| Stack | Launch | Loop closure | Owns `map` | `bridge_level_frame` |
|-------|--------|--------------|-----------|----------------------|
| Odometry only | `pepper_slam fastlio_odometry.launch.py` | none | nobody | `true` (unused) |
| + RTAB-Map | `pepper_slam bag_test/rtabmap_fastlio_bag.launch.py` | RTAB-Map ICP + visual BoW | RTAB-Map | `true` |
| + Scan-Context PGO | `fastlio_lc_pgo fastlio_lc_l2.launch.py` | GTSAM/ISAM2 on Scan Context | `pgo_map_odom_bridge` | `false` |
| + prior map in the filter | `fast_lio localization_l2.launch.py` | n/a (localization) | `fastlio_localization` | n/a (no bridge runs) |

Odometry alone has no drift correction. RTAB-Map on top is the validated mapping
configuration (best measured closure 0.19 m). PGO adds pose-graph loop closure
plus a ray-traced `/projected_map`. Prior-map localization is the lightest
runtime stack -- the map is loaded into the ikd-Tree the iEKF registers
against, so there is no correction node beside the filter and no `odom` frame
-- and is wrapped for Nav2 by `pepper_nav2_fastloc.launch.py`.

**`bridge_level_frame` is the one that bites.** FAST-LIO publishes in the raw
initial-IMU frame — the L2 IMU reads gravity along +X, so `odom` looks tilted
~90° — and `odom` is the bridge's one-time fix. Set it `false` whenever a
higher layer owns `odom`'s parent, or `odom` gets two parents and the TF tree
breaks.

## Divergence guard (`guard_enable`)

When FAST-LIO's plane correspondences collapse — feature-poor corridor, blank
wall, body-occluded lidar — the iEKF stops correcting but keeps propagating on
IMU. Accel bias and gravity misalignment get *double* integrated, so the pose
does not drift, it accelerates: the estimate sprints out of the map. Unguarded,
`lio_odom_bridge` republishes that straight onto `odom -> base_footprint`, i.e.
inside Nav2's control loop. Nav2's `max_vel_x` does not help — it clamps
*commanded* velocity and never inspects the pose.

```bash
ros2 launch pepper_slam fastlio_odometry.launch.py guard_enable:=true
```

The guard rejects any pose implying motion the base physically cannot perform
(`max_linear_speed` 0.7 m/s, `max_angular_speed` 1.0 rad/s — above Nav2's own
0.5 command caps, with margin), plus a zero-velocity cross-check: if the wheels
report stationary and LIO reports motion, LIO is wrong unconditionally.

**Rejected, never clipped.** Saturating a bad pose to the limit would
manufacture a plausible-looking but wrong estimate, which is strictly harder to
detect downstream than an obviously broken one. Clamping is right for an
actuator command, which must be feasible; wrong for a measurement, which must be
honest or absent.

While rejecting, the pose is carried forward on `wheel_odom_topic`
(`/pepper_odom`) rather than frozen — a frozen `odom` makes obstacles stream
past a robot that reports standing still. Only the wheel *delta* is borrowed, so
`pepper_odom` and `odom` staying disconnected trees is fine. Coasting is bounded
by `max_hold_duration` (3 s), after which the guard escalates to FAULT rather
than dead reckoning silently forever.

Two signals come out. `/localization/wheel_trust_scale` (`std_msgs/Float64`) is
the multiplier `pepper_odom_covariance` already subscribes to, where `<1` means
"trust wheels more". It is not a step: it starts at `degraded_trust_scale` (0.5
— wheels beat a diverged LIO) and **ramps with the distance dead reckoned**,

```
scale(d) = degraded_trust_scale + (hold_drift_k · d)²
```

the same quadratic form `pepper_odom_covariance` grows its own variance by, so
the two compose rather than duplicate: that node models drift since its last
reset, this one models having no exteroceptive correction at all right now. At
the defaults the scale crosses 1.0 at ~0.7 m — up to there the wheels are still
net *more* trustworthy than the rejected estimate; past it accumulated drift
dominates and the reported uncertainty inflates.

`d` is **path length, not displacement**: drift accrues with ground covered, so
a there-and-back excursion still counts. And the ramp is distance-driven, not
time-driven, because dead reckoning a stationary robot is exact — a long hold
that covered no ground has added no error. That is also why FAULT is not pinned
to the maximum; elapsed time is carried by the verdict and the diagnostic level
instead.

The second signal is a `/diagnostics` status carrying the verdict, held
distance, current scale, and the rolling rejection rate. A high rejection rate
convicts the guard's thresholds, not the estimator — an over-tight gate rejects
exactly the measurements that would correct it.

This is Tier 1 only: a hard physical bound, so no baseline, warm-up or tuning.
Statistical degeneracy detection (covariance spike, `effct_feat_num` collapse —
see `utils/lio_health.py`) is a separate tier, and must inform rather than reject
on its own, because those signals look healthy after a confidently-wrong relock.

**Known gap:** `pointlio_odometry.launch.py` reaches the bridge through
`point_lio`'s own launch file, which declares the `lio_odom_bridge` node itself
and does not forward these arguments. The guard defaults off there. Set the
parameters on that node directly, or add the passthrough in `point_lio`.

## 🚀 Running

```bash
# RTAB-Map, RealSense (add localization:=true to stop mapping, rviz:=true for RViz)
ros2 launch pepper_slam rtabmap_base.launch.py

# SLAM Toolbox, 2D LiDAR
ros2 launch pepper_slam slam_toolbox.launch.py

# Static sensor extrinsics (included by every bag-replay and FAST-LIO launch)
ros2 launch pepper_slam pepper_sensor_tf.launch.py
```

`pepper_sensor_tf.launch.py` exists because the older bags captured
`/tf_static` **empty** — without the rig mount and RealSense internals RTAB-Map
refuses to start. See *Recording bags* below for why, and for how to stop it
happening again.

It takes `scope:`:

| `scope` | publishes | use for |
|---------|-----------|---------|
| `mount` (default) | only `base_footprint→l2lidar_frame`, `l2lidar_frame→camera_camera_link`, `l2lidar_frame→l2lidar_frame_imu` | the real robot, and any bag recorded with `/tf_static` |
| `all` | the above **plus** the seven RealSense-internal edges | replaying `slam_recording*`, `slam_bench_run*` — the bags with no `/tf_static` |

`mount` is the default because `realsense2_camera` publishes its own internal
extrinsics, read off the device. Publishing them here too gives those edges two
publishers, and since `/tf_static` is latched and tf2 keys its buffer on the
child frame, **whichever message lands last silently wins** — which one that is
varies between launches.

## Recording bags

**Record `/tf_static` with a transient-local QoS override, or you will lose it.**

```bash
ros2 bag record -a \
  --qos-profile-overrides-path $(ros2 pkg prefix pepper_slam)/share/pepper_slam/config/record_qos.yaml \
  -o my_bag
```

`/tf_static` is published **once** and latched (`transient_local`).
`ros2 bag record` subscribes **volatile** by default, so it only receives
messages sent *after* it subscribes — capturing the transforms is a race
between the recorder and your launch files. That race is why:

| bag | `/tf_static` |
|-----|--------------|
| `lidar_cam_calib`, `lidar_cam_calib2`, `lidar_cam_calib3` | 4 msgs (won the race) |
| `slam_session_20260709_085909` | 4 msgs |
| `July_22` | 11 msgs |
| `slam_recording`, `slam_recording2`, `slam_bench_run1..3` | **0 msgs** |

The override makes the recorder subscribe transient-local, so it receives the
already-latched message regardless of start order. Verified: recorder started
8 s *after* the publisher still captured it.

**A bag recorded this way needs no `pepper_sensor_tf.launch.py` at all** — the
transforms come from the bag, exactly as published, with no hand-maintained
copy to drift from the device. `ros2 bag play` re-offers `/tf_static` with the
recorded durability, so late-joining subscribers (RViz, RTAB-Map) still get it.

Note `-a` records everything including 30 Hz colour; existing bags run 8–22 GB
for 3–8 minutes. Name topics explicitly for long sessions.

### Where the rig values come from

`base_footprint→l2lidar_frame` and the camera internals were recovered from
`bags/lidar_cam_calib2`. `l2lidar_frame→camera_camera_link` is **measured**
(tape measure, 2026-08-04) — its rotation is inherited and **not** measured.
See the header of `config/sensor_tf.yaml` for the full provenance, including
why the previous `direct_visual_lidar_calibration` value was replaced.

### Bag-replay experiments

Each wraps `rtabmap_base.launch.py` unchanged and writes to a throwaway
database, so recorded maps are never at risk. Playback commands are in each
file's header.

| Launch file | Sensor setup | Odometry source |
|-------------|--------------|-----------------|
| `bag_test/rtabmap_rgbd_wheel_bag.launch.py` | RealSense RGB-D (infra1 + depth) | bag TF (`pepper_odom`) |
| `bag_test/rtabmap_l2_bag.launch.py` | L2 lidar + IMU, no camera | RTAB-Map `icp_odometry` |
| `bag_test/rtabmap_fastlio_bag.launch.py` | L2 lidar + RGB for loop closure | FAST-LIO (best measured: 0.19 m closure) |
| `bag_test/rtabmap_fused_bag.launch.py` | L2 + RGB + aligned depth + fused IMU | `odom_source:=wheel` (default) or `icp` |

`rtabmap_fused_bag.launch.py` is the all-three-sensors variant: visual
bag-of-words proposes the loop closure, lidar ICP refines it, and
`imu_filter_madgwick` supplies the gravity constraint the orientation-less
`/camera/imu` can't. Wheel odometry is the default because it is by far the
smoothest measured (median step 0.35 cm vs 5.74 cm for `icp_odometry`) and its
drift is what loop closure corrects.

## 📁 Package Structure

Launch, params, and a few standalone TF/odometry helper scripts (`ament_cmake`, no compiled targets):

```
pepper_slam/
├── config/
│   ├── ekf_lio_wheel.yaml                    # robot_localization EKF (wheel + LIO fusion)
│   ├── mapper_params_online_async.yaml       # SLAM Toolbox parameters
│   ├── record_qos.yaml                       # QoS overrides for bag recording
│   └── sensor_tf.yaml                        # static sensor-rig transform provenance
├── launch/
│   ├── rtabmap_base.launch.py                 # vendored upstream; excluded from flake8
│   ├── slam_toolbox.launch.py
│   ├── pepper_sensor_tf.launch.py
│   ├── ekf_fusion.launch.py                   # robot_localization EKF, alternative to lio_odom_bridge
│   ├── fastlio_odometry.launch.py             # FAST-LIO odometry (no loop closure); wraps fast_lio's mapping.launch.py
│   ├── pointlio_odometry.launch.py            # same, for Point-LIO
│   ├── view_rig.launch.py                     # sensor rig visualization
│   └── bag_test/                              # manual bag-replay validation, not automated tests
│       ├── rtabmap_rgbd_wheel_bag.launch.py
│       ├── rtabmap_l2_bag.launch.py
│       ├── rtabmap_fastlio_bag.launch.py
│       └── rtabmap_fused_bag.launch.py
├── rviz/{rtabmap_fastlio_mapping,rtabmap_fused_mapping,view_rig}.rviz
│   (compute_lidar_camera_bridge.py moved to ros2_ws/utils/)
├── scripts/
│   ├── check_frame_contract.py                # asserts the LIO frame contract holds, whichever backend runs
│   ├── lio_odom_guard.py                      # divergence gate: reject implausible LIO poses, dead reckon on wheels
│   ├── leveled_odometry_publisher.py          # republishes LIO odometry rotated into the gravity-level odom frame
│   ├── pepper_odom_relabel.py                 # republishes /pepper_odom with frame_id overridden to odom
│   └── static_tf_publisher.py                 # publishes a whole static TF chain from one node
├── urdf/
│   ├── pepper_display_with_rig.urdf.xacro     # DISPLAY ONLY: Pepper body + sensor rig together
│   ├── pepper_sensor_rig.urdf.xacro           # sensor rig only, rooted for the real robot
│   └── sensor_rig.xacro                       # the one xacro macro definition of the L2 + D435i rig
├── ament_flake8.ini
├── CMakeLists.txt
├── FRAMES.md
└── package.xml
```

Saved maps and keepout masks live in `pepper_navigation/map/` (Nav2's map server
and costmap filters consume them); RTAB-Map `.db` files in `~/.ros/`, not
version-controlled.

## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
