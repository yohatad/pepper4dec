<div align="center">
<h1>Pepper Social Navigation</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Pepper Social Navigation** package makes Pepper navigate *around people*
rather than around obstacles that happen to be people. It contributes three
things to the existing Nav2 stack: a multi-camera world-frame people tracker, a
proxemic costmap layer, and a social force model embedded as an MPPI critic.

## ✨ Key Features
- **World-frame people tracking** fusing Pepper's head camera and the body-mounted
  RealSense into one set of metric tracks with velocity
- **Range-split cameras**: the head camera works close (0.4-3.5 m), the RealSense
  works long (2.0-8.0 m), with a deliberate handover band between them
- **Three-stage ranging** (depth, feet, head plane) with **anisotropic** noise, so a
  depthless camera still contributes the bearing it measures well
- **Social force model as an MPPI cost**, using the repulsive *potential* the SFM
  force is the gradient of — not an ad-hoc penalty
- **Constant-velocity anticipation**, which removes plain SFM's worst weakness
- **Audience/bystander split**, so the tour group following Pepper does not repel it
- **Offboard-ready**: perception runs on a laptop, control stays on the robot
- **Measurable**: a metrics node that produces numbers, not impressions

## ✅ Prerequisites

```bash
sudo apt install ros-humble-social-nav-msgs ros-humble-nav2-mppi-controller
```

`social_nav_msgs` is **not** pulled in by nav2 and is the people interface this
package publishes and both plugins consume. Also required in the workspace:
`pepper_navigation`, `pepper_slam`, `person_detection`, `dec_interfaces`,
`image_decompressor`.

```bash
cd ~/ros2_ws
colcon build --packages-select pepper_social_nav
source install/setup.bash
```

## 🧠 Design rationale

### Why a tracker at all, and why in the world frame

`person_detection` publishes **pixels**: `PersonDetection.msg` carries `u`, `v`
and an optional depth, with ByteTrack IDs associated in *image* space. A social
force model needs metric positions and **velocities** in a world frame. Nothing
in the repo produced those, which is why this package leads with the tracker.

Associating in metres rather than pixels also removes the failure mode that
would otherwise dominate: `overt_attention` pans the head continuously, so a
fast pan moves every bounding box far enough that IoU matching fails, the track
is reborn with a new ID, and its velocity resets to **zero** — precisely when
the robot is closest to someone. In the world frame TF factors the head
rotation out and the person's position is continuous through the pan.

### Why there is no person re-identification

ReID answers "is this the same person as before". SFM never asks: it is
memoryless, and computes forces from current positions and velocities. Two
pedestrians swapping IDs produces identical forces.

The real motivation for ReID here would be ID churn from head motion — and
world-frame tracking fixes that at the root, rather than patching it with an
appearance model. ReID would also cost a second ONNX model running a crop per
detection, and its embeddings degrade exactly where this deployment needs them:
across the viewpoint flip that happens every time Pepper turns to lead the tour.

### Why two cameras, split by range

Neither is sufficient alone, and their weaknesses are opposites — so each is
trusted over the band it is actually good at.

**`front` — close work, 0.4–3.5 m.** Pepper's `CameraTop` sits at ~1.15 m
(0.333 m above `base_link` per `pepper.urdf`), 55.2°×44.3°, VGA 640×480, panning
with the head. At conversational range it frames a whole person. It fades with
distance because VGA runs out of pixels on a small target.

**`realsense` — long work, 2.0–8.0 m.** Better sensor and optics, so it carries
distance. Computed from `pepper_slam/urdf/sensor_rig.xacro` it sits at
**z = 0.308 m with the optical axis pitched *down* 2.31°**, so with the D435's
42.5° colour FOV the tallest thing it can see is `0.308 + 0.343 × range`:

| Range | Sees up to | |
|---|---|---|
| 1.0 m | 0.65 m | mid-thigh |
| 2.0 m | 0.99 m | waist |
| 3.0 m | 1.34 m | chest |
| ~4.5 m | 1.85 m | first full body |

Its framing *improves* with range exactly where the front camera's degrades.
Below ~2 m it is looking at legs, which is where the front camera takes over.

The 2.0–3.5 m overlap is deliberate: it is the handover band, and because
association happens in metres the handover is a continuation of one track,
costing neither the ID nor the velocity estimate.

### Why ranging has three methods, and why one is deliberately weak

Ranging degrades gracefully rather than failing:

1. **Depth**, where the camera has it. The box centroid lands on torso or legs —
   solid geometry — and the RealSense's depth is best inside its band.
2. **Ground plane through the feet.** Intersects the ray through the box's
   bottom-centre pixel with the floor. Needs no depth and puts the estimate
   where the person actually stands. Requires the box bottom to be clear of the
   image border, or the feet are cropped and the ray lands past the person.
3. **Head plane through the top of the box.** The close-range case for the front
   camera.

Method 3 exists because of a hard geometric fact: **with the head level, the
front camera's lower frame edge does not reach the floor until 2.82 m**, and
`overt_attention`'s `default_pitch: -0.2` rad tilts it *up*, pushing that to
~6 m. So in exactly the close band this camera is assigned, the feet are never
in frame and method 2 cannot run.

Intersecting with `z = person_height` works there, but the range it yields is
weak: the camera is only ~0.55 m below a standing head, so ordinary height
variation (1.55–1.90 m) swings range by ~20%. That is not a reason to discard
the detection — the **bearing is still excellent**.

So measurement noise is **anisotropic**: built along the line of sight
(`range_stddev`) and across it (`cross_stddev`), then rotated into the tracking
frame. A head-plane detection enters as effectively **bearing-only** — large
range sigma, small cross sigma — and the filter sharpens direction while leaving
distance to depth-bearing sources and the motion model. An isotropic sigma would
have to be as bad as the worst axis, throwing the good half away.

### Why the SFM lives in the controller

The repulsive force `f(d) = A·exp((r−d)/B)` is the negative gradient of the
potential `V(d) = A·B·exp((r−d)/B)`. Scoring MPPI trajectories with **V** makes
the optimum MPPI settles on consistent with SFM dynamics, while keeping MPPI's
constraint handling, obstacle awareness and goal seeking. A separate SFM
velocity node bolted onto Nav2 would have to reimplement all of that.

It could not have gone into DWB: that stack weights `BaseObstacle.scale` at
`0.02` against `PathDist.scale` `24.0`, so any social cost fed to DWB would be
arithmetically ignored.

### Why the tour group is excluded

`behavior_controller/data/dec_Tour.xml` has Pepper say *"follow me"*. The
visitors are **behind the robot by design**. Vanilla SFM reads that cluster as a
large repulsive force and the robot accelerates away from its own audience. The
tracker therefore publishes two topics: `~/pedestrians` (everyone, for metrics
and logging) and `~/bystanders` (the avoid set, what both plugins consume).
Someone close and behind the robot while `/behavior_controller/tour_active` is
true is audience. With no tour running, everyone is a bystander.

## 🚀 Running

### On the robot

```bash
ros2 launch pepper_social_nav pepper_nav2_social.launch.py
```

Same localization, maps and frames as `pepper_nav2_fastlio_loc.launch.py`. It
does **not** start any camera or detector — perception is offboard.

### On the laptop

```bash
ros2 launch pepper_social_nav perception_offboard.launch.py
```

Runs decompression, two `person_detection` instances and the tracker. Only
compressed images go robot→laptop; only `Pedestrians` (a few hundred bytes at
15 Hz) comes back.

Both machines need the same `ROS_DOMAIN_ID`, the same DDS vendor and the same
subnet, and `/tf`, `/tf_static` and `/joint_states` must reach the laptop — the
tracker cannot deproject without `map → camera` at the image timestamp.

To run everything on the robot instead:

```bash
ros2 launch pepper_social_nav pepper_nav2_social.launch.py run_tracker:=true
ros2 launch pepper_social_nav perception_offboard.launch.py tracker:=false
```

### Latency is handled, do not "simplify" it

Compress → WiFi → decompress → YOLO is easily 100–300 ms. The tracker filters at
the **measurement** timestamp (taken from `CameraInfo`, not arrival time) and
predicts forward to now; the SFM critic starts its extrapolation at the
message's actual age. Stamping anything with `now()` silently reintroduces the
error the whole design avoids.

## 🖥️ ROS Interface

### `people_tracker`

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/person_detection/*/data` | `dec_interfaces/PersonDetection` | in | Per-camera detections |
| `/camera/*/camera_info` | `sensor_msgs/CameraInfo` | in | Intrinsics, frame and **timestamp** |
| `/behavior_controller/tour_active` | `std_msgs/Bool` | in | Enables the audience split |
| `~/pedestrians` | `social_nav_msgs/Pedestrians` | out | All tracks |
| `~/bystanders` | `social_nav_msgs/Pedestrians` | out | Avoid set (excludes the tour group) |
| `~/markers` | `visualization_msgs/MarkerArray` | out | RViz bodies, velocity arrows, labels |

### Plugins

| Plugin | Loaded by | Consumes |
|---|---|---|
| `pepper_social_nav::ProxemicLayer` | `nav2_costmap_2d` | `~/bystanders` |
| `pepper_social_nav::SocialForceCritic` | `nav2_mppi_controller` | `~/bystanders` |

Both **fail open**: if the people topic goes stale past `message_timeout`, they
contribute no cost rather than steering on stale data. The on-robot,
lidar-driven collision monitor remains authoritative regardless of the link.

## 🧪 Validation order

The steps are ordered so a regression is attributable. Do not collapse them.

**0. Record a people bag.** The existing recordings are SLAM runs with people
only by accident. `scripts/record_people_bag.sh` lists the seven cases to
capture (crossing, approaching, standing, two people crossing paths, leaving
and returning, a head pan across a person, and range steps at 1/2/3 m).

**1. Tracker, on that bag.** Confirm a walking person reports ~1 m/s, a standing
person ~0, and that a head pan does not break the track. Run
`scripts/social_metrics.py` from here on.

**2. Safety fix.** See below — this must precede any close-range human testing.

**3. MPPI + holonomic at parity, social terms OFF.** Set
`SocialForceCritic.cost_weight: 0.0` and `proxemic_layer.enabled: false`, and
confirm the robot drives its tour as well as the DWB stack did. Switching
controller *and* adding social costs at once makes a regression unattributable.

**4. Enable the proxemic layer.** The global plan should now route around groups.

**5. Enable the SFM critic.** Watch `/controller_server/trajectories` in RViz:
the candidate bundle should bend away from a person before the base moves.

## ⚠️ The safety fix, and what still needs measuring

`nav2_params_fastlio_loc.yaml` sets `PolygonStop` to 0.40 m, but its only source
is `/points_safety`, built with `cloud_range_filter` `min_range=0.8` — radial,
from the lidar. With the L2 at z = 0.2582 and the monitor's `min_height: 0.10`,
the closest surviving point is `sqrt(0.8² − 0.16²) ≈ 0.78 m` horizontally.
**Nothing can ever fall inside 0.40 m**, so that stop zone has never fired.

`cloud_range_filter.py` now takes a geometric `self_filter_box` (off by default,
so the existing stacks are unchanged), which removes Pepper's body by shape
instead of by radius. `pepper_nav2_social.launch.py` uses it with
`min_range=0.30`, and both polygons then see real points.

> **The box is a starting point from nominal dimensions, not a measurement.**
> Pepper's base is ~0.48 m across, but the arms and tablet extend further and
> the measured self-hit shell reached ~0.6 m from the lidar. Verify before
> driving: view `/points_safety` in RViz and confirm no returns sit on the
> robot. If the robot freezes in place, the box is leaking self-hits — widen
> it, or set `min_range` back to 0.8 and accept the dead zone until it is tuned.

## 📊 Measuring it

```bash
ros2 run pepper_social_nav social_metrics.py --ros-args -p label:=baseline
ros2 run pepper_social_nav social_metrics.py --ros-args -p label:=social
```

Same route, both stacks. Writes `~/.ros/social_metrics_<label>.json`. Expect the
social stack to improve `min_distance_m` and cut `personal_zone_s` and
`collision_monitor_interventions`, at some cost in `path_length_m`. Watch
`stopped_s`: a robot that achieves great proxemics by never moving is not better.

## 📁 Package Structure

```
pepper_social_nav/
├── config/
│   ├── people_tracker.yaml          # camera sources, filter and audience tuning
│   └── nav2_params_social.yaml      # MPPI + holonomic + proxemic + SFM critic
├── include/pepper_social_nav/
│   ├── people_tracker.hpp           # design rationale lives here
│   ├── proxemic_layer.hpp
│   └── social_force_critic.hpp
├── src/
│   ├── people_tracker.cpp           # deprojection, KF, association, audience split
│   ├── people_tracker_node.cpp
│   ├── proxemic_layer.cpp           # costmap_2d plugin
│   └── social_force_critic.cpp      # mppi critic plugin
├── launch/
│   ├── pepper_nav2_social.launch.py # ROBOT: nav2 + safety fix + optional tracker
│   └── perception_offboard.launch.py# LAPTOP: decompress + 2x detect + track
├── rviz/social_nav.rviz
├── scripts/
│   ├── record_people_bag.sh         # the validation bag, and what to capture
│   └── social_metrics.py            # proxemic + efficiency numbers
└── test/test_people_tracker_geometry.cpp
```

`nav2_params_social.yaml` lives here rather than in `pepper_navigation` because
it names `pepper_social_nav::ProxemicLayer`; keeping it there would make
`pepper_navigation` implicitly depend on this package. The dependency is
one-way: `pepper_navigation` never depends on `pepper_social_nav`.

It is also deliberately outside `test_shared_nav2_params.py`'s file list — that
test asserts `controller_server` is byte-identical across the four DWB profiles,
and this file replaces DWB with MPPI on purpose.

## 💡 Support

- Issues: [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License.
