# Test plan: the safety chain

Unlike `TEST_PLAN_map_odom.md`, this one is **not temporary**. It covers the
layers that stop the robot, which survive every branch merge. Re-run the robot
section after any change to the collision monitor, the safety cloud filter, the
costmap geometry or the driver.

Layers under test, outermost first. Each covers a failure the others
structurally cannot, so each is tested on its own:

| # | Layer | Where it runs |
|---|---|---|
| 1 | ALMemory heartbeat + `robot_watchdog.py` | on the robot |
| 2 | `cmd_vel_timeout`, speed caps | naoqi_driver |
| 3 | collision monitor (stop / slowdown polygons) | host, Nav2 |
| 4 | safety cloud filter (`points_safety_filter`) | host |
| 5 | costmap inflation + MPPI critics | host, Nav2 |
| 6 | localization watchdog | host |
| 7 | lifecycle gating (`wait_for_map_then_start`) | host |
| 8 | `bumper_guard` (post-contact directional gate) | host |

---

## A. Bag tests (no robot needed)

### A1. The side-body cut removes Pepper's own flanks

The L2 sees the robot's own body at 0.20-0.34 m, 0.23-0.45 m high, in two
lobes at +-30..90 deg. Without the cut the stop zone fires on the robot itself.

```bash
ros2 launch pepper_navigation costmap_bag.launch.py localize:=true rviz:=true
ros2 bag play ~/ros2_ws/bags/slam_20260823_aligned --clock --read-ahead-queue-size 2000
```

| Check | Expected |
|---|---|
| `ros2 topic hz /points_safety` | Publishing. **If silent, the QoS regressed to RELIABLE** -- the L2 publishes BEST_EFFORT |
| Point count `/points` vs `/points_safety` | Reduced, not emptied |
| In RViz, `/points_safety` around the base | The two flank lobes inside 0.35 m are gone; the forward cone is untouched at every range |
| Points inside the stop polygon, per scan | ~1 or fewer (was ~92 before the cut) |
| Scans that would trigger the stop | Under ~10% in a corridor (was ~72%) |
| `ros2 param get /points_safety_filter self_cut_range` | 0.35 |

Set `self_cut_range:=0.0` and replay to see the failure mode: the stop zone
fires almost continuously.

### A2. The forward cone is never cut

The one thing this filter must never do.

| Check | Expected |
|---|---|
| Walk toward the robot head-on in the replay | The approaching points survive at every range, including below 0.35 m |
| Approach from the side at >0.35 m | Survives (only CLOSE side returns are cut) |
| Approach from the side at <0.35 m | Cut. Accepted: ~6-10 cm from the skin, unreachable by a forward-driving diff base, and the slowdown wedge still covers it |

### A3. Costmap marks obstacles, not floor

| Check | Expected |
|---|---|
| Reflective floor stretch | No lethal cells on open floor. The local costmap uses the RealSense only; the L2 is in the config as an inert `l2_voxel` |
| An actual obstacle (chair, person) | Marked, and cleared again once it leaves |
| `scripts/costmap_explain.py` | Cells attributable to the expected sensor |

### A4. Geometry consistency (no robot, no bag)

```bash
colcon test --packages-select pepper_navigation pepper_slam
colcon test-result
```

| Check | Expected |
|---|---|
| `test_robot_geometry_identical_across_param_files` | Passes: radius, inflation and BOTH polygons identical in all six profiles |
| `test_cloud_filters` | Passes, including the `SideBodyKeep` cases |
| Whole workspace | 0 failures |

### Recording for future bag tests

`/laser`, `/sonar/front` and `/bumper` are **absent from every existing bag**
because all three are `"enabled": false` in naoqi's `boot_config.json`, so
B9-B11 below are robot-only by construction, not by choice. Add these to the
record list on the next session and they become bag-testable:

    /laser /sonar/front /bumper /joint_states

`/joint_states` is not optional for the bumper: it is `bumper_guard`'s liveness
witness, and without it every replay reads as blind and holds its latches.
A bag that carries `/tf` with the body tree also lets `sonar_tf_bridge` be
replayed; the sensor-rig launch alone does not publish `SonarFront_frame`
(see `pepper_sensor_tf.launch.py` -- it deliberately omits `pepper.urdf`).

---

## B. Robot tests

Run in order. **Have the emergency stop within reach for B1-B3**, since those
tests deliberately provoke uncommanded motion.

### B0. Prerequisites: turn the sensors on

All three ship disabled in naoqi's `boot_config.json`. Nothing below runs
until they report.

| Step | Expected |
|---|---|
| Set `"laser"`, `"sonar"`, `"bumper"` to `"enabled": true`, restart the driver | `/laser`, `/sonar/front`, `/sonar/back`, `/bumper` all present |
| `ros2 topic hz /laser` | ~10 Hz |
| Driver log | No `no laser board has advanced its frame counter` |
| Press each bumper by hand, `ros2 topic echo /bumper` | Three distinct ids: 0 RIGHT, 1 LEFT, 2 BACK. **The BACK event is behind a `robot_ == PEPPER` guard -- confirm it specifically** |

`/sonar/back` arrives whether or not it is wanted: `converters/sonar.cpp`
pushes both Front and Back keys for Pepper. It is simply not consumed.

### B1. The robot-side watchdog (layer 1)

The only layer that survives the host disappearing. ALMotion.move() holds the
last velocity forever, so this is the one that matters most.

Preconditions: `robot_watchdog.py` running **on the robot**, driver up.

| Step | Expected |
|---|---|
| `ros2 topic pub -r 10 /cmd_vel geometry_msgs/Twist '{linear: {x: 0.1}}'`, robot driving slowly | Moves |
| **Kill the driver with `kill -9`** (not Ctrl+C) while it drives | Base stops within ~1 s (`--timeout`), from the robot, not the host |
| Repeat but **unplug the network / drop Wi-Fi** | Same: stops within ~1 s |
| Stop the driver cleanly (Ctrl+C) | Stops immediately, and the watchdog does NOT act (the driver writes -1 on a clean exit) |
| No driver running at all | Watchdog idle; it does not interfere with other programs |

### B2. Driver velocity guards (layer 2)

| Step | Expected |
|---|---|
| Publish `/cmd_vel` at 10 Hz, then stop publishing | Base stops after `cmd_vel_timeout` (default 0.5 s), with `No /cmd_vel for the configured timeout, stopping the base` |
| Set `max_linear_speed: 0.2`, command 0.5 m/s | Robot moves at 0.2 m/s, not 0.5 |
| Set `max_angular_speed: 0.3`, command 1.0 rad/s | Clamped to 0.3 |

### B3. Collision monitor polygons (layer 3)

The numbers below come from the config; the point of the test is that the
measured distances match and that the robot's own body never triggers them.

```bash
ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py
ros2 topic echo /collision_monitor_state
```

| Step | Expected |
|---|---|
| Stand still in an open room, nobody near | State `DO_NOTHING`. **If it stops while alone, the side-body cut is not working** -- check A1 |
| Drive down a corridor | Mostly `DO_NOTHING`. Occasional `SLOWDOWN` near the walls is acceptable; constant stopping is not |
| Walk slowly toward the front of the robot | `SLOWDOWN` at ~0.90 m, `STOP` at ~0.40 m. Measure with a tape once |
| Approach from the side | `SLOWDOWN` at ~0.45 m. The stop zone sides are ~0.31 m |
| Put an obstacle 0.25 m from the flank, robot stationary | Expected NOT to stop (inside the cut). Confirm the robot cannot reach it by driving forward |
| Block the robot's path while it drives to a goal | Stops, holds, and resumes when cleared |
| Check the polygons in RViz (`polygon_stop`, `polygon_slowdown`) | Wedges, not circles; the stop wedge clears the 0.255 m body |

### B4. Costmap and planner margins (layer 5)

| Step | Expected |
|---|---|
| Costmap startup log | No "inflation radius is smaller than the inscribed radius" warning |
| RViz footprint vs the real base | The 0.25 m circle covers Pepper's base |
| Narrowest doorway on the route | Passable at ~0.70 m or wider. Note the width if the planner refuses |
| Normal route | Reaches the goal; MPPI does not oscillate near walls (that would mean the inflation band is too thin to steer on) |

### B5. Localization watchdog (layer 6)

| Step | Expected |
|---|---|
| Startup log | `watching [...pose lock..., ...map -> odom gate...]` |
| Normal run | Pose lock OK; no cancellations |
| Cover the lidar / drive into a large unmapped space for >5 s | `localization LOST (...unhealthy for 5s) -- holding navigation`; the active goal is cancelled |
| Issue a new goal while still lost | Cancelled too (it re-cancels every 3 s) |
| Uncover and let it recover | `localization healthy again`; goals accepted |
| Kill the localizer outright | After `stale_timeout` (10 s) the watchdog treats silence as lost |
| During all of the above | **`cmd_vel` is never touched by the watchdog** -- only the collision monitor stops the wheels |

### B6. Lifecycle gating (layer 7)

| Step | Expected |
|---|---|
| Launch fastloc with the robot somewhere unmapped so the lock is slow | Nav2 servers stay inactive; no costmap errors spinning in the log |
| RViz while searching | The prior map draws (the static `map -> lio_init` anchor exists) but the robot is NOT drawn at the origin |
| Once it locks | `wait_for_map_then_start` calls startup; planner/controller/bt_navigator activate |

### B7. Teleop dead-man switch

```bash
ros2 launch naoqi_driver safe_teleop.launch.py
```

| Step | Expected |
|---|---|
| Push the stick without holding enable | No motion |
| Hold enable and push | Moves, capped at 0.2 m/s / 0.5 rad/s |
| Release enable mid-motion | Stops immediately |
| Drive toward a wall with enable held | Collision monitor slows then stops it anyway |

### B8. End-to-end smoke

| Step | Expected |
|---|---|
| Full route with a person crossing the path | Slows, stops if needed, resumes, reaches the goal |
| Repeat 3 times | No spurious stops on open floor; no missed stops |

### B9. Bumper gate (layer 8)

The one test that cannot be inferred from any other: it proves the robot can
get **off** an obstacle. With `vx_min: 0.0` nothing but `BackUp` emits reverse,
so if reverse does not survive a front contact, a bump is a permanent deadlock.

```bash
ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py
ros2 topic echo /diagnostics --field status    # bumper_guard: contact
```

| Step | Expected |
|---|---|
| `ros2 topic info /cmd_vel` and `/cmd_vel_safe` | `cmd_vel` published by `bumper_guard`, `cmd_vel_safe` by `collision_monitor`. If `collision_monitor` still owns `cmd_vel`, the gate is not in the chain |
| Press the left bumper, robot idle | Status WARN `contact front`; `left=true`, `block_forward=true` |
| Command +0.2 m/s while pressed | No motion; log `bumper latched: vx 0.20 -> 0.00` |
| Command -0.1 m/s while pressed | **Robot reverses.** If it does not, `BackUp` can never free it |
| Command rotation while pressed | Robot turns -- rotation is never gated |
| Release, command forward within 1 s | Still blocked (`clear_delay`) |
| Release, wait > 1 s, command forward | Moves |
| Press the back bumper, command reverse, then forward | Reverse blocked, forward moves |
| Hold a bumper > 10 s | Status ERROR `bumper held down -- stuck switch or debris` |
| `ros2 service call /bumper_guard/clear std_srvs/srv/Trigger` while pressed | Refused: `a bumper is still pressed` |
| Same call after release | `bumper latches cleared`; forward allowed at once |
| Kill `bumper_guard` while driving | **Robot stops** -- nothing publishes `cmd_vel`, the driver times out |
| Stop the driver with a latch set | Latch holds; status WARN `no joint_states` |
| Drive slowly into a soft obstacle | Gate blocks forward on contact; a `BackUp` recovery frees the robot |

### B10. Pepper laser as a monitor source (layer 3)

Three base boards, one `/laser` scan: 61 slots, 45 live beams, two 8-slot blind
gaps. Already in `base_footprint`, so no TF work. **Measure before enabling** --
the boards can read the floor, and 4 floor returns past `max_points: 3` is a
permanent stop. This is the same failure class as `self_cut_range: 0.35`.

| Step | Expected |
|---|---|
| `ros2 topic echo /laser --once` | 61 ranges; the two blind gaps are NaN; **no value near 5.40 or 7.09** -- those are the saturation sentinels `encodeRange` exists to suppress (the right bank has been seen to emit 7.09) |
| Stand still in an open room, source still `enabled: False`. Count beams < 0.40 m | **Must be <= 3.** More than that and enabling the source stops the robot where it stands |
| Drive over a threshold, up to a ramp, onto dark or glossy floor | Count beams < 0.40 m in each. Record the worst case |
| Flip `laser.enabled: True`, re-run **all of B3** | Stop rate must not rise materially. Standing alone in an open room is still `DO_NOTHING` |
| Approach the flank at 0.30 m (inside the L2 self-cut) | Now detected -- this is what the laser buys that the L2 cannot |

### B11. Front sonar as a monitor source (layer 3)

Needs `sonar_tf_bridge`: the real chain is
`base_footprint -> base_link -> torso -> Hip -> Pelvis -> Tibia -> SonarFront_frame`,
and `tf_nav_relay` drops it at `base_link`, so without the bridge the source is
silently inert. Three of those joints are revolute, which is why a
`static_transform_publisher` is not a substitute.

| Step | Expected |
|---|---|
| Bridge log | `bridging ['SonarFront_frame'] -> base_footprint`, and no repeating `no TF` warning |
| `ros2 topic echo /tf_nav` | Exactly one `base_footprint -> SonarFront_frame` edge at ~20 Hz |
| Compare with `ros2 run tf2_ros tf2_echo base_footprint SonarFront_frame` (full `/tf`) | Same translation and rotation |
| Change the hip/knee posture | The bridged value tracks it. A static transform would not -- MEASURED 8.5 cm of z error on a synthetic chain |
| Kill the bridge with the source enabled | **Record what the monitor does.** See the stale-source test below -- this is the same question |
| Flip `sonar_front.enabled: True`. Stand still, open room | `DO_NOTHING`. A sonar reading inside 0.40 m from nothing means it is unusable as a safety source |
| Walk toward the front | `SLOWDOWN` near 0.90 m |
| Approach closer than 0.25 m | The sonar reads nothing -- its dead zone sits **inside** the 0.40 m stop wedge, so the L2 forward cone must be what stops the robot. Confirm it still does |
| Cover the sonar with a hand | Reads minimum, `STOP`, and recovers when uncovered |

### B12. What a stale source actually does (layer 3)

**Unverified assumption, and safety-relevant.** `source_timeout: 1.0` is
documented as the window in which data counts as valid, but Humble's
`collision_monitor` does not ship its source, and the installed header does not
say whether an invalid source causes a stop or is silently skipped. Everything
written elsewhere about the chain "failing safe on a dead lidar" rests on this.

| Step | Expected |
|---|---|
| While driving, stop `points_safety_filter` | **Record the truth.** Either `STOP` within ~1.0 s (fails safe) or continued motion with only a log warning (fails open) |
| Same with the laser or sonar source enabled and only *that* source stopped | Does one dead source of three stop the robot, or do the survivors carry it? |

If it fails open, the chain needs a source-liveness watchdog and every
"fails safe" claim in `pepper4dec_system_overview.txt` needs correcting.

---

## What a failure means

| Symptom | Most likely layer |
|---|---|
| Robot stops constantly on open floor | 4 -- side-body cut off or `self_cut_range` 0 |
| Robot never stops for obstacles | 3 -- `/points_safety` silent (QoS!), or the monitor is not in the `cmd_vel` chain |
| Robot keeps driving after the driver dies | 1 -- `robot_watchdog.py` not running on the robot |
| Robot drives confidently to the wrong place | 6 -- the pose lock is wrong and the watchdog is not acting |
| Nav2 never activates | 7 -- the lock never happened; check the localizer, not Nav2 |
| MPPI oscillates near walls | 5 -- inflation band too thin relative to the inscribed radius |
| Robot stops dead on contact and never recovers | 8 -- reverse is being gated too; `BackUp` cannot escape |
| Robot will not move and nothing is in the way | 8 -- a stuck bumper latch; check `bumper_guard: contact` for ERROR |
| Robot stops whenever the driver hiccups | 8 -- `/joint_states` gaps are reading as blind |
| Stops constantly after enabling the laser | 3 -- the boards are reading the floor; re-measure B10 |
| Laser or sonar source has no effect at all | 3 -- source `enabled: False`, or for the sonar the bridge is not running |
