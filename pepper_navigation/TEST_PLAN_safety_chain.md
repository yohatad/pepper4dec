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
| 8 | `bumper_guard` (directional gate + sensor liveness) | host |
| 9 | `naoqi_driver` proximity backstop (laser + sonar + bumper) | naoqi_driver |

---

## Status board

`DONE` verified by running it. `STALE` passed before a change that invalidates
it. `TODO` never run.

| Test | Status | Evidence |
|---|---|---|
| A1 side-body cut | **DONE** 2026-10-19 | found `self_cut_range: 0.35` wrong; at 0.40, 1019 scans gave 0.1 in-zone pts/scan and 0.8% would-stop -- but see A12: the whole-bag stop rate is **6.60%**, and that 0.8% was a window |
| A2 forward cone never cut | **DONE** | 680 stamp-matched pairs, zero forward-cone points lost |
| A3 costmap marks obstacles not floor | **DONE** | 140 updates on a healthy lock, 0/140 lethal under the footprint |
| A4 geometry consistency | **DONE** | `test_shared_nav2_params`, runs in CI |
| A5-A7 laser/sonar as Nav2 sources | **SUPERSEDED** | They worked (STOP 98% / SLOWDOWN 100%, bridge exact to 0.000 mm over 1500 samples) but the sensors moved to `naoqi_driver`'s backstop, so the sources and `sonar_tf_bridge` were removed. Nav2's monitor keeps the L2 alone |
| A8 gate in the real chain | **DONE** 2026-10-19 | pass-through exact when idle; forward clamped to 0 under latch; reverse survives; latch timing correct |
| A9 TF load of the bridge | **DONE** 2026-10-19 | `/tf` 3409 transforms/s over 83 edges vs `/tf_nav` 20/s over 1 |
| A10 launch topology | **DONE** 2026-10-19 | `fastloc` wires `cmd_vel_raw -> monitor -> cmd_vel_safe -> gate -> cmd_vel`; `bumper_guard:=false` falls back correctly |
| B12 stale-source behaviour | **DONE, FAILED, FIXED** 2026-10-19 | held last verdict, 161 full-speed passes; `required_topics` now clamps in 1.05 s |
| B0 sensors enabled | **PART DONE** 2026-10-19 | enabled in `boot_config.json` and the four topics verified live (laser 9.7-10.0 Hz, both sonars 10.0 Hz). Bumper EVENTS still unconfirmed -- nothing pressed while watching |
| B1-B10 robot tests | **TODO** | need the robot. B7 (teleop dead-man) removed as not wanted; B11 folded into B10 |
| J1 L2 rate / stamp lag | **DONE** 2026-10-19 | 11.3 Hz, 4923 pts/cloud, stamp lag +102.4 ms live, confirming the bag's 98.8 ms |
| J8 L2 extrinsic | **DONE** 2026-10-19 | `[0.133, 0.000, 0.258]` with a non-identity rotation; resolves on the Jetson once `pepper_sensor_tf` is up |
| J4, J9, J10, J11 L2 on the Jetson | **DONE** 2026-10-19 | floor, bandwidth, QoS, near-range floor -- see the J table |
| J5 detection at the stop boundary | **DONE** 2026-10-19 | the stop zone is **not blind**: STOP on 100% of scans from against the base out to 0.445 m. It fires **early** (ghost points + a ~5 cm short reading), fixed for ghosts by `max_points: 10` -- see the J table |
| J3 self-hit shell | **PART DONE** 2026-10-19 | arms at rest: no 0.36 m shell; body at 0.11-0.25 m. Arms-moving run outstanding |
| J6, J7 L2 blind spots | **TODO** | need an object placed |
| B13 stopping distance | **PART DONE** 2026-10-19 | Latency measured on the bag: 135 ms software (98.8 ms of it the lidar sweep), so 4.7 cm of the 15 cm clearance goes before braking and the hardware must deliver **a >= 0.60 m/s²**. 0.90 m has ~5x margin and is not the constraint. Outstanding: the static detection checks, and `a` from an open-floor stop -- neither needs driving at an obstacle |
| A12 full-bag monitor vs geometry | **DONE** 2026-10-19 | **99.85% exact, and 0 missed STOPs** (377/377 matched). Whole-bag geometry rates over 21526 scans: **stop 6.60%, slowdown 30.18%** |
| A11 slow-zone body-vs-wall split | **DONE** 2026-10-19 | **All environment, no body component.** With the shipped cut, zero robot-fixed polar bins at even a 10% threshold; the most persistent appears in 29/1133 scans (2.6%) |

**Only `fastloc` is wired.** `amcl`, `kissicp`, `pointloc` and `rtabmap_loc`
have the config but not the launch wiring, so they have no gate and **no
`required_topics`** -- the B12 fail-open is still live in those four.

### Seeing what the safety filter does

Layers 3-4 are easiest to judge by eye. `safety_cut_viz` colours every L2 point
by the stage that removes it, or the polygon it reaches, and prints the
verdict the monitor would give. It is read-only (publishes no velocity) and
every threshold is a live parameter:

```bash
ros2 launch pepper_navigation safety_cut_viz.launch.py sensor_tf:=true   # robot, no Nav2
ros2 launch pepper_navigation safety_cut_viz.launch.py                   # Nav2 already up
ros2 param set /safety_cut_viz self_cut_range 0.25                       # try a value
```

Red = in the stop wedge, yellow = slowdown wedge, blue = `min_range`,
magenta = side-body cut, orange = body box, grey = outside the 0.22-1.80 m
height band, green = reaches the monitor outside both wedges. Grey and green
start hidden; every cloud keeps 10 s of scans so fixed things build up.

**Humble's collision monitor publishes no state topic** -- `state_topic` in the
yaml is ignored (measured in test 1a). Read its verdict from the viz text,
or from its output: `ros2 topic echo /cmd_vel` shows 0 for STOP and 0.3x
the command for SLOWDOWN.

Current filter values (`points_safety_filter`, all Nav2 launches), measured
2026-10-19 on the robot, arms at rest:

| Parameter | Value | Why |
|---|---|---|
| `min_range` | 0.12 m from the lidar | 0.22 hid a box ~0.10 m from the lidar entirely; no in-band body returns lie between 0.12 and 0.22 |
| `self_cut_range` / `self_cut_angle` | 0.30 m / 25 deg | no cut: 65% of scans false-STOP; 0.30-0.40: 0%; 0.25: 0% but max 3 pts (trigger > 3) |
| `body_boxes` | x 0.10-0.22, \|y\| <= 0.13, z 0.30-1.00 | Pepper's front face, straight ahead, so no range cut can take it without blinding the forward cone |
| `max_points` (both polygons, in the yaml) | 10 (was 3) | J5: ghost points along the line of sight peak at 8/scan and STOPped a box at 0.635 m on 25% of scans; real objects give >= 50 in the stop wedge. A12's rates were taken at 3 |

---

## A. Bag tests (no robot needed)

### A1. The side-body cut removes Pepper's own flanks

**Status: DONE** 2026-10-19 -- this is where the `self_cut_range` bug was found,
and it was re-measured live the same day; the values below are the live ones.

The L2 sees the robot's own body beside it 0.11-0.25 m from the lidar (x
0.18-0.22, |y| <= 0.20, z 0.22-0.28, bearings 55-80 deg), and Pepper's front
face straight ahead (x ~0.18-0.22, z 0.30-0.90). Without the cuts the stop
zone fires on the robot itself. (Re-measured 2026-10-19; the earlier figure,
a shell at 0.355-0.368 m, no longer holds.)

`costmap_bag.launch.py` does not run the safety filter; use the collision
monitor bag rig, which does, plus the viz:

```bash
ros2 launch pepper_navigation collision_monitor_bag.launch.py out_prefix:=$HOME/cm_a1
ros2 launch pepper_navigation safety_cut_viz.launch.py   # rig TF comes from the bag launch
ros2 bag play ~/ros2_ws/bags/slam_20260823_aligned --clock --read-ahead-queue-size 2000
```

| Check | Expected |
|---|---|
| `ros2 topic hz /points_safety` | Publishing. **If silent, the QoS regressed to RELIABLE** -- the L2 publishes BEST_EFFORT |
| Point count `/points` vs `/points_safety` | Reduced, not emptied |
| In RViz (safety_cut_viz) around the base | The flanks inside 0.30 m are magenta, the front face orange; nothing red with nobody near |
| Points inside the stop polygon, per scan | ~1 or fewer with nobody near |
| Scans that would trigger the stop | Under ~10% in a corridor |
| `ros2 param get /points_safety_filter self_cut_range` | 0.30 |
| `ros2 param get /points_safety_filter body_boxes` | `[0.10, 0.22, -0.13, 0.13, 0.30, 1.00]` |

`ros2 param set /safety_cut_viz self_cut_range 0.0` shows the failure mode
without a relaunch: red appears beside the lidar on most scans.

### A2. The forward cone is never cut

**Status: DONE** -- 680 stamp-matched pairs, zero forward-cone points lost.

The one thing this filter must never do.

| Check | Expected |
|---|---|
| Walk toward the robot head-on in the replay | The approaching points survive at every range down to `min_range` (0.12 m from the lidar) |
| Approach from the side at >0.30 m from the lidar | Survives (only CLOSE side returns are cut) |
| Approach from the side at <0.30 m from the lidar | Cut. Accepted: a few cm from the skin, unreachable by a forward-driving diff base, and the slowdown wedge still covers it |
| Anything inside the orange body box (x <= 0.22) | Cut. The box stays inside robot_radius 0.25, so only the body itself can be there |

### A3. Costmap marks obstacles, not floor

**Status: DONE** -- 0/140 costmap updates lethal under the footprint.

| Check | Expected |
|---|---|
| Reflective floor stretch | No lethal cells on open floor. The local costmap uses the RealSense only; the L2 is in the config as an inert `l2_voxel` |
| An actual obstacle (chair, person) | Marked, and cleared again once it leaves |
| `scripts/costmap_explain.py` | Cells attributable to the expected sensor |

### A4. Geometry consistency (no robot, no bag)

**Status: DONE** -- runs as `test_shared_nav2_params` in the package test suite.

```bash
colcon test --packages-select pepper_navigation pepper_slam
colcon test-result
```

| Check | Expected |
|---|---|
| `test_robot_geometry_identical_across_param_files` | Passes: radius, inflation and BOTH polygons identical in all six profiles |
| `test_cloud_filters` | Passes, including the `SideBodyKeep` and `BodyBoxKeep` cases |
| Whole workspace | 0 failures |

### A11. Slow zone: body or environment?

**Answer: environment.** The slow zone fires because of walls and furniture,
not because Pepper sees itself. Nothing to fix -- it is the cost of a 0.90 m
slow wedge in a corridor.

Method: a body return sits at a FIXED range and bearing from the sensor scan
after scan, because it is rigidly attached; an environment return moves as the
robot drives. So bin in-zone points into polar cells (2 cm x 2 deg) about the
sensor and count how many distinct scans each cell appears in.

| Run | slow | stop | robot-fixed bins (>=10% of scans) |
|---|---|---|---|
| shipped (`self_cut_range: 0.40`) | 20.6% | 1.3% | **0** -- most persistent bin 29/1133 scans |
| negative control (`self_cut_range: 0`) | 92.4% | 81.1% | **313**, at 0.14-0.50 m from the sensor, 36.7% of all in-zone points |

The control matters: a method that finds nothing has to be shown capable of
finding something. With the cut off it correctly recovers Pepper's own shell at
exactly the expected range.

**Use polar bins, not Cartesian ones.** 5 cm Cartesian cells found zero
robot-fixed cells even in the control, where the stop rate was 57-81% and
therefore obviously full of body returns. A spinning lidar samples fixed
azimuths, so a rigid return lands in a stable (range, bearing) cell but
scatters across neighbouring x/y cells.

**Trigger rates are strongly window-dependent.** The same shipped config gave a
1.3% stop rate over one 100 s window and 7.2% over a 150 s one. Quote
whole-bag numbers, not slices.

### A12. The real monitor vs plain geometry, whole bag

Nav2's actual `collision_monitor` against an independent implementation of the
same polygons, height band and `max_points`, scan by scan, over all 1946 s of
`slam_20260823_aligned`.

| | |
|---|---|
| exact agreement | **99.85%** (4524/4531 on the paired window) |
| **geometry said STOP, monitor did not** | **0** -- 377 of 377 matched |
| disagreements | 7, all benign: 2 SLOWDOWN->CLEAR and 5 CLEAR->SLOWDOWN, boundary scans a tick early or late |
| whole-bag geometry rates (21526 scans) | **stop 6.60%, slowdown 30.18%** |

This is what validates every polygon number in this plan: until now they were
all computed by our own point-in-polygon code and never by the node that
enforces them. Independently reproduced elsewhere at 99.4% exact / 99.58%
within one scan.

**CORRECTION: the stop rate is 6.60%, not 0.8%.** The 0.8% figure quoted while
tuning `self_cut_range` came from a 1019-scan window. The whole-bag figure is
**6.60%**, about eight times higher. The slowdown figure held up exactly
(30.18% vs the 30.2% quoted). Combined with A11 -- zero robot-fixed bins, so
every in-zone point is environment -- that 6.60% is real obstacles on a drive
that passed close to things, not false triggering. But it is the number to
quote, and window figures are not.

**Pair each cloud with the NEXT monitor verdict, not the last one seen.** The
first run of this test reported 59.71% agreement, which was entirely a pairing
bug: the verdict for a cloud arrives 20-36 ms after it (see B13), so using the
latest verdict compares every cloud against the previous one's answer. The tell
was that the marginals were identical -- CLEAR 13608/13607, SLOWDOWN 6497/6497,
STOP 1421/1421 -- which is a temporal offset, not a disagreement. Fixing the
pairing took agreement from 59.71% to 99.85% on the same stack.

### Recording for future bag tests

Only `/laser`, `/sonar/front` and `/bumper` are absent from the existing bags,
because those three are `"enabled": false` in naoqi's `boot_config.json`. Add
them to the record list on the next session:

    /laser /sonar/front /bumper

Everything else these tests need is already in `slam_20260823_aligned`:
`/joint_states` (76668 msgs -- `bumper_guard`'s liveness witness), `/cmd_vel`,
`/robot_description`, and `/tf` carrying naoqi_driver's **complete 83-frame
body tree** rooted at `base_footprint`:

    base_footprint -> base_link -> torso -> Hip -> Pelvis -> Tibia
        -> SonarFront_frame, BumperFL/FR/B_frame, ShovelLaser_frame, ...

So `sonar_tf_bridge` is testable against the real dynamic chain on replay, and
the gate's liveness is real rather than synthetic. Only the sensor READINGS
have to be injected. Note the sensor-rig launch does not publish these frames
(`pepper_sensor_tf.launch.py` deliberately omits `pepper.urdf`) -- they come
from the bag's own `/tf`, or from the driver on the robot.

### Three things that bite on replay

**The monitor only evaluates when `cmd_vel_in` arrives.** It is driven by the
`cmd_vel_raw` callback, not a timer: MEASURED 1 output in 4 s with no
`cmd_vel_raw` against 80 with it. And the bag's own recorded `/cmd_vel` is only
**0.633 Hz**, far too sparse to drive it -- inject `cmd_vel_raw` at the
controller's 20 Hz instead of replaying it.

**A disabled source still logs `Creating Scan` / `Creating Range`.** `enabled`
gates only whether the data is used, so the log cannot tell you which sources
are live. Check `ros2 param get` instead.

**`cloud_range_filter` takes `input_topic` / `output_topic` as PARAMETERS**, not
remaps. Remapping `/points_in` silently leaves it on its defaults
(`/cloud_registered_body`), and the monitor then sees no cloud at all.

---

## B. Robot tests

Run in order. **Have the emergency stop within reach for B1-B3**, since those
tests deliberately provoke uncommanded motion.

### B0. Prerequisites: turn the sensors on

**Status: PART DONE** 2026-10-19. All three are now enabled in
`boot_config.json` (committed), and the topics were verified live. The bumper
EVENTS are still unconfirmed -- nothing was pressed while watching.

| Step | Expected | Result |
|---|---|---|
| Set `"laser"`, `"sonar"`, `"bumper"` to `"enabled": true`, restart the driver | `/laser`, `/sonar/front`, `/sonar/back`, `/bumper` all present | **DONE** -- all four present |
| `ros2 topic hz /laser` | ~10 Hz | **DONE** -- 9.66-10.04 Hz. `/sonar/front` and `/sonar/back` 10.0 Hz, `/joint_states` 50 Hz |
| Driver log | No `no laser board has advanced its frame counter` | **TODO** -- note `first_poll_s_` now holds this error back for a full `kFrameStaleSeconds`, so a start-up stall is no longer reported spuriously |
| Press each bumper by hand, `ros2 topic echo /bumper` | Three distinct ids: 0 RIGHT, 1 LEFT, 2 BACK. **The BACK event is behind a `robot_ == PEPPER` guard -- confirm it specifically** | **TODO** -- 0 messages seen, which is correct for an untouched event topic but proves nothing |

`/sonar/back` arrives whether or not it is wanted: `converters/sonar.cpp`
pushes both Front and Back keys for Pepper. It is simply not consumed.

### B1. The robot-side watchdog (layer 1)

**Status: TODO**

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

**Status: TODO**

| Step | Expected |
|---|---|
| Publish `/cmd_vel` at 10 Hz, then stop publishing | Base stops after `cmd_vel_timeout` (default 0.5 s), with `No /cmd_vel for the configured timeout, stopping the base` |
| Set `max_linear_speed: 0.2`, command 0.5 m/s | Robot moves at 0.2 m/s, not 0.5 |
| Set `max_angular_speed: 0.3`, command 1.0 rad/s | Clamped to 0.3 |

### B3. Collision monitor polygons (layer 3)

**Status: TODO on the robot.** The polygon geometry was validated on the bag instead (A5-A8), and a 150 s slice matched plain geometry on 99.4% of scans.

The numbers below come from the config; the point of the test is that the
measured distances match and that the robot's own body never triggers them.

**`/collision_monitor_state` does not exist.** `state_topic` is not a declared
parameter in Humble's `collision_monitor` (MEASURED: `ros2 param list` has no
such entry, and the topic is never advertised). Read the verdict off the gated
velocity instead -- compare the monitor's input to its output:

| `cmd_vel` out | verdict |
|---|---|
| `0.0` | STOP |
| `0.3 x vx_in` | SLOWDOWN |
| `vx_in` | clear |

The node also logs `Robot to stop due to PolygonStop polygon` /
`Robot to slowdown for 30.000000 percents` on each change, but only on change.

```bash
ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py
ros2 launch pepper_navigation safety_cut_viz.launch.py      # verdict + colours
ros2 topic echo /cmd_vel_safe                               # the monitor's output
ros2 topic echo /cmd_vel                                    # after bumper_guard
```

There is no `/collision_monitor_state` on Humble (see "Seeing what the safety
filter does"); "no action" below means the viz says `clear` and `/cmd_vel`
passes the command through unchanged.

| Step | Expected |
|---|---|
| Stand still in an open room, nobody near, for 60 s | `clear` on every scan, at most 1-2 red points. **If it stops while alone, find the red points in the viz**: beside the lidar = side-body cut (A1); straight ahead = the body box no longer covers Pepper's front |
| Same, with autonomous life on (arms and stance moving) | Still `clear`. Pepper's front face moves several cm with posture -- see the open issue below |
| Box straight ahead touching the base (~0.10 m from the lidar) | `STOP`. With the old `min_range` 0.22 this box was invisible |
| Drive down a corridor | Mostly no action. Occasional `SLOWDOWN` near the walls is acceptable; constant stopping is not |
| Walk slowly toward the front of the robot | `SLOWDOWN` at ~0.90 m, `STOP` at ~0.40 m. Measure with a tape once |
| Approach from the side | `SLOWDOWN` at ~0.45 m. The stop zone sides are ~0.31 m |
| Put an obstacle at the flank, closer than 0.30 m to the lidar, robot stationary | Expected NOT to stop (inside the cut). Confirm the robot cannot reach it by driving forward |
| Block the robot's path while it drives to a goal | Stops, holds, and resumes when cleared |
| Check the polygons in RViz (`polygon_stop`, `polygon_slowdown`) | Wedges, not circles; the stop wedge clears the 0.255 m body |

**Open issue (2026-10-19):** with nobody near, minutes after the measurements
above, red points appeared at x 0.22-0.28, |y| <= 0.06, z 0.73-0.88 -- 0.5 m
straight ahead of the lidar, just past the body box -- and 40% of scans would
have stopped. Not yet identified (Pepper's hands or tablet edge in a shifted
stance, or something hanging in front). Settle it before trusting the first
row: if it is Pepper, the box must reach x ~0.28, past robot_radius, and needs
its own check that it hides nothing real.

### B4. Costmap and planner margins (layer 5)

**Status: TODO**

| Step | Expected |
|---|---|
| Costmap startup log | No "inflation radius is smaller than the inscribed radius" warning |
| RViz footprint vs the real base | The 0.25 m circle covers Pepper's base |
| Narrowest doorway on the route | Passable at ~0.70 m or wider. Note the width if the planner refuses |
| Normal route | Reaches the goal; MPPI does not oscillate near walls (that would mean the inflation band is too thin to steer on) |

### B5. Localization watchdog (layer 6)

**Status: TODO on the robot.** Its logic now has 18 unit tests (`test_localization_watchdog.py`).

| Step | Expected |
|---|---|
| Startup log | `watching [...pose lock..., ...map -> odom gate...]` |
| Normal run | Pose lock OK; no cancellations |
| Cover the lidar / drive into a large unmapped space for >5 s | `localization LOST (...unhealthy for 5s) -- holding navigation`; the active goal is cancelled |
| Issue a new goal while still lost | Cancelled too (it re-cancels every 3 s) |
| Uncover and let it recover | `localization healthy again`; goals accepted |
| Kill the localizer outright | After `stale_timeout` (5 s in the Nav2 launches) the watchdog treats silence as lost |
| During all of the above | **`cmd_vel` is never touched by the watchdog** -- only the collision monitor stops the wheels |

### B6. Lifecycle gating (layer 7)

**Status: TODO**

| Step | Expected |
|---|---|
| Launch fastloc with the robot somewhere unmapped so the lock is slow | Nav2 servers stay inactive; no costmap errors spinning in the log |
| RViz while searching | The prior map draws (the static `map -> lio_init` anchor exists) but the robot is NOT drawn at the origin |
| Once it locks | `wait_for_map_then_start` calls startup; planner/controller/bt_navigator activate |

<!-- B7 (teleop dead-man) and B11 (front sonar, folded into B10) were removed.
     The numbering is left alone so earlier commits and the system overview
     still point at the right tests. -->

### B8. End-to-end smoke

**Status: TODO** -- run last, after everything else.

| Step | Expected |
|---|---|
| Full route with a person crossing the path | Slows, stops if needed, resumes, reaches the goal |
| Repeat 3 times | No spurious stops on open floor; no missed stops |

### B9. Bumper gate (layer 8)

**Status: PART DONE.** The chain and the direction logic are verified on the bag (A8) and by 17 unit tests. What remains needs hardware: that the three events fire at all, and that reversing physically frees Pepper.

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

### B10. Pepper laser + sonar via the driver backstop (layer 9)

**Status: PART DONE** 2026-10-19. The sensors are characterised live; the
backstop itself is untested on hardware and ships `proximity_guard: false`.

These sensors are **no longer Nav2 sources.** They are owned by
`naoqi_driver`'s proximity backstop (`src/subscribers/proximity_guard.hpp`),
because the laser, the sonar and `/cmd_vel` all pass through that process and a
check there cannot be orphaned when the Jetson or the network goes. Nav2's
collision monitor keeps the L2 alone. See that header for why the backstop is a
coarse arc-and-range rule rather than a copy of the wedges.

What the live session established, all MEASURED:

| finding | detail |
|---|---|
| laser layout | 3 boards, 61 slots: live `0-14` (-120..-65 deg), **blind gap `15-22`**, live `23-37` (-29..+26, forward), **blind gap `38-45`**, live `46-60` (+61..+116) |
| the gaps publish **-1.0** | the `msg_.ranges` initialiser leaking out; below `range_min`, so well-behaved consumers drop it, but it passes an `isfinite` check and became a phantom point in a first analysis |
| `+inf` is ambiguous | it means "clear" OR "too close / unresolvable" -- the boards emit a ~5.40 m saturation sentinel (7.09 on the right bank) which `encodeRange` maps to `+inf` |
| detection works | a box ahead at 0.36 m gave **8 beams, 101/101 scans**; a wall-like object left at 0.39 m gave **10 beams** |
| beam budget | 3.93 deg per slot, so 4 beams needs ~11 cm of width at 0.40 m and ~25 cm at 0.90 m. Thin obstacles (chair leg, cable) give 1-2 beams and never reach a threshold of 3 |
| single dropout | slot 31 (+2.0 deg, dead ahead) read `+inf` while slots 30 and 32 both saw the box -- specular loss at normal incidence. This is why `min_hits` defaults to 2 |
| front sonar | 83.6% of readings **above** the declared 2.55 m max, up to 4.77 m. Unusable, not clear |
| back sonar | pinned 0.360-0.370 m, std 0.001 -- physically obstructed. **Never enable it** |
| ranges are from | laser: **base_footprint** (the driver transforms all three boards before taking the norm). Sonar: **its own frame**, forward of base_footprint |

| Step | Expected |
|---|---|
| Object 1-1.5 m dead ahead | slots 23-37 report it. Confirmed on 2026-10-19 |
| Walk an object around at ~1 m | Dropouts at -61..-33 and +29..+57 deg -- map the blind wedges physically |
| Slide a narrow object across dead ahead | Does the slot-31 dropout follow the geometry or stay pinned to slot 31? Pinned means a weak beam |
| 0.15 m tall object at 0.5 m | The laser **should** see it; the L2 cannot (0.22 m height band). This is the backstop's real value-add |
| Clear everything within 3 m, read `/sonar/front` | Records how it signals "nothing": currently 4.77 m |
| Set `proximity_guard: true`, drive toward a soft obstacle | Forward blocked at `proximity_stop_range`; reverse and rotation still work; `/diagnostics` shows `naoqi_driver: proximity backstop` |
| Kill `/laser` and `/sonar/front` while driving | Backstop reports "no opinion" and does **not** stop the robot -- it fails quiet by design, because Nav2 plus `required_topics` already fail safe |

### B12. Stale-source behaviour -- ANSWERED on the bag, FAIL

**Run, measured, and fixed.** It does NOT fail safe. With the laser as the only
enabled source, on `slam_20260823_aligned`:

```
while fed CLEAR at 2.50 m   : {'PASS': 80}
after the laser goes silent : {'PASS': 161}      <- full speed, no sensor data
```

The monitor **holds its last verdict indefinitely**: last verdict SLOWDOWN and
it slows forever, last verdict clear and it passes `vx_in` forever.
`source_timeout` only decides whether a reading is USED. Nothing else caught
it -- the driver's `cmd_vel_timeout` fires only if `cmd_vel` stops, and it does
not.

The fix is `bumper_guard`'s `required_topics`, which clamps translation when a
required sensor topic goes quiet. Verified: 100/121 outputs clamped, the first
21 being the 1.0 s `required_timeout` window.

| Step | Expected |
|---|---|
| While driving, stop `points_safety_filter` | `cmd_vel` linear clamps to 0 within ~1 s; `bumper_guard: sensor liveness` goes ERROR naming the topic |
| Restart the filter | Motion resumes; the status returns to OK |
| Launch with the filter never started | Clamped from the start -- never-seen counts as stale, not as fine |
| Rotation while clamped | Still allowed: the footprint is a 0.25 m circle, so turning in place sweeps nothing new |
| `localization_watchdog` log | Cancels goals too, since the status is in its `status_names` |

### B13. Does it actually stop in time? (this one sets the polygon distances)

**Why it exists.** 0.40 m and 0.90 m were derived from reasoning about braking
and latency, and the bag only ever validated the FALSE-TRIGGER rate (A1, A11).
Nothing has measured whether the robot physically stops before contact. The
budget is tighter than it looks:

    stop wedge reach         0.40 m
    robot_radius (the body)  0.25 m
    clearance ahead of body  0.15 m   <- the entire budget

That 0.15 m must cover latency travel PLUS braking:

    0.15 m  >=  v * t_latency  +  v^2 / (2a)

At `vx_max` 0.35 m/s with ~0.15 s of latency, 5.3 cm goes to latency and 9.7 cm
is left for braking, which demands **a >= 0.63 m/s²**. No acceleration limit is
configured anywhere -- MPPI sets no `ax_min` and `boot_config.json` sets no
accel cap -- so deceleration is whatever the hardware gives when commanded to
zero, and that value is unknown.

**Latency is MEASURED, and it is mostly the lidar.** 2026-10-19, on the bag,
109 rising edges of "a point entered the stop polygon" matched to the gated
`cmd_vel` reaching zero:

| term | value | source |
|---|---|---|
| lidar sweep + stamp lag | **98.8 ms** | `/points` header stamps sit -98.8 ms from bag receive time: the L2 accumulates a full sweep before publishing |
| filter + monitor | **36.2 ms** p90 (p50 20.6, max 49.5) | measured; the 49.5 ms max is the 20 Hz controller tick, since the monitor is `cmd_vel`-triggered, not timer-driven |
| **software subtotal** | **135 ms** | |
| driver -> wheels | unmeasured | naoqi side, needs the robot |

At `vx_max` 0.35 m/s, 135 ms is **4.7 cm** travelled before braking begins,
leaving **10.3 cm** of the 15 cm clearance, which **requires a >= 0.60 m/s²**.
Sensitivity: if the driver adds another 50 ms the requirement rises to about
0.72 m/s². Note this is dominated by the lidar's own sweep time, which no
software tuning removes.

**Is 0.90 m reasonable? Yes, with ~5x margin.** The slow zone's job is to get
from 0.35 to 0.105 m/s before the stop boundary, over 0.90 - 0.40 = 0.50 m. At
0.60 m/s² that takes `(0.35^2 - 0.105^2) / (2 * 0.60)` = **9.3 cm** of the 50 cm
available. The slow zone is not the binding constraint; the stop zone is.

**What can be checked with the robot STATIONARY.** Do these before any driving:

| # | Check | Expected |
|---|---|---|
| 1 | Park the robot. Put a board at 0.45, 0.40, 0.35, 0.30 m dead ahead, and at each flank angle. Count points inside the stop polygon | More than 3 at every distance inside 0.40 m. **If the self-cut or the 0.22 m height band eats the returns, the zone is BLIND and no braking arithmetic matters.** This is the first thing to check |
| 2 | Same, at the slow boundary 0.90 m and just outside it | Triggers inside, clear outside |
| 3 | Board at 0.40 m but only 0.15 m tall | NOT detected -- confirms the height band's lower edge, and that low obstacles are a known blind spot |

**Deceleration needs motion but NOT an obstacle.** In open floor with nothing
ahead, command 0.35 m/s, then command 0, and measure the distance travelled
after the command from TF or `/pepper_odom`. Repeat five times. That yields `a`
with zero collision risk, and `a` plus the measured latency PREDICTS the gap:

    predicted gap = 0.15 m - v * t_latency - v^2 / (2a)

Only run a confirmation approach at a soft obstacle if that prediction is thin.

**Two scenarios, and only the second sets 0.40 m.**

*S1, gradual approach (nominal).* Drive at a static obstacle from 3 m. The slow
zone should engage near 0.90 m and drop the robot to 0.105 m/s, so by the stop
boundary braking is trivial (a few cm). Measure the gap at rest.

*S2, sudden appearance (worst case).* The robot is at full speed and sees
nothing until the obstacle is at the stop boundary -- a board swung in from the
side at ~0.45 m, or a helper stepping out of a doorway as the robot passes.
There is no slow-zone warning here, so this is the case that decides whether
0.40 m is enough.

**Protocol.** A compliant obstacle only -- foam board or a cardboard box, never
a person and never a wall for the first runs. Start at 0.1 m/s and step up
0.1 -> 0.2 -> 0.35. Bumper enabled, since it is the backstop for exactly this
test (and this doubles as B9's contact case). A way to cut motion within reach
-- the e-stop or the driver's own shutdown -- and nobody between robot and
obstacle. Five repeats per speed, and **record the minimum
gap, not the mean** -- the worst case is the safety case.

| Result | What it means | Action |
|---|---|---|
| gap > 0 at every speed in S1 **and** S2 | 0.40 m is genuinely safe | keep it |
| gap > 0 in S1, gap <= 0 in S2 | the stop zone only works because the slow zone warned it | raise the stop reach to `0.25 + v*t + v^2/(2a) + 0.05`, or cap `vx_max` |
| gap <= 0 in S1 | the stop zone is too small outright | raise the reach |
| speed at the 0.40 m boundary in S1 is above ~0.12 m/s | 0.90 m is not deep enough to decelerate in | raise the slow reach, or cap `vx_max` |

**Any change to the stop wedge re-opens the self-cut tuning.** That is exactly
how the `self_cut_range: 0.35` bug happened: the wedge was widened to +-0.31 m
at the flanks, which pushed its diagonal corners to ~0.465 m from
`base_footprint` and into a band the 0.35 m cut did not reach, so Pepper's own
torso sat inside the stop zone on 30% of scans. So after changing any polygon
distance, re-run A1 and A11 on the bag before trusting the new value.

---

## J. L2 lidar tests (run ON THE JETSON)

The L2 is connected to the Jetson, and since the laser and sonar moved to
`naoqi_driver`'s backstop it is the collision monitor's ONLY source. Run these
on the Jetson itself: a `/points` subscriber on the laptop pulls ~4900 points
at 11 Hz across WiFi, and that is the safety chain's own input riding the link.

Every value the monitor is tuned to -- `self_cut_range`, `min_range`, the
polygon reaches -- was derived from BAG data. These are what check it against
the real sensor.

`pepper_sensor_tf` must be running or nothing transforms; MEASURED 2026-10-19,
`base_footprint <- l2lidar_frame` resolves on the Jetson once it is up.

```bash
cd ~/ros2_ws && source install/setup.bash
export ROS_DOMAIN_ID=5 ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/upanzi/ros2_ws/config/cyclonedds/robot.xml
```

| # | Test | Status | Result / expectation |
|---|---|---|---|
| J1 | Rate, points per cloud, header-stamp lag | **DONE** 2026-10-19 | 11.3 Hz, 4923 pts/cloud, stamp lag **+102.4 ms** -- confirms the bag's 98.8 ms, and it is 73% of B13's latency budget |
| J8 | Extrinsic `base_footprint <- l2lidar_frame` | **DONE** 2026-10-19 | `[0.133, 0.000, 0.258]`, rotation RPY `[-90.6, -66.0, -89.4]` deg. **Not identity**, so no translation-only shortcut is valid |
| J5 | Board at 0.45 / 0.40 / 0.35 / 0.30 m dead ahead (from base_footprint, the base centre -- NOT the lidar, which sits 0.133 m further forward); count points inside `PolygonStop` | **DONE** 2026-10-19 | 0.5 m box, current filter, distances by the RealSense (placing by eye was off by up to 0.1 m -- use a reading, not a guess). Stop-wedge points per scan and verdict at `max_points: 3`: against the base (~0.10 m from the lidar) 31-113, STOP (invisible at the old `min_range: 0.22`); **0.315 m** ~2000, STOP 100%; **0.355 m** ~1550, STOP 100%; **0.445 m** 287, STOP 100%; **0.475 m** 50, STOP 95%; **0.635 m** 2 (max 8), **STOP 25%**. **Not blind anywhere inside the reach.** Two effects make it fire EARLY, both on the safe side: (1) the L2 reads **~5 cm short at every distance** vs the RealSense (5.7 / 3.9 / 5.0 / 5.1 / 6.8 cm) -- constant with range, so more likely the tape-measured `x: 0.133` in `sensor_tf.yaml` than range noise; settle it on a wall at 1-2 m; (2) **ghost points**: anything ahead puts ~6 pts/scan along the line of sight at lidar height (z 0.22-0.28, y ~0, intensity like real returns), 2-8 of them inside 0.40 m -- the whole cause of the 0.635 m STOPs. **Fix: `max_points: 10`** (was 3): ghosts peak at 8, real objects give >= 50 in the wedge, empty scene max 1-2, so 0.635 m goes to 0% STOP and every real case still stops. Not yet re-checked live (sensors down 2026-10-20); the 0.475 m early stop remains (offset, not ghosts) |
| J3 | Self-hit shell, stationary, nothing within 2 m: histogram returns by range and bearing from the lidar | **PART DONE** 2026-10-19 | **Arms at rest: no 0.355-0.368 m shell.** 510 scans: the flank body sits **0.11-0.25 m** from the lidar at bearings 55-80 deg (x 0.18-0.22, \|y\| <= 0.20, z 0.22-0.28); Pepper's front face straight ahead at x ~0.18, z 0.30-0.90 (now the body box). Live `self_cut_range` sweep, stop-wedge false-STOP rate: off 65%, 0.25 m 0% (max 3 pts, no margin), **0.30-0.40 m 0%** (max 1-2; 510 scans at 0.30). Posture matters: minutes apart the front face moved from x 0.18 to x 0.22-0.28 at z 0.73-0.88 (naoqi_driver off, stance changed) and 40% of scans false-STOPped, then went back. **Outstanding: repeat with naoqi_driver on and the arms moving** |
| J4 | Count points with `z < 0.22` in `base_footprint` at close range | **DONE** 2026-10-19 | **`min_height` did not move -- it is still 0.22.** What moved 0.22 -> 0.12 is `min_range` (3D distance from the lidar, in the filter). Live: below-band returns are 92-99% of everything within 0.9 m; the floor tops out at z **0.056** at 0.6-0.9 m (no below-floor reflections); Pepper's base reaches z 0.19-0.29 within 0.45 m (the part above 0.22 is taken by the side cut and the body box). Lowering `min_height` to 0.15 / 0.10 / 0.05 false-STOPped 65-100% of empty scans, so 0.22 stays |
| J11 | Nearest return with nothing around | **DONE** 2026-10-19 | 115 scans live: **nothing below 0.05 m** (no 0.007 m housing returns, no NaN, no zero points); 0.4 pts/scan at 0.05-0.08 m, 7.7 at 0.08-0.10, 19 at 0.10-0.12 (Pepper's front and the floor). `min_range: 0.12` clears the housing with margin; what it lets through at 0.12-0.15 m is body (body box) or floor (height band). An empty 45 s capture had no in-band returns between 0.12 and 0.22 |
| J6 | 0.15 m tall object at 0.40 m | **TODO** | **Not** seen -- the height band's documented blind spot, and the reason the driver backstop's laser matters |
| J7 | Object at 0.30 m on the flank | **TODO** | **Not** seen -- the self-cut's deliberate blind shell. Re-check, since the cut shrank from 0.40 to 0.30 |
| J9 | `ros2 topic bw /points` | **DONE** 2026-10-19 | **`ros2 topic bw` prints nothing** on this BEST_EFFORT topic (the J10 trap). Computed from the stream: 4828 pts x **24 B** (x, y, z, intensity, range, time) x 11.5 Hz = **~1.3 MB/s, ~10.6 Mbit/s** -- higher than the 6.5 estimate. `/points_safety` carries x, y, z only (12 B/pt) after the cuts, under half that |
| J10 | `ros2 topic info -v /points` | **DONE** 2026-10-19 | `l2lidar_node` publishes **BEST_EFFORT**, KEEP_LAST 5, VOLATILE |

Order: J5 first (a blind stop zone voids everything else), then J3 with the arms
moving (it blocks the `self_cut_range: 0.30` decision), then J4 and J11 for the
`min_range` change, then J6 and J7, then J9 and J10.

`sensor_msgs_py` IS available on the Jetson (the J results above were taken
with it, and `safety_cut_viz` uses it), along with numpy, scipy, yaml, rclpy
and tf2_ros.

---

## What a failure means

| Symptom | Most likely layer |
|---|---|
| Robot stops constantly on open floor | 4 -- side-body cut off, `self_cut_range` 0 or `body_boxes` missing / not covering Pepper's front. Look with `safety_cut_viz` |
| Robot never stops for obstacles | 3 -- `/points_safety` silent (QoS!), or the monitor is not in the `cmd_vel` chain |
| Robot keeps driving after the driver dies | 1 -- `robot_watchdog.py` not running on the robot |
| Robot drives confidently to the wrong place | 6 -- the pose lock is wrong and the watchdog is not acting |
| Nav2 never activates | 7 -- the lock never happened; check the localizer, not Nav2 |
| MPPI oscillates near walls | 5 -- inflation band too thin relative to the inscribed radius |
| Robot touches an obstacle it did stop for | 3 -- the stop reach does not cover latency plus braking; run B13 |
| Robot stops dead on contact and never recovers | 8 -- reverse is being gated too; `BackUp` cannot escape |
| Robot will not move and nothing is in the way | 8 -- a stuck bumper latch; check `bumper_guard: contact` for ERROR |
| Robot stops whenever the driver hiccups | 8 -- `/joint_states` gaps are reading as blind |
| Stops constantly after enabling the laser | 3 -- the boards are reading the floor; re-measure B10 |
| Laser or sonar source has no effect at all | 3 -- source `enabled: False`, or for the sonar the bridge is not running |
| Robot keeps driving with a dead lidar | 8 -- `required_topics` empty; the monitor holds its last verdict and will not stop on its own |
| Robot clamped and will not translate, bumpers clear | 8 -- a required sensor topic is stale; check `bumper_guard: sensor liveness` |
