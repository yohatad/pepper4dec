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

---

## B. Robot tests

Run in order. **Have the emergency stop within reach for B1-B3**, since those
tests deliberately provoke uncommanded motion.

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
