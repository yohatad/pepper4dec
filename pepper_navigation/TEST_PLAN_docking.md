# Robot test plan: autonomous docking (opennav_docking, L2 + bottom camera)

Temporary. Delete this file once the tests below have passed on the robot and
the docking branch is merged.

Everything here has only run against a simulation so far (the docking tests in
`test/`). Expect to change numbers in `config/docking.yaml` and
`config/docks.yaml` as you go: tests 3 and 4 exist to measure them.

## What is under test

| Repo | Branch | What |
|---|---|---|
| pepper4dec | `feature/auto-docking` | `launch/pepper_docking.launch.py`, `config/docking.yaml`, `config/docks.yaml`, `scripts/dock_detector.py`, `scripts/dock_perception.py`, `scripts/naoqi_battery_state.py`; `dec_launch` `naoqi_driver.launch.py bottom_camera:=true` |
| opennav_docking | `humble` at `9d7819a` | the docking server, built from source (`rotate_to_dock`) |
| naoqi_driver2 | `feature/modern-cpp` | boot-config parameters, so `bottom_camera:=true` can switch the bottom camera on |

Nothing here is merged into `devel` or `main`.

## Safety, for every test that moves the robot

- Stand next to Pepper with a hand on the chest button, and keep a terminal
  ready with `ros2 action cancel`-able goals (Ctrl-C on `send_goal` cancels).
- The backward approach is blind: Pepper's L2 and cameras only look forward.
  Keep the area behind the station's approach line clear.
- For the first docks (test 6) halve the speed: `controller.v_linear_max: 0.08`
  in `config/docking.yaml`.

## Setup

```bash
cd ~/ros2_ws/src
git -C pepper4dec fetch && git -C pepper4dec switch feature/auto-docking
git clone -b humble https://github.com/open-navigation/opennav_docking.git
git -C opennav_docking checkout 9d7819a
touch opennav_docking/opennav_following/COLCON_IGNORE \
  opennav_docking/opennav_following_bt/COLCON_IGNORE \
  opennav_docking/opennav_following_demo/COLCON_IGNORE
git -C naoqi_driver2 switch feature/modern-cpp
cd .. && rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select naoqi_driver dec_launch \
  opennav_docking_msgs opennav_docking_core opennav_docking pepper_navigation
source install/setup.bash
colcon test --packages-select pepper_navigation && colcon test-result --verbose
```

Expected: everything builds; all tests pass, including `test_dock_perception`,
`test_docking_config` and `test_docking_sim_launch` (the full simulated dock,
about 30 s). If `test_docking_sim_launch` is missing, opennav_docking was not
found when pepper_navigation was configured: rebuild pepper_navigation.

Bring-up used by the tests below (each in its own terminal):

```bash
ros2 launch dec_launch naoqi_driver.launch.py nao_ip:=<robot ip> bottom_camera:=true
ros2 launch dec_launch dec_robot.launch.py            # or your usual sensor bring-up
ros2 launch pepper_navigation pepper_nav2_fastloc.launch.py
ros2 launch pepper_navigation pepper_docking.launch.py   # from test 3 on, once docks.yaml is set
```

Overt attention off and head pitched down, before any detection or dock:

```bash
ros2 service call /overt_attention/set_enabled std_srvs/srv/SetBool "{data: false}"
ros2 topic pub --once /joint_angles naoqi_bridge_msgs/msg/JointAnglesWithSpeed \
  "{joint_names: [HeadYaw, HeadPitch], joint_angles: [0.0, 0.2], speed: 0.1}"
```

## 1. Signals (robot standing still)

| Check | Expected |
|---|---|
| `ros2 topic hz /naoqi_driver/camera/bottom/image_raw` | ~10 Hz. If the topic is named differently, set `camera.image_topic` / `camera.info_topic` in `config/docking.yaml` |
| `ros2 topic echo --once /naoqi_driver/camera/bottom/camera_info` | 640 x 480, `frame_id: CameraBottom_optical_frame` |
| `ros2 run tf2_ros tf2_echo map CameraBottom_optical_frame` | Resolves. After the head command above, the camera is ~1.03 m above the floor |
| `ros2 run rqt_image_view rqt_image_view` on the bottom camera, at the staging pose | The whole white station is in view, with floor around it |
| `ros2 run pepper_navigation naoqi_battery_state.py`, then `ros2 topic echo /battery_state` | `percentage` is Pepper's charge (0-1), `power_supply_status` 2 (discharging) |
| Plug the charging cable in by hand | `/battery_state` `power_supply_status` 1 (charging) or 4 (full); `/docking/battery_state` `current: 1.0` |
| `/battery_state` `current` while charging | Positive. If negative, set `naoqi_battery_state.current_sign: -1.0` |

## 2. NAOqi safety while backing towards the station (teleop, no docking)

naoqi_driver moves the base through `ALMotion.move` with NAOqi's collision
protection on. Docking backs Pepper into the station until the contacts touch.

| Check | Expected |
|---|---|
| Face away from the station ~0.5 m in front of it; back up slowly (`ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: -0.05}}"`, Ctrl-C to stop) | Pepper reaches the station and `/docking/battery_state` flips to `current: 1.0`. Write down if NAOqi instead stops it short, and at what distance: then the cmd_vel route cannot finish a dock, and NAOqi's external collision protection must be relaxed for the last centimetres (follow-up work) |

## 3. Calibration: station size, docked pose, contact offset

| Step | What to do | Where it goes |
|---|---|---|
| 3a | Measure the station: depth (along the docking direction), width, height | `dock_detector.station` in `config/docking.yaml` (now 0.37 / 0.39 / 0.075) |
| 3b | Dock with NAOqi's own routine: `ssh nao@<robot ip>`, `qicli call ALRecharge.goToStation`, wait for `qicli call ALRecharge.getStatus` = 4. Then `ros2 run tf2_ros tf2_echo map base_footprint` | `config/docks.yaml` `pose: [x, y, yaw + pi]` (yaw in radians, pointing INTO the station) |
| 3c | `qicli call ALRecharge.leaveStation`, drive to face the station from ~0.75 m, start `pepper_docking.launch.py`, `ros2 topic echo /dock_detector/lidar_pose` | `station_offset` = distance from the docked x, y (3b) to the measured station centre, along the dock heading. Set `external_detection_translation_x: -<station_offset>` (now -0.25) |
| 3d | Keep the L2 0.6 m from the front face: `staging_x_offset = station_offset - depth/2 - 0.6 - 0.133` | `staging_x_offset` (now -0.67); `colcon test` re-checks it (`test_docking_config`) |

## 4. Detection (robot parked at the staging pose, head down, overt attention off)

Read `ros2 topic echo /dock_detector/status`; watch `/dock_detector/camera_debug`
in rqt_image_view (white blob green, map pose orange, measured pose red).

| Check | Expected |
|---|---|
| 4a Station where the map says | Both "lidar: found" and "camera: found", within 2 cm and 2 deg of the map; "fused: published from 2 sensor(s)" |
| 4b Repeatability: 10 readings of `/dock_detector/lidar_pose` and `camera_pose` | Spread within 1 cm and 1 deg each |
| 4c Move the station by measured amounts (tape marks on the floor): 5 cm sideways; 10 cm back; 10 deg turned | Status reports the same offsets within 1.5 cm and 1.5 deg, for both sensors |
| 4d Someone standing right next to the station; a white sheet of paper near it | Still the right pose, or nothing published. Never a wrong pose |
| 4e Station moved 40 cm (beyond `max_shift` 0.30) | "not found", nothing on `/detected_dock_pose` |
| 4f Lights dimmed or off | Camera may drop out ("no white blob"); lidar alone keeps publishing |
| 4g Turn the head while measuring | Camera reports "head moving: skipped"; the lidar carries on |
| 4h Drive or turn the robot | Both report "robot moving: not measuring" |

## 5. Docking server bring-up

| Check | Expected |
|---|---|
| `pepper_docking.launch.py` with `docks.yaml` still NaN | Refuses to start: "home_dock has no pose" |
| With the pose set | `ros2 lifecycle get /docking_server` = active; `ros2 param get /docking_server rotate_to_dock` = True |
| RViz: add `/staging_pose` and `/dock_pose` (published once a dock goal starts) | Staging ~0.67 m in front of the station, facing it; dock pose on the station's contact point |

## 6. First dock (slow: `v_linear_max: 0.08`)

From about 1 m in front of the station:
`ros2 action send_goal /dock_robot opennav_docking_msgs/action/DockRobot "{use_dock_id: true, dock_id: home_dock}" --feedback`

| Check | Expected |
|---|---|
| Feedback states | NAV_TO_STAGING_POSE (or "already within pre-staging pose tolerance"), INITIAL_PERCEPTION, CONTROLLING, WAIT_FOR_CHARGE |
| Docking server log | "Successful initial dock detection" |
| The turn | 180 deg in place at the staging pose; does not touch the station |
| The approach | Straight backwards onto the station, slowing near it; no oscillation |
| Contact | "Made contact with dock", then "Robot is charging!"; result `success: True` |
| Docked pose (`tf2_echo map base_footprint`) | Within ~2 cm and ~3 deg of the ALRecharge dock (3b) |
| The station | Not pushed out of place (check the tape marks) |

Then restore `v_linear_max: 0.15` and repeat once at normal speed.

## 7. Undock

`ros2 action send_goal /undock_robot opennav_docking_msgs/action/UndockRobot "{dock_type: pepper_station}"`

| Check | Expected |
|---|---|
| Undock | Drives straight off to the staging pose; `success: True`; `/docking/battery_state` back to -1 |
| A Nav2 goal afterwards | Executes normally |

## 8. Repeatability and starting points

| Check | Expected |
|---|---|
| 5 dock / undock cycles, starting from across the room (Nav2 drives to staging) | 5 successes; note each final yaw error and the total time |
| One dock with the station moved 10 cm and 8 deg (tape marks) | Succeeds on the station's real position |

## 9. Failure behaviour

| Check | Expected |
|---|---|
| Station removed (or moved > 30 cm) | Fails with 904 (FAILED_TO_DETECT_DOCK) after its retries; never backs into empty space |
| Person in front of the station during INITIAL_PERCEPTION | Waits or fails with 904; no wrong pose |
| Station unplugged from mains | Backs on, waits, fails with 906 (FAILED_TO_CHARGE); does not keep pushing |
| Cancel during CONTROLLING (Ctrl-C on `send_goal`) | Robot stops at once |
| Battery full | Still docks: "plugged in" counts, not the charging current |

## 10. Other profiles

| Check | Expected |
|---|---|
| `pepper_nav2_amcl.launch.py` + `pepper_docking.launch.py odom_topic:=/odom_lio` | One successful dock and undock |

## 11. Docker image

| Check | Expected |
|---|---|
| `docker compose build` | opennav_docking clones, builds; `nav2_graceful_controller` comes from rosdep |

## Results

| Test | Date | Result | Notes / measured values |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | station size; docked pose; station_offset |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |
| 11 | | | |
