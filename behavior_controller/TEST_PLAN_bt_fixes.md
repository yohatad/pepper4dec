# Robot test plan: behavior tree fixes (tours, phrases, gestures, halt crash)

Temporary. Delete this file once the tests below have passed on the robot and
`main` has been fast-forwarded to `devel`.

## What is under test

| Repo | Branch | Commits |
|---|---|---|
| pepper4dec | `devel` | `38b590a` BehaviorTree.ROS2 halt-crash patch, `ab42f61` tour trees (loop, SubTree blackboards, empty messages, gesture types), `fa9078d` phrases, radian headings, DEC target heights, `53e33b3` BT tests |
| BehaviorTree.ROS2 | `pepper4dec` (fork `yohatad/BehaviorTree.ROS2`) | `cc5294d` halt-crash fix (was `patches/behaviortree_ros2-reset-goal-handle.patch`) |

Nothing here is merged into `main`. Fast-forward it only after the tests pass.

## Setup

```bash
cd ~/ros2_ws/src
git -C pepper4dec switch devel && git -C pepper4dec pull
# The fork with the halt-crash fix; skip if `git -C BehaviorTree.ROS2 branch` already shows pepper4dec
git -C BehaviorTree.ROS2 remote set-url origin https://github.com/yohatad/BehaviorTree.ROS2.git
git -C BehaviorTree.ROS2 fetch origin && git -C BehaviorTree.ROS2 switch pepper4dec
cd .. && colcon build --symlink-install --packages-select behaviortree_ros2 behavior_controller
source install/setup.bash
colcon test --packages-select behavior_controller && colcon test-result --verbose
```

Expected: everything builds and all tests pass (112 at the time of writing,
8 cppcheck skips).

The tours read visitor speech from `/speech_event/text`, so for tests 2 to 5
set `action_server: false` in `speech_event/config/speech_event_configuration.yaml`.
Choose the scenario in `behavior_controller/config/behavior_controller_configuration.yaml`
(`scenario_specification`, `environment_knowledge_base`). With
`--symlink-install` both take effect on the next launch, no rebuild. Put
them back afterwards.

All tests: `ros2 launch dec_launch dec_system.launch.py` (navigation on: the
pointing gesture needs `/localization/pose`).

## 1. Default scenario (regression)

Shipped config: `asr_cm_tts_pipeline`, `decEnvironmentKnowledgeBase_short.yaml`.

| Check | Expected |
|---|---|
| behavior_controller log | `behavior_controller: configured`, then `activated`. No "Knowledge base not loaded" or "Environment knowledge base validation failed" |
| Ask a question | Robot answers as before |

## 2. Lab tour, visitor accepts

`scenario_specification: "lab_tour"`, `environment_knowledge_base: "labEnvironmentKnowledgeBase.yaml"`

| Check | Expected |
|---|---|
| Stand in view | Robot stops idling and plays the iconic `welcome` gesture |
| Welcome speech | "Hello! Welcome to the Robotics Lab! I'm Pepper, your friendly robot tour guide." (was: tree failed here) |
| Answer "yes" | Drives to the start pose (4.0, 6.8) |
| Each exhibit, in order 1 to 5 | Says "Please follow me", drives there, points at the exhibit with the arm (deictic), says its description. All five, not just the first |
| Heading on arrival (`ros2 run tf2_ros tf2_echo map base_footprint`) | Yaw about -90 deg (270) at exhibits 1, 2, 3, 5; 180 deg at exhibit 4. Before the fix it was about -10 deg everywhere |
| Pointing | Arm aims at the exhibit, not straight ahead and not up |
| End | Waves (iconic `wave`), says "Thank you for taking the tour with me. Goodbye!", drives back to (4.0, 6.8) |

## 3. Lab tour, visitor declines

Same config as 2.

| Check | Expected |
|---|---|
| Answer "no" | "Maybe another time", then "Goodbye", a wave, back to (4.0, 6.8). No exhibit is visited |

## 4. Two tours in a row (halt crash)

Same config as 2. Do 3 (decline), then stay in view of the camera.

| Check | Expected |
|---|---|
| After the first tour ends | A new tour starts at once. behavior_controller keeps running: `ros2 node list` still shows `/behavior_controller`, and the log has no `terminate called` / `UnknownGoalHandleError` (before the patch it aborted here) |
| Answer "yes" this time | Full tour, ending with the long goodbye, not the short "Goodbye" left over from the first tour |

## 5. DEC tour

`scenario_specification: "dec_Tour"`, `environment_knowledge_base: "decEnvironmentKnowledgeBase.yaml"`

| Check | Expected |
|---|---|
| Stops 7 and 8 | Robot points without speaking and carries on (they have no `gesture_message_english`). Before the fix the tour ended at 7 |
| All 9 stops | Visited in order, then the goodbye |
| Pointing at the booths (stops 2 to 8) | Every booth target is 0.82 m high, the lab's exhibit height (stops 4 to 8 had headings, 270 / 180, typed into the height). Note where the arm points at each booth, and measure the real heights to put in the KB if they differ |

## 6. Docker image

| Check | Expected |
|---|---|
| `docker compose build` | Builds, with BehaviorTree.ROS2 cloned from the fork's `pepper4dec` branch |
