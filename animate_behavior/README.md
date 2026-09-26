<div align="center">
<h1>Animate Behavior</h1>
</div>

<div align="center">
  <img src="../images/upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Animate Behavior** package is a ROS2 action server that keeps Pepper looking alive during idle periods and conversation: small randomized movements of the arms, hands, legs and base, smoothed at 30 Hz, with a cascade wave on the face LEDs running alongside.

## ✨ Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Multiple Behavior Types**: Supports All, body, arms, hands, idle, rotation, and home behaviors
- **High-Frequency Updates**: 30Hz motion updates for smooth animation
- **Exponential Smoothing**: Natural, fluid movements with configurable smoothing factor
- **BehaviorTree Integration**: Action-based stop via `home` behavior type
- **Real-time Feedback**: Continuous feedback on animated limb, gestures completed, and elapsed time
- **LED Cascade Animation**: Synchronized face LED wave effect driven via the `naoqi_driver` `/naoqi_driver/run_led` action server

## ✅ Prerequisites
- **ROS2 Humble** or newer
- **C++17 toolchain** (compiled package, built via `colcon build`)
- **Physical Pepper robot** or compatible simulator
- **naoqi_bridge_msgs** for joint command publishing

## 🛠️ Installation

### Package Installation

```bash
cd ~/ros2_ws
colcon build --packages-up-to animate_behavior
source install/setup.bash
```

## 🔧 Configuration

Configuration is managed via `config/animate_behavior_configuration.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `verbose_mode` | Enable verbose logging | `true` |
| `gesture_update_rate` | Animation loop frequency (Hz) | `30.0` |
| `gesture_interval_min` | Minimum time between gesture targets (sec) | `2.5` |
| `gesture_interval_max` | Maximum time between gesture targets (sec) | `4.5` |
| `gesture_rotation_interval` | Time between base rotation changes (sec) | `5.0` |
| `gesture_smoothing_factor` | Exponential smoothing coefficient | `0.15` |
| `gesture_motion_speed` | ALMotion speed parameter | `0.08` |
| `led_enabled` | Enable/disable LED cascade animation | `true` |
| `led_white_step` | Delay between each LED layer fading white (sec) | `0.06` |
| `led_dark_step` | Delay between each LED layer fading dark (sec) | `0.04` |
| `led_fade_duration` | Duration of each RGB fade transition (sec) | `0.10` |
| `led_white_hold` | Time all LEDs hold white before fading out (sec) | `2.0` |
| `led_dark_pause` | Pause between cascade wave cycles (sec) | `0.2` |

## 🚀 Running

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Run the animate behavior action server
ros2 run animate_behavior animate_behavior
```

> **Note:** Ensure your robot is properly launched and the required topics (`/joint_states`) are available before running this node.

## 🖥️ ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions from robot (used for smooth interpolation) |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_angles` | `naoqi_bridge_msgs/JointAnglesWithSpeed` | Joint angle commands sent to Pepper robot at 30Hz |
| `/cmd_vel` | `geometry_msgs/Twist` | Base rotation velocity commands |

### Action Servers

| Action | Type | Description |
|--------|------|-------------|
| `/animate_behavior` | `dec_interfaces/action/AnimateBehavior` | Main animation control interface |

### Action Clients

| Action | Type | Description |
|--------|------|-------------|
| `/naoqi_driver/run_led` | `naoqi_bridge_msgs/action/RunLed` | LED commands sent to the naoqi_driver for cascade wave animation |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/animate_behavior/stop` | `std_srvs/Trigger` | Alternative stop mechanism (explicit service call) |

## 🔌 Action Interface

**Action Type:** `dec_interfaces/action/AnimateBehavior`

### Goal

| Field | Type | Description |
|-------|------|-------------|
| `behavior_type` | string | "All", "body", "arms", "hands", "idle", "rotation", "home" |
| `selected_range` | float32 | Movement amplitude scaling (0.0 to 1.0) |
| `duration_seconds` | int32 | How long to run (0 = infinite until cancelled) |

### Result

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether execution succeeded |
| `message` | string | Status message describing outcome |
| `total_duration` | float32 | Actual elapsed time in seconds |

### Feedback (2Hz)

| Field | Type | Description |
|-------|------|-------------|
| `current_limb` | string | Currently animated limb |
| `gestures_completed` | int32 | Total number of gestures completed |
| `elapsed_time` | float32 | Elapsed time since start |
| `is_running` | bool | Animation still active flag |

### Action Usage Examples

**Animate all limbs for 30 seconds with moderate range:**
```bash
ros2 action send_goal /animate_behavior dec_interfaces/action/AnimateBehavior \
  "{behavior_type: 'All', selected_range: 0.5, duration_seconds: 30}"
```

**Return to home position (stop animation):**
```bash
ros2 action send_goal /animate_behavior dec_interfaces/action/AnimateBehavior \
  "{behavior_type: 'home', selected_range: 0.0, duration_seconds: 0}"
```

### BehaviorTree.CPP Integration

`behavior_controller` registers the `AnimateBehavior` and `StopAnimateBehavior`
nodes:

```xml
<!-- Animate the body until stopped -->
<AnimateBehavior behavior_type="body" selected_range="0.4" duration_seconds="0"/>

<!-- Stop it, through the /animate_behavior/stop service -->
<StopAnimateBehavior/>
```

## 🎭 Behavior Types

The node supports multiple animation modes for different interaction scenarios:

| Behavior Type | Limbs Animated | Description |
|--------------|----------------|-------------|
| `All` | Arms, Hands, Legs, Base | Full-body animation including all limbs and base rotation |
| `body` | Arms, Hands, Legs | Torso and limb movements |
| `arms` | Arms only | Arm movements (excludes hands) |
| `hands` | Hands only | Hand opening/closing gestures |
| `idle` | None | LEDs only; no limb gestures or rotation |
| `rotation` | Base only | Base rotation without limb movement |
| `home` | All limbs | Moves all joints to neutral home position then stops |

> **Note:** The `home` behavior type provides an action-based stop mechanism, immediately returning all limbs to their neutral positions, canceling any ongoing animation, and turning off all face LEDs.

## 🦾 Joint Movement Ranges

Joint limits, home positions and per-joint amplitude factors are constants in
`src/animate_behavior_implementation.cpp`, taken from the CSSR4Africa D5.1
Actuator Tests deliverable. `selected_range` scales every joint's random
amplitude on top of its factor.

## 🌈 LED Animation

With `led_enabled`, a cascade wave runs on the face LEDs alongside the body
animation: the rings around each eye light white from the outside in, hold
for `led_white_hold`, fade out from the inside out, then pause. The other
`led_*` parameters set the timing. The LEDs turn off when animation stops. If
`/naoqi_driver/run_led` is not available within 5 s of startup, the LEDs are
disabled and body animation continues.

## 📁 Package Structure

```
animate_behavior/
├── config/
│   └── animate_behavior_configuration.yaml   # ROS2 parameters
├── data/
│   └── pepper_topics.yaml                    # topic name overrides
├── include/
│   └── animate_behavior/
│       └── animate_behavior_interface.h      # shared class/struct declarations
├── launch/
│   └── animate_behavior.launch.py
├── src/
│   ├── animate_behavior_application.cpp      # node entry point, lifecycle + action server
│   └── animate_behavior_implementation.cpp   # animation/LED synthesis helpers
├── CMakeLists.txt
├── package.xml
└── README.md
```

## 🏗️ Architecture

- **Animation loop** (`gesture_update_rate`, 30 Hz): picks a new random target
  per joint group every `gesture_interval_min`-`gesture_interval_max` seconds and
  moves toward it with exponential smoothing, publishing `/joint_angles`.
- **Feedback** at 2 Hz on the action.
- **LED cascade**: timers that send `MODE_RGB_FADE` goals to `/naoqi_driver/run_led`.

## 🧪 Testing

```bash
cd ~/ros2_ws
colcon test --packages-select animate_behavior
colcon test-result --verbose
```

Runs unit tests for the motion math: joint soft-limit clamping, randomized gesture targets and smoothing.

## 💡 Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## 📜 License
Copyright (C) 2025 Carnegie Mellon University Africa
Licensed under the BSD-3-Clause License. See individual package licenses for details.
