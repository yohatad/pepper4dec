<div align="center">
<h1>Animate Behavior</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Animate Behavior** package is a ROS2 action server that provides natural, lifelike animation for the Pepper humanoid robot during idle periods or social interactions. It generates smooth, randomized gestural movements across various body parts (arms, hands, legs, and base rotation) to enhance the robot's expressiveness and engagement during human-robot interaction. The module uses high-frequency motion updates (30Hz) with exponential smoothing to achieve natural, fluid movements that avoid mechanical or jerky appearance. A synchronized LED cascade wave on the face LEDs runs in parallel with body animation to further enhance expressiveness.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Multiple Behavior Types**: Supports All, body, arms, hands, idle, rotation, and home behaviors
- **High-Frequency Updates**: 30Hz motion updates for smooth animation
- **Exponential Smoothing**: Natural, fluid movements with configurable smoothing factor
- **BehaviorTree Integration**: Action-based stop via `home` behavior type
- **Real-time Feedback**: Continuous feedback on animated limb, gestures completed, and elapsed time
- **LED Cascade Animation**: Synchronized face LED wave effect driven via the `naoqi_driver` `/naoqi_driver/run_led` action server

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **Physical Pepper robot** or compatible simulator
- **naoqi_bridge_msgs** for joint command publishing

## Installation

### Package Installation

```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select animate_behavior
source install/setup.bash
```

## Configuration

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

## Joint Movement Ranges

Joint limits and home positions are taken from the CSSR4Africa D5.1 Actuator Tests deliverable. Movement factors scale the random gesture amplitude relative to `selected_range`.

### Right Arm

| Joint | Min (rad) | Max (rad) | Home (rad) | Factor |
|-------|-----------|-----------|------------|--------|
| RShoulderPitch | -2.0857 | 2.0857 | 1.7410 | 0.6 |
| RShoulderRoll | -1.5620 | -0.0087 | -0.09664 | 0.4 |
| RElbowYaw | -2.0857 | 2.0857 | 1.6981 | 0.6 |
| RElbowRoll | 0.0087 | 1.5620 | 0.09664 | 0.4 |
| RWristYaw | -1.8239 | 1.8239 | -0.05679 | 0.5 |

### Left Arm

| Joint | Min (rad) | Max (rad) | Home (rad) | Factor |
|-------|-----------|-----------|------------|--------|
| LShoulderPitch | -2.0857 | 2.0857 | 1.7625 | 0.6 |
| LShoulderRoll | 0.0087 | 1.5620 | 0.09970 | 0.4 |
| LElbowYaw | -2.0857 | 2.0857 | -1.7150 | 0.6 |
| LElbowRoll | -1.5620 | -0.0087 | -0.1334 | 0.4 |
| LWristYaw | -1.8239 | 1.8239 | 0.06592 | 0.5 |

### Hands

| Joint | Min | Max | Home | Factor |
|-------|-----|-----|------|--------|
| LHand | 0.0 | 1.0 | 0.67 | 0.5 |
| RHand | 0.0 | 1.0 | 0.67 | 0.5 |

`0.0` = fully closed, `1.0` = fully open.

### Legs

| Joint | Min (rad) | Max (rad) | Home (rad) | Factor |
|-------|-----------|-----------|------------|--------|
| HipPitch | -1.0385 | 1.0385 | -0.0107 | 0.2 |
| HipRoll | -0.5149 | 0.5149 | -0.00766 | 0.2 |
| KneePitch | -0.5149 | 0.5149 | 0.03221 | 0.1 |

## Running the Node

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Run the animate behavior action server
ros2 run animate_behavior animate_behavior
```

> **Note:** Ensure your robot is properly launched and the required topics (`/joint_states`) are available before running this node.

## ROS Interface

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

## Action Interface

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

**Animate arms only indefinitely (until cancelled):**
```bash
ros2 action send_goal /animate_behavior dec_interfaces/action/AnimateBehavior \
  "{behavior_type: 'arms', selected_range: 0.7, duration_seconds: 0}"
```

**Return to home position (stop animation):**
```bash
ros2 action send_goal /animate_behavior dec_interfaces/action/AnimateBehavior \
  "{behavior_type: 'home', selected_range: 0.0, duration_seconds: 0}"
```

**Gentle hand gestures for 60 seconds:**
```bash
ros2 action send_goal /animate_behavior dec_interfaces/action/AnimateBehavior \
  "{behavior_type: 'hands', selected_range: 0.3, duration_seconds: 60}"
```

### BehaviorTree.CPP Integration

```xml
<!-- Animate during idle periods -->
<Action ID="AnimateIdle"
        name="animate_behavior"
        behavior_type="body"
        selected_range="0.4"
        duration_seconds="0"/>

<!-- Return to home position when stopping -->
<Action ID="StopAnimation"
        name="animate_behavior"
        behavior_type="home"
        selected_range="0.0"
        duration_seconds="0"/>
```

## LED Animation

When `led_enabled` is `true` and the `/naoqi_driver/run_led` action server is available, the node runs a continuous **cascade wave** on Pepper's face LEDs in parallel with body animation.

### Cascade Wave Pattern

The 10 individual face LED actuators are grouped into 5 layers radiating outward from the center of each eye. On each cycle:

1. **Fade in** — layers light up white one at a time from the outermost ring inward, with `led_white_step` seconds between layers.
2. **Hold** — all LEDs stay white for `led_white_hold` seconds.
3. **Fade out** — layers fade to dark from the innermost ring outward, with `led_dark_step` seconds between layers.
4. **Pause** — `led_dark_pause` seconds before the next cycle starts.

| Layer | Left actuators | Right actuators |
|-------|---------------|-----------------|
| 0 (outermost) | `FaceLedLeft5` | `FaceLedRight5` |
| 1 | `FaceLedLeft6`, `FaceLedLeft4` | `FaceLedRight6`, `FaceLedRight4` |
| 2 | `FaceLedLeft7`, `FaceLedLeft3` | `FaceLedRight7`, `FaceLedRight3` |
| 3 | `FaceLedLeft0`, `FaceLedLeft2` | `FaceLedRight0`, `FaceLedRight2` |
| 4 (innermost) | `FaceLedLeft1` | `FaceLedRight1` |

Each fade uses `MODE_RGB_FADE` with a `led_fade_duration` second transition. LEDs are turned off automatically when animation stops.

### Dependency

The LED animation requires the `naoqi_driver` node to be running with the `/naoqi_driver/run_led` action server available. If the server is not found within 5 seconds at startup, LED animation is automatically disabled and body animation continues normally.

## Package Structure

```
animate_behavior/
├── config/
│   └── animate_behavior_configuration.yaml   # all ROS2 parameters
├── data/
│   └── pepper_topics.yaml
├── resource/
│   └── animate_behavior
├── animate_behavior/
│   ├── __init__.py
│   ├── animate_behavior_application.py
│   └── animate_behavior_implementation.py
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## Architecture

The animation system uses high-frequency motion updates (30Hz) with exponential smoothing to achieve natural, fluid movements:

1. **Animation Loop**: Runs at 30Hz, publishing joint angle commands
2. **Gesture Generation**: Randomizes target positions for each joint group
3. **Exponential Smoothing**: Interpolates between current and target positions
4. **Feedback Thread**: Publishes status updates at 2Hz
5. **Action Server**: Handles goal requests and cancellation
6. **LED Cascade**: ROS timer-based scheduler fires `MODE_RGB_FADE` goals against `/naoqi_driver/run_led` in a looping wave pattern; cancelled and turned off when animation stops

## Testing

```bash
# Check node is running
ros2 node list

# Verify action server is available
ros2 action list

# Send a test goal
ros2 action send_goal /animate_behavior dec_interfaces/action/AnimateBehavior \
  "{behavior_type: 'All', selected_range: 0.5, duration_seconds: 10}"

# Monitor joint commands
ros2 topic echo /joint_angles
```

## Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
