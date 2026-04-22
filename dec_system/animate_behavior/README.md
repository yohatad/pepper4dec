<div align="center">
<h1>Animate Behavior</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Animate Behavior** package is a ROS2 action server that provides natural, lifelike animation for the Pepper humanoid robot during idle periods or social interactions. It generates smooth, randomized gestural movements across various body parts (arms, hands, legs, and base rotation) to enhance the robot's expressiveness and engagement during human-robot interaction. The module uses high-frequency motion updates (30Hz) with exponential smoothing to achieve natural, fluid movements that avoid mechanical or jerky appearance.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Multiple Behavior Types**: Supports All, body, hands, arms, rotation, and home behaviors
- **High-Frequency Updates**: 30Hz motion updates for smooth animation
- **Exponential Smoothing**: Natural, fluid movements with configurable smoothing factor
- **BehaviorTree Integration**: Action-based stop via `home` behavior type
- **Real-time Feedback**: Continuous feedback on animated limb, gestures completed, and elapsed time

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
| `update_rate` | Animation loop frequency (Hz) | `30.0` |
| `feedback_rate` | Feedback publishing frequency (Hz) | `2.0` |
| `gesture_interval_min` | Minimum time between gesture targets (sec) | `2.5` |
| `gesture_interval_max` | Maximum time between gesture targets (sec) | `4.0` |
| `rotation_interval` | Time between base rotation changes (sec) | `5.0` |
| `smoothing_factor` | Exponential smoothing coefficient | `0.15` |
| `motion_speed` | ALMotion speed parameter | `0.08` |

## 🎭 Behavior Types

The node supports multiple animation modes for different interaction scenarios:

| Behavior Type | Limbs Animated | Description |
|--------------|----------------|-------------|
| `All` | Arms, Hands, Legs, Base | Full-body animation including all limbs and base rotation |
| `body` | Arms, Legs | Torso and leg movements (excludes hands) |
| `hands` | Hands only | Hand opening/closing gestures |
| `arms` | Arms, Hands | Arm and hand movements (excludes hip joints) |
| `rotation` | Base only | Base rotation without limb movement |
| `home` | All limbs | Returns all limbs to neutral home position (immediate stop) |

> **Note:** The `home` behavior type provides an action-based stop mechanism, immediately returning all limbs to their neutral positions and canceling any ongoing animation. This is useful for stopping animations from behavior tree controllers.

## Joint Movement Ranges

Each joint group has defined movement factors that scale with the `selected_range` parameter:

| Joint Group | Joints | Movement Factors |
|-------------|--------|------------------|
| **Arms** | ShoulderPitch, ShoulderRoll, ElbowYaw, ElbowRoll, WristYaw | 0.6, 0.4, 0.6, 0.4, 0.5 |
| **Hands** | Hand (open/close) | 0.5 |
| **Legs** | HipPitch, HipRoll, KneePitch | 0.2, 0.2, 0.1 (conservative for stability) |

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

### Services

| Service | Type | Description |
|---------|------|-------------|
| `/animate_behavior/stop` | `std_srvs/Trigger` | Alternative stop mechanism (explicit service call) |

## Action Interface

**Action Type:** `dec_interfaces/action/AnimateBehavior`

### Goal

| Field | Type | Description |
|-------|------|-------------|
| `behavior_type` | string | "All", "body", "hands", "arms", "rotation", "home" |
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

## Package Structure

```
animate_behavior/
├── config/
│   └── animate_behavior_configuration.yaml
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
