<div align="center">
<h1>Animate Behavior for Human-Robot Interaction</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Animate Behavior** package is a ROS2 action server that provides natural, lifelike animation for the Pepper humanoid robot during idle periods or social interactions. It generates smooth, randomized gestural movements across various body parts (arms, hands, legs, and base rotation) to enhance the robot's expressiveness and engagement during human-robot interaction. The module uses high-frequency motion updates (30Hz) with exponential smoothing to achieve natural, fluid movements that avoid mechanical or jerky appearance.

# 📄 Documentation
The main documentation for this deliverable is found in the CSSR4Africa project deliverables. For technical details about the animation pipeline and motion control, refer to the source code and inline comments.

# 🛠️ Installation

Install the required software components to instantiate and set up the development environment for controlling the Pepper robot. Use the [CSSR4Africa Software Installation Manual](https://cssr4africa.github.io/deliverables/CSSR4Africa_Deliverable_D3.3.pdf).

## Prerequisites
- Ubuntu 22.04
- ROS2 Humble
- Python 3.10+ (tested with Python 3.10.12)
- Physical Pepper robot or compatible simulator

## Building the ROS2 Package

```sh
cd ~/ros2_ws
source /opt/ros/humble/setup.bash  # or your ROS2 distribution
colcon build --packages-select animate_behavior
source install/setup.bash
```

# 🎭 Behavior Types
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

# 🔧 Motion Parameters

The animation system uses several hardcoded parameters optimized for natural human-robot interaction:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `update_rate` | 30.0 Hz | Animation loop frequency for smooth motion updates |
| `feedback_rate` | 2.0 Hz | Feedback publishing frequency |
| `gesture_interval` | 2.5-4.5 sec | Randomized time between new gesture targets |
| `rotation_interval` | 5.0 sec | Time between base rotation changes |
| `smoothing_factor` | 0.15 | Exponential smoothing coefficient (0.1-0.2 range) |
| `motion_speed` | 0.08 | ALMotion speed parameter for joint commands |

These parameters are tuned for natural, engaging motion and are defined in the implementation code.

## Joint Movement Ranges

Each joint group has defined movement factors that scale with the `selected_range` parameter:

| Joint Group | Joints | Movement Factors |
|-------------|--------|------------------|
| **Arms** | ShoulderPitch, ShoulderRoll, ElbowYaw, ElbowRoll, WristYaw | 0.6, 0.4, 0.6, 0.4, 0.5 |
| **Hands** | Hand (open/close) | 0.5 |
| **Legs** | HipPitch, HipRoll, KneePitch | 0.2, 0.2, 0.1 (conservative for stability) |

# 🚀 Running the Node

**Run the `animate_behavior` node from the `animate_behavior` package:**

Source the workspace:
```bash
cd ~/ros2_ws && source install/setup.bash
```

Run the animate behavior action server:
```bash
ros2 run animate_behavior animate_behavior
```

> **Note:** Ensure your robot is properly launched and the required topics (`/joint_states`) are available before running this node.

# 🎯 Action Server Interface

The node provides an action server `/animate_behavior` for controlling animation behaviors. The action interface allows clients to request specific animation behaviors with configurable parameters.

## Action Definition

**Action Type:** `cssr_interfaces/action/AnimateBehavior`

### Goal (Request)
```
string behavior_type      # "All", "body", "hands", "arms", "rotation", "home"
float32 selected_range    # 0.0 to 1.0 (movement amplitude scaling)
int32 duration_seconds    # How long to run (0 = infinite until cancelled)
```

### Result (Response)
```
bool success              # Whether execution succeeded
string message            # Status message describing outcome
float32 total_duration    # Actual elapsed time in seconds
```

### Feedback (Continuous Updates at 2Hz)
```
string current_limb       # Currently animated limb
int32 gestures_completed  # Total number of gestures completed
float32 elapsed_time      # Elapsed time since start
bool is_running           # Animation still active flag
```

## Action Usage Examples

### Using ROS2 CLI

**Animate all limbs for 30 seconds with moderate range:**
```bash
ros2 action send_goal /animate_behavior cssr_interfaces/action/AnimateBehavior \
  "{behavior_type: 'All', selected_range: 0.5, duration_seconds: 30}"
```

**Animate arms only indefinitely (until cancelled):**
```bash
ros2 action send_goal /animate_behavior cssr_interfaces/action/AnimateBehavior \
  "{behavior_type: 'arms', selected_range: 0.7, duration_seconds: 0}"
```

**Return to home position (stop animation):**
```bash
ros2 action send_goal /animate_behavior cssr_interfaces/action/AnimateBehavior \
  "{behavior_type: 'home', selected_range: 0.0, duration_seconds: 0}"
```

**Gentle hand gestures for 60 seconds:**
```bash
ros2 action send_goal /animate_behavior cssr_interfaces/action/AnimateBehavior \
  "{behavior_type: 'hands', selected_range: 0.3, duration_seconds: 60}"
```

### Using Python Action Client

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from cssr_interfaces.action import AnimateBehavior

class AnimationClient(Node):
    def __init__(self):
        super().__init__('animation_client')
        self._action_client = ActionClient(self, AnimateBehavior, '/animate_behavior')

    def send_goal(self, behavior_type='All', selected_range=0.5, duration_seconds=30):
        goal_msg = AnimateBehavior.Goal()
        goal_msg.behavior_type = behavior_type
        goal_msg.selected_range = selected_range
        goal_msg.duration_seconds = duration_seconds

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Gestures: {feedback.gestures_completed}, '
                              f'Elapsed: {feedback.elapsed_time:.1f}s')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.message}, '
                              f'Duration: {result.total_duration:.1f}s')

def main():
    rclpy.init()
    client = AnimationClient()
    client.send_goal(behavior_type='arms', selected_range=0.6, duration_seconds=45)
    rclpy.spin(client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Integration with BehaviorTree.CPP

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

# 🖥️ Topics and Services

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_angles` | `naoqi_bridge_msgs/JointAnglesWithSpeed` | Joint angle commands sent to Pepper robot at 30Hz |
| `/cmd_vel` | `geometry_msgs/Twist` | Base rotation velocity commands |

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions from robot (used for smooth interpolation) |

## Action Servers

| Action Server | Type | Description |
|---------------|------|-------------|
| `/animate_behavior` | `cssr_interfaces/action/AnimateBehavior` | Main animation control interface |

## Services

| Service | Type | Description |
|---------|------|-------------|
| `/animate_behavior/stop` | `std_srvs/Trigger` | Alternative stop mechanism (explicit service call) |

> **Note:** For BehaviorTree integration, it's recommended to use the action-based stop via the `home` behavior type rather than the service interface.

# 📁 Package Structure
```
animate_behavior/
├── config/
│   └── animate_behavior_configuration.yaml  # Configuration placeholder
├── data/
│   └── pepper_topics.yaml                   # Topic mappings
├── resource/
│   └── animate_behavior                     # Package marker for ament
├── animate_behavior/
│   ├── __init__.py
│   ├── animate_behavior_application.py      # Main node entry point
│   └── animate_behavior_implementation.py   # Core animation server logic
├── package.xml                              # ROS2 package manifest
├── setup.py                                 # Python package setup
├── setup.cfg                                # Setup configuration
└── README.md                                # This file
```

# 🔍 Debugging Tips

## Animation Server
- Check that the action server is running: `ros2 action list | grep animate_behavior`
- Monitor action server status: `ros2 action info /animate_behavior`
- Send a test goal: `ros2 action send_goal /animate_behavior cssr_interfaces/action/AnimateBehavior "{behavior_type: 'hands', selected_range: 0.5, duration_seconds: 10}"`
- Monitor feedback during execution: Add `--feedback` flag to the action send_goal command
- Check for joint state reception: Look for "✓ Joint states received" in the node logs

## Topic Monitoring
- Verify joint commands are published: `ros2 topic echo /joint_angles`
- Check velocity commands: `ros2 topic echo /cmd_vel`
- Monitor joint state feedback: `ros2 topic echo /joint_states --once`
- Check topic rates: `ros2 topic hz /joint_angles` (should be ~30Hz during animation)

## Common Issues

### Animation not starting
- Ensure the robot is launched and publishing `/joint_states`
- Check that the goal parameters are valid (behavior_type must be one of: All, body, hands, arms, rotation, home)
- Verify `selected_range` is between 0.0 and 1.0

### Jerky or unnatural motion
- The node uses 30Hz updates with exponential smoothing for smooth motion
- If motion appears jerky, check CPU load (high load may affect timing)
- Verify `/joint_states` is being received (check logs for "Joint states received")

### Animation won't stop
- Use the `home` behavior type to stop: `ros2 action send_goal /animate_behavior cssr_interfaces/action/AnimateBehavior "{behavior_type: 'home', selected_range: 0.0, duration_seconds: 0}"`
- Alternatively, use the stop service: `ros2 service call /animate_behavior/stop std_srvs/srv/Trigger`
- Cancel via action: `ros2 action send_goal /animate_behavior cssr_interfaces/action/AnimateBehavior "{behavior_type: 'All', selected_range: 0.5, duration_seconds: 10}" --cancel`

## Logging
- View detailed logs: `ros2 run animate_behavior animate_behavior --ros-args --log-level debug`
- Monitor ROS2 logs: `ros2 topic echo /rosout | grep animate_behavior`

# 🎨 Implementation Details

## Motion Generation Algorithm

The animation system uses a sophisticated motion generation approach:

1. **Target Generation:** New random target positions are generated every 2.5-4.5 seconds (randomized for natural variation)
2. **Smooth Interpolation:** 30Hz update loop smoothly interpolates from current to target positions using exponential smoothing
3. **Natural Bias:** 30% probability of returning toward home position to create natural oscillatory motion
4. **Range Limiting:** All movements constrained by joint limits and scaled by `selected_range` parameter
5. **Staggered Timing:** Limb animations start with random time offsets for asynchronous, natural motion

## Thread Safety

- Uses `MultiThreadedExecutor` with 4 threads for concurrent callback processing
- `ReentrantCallbackGroup` enables thread-safe timer and action callbacks
- Thread synchronization via `threading.Event` for goal completion signaling
- Safe shutdown handling with exception guards for cleanup operations

## Performance Characteristics

- **Update Rate:** 30Hz joint position commands for smooth motion
- **Feedback Rate:** 2Hz feedback updates to clients
- **Gesture Frequency:** New gesture every 2.5-4.5 seconds (randomized)
- **Rotation Frequency:** Base rotation every 5 seconds
- **CPU Usage:** Low CPU footprint (~1-2% on typical systems)
- **Latency:** <100ms goal acceptance, immediate feedback stream

# 💡 Support

For issues or questions:
- Create an issue on the CSSR4Africa GitHub repository
- Contact: <a href="mailto:yohanneh@andrew.cmu.edu">yohanneh@andrew.cmu.edu</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2025 CSSR4Africa Consortium
Funded by African Engineering and Technology Network (Afretec)
Inclusive Digital Transformation Research Grant Programme

**Author:** Yohannes Tadesse Haile, Carnegie Mellon University Africa
**Version:** v1.0
**Last Updated:** February 2026
