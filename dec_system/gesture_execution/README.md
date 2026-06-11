<div align="center">
<h1>Gesture Execution for Pepper Robot</h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:70%; height:auto;">
</div>

The **Gesture Execution** package is a ROS2 action server that executes various types of gestures on the Pepper humanoid robot. It provides an action server interface for executing deictic (pointing), iconic (predefined arm motions), bowing, and nodding gestures with smooth Bézier interpolation for natural motion. The system includes real-time feedback on gesture execution progress and comprehensive visualization support for debugging.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble with action-based interface
- **Multiple Gesture Types**: Supports deictic, iconic, bowing, and nodding gestures
- **Bézier Interpolation**: Smooth motion with continuous velocity and acceleration profiles
- **Inverse Kinematics**: Calculates joint angles for pointing to specific 3D locations
- **Joint Limit Safety**: Validates all motions against robot joint limits
- **RViz2 Visualization**: Publishes markers for visualizing pointing gestures
- **Real-time Feedback**: Provides elapsed time feedback during gesture execution

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **Pepper Robot** or simulator with NAOqi bridge
- **dec_interfaces** package for action definitions

## Installation

### Package Installation

```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone https://github.com/yohatad/pepper4dec.git

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select gesture_execution dec_interfaces
source install/setup.bash
```

### Python Dependencies

```bash
pip install PyYAML
```

## Configuration

Configuration is managed via `config/gesture_execution_configuration.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `gestureDescriptors` | Path to gesture descriptor file | `gesture.yaml` |
| `robotTopics` | Path to robot topic mapping file | `pepperTopics.yaml` |
| `verboseMode` | Enable detailed logging and debugging | `true` |

### Gesture Definitions

Gestures are defined in `data/gesture.yaml` with the following structure:
```yaml
gestures:
  wave:
    joints: ["RArm"]
    joint_angles:
      RArm:
        - [1.7410, -0.09664, 1.6981, 0.09664, -0.05679]  # Waypoint 1
        - [0.0414, -0.7725, 1.4900, 0.5236, 0.0]          # Waypoint 2
    times:
      RArm: [1.0, 2.0]  # Timing for each waypoint
```

### Topic Mappings

Robot topics are configured in `data/pepper_topics.yaml`:
```yaml
topics:
  JointAngles: "/joint_angles"
  Wheels: "/cmd_vel"
  JointStates: "/joint_states"
  RobotPose: "/robot_localization/pose"
```

## Running the Node

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Run the gesture execution node (Action Server)
ros2 run gesture_execution gesture_execution
```

## ROS Interface

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_states` | `sensor_msgs/JointState` | Current joint positions |
| `/robot_localization/pose` | `geometry_msgs/Pose2D` | Robot pose for IK |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/joint_angles_trajectory` | `naoqi_bridge_msgs/JointAnglesTrajectory` | Joint angle trajectory commands |
| `/gesture_execution/visualization` | `visualization_msgs/Marker` | RViz2 visualization markers |

### Action Servers

| Action | Type | Description |
|--------|------|-------------|
| `/gesture_execution` | `dec_interfaces/action/Gesture` | Main gesture execution interface |

## Action Interface

**Action Type:** `dec_interfaces/action/Gesture`

### Goal

| Field | Type | Description |
|-------|------|-------------|
| `gesture_type` | string | "deictic", "iconic", "bow", "nod" |
| `gesture_name` | string | Name of predefined gesture (for iconic gestures) |
| `gesture_duration` | int64 | Duration in milliseconds (1000-5000) |
| `bow_nod_angle` | int64 | Angle for bowing/nodding in degrees |
| `location_x` | float64 | X coordinate for pointing (meters) |
| `location_y` | float64 | Y coordinate for pointing (meters) |
| `location_z` | float64 | Z coordinate for pointing (meters) |

### Result

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether execution succeeded |
| `message` | string | Status message |
| `actual_duration_seconds` | float32 | Actual duration of gesture execution |

### Feedback

| Field | Type | Description |
|-------|------|-------------|
| `elapsed_seconds` | float32 | Elapsed time during gesture execution |

## Gesture Types

### 1. Deictic Gestures (Pointing)
Points to a specific 3D location in the environment using inverse kinematics.

```bash
ros2 action send_goal /gesture_execution dec_interfaces/action/Gesture \
  "{gesture_type: 'deictic', gesture_name: '', gesture_duration: 2000, bow_nod_angle: 0, \
    location_x: 2.0, location_y: 1.5, location_z: 0.8}"
```

### 2. Iconic Gestures (Predefined Arm Motions)
Executes predefined arm motions from gesture.yaml:
- `welcome` - Welcome gesture (both arms + leg)
- `wave` - Wave gesture (right arm)
- `shake` - Handshake gesture (right arm)

```bash
ros2 action send_goal /gesture_execution dec_interfaces/action/Gesture \
  "{gesture_type: 'iconic', gesture_name: 'wave', gesture_duration: 3000, bow_nod_angle: 0, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"
```

### 3. Bowing Gestures
Bows the robot forward at a specified angle (5-45°).

```bash
ros2 action send_goal /gesture_execution dec_interfaces/action/Gesture \
  "{gesture_type: 'bow', gesture_name: '', gesture_duration: 2500, bow_nod_angle: 30, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"
```

### 4. Nodding Gestures
Nods the robot's head at a specified angle (5-30°).

```bash
ros2 action send_goal /gesture_execution dec_interfaces/action/Gesture \
  "{gesture_type: 'nod', gesture_name: '', gesture_duration: 1500, bow_nod_angle: 15, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"
```

## Package Structure

```
gesture_execution/
├── gesture_execution/
│   ├── __init__.py
│   ├── gesture_execution_application.py
│   ├── gesture_execution_implementation.py
│   ├── gesture_test_visualization.py
│   └── pepper_kinematics_utilities.py
├── config/
│   └── gesture_execution_configuration.yaml
├── data/
│   ├── gesture.yaml
│   └── pepper_topics.yaml
├── resource/
│   └── gesture_execution
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## Architecture

The gesture execution system uses Bézier interpolation for smooth motion:

1. **Action Goal**: Receive gesture parameters via ROS2 Action Server
2. **Gesture Validation**: Validate parameters against constraints and joint limits
3. **Trajectory Generation**: Calculate joint trajectories using Bézier interpolation
4. **Feedback Thread**: Start real-time feedback publishing of elapsed time
5. **Motion Execution**: Publish joint trajectories to robot controllers
6. **Visualization**: Publish RViz2 markers for deictic gestures (optional)
7. **Result**: Return success/failure status with actual duration

### Joint Coordinate System
1. ShoulderPitch
2. ShoulderRoll
3. ElbowYaw
4. ElbowRoll
5. WristYaw

### Safety Limits
- **Shoulder Pitch**: ±2.0857 rad (±119.5°)
- **Shoulder Roll**: Right: -1.5621 to -0.0087 rad, Left: 0.0087 to 1.5621 rad
- **Head Yaw**: ±2.0857 rad (±119.5°)
- **Head Pitch**: -0.7068 to 0.6371 rad (-40.5° to 36.5°)
- **Bowing Angle**: 5-45 degrees
- **Nodding Angle**: 5-30 degrees
- **Gesture Duration**: 1000-5000 milliseconds

## Testing

```bash
# Check node is running
ros2 node list

# Verify action server is available
ros2 action list

# Send a test gesture
ros2 action send_goal /gesture_execution dec_interfaces/action/Gesture \
  "{gesture_type: 'iconic', gesture_name: 'wave', gesture_duration: 3000, bow_nod_angle: 0, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"

# Monitor joint commands
ros2 topic echo /joint_angles_trajectory
```

### RViz2 Visualization
```bash
# Run the visualization test
ros2 run gesture_execution test_visualization

# In another terminal, launch RViz2
ros2 run rviz2 rviz2
```
**RViz2 Setup:**
1. Add a "Marker" display
2. Set topic to: `/gesture_execution/visualization`
3. Ensure "Global Options" → "Fixed Frame" is set to `base_link`

## Support

For issues or questions:
- Create an issue on the [pepper4dec GitHub repository](https://github.com/yohatad/pepper4dec/issues)
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a>

## License
Copyright (C) 2026 Upanzi Network
Licensed under the BSD-3-Clause License. See individual package licenses for details.
