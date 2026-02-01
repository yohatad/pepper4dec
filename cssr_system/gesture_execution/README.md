<div align="center">
<h1> Gesture Execution for Pepper Robot (ROS2) </h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Gesture Execution** package is a **ROS2** package designed to execute various types of gestures on the Pepper humanoid robot. It provides a service interface for executing deictic (pointing), iconic (predefined arm motions), bowing, and nodding gestures with smooth Bézier interpolation for natural motion.

## Key Features
- **ROS2 Native**: Built for ROS2 Humble
- **Multiple Gesture Types**: Supports deictic (pointing), iconic, bowing, and nodding gestures
- **Smooth Motion**: Uses Bézier interpolation for natural, fluid movements
- **Inverse Kinematics**: Calculates joint angles for pointing to specific 3D locations
- **Configurable**: YAML-based configuration and gesture definitions
- **Service Interface**: ROS2 service for easy gesture triggering
- **Real-time Adaptation**: Adapts pointing gestures based on robot's current pose
- **Joint Limit Safety**: Validates all motions against robot joint limits

# 🛠️ Installation 

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **Pepper Robot** or simulator with NAOqi bridge
- **cssr_interfaces** package for service definitions

## Package Installation

1. **Clone and Build the Workspace**
```bash
# Clone the repository (if not already done)
cd ~/ros2_ws/src
git clone <repository-url>

# Build the workspace
cd ~/ros2_ws
colcon build --packages-select gesture_execution cssr_interfaces
source install/setup.bash
```

2. **Install Python Dependencies**
```bash
# Install required Python packages
pip install PyYAML
```

# 🔧 Configuration Parameters
The configuration is managed via `config/gesture_execution_configuration.yaml`:

| Parameter | Description | Values | Default |
|-----------|-------------|---------|---------|
| `gestureDescriptors` | Path to gesture descriptor file | string | `gestureDescriptors.dat` |
| `robotTopics` | Path to robot topic mapping file | string | `pepperTopics.dat` |
| `verboseMode` | Enable detailed logging and debugging | `true`, `false` | `false` |

## Gesture Definitions
Gestures are defined in `data/gesture.yaml` with the following structure:
```yaml
gestures:
  wave:
    id: 2
    joints: ["RArm"]
    joint_angles:
      RArm:
        - [1.7410, -0.09664, 1.6981, 0.09664, -0.05679]  # Waypoint 1
        - [0.0414, -0.7725, 1.4900, 0.5236, 0.0]          # Waypoint 2
    times:
      RArm: [1.0, 2.0]  # Timing for each waypoint
```

## Topic Mappings
Robot topics are configured in `data/pepper_topics.yaml`:
```yaml
topics:
  JointAngles: "/joint_angles"
  Wheels: "/cmd_vel"
  JointStates: "/joint_states"
  RobotPose: "/robotLocalization/pose"
```

# 🚀 Running the Node

## Building the Package
```bash
# Build the gesture_execution package
cd ~/ros2_ws
colcon build --packages-select gesture_execution
source install/setup.bash
```

## Starting the Node
```bash
# Run the gesture execution node
ros2 run gesture_execution gesture_execution
```

## Verification
To verify the node is running and services are available:
```bash
# Check node status
ros2 node list

# Check available services
ros2 service list | grep gesture

# Expected output:
# /gesture_execution/perform_gesture
```

# 🖥️ Service Interface
The node provides a single service `/gesture_execution/perform_gesture` with the following request parameters:

## Service Request (`cssr_interfaces/srv/PerformGesture`)

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `gesture_type` | string | Type of gesture to execute | `"deictic"`, `"iconic"`, `"bow"`, `"nod"` |
| `gesture_id` | int32 | ID of predefined gesture (for iconic gestures) | Positive integer |
| `speed` | int32 | Duration of gesture in milliseconds | 1000-10000 ms |
| `bow_nod_angle` | int32 | Angle for bowing/nodding in degrees | 5-45° for bow, 5-30° for nod |
| `location_x` | float32 | X coordinate for pointing (meters) | Any real number |
| `location_y` | float32 | Y coordinate for pointing (meters) | Any real number |
| `location_z` | float32 | Z coordinate for pointing (meters) | Any real number |

## Service Response
- `gesture_success`: 1 for success, 0 for failure

# 📋 Gesture Types

## 1. Deictic Gestures (Pointing)
Points to a specific 3D location in the environment. The robot uses inverse kinematics to calculate the appropriate joint angles.

**Example Service Call:**
```bash
ros2 service call /gesture_execution/perform_gesture cssr_interfaces/srv/PerformGesture \
  "{gesture_type: 'deictic', gesture_id: 0, speed: 2000, bow_nod_angle: 0, \
    location_x: 2.0, location_y: 1.5, location_z: 0.8}"
```

## 2. Iconic Gestures (Predefined Arm Motions)
Executes predefined arm motions from the gesture.yaml file. Currently supported gestures:
- `id: 1` - Welcome gesture (both arms)
- `id: 2` - Wave gesture (right arm)
- `id: 3` - Handshake gesture (both arms)

**Example Service Call:**
```bash
ros2 service call /gesture_execution/perform_gesture cssr_interfaces/srv/PerformGesture \
  "{gesture_type: 'iconic', gesture_id: 2, speed: 3000, bow_nod_angle: 0, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"
```

## 3. Bowing Gestures
Bows the robot forward at a specified angle.

**Example Service Call:**
```bash
ros2 service call /gesture_execution/perform_gesture cssr_interfaces/srv/PerformGesture \
  "{gesture_type: 'bow', gesture_id: 0, speed: 2500, bow_nod_angle: 30, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"
```

## 4. Nodding Gestures
Nods the robot's head at a specified angle.

**Example Service Call:**
```bash
ros2 service call /gesture_execution/perform_gesture cssr_interfaces/srv/PerformGesture \
  "{gesture_type: 'nod', gesture_id: 0, speed: 1500, bow_nod_angle: 15, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"
```

# 🏗️ Architecture

## Core Components
1. **GestureExecutionSystem**: Main ROS2 node managing service interface and gesture execution
2. **GestureDescriptorManager**: Loads and manages gesture definitions from YAML files
3. **ConfigManager**: Handles configuration and topic mappings
4. **PepperKinematicsUtilities**: Provides inverse kinematics for pointing gestures

## Motion Execution Pipeline
1. **Service Request**: Receive gesture parameters via ROS2 service
2. **Gesture Validation**: Validate parameters against constraints
3. **Trajectory Generation**: Calculate joint trajectories using Bézier interpolation
4. **Joint Limit Checking**: Ensure all motions are within safe limits
5. **Motion Execution**: Publish joint trajectories to robot controllers
6. **Response**: Return success/failure status

## Bézier Interpolation
The system uses cubic Bézier curves for smooth motion between waypoints, providing:
- Continuous velocity and acceleration profiles
- Natural-looking movements
- Reduced stress on robot joints
- Precise timing control

# ⚙️ Technical Details

## Joint Coordinate System
The system uses the following joint order for each arm:
1. ShoulderPitch
2. ShoulderRoll  
3. ElbowYaw
4. ElbowRoll
5. WristYaw

## Kinematic Parameters
- Upper arm length: 150 mm
- Shoulder offsets: X=-57.0mm, Y=±149.74mm, Z=86.82mm
- Torso height: 0.0mm (adjustable based on robot configuration)

## Safety Limits
- **Shoulder Pitch**: ±2.0857 rad (±119.5°)
- **Shoulder Roll**: Right: -1.5621 to -0.0087 rad, Left: 0.0087 to 1.5621 rad
- **Bowing Angle**: 5-45 degrees
- **Nodding Angle**: 5-30 degrees
- **Gesture Duration**: 1000-10000 milliseconds

# 🐛 Debugging and Troubleshooting

## Enabling Verbose Mode
Set `verboseMode: true` in `config/gesture_execution_configuration.yaml` to enable detailed logging.

## Common Issues
1. **Service Not Found**: Ensure the node is running and the workspace is sourced
2. **Joint Limit Errors**: Check that pointing locations are within reachable workspace
3. **Configuration Errors**: Verify YAML file syntax and file paths
4. **Motion Execution Failures**: Check robot connection and joint state topics

## Logging
The node provides different log levels:
- **INFO**: General operation messages
- **WARNING**: Parameter validation issues
- **ERROR**: Execution failures
- **DEBUG**: Detailed trajectory information (with verbose mode)

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohanneh@andrew.cmu.edu">yohanneh@andrew.cmu.edu</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2025 Upanzi Network   
Funded by African Engineering and Technology Network (Afretec)  
Inclusive Digital Transformation Research Grant Programme

2025-10-11