<div align="center">
<h1> Gesture Execution for Pepper Robot (ROS2) </h1>
</div>

<div align="center">
  <img src="../upanzi-logo.svg" alt="Upanzi Logo" style="width:50%; height:auto;">
</div>

The **Gesture Execution** package is a **ROS2** package designed to execute various types of gestures on the Pepper humanoid robot. It provides an **Action Server** interface for executing deictic (pointing), iconic (predefined arm motions), bowing, and nodding gestures with smooth Bézier interpolation for natural motion. The system includes real-time feedback on gesture execution progress and comprehensive visualization support for debugging.

## Key Features
- **ROS2 Action Server**: Built for ROS2 Humble with action-based interface for long-running gestures with feedback
- **Multiple Gesture Types**: Supports deictic (pointing), iconic, bowing, and nodding gestures
- **Smooth Motion**: Uses Bézier interpolation for natural, fluid movements
- **Inverse Kinematics**: Calculates joint angles for pointing to specific 3D locations
- **Configurable**: YAML-based configuration and gesture definitions
- **Real-time Feedback**: Provides elapsed time feedback during gesture execution
- **Real-time Adaptation**: Adapts pointing gestures based on robot's current pose
- **Joint Limit Safety**: Validates all motions against robot joint limits
- **RViz2 Visualization**: Publishes markers for visualizing pointing gestures in RViz2

# 🛠️ Installation 

## Prerequisites
- **ROS2 Humble** or newer
- **Python 3.10** or compatible version
- **Pepper Robot** or simulator with NAOqi bridge
- **cssr_interfaces** package for action definitions

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
| `gestureDescriptors` | Path to gesture descriptor file | string | `gesture.yaml` |
| `robotTopics` | Path to robot topic mapping file | string | `pepper_topics.yaml` |
| `verboseMode` | Enable detailed logging and debugging | `true`, `false` | `true` |

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
# Run the gesture execution node (Action Server)
ros2 run gesture_execution gesture_execution
```

## Running Visualization Test
```bash
# Run the visualization test to verify RViz2 markers
ros2 run gesture_execution test_visualization
```

## Verification
To verify the node is running and actions are available:
```bash
# Check node status
ros2 node list

# Check available actions
ros2 action list

# Expected output:
# /gesture_execution/execute
```

# 🖥️ Action Interface
The node provides a single action server `/gesture_execution/execute` with the following goal structure:

## Action Goal (`cssr_interfaces/action/Gesture`)

| Parameter | Type | Description | Constraints |
|-----------|------|-------------|-------------|
| `gesture_type` | string | Type of gesture to execute | `"deictic"`, `"iconic"`, `"bow"`, `"nod"` |
| `gesture_id` | int32 | ID of predefined gesture (for iconic gestures) | Positive integer |
| `gesture_duration` | int32 | Duration of gesture in milliseconds | 1000-5000 ms |
| `bow_nod_angle` | int32 | Angle for bowing/nodding in degrees | 5-45° for bow, 5-30° for nod |
| `location_x` | float32 | X coordinate for pointing (meters) | Any real number |
| `location_y` | float32 | Y coordinate for pointing (meters) | Any real number |
| `location_z` | float32 | Z coordinate for pointing (meters) | Any real number |

## Action Feedback
- `elapsed_seconds`: Real-time feedback of elapsed time during gesture execution

## Action Result
- `success`: Boolean indicating gesture completion status
- `actual_duration_seconds`: Actual duration of gesture execution
- `message`: Status message

# 📋 Gesture Types

## 1. Deictic Gestures (Pointing)
Points to a specific 3D location in the environment. The robot uses inverse kinematics to calculate the appropriate joint angles. The robot's head will also track the pointing target for natural interaction.

**Example Action Goal:**
```bash
ros2 action send_goal /gesture_execution/execute cssr_interfaces/action/Gesture \
  "{gesture_type: 'deictic', gesture_id: 0, gesture_duration: 2000, bow_nod_angle: 0, \
    location_x: 2.0, location_y: 1.5, location_z: 0.8}"
```

## 2. Iconic Gestures (Predefined Arm Motions)
Executes predefined arm motions from the gesture.yaml file. Currently supported gestures:
- `id: 1` - Welcome gesture (both arms)
- `id: 2` - Wave gesture (right arm)
- `id: 3` - Handshake gesture (both arms)

**Example Action Goal:**
```bash
ros2 action send_goal /gesture_execution/execute cssr_interfaces/action/Gesture \
  "{gesture_type: 'iconic', gesture_id: 2, gesture_duration: 3000, bow_nod_angle: 0, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"
```

## 3. Bowing Gestures
Bows the robot forward at a specified angle.

**Example Action Goal:**
```bash
ros2 action send_goal /gesture_execution/execute cssr_interfaces/action/Gesture \
  "{gesture_type: 'bow', gesture_id: 0, gesture_duration: 2500, bow_nod_angle: 30, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"
```

## 4. Nodding Gestures
Nods the robot's head at a specified angle.

**Example Action Goal:**
```bash
ros2 action send_goal /gesture_execution/execute cssr_interfaces/action/Gesture \
  "{gesture_type: 'nod', gesture_id: 0, gesture_duration: 1500, bow_nod_angle: 15, \
    location_x: 0.0, location_y: 0.0, location_z: 0.0}"
```

# 🏗️ Architecture

## Core Components
1. **GestureExecutionSystem**: Main ROS2 Action Server managing gesture execution with feedback
2. **GestureDescriptorManager**: Loads and manages gesture definitions from YAML files
3. **ConfigManager**: Handles configuration and topic mappings
4. **PepperKinematicsUtilities**: Provides inverse kinematics for pointing gestures
5. **VisualizationTestNode**: Test utility for verifying RViz2 marker visualization

## Motion Execution Pipeline
1. **Action Goal**: Receive gesture parameters via ROS2 Action Server
2. **Gesture Validation**: Validate parameters against constraints and joint limits
3. **Trajectory Generation**: Calculate joint trajectories using Bézier interpolation
4. **Feedback Thread**: Start real-time feedback publishing of elapsed time
5. **Motion Execution**: Publish joint trajectories to robot controllers
6. **Visualization**: Publish RViz2 markers for deictic gestures (optional)
7. **Result**: Return success/failure status with actual duration

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
- **Head Yaw**: ±2.0857 rad (±119.5°)
- **Head Pitch**: -0.7068 to 0.6371 rad (-40.5° to 36.5°)
- **Bowing Angle**: 5-45 degrees
- **Nodding Angle**: 5-30 degrees
- **Gesture Duration**: 1000-5000 milliseconds

# 🎯 Visualization

## RViz2 Markers
The system publishes visualization markers to `/gesture_execution/visualization` for deictic gestures:
- **Red Sphere**: Target pointing location
- **Blue Sphere**: Robot shoulder position
- **Blue Arrow**: Pointing line from shoulder to target
- **White Text**: Coordinate labels

## Testing Visualization
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

# 🐛 Debugging and Troubleshooting

## Enabling Verbose Mode
Set `verboseMode: true` in `config/gesture_execution_configuration.yaml` to enable detailed logging.

## Common Issues
1. **Action Not Found**: Ensure the node is running and the workspace is sourced
2. **Joint Limit Errors**: Check that pointing locations are within reachable workspace
3. **Configuration Errors**: Verify YAML file syntax and file paths
4. **Motion Execution Failures**: Check robot connection and joint state topics
5. **Visualization Not Appearing**: Verify RViz2 configuration and marker topic

## Logging
The node provides different log levels:
- **INFO**: General operation messages
- **WARNING**: Parameter validation issues
- **ERROR**: Execution failures
- **DEBUG**: Detailed trajectory information (with verbose mode)

# 📦 Package Structure
```
gesture_execution/
├── gesture_execution/
│   ├── __init__.py
│   ├── gesture_execution_application.py     # Main entry point
│   ├── gesture_execution_implementation.py  # Action server implementation
│   ├── gesture_test_visualization.py        # RViz2 visualization test
│   └── pepper_kinematics_utilities.py       # Inverse kinematics utilities
├── config/
│   └── gesture_execution_configuration.yaml # Configuration file
├── data/
│   ├── gesture.yaml                         # Gesture definitions
│   └── pepper_topics.yaml                   # Topic mappings
├── launch/                                  # Launch files (if any)
├── resource/
│   └── gesture_execution                    # Package resource file
├── setup.py                                 # Python setup
├── setup.cfg                                # Configuration
├── package.xml                              # ROS2 package definition
└── README.md                                # This file
```

# 💡 Support

For issues or questions:
- Create an issue on GitHub
- Contact: <a href="mailto:yohatad123@gmail.com">yohatad123@gmail.com</a><br>
- Visit: <a href="http://www.cssr4africa.org">www.cssr4africa.org</a>

# 📜 License
Copyright (C) 2025 Upanzi Network   
Funded by African Engineering and Technology Network (Afretec)  
Inclusive Digital Transformation Research Grant Programme

2025-10-11