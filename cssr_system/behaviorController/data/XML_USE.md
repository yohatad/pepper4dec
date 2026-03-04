# Behavior Controller - XML Behavior Tree Reference

This document describes how to write XML behavior tree files for the `behaviorController` package. Tree files are placed in the `data/` folder and selected via the `scenarioSpecification` field in `behaviorControllerConfiguration.yaml`.

## File Structure

Every XML file must follow the BehaviorTree.CPP v4 format:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<root BTCPP_format="4"
      main_tree_to_execute="MainTreeID">

  <BehaviorTree ID="MainTreeID">
    <!-- nodes go here -->
  </BehaviorTree>

  <TreeNodesModel>
    <!-- node model declarations for Groot editor -->
  </TreeNodesModel>

</root>
```

- `main_tree_to_execute` specifies which `<BehaviorTree>` is the entry point.
- Multiple `<BehaviorTree>` blocks can be defined and referenced as subtrees.

## Control Flow Nodes

These are built-in BehaviorTree.CPP nodes:

| Node | Description |
|------|-------------|
| `<Sequence>` | Executes children left-to-right; fails on first failure |
| `<Fallback>` | Executes children left-to-right; succeeds on first success |
| `<Parallel success_count="N" failure_count="M">` | Runs children in parallel; succeeds when N succeed, fails when M fail |
| `<Inverter>` | Inverts child result (SUCCESS <-> FAILURE) |
| `<ForceFailure>` | Always returns FAILURE regardless of child result |
| `<ForceSuccess>` | Always returns SUCCESS regardless of child result |
| `<KeepRunningUntilFailure>` | Re-ticks child until it returns FAILURE |
| `<RetryUntilSuccessful num_attempts="N">` | Retries child up to N times until SUCCESS |
| `<SubTree ID="TreeID"/>` | Executes another `<BehaviorTree>` block by ID |

## Available Action Nodes

### ROS2 Action Nodes

These nodes communicate with ROS2 action servers. They return RUNNING while the action is executing, SUCCESS when it completes, or FAILURE on error.

#### TTSRosAction - Text-to-Speech

Sends text to the `/tts` action server.

```xml
<!-- Speak using a utility phrase key (looked up from knowledge base) -->
<Action ID="TTSRosAction"
        name="SayWelcome"
        phrase_key="welcome_message"
        language="English"/>

<!-- Speak direct text -->
<Action ID="TTSRosAction"
        name="SayHello"
        text="Hello, welcome to the lab!"/>
```

| Port | Type | Default | Description |
|------|------|---------|-------------|
| `action_name` | string | `/tts` | Action server name |
| `phrase_key` | string | — | Key to look up in utility phrases |
| `text` | string | — | Direct text to speak (takes priority over phrase_key) |
| `language` | string | config language | Language code |

#### Navigate - Navigation

Sends navigation goals to the `/navigation` action server.

```xml
<!-- Navigate using explicit coordinates -->
<Action ID="Navigate"
        name="GoToLobby"
        goal_x="1.5"
        goal_y="2.0"
        goal_theta="0.0"/>

<!-- Navigate using blackboard variable "exhibitLocation" (set by SelectExhibit) -->
<Action ID="Navigate"
        name="Navigate"/>
```

| Port | Type | Default | Description |
|------|------|---------|-------------|
| `action_name` | string | `/navigation` | Action server name |
| `goal_x` | double | — | Goal X coordinate |
| `goal_y` | double | — | Goal Y coordinate |
| `goal_theta` | double | — | Goal orientation (radians) |

If `goal_x`, `goal_y`, `goal_theta` are not provided, the node reads `exhibitLocation` from the blackboard.

#### GestureRosAction - Gestures

Sends gesture commands to the `/gesture` action server.

```xml
<!-- Deictic gesture pointing at a location -->
<Action ID="GestureRosAction"
        name="PointAtExhibit"
        gesture_type="deictic"
        gesture_id="1"
        gesture_duration="3000"
        location_x="2.0"
        location_y="1.0"
        location_z="1.2"/>

<!-- Gesture using blackboard target "exhibitGestureTarget" (set by SelectExhibit) -->
<Action ID="GestureRosAction"
        name="PerformDeicticGesture"/>
```

| Port | Type | Default | Description |
|------|------|---------|-------------|
| `action_name` | string | `/gesture` | Action server name |
| `gesture_type` | string | `deictic` | Gesture type |
| `gesture_id` | int | `1` | Gesture ID |
| `gesture_duration` | int | `3000` | Duration in milliseconds |
| `bow_nod_angle` | int | `0` | Bow/nod angle |
| `location_x` | double | — | Target X coordinate |
| `location_y` | double | — | Target Y coordinate |
| `location_z` | double | — | Target Z coordinate |

If location ports are not provided, the node reads `exhibitGestureTarget` from the blackboard.

#### SpeechRecognitionRosAction - Speech Recognition

Listens for visitor speech via the `/speech_recognition` action server. Stores the transcription in blackboard variable `visitorSpeech` and sets `visitorResponse` to `"yes"` or `"no"`.

```xml
<Action ID="SpeechRecognitionRosAction"
        name="ListenToVisitor"
        wait="5.0"/>
```

| Port | Type | Default | Description |
|------|------|---------|-------------|
| `action_name` | string | `/speech_recognition` | Action server name |
| `wait` | float | `5.0` | Wait time in seconds |

#### AnimateBehaviorRosAction - Idle Animation

Controls the `/animate_behavior` action server to start or stop idle body animations. Use `behavior_type="home"` to stop animation and return to the home position.

```xml
<!-- Start idle body animation (runs indefinitely) -->
<Action ID="AnimateBehaviorRosAction"
        name="AnimateIdle"
        behavior_type="body"
        selected_range="0.4"
        duration_seconds="0"/>

<!-- Stop animation and return to home position -->
<Action ID="AnimateBehaviorRosAction"
        name="StopAnimation"
        behavior_type="home"
        selected_range="0.0"
        duration_seconds="0"/>
```

| Port | Type | Default | Description |
|------|------|---------|-------------|
| `action_name` | string | `/animate_behavior` | Action server name |
| `behavior_type` | string | `body` | `All`, `body`, `hands`, `arms`, `rotation`, or `home` |
| `selected_range` | float | `0.4` | Movement amplitude (0.0 - 1.0) |
| `duration_seconds` | int | `0` | Duration in seconds (0 = infinite until cancelled) |

**Behavior types:**

| Type | Effect |
|------|--------|
| `All` | Full-body animation (arms, hands, legs, base) |
| `body` | Arms and legs only |
| `hands` | Hand open/close gestures only |
| `arms` | Arms and hands only |
| `rotation` | Base rotation only |
| `home` | Returns all limbs to neutral position (stops animation) |

### ROS2 Service Nodes

These nodes call ROS2 services. They block briefly and return SUCCESS or FAILURE.

#### SetOvertAttentionModeRosService - Attention Control

Sets the overt attention mode via `/overtAttention/set_mode`.

```xml
<Action ID="SetOvertAttentionModeRosService"
        name="SetScanning"
        state="scanning"/>

<Action ID="SetOvertAttentionModeRosService"
        name="LookAtExhibit"
        state="location"
        location_x="2.0"
        location_y="1.0"
        location_z="1.2"/>

<!-- Look at blackboard target "exhibitGestureTarget" -->
<Action ID="SetOvertAttentionModeRosService"
        name="LookAtExhibit"
        state="location"/>
```

| Port | Type | Default | Description |
|------|------|---------|-------------|
| `service_name` | string | `/overtAttention/set_mode` | Service name |
| `state` | string | node name | `scanning`, `seeking`, `social`, `location`, `disabled` |
| `location_x` | float | — | Target X (used when state=`location`) |
| `location_y` | float | — | Target Y |
| `location_z` | float | — | Target Z |

If `state` is not set, the node's `name` attribute is used as the state value.

#### SetAnimateBehaviorRosService - Activation Toggle

Enables or disables the animate behavior module via `/animateBehaviour/setActivation`.

```xml
<Action ID="SetAnimateBehaviorRosService"
        name="EnableAnimation"
        state="enabled"/>

<Action ID="SetAnimateBehaviorRosService"
        name="DisableAnimation"
        state="disabled"/>
```

| Port | Type | Default | Description |
|------|------|---------|-------------|
| `service_name` | string | `/animateBehaviour/setActivation` | Service name |
| `state` | string | node name | `enabled` or `disabled` |

If `state` is not set, the node's `name` attribute is used (e.g., `name="enabled"`).

#### ConversationPromptRosService - Conversation

Sends a prompt to the conversation manager via `/conversation/prompt`.

```xml
<!-- Send a direct prompt -->
<Action ID="ConversationPromptRosService"
        name="AskAboutExhibit"
        prompt="Tell me about this painting"/>

<!-- Use blackboard variable "visitorSpeech" (set by SpeechRecognitionRosAction) -->
<Action ID="ConversationPromptRosService"
        name="ProcessVisitorSpeech"/>
```

| Port | Type | Default | Description |
|------|------|---------|-------------|
| `service_name` | string | `/conversation/prompt` | Service name |
| `prompt` | string | — | Text prompt (falls back to blackboard `visitorSpeech`) |

Stores the response in blackboard variable `conversationResponse`.

### Custom Nodes

These are standalone nodes that do not call external ROS2 services/actions.

| Node ID | Type | Description |
|---------|------|-------------|
| `StartOfTree` | Action | Logs tree start and current language. Place at beginning of main tree. |
| `IsVisitorDiscovered` | Condition | Subscribes to `/faceDetection/data`; returns SUCCESS when a face is detected. Stays RUNNING until then. |
| `IsMutualGazeDiscovered` | Condition | Subscribes to `/overtAttention/status`; returns SUCCESS when mutual gaze is detected. Stays RUNNING until then. |
| `SelectExhibit` | Action | Reads the current visit index from blackboard, loads exhibit info, and sets blackboard variables: `exhibitPreGestureMessage`, `exhibitPostGestureMessage`, `exhibitLocation`, `exhibitGestureTarget`. Increments `visits`. |
| `IsListWithExhibit` | Condition | Returns SUCCESS if there are more exhibits to visit, FAILURE if all have been visited. |
| `RetrieveListOfExhibits` | Action | Loads the tour specification and initializes `visits` to 0 on the blackboard. |
| `IsVisitorResponseYes` | Condition | Checks blackboard `visitorResponse`; returns SUCCESS if `"yes"`, FAILURE otherwise. |
| `HandleFallBack` | Action | Always returns SUCCESS. Used inside `<Fallback>` to gracefully handle errors. |

Usage:

```xml
<StartOfTree/>
<IsVisitorDiscovered/>
<SelectExhibit/>
<IsListWithExhibit/>
<RetrieveListOfExhibits/>
<IsMutualGazeDiscovered/>
<IsVisitorResponseYes/>
<HandleFallBack/>
```

## Blackboard Variables

The blackboard is shared state accessible by all nodes in the tree.

| Variable | Type | Set By | Used By |
|----------|------|--------|---------|
| `visits` | int | `RetrieveListOfExhibits`, `SelectExhibit` | `SelectExhibit`, `IsListWithExhibit` |
| `exhibitLocation` | RobotPose | `SelectExhibit` | `Navigate` |
| `exhibitGestureTarget` | Position3D | `SelectExhibit` | `GestureRosAction`, `SetOvertAttentionModeRosService` |
| `exhibitPreGestureMessage` | string | `SelectExhibit` | TTS nodes |
| `exhibitPostGestureMessage` | string | `SelectExhibit` | TTS nodes |
| `visitorSpeech` | string | `SpeechRecognitionRosAction` | `ConversationPromptRosService` |
| `visitorResponse` | string | `SpeechRecognitionRosAction` | `IsVisitorResponseYes` |
| `conversationResponse` | string | `ConversationPromptRosService` | — |

## Common Patterns

### Error Handling with Fallback

Wrap actions in `<Fallback>` with `<HandleFallBack/>` so the tree continues even if an action fails:

```xml
<Fallback>
  <Action ID="TTSRosAction" name="Greet" text="Hello!"/>
  <HandleFallBack/>
</Fallback>
```

### Start/Stop Animation Around an Action

Disable animation before a precise action, then re-enable:

```xml
<Sequence>
  <Action ID="SetAnimateBehaviorRosService" name="DisableAnim" state="disabled"/>
  <Action ID="GestureRosAction" name="PointAtExhibit"/>
  <Action ID="SetAnimateBehaviorRosService" name="EnableAnim" state="enabled"/>
</Sequence>
```

### Idle Animation with AnimateBehaviorRosAction

Start body animation during idle, stop it when transitioning to a precise task:

```xml
<Sequence>
  <!-- Start idle animation -->
  <Action ID="AnimateBehaviorRosAction"
          name="AnimateIdle"
          behavior_type="body"
          selected_range="0.4"
          duration_seconds="0"/>

  <!-- ... do other things ... -->

  <!-- Stop and return to home -->
  <Action ID="AnimateBehaviorRosAction"
          name="StopAnimation"
          behavior_type="home"
          selected_range="0.0"
          duration_seconds="0"/>
</Sequence>
```

### Parallel Actions

Run attention and animation setup simultaneously:

```xml
<Parallel success_count="2" failure_count="2">
  <Fallback>
    <Action ID="SetAnimateBehaviorRosService" name="enabled"/>
    <HandleFallBack/>
  </Fallback>
  <Fallback>
    <Action ID="SetOvertAttentionModeRosService" name="scanning"/>
    <HandleFallBack/>
  </Fallback>
</Parallel>
```

### Subtree Decomposition

Break complex scenarios into named subtrees:

```xml
<BehaviorTree ID="MainTree">
  <Sequence>
    <StartOfTree/>
    <SubTree ID="Phase1_Detect"/>
    <SubTree ID="Phase2_Engage"/>
    <SubTree ID="Phase3_Tour"/>
  </Sequence>
</BehaviorTree>

<BehaviorTree ID="Phase1_Detect">
  <!-- detection logic -->
</BehaviorTree>
```

## TreeNodesModel

For Groot editor compatibility, declare all custom nodes in `<TreeNodesModel>`:

```xml
<TreeNodesModel>
  <Action ID="TTSRosAction" editable="true"/>
  <Action ID="Navigate" editable="true"/>
  <Action ID="GestureRosAction" editable="true"/>
  <Action ID="SpeechRecognitionRosAction" editable="true"/>
  <Action ID="AnimateBehaviorRosAction" editable="true"/>
  <Action ID="SetOvertAttentionModeRosService" editable="true"/>
  <Action ID="SetAnimateBehaviorRosService" editable="true"/>
  <Action ID="ConversationPromptRosService" editable="true"/>
  <Action ID="StartOfTree" editable="true"/>
  <Action ID="IsVisitorDiscovered" editable="true"/>
  <Action ID="SelectExhibit" editable="true"/>
  <Condition ID="IsListWithExhibit" editable="true"/>
  <Action ID="RetrieveListOfExhibits" editable="true"/>
  <Condition ID="IsMutualGazeDiscovered" editable="true"/>
  <Condition ID="IsVisitorResponseYes" editable="true"/>
  <Action ID="HandleFallBack" editable="true"/>
</TreeNodesModel>
```

## Adding a New Tree

1. Create a new `.xml` file in the `data/` folder (e.g., `my_scenario.xml`)
2. Set `scenarioSpecification: my_scenario` in `behaviorControllerConfiguration.yaml`
3. Rebuild the package so the new file is installed:
   ```bash
   cd ~/ros2_ws && colcon build --packages-select behavior_controller && source install/setup.bash
   ```
