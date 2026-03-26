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
- Multiple `<BehaviorTree>` blocks can be defined and referenced as subtrees via `<SubTree ID="..."/>`.

---

## Control Flow Nodes

Built-in BehaviorTree.CPP nodes — no registration required.

| Node | Description |
|------|-------------|
| `<Sequence>` | Runs children left-to-right; stops and returns FAILURE on the first failure |
| `<ReactiveSequence>` | Like Sequence but re-ticks all previous children every tick |
| `<Fallback>` | Runs children left-to-right; stops and returns SUCCESS on the first success |
| `<Parallel success_count="N" failure_count="M">` | Runs children concurrently; succeeds when N succeed, fails when M fail |
| `<IfThenElse>` | First child is the condition; runs second child on SUCCESS, third on FAILURE |
| `<WhileDoElse>` | First child is the condition; runs second child while true, third child when false |
| `<Inverter>` | Inverts child result (SUCCESS ↔ FAILURE) |
| `<ForceSuccess>` | Always returns SUCCESS regardless of child result |
| `<ForceFailure>` | Always returns FAILURE regardless of child result |
| `<AlwaysSuccess/>` | Leaf that always returns SUCCESS |
| `<AlwaysFailure/>` | Leaf that always returns FAILURE |
| `<RetryUntilSuccessful num_attempts="N">` | Retries child up to N times until it returns SUCCESS |
| `<KeepRunningUntilFailure>` | Re-ticks child until it returns FAILURE |
| `<SubTree ID="TreeID"/>` | Executes another `<BehaviorTree>` block by ID |

---

## Available Action Nodes

### `AnimateBehavior` — Idle Body Animation

Sends a goal to the `animate_behavior` action server to start continuous idle animations. Returns RUNNING while the animation is executing, SUCCESS when it completes (or `duration_seconds=0` for indefinite).

```xml
<!-- Start full-body idle animation (runs indefinitely) -->
<Action ID="AnimateBehavior"
        behavior_type="All"
        selected_range="0.5"
        duration_seconds="0"/>

<!-- Animate hands only at low amplitude for 5 seconds -->
<Action ID="AnimateBehavior"
        behavior_type="hands"
        selected_range="0.3"
        duration_seconds="5"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `behavior_type` | input | string | `All` | `All`, `body`, `hands`, `rotation` |
| `selected_range` | input | float | `0.5` | Movement amplitude `[0.0, 1.0]` |
| `duration_seconds` | input | int | `0` | Duration in seconds (`0` = indefinite) |
| `message` | output | string | — | Result message from the action server |

**Behavior types:**

| Type | Effect |
|------|--------|
| `All` | Full-body animation |
| `body` | Arms and legs only |
| `hands` | Hand open/close gestures only |
| `rotation` | Base rotation only |

---

### `StopAnimateBehavior` — Stop Animation

Calls the `animate_behavior/stop` service (std_srvs/Trigger) to immediately halt any ongoing animation.

```xml
<Action ID="StopAnimateBehavior"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `message` | output | string | — | Response message from the stop service |

---

### `Gesture` — Perform a Gesture

Sends a goal to the `Gesture` action server. Supports both iconic gestures (wave, welcome, goodbye) and deictic gestures (pointing to a 3D location).

```xml
<!-- Iconic gesture (e.g., wave, welcome, goodbye) -->
<Action ID="Gesture"
        gesture_type="welcome"
        gesture_duration="2000"/>

<!-- Deictic gesture pointing at a 3D location -->
<Action ID="Gesture"
        gesture_type="point"
        location_x="{exhibit_location_x}"
        location_y="{exhibit_location_y}"
        location_z="{exhibit_location_z}"
        gesture_duration="2000"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `gesture_type` | input | string | `""` | Gesture type (e.g. `wave`, `point`, `welcome`, `goodbye`) |
| `gesture_id` | input | int64 | `0` | Gesture ID |
| `gesture_duration` | input | int64 | `0` | Duration in **milliseconds** |
| `bow_nod_angle` | input | int64 | `0` | Bow/nod angle in degrees |
| `location_x` | input | double | `0.0` | Target X (metres) |
| `location_y` | input | double | `0.0` | Target Y (metres) |
| `location_z` | input | double | `0.0` | Target Z (metres) |
| `message` | output | string | — | Result message from the action server |
| `elapsed_seconds` | output | float | — | Feedback: elapsed time in seconds |

---

### `Navigate` — Navigate to Pose

Sends a `nav2_msgs/NavigateToPose` goal to the `/navigate_to_pose` action server. Returns RUNNING during navigation, SUCCESS on arrival, FAILURE otherwise.

```xml
<!-- Navigate to explicit coordinates -->
<Action ID="Navigate"
        goal_x="1.5"
        goal_y="2.0"
        goal_theta="0.0"
        frame_id="map"/>

<!-- Navigate to exhibit location stored on blackboard -->
<Action ID="Navigate"
        goal_x="{exhibit_goal_x}"
        goal_y="{exhibit_goal_y}"
        goal_theta="{exhibit_goal_theta}"
        frame_id="map"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `goal_x` | input | double | — | Goal X position (metres) — **required** |
| `goal_y` | input | double | — | Goal Y position (metres) — **required** |
| `goal_theta` | input | double | `0.0` | Goal heading (radians) |
| `frame_id` | input | string | `map` | Coordinate frame for the goal pose |
| `distance_remaining` | output | float | — | Feedback: metres remaining to goal |
| `recoveries` | output | int | — | Feedback: number of Nav2 recovery attempts |

---

### `SpeechWithFeedback` — Text-to-Speech

Sends text to the `/speech_with_feedback` action server (naoqi_bridge_msgs/SpeechWithFeedback). Supports `\mrk=N\` bookmark syntax for synchronising gestures with speech.

```xml
<!-- Speak a fixed string -->
<Action ID="SpeechWithFeedback" say="Hello, welcome to the museum!"/>

<!-- Speak text stored on the blackboard -->
<Action ID="SpeechWithFeedback" say="{welcome_speech}"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `say` | input | string | `""` | Text to speak (supports `\mrk=N\` bookmarks) |
| `started` | output | bool | — | Feedback: true once speech has started |
| `bookmark` | output | int | — | Feedback: current bookmark ID (`-1` if none) |
| `current_word` | output | string | — | Feedback: word currently being spoken |

---

### `SpeechRecognition` — Automatic Speech Recognition

Sends a goal to the `/speech_recognition_action` action server. Returns FAILURE if the transcription is empty.

```xml
<Action ID="SpeechRecognition"
        wait="5.0"
        transcription="{visitor_speech}"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `action_name` | input | string | `/speech_recognition_action` | Action server name |
| `wait` | input | float | `2.0` | Seconds to wait for speech input |
| `transcription` | output | string | — | Recognised speech text |
| `status` | output | string | — | Feedback: `waiting` \| `speech` \| `transcribing` |

---

### `ConversationManager` — LLM Conversation

Sends a natural-language prompt to the `/prompt` action server and stores the response.

```xml
<!-- Send a fixed prompt -->
<Action ID="ConversationManager"
        prompt="Tell me about this painting."
        response="{llm_response}"/>

<!-- Forward visitor speech to the conversation manager -->
<Action ID="ConversationManager"
        prompt="{visitor_speech}"
        response="{llm_response}"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `action_name` | input | string | `/prompt` | Action server name |
| `prompt` | input | string | `""` | Natural-language prompt — **required** |
| `response` | output | string | — | Reply from the conversation manager |
| `status` | output | string | — | Feedback: `searching` \| `generating` |

---

### `SetOvertAttention` — Attention Enable/Disable

Calls the `/attn/set_enabled` service (std_srvs/SetBool) to enable or disable the overt attention system.

```xml
<!-- Enable attention (robot tracks faces/saliency) -->
<Action ID="SetOvertAttention" enabled="true"/>

<!-- Disable attention (head returns to default pose) -->
<Action ID="SetOvertAttention" enabled="false"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `enabled` | input | bool | `true` | `true` = enable attention, `false` = disable |
| `message` | output | string | — | Response message from the service |

---

### `SetSpeechListening` — Microphone Enable/Disable

Calls the `/speech_event/set_enabled` service (std_srvs/SetBool) to mute or unmute the speech recognition microphone. Mute during TTS to avoid self-echo.

```xml
<!-- Enable microphone -->
<Action ID="SetSpeechListening" enabled="true"/>

<!-- Mute microphone (e.g., while robot is speaking) -->
<Action ID="SetSpeechListening" enabled="false"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `enabled` | input | bool | `true` | `true` = listen, `false` = mute |
| `message` | output | string | — | Response message from the service |

---

## Available Condition Nodes

### `CheckFaceDetected` — Face Detection

Subscribes to `/faceDetection/data`. Returns RUNNING until the face condition is met, then SUCCESS. Never times out on its own — wrap in a timeout decorator if needed.

```xml
<!-- Succeed on any face detected -->
<Condition ID="CheckFaceDetected" require_mutual_gaze="false"/>

<!-- Succeed only when mutual gaze is established -->
<Condition ID="CheckFaceDetected" require_mutual_gaze="true"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `require_mutual_gaze` | input | bool | `true` | If true, only succeeds when mutual gaze is detected |
| `face_count` | output | int | — | Number of faces in the latest message |
| `mutual_gaze` | output | bool | — | True if any detected face has mutual gaze |
| `face_id` | output | string | — | Label ID of the first detected face |
| `face_x` | output | double | — | Centroid X of first face (pixels) |
| `face_y` | output | double | — | Centroid Y of first face (pixels) |
| `face_depth` | output | double | — | Depth of first face (metres) |

---

### `ListenForSpeech` — Speech Event Listener

Subscribes to `/speech_event/text`. Returns RUNNING until a new transcription arrives **after** this node started (stale messages are ignored). Returns SUCCESS with the new transcription.

```xml
<Condition ID="ListenForSpeech" transcription="{visitor_speech}"/>
```

| Port | Direction | Type | Default | Description |
|------|-----------|------|---------|-------------|
| `transcription` | output | string | — | Recognised speech text |

---

## Blackboard

The blackboard is a shared key-value store accessible by all nodes in the tree. Reference blackboard values in XML using the `{key}` syntax.

### Writing to the Blackboard

Use the built-in `SetBlackboardValue` action:

```xml
<Action ID="SetBlackboardValue" key="tour_declined" value="true"/>
```

### Reading from the Blackboard

Use the built-in `CheckBlackboard` condition:

```xml
<Condition ID="CheckBlackboard" key="tour_declined" expected="true"/>
```

### Blackboard Variables Used in `dec_Tour.xml`

| Key | Type | Set By | Used By |
|-----|------|--------|---------|
| `welcome_speech` | string | caller / init | `SpeechWithFeedback` |
| `query_tour_speech` | string | caller / init | `SpeechWithFeedback` |
| `press_yes_no_speech` | string | caller / init | `SpeechWithFeedback` |
| `maybe_another_time_speech` | string | caller / init | `SpeechWithFeedback` |
| `follow_me_speech` | string | caller / init | `SpeechWithFeedback` |
| `brief_goodbye_speech` | string | caller / init | `SpeechWithFeedback` |
| `goodbye_speech` | string | caller / init | `SpeechWithFeedback` |
| `exhibit_goal_x` | double | `SelectExhibit` | `Navigate` |
| `exhibit_goal_y` | double | `SelectExhibit` | `Navigate` |
| `exhibit_goal_theta` | double | `SelectExhibit` | `Navigate` |
| `exhibit_location_x` | double | `SelectExhibit` | `Gesture` |
| `exhibit_location_y` | double | `SelectExhibit` | `Gesture` |
| `exhibit_location_z` | double | `SelectExhibit` | `Gesture` |
| `tour_declined` | string | `SetBlackboardValue` | `CheckBlackboard` |

---

## Common Patterns

### Stop Animation Before a Gesture, Restart After

```xml
<Sequence>
  <Action ID="StopAnimateBehavior"/>
  <Action ID="Gesture" gesture_type="welcome" gesture_duration="2000"/>
  <Action ID="AnimateBehavior" behavior_type="All" selected_range="0.5" duration_seconds="0"/>
</Sequence>
```

### Disable Attention During Navigation, Re-enable After

```xml
<Sequence>
  <Action ID="SetOvertAttention" enabled="false"/>
  <Action ID="Navigate" goal_x="{exhibit_goal_x}" goal_y="{exhibit_goal_y}" goal_theta="{exhibit_goal_theta}" frame_id="map"/>
  <Action ID="SetOvertAttention" enabled="true"/>
</Sequence>
```

### Mute Microphone During TTS

```xml
<Sequence>
  <Action ID="SetSpeechListening" enabled="false"/>
  <Action ID="SpeechWithFeedback" say="{welcome_speech}"/>
  <Action ID="SetSpeechListening" enabled="true"/>
</Sequence>
```

### Try ASR, Fallback to Button

```xml
<Fallback name="GetResponseMethod">
  <Sequence name="ASRResponse">
    <Condition ID="IsASREnabled"/>
    <Action ID="SetSpeechListening" enabled="true"/>
    <Action ID="GetVisitorResponse" timeout="10.0" response_type="asr"/>
    <Action ID="SetSpeechListening" enabled="false"/>
  </Sequence>
  <RetryUntilSuccessful num_attempts="3">
    <Sequence>
      <Action ID="SpeechWithFeedback" say="{press_yes_no_speech}"/>
      <Action ID="PressYesNoDialogue" timeout="15.0"/>
    </Sequence>
  </RetryUntilSuccessful>
</Fallback>
```

### Navigation with Retry and Recovery

```xml
<RetryUntilSuccessful num_attempts="3">
  <Fallback>
    <Action ID="Navigate"
            goal_x="{exhibit_goal_x}"
            goal_y="{exhibit_goal_y}"
            goal_theta="{exhibit_goal_theta}"
            frame_id="map"/>
    <Sequence>
      <Action ID="LogEvent" level="warn" message="Navigation failed, attempting recovery"/>
      <Action ID="ClearCostmaps"/>
      <Action ID="BackUp" distance="0.3" timeout="5.0"/>
      <AlwaysFailure/>
    </Sequence>
  </Fallback>
</RetryUntilSuccessful>
```

### Optional Condition with Graceful Fallback

```xml
<Fallback name="EstablishGaze">
  <Condition ID="CheckFaceDetected" require_mutual_gaze="true"/>
  <Action ID="LogEvent" level="warn" message="Mutual gaze not established, continuing"/>
</Fallback>
```

### Subtree Decomposition

```xml
<BehaviorTree ID="TourGuide">
  <Sequence>
    <SubTree ID="DetectVisitor"/>
    <SubTree ID="EngageVisitor"/>
    <SubTree ID="VisitExhibit"/>
    <SubTree ID="EndTour"/>
  </Sequence>
</BehaviorTree>

<BehaviorTree ID="DetectVisitor">
  <!-- detection logic -->
</BehaviorTree>
```

---

## TreeNodesModel

Declare all custom nodes in `<TreeNodesModel>` for Groot editor compatibility. Port declarations must match the C++ `providedPorts()` definitions exactly.

```xml
<TreeNodesModel>
  <!-- Actions -->
  <Action ID="AnimateBehavior">
    <input_port name="behavior_type" default="All">all | body | hands | rotation</input_port>
    <input_port name="selected_range" default="0.5">Movement range [0.0, 1.0]</input_port>
    <input_port name="duration_seconds" default="0">Duration in seconds (0 = indefinite)</input_port>
    <output_port name="message">Result message from action server</output_port>
  </Action>

  <Action ID="StopAnimateBehavior">
    <output_port name="message">Response message from the stop service</output_port>
  </Action>

  <Action ID="Gesture">
    <input_port name="gesture_type" default="">Gesture type (e.g. wave, point)</input_port>
    <input_port name="gesture_id" default="0">Gesture ID</input_port>
    <input_port name="gesture_duration" default="0">Duration in ms</input_port>
    <input_port name="bow_nod_angle" default="0">Bow/nod angle in degrees</input_port>
    <input_port name="location_x" default="0.0">Target x (metres)</input_port>
    <input_port name="location_y" default="0.0">Target y (metres)</input_port>
    <input_port name="location_z" default="0.0">Target z (metres)</input_port>
    <output_port name="message">Result message from action server</output_port>
    <output_port name="elapsed_seconds">Feedback: elapsed time in seconds</output_port>
  </Action>

  <Action ID="Navigate">
    <input_port name="goal_x">Goal x position (metres)</input_port>
    <input_port name="goal_y">Goal y position (metres)</input_port>
    <input_port name="goal_theta" default="0.0">Goal heading (radians)</input_port>
    <input_port name="frame_id" default="map">Coordinate frame for the goal pose</input_port>
    <output_port name="distance_remaining">Feedback: metres remaining to goal</output_port>
    <output_port name="recoveries">Feedback: number of recovery attempts</output_port>
  </Action>

  <Action ID="SpeechWithFeedback">
    <input_port name="say" default="">Text to speak (supports \mrk=N\ bookmarks)</input_port>
    <output_port name="started">Feedback: true once speech begins</output_port>
    <output_port name="bookmark">Feedback: current bookmark ID (-1 if none)</output_port>
    <output_port name="current_word">Feedback: word currently being spoken</output_port>
  </Action>

  <Action ID="SpeechRecognition">
    <input_port name="action_name" default="/speech_recognition_action">Action server name</input_port>
    <input_port name="wait" default="2.0">Seconds to wait for speech input</input_port>
    <output_port name="transcription">Recognised speech text</output_port>
    <output_port name="status">Feedback: waiting | speech | transcribing</output_port>
  </Action>

  <Action ID="ConversationManager">
    <input_port name="action_name" default="/prompt">Action server name</input_port>
    <input_port name="prompt" default="">Natural-language prompt to send</input_port>
    <output_port name="response">Reply from the conversation manager</output_port>
    <output_port name="status">Feedback: searching | generating</output_port>
  </Action>

  <Action ID="SetOvertAttention">
    <input_port name="enabled" default="true">true = enable attention, false = disable</input_port>
    <output_port name="message">Response message from the service</output_port>
  </Action>

  <Action ID="SetSpeechListening">
    <input_port name="enabled" default="true">true = listen, false = mute</input_port>
    <output_port name="message">Response message from the service</output_port>
  </Action>

  <!-- Conditions -->
  <Condition ID="CheckFaceDetected">
    <input_port name="require_mutual_gaze" default="true">Succeed only when mutual gaze is detected</input_port>
    <output_port name="face_count">Number of faces in the latest message</output_port>
    <output_port name="mutual_gaze">True if any face has mutual gaze</output_port>
    <output_port name="face_id">Label ID of the first detected face</output_port>
    <output_port name="face_x">Centroid x of first face (pixels)</output_port>
    <output_port name="face_y">Centroid y of first face (pixels)</output_port>
    <output_port name="face_depth">Depth of first face (metres)</output_port>
  </Condition>

  <Condition ID="ListenForSpeech">
    <output_port name="transcription">Recognised speech text from /speech_event/text</output_port>
  </Condition>
</TreeNodesModel>
```

---

## Adding a New Tree

1. Create a new `.xml` file in the `data/` folder (e.g., `my_scenario.xml`)
2. Set `scenarioSpecification: my_scenario` in `behaviorControllerConfiguration.yaml`
3. Rebuild the package so the new file is installed:
   ```bash
   cd ~/ros2_ws && colcon build --packages-select behavior_controller && source install/setup.bash
   ```
