"""remote_dialogue_implementation.py

RemoteDialogueNode: Pepper's side of the /ws/pepper voice server.

The server does speech recognition, voice activity detection, turn-taking, the
language model and speech synthesis. This node only moves audio and messages:

    /audio ──► one mic channel, 48 kHz → 16 kHz, 20 ms frames ──► server
               (zeros instead of live audio whenever the gate is closed)
    server ──► reply PCM ──► 48 kHz stereo, paced pieces ──► send_audio_buffer
    server ──► state / transcript / language ──► ROS topics

The microphone gate is open only while a RemoteDialogueTurn goal is listening
(or set_listening has enabled it), and never while the robot's own reply is
playing or within `mic_guard_s` after it ends. `playback_done` is sent when
the last sample handed to the robot has played, computed from the samples
sent; see remote_dialogue_logic.PlaybackScheduler.

The ROS interface is documented in remote_dialogue_application.py.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 26, 2026
Version: v1.0

Copyright (C) 2026 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import os
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from dec_interfaces.action import RemoteDialogueTurn
from dec_interfaces.msg import DialogueTranscript
from dec_interfaces.srv import ResetDialogue, SetDialogueLanguage
from naoqi_bridge_msgs.msg import AudioBuffer
from naoqi_bridge_msgs.srv import SendAudioBuffer
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from .remote_dialogue_audio import (FrameChunker, MicrophoneConverter,
                                    mono_to_robot_stereo)
from .remote_dialogue_logic import MicrophoneGate, PlaybackScheduler, TurnTracker
from .remote_dialogue_session import DialogueSession, SUPPORTED_OUTPUT_RATES

TOKEN_ENV = 'REMOTE_DIALOGUE_TOKEN'
URL_ENV = 'REMOTE_DIALOGUE_URL'


class RemoteDialogueNode(LifecycleNode):
    """Lifecycle node bridging Pepper's microphone and speaker to the voice server."""

    def __init__(self, node_name='remote_dialogue', **kwargs):
        """Declare parameters; everything else happens in on_configure."""
        super().__init__(node_name, **kwargs)
        self.declare_parameter('server_url', '')
        self.declare_parameter('language', 'eng')
        self.declare_parameter('output_sample_rate', 48000)
        self.declare_parameter('greet_on_connect', False)
        self.declare_parameter('microphone_topic', '/audio')
        self.declare_parameter('microphone_channel', int(AudioBuffer.CHANNEL_FRONT_LEFT))
        self.declare_parameter('send_audio_buffer_service', '/naoqi_driver/send_audio_buffer')
        self.declare_parameter('playback_piece_s', 0.2)
        self.declare_parameter('playback_lead_s', 0.15)
        self.declare_parameter('playback_tail_s', 0.15)
        self.declare_parameter('mic_guard_s', 0.3)
        self.declare_parameter('turn_timeout_s', 90.0)
        self.declare_parameter('language_timeout_s', 5.0)
        self.declare_parameter('reconnect_interval_s', 3.0)
        self.declare_parameter('verbose_mode', False)

        self.session = None
        self.audio_sub = None
        self._lock = threading.RLock()
        self._turn = None
        self._turn_changed = threading.Condition(self._lock)
        self._language_confirmed = threading.Event()
        self._playback_thread = None
        self._playback_stop = threading.Event()
        self._speaker_missing_warned = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _vlog(self, msg):
        if self.verbose:
            self.get_logger().info(f'[VERBOSE] {msg}')

    def _read_parameters(self):
        p = self.get_parameter
        self.server_url = os.environ.get(URL_ENV) or p('server_url').value
        self.token = os.environ.get(TOKEN_ENV, '')
        self.language = p('language').value
        self.output_rate = int(p('output_sample_rate').value)
        self.greet_on_connect = bool(p('greet_on_connect').value)
        self.microphone_topic = p('microphone_topic').value
        self.microphone_channel = int(p('microphone_channel').value)
        self.speaker_service = p('send_audio_buffer_service').value
        self.piece_s = float(p('playback_piece_s').value)
        self.lead_s = float(p('playback_lead_s').value)
        self.tail_s = float(p('playback_tail_s').value)
        self.mic_guard_s = float(p('mic_guard_s').value)
        self.turn_timeout_s = float(p('turn_timeout_s').value)
        self.language_timeout_s = float(p('language_timeout_s').value)
        self.reconnect_interval_s = float(p('reconnect_interval_s').value)
        self.verbose = bool(p('verbose_mode').value)

        errors = []
        if not self.server_url.startswith(('ws://', 'wss://')):
            errors.append(f'server_url must start with ws:// or wss:// (set it or ${URL_ENV})')
        if not self.token:
            errors.append(f'${TOKEN_ENV} is not set')
        if self.output_rate not in SUPPORTED_OUTPUT_RATES:
            errors.append(f'output_sample_rate must be one of {SUPPORTED_OUTPUT_RATES}')
        if not self.language:
            errors.append('language must not be empty')
        for name in ('playback_piece_s', 'playback_lead_s', 'turn_timeout_s',
                     'language_timeout_s', 'reconnect_interval_s'):
            if float(p(name).value) <= 0.0:
                errors.append(f'{name} must be > 0')
        for name in ('playback_tail_s', 'mic_guard_s'):
            if float(p(name).value) < 0.0:
                errors.append(f'{name} must be >= 0')
        return errors

    # ── lifecycle ────────────────────────────────────────────────────────────

    def on_configure(self, state):
        """Validate parameters and create the ROS interface."""
        errors = self._read_parameters()
        if errors:
            for error in errors:
                self.get_logger().error(f'Configuration error: {error}')
            return TransitionCallbackReturn.FAILURE

        self._services_group = ReentrantCallbackGroup()
        self._audio_group = MutuallyExclusiveCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, '/remote_dialogue/state', 10)
        self.transcript_pub = self.create_publisher(
            DialogueTranscript, '/remote_dialogue/transcript', 10)
        self.language_pub = self.create_publisher(String, '/remote_dialogue/language', latched)
        self.connected_pub = self.create_publisher(Bool, '/remote_dialogue/connected', latched)

        self.speaker_client = self.create_client(
            SendAudioBuffer, self.speaker_service, callback_group=self._services_group)
        self.set_language_srv = self.create_service(
            SetDialogueLanguage, '/remote_dialogue/set_language', self._set_language_cb,
            callback_group=self._services_group)
        self.reset_srv = self.create_service(
            ResetDialogue, '/remote_dialogue/reset', self._reset_cb,
            callback_group=self._services_group)
        self.set_listening_srv = self.create_service(
            SetBool, '/remote_dialogue/set_listening', self._set_listening_cb,
            callback_group=self._services_group)
        self.action_server = ActionServer(
            self, RemoteDialogueTurn, '/remote_dialogue', self._execute_turn,
            goal_callback=self._goal_cb, cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            callback_group=self._services_group)

        self.converter = MicrophoneConverter(channel=self.microphone_channel)
        self.chunker = FrameChunker()
        self.gate = MicrophoneGate(guard_s=self.mic_guard_s)
        self.scheduler = PlaybackScheduler(self.output_rate, piece_s=self.piece_s,
                                           lead_s=self.lead_s)
        self._publish_connected(False)
        self._publish_language(self.language)
        self.get_logger().info(
            f'remote_dialogue configured: language={self.language}, '
            f'output {self.output_rate} Hz, speaker {self.speaker_service}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Connect to the server and start streaming the microphone."""
        self.session = DialogueSession(
            self.server_url, self.token, language=self.language,
            output_sample_rate=self.output_rate, greet_on_connect=self.greet_on_connect,
            on_event=self._on_server_event, on_audio=self._on_server_audio,
            on_connection=self._on_connection, logger=self.get_logger(),
            reconnect_interval_s=self.reconnect_interval_s)
        self.session.start()
        self._playback_stop.clear()
        self._playback_thread = threading.Thread(
            target=self._playback_loop, name='remote_dialogue_playback', daemon=True)
        self._playback_thread.start()
        self.audio_sub = self.create_subscription(
            AudioBuffer, self.microphone_topic, self._audio_cb, 10,
            callback_group=self._audio_group)
        self.get_logger().info('remote_dialogue activated')
        return super().on_activate(state)

    def on_deactivate(self, state):
        """Stop the microphone stream, playback and the server session."""
        self._shutdown_runtime()
        self.get_logger().info('remote_dialogue deactivated')
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        """Destroy the ROS interface."""
        self.action_server.destroy()
        for srv in (self.set_language_srv, self.reset_srv, self.set_listening_srv):
            self.destroy_service(srv)
        self.destroy_client(self.speaker_client)
        for pub in (self.state_pub, self.transcript_pub, self.language_pub,
                    self.connected_pub):
            self.destroy_publisher(pub)
        self.get_logger().info('remote_dialogue cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        """Close the session if the node is shut down from any state."""
        self._shutdown_runtime()
        return TransitionCallbackReturn.SUCCESS

    def _shutdown_runtime(self):
        if self.audio_sub is not None:
            try:
                self.destroy_subscription(self.audio_sub)
            except Exception:  # noqa: B902 - the context may already be shut down
                pass
            self.audio_sub = None
        self._playback_stop.set()
        if self._playback_thread is not None:
            self._playback_thread.join(2.0)
            self._playback_thread = None
        if self.session is not None:
            self.session.stop()
            self.session = None
        with self._lock:
            if self._turn is not None:
                self._turn.on_disconnect()
                self._turn_changed.notify_all()

    # ── publishing ───────────────────────────────────────────────────────────

    def _publish_connected(self, ready):
        self.connected_pub.publish(Bool(data=bool(ready)))

    def _publish_language(self, language):
        self.language_pub.publish(String(data=language))

    # ── server callbacks (session thread) ────────────────────────────────────

    def _on_connection(self, connected, ready):
        if rclpy.ok(context=self.context):
            self._publish_connected(ready)
        with self._lock:
            self._turn_changed.notify_all()   # wakes turns waiting for ready
        if not ready:
            with self._lock:
                self.gate.server_speaking = False
                self.scheduler.flush(time.monotonic())
                self.scheduler.reply_complete = False
                if not connected and self._turn is not None:
                    self._turn.on_disconnect()
                    self._turn_changed.notify_all()

    def _on_server_event(self, event):
        kind = event.get('type')
        now = time.monotonic()
        if kind == 'state':
            state = str(event.get('state', ''))
            self.state_pub.publish(String(data=state))
            self._vlog(f'server state: {state}')
            with self._lock:
                if state == 'speaking':
                    self.gate.server_speaking = True
                    self.scheduler.reply_complete = False
                elif state == 'idle' and self.gate.server_speaking:
                    self.scheduler.mark_reply_complete()
                if self._turn is not None and self._turn.on_state(state):
                    self._turn_changed.notify_all()
        elif kind == 'transcript':
            msg = DialogueTranscript(role=str(event.get('role', '')),
                                     text=str(event.get('text', '')),
                                     turn_id=int(event.get('turn_id', 0)))
            self.transcript_pub.publish(msg)
            self._vlog(f'{msg.role} [{msg.turn_id}]: {msg.text}')
            with self._lock:
                if self._turn is not None:
                    self._turn.on_transcript(msg.role, msg.text, msg.turn_id)
                    self._turn_changed.notify_all()
        elif kind == 'language':
            self.language = str(event.get('language', self.language))
            self._publish_language(self.language)
            self._language_confirmed.set()
            self.get_logger().info(f'Server language: {self.language}')
        elif kind == 'ready':
            self.language = str(event.get('language', self.language))
            self._publish_language(self.language)
        elif kind == 'interrupt':
            with self._lock:
                self.scheduler.flush(now)
        elif kind == 'error':
            self.get_logger().warning(f'Voice server error: {event.get("message", "")}')
        elif kind == 'sources':
            self._vlog(f'sources: {[i.get("title") for i in event.get("items", [])]}')

    def _on_server_audio(self, data):
        with self._lock:
            self.scheduler.push(data, time.monotonic())

    # ── microphone (ROS audio thread) ────────────────────────────────────────

    def _audio_cb(self, msg):
        session = self.session
        if session is None or not session.connected:
            return
        try:
            pcm = self.converter.convert(msg.data, msg.channel_map, msg.frequency)
        except ValueError as e:
            self.get_logger().error(f'Microphone audio rejected: {e}',
                                    throttle_duration_sec=10.0)
            return
        now = time.monotonic()
        with self._lock:
            live = self.gate.is_live(now, self.scheduler.end_time)
        for frame in self.chunker.push(pcm):
            session.send_audio(frame if live else bytes(len(frame)))

    # ── speaker (playback thread) ────────────────────────────────────────────

    def _send_to_speaker(self, stereo):
        if not self.speaker_client.service_is_ready():
            if not self._speaker_missing_warned:
                self.get_logger().warning(
                    f'{self.speaker_service} is not available; reply audio is dropped '
                    '(is naoqi_driver running?)')
                self._speaker_missing_warned = True
            return
        self._speaker_missing_warned = False
        request = SendAudioBuffer.Request()
        request.audio_data = stereo
        done = threading.Event()
        future = self.speaker_client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(1.0):
            self.get_logger().warning('send_audio_buffer did not answer within 1 s',
                                      throttle_duration_sec=5.0)
            return
        result = future.result()
        if result is not None and not result.success:
            self.get_logger().warning(f'send_audio_buffer failed: {result.message}',
                                      throttle_duration_sec=5.0)

    def _playback_loop(self):
        while not self._playback_stop.wait(0.01):
            now = time.monotonic()
            with self._lock:
                piece = self.scheduler.next_piece(now)
                done = piece is None and self.scheduler.take_playback_done(now, self.tail_s)
            if piece is not None:
                self._send_to_speaker(mono_to_robot_stereo(piece, self.output_rate))
                continue
            if done:
                session = self.session
                if session is not None:
                    session.send_json({'type': 'playback_done'})
                self._vlog('playback_done sent')
                with self._lock:
                    self.gate.robot_speech_ended(time.monotonic())
                    if self._turn is not None:
                        self._turn.on_playback_done()
                        self._turn_changed.notify_all()

    # ── services ─────────────────────────────────────────────────────────────

    def _set_listening_cb(self, request, response):
        with self._lock:
            self.gate.listening_enabled = bool(request.data)
        response.success = True
        response.message = f'listening {"enabled" if request.data else "disabled"}'
        return response

    def _set_language_cb(self, request, response):
        language = request.language.strip()
        session = self.session
        if not language:
            response.success, response.message = False, 'language is empty'
            return response
        if session is None or not session.ready:
            response.success, response.message = False, 'voice server not connected'
            return response
        self._language_confirmed.clear()
        session.set_language_hint(language)
        session.send_json({'type': 'set_language', 'language': language})
        if not self._language_confirmed.wait(self.language_timeout_s):
            response.success = False
            response.message = f'no confirmation within {self.language_timeout_s:.0f} s'
            return response
        response.success = self.language == language
        response.message = f'language is {self.language}'
        return response

    def _reset_cb(self, request, response):
        session = self.session
        if session is None or not session.ready:
            response.success, response.message = False, 'voice server not connected'
            return response
        message = {'type': 'reset', 'greet': bool(request.greet)}
        if request.language.strip():
            message['language'] = request.language.strip()
            session.set_language_hint(message['language'])
        session.send_json(message)
        response.success, response.message = True, 'reset sent'
        return response

    # ── action ───────────────────────────────────────────────────────────────

    def _goal_cb(self, _goal):
        with self._lock:
            return GoalResponse.REJECT if self._turn is not None else GoalResponse.ACCEPT

    def _execute_turn(self, goal_handle):
        goal = goal_handle.request
        result = RemoteDialogueTurn.Result()
        # Not ready yet (models loading, or reconnecting): wait for it up to
        # the goal's `wait` rather than failing at once, so a behavior tree
        # that loops turns does not spin while the server comes up.
        deadline = time.monotonic() + max(0.0, float(goal.wait))
        with self._lock:
            while not (self.session is not None and self.session.ready):
                remaining = deadline - time.monotonic()
                if (remaining <= 0.0 or goal_handle.is_cancel_requested or
                        not rclpy.ok(context=self.context)):
                    break
                self._turn_changed.wait(min(remaining, 0.2))
        session = self.session
        if session is None or not session.ready:
            result.status = 'not_connected'
            if goal_handle.is_cancel_requested:
                result.status = 'cancelled'
                goal_handle.canceled()
            elif rclpy.ok(context=self.context):
                goal_handle.abort()
            return result

        text = goal.text.strip()
        now = time.monotonic()
        tracker = TurnTracker(now, max(0.0, float(goal.wait)), self.turn_timeout_s,
                              text_mode=bool(text), user_text=text)
        with self._lock:
            if self._turn is not None:
                result.status = 'busy'
                goal_handle.abort()
                return result
            self._turn = tracker
            self.gate.turn_open = not text
        if text:
            session.send_json({'type': 'text_input', 'text': text})
        self._vlog(f'turn started ({"text" if text else f"listening {goal.wait:.1f} s"})')

        outcome = None
        last_state = None
        try:
            with self._lock:
                while True:
                    now = time.monotonic()
                    outcome = tracker.outcome(now)
                    if outcome is not None:
                        break
                    if not rclpy.ok(context=self.context):
                        outcome = (False, 'cancelled')
                        break
                    if goal_handle.is_cancel_requested:
                        outcome = (False, 'cancelled')
                        break
                    if tracker.state != last_state:
                        last_state = tracker.state
                        goal_handle.publish_feedback(
                            RemoteDialogueTurn.Feedback(state=tracker.state))
                    self._turn_changed.wait(0.1)
        finally:
            with self._lock:
                self._turn = None
                self.gate.turn_open = False

        success, status = outcome
        result.success = success
        result.status = status
        result.user_text = tracker.user_text
        result.assistant_text = tracker.assistant_text()
        result.language = self.language
        result.turn_id = max(tracker.turn_id, 0)
        self._vlog(f'turn ended: {status} user="{result.user_text}" '
                   f'reply="{result.assistant_text}"')
        if not rclpy.ok(context=self.context):
            return result
        if status == 'cancelled':
            goal_handle.canceled()
        elif success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result
