"""End-to-end tests of RemoteDialogueNode with a fake Pepper and a fake server.

The fake Pepper publishes /audio at real-time rate (48 kHz, 4 channels, 4096
frames per buffer, like naoqi_driver), serves send_audio_buffer, and echoes
whatever it is playing back into its own microphones, as the real robot
does. The fake server counts any live audio that reaches it while a reply is
playing, so a leak in the microphone gate fails the test.
"""

import threading
import time

import domain_coordinator
from fake_voice_server import FakeVoiceServer
import numpy as np
import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from dec_interfaces.action import RemoteDialogueTurn
from dec_interfaces.msg import DialogueTranscript
from dec_interfaces.srv import ResetDialogue, SetDialogueLanguage
from naoqi_bridge_msgs.msg import AudioBuffer
from naoqi_bridge_msgs.srv import SendAudioBuffer
from remote_dialogue.remote_dialogue_implementation import RemoteDialogueNode
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

RATE = 48000
BUFFER_FRAMES = 4096
REPLY_S = 0.6


class FakePepper(Node):
    """Microphones on /audio and a speaker on send_audio_buffer."""

    def __init__(self):
        super().__init__('fake_pepper')
        self.pub = self.create_publisher(AudioBuffer, '/audio', 10)
        self.srv = self.create_service(SendAudioBuffer, '/naoqi_driver/send_audio_buffer',
                                       self._play)
        self.lock = threading.Lock()
        self.pieces = []            # (time received, stereo frames)
        self.playing_until = 0.0
        self.visitor_until = 0.0    # visitor speaks until this time
        self.echo = True
        self.states, self.transcripts = [], []
        self.connected = None
        self.language = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/remote_dialogue/state',
                                 lambda m: self.states.append(m.data), 10)
        self.create_subscription(DialogueTranscript, '/remote_dialogue/transcript',
                                 lambda m: self.transcripts.append((m.role, m.text)), 10)
        self.create_subscription(Bool, '/remote_dialogue/connected',
                                 lambda m: setattr(self, 'connected', m.data), latched)
        self.create_subscription(String, '/remote_dialogue/language',
                                 lambda m: setattr(self, 'language', m.data), latched)
        self.turn = ActionClient(self, RemoteDialogueTurn, '/remote_dialogue')
        self.set_language = self.create_client(SetDialogueLanguage,
                                               '/remote_dialogue/set_language')
        self.reset = self.create_client(ResetDialogue, '/remote_dialogue/reset')
        self.set_listening = self.create_client(SetBool, '/remote_dialogue/set_listening')
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._mic_loop, daemon=True)
        self._thread.start()

    def _play(self, request, response):
        frames = len(request.audio_data) // 4
        now = time.monotonic()
        with self.lock:
            self.pieces.append((now, frames))
            self.playing_until = max(self.playing_until, now) + frames / RATE
        response.success = True
        return response

    def speaker_seconds(self):
        with self.lock:
            return sum(f for _, f in self.pieces) / RATE

    def first_piece_time(self):
        with self.lock:
            return self.pieces[0][0] if self.pieces else None

    def _mic_loop(self):
        period = BUFFER_FRAMES / RATE
        t = np.arange(BUFFER_FRAMES) / RATE
        speech = (6000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
        echo = (6000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        silence = np.zeros(BUFFER_FRAMES, np.int16)
        next_time = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            with self.lock:
                playing = now < self.playing_until
            if now < self.visitor_until:
                mono = speech
            elif playing and self.echo:
                mono = echo            # Pepper hears its own speaker
            else:
                mono = silence
            msg = AudioBuffer()
            msg.frequency = RATE
            msg.channel_map = [0, 2, 3, 5]
            msg.data = np.repeat(mono, 4).tolist()
            self.pub.publish(msg)
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))

    def stop(self):
        self._stop.set()
        self._thread.join(2.0)


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def call(client, request, timeout=10.0):
    assert client.wait_for_service(timeout_sec=5.0)
    future = client.call_async(request)
    assert wait_for(future.done, timeout)
    return future.result()


def run_turn(pepper, wait=5.0, text='', timeout=30.0):
    assert pepper.turn.wait_for_server(timeout_sec=5.0)
    goal = RemoteDialogueTurn.Goal(wait=wait, text=text)
    send = pepper.turn.send_goal_async(goal)
    assert wait_for(send.done)
    handle = send.result()
    assert handle.accepted
    result = handle.get_result_async()
    assert wait_for(result.done, timeout)
    return result.result().result


@pytest.fixture(scope='module', autouse=True)
def ros():
    # A domain of our own (as ament_add_ros_isolated_gtest gives the C++
    # tests), so a robot stack or another package's tests running on this
    # machine cannot publish into these tests, or answer their goals.
    with domain_coordinator.domain_id() as domain:
        rclpy.init(domain_id=domain)
        yield
        rclpy.shutdown()


@pytest.fixture
def server():
    srv = FakeVoiceServer(token='tok', reply_seconds=REPLY_S).start()
    yield srv
    srv.stop()


@pytest.fixture
def system(server, monkeypatch):
    monkeypatch.setenv('REMOTE_DIALOGUE_TOKEN', 'tok')
    monkeypatch.setenv('REMOTE_DIALOGUE_URL', server.url)
    node = RemoteDialogueNode(parameter_overrides=[
        Parameter('language_timeout_s', value=1.5),
        Parameter('reconnect_interval_s', value=0.3),
        Parameter('verbose_mode', value=True),
    ])
    pepper = FakePepper()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    executor.add_node(pepper)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    assert node.trigger_configure() == TransitionCallbackReturn.SUCCESS
    assert node.trigger_activate() == TransitionCallbackReturn.SUCCESS
    assert wait_for(lambda: pepper.connected is True)
    yield node, pepper, server
    pepper.stop()
    node.trigger_deactivate()
    executor.shutdown()
    node.destroy_node()
    pepper.destroy_node()
    spinner.join(2.0)


def assert_playback_done_after_audio_ended(pepper, server):
    """playback_done must not arrive before the robot finished playing."""
    first = pepper.first_piece_time()
    assert first is not None
    audio_end = first + pepper.speaker_seconds()
    assert server.playback_done_times, 'playback_done never sent'
    assert server.playback_done_times[-1] >= audio_end - 0.01
    # ...and not long after (lead + tail + loop slack)
    assert server.playback_done_times[-1] - audio_end < 0.6


def test_connects_with_config_and_streams_zeros_between_turns(system):
    node, pepper, server = system
    assert server.configs[0] == {'type': 'config', 'language': 'eng',
                                 'output_sample_rate': 48000, 'greet': False}
    assert pepper.language == 'eng'
    pepper.visitor_until = time.monotonic() + 1.0   # someone talks, no turn open
    time.sleep(1.5)
    assert server.mic_frames > 50                    # stream is continuous...
    assert server.mic_bytes % 640 == 0               # ...in 20 ms frames
    assert 'listening' not in pepper.states          # ...but it was all zeros


def test_typed_turn_plays_reply_and_reports_it(system):
    node, pepper, server = system
    result = run_turn(pepper, text='What is the Upanzi Network?')
    assert result.success and result.status == 'completed'
    assert result.user_text == 'What is the Upanzi Network?'
    assert result.assistant_text == \
        'You asked: What is the Upanzi Network? This is the second sentence.'
    assert result.turn_id == 1 and result.language == 'eng'
    assert pepper.speaker_seconds() == pytest.approx(REPLY_S, abs=0.01)
    assert_playback_done_after_audio_ended(pepper, server)
    assert server.live_audio_while_speaking == 0


def test_spoken_turn_with_echo_never_leaks_robot_voice(system):
    node, pepper, server = system
    threading.Timer(0.5, lambda: setattr(pepper, 'visitor_until',
                                         time.monotonic() + 0.8)).start()
    result = run_turn(pepper, wait=5.0)
    assert result.success, result.status
    assert result.user_text == 'Tell me about the Digital Experience Center.'
    assert 'listening' in pepper.states and 'thinking' in pepper.states
    assert ('assistant', 'This is the second sentence.') in pepper.transcripts
    assert_playback_done_after_audio_ended(pepper, server)
    # Pepper's speaker was echoed into its microphones the whole reply.
    assert server.live_audio_while_speaking == 0


def test_two_turns_in_a_row(system):
    node, pepper, server = system
    first = run_turn(pepper, text='one')
    second = run_turn(pepper, text='two')
    assert first.success and second.success
    assert (first.turn_id, second.turn_id) == (1, 2)
    assert len(server.playback_done_times) == 2
    assert server.live_audio_while_speaking == 0


def test_no_speech_times_out(system):
    node, pepper, server = system
    start = time.monotonic()
    result = run_turn(pepper, wait=1.0)
    assert not result.success and result.status == 'no_speech'
    assert 0.9 <= time.monotonic() - start < 2.5


def test_second_goal_while_busy_is_rejected(system):
    node, pepper, server = system
    assert pepper.turn.wait_for_server(timeout_sec=5.0)
    first = pepper.turn.send_goal_async(RemoteDialogueTurn.Goal(wait=3.0))
    assert wait_for(first.done) and first.result().accepted
    second = pepper.turn.send_goal_async(RemoteDialogueTurn.Goal(wait=3.0))
    assert wait_for(second.done)
    assert not second.result().accepted
    cancel = first.result().cancel_goal_async()
    assert wait_for(cancel.done)
    result = first.result().get_result_async()
    assert wait_for(result.done)
    assert result.result().result.status == 'cancelled'


def test_set_language_waits_for_confirmation(system):
    node, pepper, server = system
    ok = call(pepper.set_language, SetDialogueLanguage.Request(language='kin'))
    assert ok.success, ok.message
    assert wait_for(lambda: pepper.language == 'kin')
    bad = call(pepper.set_language, SetDialogueLanguage.Request(language='xx'))
    assert not bad.success
    assert run_turn(pepper, text='Muraho').language == 'kin'


def test_reset_with_greeting_plays_welcome_line(system):
    node, pepper, server = system
    response = call(pepper.reset, ResetDialogue.Request(greet=True, language='kin'))
    assert response.success
    assert wait_for(lambda: server.playback_done_times)
    assert ('assistant', 'Muraho, murakaza neza.') in pepper.transcripts
    assert wait_for(lambda: pepper.language == 'kin')
    assert_playback_done_after_audio_ended(pepper, server)
    assert {'type': 'reset', 'greet': True, 'language': 'kin'} in server.controls


def test_set_listening_opens_the_gate_outside_turns(system):
    node, pepper, server = system
    assert call(pepper.set_listening, SetBool.Request(data=True)).success
    pepper.visitor_until = time.monotonic() + 0.8
    assert wait_for(lambda: server.playback_done_times, timeout=10.0)
    assert 'listening' in pepper.states
    assert server.live_audio_while_speaking == 0
    assert call(pepper.set_listening, SetBool.Request(data=False)).success


def test_reconnects_after_drop_and_turns_work_again(system):
    node, pepper, server = system
    server.drop_connections()
    assert wait_for(lambda: pepper.connected is False, timeout=5.0)
    assert wait_for(lambda: pepper.connected is True, timeout=5.0)
    assert [c['greet'] for c in server.configs] == [False, False]
    assert run_turn(pepper, text='still there?').success


def test_turn_without_server_reports_not_connected(server, monkeypatch):
    monkeypatch.setenv('REMOTE_DIALOGUE_TOKEN', 'tok')
    monkeypatch.setenv('REMOTE_DIALOGUE_URL', 'ws://127.0.0.1:9/ws/pepper')
    node = RemoteDialogueNode(node_name='remote_dialogue')
    client = Node('client')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(client)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    try:
        node.trigger_configure()
        node.trigger_activate()
        action = ActionClient(client, RemoteDialogueTurn, '/remote_dialogue')
        assert action.wait_for_server(timeout_sec=5.0)
        send = action.send_goal_async(RemoteDialogueTurn.Goal(wait=1.0))
        assert wait_for(send.done)
        result = send.result().get_result_async()
        assert wait_for(result.done)
        assert result.result().result.status == 'not_connected'
    finally:
        node.trigger_deactivate()
        executor.shutdown()
        node.destroy_node()
        client.destroy_node()


def test_configure_fails_without_token(monkeypatch):
    monkeypatch.delenv('REMOTE_DIALOGUE_TOKEN', raising=False)
    monkeypatch.setenv('REMOTE_DIALOGUE_URL', 'ws://127.0.0.1:9/ws/pepper')
    node = RemoteDialogueNode()
    try:
        assert node.trigger_configure() == TransitionCallbackReturn.FAILURE
    finally:
        node.destroy_node()


def test_token_is_not_a_ros_parameter(monkeypatch):
    node = RemoteDialogueNode()
    try:
        params = node.get_parameters_by_prefix('')
        assert 'server_url' in params
        assert not any('token' in name for name in params)
        monkeypatch.setenv('REMOTE_DIALOGUE_TOKEN', 'very-secret-value')
        assert 'very-secret-value' not in str([p.value for p in params.values()])
    finally:
        node.destroy_node()
