"""Tests for remote_dialogue_session against the fake /ws/pepper server."""

import json
import threading
import time

from fake_voice_server import FakeVoiceServer
import pytest

from remote_dialogue.remote_dialogue_session import build_url, DialogueSession, redact_url


class Recorder:
    """Collects session callbacks and log lines."""

    def __init__(self):
        self.events = []
        self.audio = bytearray()
        self.connection = []
        self.logs = []
        self.lock = threading.Lock()

    def on_event(self, event):
        with self.lock:
            self.events.append(event)

    def on_audio(self, data):
        with self.lock:
            self.audio.extend(data)

    def on_connection(self, connected, ready):
        with self.lock:
            self.connection.append((connected, ready))

    def info(self, msg):
        self.logs.append(msg)

    warning = error = info

    def types(self):
        with self.lock:
            return [e.get('type') for e in self.events]


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def server():
    srv = FakeVoiceServer(token='s3cret', reply_seconds=0.3).start()
    yield srv
    srv.stop()


def make_session(server, rec, token='s3cret', **kwargs):
    return DialogueSession(server.url, token, on_event=rec.on_event, on_audio=rec.on_audio,
                           on_connection=rec.on_connection, logger=rec,
                           reconnect_interval_s=0.2, **kwargs)


def test_url_helpers():
    assert build_url('ws://h:1/ws/pepper', 'a b') == 'ws://h:1/ws/pepper?token=a+b'
    assert build_url('ws://h:1/ws/pepper?token=old&x=1', 't') == \
        'ws://h:1/ws/pepper?x=1&token=t'
    assert redact_url('ws://h:1/ws/pepper?token=t') == 'ws://h:1/ws/pepper'


def test_rejects_unsupported_output_rate(server):
    with pytest.raises(ValueError):
        DialogueSession(server.url, 't', output_sample_rate=44100)


def test_connects_sends_config_and_reports_ready(server):
    rec = Recorder()
    session = make_session(server, rec, language='kin', output_sample_rate=48000,
                           greet_on_connect=False)
    session.start()
    try:
        assert wait_for(lambda: session.ready)
        assert server.configs == [{'type': 'config', 'language': 'kin',
                                   'output_sample_rate': 48000, 'greet': False}]
        assert rec.connection[-1] == (True, True)
        assert session.language == 'kin'
        assert 'ready' in rec.types()
    finally:
        session.stop()
    assert wait_for(lambda: rec.connection[-1] == (False, False))


def test_greeting_audio_is_received_at_requested_rate(server):
    rec = Recorder()
    session = make_session(server, rec, output_sample_rate=16000, greet_on_connect=True)
    session.start()
    try:
        assert wait_for(lambda: rec.types().count('state') >= 2, timeout=5.0)
        assert wait_for(lambda: server.reply_end_times)
        assert len(rec.audio) == int(0.3 * 16000) * 2
        states = [e['state'] for e in rec.events if e.get('type') == 'state']
        assert states[:2] == ['speaking', 'idle']
    finally:
        session.stop()


def test_bad_token_is_refused_and_retried_without_leaking_it(server):
    rec = Recorder()
    session = make_session(server, rec, token='wrong-token')
    session.start()
    try:
        assert wait_for(lambda: server.rejected_handshakes >= 2)
        assert not session.ready
        assert not any('wrong-token' in line for line in rec.logs)
        assert any('403' in line for line in rec.logs)
    finally:
        session.stop()


def test_reconnects_after_drop_with_greet_false(server):
    rec = Recorder()
    session = make_session(server, rec, greet_on_connect=True)
    session.start()
    try:
        assert wait_for(lambda: session.ready)
        server.drop_connections()
        assert wait_for(lambda: (False, False) in rec.connection)
        assert wait_for(lambda: session.ready and session.connections == 2)
        assert [c['greet'] for c in server.configs] == [True, False]
    finally:
        session.stop()


def test_audio_and_control_reach_server_in_order(server):
    rec = Recorder()
    session = make_session(server, rec, greet_on_connect=False)
    assert not session.send_audio(b'\x00' * 640)      # not connected yet
    session.start()
    try:
        assert wait_for(lambda: session.ready)
        for _ in range(50):
            assert session.send_audio(b'\x00' * 640)
        assert session.send_json({'type': 'set_language', 'language': 'amh'})
        assert wait_for(lambda: 'language' in rec.types())
        assert server.mic_frames == 50                # all before the control message
        assert session.language == 'amh'
        assert session.send_json({'type': 'text_input', 'text': 'What is HumekaFL?'})
        assert wait_for(lambda: server.reply_end_times)
        user = [e for e in rec.events if e.get('role') == 'user']
        assert user and user[0]['text'] == 'What is HumekaFL?'
    finally:
        session.stop()


def test_ignores_non_json_and_unknown_messages(server):
    rec = Recorder()
    session = make_session(server, rec, greet_on_connect=False)
    session.start()
    try:
        assert wait_for(lambda: session.ready)
        session.send_json({'type': 'something_new'})
        session.send_json({'type': 'set_language', 'language': 'xx'})
        assert wait_for(lambda: 'error' in rec.types())
        assert session.ready
        assert json.dumps(server.controls[0]) == '{"type": "something_new"}'
    finally:
        session.stop()


def test_failing_callbacks_do_not_kill_the_session(server):
    """A callback that raises (a ROS publish after shutdown) is contained."""
    rec = Recorder()

    def boom(*_args):
        raise RuntimeError('publisher context is invalid')

    session = DialogueSession(server.url, 's3cret', on_event=boom, on_audio=boom,
                              on_connection=boom, logger=rec, reconnect_interval_s=0.2,
                              greet_on_connect=True)
    session.start()
    try:
        assert wait_for(lambda: session.ready)
        assert wait_for(lambda: server.reply_end_times)   # greeting audio received
        server.drop_connections()
        assert wait_for(lambda: session.ready and session.connections == 2)
        assert any('session callback failed' in line for line in rec.logs)
    finally:
        session.stop()
