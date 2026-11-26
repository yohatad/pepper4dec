"""fake_voice_server.py

A stand-in for the /ws/pepper voice server (PEPPER_API.md, v1.1), for tests.

It follows the documented protocol: token check during the handshake, the
`config` message within 10 s, `ready`, an optional welcome line, server-side
turn-taking on the microphone stream (an energy detector instead of a real
VAD), replies as real-time-paced PCM with `speaking`/`idle` states and one
`assistant` transcript per sentence, `playback_done`, `reset`,
`set_language` and `text_input`.

It also records what the client did, so tests can check the half-duplex
rules: `live_audio_while_speaking` counts non-silent microphone frames that
arrived while a reply was being sent or played.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 26, 2026
Version: v1.0

Copyright (C) 2026 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import asyncio
import http
import json
import threading
import time
import urllib.parse

import numpy as np
from websockets.asyncio.server import serve

LANGUAGES = ('eng', 'kin', 'amh', 'swh', 'fra')
WELCOME = {'eng': 'Hello, and welcome.', 'kin': 'Muraho, murakaza neza.'}


def tone(seconds, rate, freq=440.0, amplitude=8000):
    """Mono int16 LE sine bytes."""
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype('<i2').tobytes()


class FakeVoiceServer:
    """Run the fake server on a background thread."""

    def __init__(self, token='secret', reply_seconds=0.6, speech_rms=500.0,
                 end_silence_frames=15, chunk_s=0.04):
        """Configure the fake server.

        Args:
            token: the accepted ?token=
            reply_seconds: length of each spoken reply
            speech_rms: RMS above which a 20 ms mic frame counts as speech
            end_silence_frames: quiet frames that end a visitor's turn
            chunk_s: reply audio chunk length (sent at playback speed)
        """
        self.token = token
        self.reply_seconds = reply_seconds
        self.speech_rms = speech_rms
        self.end_silence_frames = end_silence_frames
        self.chunk_s = chunk_s
        self.port = None

        # Recorded client behaviour
        self.configs = []
        self.controls = []
        self.mic_frames = 0
        self.mic_bytes = 0
        self.live_audio_while_speaking = 0
        self.playback_done_times = []
        self.reply_end_times = []     # when the last reply byte was sent
        self.reply_seconds_sent = []
        self.rejected_handshakes = 0
        self.connections = 0

        self._loop = None
        self._thread = None
        self._server = None
        self._ready = threading.Event()
        self._connections = set()

    # ── control from the test thread ─────────────────────────────────────────

    def start(self):
        """Start serving on a free localhost port."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        assert self._ready.wait(5.0), 'fake server did not start'
        return self

    def stop(self):
        """Stop the server."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(5.0)

    def drop_connections(self):
        """Close every open connection (simulates a network drop)."""
        async def close_all():
            for ws in list(self._connections):
                await ws.close()
        asyncio.run_coroutine_threadsafe(close_all(), self._loop).result(5.0)

    @property
    def url(self):
        """The ws:// URL without token."""
        return f'ws://127.0.0.1:{self.port}/ws/pepper'

    # ── server ───────────────────────────────────────────────────────────────

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def start():
            self._server = await serve(self._handler, '127.0.0.1', 0,
                                       process_request=self._check_token, max_size=None)
            self.port = self._server.sockets[0].getsockname()[1]
            self._ready.set()

        self._loop.run_until_complete(start())
        self._loop.run_forever()
        self._server.close()
        self._loop.run_until_complete(self._server.wait_closed())
        self._loop.close()

    def _check_token(self, connection, request):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.path).query)
        if query.get('token', [''])[0] != self.token:
            self.rejected_handshakes += 1
            return connection.respond(http.HTTPStatus.FORBIDDEN, 'bad token\n')
        return None

    async def _handler(self, ws):
        self._connections.add(ws)
        self.connections += 1
        try:
            await Session(self, ws).run()
        finally:
            self._connections.discard(ws)


class Session:
    """One client session on the fake server."""

    def __init__(self, server, ws):
        """Bind to the server's recorder and the socket."""
        self.server = server
        self.ws = ws
        self.language = 'eng'
        self.rate = 24000
        self.turn_id = 0
        self.state = 'idle'
        self.replying = False       # from `speaking` until playback_done (or 1.5 s)
        self.mic_open = True
        self.speech_frames = 0
        self.quiet_frames = 0
        self.in_speech = False
        self.reply_task = None
        self.reopen_task = None

    async def send_state(self, state):
        if state != self.state:
            self.state = state
            await self.ws.send(json.dumps({'type': 'state', 'state': state}))

    async def run(self):
        try:
            raw = await asyncio.wait_for(self.ws.recv(), 10.0)
            config = json.loads(raw)
        except (asyncio.TimeoutError, ValueError, TypeError):
            await self.ws.close(4000, 'config expected')
            return
        self.server.configs.append(config)
        if (config.get('type') != 'config' or
                config.get('output_sample_rate', 24000) not in (16000, 24000, 48000) or
                config.get('language', 'eng') not in LANGUAGES):
            await self.ws.send(json.dumps({'type': 'error', 'message': 'bad config'}))
            await self.ws.close(4000, 'bad config')
            return
        self.language = config.get('language', 'eng')
        self.rate = config.get('output_sample_rate', 24000)
        await self.ws.send(json.dumps({
            'type': 'ready', 'session_id': f's{self.server.connections}',
            'profile': 'upanzi', 'language': self.language,
            'input_sample_rate': 16000, 'output_sample_rate': self.rate}))
        if config.get('greet', True):
            self.start_reply(WELCOME.get(self.language, 'Welcome.'))

        async for message in self.ws:
            if isinstance(message, bytes):
                await self.on_mic(message)
            else:
                await self.on_control(json.loads(message))

    async def on_mic(self, frame):
        server = self.server
        server.mic_frames += 1
        server.mic_bytes += len(frame)
        samples = np.frombuffer(frame[:len(frame) - len(frame) % 2], dtype='<i2')
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
        loud = rms > server.speech_rms
        if self.replying:
            if loud:
                server.live_audio_while_speaking += 1
            return
        if not self.mic_open:
            return
        if loud:
            self.speech_frames += 1
            self.quiet_frames = 0
            if not self.in_speech and self.speech_frames >= 3:
                self.in_speech = True
                await self.send_state('listening')
        elif self.in_speech:
            self.quiet_frames += 1
            if self.quiet_frames >= server.end_silence_frames:
                self.in_speech = False
                self.speech_frames = 0
                await self.answer('Tell me about the Digital Experience Center.')
        else:
            self.speech_frames = 0

    async def answer(self, question, typed=False):
        self.turn_id += 1
        if not typed:
            await self.send_state('thinking')
        await self.ws.send(json.dumps({'type': 'transcript', 'role': 'user',
                                       'text': question, 'turn_id': self.turn_id}))
        self.start_reply(f'You asked: {question}', 'This is the second sentence.')

    def start_reply(self, *sentences):
        self.replying = True
        self.reply_task = asyncio.ensure_future(self.reply(sentences))

    async def reply(self, sentences):
        server = self.server
        await self.send_state('speaking')
        audio = tone(server.reply_seconds, self.rate)
        chunk = int(server.chunk_s * self.rate) * 2
        start = time.monotonic()
        sent = 0
        for i, sentence in enumerate(sentences):
            await self.ws.send(json.dumps({'type': 'transcript', 'role': 'assistant',
                                           'text': sentence, 'turn_id': self.turn_id}))
            part = audio[i * len(audio) // len(sentences):(i + 1) * len(audio) // len(sentences)]
            for offset in range(0, len(part), chunk):
                piece = part[offset:offset + chunk]
                await self.ws.send(piece)
                sent += len(piece)
                # Real-time pacing, like the real server.
                delay = start + sent / 2 / self.rate - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
        server.reply_end_times.append(time.monotonic())
        server.reply_seconds_sent.append(sent / 2 / self.rate)
        await self.send_state('idle')
        # The API's fallback: reopen 1.5 s after sending if playback_done never comes.
        self.reopen_task = asyncio.ensure_future(self.reopen_after(1.5))

    async def reopen_after(self, seconds):
        await asyncio.sleep(seconds)
        self.replying = False

    async def on_control(self, msg):
        server = self.server
        server.controls.append(msg)
        kind = msg.get('type')
        if kind == 'playback_done':
            server.playback_done_times.append(time.monotonic())
            if self.reopen_task is not None:
                self.reopen_task.cancel()
            self.replying = False
        elif kind == 'set_language':
            if msg.get('language') in LANGUAGES:
                self.language = msg['language']
                await self.ws.send(json.dumps({'type': 'language', 'language': self.language}))
            else:
                await self.ws.send(json.dumps({'type': 'error',
                                               'message': 'unsupported language'}))
        elif kind == 'reset':
            self.turn_id = 0
            if msg.get('language') in LANGUAGES:
                self.language = msg['language']
                await self.ws.send(json.dumps({'type': 'language', 'language': self.language}))
            if msg.get('greet', True):
                self.start_reply(WELCOME.get(self.language, 'Welcome.'))
        elif kind == 'text_input':
            await self.answer(msg.get('text', ''), typed=True)
