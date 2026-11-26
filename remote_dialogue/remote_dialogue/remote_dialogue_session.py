"""remote_dialogue_session.py

WebSocket client for the /ws/pepper voice server (PEPPER_API.md, v1.1).

DialogueSession runs its own asyncio event loop on a background thread and
keeps one session open: it connects, sends the `config` message, reports
`ready`, forwards every server message to callbacks, and reconnects after a
drop (with `"greet": false`, as the API asks). Other threads hand it
microphone frames and control messages through thread-safe methods; both go
through one outgoing queue, so they reach the server in the order given.

The token is part of the URL query; it is never logged.

None of this module touches ROS, so it is tested against a local fake server.

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
import json
import threading
import urllib.parse

from websockets.asyncio.client import connect

SUPPORTED_OUTPUT_RATES = (16000, 24000, 48000)


def build_url(server_url, token):
    """Append the token to the server URL as the `token` query parameter."""
    parts = urllib.parse.urlsplit(server_url)
    query = urllib.parse.parse_qsl(parts.query)
    query = [(k, v) for k, v in query if k != 'token'] + [('token', token)]
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def redact_url(server_url):
    """Return the URL without its query string, safe to log."""
    parts = urllib.parse.urlsplit(server_url)
    return urllib.parse.urlunsplit(parts._replace(query='', fragment=''))


class DialogueSession:
    """One persistent, self-reconnecting session with the voice server."""

    def __init__(self, server_url, token, language='eng', output_sample_rate=48000,
                 greet_on_connect=False, on_event=None, on_audio=None,
                 on_connection=None, logger=None, reconnect_interval_s=3.0,
                 open_timeout_s=10.0, max_queue=500):
        """Configure the session; nothing connects until start().

        Args:
            server_url: ws://host:port/ws/pepper (no token)
            token: shared secret appended as ?token=
            language: language for the config message
            output_sample_rate: reply audio rate, one of 16000/24000/48000
            greet_on_connect: `greet` for the very first config message;
                reconnects always send false
            on_event: callback(dict) for every JSON message from the server
            on_audio: callback(bytes) for every binary (speech) frame
            on_connection: callback(connected: bool, ready: bool)
            logger: object with info/warning/error methods (optional)
            reconnect_interval_s: pause between reconnect attempts
            open_timeout_s: WebSocket handshake timeout
            max_queue: outgoing frames kept while the socket is slow; the
                oldest are dropped beyond this
        """
        if output_sample_rate not in SUPPORTED_OUTPUT_RATES:
            raise ValueError(f'output_sample_rate must be one of {SUPPORTED_OUTPUT_RATES}')
        self.server_url = server_url
        self._url = build_url(server_url, token)
        self.language = language
        self.output_sample_rate = int(output_sample_rate)
        self._greet_next = bool(greet_on_connect)
        self._on_event = on_event or (lambda event: None)
        self._on_audio = on_audio or (lambda data: None)
        self._on_connection = on_connection or (lambda connected, ready: None)
        self._log = logger
        self.reconnect_interval_s = float(reconnect_interval_s)
        self.open_timeout_s = float(open_timeout_s)
        self.max_queue = int(max_queue)

        self.connected = False   # socket open and config sent
        self.ready = False       # server sent `ready`
        self.session_id = ''
        self.connections = 0     # successful handshakes so far

        self._loop = None
        self._thread = None
        self._queue = None
        self._stop = None
        self._started = threading.Event()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Start the background thread and begin connecting."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name='remote_dialogue_ws',
                                        daemon=True)
        self._thread.start()
        self._started.wait(5.0)

    def stop(self, timeout=5.0):
        """Close the session and join the thread."""
        if self._thread is None:
            return
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout)
        self._thread = None

    # ── thread-safe senders ──────────────────────────────────────────────────

    def send_audio(self, frame):
        """Queue one microphone frame. Dropped if the socket is not open."""
        if not self.connected or self._loop is None:
            return False
        self._loop.call_soon_threadsafe(self._enqueue, bytes(frame))
        return True

    def send_json(self, message):
        """Queue a control message (dict). Returns False if not connected."""
        if not self.connected or self._loop is None:
            return False
        self._loop.call_soon_threadsafe(self._enqueue, json.dumps(message))
        return True

    def set_language_hint(self, language):
        """Language to send in the config of the next (re)connection."""
        self.language = language

    # ── internals (event-loop thread) ────────────────────────────────────────

    def _enqueue(self, item):
        if self._queue is None:
            return
        while self._queue.qsize() >= self.max_queue:
            self._queue.get_nowait()
        self._queue.put_nowait(item)

    def _info(self, msg):
        if self._log is not None:
            try:
                self._log.info(msg)
            except Exception:  # noqa: B902 - logging must never kill the thread
                pass

    def _warn(self, msg):
        if self._log is not None:
            try:
                self._log.warning(msg)
            except Exception:  # noqa: B902 - logging must never kill the thread
                pass

    def _callback(self, fn, *args):
        # A failing callback (e.g. a ROS publish after the context is shut
        # down) must not kill the session thread.
        try:
            fn(*args)
        except Exception as e:  # noqa: B902
            self._warn(f'session callback failed: {type(e).__name__}: {e}')

    def _set_connection(self, connected, ready):
        changed = (connected, ready) != (self.connected, self.ready)
        self.connected, self.ready = connected, ready
        if changed:
            self._callback(self._on_connection, connected, ready)

    def _run(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._stop = asyncio.Event()
        self._queue = asyncio.Queue()
        self._started.set()
        try:
            loop.run_until_complete(self._main())
        finally:
            loop.close()

    async def _main(self):
        while not self._stop.is_set():
            try:
                await self._session_once()
            except Exception as e:  # noqa: B902 - any failure means reconnect
                self._warn(f'voice server {redact_url(self.server_url)}: {type(e).__name__}: '
                           f'{str(e).replace(self._url, redact_url(self.server_url))}')
            self._set_connection(False, False)
            if self._stop.is_set():
                break
            try:
                await asyncio.wait_for(self._stop.wait(), self.reconnect_interval_s)
            except asyncio.TimeoutError:
                pass

    async def _session_once(self):
        async with connect(self._url, max_size=None, open_timeout=self.open_timeout_s,
                           ping_interval=20, ping_timeout=20) as ws:
            config = {'type': 'config', 'language': self.language,
                      'output_sample_rate': self.output_sample_rate,
                      'greet': self._greet_next}
            await ws.send(json.dumps(config))
            # Only the first session may greet; a reconnect is mid-conversation.
            self._greet_next = False
            self.connections += 1
            while not self._queue.empty():  # stale audio from the last session
                self._queue.get_nowait()
            self._set_connection(True, False)
            self._info(f'Connected to voice server {redact_url(self.server_url)} '
                       f'(language={self.language}); waiting for ready')

            sender = asyncio.ensure_future(self._sender(ws))
            receiver = asyncio.ensure_future(self._receiver(ws))
            stopper = asyncio.ensure_future(self._stop.wait())
            done, pending = await asyncio.wait(
                {sender, receiver, stopper}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                if task is not stopper and task.exception() is not None:
                    raise task.exception()

    async def _sender(self, ws):
        while True:
            item = await self._queue.get()
            await ws.send(item)

    async def _receiver(self, ws):
        async for message in ws:
            if isinstance(message, bytes):
                self._callback(self._on_audio, message)
                continue
            try:
                event = json.loads(message)
            except ValueError:
                self._warn('voice server sent a non-JSON text frame; ignored')
                continue
            if not isinstance(event, dict):
                continue
            if event.get('type') == 'ready':
                self.session_id = str(event.get('session_id', ''))
                if event.get('language'):
                    self.language = event['language']
                self._set_connection(True, True)
                self._info(f'Voice server ready (session {self.session_id}, '
                           f'language {self.language})')
            elif event.get('type') == 'language' and event.get('language'):
                self.language = event['language']
            self._callback(self._on_event, event)
