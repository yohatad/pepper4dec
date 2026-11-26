"""remote_dialogue_logic.py

Clock-driven state for the remote dialogue node: speaker pacing, the
microphone gate, and the outcome of one conversation turn. Every method takes
the current time as an argument, so tests drive it with a fake clock.

PlaybackScheduler
    Queues reply audio as it streams in from the server and hands it to the
    robot in short pieces, never more than `lead_s` ahead of what Pepper is
    playing. Because every sample that reaches the speaker passes through
    here, `end_time` is when Pepper's speaker goes quiet; that is what
    `playback_done` is timed from (see PEPPER_API.md, "Half duplex").

MicrophoneGate
    Decides, per 20 ms frame, whether live microphone audio or zeros go to
    the server. The stream itself never stops; muting means sending zeros.

TurnTracker
    Follows the server's state and transcript messages through one turn and
    reports how it ended.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: November 26, 2026
Version: v1.0

Copyright (C) 2026 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

from .remote_dialogue_audio import pcm_duration


class PlaybackScheduler:
    """Pace mono reply audio out to the robot speaker."""

    def __init__(self, rate, piece_s=0.2, lead_s=0.15, max_hold_s=0.08):
        """Configure the pacing.

        Args:
            rate: sample rate of the reply audio from the server (Hz)
            piece_s: target length of each piece sent to the robot
            lead_s: send the next piece once the robot has at most this much
                audio left to play
            max_hold_s: send a short piece rather than hold audio longer
                than this waiting for a full one
        """
        self.rate = int(rate)
        self.piece_bytes = max(2, int(piece_s * self.rate) * 2)
        self.lead_s = float(lead_s)
        self.max_hold_s = float(max_hold_s)
        self._pending = bytearray()
        self._pending_since = None
        self.end_time = 0.0      # when the robot speaker finishes what it has
        self.reply_complete = False
        self.total_bytes = 0

    def push(self, pcm, now):
        """Queue audio received from the server."""
        if not pcm:
            return
        if not self._pending:
            self._pending_since = now
        self._pending.extend(pcm)

    def mark_reply_complete(self):
        """The server sent `idle`: no more audio is coming for this reply."""
        self.reply_complete = True

    def flush(self, now):
        """Server `interrupt`: drop everything not yet handed to the robot."""
        self._pending.clear()
        self._pending_since = None
        # Audio already sent to Pepper cannot be recalled; end_time stands.

    def next_piece(self, now):
        """Return the next piece to send to the robot, or None if not yet due."""
        if not self._pending:
            return None
        if self.end_time - now > self.lead_s:
            return None
        full = len(self._pending) >= self.piece_bytes
        held_too_long = (self._pending_since is not None and
                         now - self._pending_since >= self.max_hold_s)
        if not (full or self.reply_complete or held_too_long):
            return None
        size = min(self.piece_bytes, len(self._pending))
        size -= size % 2
        piece = bytes(self._pending[:size])
        del self._pending[:size]
        self._pending_since = now if self._pending else None
        self.end_time = max(self.end_time, now) + pcm_duration(size, self.rate)
        self.total_bytes += size
        return piece

    def has_pending(self):
        """True while audio is queued but not yet sent to the robot."""
        return bool(self._pending)

    def drained(self, now):
        """True once every queued sample has been sent and played."""
        return not self._pending and now >= self.end_time

    def take_playback_done(self, now, tail_s=0.0):
        """Return True once, when a complete reply has finished playing.

        `tail_s` covers the robot's output latency beyond the computed end.
        """
        if self.reply_complete and not self._pending and now >= self.end_time + tail_s:
            self.reply_complete = False
            return True
        return False


class MicrophoneGate:
    """Choose live audio or zeros for each microphone frame."""

    def __init__(self, guard_s=0.3):
        """`guard_s`: keep muting this long after the robot's own speech ends."""
        self.guard_s = float(guard_s)
        self.turn_open = False       # a RemoteDialogueTurn goal is listening
        self.listening_enabled = False  # set_listening service override
        self.server_speaking = False    # from `speaking` until playback_done
        self.muted_until = 0.0

    def robot_speech_ended(self, now):
        """Playback finished; keep the guard window."""
        self.server_speaking = False
        self.muted_until = max(self.muted_until, now + self.guard_s)

    def is_live(self, now, playback_end):
        """True if this frame may carry live microphone audio."""
        if not (self.turn_open or self.listening_enabled):
            return False
        if self.server_speaking:
            return False
        return now >= max(self.muted_until, playback_end + self.guard_s)


class TurnTracker:
    """Follow one conversation turn to its outcome."""

    def __init__(self, now, wait_s, timeout_s, text_mode=False, user_text=''):
        """Start tracking a turn at time `now`.

        Args:
            now: start time
            wait_s: how long to wait for the visitor to start speaking
            timeout_s: overall limit for the whole turn
            text_mode: the turn is a typed question (no listening phase)
            user_text: the typed question, in text mode
        """
        self.started = now
        self.wait_deadline = now + wait_s
        self.timeout_deadline = now + timeout_s
        self.text_mode = text_mode
        self.heard = text_mode
        self.replying = False
        self.finished = False
        self.disconnected = False
        self.user_text = user_text
        self.assistant = []
        self.turn_id = -1
        self.state = 'waiting'

    def on_state(self, state):
        """Handle a server `state` message. Returns True if the state changed."""
        changed = state != self.state
        self.state = state
        if state in ('listening', 'thinking'):
            self.heard = True
        elif state == 'speaking':
            # Only a reply to this turn counts: a welcome line that starts
            # while the turn is still waiting for the visitor does not.
            if self.heard:
                self.replying = True
        elif state == 'idle' and self.heard and not self.replying and not self.text_mode:
            # The server opened a turn on noise and dropped it: keep waiting.
            if not self.user_text:
                self.heard = False
        return changed

    def on_transcript(self, role, text, turn_id):
        """Handle a server `transcript` message."""
        if role == 'user':
            self.heard = True
            if not self.text_mode:
                self.user_text = text
            self.turn_id = turn_id
        elif role == 'assistant' and self.heard:
            if self.turn_id < 0:
                self.turn_id = turn_id
            if turn_id == self.turn_id:
                self.assistant.append(text)

    def on_playback_done(self):
        """The reply finished playing on the robot."""
        if self.replying:
            self.finished = True

    def on_disconnect(self):
        """The server connection dropped."""
        self.disconnected = True

    def assistant_text(self):
        """The reply sentences joined."""
        return ' '.join(t.strip() for t in self.assistant if t.strip())

    def outcome(self, now):
        """Return None while running, else (success, status)."""
        if self.finished:
            return True, 'completed'
        if self.disconnected:
            return False, 'disconnected'
        if not self.heard and now >= self.wait_deadline:
            return False, 'no_speech'
        if now >= self.timeout_deadline:
            return False, 'timeout'
        return None
