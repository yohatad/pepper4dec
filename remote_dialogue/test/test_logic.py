"""Tests for remote_dialogue_logic: speaker pacing, mic gate, turn outcome."""

import pytest

from remote_dialogue.remote_dialogue_logic import (MicrophoneGate, PlaybackScheduler,
                                                   TurnTracker)

RATE = 48000


def pcm(seconds):
    return b'\x00\x01' * int(seconds * RATE)


# ── PlaybackScheduler ────────────────────────────────────────────────────────

def drive(scheduler, start, stop, step=0.01, feed=None):
    """Advance a fake clock, collecting (time, piece) for everything sent."""
    sent = []
    t = start
    while t < stop:
        if feed:
            feed(t)
        piece = scheduler.next_piece(t)
        if piece is not None:
            sent.append((t, piece))
        t = round(t + step, 6)
    return sent


def test_scheduler_paces_pieces_and_end_time_matches_audio():
    s = PlaybackScheduler(RATE, piece_s=0.2, lead_s=0.15)
    s.push(pcm(1.0), now=0.0)
    s.mark_reply_complete()
    sent = drive(s, 0.0, 3.0)

    assert sum(len(p) for _, p in sent) == len(pcm(1.0))
    assert all(len(p) <= int(0.2 * RATE) * 2 for _, p in sent)
    # Never more than lead + one piece ahead of playback.
    clock_end = 0.0
    for t, piece in sent:
        assert clock_end - t <= 0.15 + 1e-9
        clock_end = max(clock_end, t) + len(piece) / 2 / RATE
    assert s.end_time == pytest.approx(1.0, abs=0.02)


def test_scheduler_holds_partial_audio_briefly_then_sends_it():
    s = PlaybackScheduler(RATE, piece_s=0.2, lead_s=0.15, max_hold_s=0.08)
    s.push(pcm(0.05), now=0.0)
    assert s.next_piece(0.01) is None           # waiting for a fuller piece
    piece = s.next_piece(0.09)                  # held too long: send what there is
    assert piece is not None and len(piece) == len(pcm(0.05))


def test_scheduler_streaming_input_keeps_gapless_end_time():
    """Audio arriving at playback speed in 40 ms chunks plays without gaps."""
    s = PlaybackScheduler(RATE)
    chunk = pcm(0.04)

    def feed(t):
        # one chunk every 40 ms for 1 s
        if round(t * 100) % 4 == 0 and t < 1.0:
            s.push(chunk, t)

    sent = drive(s, 0.0, 2.0, feed=feed)
    s.mark_reply_complete()
    sent += drive(s, 2.0, 3.0)
    total = sum(len(p) for _, p in sent) / 2 / RATE
    assert total == pytest.approx(1.0, abs=0.001)
    # Once started, playback ends within one hold window of the audio length.
    first = sent[0][0]
    assert s.end_time - first == pytest.approx(1.0, abs=0.15)


def test_playback_done_only_after_complete_reply_has_played():
    s = PlaybackScheduler(RATE)
    s.push(pcm(0.5), now=0.0)
    drive(s, 0.0, 0.3)
    assert not s.take_playback_done(0.3)        # reply not complete
    s.mark_reply_complete()
    drive(s, 0.3, 0.45)
    assert not s.take_playback_done(0.45)       # still playing
    end = s.end_time
    assert not s.take_playback_done(end + 0.05, tail_s=0.1)   # inside the tail
    assert s.take_playback_done(end + 0.1, tail_s=0.1)
    assert not s.take_playback_done(end + 1.0)  # reported once


def test_flush_drops_unsent_audio_but_keeps_end_time():
    s = PlaybackScheduler(RATE)
    s.push(pcm(1.0), now=0.0)
    s.next_piece(0.0)
    end = s.end_time
    s.flush(0.05)
    assert not s.has_pending()
    assert s.end_time == end
    assert s.drained(end)


# ── MicrophoneGate ───────────────────────────────────────────────────────────

def test_gate_closed_without_a_turn_or_listening():
    g = MicrophoneGate(guard_s=0.3)
    assert not g.is_live(10.0, playback_end=0.0)
    g.listening_enabled = True
    assert g.is_live(10.0, playback_end=0.0)
    g.listening_enabled = False
    g.turn_open = True
    assert g.is_live(10.0, playback_end=0.0)


def test_gate_mutes_while_robot_speaks_and_for_the_guard_after():
    g = MicrophoneGate(guard_s=0.3)
    g.turn_open = True
    g.server_speaking = True
    assert not g.is_live(1.0, playback_end=0.0)
    g.robot_speech_ended(now=2.0)
    assert not g.is_live(2.2, playback_end=1.9)
    assert g.is_live(2.31, playback_end=1.9)
    # Playback end later than the guard still mutes.
    assert not g.is_live(2.31, playback_end=2.2)


# ── TurnTracker ──────────────────────────────────────────────────────────────

def full_turn(t, user='hi', turn_id=1):
    t.on_state('listening')
    t.on_state('thinking')
    t.on_transcript('user', user, turn_id)
    t.on_state('speaking')
    t.on_transcript('assistant', 'First.', turn_id)
    t.on_transcript('assistant', ' Second. ', turn_id)
    t.on_state('idle')


def test_turn_completes_after_playback_done():
    t = TurnTracker(0.0, wait_s=5.0, timeout_s=60.0)
    full_turn(t)
    assert t.outcome(1.0) is None               # idle, but still playing
    t.on_playback_done()
    assert t.outcome(1.0) == (True, 'completed')
    assert t.user_text == 'hi'
    assert t.assistant_text() == 'First. Second.'
    assert t.turn_id == 1


def test_turn_no_speech_after_wait():
    t = TurnTracker(0.0, wait_s=5.0, timeout_s=60.0)
    assert t.outcome(4.9) is None
    assert t.outcome(5.0) == (False, 'no_speech')


def test_turn_listening_keeps_it_alive_past_wait_until_timeout():
    t = TurnTracker(0.0, wait_s=5.0, timeout_s=60.0)
    t.on_state('listening')
    assert t.outcome(30.0) is None
    assert t.outcome(60.0) == (False, 'timeout')


def test_turn_noise_that_server_drops_counts_as_not_heard():
    t = TurnTracker(0.0, wait_s=5.0, timeout_s=60.0)
    t.on_state('listening')
    t.on_state('idle')                          # no transcript, no reply
    assert t.outcome(5.0) == (False, 'no_speech')


def test_turn_ignores_welcome_line_that_starts_while_waiting():
    t = TurnTracker(0.0, wait_s=5.0, timeout_s=60.0)
    t.on_state('speaking')
    t.on_transcript('assistant', 'Hello, and welcome.', 0)
    t.on_state('idle')
    t.on_playback_done()
    assert t.outcome(1.0) is None
    full_turn(t)
    t.on_playback_done()
    assert t.outcome(2.0) == (True, 'completed')
    assert t.assistant_text() == 'First. Second.'


def test_turn_transcripts_in_any_order():
    t = TurnTracker(0.0, wait_s=5.0, timeout_s=60.0)
    t.on_state('thinking')
    t.on_state('speaking')
    t.on_transcript('assistant', 'Answer.', 4)
    t.on_transcript('user', 'question', 4)
    t.on_state('idle')
    t.on_playback_done()
    assert t.outcome(1.0) == (True, 'completed')
    assert (t.user_text, t.assistant_text(), t.turn_id) == ('question', 'Answer.', 4)


def test_text_turn_has_no_listening_phase():
    t = TurnTracker(0.0, wait_s=0.0, timeout_s=60.0, text_mode=True, user_text='typed')
    assert t.outcome(10.0) is None              # no no_speech in text mode
    t.on_state('speaking')
    t.on_transcript('assistant', 'Reply.', 2)
    t.on_transcript('user', 'server echo of typed', 2)
    t.on_state('idle')
    t.on_playback_done()
    assert t.outcome(11.0) == (True, 'completed')
    assert t.user_text == 'typed'


def test_turn_disconnect():
    t = TurnTracker(0.0, wait_s=5.0, timeout_s=60.0)
    t.on_state('listening')
    t.on_disconnect()
    assert t.outcome(1.0) == (False, 'disconnected')
