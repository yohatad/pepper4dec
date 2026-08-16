#!/usr/bin/env python3
"""
Unit tests for the LLM response-parsing helpers in
conversation_manager_implementation.py: JSON extraction from raw model output
(including <think> prefixes and NAOqi prosody tags), answer/intent extraction,
streaming JSON string decoding, and sentence-level speech tagging.

Pure string/JSON logic — no ROS graph, no OpenAI call, no Chroma database.

The module imports openai and chromadb at import time, so the whole file skips
when those aren't installed (same convention as person_detection's bag-replay
test skipping on a missing ONNX model).

Run via: colcon test --packages-select conversation_manager

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Date: Aug 16, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
"""

import json

import pytest

# openai / chromadb are heavy optional deps pulled in at module import.
#
# Deliberately NOT pytest.importorskip: under pytest 6.2.5 (the version ament
# ships) a module-level Skipped raised during collection aborts collection for
# the whole session, which silently disables the sibling flake8/pep257 tests.
# A pytestmark skipif collects normally and skips at run time instead.
try:
    from conversation_manager import conversation_manager_implementation as impl
except ImportError:  # pragma: no cover - exercised only where deps are absent
    impl = None

pytestmark = pytest.mark.skipif(
    impl is None,
    reason='conversation_manager deps (openai, chromadb) not installed')


# ─────────────────────────────────────────────────────────────────────────────
# _parse_llm_json
# ─────────────────────────────────────────────────────────────────────────────

def test_parses_plain_json():
    assert impl._parse_llm_json('{"intent": "STOP", "answer": "ok"}') == {
        'intent': 'STOP', 'answer': 'ok'}


def test_strips_think_prefix():
    """Thinking models emit <think>…</think> before the JSON payload."""
    raw = '<think>Let me consider the question.</think>{"answer": "Hello"}'
    assert impl._parse_llm_json(raw) == {'answer': 'Hello'}


def test_strips_think_prefix_with_surrounding_whitespace():
    raw = '  <think>\nreasoning\n</think>\n\n  {"answer": "Hi"}  '
    assert impl._parse_llm_json(raw) == {'answer': 'Hi'}


def test_recovers_json_containing_naoqi_prosody_tags():
    r"""
    Prosody tags like \rspd=82\ contain lone backslashes that are invalid JSON
    escapes. The parser must escape them and retry rather than give up — this
    is the single most common real-world payload shape.
    """
    raw = r'{"answer": "\rspd=82\Hello there"}'
    # Sanity check that this really is invalid JSON, so the test is exercising
    # the retry path rather than the happy path.
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)

    parsed = impl._parse_llm_json(raw)
    assert parsed.get('answer') == r'\rspd=82\Hello there'


def test_returns_empty_dict_on_unparseable_input():
    """Unparseable output must degrade to {}, never raise into the node."""
    for raw in ('not json at all', '', '   ', '{"unclosed": ', '[1, 2, 3'):
        assert impl._parse_llm_json(raw) == {}, raw


def test_json_array_is_not_treated_as_an_object():
    """A JSON array parses, but callers do .get() on the result."""
    parsed = impl._parse_llm_json('[1, 2, 3]')
    assert not isinstance(parsed, dict) or parsed == {}


# ─────────────────────────────────────────────────────────────────────────────
# extract_answer_from_raw
# ─────────────────────────────────────────────────────────────────────────────

def test_extracts_answer_field():
    assert impl.extract_answer_from_raw(
        '{"intent": "SOCIAL_SMALL_TALK", "answer": "Hello!"}') == 'Hello!'


def test_falls_back_to_plain_text():
    """Models that ignore the JSON instruction still have to be usable."""
    assert impl.extract_answer_from_raw('Just plain text.') == 'Just plain text.'


def test_plain_text_fallback_strips_think_block():
    raw = '<think>hmm</think>The museum opens at nine.'
    assert impl.extract_answer_from_raw(raw) == 'The museum opens at nine.'


def test_strips_surrounding_ascii_quotes():
    """Models like to wrap the answer in quotes; those must not be spoken."""
    assert impl.extract_answer_from_raw('{"answer": "\\"Hello\\""}') == 'Hello'
    assert impl.extract_answer_from_raw('{"answer": "\'Hello\'"}') == 'Hello'


def test_strips_surrounding_curly_quotes():
    r"""
    Typographic quotes must be stripped too.

    Regression guard: the strip set used to be written as adjacent string
    literals, which Python concatenated down to the two ASCII quotes, so the
    curly characters silently were not in the set at all despite the comment
    saying otherwise. They are now spelled as explicit \u escapes.
    """
    assert impl.extract_answer_from_raw('{"answer": "“Hello”"}') == 'Hello'
    assert impl.extract_answer_from_raw('{"answer": "‘Hello’"}') == 'Hello'


def test_quote_strip_set_really_contains_the_curly_characters():
    """Guards the literal itself, not just one round trip through the parser.

    Written against the constant so a future edit that collapses the escapes
    back into adjacent literals fails here with an obvious message.
    """
    for char in ('"', "'", '“', '”', '‘', '’'):
        assert char in impl._SURROUNDING_QUOTES, f'{char!r} (U+{ord(char):04X}) missing'


def test_does_not_strip_quotes_from_the_middle():
    """Only surrounding quotes go; quoted phrases inside the answer stay."""
    assert impl.extract_answer_from_raw(
        '{"answer": "He said “hello” to me"}') == 'He said “hello” to me'


def test_converts_star_placeholders_to_naoqi_tags():
    r"""
    The prompt asks the LLM to write *pau=300* instead of \pau=300\ so the
    payload stays valid JSON; this is where that gets converted back.
    """
    out = impl.extract_answer_from_raw('{"answer": "Wait *pau=300* then go"}')
    assert out == r'Wait \pau=300\ then go'


def test_converts_multiple_placeholders():
    out = impl.extract_answer_from_raw('{"answer": "*rspd=85*Hi *pau=200* there"}')
    assert out == r'\rspd=85\Hi \pau=200\ there'


def test_leaves_non_tag_asterisks_alone():
    """Only *name=digits* is a tag; ordinary emphasis must survive untouched."""
    out = impl.extract_answer_from_raw('{"answer": "That is *really* nice"}')
    assert out == 'That is *really* nice'


def test_result_is_stripped():
    assert impl.extract_answer_from_raw('{"answer": "  padded  "}') == 'padded'


# ─────────────────────────────────────────────────────────────────────────────
# extract_intent_from_raw
# ─────────────────────────────────────────────────────────────────────────────

def test_extracts_intent_and_confidence():
    intent, confidence = impl.extract_intent_from_raw(
        '{"intent": "NAVIGATION_REQUEST", "confidence": 0.87}')
    assert intent == 'NAVIGATION_REQUEST'
    assert confidence == pytest.approx(0.87)


def test_intent_defaults_to_unknown():
    """Unparseable or field-less output must not crash the intent router."""
    for raw in ('garbage', '{}', ''):
        assert impl.extract_intent_from_raw(raw) == ('UNKNOWN', 0.0), raw


def test_missing_confidence_defaults_to_zero():
    intent, confidence = impl.extract_intent_from_raw('{"intent": "STOP"}')
    assert (intent, confidence) == ('STOP', 0.0)


def test_confidence_strings_are_coerced_to_float():
    """Models sometimes quote the number; a string here would break comparisons."""
    _, confidence = impl.extract_intent_from_raw(
        '{"intent": "STOP", "confidence": "0.5"}')
    assert isinstance(confidence, float)
    assert confidence == pytest.approx(0.5)


def test_non_numeric_confidence_falls_back_to_zero():
    _, confidence = impl.extract_intent_from_raw(
        '{"intent": "STOP", "confidence": "high"}')
    assert confidence == 0.0


def test_intent_is_always_a_string():
    intent, _ = impl.extract_intent_from_raw('{"intent": 42}')
    assert isinstance(intent, str)


# ─────────────────────────────────────────────────────────────────────────────
# apply_speech_tags
# ─────────────────────────────────────────────────────────────────────────────

def test_slow_intents_get_a_speed_prefix():
    for intent in ('ASK_EXHIBIT_QUESTION', 'ASK_TOUR_META'):
        assert impl.apply_speech_tags('Some explanation.', intent) == \
            r'\rspd=85\Some explanation.'


def test_other_intents_are_untouched():
    for intent in ('SOCIAL_SMALL_TALK', 'NAVIGATION_REQUEST', 'STOP', 'UNKNOWN'):
        assert impl.apply_speech_tags('Hello.', intent) == 'Hello.'


def test_short_confirmations_never_get_a_speed_tag():
    """A drawn-out "yes" sounds wrong; these stay at normal speed."""
    for answer in ('yes', 'no', 'Yes', 'NO', '  yes  '):
        assert impl.apply_speech_tags(answer, 'ASK_EXHIBIT_QUESTION') == answer


def test_empty_answer_passes_through():
    assert impl.apply_speech_tags('', 'ASK_EXHIBIT_QUESTION') == ''


def test_tagging_is_not_cumulative_across_calls():
    """Applying to an already-tagged answer is the caller's bug to avoid, but
    the function must at least be a pure function of its inputs.
    """
    once = impl.apply_speech_tags('Hi there.', 'ASK_TOUR_META')
    assert impl.apply_speech_tags('Hi there.', 'ASK_TOUR_META') == once


# ─────────────────────────────────────────────────────────────────────────────
# parse_json_string_value — incremental decoder used by the streaming path
# ─────────────────────────────────────────────────────────────────────────────

def test_reads_up_to_the_closing_quote():
    assert impl.parse_json_string_value('hello", "next": 1') == ('hello', True)


def test_incomplete_string_reports_not_complete():
    """Mid-stream buffers have no closing quote yet; that is not an error."""
    assert impl.parse_json_string_value('hello wor') == ('hello wor', False)


def test_decodes_standard_escapes():
    text, complete = impl.parse_json_string_value(r'a\nb\tc\"d\\e"')
    assert (text, complete) == ('a\nb\tc"d\\e', True)


def test_decodes_unicode_escapes():
    text, complete = impl.parse_json_string_value(r'café"')
    assert (text, complete) == ('café', True)


def test_stops_cleanly_on_escape_split_across_buffers():
    r"""A buffer ending mid-escape (trailing \ or partial \uXX) must stop at the
    boundary and report incomplete, not raise or emit a mangled character.
    """
    assert impl.parse_json_string_value('abc\\') == ('abc', False)
    assert impl.parse_json_string_value(r'abc\u00') == ('abc', False)


def test_escaped_quote_does_not_terminate_the_string():
    text, complete = impl.parse_json_string_value(r'say \"hi\" now"')
    assert (text, complete) == ('say "hi" now', True)


def test_empty_input_is_empty_and_incomplete():
    assert impl.parse_json_string_value('') == ('', False)


def test_immediate_quote_is_an_empty_complete_string():
    assert impl.parse_json_string_value('"') == ('', True)


def test_incremental_decode_matches_whole_buffer_decode():
    """Feeding the whole buffer at once must agree with json.loads on the same
    string — the streaming decoder is only useful if it is faithful.
    """
    payload = r'Hello \"world\", café\nline two'
    text, complete = impl.parse_json_string_value(payload + '"')
    assert complete
    assert text == json.loads('"' + payload + '"')
