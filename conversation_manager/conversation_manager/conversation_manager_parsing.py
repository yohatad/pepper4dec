"""conversation_manager_parsing.py

Pure text handling for LLM responses: parsing the JSON answer/intent object,
the incremental JSON-string decoder the streaming path uses, and the NAOqi
speech-tag post-processing. Standard library only, so it can be imported and
tested without openai, chromadb or a running ROS graph.

conversation_manager_implementation re-exports every name here, so existing
imports from that module keep working.

Author: Yohannes Tadesse Haile
Affiliation: Carnegie Mellon University Africa
Email: yohatad123@gmail.com
Date: February 28, 2026
Version: v1.0

Copyright (C) 2025 Carnegie Mellon University Africa
This software is provided 'as-is' for research and educational purposes
within the DEC project.
"""

import json
import re
from typing import List, Tuple


def parse_json_string_value(s: str) -> Tuple[str, bool]:
    """
    Parse characters of a JSON string value after the opening quote.

    Returns (decoded_text, is_complete).
    is_complete is True when the unescaped closing quote is found.
    Stops at buffer boundary (incomplete escape) without error.
    """
    result: List[str] = []
    i = 0
    escape_map = {'n': '\n', 't': '\t', 'r': '\r', '"': '"', '\\': '\\', '/': '/'}
    while i < len(s):
        c = s[i]
        if c == '\\':
            if i + 1 >= len(s):
                break  # Incomplete escape at buffer boundary
            next_c = s[i + 1]
            if next_c == 'u':
                if i + 5 < len(s):
                    result.append(chr(int(s[i + 2:i + 6], 16)))
                    i += 6
                else:
                    break  # Incomplete \uXXXX at buffer boundary
            else:
                result.append(escape_map.get(next_c, next_c))
                i += 2
        elif c == '"':
            return ''.join(result), True
        else:
            result.append(c)
            i += 1
    return ''.join(result), False


def _parse_llm_json(raw: str) -> dict:
    r"""
    Parse the JSON object from a raw LLM response, stripping any
    <think>…</think> chain-of-thought prefix first.

    NAOqi prosody tags (e.g. \vct=108\, \rspd=82\, \pau=200\) contain
    backslashes that are not valid JSON escape sequences.  If the first parse
    attempt fails we escape those lone backslashes and retry.

    Returns an empty dict if both attempts fail.
    """
    cleaned = raw.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()

    # First attempt — try as-is (handles correctly-escaped JSON)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError, ValueError):
        pass

    # Second attempt — escape lone backslashes from NAOqi tags so JSON can parse.
    # Only preserve \" \\ \/ and \uXXXX as valid JSON escapes.
    # \r \n \b \f \t are intentionally NOT preserved because the LLM uses \r
    # as the start of \rspd= tags, not as a carriage-return escape.
    try:
        escaped = re.sub(r'\\(?!["\\/u])', r'\\\\', cleaned)
        return json.loads(escaped)
    except (json.JSONDecodeError, AttributeError, ValueError):
        return {}


# Quote characters an LLM may wrap the answer in: ASCII double/single, plus the
# typographic pairs models emit when they "prettify" output. Written as explicit
# \u escapes on purpose — spelling the curly characters as adjacent string
# literals concatenates them down to the two ASCII quotes, which is exactly how
# the curly ones silently went missing from this set before.
_SURROUNDING_QUOTES = (
    '"'          # U+0022 quotation mark
    "'"          # U+0027 apostrophe
    '\u201c'   # left double quotation mark
    '\u201d'   # right double quotation mark
    '\u2018'   # left single quotation mark
    '\u2019'   # right single quotation mark
)


def extract_answer_from_raw(raw: str) -> str:
    r"""
    Extract the spoken answer from a raw LLM response.

    Handles:
      - Thinking-capable models:  <think>...</think>{"answer": "..."}
      - JSON structured response: {"intent": "...", "answer": "..."}
      - Plain-text fallback:      raw text returned as-is

    Post-processing:
      - Strips surrounding curly/smart quotes the LLM sometimes adds.
      - Converts *tag=N* placeholders → \tag=N\ NAOqi control sequences.

    Note: sentence-level speed tags are applied separately in apply_speech_tags().
    """
    parsed = _parse_llm_json(raw)
    if parsed:
        answer = parsed.get("answer", raw)
    else:
        # Plain-text fallback
        answer = raw.strip()
        if "</think>" in answer:
            answer = answer.split("</think>", 1)[1].strip()

    answer = answer.strip(_SURROUNDING_QUOTES)

    # Convert *tag=N* placeholders → \tag=N\ NAOqi control sequences.
    # The LLM uses * to avoid JSON backslash escape conflicts.
    answer = re.sub(r'\*([a-z]+=\d+)\*', r'\\\1\\', answer)

    return answer.strip()


# Intents that benefit from slightly slower speech for clarity
_SLOW_INTENTS = {"ASK_EXHIBIT_QUESTION", "ASK_TOUR_META"}


def apply_speech_tags(answer: str, intent: str) -> str:
    r"""
    Prepend a sentence-level NAOqi speed tag based on intent.

    Word-level tags (*pau=N*, etc.) are already embedded in the answer by the
    LLM and converted by extract_answer_from_raw(). This function only adds
    the sentence-level \rspd=N\ prefix so the LLM never has to generate it.

    Args:
        answer: cleaned answer text (may already contain \pau=N\ tags)
        intent: LLM-classified intent string

    Returns:
        answer with \rspd=N\ prepended for slow intents, unchanged otherwise.
    """
    if not answer:
        return answer

    # No speed tag for short fixed responses
    if answer.strip().lower() in ("yes", "no"):
        return answer

    if intent in _SLOW_INTENTS:
        return "\\rspd=85\\" + answer

    return answer


def extract_intent_from_raw(raw: str) -> Tuple[str, float]:
    """
    Extract the intent label and confidence score from a raw LLM response.

    Returns ``(intent, confidence)`` where intent is one of:
      ASK_EXHIBIT_QUESTION | ASK_TOUR_META | NAVIGATION_REQUEST |
      SOCIAL_SMALL_TALK | OFF_TOPIC | STOP | AFFIRMATIVE | NEGATIVE
    and confidence is in [0.0, 1.0].

    Falls back to ("UNKNOWN", 0.0) if the JSON cannot be parsed or the
    fields are missing.
    """
    parsed = _parse_llm_json(raw)
    intent = parsed.get("intent", "UNKNOWN") if parsed else "UNKNOWN"
    confidence = parsed.get("confidence", 0.0) if parsed else 0.0
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return str(intent), confidence
