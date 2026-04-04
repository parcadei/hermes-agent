"""Parse raw LongMemEval chunks into structured conversation turns.

LongMemEval chunks are stringified Python lists with three patterns:

1. Full chunk: ``['Chat Time: ...', [{role/content dicts}]]``
2. Multi-session: tail text + ``'Chat Time: ...'`` + new turns
3. Continuation: raw text with no Chat Time marker

This module uses regex parsing (not ast.literal_eval) because the chunks
contain escaped quotes, truncated content, and other artifacts that make
safe Python evaluation unreliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    timestamp: str  # "2023/05/22 (Mon) 10:50" - raw string
    role: str  # "user", "assistant", or "continuation"
    content: str  # the message text
    turn_index: int  # global ordering


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches Chat Time headers: 'Chat Time: 2023/05/22 (Mon) 10:50'
_CHAT_TIME_RE = re.compile(
    r"""['"]Chat\s+Time:\s*(\d{4}/\d{2}/\d{2}\s*\([^)]+\)\s*\d{1,2}:\d{2})['"]"""
)

# Matches role/content dict entries.  Handles both quote styles for content.
# {'role': 'user', 'content': "..."}  or  {'role': 'user', 'content': '...'}
_ROLE_CONTENT_RE = re.compile(
    r"""\{\s*['"]role['"]\s*:\s*['"](\w+)['"]\s*,\s*['"]content['"]\s*:\s*"""
)


# Matches dict boundary artifacts left by MABench chunking:
#   "}, {'role': 'assistant', 'content': "   or   '}, {'role': 'user', 'content': '
# Also handles trailing/leading fragments like  "}]  or  [{
_DICT_BOUNDARY_RE = re.compile(
    r"""['"]?\s*\}\s*,?\s*\{?\s*['"]?role['"]?\s*:\s*['"](?:user|assistant)['"]"""
    r"""\s*,\s*['"]content['"]?\s*:\s*['"]?""",
    re.IGNORECASE,
)

# Residual list/dict envelope: leading [{ or trailing }] and variants
_ENVELOPE_RE = re.compile(r"""^[\[\]{}',\s]+|[\[\]{}',\s]+$""")

# Escaped whitespace literals from Python repr
_ESCAPED_WS_RE = re.compile(r"\\[nt]")


def _strip_dict_syntax(text: str) -> str:
    """Remove Python dict/list serialization artifacts from chunk text.

    Splits on ``{'role': '...', 'content': "`` boundaries, keeping only
    the content fragments (the actual conversation text).
    """
    # Split on dict boundaries — each piece is a content fragment
    parts = _DICT_BOUNDARY_RE.split(text)
    cleaned_parts: list[str] = []
    for part in parts:
        # Strip envelope brackets
        part = _ENVELOPE_RE.sub("", part)
        # Unescape whitespace
        part = _ESCAPED_WS_RE.sub(" ", part)
        # Strip dangling quotes/backslashes
        part = part.strip("'\"\\, \t\n\r")
        if part:
            cleaned_parts.append(part)
    result = " ".join(cleaned_parts)
    # Collapse multi-space
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def _extract_string_value(text: str, start: int) -> str:
    """Extract a quoted string value starting at position *start*.

    Handles both single- and double-quoted strings, including escaped quotes
    within the string.  Returns the content between the outermost quotes.
    """
    # Skip whitespace
    pos = start
    while pos < len(text) and text[pos] in " \t\n\r":
        pos += 1

    if pos >= len(text):
        return ""

    quote = text[pos]
    if quote not in ('"', "'"):
        # Not a quoted string; grab until next obvious delimiter
        end = text.find("}", pos)
        if end == -1:
            return text[pos:].strip()
        return text[pos:end].strip().rstrip("'\"")

    # Walk through the string respecting escapes
    pos += 1  # skip opening quote
    parts: list[str] = []
    while pos < len(text):
        ch = text[pos]
        if ch == "\\" and pos + 1 < len(text):
            next_ch = text[pos + 1]
            if next_ch == quote:
                parts.append(quote)
                pos += 2
                continue
            elif next_ch == "n":
                parts.append("\n")
                pos += 2
                continue
            elif next_ch == "t":
                parts.append("\t")
                pos += 2
                continue
            elif next_ch == "\\":
                parts.append("\\")
                pos += 2
                continue
            else:
                parts.append(ch)
                pos += 1
                continue
        if ch == quote:
            break
        parts.append(ch)
        pos += 1

    return "".join(parts)


def _extract_turns_from_segment(
    text: str, timestamp: str, turn_counter: int
) -> tuple[list[ConversationTurn], int]:
    """Extract role/content turns from a text segment that belongs to one session."""
    turns: list[ConversationTurn] = []
    for m in _ROLE_CONTENT_RE.finditer(text):
        role = m.group(1)
        content_start = m.end()
        content = _extract_string_value(text, content_start)
        turns.append(
            ConversationTurn(
                timestamp=timestamp,
                role=role,
                content=content,
                turn_index=turn_counter,
            )
        )
        turn_counter += 1
    return turns, turn_counter


def parse_chunk(
    text: str,
    turn_counter: int = 0,
    last_timestamp: str = "",
) -> tuple[list[ConversationTurn], int, str]:
    """Parse a raw LongMemEval chunk into conversation turns.

    Args:
        text: Raw chunk string from ConversationCreator.
        turn_counter: Current global turn index (for ordering across chunks).
        last_timestamp: Timestamp from previous chunk (for continuations).

    Returns:
        ``(turns, updated_turn_counter, last_timestamp_seen)``
    """
    if not text or not text.strip():
        return [], turn_counter, last_timestamp

    # Find all Chat Time markers and their positions
    chat_times: list[tuple[int, str]] = []
    for m in _CHAT_TIME_RE.finditer(text):
        chat_times.append((m.start(), m.group(1).strip()))

    all_turns: list[ConversationTurn] = []
    current_timestamp = last_timestamp

    if not chat_times:
        # Pattern 3: Continuation chunk — no Chat Time at all.
        # Try to extract role/content turns from embedded dict syntax.
        embedded_turns, turn_counter = _extract_turns_from_segment(
            text, last_timestamp, turn_counter
        )
        if embedded_turns:
            # Got structured turns — also capture any leading text
            # before the first {'role':...} match.
            first_match = _ROLE_CONTENT_RE.search(text)
            if first_match and first_match.start() > 0:
                leading = _strip_dict_syntax(text[:first_match.start()])
                if leading:
                    all_turns.append(
                        ConversationTurn(
                            timestamp=last_timestamp,
                            role="continuation",
                            content=leading,
                            turn_index=turn_counter,
                        )
                    )
                    turn_counter += 1
            all_turns.extend(embedded_turns)
        else:
            # No dict syntax found — clean and return as continuation
            cleaned = _strip_dict_syntax(text)
            if cleaned:
                all_turns.append(
                    ConversationTurn(
                        timestamp=last_timestamp,
                        role="continuation",
                        content=cleaned,
                        turn_index=turn_counter,
                    )
                )
                turn_counter += 1
        return all_turns, turn_counter, last_timestamp

    # There is at least one Chat Time marker.
    # Check for leading text before the first Chat Time (tail of previous session).
    first_ct_pos = chat_times[0][0]
    if first_ct_pos > 0:
        leading_text = text[:first_ct_pos].strip()
        # Strip wrapper brackets/punctuation from the stringified list envelope
        leading_text = leading_text.strip("[]'\"}, \t\n\r")
        leading_text = leading_text.strip()
        if leading_text:
            all_turns.append(
                ConversationTurn(
                    timestamp=last_timestamp,
                    role="continuation",
                    content=leading_text,
                    turn_index=turn_counter,
                )
            )
            turn_counter += 1

    # Process each Chat Time segment
    for i, (ct_pos, ts) in enumerate(chat_times):
        current_timestamp = ts
        # Determine the segment boundaries
        seg_start = ct_pos
        if i + 1 < len(chat_times):
            seg_end = chat_times[i + 1][0]
        else:
            seg_end = len(text)
        segment = text[seg_start:seg_end]

        segment_turns, turn_counter = _extract_turns_from_segment(
            segment, current_timestamp, turn_counter
        )
        all_turns.extend(segment_turns)

    return all_turns, turn_counter, current_timestamp
