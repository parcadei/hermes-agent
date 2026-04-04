"""Always-on temporal pipeline -- no intent routing.

Provides session-time parsing, query-date extraction, temporal proximity
scoring, and fact enrichment for answer prompts. All functions are designed
to be neutral (no-op) when temporal information is absent, so they can run
unconditionally on every ingestion and query path.

PIECE 1: parse_chat_time   -- extract session timestamp from LME chunk text
PIECE 2: parse_query_date  -- extract query anchor from LME query text
PIECE 3: temporal_proximity -- continuous decay score for retrieval
PIECE 4: enrich_with_temporal + format_temporal_context -- pre-computed
         deltas for answer prompt
"""

from __future__ import annotations

import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

CHAT_TIME_RE = re.compile(
    r"Chat\s*Time\s*:\s*(\d{4})[/\-](\d{2})[/\-](\d{2})\s*\([^)]*\)\s*(\d{2}):(\d{2})"
)
CHAT_TIME_SPACED_RE = re.compile(
    r"Chat\s+Time\s*:\s*(\d{4})\s*/\s*(\d{2})\s*/\s*(\d{2})\s*\(\s*\w+\s*\)\s*(\d{2})\s*:\s*(\d{2})"
)
QUERY_DATE_RE = re.compile(
    r"Current\s+Date\s*:\s*(\d{4})[/\-](\d{2})[/\-](\d{2})\s*\([^)]*\)\s*(\d{2}):(\d{2})"
)


# ---------------------------------------------------------------------------
# PIECE 1: parse_chat_time
# ---------------------------------------------------------------------------

def parse_chat_time(text: str) -> datetime | None:
    """Extract Chat Time from an LME chunk. Returns datetime or None."""
    for pattern in [CHAT_TIME_RE, CHAT_TIME_SPACED_RE]:
        m = pattern.search(text)
        if m:
            y, mo, d, h, mi = (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                int(m.group(4)),
                int(m.group(5)),
            )
            return datetime(y, mo, d, h, mi)
    return None


# ---------------------------------------------------------------------------
# PIECE 2: parse_query_date
# ---------------------------------------------------------------------------

def parse_query_date(query_text: str) -> datetime | None:
    """Extract Current Date from an LME query. Returns datetime or None."""
    m = QUERY_DATE_RE.search(query_text)
    if m:
        y, mo, d, h, mi = (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            int(m.group(5)),
        )
        return datetime(y, mo, d, h, mi)
    return None


# ---------------------------------------------------------------------------
# PIECE 3: temporal_proximity
# ---------------------------------------------------------------------------

def temporal_proximity(
    session_time: datetime | None,
    query_date: datetime | None,
    alpha: float = 0.01,
) -> float:
    """Continuous temporal proximity score. Always-on, neutral when missing."""
    if session_time is None or query_date is None:
        return 1.0
    days_apart = abs((query_date - session_time).total_seconds()) / 86400.0
    return 1.0 / (1.0 + alpha * days_apart)


# ---------------------------------------------------------------------------
# PIECE 4: enrich_with_temporal + format_temporal_context
# ---------------------------------------------------------------------------

def enrich_with_temporal(
    facts: list[dict],
    query_date: datetime | None,
) -> list[dict]:
    """Enrich retrieved facts with pre-computed temporal deltas.

    Each fact dict must have:
        "text": str -- the fact text
        "session_time": datetime | None -- when the session occurred

    Returns enriched fact dicts with added fields:
        "temporal_prefix": str -- e.g. "[2023-01-15, 10 days before query]"
        "days_from_query": int | None -- signed days (negative = before query)

    Also adds pairwise deltas between adjacent (by time) facts.
    If query_date is None, returns facts unchanged (neutral).
    """
    if query_date is None:
        for f in facts:
            f["temporal_prefix"] = ""
            f["days_from_query"] = None
        return facts

    # Sort by session_time (None last)
    def _sort_key(f):
        st = f.get("session_time")
        if st is None:
            return (1, datetime.max)
        return (0, st)

    sorted_facts = sorted(facts, key=_sort_key)

    for f in sorted_facts:
        st = f.get("session_time")
        if st is None:
            f["temporal_prefix"] = ""
            f["days_from_query"] = None
            continue
        delta_days = (query_date - st).days
        date_str = st.strftime("%Y-%m-%d")
        if delta_days > 0:
            f["temporal_prefix"] = f"[{date_str}, {delta_days} days before query]"
        elif delta_days < 0:
            f["temporal_prefix"] = f"[{date_str}, {abs(delta_days)} days after query]"
        else:
            f["temporal_prefix"] = f"[{date_str}, same day as query]"
        f["days_from_query"] = -delta_days  # negative = before query

    # Pairwise deltas between adjacent facts (by time)
    timed = [f for f in sorted_facts if f.get("session_time") is not None]
    pairwise_deltas = []
    for i in range(len(timed) - 1):
        t1 = timed[i]["session_time"]
        t2 = timed[i + 1]["session_time"]
        delta = abs((t2 - t1).days)
        pairwise_deltas.append(delta)

    # Attach pairwise info to each timed fact (delta to next)
    for i, f in enumerate(timed):
        if i < len(pairwise_deltas):
            f["days_to_next_fact"] = pairwise_deltas[i]
        else:
            f["days_to_next_fact"] = None

    return sorted_facts


def format_temporal_context(enriched_facts: list[dict]) -> str:
    """Format enriched facts into a prompt string with temporal context.

    Output example:
        [2023-01-15, 10 days before query] User mentioned loving sushi.
        [2023-01-20, 5 days before query] User said they started a new job.
        (5 days between these events)
        [2023-01-25, same day as query] User asked about restaurants.
    """
    lines = []
    timed_idx = 0
    timed_facts = [f for f in enriched_facts if f.get("session_time") is not None]

    for f in enriched_facts:
        prefix = f.get("temporal_prefix", "")
        text = f["text"]
        if prefix:
            lines.append(f"{prefix} {text}")
        else:
            lines.append(text)

        # Insert pairwise delta after each timed fact (except last)
        if f.get("session_time") is not None:
            delta = f.get("days_to_next_fact")
            if delta is not None and delta > 0:
                lines.append(f"  ({delta} days between these events)")
            timed_idx += 1

    return "\n".join(lines)
