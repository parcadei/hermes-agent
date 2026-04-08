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
from datetime import datetime, timedelta

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
# PIECE 1b: extract_content_dates -- in-conversation date expressions
# ---------------------------------------------------------------------------

_DAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_LAST_DAY_RE = re.compile(
    r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_NEXT_DAY_RE = re.compile(
    r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_THIS_DAY_RE = re.compile(
    r"\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)


def _resolve_last_day(day_name: str, ref: datetime) -> datetime:
    """Resolve 'last <day>' relative to reference date."""
    target = _DAYS[day_name.lower()]
    days_back = (ref.weekday() - target) % 7
    if days_back == 0:
        days_back = 7  # "last Tuesday" on a Tuesday = 7 days ago
    return ref - timedelta(days=days_back)


def _resolve_next_day(day_name: str, ref: datetime) -> datetime:
    """Resolve 'next <day>' relative to reference date."""
    target = _DAYS[day_name.lower()]
    days_fwd = (target - ref.weekday()) % 7
    if days_fwd == 0:
        days_fwd = 7  # "next Friday" on a Friday = 7 days ahead
    return ref + timedelta(days=days_fwd)


def extract_content_dates(
    text: str,
    ref: datetime | None = None,
) -> list[dict]:
    """Extract date expressions from conversation text, resolved to datetimes.

    Uses deterministic regex for day-of-week ("last Tuesday", "next Monday"),
    then falls back to dateparser for other expressions ("three years ago",
    "March 15th", "yesterday").

    Returns list of {"expression": str, "resolved": datetime, "source": str}.
    Empty list if no dates found or ref is None.
    """
    if ref is None:
        return []

    results: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()

    # Pass 1: "last <day>" — deterministic
    for m in _LAST_DAY_RE.finditer(text):
        resolved = _resolve_last_day(m.group(1), ref)
        results.append({
            "expression": m.group(0),
            "resolved": resolved,
            "source": "regex_last_day",
        })
        seen_spans.add(m.span())

    # Pass 2: "next <day>" — deterministic
    for m in _NEXT_DAY_RE.finditer(text):
        resolved = _resolve_next_day(m.group(1), ref)
        results.append({
            "expression": m.group(0),
            "resolved": resolved,
            "source": "regex_next_day",
        })
        seen_spans.add(m.span())

    # Pass 3: "this <day>" — same week
    for m in _THIS_DAY_RE.finditer(text):
        target = _DAYS[m.group(1).lower()]
        delta = target - ref.weekday()
        resolved = ref + timedelta(days=delta)
        results.append({
            "expression": m.group(0),
            "resolved": resolved,
            "source": "regex_this_day",
        })
        seen_spans.add(m.span())

    # Pass 4: dateparser fallback for remaining expressions
    # Common relative patterns not covered by regex above
    _DATEPARSER_PATTERNS = [
        re.compile(r"\b(yesterday)\b", re.I),
        re.compile(r"\b(today)\b", re.I),
        re.compile(r"\b(tomorrow)\b", re.I),
        re.compile(r"\b(last (?:week|month|year))\b", re.I),
        re.compile(r"\b(next (?:week|month|year))\b", re.I),
        re.compile(r"\b(this (?:week|month|year))\b", re.I),
        re.compile(r"\b(\d+ (?:days?|weeks?|months?|years?) ago)\b", re.I),
        re.compile(r"\b((?:three|four|five|six|seven|eight|nine|ten) (?:days?|weeks?|months?|years?) ago)\b", re.I),
        # Absolute dates
        re.compile(
            r"\b((?:january|february|march|april|may|june|july|august"
            r"|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)\b",
            re.I,
        ),
        re.compile(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"),
        re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b"),
    ]

    try:
        import dateparser
        dp_settings = {"RELATIVE_BASE": ref, "PREFER_DATES_FROM": "past"}
    except ImportError:
        dateparser = None
        dp_settings = {}

    if dateparser is not None:
        for pat in _DATEPARSER_PATTERNS:
            for m in pat.finditer(text):
                # Skip if this span overlaps with an already-resolved span
                if any(m.start() < e and m.end() > s for s, e in seen_spans):
                    continue
                parsed = dateparser.parse(m.group(1), settings=dp_settings)
                if parsed is not None:
                    results.append({
                        "expression": m.group(1),
                        "resolved": parsed,
                        "source": "dateparser",
                    })
                    seen_spans.add(m.span())

    return results


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
        "temporal_prefix": str -- e.g. "[Date: 2023-01-15, 10 days ago / ~1.4 weeks ago]"
        "days_from_query": int | None -- signed days (negative = before query)

    Also adds pairwise deltas between adjacent (by time) facts.
    If query_date is None, returns facts unchanged (neutral).
    """
    if query_date is None:
        for f in facts:
            f["temporal_prefix"] = ""
            f["days_from_query"] = None
        return facts

    # Preserve retrieval order — do NOT re-sort by session_time.
    # BM25 relevance ranking is critical; re-sorting destroys it and
    # causes -2.3% SubEM regression on LME (77/300 vs 83/300).

    for f in facts:
        st = f.get("session_time")
        if st is None:
            f["temporal_prefix"] = ""
            f["days_from_query"] = None
            continue
        delta_days = (query_date - st).days
        date_str = st.strftime("%Y-%m-%d")
        if delta_days > 0:
            # Build natural language relative durations the LLM can read directly.
            # Validated: all 4 local models (3B-27B) correctly compute temporal
            # answers from this format (e.g. "~3.0 weeks ago" → "3 weeks ago").
            parts = [f"{delta_days} days ago"]
            if delta_days >= 7:
                weeks = delta_days / 7
                parts.append(f"~{weeks:.1f} weeks ago")
            if delta_days >= 30:
                months = delta_days / 30.44
                parts.append(f"~{months:.1f} months ago")
            if delta_days >= 365:
                years = delta_days / 365.25
                parts.append(f"~{years:.1f} years ago")
            f["temporal_prefix"] = f"[Date: {date_str}, {' / '.join(parts)}]"
        elif delta_days < 0:
            f["temporal_prefix"] = f"[Date: {date_str}, {abs(delta_days)} days in the future]"
        else:
            f["temporal_prefix"] = f"[Date: {date_str}, same day]"
        f["days_from_query"] = -delta_days  # negative = before query

    return facts


def format_temporal_context(enriched_facts: list[dict]) -> str:
    """Format enriched facts into a prompt string with temporal context.

    Output example:
        [Date: 2023-01-15, 10 days ago / ~1.4 weeks ago] User mentioned loving sushi.
        [Date: 2023-01-20, 5 days ago] User said they started a new job.
        (5 days between these events)
        [Date: 2023-01-25, same day] User asked about restaurants.
    """
    lines = []
    for f in enriched_facts:
        prefix = f.get("temporal_prefix", "")
        text = f["text"]
        if prefix:
            lines.append(f"{prefix} {text}")
        else:
            lines.append(text)
    return "\n".join(lines)
