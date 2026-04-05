"""Deterministic event calendar for LME temporal queries.

Ingestion-time indexing:
  1. For each turn, extract date mentions using dateparser with the turn's
     timestamp as anchor ("3 weeks ago" -> absolute date).
  2. Store (event_date, turn_idx, mention_text, snippet) in a sorted table
     indexed by date.

Query-time lookup:
  1. For temporal questions ("What date was X?", "How many weeks ago?"),
     look up events by date window.
  2. Pair with dateparser to compute date arithmetic deterministically,
     instead of relying on the LLM.

No LLM calls. Fully reproducible. Generalizable to any timestamped conversation.

This module DOES NOT replace mabench/event_calendar.py (FC keyword overlap
channel). It lives alongside as an independent LME-focused calendar.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from dateparser.search import search_dates

# Turn text often starts with its own timestamp like "[2023/06/03 (Sat) 14:30]".
# We strip that prefix before date extraction to avoid polluting the calendar
# with the anchor itself.
_TURN_TIMESTAMP_RE = re.compile(
    r"^\s*\[\s*\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\s*\([^)]*\)\s*\d{1,2}:\d{2}\s*\]\s*"
)

# dateparser settings: anchor to turn time, prefer past (LME conversations are
# typically about past events). Month-only mentions ("in March") resolve to
# day=1, which is useful for event-ordering queries even when imprecise.
_DATEPARSER_BASE_SETTINGS = {
    "PREFER_DATES_FROM": "past",
    "RETURN_AS_TIMEZONE_AWARE": False,
    "REQUIRE_PARTS": ["month"],  # skip noise like bare "3" or "tomorrow"
}

# Minimum character length for a mention to be considered. search_dates
# sometimes splits dates awkwardly ("3 last" from "June 3 last month");
# we filter obvious noise.
_MIN_MENTION_LEN = 4

# ─── Strict signal validators ────────────────────────────────────────────
# A mention must contain at least ONE strong date signal, otherwise we
# reject it as noise. dateparser.search_dates is aggressive — it interprets
# durations like "2 months" and "a 30-minute" as dates, which pollutes the
# calendar. We require one of these:
#   - month name (january, feb, etc.)
#   - ordinal day (1st, 2nd, 3rd, ..., 31st)
#   - weekday + date context ("last monday", "next friday")
#   - explicit relative anchor ("ago", "yesterday", "today", "tomorrow")
#   - last/next/this + time-unit or weekday
#   - 4-digit year (1900-2099)
_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
    re.IGNORECASE,
)
_ORDINAL_RE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)\b", re.IGNORECASE)
_REL_ANCHOR_RE = re.compile(
    r"\b(ago|yesterday|today|tomorrow|tonight)\b",
    re.IGNORECASE,
)
_LAST_NEXT_RE = re.compile(
    r"\b(last|next|this)\s+(week|month|year|weekend|night|morning|evening|"
    r"sun(?:day)?|mon(?:day)?|tues?(?:day)?|wed(?:nesday)?|thur?s?(?:day)?|"
    r"fri(?:day)?|sat(?:urday)?)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# Numeric slash-dates like "3/8", "3/8/2023", "12/25" (plausible M/D or M/D/Y).
# Month 1-12, day 1-31. Matched with word-boundary to avoid "2/3 of".
_NUMERIC_DATE_RE = re.compile(
    r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])(?:/\d{2,4})?\b"
)

# Blacklist patterns — reject mentions matching any of these
# "10-minute", "30-minute", "1-year" (number-word compounds = durations)
_DURATION_HYPHEN_RE = re.compile(r"\d+-[a-zA-Z]")
# "5-10", "4-6", "2 / 2" (numeric ranges, hyphen-separated).
# NOTE: does not cover slash-dates (handled by _NUMERIC_DATE_RE exemption).
_NUM_RANGE_RE = re.compile(r"\d+\s*-\s*\d+")
# "30%", "5-10%" (percentages)
_PCT_RE = re.compile(r"%")
# "#4, #6" (numeric lists)
_HASH_NUM_RE = re.compile(r"#\s*\d+")


def _has_strong_date_signal(mention: str) -> bool:
    """Return True if mention contains a strong date keyword."""
    return bool(
        _MONTH_RE.search(mention)
        or _ORDINAL_RE.search(mention)
        or _REL_ANCHOR_RE.search(mention)
        or _LAST_NEXT_RE.search(mention)
        or _YEAR_RE.search(mention)
        or _NUMERIC_DATE_RE.search(mention)
    )


def _has_noise_pattern(mention: str) -> bool:
    """Return True if mention matches a known noise pattern."""
    return bool(
        _DURATION_HYPHEN_RE.search(mention)
        or _NUM_RANGE_RE.search(mention)
        or _PCT_RE.search(mention)
        or _HASH_NUM_RE.search(mention)
    )


@dataclass(frozen=True)
class CalendarEvent:
    """A single date mention extracted from a turn."""
    event_date: date
    turn_idx: int
    mention_text: str  # The phrase that was parsed (e.g. "on June 3")
    snippet: str       # The full turn text (or relevant substring)

    def __repr__(self):  # pragma: no cover
        return (
            f"CalendarEvent({self.event_date.isoformat()}, "
            f"turn={self.turn_idx}, mention='{self.mention_text}')"
        )


def extract_date_mentions(
    text: str,
    anchor: datetime,
) -> list[tuple[str, date]]:
    """Extract date mentions from text, resolving relative dates against anchor.

    Returns list of (mention_phrase, parsed_date) tuples.

    Skips the turn's own timestamp prefix (e.g. '[2023/06/03 (Sat) 14:30]').
    Falls back to empty list on parse failures or no-hit text.

    Args:
        text: free-form turn text, possibly with leading timestamp prefix.
        anchor: the turn's chat_time, used as RELATIVE_BASE for dateparser.

    Returns:
        List of (mention_text, date) pairs. Empty if no dates found.
    """
    if not text:
        return []

    # Strip leading turn timestamp so it doesn't get parsed as a mention.
    clean_text = _TURN_TIMESTAMP_RE.sub("", text, count=1)
    if not clean_text.strip():
        return []

    settings = {**_DATEPARSER_BASE_SETTINGS, "RELATIVE_BASE": anchor}

    try:
        hits = search_dates(clean_text, settings=settings, languages=["en"])
    except Exception:
        return []

    if not hits:
        return []

    results: list[tuple[str, date]] = []
    for mention, parsed in hits:
        if not mention or parsed is None:
            continue
        m = mention.strip()
        if len(m) < _MIN_MENTION_LEN:
            continue
        if _has_noise_pattern(m):
            continue
        if not _has_strong_date_signal(m):
            continue
        results.append((m, parsed.date()))

    return results


class EventCalendarLME:
    """Sorted calendar of date-mentions extracted from turns.

    Build via repeated ingest_turn() calls, query via lookup_window() or
    lookup_month(). All queries return CalendarEvent instances with turn_idx
    back-references for downstream retrieval.
    """

    def __init__(self):
        self._events: list[CalendarEvent] = []
        # Sorted-by-date list gets rebuilt lazily after inserts.
        self._sorted_dirty = True

    def __len__(self) -> int:
        return len(self._events)

    def ingest_turn(
        self,
        turn_idx: int,
        turn_text: str,
        anchor: datetime,
    ) -> int:
        """Parse dates in turn_text, append CalendarEvent for each.

        Returns the number of events added from this turn.
        """
        mentions = extract_date_mentions(turn_text, anchor)
        added = 0
        for mention, parsed_date in mentions:
            self._events.append(
                CalendarEvent(
                    event_date=parsed_date,
                    turn_idx=turn_idx,
                    mention_text=mention,
                    snippet=turn_text,
                )
            )
            added += 1
        if added:
            self._sorted_dirty = True
        return added

    def _ensure_sorted(self):
        if self._sorted_dirty:
            self._events.sort(key=lambda e: e.event_date)
            self._sorted_dirty = False

    def lookup_window(
        self,
        target_date: date,
        window_days: int = 3,
    ) -> list[CalendarEvent]:
        """Return events within ±window_days of target_date."""
        self._ensure_sorted()
        return [
            e for e in self._events
            if abs((e.event_date - target_date).days) <= window_days
        ]

    def lookup_month(self, year: int, month: int) -> list[CalendarEvent]:
        """Return all events in a given (year, month)."""
        self._ensure_sorted()
        return [
            e for e in self._events
            if e.event_date.year == year and e.event_date.month == month
        ]

    def lookup_range(
        self,
        start_date: date,
        end_date: date,
    ) -> list[CalendarEvent]:
        """Return events with start_date <= event_date <= end_date."""
        self._ensure_sorted()
        return [
            e for e in self._events
            if start_date <= e.event_date <= end_date
        ]

    def all_events(self) -> list[CalendarEvent]:
        """Return all events, sorted by date."""
        self._ensure_sorted()
        return list(self._events)
