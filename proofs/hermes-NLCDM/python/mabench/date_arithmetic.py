"""Deterministic date-arithmetic post-processor for LME temporal questions.

Classifies temporal-reasoning questions that require date arithmetic
(e.g. "How many weeks ago did I attend the festival?"), extracts
candidate event dates from retrieved turns, and computes the answer
deterministically using dateparser + datetime arithmetic.

The LLM frequently hallucinates dates even when retrieval surfaces the
correct turn. This module bypasses that by doing the arithmetic locally.

No LLM calls. Pure library. Safe to call from any retrieval pipeline.

See mabench.event_calendar_lme for shared date-mention extraction logic.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime

from mabench.event_calendar_lme import extract_date_mentions

# ---------------------------------------------------------------------------
# Classification patterns
# ---------------------------------------------------------------------------
# Each pattern returns a label naming the arithmetic to perform.
# Patterns are tried in order; the first match wins.
#
# Labels:
#   weeks_ago   -> (query_date - event_date).days // 7
#   days_ago    -> (query_date - event_date).days
#   months_ago  -> (query_date - event_date).days // 30
#   years_ago   -> (query_date - event_date).days // 365
#   days_passed -> (query_date - event_date).days   (formatted "N days")
#   days_between-> |d1 - d2|                         (formatted "N days")
#   date_of     -> event_date                        (formatted "Month Dth")
#   how_long_ago-> alias for weeks_ago (rare)
#   none        -> not an arithmetic question

_UNIT_TO_LABEL = {
    "week": "weeks_ago",
    "weeks": "weeks_ago",
    "day": "days_ago",
    "days": "days_ago",
    "month": "months_ago",
    "months": "months_ago",
    "year": "years_ago",
    "years": "years_ago",
    "hour": "days_ago",   # degenerate; map to days
    "hours": "days_ago",
}

_X_AGO_RE = re.compile(
    r"how\s+many\s+(weeks?|days?|months?|years?|hours?)\s+ago",
    re.IGNORECASE,
)
# "how many days had passed BETWEEN ... and ..." -> days_between
_X_BETWEEN_RE = re.compile(
    r"how\s+many\s+(weeks?|days?|months?|years?|hours?)\s+(?:had|have|has)?\s*"
    r"passed\s+between\b",
    re.IGNORECASE,
)
# "how many days had passed since X" / "how many days did I spend" -> days_passed
_X_PASSED_RE = re.compile(
    r"how\s+many\s+(weeks?|days?|months?|years?|hours?)\s+"
    r"(?:had|have|has|did)\s+(?:passed|spend|spent|take)",
    re.IGNORECASE,
)
_HOW_LONG_AGO_RE = re.compile(r"how\s+long\s+ago", re.IGNORECASE)
_WHAT_DATE_RE = re.compile(
    r"what\s+(?:was\s+the\s+|is\s+the\s+)?date\b",
    re.IGNORECASE,
)
_ON_WHAT_DATE_RE = re.compile(r"on\s+what\s+date\b", re.IGNORECASE)


def classify_date_arithmetic_question(question: str) -> str:
    """Return a classification label for this question.

    Labels: weeks_ago, days_ago, months_ago, years_ago, days_passed,
    days_between, date_of, how_long_ago, none.

    Args:
        question: raw question text (may contain a "Current Date:" prefix).

    Returns:
        A label string. "none" if the question does not require arithmetic.
    """
    if not question:
        return "none"
    q = question.lower()

    # "between X and Y" takes precedence over plain "passed"
    m_between = _X_BETWEEN_RE.search(q)
    if m_between:
        return "days_between"

    m_passed = _X_PASSED_RE.search(q)
    if m_passed:
        unit = m_passed.group(1).rstrip("s")
        # "days passed" is different from "days ago" — emits "N days" format
        if unit in ("day", "hour"):
            return "days_passed"
        # months/years passed -> weeks/months-ago shape
        return _UNIT_TO_LABEL.get(m_passed.group(1), "days_passed")

    m_ago = _X_AGO_RE.search(q)
    if m_ago:
        return _UNIT_TO_LABEL[m_ago.group(1).lower()]

    if _HOW_LONG_AGO_RE.search(q):
        return "how_long_ago"

    if _ON_WHAT_DATE_RE.search(q) or _WHAT_DATE_RE.search(q):
        return "date_of"

    return "none"


# ---------------------------------------------------------------------------
# Query-date extraction
# ---------------------------------------------------------------------------

_CURRENT_DATE_RE = re.compile(
    r"Current Date:\s*(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})"
)


def extract_query_date(question: str) -> date | None:
    """Parse the "Current Date: YYYY/MM/DD" prefix from a question."""
    if not question:
        return None
    m = _CURRENT_DATE_RE.search(question)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Arithmetic primitives
# ---------------------------------------------------------------------------

_UNIT_DIVISOR = {
    "days": 1,
    "weeks": 7,
    "months": 30,
    "years": 365,
}


def compute_units_ago(
    query_date: date, event_date: date, unit: str
) -> int:
    """Compute (query_date - event_date) in the requested unit.

    Uses integer division. "months" uses 30 days, "years" uses 365 days —
    this is what LME gold answers expect (calendar-aware arithmetic would
    not match "3 months ago" for 92-day spans).

    Args:
        query_date: the "now" date (from Current Date prefix).
        event_date: the dated event resolved from retrieval.
        unit: one of "days", "weeks", "months", "years".

    Returns:
        Number of units between the two dates. Always non-negative.
    """
    if unit not in _UNIT_DIVISOR:
        raise ValueError(f"Unknown unit: {unit!r}")
    delta_days = (query_date - event_date).days
    if delta_days < 0:
        delta_days = -delta_days
    return delta_days // _UNIT_DIVISOR[unit]


def compute_days_between(d1: date, d2: date) -> int:
    """Return absolute number of days between two dates."""
    return abs((d1 - d2).days)


# ---------------------------------------------------------------------------
# Event-date extraction from retrieved turns
# ---------------------------------------------------------------------------

_TURN_TS_RE = re.compile(
    r"\[\s*(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})\s*\([^)]*\)\s*(\d{1,2}):(\d{2})\s*\]"
)


def _turn_anchor(turn_text: str, fallback: datetime) -> datetime:
    """Extract the turn's own timestamp as anchor, or fallback."""
    m = _TURN_TS_RE.search(turn_text)
    if not m:
        return fallback
    try:
        return datetime(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
            int(m.group(5)),
        )
    except (ValueError, TypeError):
        return fallback


def extract_event_date_from_turns(
    turns: Iterable[str],
    query_anchor: datetime,
    include_turn_timestamps: bool = True,
) -> date | None:
    """Extract the most likely event date from a bag of retrieved turns.

    Strategy:
      1. For each turn, use its own timestamp as dateparser anchor (if
         present), else use query_anchor.
      2. Extract explicit date mentions via event_calendar_lme.extract_date_mentions
         AND collect the turn's own timestamp as a candidate.
      3. Score each candidate date by:
           - Count of turns mentioning / associated with it
           - Inverse distance to the cluster centroid of turn-timestamps
             (users who report recent events co-locate in time).
      4. Return the highest-scoring candidate. Ties broken by earliest date.

    Args:
        turns: iterable of turn strings (may include timestamp prefixes).
        query_anchor: fallback datetime anchor (typically the query date).
        include_turn_timestamps: if True, include turn timestamps as candidates.
            Default True.

    Returns:
        The dominant event date, or None.
    """
    turns = list(turns)
    if not turns:
        return None

    explicit_dates: list[date] = []
    turn_ts_counts: dict[date, int] = {}

    for turn in turns:
        if not turn:
            continue
        ts_anchor = _turn_anchor(turn, query_anchor)
        has_turn_ts = ts_anchor is not query_anchor
        if include_turn_timestamps and has_turn_ts:
            d = ts_anchor.date()
            turn_ts_counts[d] = turn_ts_counts.get(d, 0) + 1
        try:
            mentions = extract_date_mentions(turn, ts_anchor)
        except Exception:
            continue
        for _, d in mentions:
            explicit_dates.append(d)

    # Build candidate pool:
    #   - explicit mentions cumulate (weight 2 each): multiple turns that
    #     agree on the same date reinforce it.
    #   - turn-timestamps use log-scaled counts: turns at the same chat time
    #     are somewhat dependent (one session) but a cluster of 10 turns at
    #     the same time is stronger signal than 1 turn.
    #     weight = 1 + log(count): 1->1, 2->1.69, 5->2.61, 10->3.30
    scores: dict[date, float] = {}
    for d in explicit_dates:
        scores[d] = scores.get(d, 0.0) + 2.0
    for d, count in turn_ts_counts.items():
        scores[d] = scores.get(d, 0.0) + 1.0 + math.log(count)

    turn_ts_dates = list(turn_ts_counts.keys())

    if not scores:
        return None

    # Penalize dates far from the cluster of retrieved turn timestamps.
    # Rationale: if 15 retrieved turns all cluster around mid-July 2023,
    # a stray "September 2022" mention is likely unrelated noise.
    if turn_ts_dates:
        # Use median turn-timestamp date as cluster center
        sorted_ts = sorted(turn_ts_dates)
        center = sorted_ts[len(sorted_ts) // 2]
        for d in list(scores.keys()):
            delta_days = abs((d - center).days)
            # Decay: bonus for being within 90 days, penalty beyond
            if delta_days <= 90:
                scores[d] *= 1.0 + (90 - delta_days) / 180.0
            else:
                # hard penalty for distant candidates
                scores[d] *= max(0.05, 1.0 - (delta_days - 90) / 365.0)

    # Also prefer dates that are strictly before the query_anchor
    # (past events). query_anchor is typically the query date.
    query_date = query_anchor.date()
    for d in list(scores.keys()):
        if d > query_date:
            scores[d] *= 0.2  # heavy penalty for future dates

    # Pick highest score; tie-break by earliest date
    top_score = max(scores.values())
    candidates = [d for d, s in scores.items() if s >= top_score - 1e-9]
    return min(candidates)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _ordinal(n: int) -> str:
    """Return English ordinal suffix for n ("1st", "2nd", "3rd", "4th"...)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    last = n % 10
    if last == 1:
        return f"{n}st"
    if last == 2:
        return f"{n}nd"
    if last == 3:
        return f"{n}rd"
    return f"{n}th"


def format_answer(classification: str, value) -> str:
    """Format a computed value into the string shape gold answers expect.

    Args:
        classification: one of the labels from classify_date_arithmetic_question.
        value: int for ago/passed/between, date for date_of.

    Returns:
        A string formatted to match LME gold answer style:
        - "3 weeks ago", "2 days ago", "3 months ago"
        - "7 days" (for passed/between)
        - "June 3rd" (for date_of)
    """
    if classification in ("weeks_ago", "days_ago", "months_ago", "years_ago"):
        unit = classification.split("_")[0]  # weeks, days, months, years
        return f"{int(value)} {unit} ago"
    if classification == "how_long_ago":
        # default to weeks for ambiguous "how long ago"
        return f"{int(value)} weeks ago"
    if classification in ("days_passed", "days_between"):
        return f"{int(value)} days"
    if classification == "date_of":
        if not isinstance(value, date):
            raise ValueError(
                f"date_of expects a date value, got {type(value).__name__}"
            )
        return f"{_MONTH_NAMES[value.month - 1]} {_ordinal(value.day)}"
    raise ValueError(f"Unknown classification: {classification!r}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnsweredQuery:
    """Result of running the date-arithmetic post-processor on a question."""
    classification: str
    computed_answer: str | None
    event_date: date | None
    confidence: float

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AnsweredQuery(class={self.classification!r}, "
            f"answer={self.computed_answer!r}, "
            f"event={self.event_date}, conf={self.confidence:.2f})"
        )


_EMPTY_RESULT_TEMPLATE = {
    "computed_answer": None,
    "event_date": None,
    "confidence": 0.0,
}


def answer_date_arithmetic(
    question: str,
    retrieved_turns: Iterable[str],
) -> AnsweredQuery:
    """Full pipeline: classify, extract dates, compute, format.

    Args:
        question: raw question text with "Current Date: ..." prefix.
        retrieved_turns: iterable of top-K retrieved turn strings.

    Returns:
        AnsweredQuery with computed_answer populated if arithmetic succeeded.
    """
    classification = classify_date_arithmetic_question(question)
    if classification == "none":
        return AnsweredQuery(
            classification="none",
            computed_answer=None,
            event_date=None,
            confidence=0.0,
        )

    query_date = extract_query_date(question)
    query_anchor = (
        datetime(query_date.year, query_date.month, query_date.day, 12, 0)
        if query_date
        else datetime(2023, 1, 1, 12, 0)
    )

    event_date = extract_event_date_from_turns(retrieved_turns, query_anchor)
    if event_date is None:
        return AnsweredQuery(
            classification=classification,
            computed_answer=None,
            event_date=None,
            confidence=0.0,
        )

    if query_date is None:
        return AnsweredQuery(
            classification=classification,
            computed_answer=None,
            event_date=event_date,
            confidence=0.25,
        )

    # Confidence: 1.0 if we have both dates cleanly; drop if event comes
    # out after query date (usually a parse-ambiguity).
    confidence = 1.0
    if event_date > query_date:
        confidence = 0.3

    try:
        if classification == "date_of":
            answer_str = format_answer("date_of", event_date)
        elif classification == "days_between":
            # days_between needs 2 dates; we only have 1 from this pass.
            # Fall back to days-from-now (less precise but usable).
            days = compute_days_between(query_date, event_date)
            answer_str = format_answer("days_between", days)
        elif classification == "days_passed":
            days = compute_units_ago(query_date, event_date, "days")
            answer_str = format_answer("days_passed", days)
        elif classification == "how_long_ago":
            weeks = compute_units_ago(query_date, event_date, "weeks")
            answer_str = format_answer("how_long_ago", weeks)
        elif classification in ("weeks_ago", "days_ago", "months_ago", "years_ago"):
            unit = classification.split("_")[0]
            units = compute_units_ago(query_date, event_date, unit)
            answer_str = format_answer(classification, units)
        else:
            return AnsweredQuery(
                classification=classification,
                computed_answer=None,
                event_date=event_date,
                confidence=0.0,
            )
    except (ValueError, TypeError):
        return AnsweredQuery(
            classification=classification,
            computed_answer=None,
            event_date=event_date,
            confidence=0.0,
        )

    return AnsweredQuery(
        classification=classification,
        computed_answer=answer_str,
        event_date=event_date,
        confidence=confidence,
    )
