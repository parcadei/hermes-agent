"""Deterministic tests for the event calendar retrieval boost.

Tests target ``_apply_event_calendar_boost`` in ``mabench/retrieval.py``.
The function gets called with already-scored retrieval candidates and boosts
any candidate whose ``index`` / ``turn_idx`` falls within the calendar's date
window around the query's Current Date.

The boost is additive (on an already-normalized score), bounded (each doc
can only be boosted once per window), and neutral (no boost) when:
  - calendar is None or empty,
  - query contains no "Current Date: YYYY/MM/DD" marker,
  - calendar.lookup_window returns no events for the window.
"""
from __future__ import annotations

from datetime import date, datetime

from mabench.event_calendar_lme import EventCalendarLME
from mabench.retrieval import _apply_event_calendar_boost

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_scored_docs(n: int = 5) -> list[dict]:
    """Return n fake scored candidates with indices 0..n-1 and descending scores."""
    return [
        {"index": i, "text": f"turn {i}", "score": float(n - i) * 0.1}
        for i in range(n)
    ]


def _make_calendar_with_events(hits: list[tuple[int, date, datetime]]) -> EventCalendarLME:
    """Build a calendar populated with the given (turn_idx, event_date, anchor) tuples.

    We don't go through the full date extractor — instead we inject events via
    a single ``ingest_turn`` call per hit, using turn text that mentions the
    target date unambiguously. The calendar's ``lookup_window`` uses only
    ``event_date`` and ``turn_idx``, so this is a faithful fixture.
    """
    cal = EventCalendarLME()
    for turn_idx, event_date, anchor in hits:
        # Format a turn text that dateparser will definitely resolve to event_date
        # when anchored at `anchor`. Use ordinal day + month name for stability.
        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        turn_text = (
            f"Visited the park on {month_names[event_date.month - 1]} "
            f"{event_date.day}, {event_date.year}."
        )
        cal.ingest_turn(turn_idx, turn_text, anchor)
    return cal


# ---------------------------------------------------------------------------
# _apply_event_calendar_boost: positive cases
# ---------------------------------------------------------------------------

class TestBoostAppliedToWindowTurns:
    def test_boosts_turns_in_window(self):
        """Turns 2 and 3 have events within the date window; they get boosted."""
        docs = _make_scored_docs(5)
        # Events on turns 2 and 3, anchored near q_date
        anchor = datetime(2023, 6, 15, 12, 0)
        cal = _make_calendar_with_events([
            (2, date(2023, 6, 10), anchor),  # 5 days before q_date
            (3, date(2023, 6, 20), anchor),  # 5 days after q_date
        ])
        query = "What happened last week? Current Date: 2023/06/15"
        boost = 0.15

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=60, boost=boost
        )

        # Same length, same doc set
        assert len(out) == 5
        assert {d["index"] for d in out} == {0, 1, 2, 3, 4}

        # Turns 2 and 3 received the boost, others unchanged
        scored_by_idx = {d["index"]: d["score"] for d in out}
        # Originals: {0:0.5, 1:0.4, 2:0.3, 3:0.2, 4:0.1}
        assert scored_by_idx[0] == 0.5
        assert scored_by_idx[1] == 0.4
        assert abs(scored_by_idx[2] - (0.3 + boost)) < 1e-9
        assert abs(scored_by_idx[3] - (0.2 + boost)) < 1e-9
        assert scored_by_idx[4] == 0.1

    def test_results_resorted_after_boost(self):
        """Boost pushes a lower-ranked candidate up the list."""
        docs = _make_scored_docs(5)
        # Only turn 4 (bottom-ranked) is in-window — should climb
        anchor = datetime(2023, 6, 15, 12, 0)
        cal = _make_calendar_with_events([
            (4, date(2023, 6, 12), anchor),
        ])
        query = "tell me. Current Date: 2023/06/15"

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=60, boost=0.5
        )

        # After boost: {0:0.5, 1:0.4, 2:0.3, 3:0.2, 4:0.1+0.5=0.6}
        # Sorted descending: 4, 0, 1, 2, 3
        assert out[0]["index"] == 4
        assert abs(out[0]["score"] - 0.6) < 1e-9

    def test_only_in_window_turns_boosted(self):
        """Events OUTSIDE the window are ignored, in-window turns boosted."""
        docs = _make_scored_docs(5)
        anchor = datetime(2023, 6, 15, 12, 0)
        cal = _make_calendar_with_events([
            (1, date(2023, 6, 14), anchor),  # in-window
            (2, date(2022, 1, 5), anchor),   # far out of window (>1yr)
        ])
        query = "Current Date: 2023/06/15"

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=30, boost=0.2
        )

        scored_by_idx = {d["index"]: d["score"] for d in out}
        # Turn 1 was 0.4, boosted to 0.6
        assert abs(scored_by_idx[1] - 0.6) < 1e-9
        # Turn 2 stays 0.3 (no event in window)
        assert abs(scored_by_idx[2] - 0.3) < 1e-9

    def test_boost_is_bounded_single_application(self):
        """Even if a turn has multiple in-window events, boost is applied once."""
        docs = _make_scored_docs(5)
        anchor = datetime(2023, 6, 15, 12, 0)
        # Turn 2 mentioned in 3 different dates, all in window
        cal = _make_calendar_with_events([
            (2, date(2023, 6, 10), anchor),
            (2, date(2023, 6, 12), anchor),
            (2, date(2023, 6, 14), anchor),
        ])
        query = "Current Date: 2023/06/15"
        boost = 0.15

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=60, boost=boost
        )

        scored_by_idx = {d["index"]: d["score"] for d in out}
        # Boost applied exactly once
        assert abs(scored_by_idx[2] - (0.3 + boost)) < 1e-9


# ---------------------------------------------------------------------------
# _apply_event_calendar_boost: pass-through / neutral cases
# ---------------------------------------------------------------------------

class TestBoostPassThrough:
    def test_none_calendar_passthrough(self):
        """When calendar is None, return docs unchanged."""
        docs = _make_scored_docs(5)
        original_scores = [d["score"] for d in docs]
        query = "Current Date: 2023/06/15"

        out = _apply_event_calendar_boost(
            docs, query, None, window_days=60, boost=0.15
        )

        assert [d["score"] for d in out] == original_scores

    def test_empty_calendar_passthrough(self):
        """Empty calendar -> no-op."""
        docs = _make_scored_docs(5)
        original = [(d["index"], d["score"]) for d in docs]
        cal = EventCalendarLME()
        query = "Current Date: 2023/06/15"

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=60, boost=0.15
        )

        assert [(d["index"], d["score"]) for d in out] == original

    def test_no_current_date_in_query_passthrough(self):
        """Query without a 'Current Date' marker -> no-op."""
        docs = _make_scored_docs(5)
        original = [(d["index"], d["score"]) for d in docs]
        anchor = datetime(2023, 6, 15, 12, 0)
        cal = _make_calendar_with_events([
            (2, date(2023, 6, 10), anchor),
            (3, date(2023, 6, 20), anchor),
        ])
        query = "What did I do last week?"  # no Current Date marker

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=60, boost=0.15
        )

        assert [(d["index"], d["score"]) for d in out] == original

    def test_empty_window_passthrough(self):
        """Window contains no events -> no-op."""
        docs = _make_scored_docs(5)
        original = [(d["index"], d["score"]) for d in docs]
        anchor = datetime(2023, 6, 15, 12, 0)
        # Events far outside the narrow window we query below
        cal = _make_calendar_with_events([
            (2, date(2021, 1, 1), anchor),
            (3, date(2024, 12, 31), anchor),
        ])
        query = "Current Date: 2023/06/15"

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=7, boost=0.15
        )

        assert [(d["index"], d["score"]) for d in out] == original

    def test_boost_zero_passthrough(self):
        """boost=0 -> no-op (defensive)."""
        docs = _make_scored_docs(5)
        original = [(d["index"], d["score"]) for d in docs]
        anchor = datetime(2023, 6, 15, 12, 0)
        cal = _make_calendar_with_events([
            (2, date(2023, 6, 10), anchor),
        ])
        query = "Current Date: 2023/06/15"

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=60, boost=0.0
        )

        assert [(d["index"], d["score"]) for d in out] == original

    def test_empty_docs_passthrough(self):
        """Empty results list -> returns empty list."""
        anchor = datetime(2023, 6, 15, 12, 0)
        cal = _make_calendar_with_events([(2, date(2023, 6, 10), anchor)])
        out = _apply_event_calendar_boost(
            [], "Current Date: 2023/06/15", cal, window_days=60, boost=0.15
        )
        assert out == []


# ---------------------------------------------------------------------------
# _apply_event_calendar_boost: regex parsing of Current Date
# ---------------------------------------------------------------------------

class TestCurrentDateParsing:
    def test_dash_separated_date(self):
        """Current Date with dashes (YYYY-MM-DD) is parsed."""
        docs = _make_scored_docs(5)
        anchor = datetime(2023, 6, 15, 12, 0)
        cal = _make_calendar_with_events([(2, date(2023, 6, 10), anchor)])
        query = "Current Date: 2023-06-15"

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=60, boost=0.15
        )

        scored_by_idx = {d["index"]: d["score"] for d in out}
        assert scored_by_idx[2] > 0.3  # boosted

    def test_date_with_day_and_time(self):
        """Current Date: 2023/06/15 (Thu) 14:30 also works."""
        docs = _make_scored_docs(5)
        anchor = datetime(2023, 6, 15, 12, 0)
        cal = _make_calendar_with_events([(2, date(2023, 6, 10), anchor)])
        query = "What happened. Current Date: 2023/06/15 (Thu) 14:30"

        out = _apply_event_calendar_boost(
            docs, query, cal, window_days=60, boost=0.15
        )

        scored_by_idx = {d["index"]: d["score"] for d in out}
        assert scored_by_idx[2] > 0.3  # boosted
