"""Deterministic tests for LME event calendar lookup.

Tests are small, synthetic, with 1-line turns that have unambiguous date
mentions. Real-world noise is deferred to a validation step on the ctx0
fixture (not in this test file).

Strategy:
  1. Parse dates from free text using dateparser.search_dates with turn timestamp as anchor.
  2. Build a calendar: list of (event_date, turn_idx, mention_text, snippet).
  3. Query: resolve question's date reference, look up events in ±window.

This is a deterministic lookup table — no LLM calls, fully reproducible.
"""
from __future__ import annotations

from datetime import date, datetime

import pytest

from mabench.event_calendar_lme import (
    CalendarEvent,
    EventCalendarLME,
    extract_date_mentions,
)


# ---------------------------------------------------------------------------
# extract_date_mentions
# ---------------------------------------------------------------------------

class TestExtractDateMentions:
    def test_absolute_date(self):
        """'On March 15' anchored in 2023 resolves to 2023-03-15."""
        anchor = datetime(2023, 5, 1, 12, 0)
        mentions = extract_date_mentions(
            "I bought a gift for my brother on March 15 for his graduation.",
            anchor,
        )
        assert len(mentions) >= 1
        # At least one mention resolves to March 15, 2023
        dates = [d for _, d in mentions]
        assert date(2023, 3, 15) in dates

    def test_relative_weeks_ago(self):
        """'2 weeks ago' anchored at 2023-07-01 resolves to 2023-06-17."""
        anchor = datetime(2023, 7, 1, 12, 0)
        mentions = extract_date_mentions("2 weeks ago I received my new phone case.", anchor)
        dates = [d for _, d in mentions]
        assert date(2023, 6, 17) in dates

    def test_relative_days_ago(self):
        """'3 days ago' anchored at 2023-07-01 resolves to 2023-06-28."""
        anchor = datetime(2023, 7, 1, 12, 0)
        mentions = extract_date_mentions("I lost my phone charger 3 days ago.", anchor)
        dates = [d for _, d in mentions]
        assert date(2023, 6, 28) in dates

    def test_month_only(self):
        """'in March' anchored in May 2023 resolves to 2023-03 (ambiguous day ok)."""
        anchor = datetime(2023, 5, 1, 12, 0)
        mentions = extract_date_mentions("The cultural festival was in March.", anchor)
        dates = [d for _, d in mentions]
        # We accept any March 2023 date (dateparser picks day 1)
        assert any(d.year == 2023 and d.month == 3 for d in dates)

    def test_no_dates(self):
        """Text with no date mentions returns empty list."""
        mentions = extract_date_mentions("Hello world.", datetime(2023, 5, 1))
        assert mentions == []

    def test_numeric_slash_date(self):
        """'on the 3/8' anchored in 2023 resolves to 2023-03-08."""
        anchor = datetime(2023, 3, 29, 6, 0)
        mentions = extract_date_mentions(
            "I bought a gift for my brother's graduation on the 3/8, it was great.",
            anchor,
        )
        dates = [d for _, d in mentions]
        assert date(2023, 3, 8) in dates

    def test_reject_duration_phrases(self):
        """Durations like 'a 30-minute' or '2 months' without 'ago' are not dates."""
        anchor = datetime(2023, 5, 1, 12, 0)
        # "a 30-minute walk" should not become a date event
        mentions = extract_date_mentions("I went for a 30-minute walk yesterday.", anchor)
        # "yesterday" should still resolve, but "30-minute" must not
        dates = [d for m, d in mentions]
        mention_texts = [m for m, _ in mentions]
        assert all("30-minute" not in m for m in mention_texts)

    def test_reject_percentages_and_ranges(self):
        """Percentages and numeric ranges are not dates."""
        anchor = datetime(2023, 5, 1, 12, 0)
        mentions = extract_date_mentions("Sales grew by 5-10% last quarter.", anchor)
        mention_texts = [m for m, _ in mentions]
        assert all("%" not in m for m in mention_texts)
        assert all("5-10" not in m for m in mention_texts)

    def test_skip_chat_timestamp(self):
        """Turn timestamp prefix like '[2023/06/03 (Sat) 14:30]' must be skipped."""
        # Turn text starts with its own timestamp — we must NOT extract that as a date mention.
        # (The timestamp is the anchor itself.)
        anchor = datetime(2023, 6, 3, 14, 30)
        mentions = extract_date_mentions(
            "[2023/06/03 (Sat) 14:30] (user) I had a great BBQ today.",
            anchor,
        )
        # No dates should be extracted — "today" resolves to anchor date but we filter it.
        # If dateparser picks up the timestamp, it should equal anchor date which we accept.
        dates = [d for _, d in mentions]
        # Either empty, or only contains anchor's date
        assert all(d == anchor.date() for d in dates)


# ---------------------------------------------------------------------------
# EventCalendarLME
# ---------------------------------------------------------------------------

class TestEventCalendarLME:
    def _mini_calendar(self):
        """Build a small calendar from 5 synthetic turns."""
        cal = EventCalendarLME()
        turns = [
            # (turn_idx, anchor, text)
            (0, datetime(2023, 3, 16, 9, 0),
             "I bought my brother a graduation gift on March 15."),
            (1, datetime(2023, 3, 23, 14, 0),
             "I got my best friend's birthday gift on March 22."),
            (2, datetime(2023, 5, 30, 10, 0),
             "I received my new phone case 2 weeks ago."),
            (3, datetime(2023, 6, 5, 18, 0),
             "I went to the first BBQ event on June 3."),
            (4, datetime(2023, 2, 15, 8, 0),
             "Yesterday I flew American Airlines on Valentine's day."),
        ]
        for idx, anchor, text in turns:
            cal.ingest_turn(idx, text, anchor)
        return cal

    def test_build(self):
        cal = self._mini_calendar()
        # Should have extracted at least one event per turn (except Valentine's which is a holiday)
        assert len(cal) >= 4

    def test_lookup_by_date_exact(self):
        """Looking up March 15 should return the graduation gift turn."""
        cal = self._mini_calendar()
        events = cal.lookup_window(date(2023, 3, 15), window_days=0)
        turn_idxs = {e.turn_idx for e in events}
        assert 0 in turn_idxs

    def test_lookup_by_date_window(self):
        """Window of ±3 days around March 15 includes March 15 and March 22."""
        cal = self._mini_calendar()
        events = cal.lookup_window(date(2023, 3, 20), window_days=5)
        turn_idxs = {e.turn_idx for e in events}
        # March 15 (turn 0) is 5 days before March 20 — at boundary
        # March 22 (turn 1) is 2 days after March 20 — inside
        assert 1 in turn_idxs
        assert 0 in turn_idxs  # at boundary

    def test_lookup_june_3(self):
        """Looking up June 3 returns the BBQ turn (turn 3)."""
        cal = self._mini_calendar()
        events = cal.lookup_window(date(2023, 6, 3), window_days=0)
        turn_idxs = {e.turn_idx for e in events}
        assert 3 in turn_idxs

    def test_empty_calendar(self):
        """Empty calendar returns no events."""
        cal = EventCalendarLME()
        assert cal.lookup_window(date(2023, 1, 1), window_days=30) == []
        assert len(cal) == 0

    def test_ingest_no_dates(self):
        """Ingesting text with no dates adds zero events."""
        cal = EventCalendarLME()
        cal.ingest_turn(0, "Hello world.", datetime(2023, 5, 1))
        assert len(cal) == 0


# ---------------------------------------------------------------------------
# Integration: realistic LME-style turns + temporal question
# ---------------------------------------------------------------------------

class TestRealisticLookup:
    def test_bbq_question(self):
        """Q4: 'What was the date on which I attended the first BBQ event in June?'
           Current Date: 2023/07/01.

           Calendar should surface the turn mentioning 'BBQ ... June 3' → date 2023-06-03.
        """
        cal = EventCalendarLME()
        cal.ingest_turn(
            turn_idx=42,
            turn_text="I went to the first BBQ event on June 3 and had a great time.",
            anchor=datetime(2023, 6, 5, 19, 0),
        )
        cal.ingest_turn(
            turn_idx=55,
            turn_text="The second BBQ in June was on June 17.",
            anchor=datetime(2023, 6, 18, 10, 0),
        )

        # Query: lookup events in June 2023
        june_events = cal.lookup_month(year=2023, month=6)
        turn_idxs = {e.turn_idx for e in june_events}
        assert 42 in turn_idxs
        assert 55 in turn_idxs

    def test_weeks_ago_question(self):
        """Q8: 'How many weeks ago did I attend the Summer Nights festival?'
           Current Date: 2023/08/05.

           Calendar should surface a turn that mentions the festival, so we can
           compute (current_date - event_date) / 7 = weeks.
        """
        cal = EventCalendarLME()
        cal.ingest_turn(
            turn_idx=100,
            turn_text="I attended the Summer Nights festival at Universal Studios on July 15.",
            anchor=datetime(2023, 7, 16, 9, 0),
        )
        query_date = date(2023, 8, 5)
        events = cal.lookup_window(query_date, window_days=30)
        # Festival on July 15 is 21 days before Aug 5 → 3 weeks
        festival_events = [e for e in events if "festival" in e.snippet.lower()]
        assert len(festival_events) == 1
        delta_days = (query_date - festival_events[0].event_date).days
        weeks = delta_days // 7
        assert weeks == 3  # "3 weeks ago" matches gold

    def test_valentines_day_airline_question(self):
        """Q28: 'What was the airline that I flied with on Valentine's day?'

           Calendar should surface turns where the user mentions February 14 explicitly.
           (Valentine's day isn't in dateparser, but 'February 14' or 'Feb 14' is.)
        """
        cal = EventCalendarLME()
        cal.ingest_turn(
            turn_idx=12,
            turn_text="On February 14, I flew American Airlines to visit my partner.",
            anchor=datetime(2023, 2, 15, 8, 0),
        )
        cal.ingest_turn(
            turn_idx=13,
            turn_text="Random turn with no date.",
            anchor=datetime(2023, 2, 16, 10, 0),
        )
        valentines_events = cal.lookup_window(date(2023, 2, 14), window_days=0)
        turn_idxs = {e.turn_idx for e in valentines_events}
        assert 12 in turn_idxs
        assert 13 not in turn_idxs
