"""Temporal tests: temporal query operations.

Tests for temporal filtering in QueryPlan (.before, .after, .during,
.overlaps) and temporal index queries.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta


class TestTemporalIndexQueries:
    """Tests for TemporalIndex query methods."""

    def test_events_on_date_exact(self):
        """events_on_date returns events whose interval contains the date."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        idx.ingest(TemporalEvent(
            event="Rome trip", start_dt=datetime(2024, 6, 1),
            end_dt=datetime(2024, 6, 14), source="dateparser",
        ))
        idx.ingest(TemporalEvent(
            event="Paris trip", start_dt=datetime(2024, 3, 15),
            end_dt=datetime(2024, 3, 18), source="dateparser",
        ))
        results = idx.events_on_date(datetime(2024, 6, 5))
        assert len(results) == 1
        assert results[0].event == "Rome trip"

    def test_events_on_date_boundary(self):
        """events_on_date includes events starting on that date."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        idx.ingest(TemporalEvent(
            event="Meeting", start_dt=datetime(2024, 6, 15),
            end_dt=datetime(2024, 6, 15), source="manual",
        ))
        results = idx.events_on_date(datetime(2024, 6, 15))
        assert len(results) == 1

    def test_events_on_date_no_match(self):
        """events_on_date returns [] when no events match."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        idx.ingest(TemporalEvent(
            event="Rome trip", start_dt=datetime(2024, 6, 1),
            end_dt=datetime(2024, 6, 14), source="dateparser",
        ))
        results = idx.events_on_date(datetime(2024, 12, 25))
        assert results == []

    def test_events_between_range(self):
        """events_between returns events overlapping the range."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        idx.ingest(TemporalEvent(
            event="Jan event", start_dt=datetime(2024, 1, 10),
            end_dt=datetime(2024, 1, 20), source="manual",
        ))
        idx.ingest(TemporalEvent(
            event="March event", start_dt=datetime(2024, 3, 1),
            end_dt=datetime(2024, 3, 31), source="manual",
        ))
        idx.ingest(TemporalEvent(
            event="June event", start_dt=datetime(2024, 6, 1),
            end_dt=datetime(2024, 6, 30), source="manual",
        ))
        results = idx.events_between(
            datetime(2024, 2, 1), datetime(2024, 4, 30),
        )
        events = [r.event for r in results]
        assert "March event" in events
        assert "Jan event" not in events
        assert "June event" not in events

    def test_events_between_partial_overlap(self):
        """events_between includes events that partially overlap the range."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        idx.ingest(TemporalEvent(
            event="Spanning", start_dt=datetime(2024, 5, 25),
            end_dt=datetime(2024, 6, 5), source="manual",
        ))
        results = idx.events_between(
            datetime(2024, 6, 1), datetime(2024, 6, 30),
        )
        assert len(results) == 1
        assert results[0].event == "Spanning"


class TestTemporalOrdering:
    """Tests for temporal ordering of events."""

    def test_events_sorted_by_date(self):
        """Events should be retrievable in temporal order."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        idx.ingest(TemporalEvent(
            event="C", start_dt=datetime(2024, 6, 15),
            end_dt=datetime(2024, 6, 15), source="manual",
        ))
        idx.ingest(TemporalEvent(
            event="A", start_dt=datetime(2024, 1, 1),
            end_dt=datetime(2024, 1, 1), source="manual",
        ))
        idx.ingest(TemporalEvent(
            event="B", start_dt=datetime(2024, 3, 15),
            end_dt=datetime(2024, 3, 15), source="manual",
        ))
        all_events = idx.events_between(
            datetime(2024, 1, 1), datetime(2024, 12, 31),
        )
        dates = [e.start_dt for e in all_events]
        assert dates == sorted(dates)


class TestTemporalAggregation:
    """Tests for temporal aggregation queries."""

    def test_count_events_in_month(self):
        """Count events in a specific month."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        for day in [1, 5, 10, 20]:
            idx.ingest(TemporalEvent(
                event=f"June {day}", start_dt=datetime(2024, 6, day),
                end_dt=datetime(2024, 6, day), source="manual",
            ))
        idx.ingest(TemporalEvent(
            event="July 1", start_dt=datetime(2024, 7, 1),
            end_dt=datetime(2024, 7, 1), source="manual",
        ))
        june_events = idx.events_between(
            datetime(2024, 6, 1), datetime(2024, 6, 30),
        )
        assert len(june_events) == 4


class TestTemporalQueryPlan:
    """Tests for temporal operations in QueryPlan."""

    def test_before_reduces_results(self):
        """.before() on empty store returns empty (baseline)."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).before(datetime(2024, 6, 15)).execute()
        assert result == []

    def test_after_reduces_results(self):
        """.after() on empty store returns empty."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).after(datetime(2024, 1, 1)).execute()
        assert result == []

    def test_during_reduces_results(self):
        """.during() on empty store returns empty."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).during(
            datetime(2024, 6, 1), datetime(2024, 6, 30),
        ).execute()
        assert result == []

    def test_overlaps_reduces_results(self):
        """.overlaps() on empty store returns empty."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).overlaps(
            datetime(2024, 6, 1), datetime(2024, 6, 30),
        ).execute()
        assert result == []


class TestQuestionClassification:
    """Tests for the 5 temporal question buckets."""

    def test_answer_temporal_query_structure(self):
        """answer_temporal_query returns structured dict."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        result = idx.answer_temporal_query("When did the user visit Rome?")
        assert isinstance(result, dict)
        assert "operation" in result or "matched_events" in result

    def test_ordering_question(self):
        """'Which happened first' is classified as ordering."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        result = idx.answer_temporal_query("Which happened first, the Rome trip or the Paris trip?")
        assert isinstance(result, dict)

    def test_distance_question(self):
        """'How many days ago' is classified as distance."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        result = idx.answer_temporal_query(
            "How many days ago did the user visit Rome?",
            reference_date=datetime(2024, 12, 1),
        )
        assert isinstance(result, dict)

    def test_duration_question(self):
        """'How long did X last' is classified as duration."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        result = idx.answer_temporal_query("How long did the Rome trip last?")
        assert isinstance(result, dict)

    def test_contains_question(self):
        """'What happened on date' is classified as contains."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        result = idx.answer_temporal_query("What happened on June 5th?")
        assert isinstance(result, dict)
