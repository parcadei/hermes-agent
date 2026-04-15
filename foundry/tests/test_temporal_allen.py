"""Temporal tests: Allen interval algebra.

Tests all 13 Allen interval relations plus convenience methods.
Each relation is tested with concrete datetime intervals.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta


def _event(name: str, start: datetime, end: datetime):
    """Helper to create a TemporalEvent."""
    from foundry.storage import TemporalEvent
    return TemporalEvent(event=name, start_dt=start, end_dt=end, source="manual")


# Fixed dates for all tests
JAN_1 = datetime(2024, 1, 1)
JAN_5 = datetime(2024, 1, 5)
JAN_10 = datetime(2024, 1, 10)
JAN_15 = datetime(2024, 1, 15)
JAN_20 = datetime(2024, 1, 20)
JAN_25 = datetime(2024, 1, 25)
JAN_31 = datetime(2024, 1, 31)


class TestAllenBefore:
    """Tests for allen_before: A ends before B starts (no overlap)."""

    def test_before_true(self):
        """[Jan 1-5] before [Jan 10-15] is True."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_5)
        b = _event("B", JAN_10, JAN_15)
        assert idx.allen_before(a, b) is True

    def test_before_false_overlap(self):
        """[Jan 1-10] before [Jan 5-15] is False (they overlap)."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_10)
        b = _event("B", JAN_5, JAN_15)
        assert idx.allen_before(a, b) is False

    def test_before_false_after(self):
        """[Jan 10-15] before [Jan 1-5] is False (A is after B)."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_10, JAN_15)
        b = _event("B", JAN_1, JAN_5)
        assert idx.allen_before(a, b) is False


class TestAllenAfter:
    """Tests for allen_after: B ends before A starts."""

    def test_after_true(self):
        """[Jan 10-15] after [Jan 1-5] is True."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_10, JAN_15)
        b = _event("B", JAN_1, JAN_5)
        assert idx.allen_after(a, b) is True

    def test_after_false(self):
        """[Jan 1-5] after [Jan 10-15] is False."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_5)
        b = _event("B", JAN_10, JAN_15)
        assert idx.allen_after(a, b) is False


class TestAllenMeets:
    """Tests for allen_meets: A.end == B.start (touching, no gap)."""

    def test_meets_true(self):
        """[Jan 1-5] meets [Jan 5-10] (end == start)."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_5)
        b = _event("B", JAN_5, JAN_10)
        assert idx.allen_meets(a, b) is True

    def test_meets_false_gap(self):
        """[Jan 1-5] does NOT meet [Jan 10-15] (gap between them)."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_5)
        b = _event("B", JAN_10, JAN_15)
        assert idx.allen_meets(a, b) is False


class TestAllenMetBy:
    """Tests for allen_met_by: inverse of meets."""

    def test_met_by_true(self):
        """[Jan 5-10] met_by [Jan 1-5]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_5, JAN_10)
        b = _event("B", JAN_1, JAN_5)
        assert idx.allen_met_by(a, b) is True


class TestAllenOverlaps:
    """Tests for allen_overlaps: A starts before B and ends during B."""

    def test_overlaps_true(self):
        """[Jan 1-10] overlaps [Jan 5-15]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_10)
        b = _event("B", JAN_5, JAN_15)
        assert idx.allen_overlaps(a, b) is True

    def test_overlaps_false_contained(self):
        """[Jan 1-20] does NOT overlap [Jan 5-10] (contains instead)."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_20)
        b = _event("B", JAN_5, JAN_10)
        assert idx.allen_overlaps(a, b) is False


class TestAllenOverlappedBy:
    """Tests for allen_overlapped_by: inverse of overlaps."""

    def test_overlapped_by_true(self):
        """[Jan 5-15] overlapped_by [Jan 1-10]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_5, JAN_15)
        b = _event("B", JAN_1, JAN_10)
        assert idx.allen_overlapped_by(a, b) is True


class TestAllenStarts:
    """Tests for allen_starts: A.start == B.start and A.end < B.end."""

    def test_starts_true(self):
        """[Jan 1-5] starts [Jan 1-15]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_5)
        b = _event("B", JAN_1, JAN_15)
        assert idx.allen_starts(a, b) is True

    def test_starts_false_different_start(self):
        """[Jan 5-10] does NOT start [Jan 1-15]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_5, JAN_10)
        b = _event("B", JAN_1, JAN_15)
        assert idx.allen_starts(a, b) is False


class TestAllenStartedBy:
    """Tests for allen_started_by: inverse of starts."""

    def test_started_by_true(self):
        """[Jan 1-15] started_by [Jan 1-5]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_15)
        b = _event("B", JAN_1, JAN_5)
        assert idx.allen_started_by(a, b) is True


class TestAllenDuring:
    """Tests for allen_during: A is strictly contained within B."""

    def test_during_true(self):
        """[Jan 5-10] during [Jan 1-15]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_5, JAN_10)
        b = _event("B", JAN_1, JAN_15)
        assert idx.allen_during(a, b) is True

    def test_during_false_equal(self):
        """[Jan 1-15] is NOT during [Jan 1-15] (equal, not during)."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_15)
        b = _event("B", JAN_1, JAN_15)
        assert idx.allen_during(a, b) is False


class TestAllenContains:
    """Tests for allen_contains_interval: B is strictly contained within A."""

    def test_contains_true(self):
        """[Jan 1-15] contains [Jan 5-10]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_15)
        b = _event("B", JAN_5, JAN_10)
        assert idx.allen_contains_interval(a, b) is True


class TestAllenFinishes:
    """Tests for allen_finishes: A.end == B.end and A.start > B.start."""

    def test_finishes_true(self):
        """[Jan 10-15] finishes [Jan 1-15]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_10, JAN_15)
        b = _event("B", JAN_1, JAN_15)
        assert idx.allen_finishes(a, b) is True


class TestAllenFinishedBy:
    """Tests for allen_finished_by: inverse of finishes."""

    def test_finished_by_true(self):
        """[Jan 1-15] finished_by [Jan 10-15]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_15)
        b = _event("B", JAN_10, JAN_15)
        assert idx.allen_finished_by(a, b) is True


class TestAllenEquals:
    """Tests for allen_equals: identical intervals."""

    def test_equals_true(self):
        """[Jan 1-15] equals [Jan 1-15]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_15)
        b = _event("B", JAN_1, JAN_15)
        assert idx.allen_equals(a, b) is True

    def test_equals_false(self):
        """[Jan 1-10] does NOT equal [Jan 1-15]."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_10)
        b = _event("B", JAN_1, JAN_15)
        assert idx.allen_equals(a, b) is False


class TestAllenRelation:
    """Tests for allen_relation: returns the name of the relation."""

    def test_relation_before(self):
        """Relation between non-overlapping intervals is 'before'."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_5)
        b = _event("B", JAN_10, JAN_15)
        assert idx.allen_relation(a, b) == "before"

    def test_relation_equals(self):
        """Relation between identical intervals is 'equals'."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_15)
        b = _event("B", JAN_1, JAN_15)
        assert idx.allen_relation(a, b) == "equals"

    def test_exactly_one_relation(self):
        """For any two events, exactly one Allen relation holds (I6)."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_10)
        b = _event("B", JAN_5, JAN_15)
        relation = idx.allen_relation(a, b)
        all_relations = [
            "before", "after", "meets", "met_by",
            "overlaps", "overlapped_by",
            "starts", "started_by",
            "during", "contains",
            "finishes", "finished_by",
            "equals",
        ]
        assert relation in all_relations

    def test_all_13_relations_covered(self):
        """We can construct pairs that produce each of the 13 relations."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        cases = {
            "before":        (_event("", JAN_1, JAN_5), _event("", JAN_10, JAN_15)),
            "after":         (_event("", JAN_10, JAN_15), _event("", JAN_1, JAN_5)),
            "meets":         (_event("", JAN_1, JAN_5), _event("", JAN_5, JAN_10)),
            "met_by":        (_event("", JAN_5, JAN_10), _event("", JAN_1, JAN_5)),
            "overlaps":      (_event("", JAN_1, JAN_10), _event("", JAN_5, JAN_15)),
            "overlapped_by": (_event("", JAN_5, JAN_15), _event("", JAN_1, JAN_10)),
            "starts":        (_event("", JAN_1, JAN_5), _event("", JAN_1, JAN_15)),
            "started_by":    (_event("", JAN_1, JAN_15), _event("", JAN_1, JAN_5)),
            "during":        (_event("", JAN_5, JAN_10), _event("", JAN_1, JAN_15)),
            "contains":      (_event("", JAN_1, JAN_15), _event("", JAN_5, JAN_10)),
            "finishes":      (_event("", JAN_10, JAN_15), _event("", JAN_1, JAN_15)),
            "finished_by":   (_event("", JAN_1, JAN_15), _event("", JAN_10, JAN_15)),
            "equals":        (_event("", JAN_1, JAN_15), _event("", JAN_1, JAN_15)),
        }
        for expected_rel, (a, b) in cases.items():
            actual = idx.allen_relation(a, b)
            assert actual == expected_rel, (
                f"Expected '{expected_rel}' for "
                f"({a.start_dt}-{a.end_dt}) vs ({b.start_dt}-{b.end_dt}), "
                f"got '{actual}'"
            )


class TestConvenienceMethods:
    """Tests for convenience temporal methods."""

    def test_before_convenience(self):
        """before(a, b) matches allen_before semantics."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_5)
        b = _event("B", JAN_10, JAN_15)
        assert idx.before(a, b) is True
        assert idx.before(b, a) is False

    def test_distance(self):
        """distance(event, reference) returns signed timedelta."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        event = _event("A", JAN_10, JAN_15)
        ref = JAN_1
        delta = idx.distance(event, ref)
        assert isinstance(delta, timedelta)
        assert delta.days > 0  # Event is after reference

    def test_duration_between(self):
        """duration_between(a, b) returns gap duration."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        a = _event("A", JAN_1, JAN_5)
        b = _event("B", JAN_10, JAN_15)
        gap = idx.duration_between(a, b)
        assert isinstance(gap, timedelta)
        assert gap.days == 5  # Jan 5 to Jan 10

    def test_duration_of(self):
        """duration_of(event) returns event span."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        event = _event("A", JAN_1, JAN_15)
        span = idx.duration_of(event)
        assert isinstance(span, timedelta)
        assert span.days == 14

    def test_contains_point(self):
        """contains(event, point) returns True if point is in interval."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        event = _event("A", JAN_1, JAN_15)
        assert idx.contains(event, JAN_5) is True
        assert idx.contains(event, JAN_20) is False
        assert idx.contains(event, JAN_1) is True   # Inclusive start
        assert idx.contains(event, JAN_15) is True   # Inclusive end
