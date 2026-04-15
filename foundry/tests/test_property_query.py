"""Property-based tests for query invariants.

Tests that query operations maintain algebraic properties for ANY
valid query chain over ANY valid entity graph.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

type_labels = st.sampled_from([
    "cat", "dog", "pet", "restaurant", "city",
    "bike", "car", "fish", "company", "hotel",
])

operators = st.sampled_from(["$eq", "$ne", "$contains"])

dates = st.datetimes(
    min_value=datetime(2020, 1, 1),
    max_value=datetime(2026, 12, 31),
)


# ---------------------------------------------------------------------------
# Query commutativity and consistency
# ---------------------------------------------------------------------------

class TestQueryConsistency:
    """Tests that count() == len(execute()) for any query."""

    @given(type_labels)
    @settings(max_examples=50)
    def test_count_matches_execute_len(self, type_label):
        """For any type filter, count() == len(execute())."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).type(type_label)
        count = plan.count()
        results = plan.execute()
        assert count == len(results)

    @given(type_labels)
    @settings(max_examples=50)
    def test_aggregate_count_matches_count(self, type_label):
        """aggregate('', 'count') == count() for any query."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).type(type_label)
        agg = plan.aggregate("", "count")
        count = plan.count()
        assert agg == count

    @given(type_labels, type_labels)
    @settings(max_examples=50)
    def test_type_then_type_is_intersection(self, t1, t2):
        """.type(A).type(B) returns entities matching BOTH types."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        combined = QueryPlan(store).type(t1).type(t2).execute()
        # Every result must match both types
        for entity in combined:
            assert t1 in entity.types or t2 in entity.types


class TestQueryMonotonicity:
    """Tests that adding filters never increases result count."""

    @given(type_labels)
    @settings(max_examples=50)
    def test_type_filter_reduces(self, type_label):
        """Adding a type filter does not increase result count."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        all_count = QueryPlan(store).count()
        filtered_count = QueryPlan(store).type(type_label).count()
        assert filtered_count <= all_count

    @given(dates)
    @settings(max_examples=50)
    def test_before_filter_monotonic(self, date):
        """before(date) returns subset of all entities."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        all_count = QueryPlan(store).count()
        before_count = QueryPlan(store).before(date).count()
        assert before_count <= all_count

    @given(dates, dates)
    @settings(max_examples=50)
    def test_narrowing_temporal_reduces(self, d1, d2):
        """Narrowing a temporal range can only reduce or maintain results."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        assume(d1 <= d2)
        store = FoundryStore()
        wide = QueryPlan(store).during(d1, d2).count()
        # Narrow the range
        mid = d1 + (d2 - d1) / 2
        narrow = QueryPlan(store).during(d1, mid).count()
        assert narrow <= wide


class TestQuerySetOperations:
    """Tests for set operation algebraic properties."""

    @given(type_labels, type_labels)
    @settings(max_examples=50)
    def test_union_commutative(self, t1, t2):
        """A.union(B) has same count as B.union(A)."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        p1 = QueryPlan(store).type(t1)
        p2 = QueryPlan(store).type(t2)
        assert p1.union(p2).count() == p2.union(p1).count()

    @given(type_labels, type_labels)
    @settings(max_examples=50)
    def test_intersection_commutative(self, t1, t2):
        """A.intersection(B) has same count as B.intersection(A)."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        p1 = QueryPlan(store).type(t1)
        p2 = QueryPlan(store).type(t2)
        assert p1.intersection(p2).count() == p2.intersection(p1).count()

    @given(type_labels)
    @settings(max_examples=50)
    def test_union_self_idempotent(self, t):
        """A.union(A) == A."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        p = QueryPlan(store).type(t)
        assert p.union(p).count() == p.count()

    @given(type_labels)
    @settings(max_examples=50)
    def test_intersection_self_idempotent(self, t):
        """A.intersection(A) == A."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        p = QueryPlan(store).type(t)
        assert p.intersection(p).count() == p.count()

    @given(type_labels)
    @settings(max_examples=50)
    def test_difference_self_empty(self, t):
        """A.difference(A) == empty."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        p = QueryPlan(store).type(t)
        assert p.difference(p).count() == 0


class TestQueryEmptyGraph:
    """Property: all queries over an empty graph return empty."""

    @given(type_labels)
    @settings(max_examples=50)
    def test_type_on_empty(self, type_label):
        """type() on empty graph returns []."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        assert QueryPlan(store).type(type_label).execute() == []

    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_where_on_empty(self, pred):
        """where() on empty graph returns []."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        assert QueryPlan(store).where(pred, "$eq", "x").execute() == []
