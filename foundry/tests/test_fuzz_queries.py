"""Fuzz tests: random query chain generation.

Generates random sequences of query operations and verifies that no
combination causes a crash.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from datetime import datetime


type_labels = st.sampled_from([
    "cat", "dog", "pet", "restaurant", "city",
    "bike", "car", "fish", "company", "hotel",
    "person", "movie", "book", "song", "game",
])

operators = st.sampled_from(["$eq", "$ne", "$contains", "$startsWith"])
directions = st.sampled_from(["forward", "backward", "both"])
agg_ops = st.sampled_from(["count", "sum", "avg", "min", "max", "list", "distinct"])
predicates = st.sampled_from([
    "name", "age", "color", "breed", "cost", "location",
    "status", "type", "rating", "owner",
])


class TestFuzzQueryChains:
    """Fuzz: random query chains must never crash."""

    @given(type_labels)
    @settings(max_examples=100)
    def test_type_then_execute(self, t):
        """type(X).execute() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type(t).execute()
        assert isinstance(result, list)

    @given(type_labels)
    @settings(max_examples=100)
    def test_type_then_count(self, t):
        """type(X).count() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type(t).count()
        assert isinstance(result, int)
        assert result >= 0

    @given(predicates, operators, st.text(min_size=0, max_size=50))
    @settings(max_examples=100)
    def test_where_then_execute(self, pred, op, val):
        """where(pred, op, val).execute() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).where(pred, op, val).execute()
        assert isinstance(result, list)

    @given(type_labels, predicates, operators, st.text(min_size=0, max_size=50))
    @settings(max_examples=100)
    def test_type_where_execute(self, t, pred, op, val):
        """type(X).where(Y, op, V).execute() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type(t).where(pred, op, val).execute()
        assert isinstance(result, list)

    @given(type_labels, directions, st.integers(min_value=1, max_value=3))
    @settings(max_examples=100)
    def test_type_related_execute(self, t, direction, hops):
        """type(X).related(direction, hops).execute() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = (
            QueryPlan(store)
            .type(t)
            .related("", direction=direction, max_hops=hops)
            .execute()
        )
        assert isinstance(result, list)

    @given(type_labels, type_labels)
    @settings(max_examples=100)
    def test_type_related_type_execute(self, t1, t2):
        """type(A).related().type(B).execute() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = (
            QueryPlan(store)
            .type(t1)
            .related()
            .type(t2)
            .execute()
        )
        assert isinstance(result, list)

    @given(type_labels, agg_ops)
    @settings(max_examples=100)
    def test_type_aggregate(self, t, op):
        """type(X).aggregate(prop, op) never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type(t).aggregate("", op)
        # Result type depends on op
        assert result is not None or result == 0

    @given(type_labels)
    @settings(max_examples=100)
    def test_type_first(self, t):
        """type(X).first() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type(t).first()
        # Returns EntityNode or None
        assert result is None or hasattr(result, "canonical")

    @given(type_labels)
    @settings(max_examples=100)
    def test_type_facts(self, t):
        """type(X).facts() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type(t).facts()
        assert isinstance(result, list)

    @given(type_labels)
    @settings(max_examples=100)
    def test_type_evidence_turns(self, t):
        """type(X).evidence_turns() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type(t).evidence_turns()
        assert isinstance(result, list)

    @given(
        type_labels, type_labels, type_labels,
        predicates, operators,
        st.text(min_size=1, max_size=20),
    )
    @settings(max_examples=50)
    def test_deep_chain(self, t1, t2, t3, pred, op, val):
        """Deep chain: .type().where().related().type().where() never crashes."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = (
            QueryPlan(store)
            .type(t1)
            .where(pred, op, val)
            .related()
            .type(t2)
            .related()
            .type(t3)
            .execute()
        )
        assert isinstance(result, list)
