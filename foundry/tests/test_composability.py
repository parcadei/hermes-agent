"""Composability tests: operator chaining.

Every query operator must chain with every other operator. Deep chains,
mixed operations, and terminal operations after any prefix must all work.
"""

from __future__ import annotations

import pytest
from datetime import datetime


class TestOperatorChaining:
    """Tests that every operator chains with every other operator."""

    def test_type_then_where(self):
        """.type().where() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type("cat").where("age", "$gt", 1).execute()
        assert isinstance(result, list)

    def test_type_then_related(self):
        """.type().related() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type("cat").related().execute()
        assert isinstance(result, list)

    def test_where_then_type(self):
        """.where().type() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).where("age", "$gt", 1).type("pet").execute()
        assert isinstance(result, list)

    def test_where_then_related(self):
        """.where().related() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).where("age", "$gt", 1).related().execute()
        assert isinstance(result, list)

    def test_related_then_type(self):
        """.related().type() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).related().type("cat").execute()
        assert isinstance(result, list)

    def test_related_then_where(self):
        """.related().where() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).related().where("age", "$gt", 1).execute()
        assert isinstance(result, list)

    def test_related_then_related(self):
        """.related().related() (multi-hop) chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).related().related().execute()
        assert isinstance(result, list)

    def test_type_where_related_type_where(self):
        """Deep chain: .type().where().related().type().where() works."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = (
            QueryPlan(store)
            .type("person")
            .where("name", "$eq", "John")
            .related()
            .type("company")
            .where("location", "$eq", "NYC")
            .execute()
        )
        assert isinstance(result, list)

    def test_before_then_type(self):
        """.before().type() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = (
            QueryPlan(store)
            .before(datetime(2024, 6, 15))
            .type("event")
            .execute()
        )
        assert isinstance(result, list)

    def test_after_then_where(self):
        """.after().where() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = (
            QueryPlan(store)
            .after(datetime(2024, 1, 1))
            .where("status", "$eq", "active")
            .execute()
        )
        assert isinstance(result, list)

    def test_during_then_related(self):
        """.during().related() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = (
            QueryPlan(store)
            .during(datetime(2024, 1, 1), datetime(2024, 12, 31))
            .related()
            .execute()
        )
        assert isinstance(result, list)

    def test_order_by_then_limit(self):
        """.order_by().limit() chains."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = (
            QueryPlan(store)
            .type("item")
            .order_by("name", "asc")
            .limit(5)
            .execute()
        )
        assert isinstance(result, list)


class TestTerminalAfterAnyPrefix:
    """Every terminal operation works after any chain prefix."""

    def test_count_after_type(self):
        """.type().count() works."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        assert isinstance(QueryPlan(store).type("cat").count(), int)

    def test_count_after_where(self):
        """.where().count() works."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        assert isinstance(QueryPlan(store).where("x", "$eq", "y").count(), int)

    def test_count_after_related(self):
        """.related().count() works."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        assert isinstance(QueryPlan(store).related().count(), int)

    def test_first_after_type(self):
        """.type().first() works."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type("cat").first()
        assert result is None  # Empty store

    def test_aggregate_after_type(self):
        """.type().aggregate() works."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type("cat").aggregate("", "count")
        assert result == 0

    def test_facts_after_where(self):
        """.where().facts() works."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).where("x", "$eq", "y").facts()
        assert isinstance(result, list)

    def test_evidence_turns_after_type(self):
        """.type().evidence_turns() works."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).type("cat").evidence_turns()
        assert isinstance(result, list)


class TestEmptyGraphQueries:
    """All queries on empty graph return empty, not errors."""

    def test_type_on_empty(self):
        """type() on empty returns []."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        assert QueryPlan(FoundryStore()).type("cat").execute() == []

    def test_where_on_empty(self):
        """where() on empty returns []."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        assert QueryPlan(FoundryStore()).where("x", "$eq", "y").execute() == []

    def test_related_on_empty(self):
        """related() on empty returns []."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        assert QueryPlan(FoundryStore()).related().execute() == []

    def test_union_on_empty(self):
        """union() of two empty queries returns []."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        p1 = QueryPlan(store).type("cat")
        p2 = QueryPlan(store).type("dog")
        assert p1.union(p2).execute() == []

    def test_no_matching_filters(self):
        """Filters with no matches return [], not error."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = (
            QueryPlan(store)
            .type("nonexistent_type_xyz")
            .where("nonexistent_prop", "$eq", "nothing")
            .execute()
        )
        assert result == []
