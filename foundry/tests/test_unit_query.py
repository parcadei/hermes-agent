"""Unit tests for the Query Language (spec section 3).

Tests QueryPlan creation, chaining operations, terminal operations,
set operations, FilterOp, LogicalOp, TraversalOp, and EntityResolver.
"""

from __future__ import annotations

import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# FilterOp
# ---------------------------------------------------------------------------

class TestFilterOp:
    """Tests for the FilterOp data structure."""

    def test_filterop_creation(self):
        """FilterOp can be created with required fields."""
        from foundry.query import FilterOp
        fop = FilterOp(field="type", operator="$eq", value="cat")
        assert fop.field == "type"
        assert fop.operator == "$eq"
        assert fop.value == "cat"
        assert fop.negate is False

    def test_filterop_negate(self):
        """FilterOp can be negated."""
        from foundry.query import FilterOp
        fop = FilterOp(field="type", operator="$eq", value="cat", negate=True)
        assert fop.negate is True

    def test_filterop_frozen(self):
        """FilterOp is frozen (immutable)."""
        from foundry.query import FilterOp
        fop = FilterOp(field="type", operator="$eq", value="cat")
        with pytest.raises(AttributeError):
            fop.field = "other"

    def test_filterop_all_operators(self):
        """All documented operators can be used."""
        from foundry.query import FilterOp
        operators = ["$eq", "$ne", "$lt", "$gt", "$lte", "$gte",
                     "$contains", "$startsWith", "$regex"]
        for op in operators:
            fop = FilterOp(field="test", operator=op, value="x")
            assert fop.operator == op


# ---------------------------------------------------------------------------
# LogicalOp
# ---------------------------------------------------------------------------

class TestLogicalOp:
    """Tests for the LogicalOp data structure."""

    def test_logical_and(self):
        """LogicalOp with $and combines multiple filters."""
        from foundry.query import FilterOp, LogicalOp
        f1 = FilterOp(field="type", operator="$eq", value="cat")
        f2 = FilterOp(field="age", operator="$gt", value=2)
        lop = LogicalOp(op="$and", children=(f1, f2))
        assert lop.op == "$and"
        assert len(lop.children) == 2

    def test_logical_or(self):
        """LogicalOp with $or creates disjunction."""
        from foundry.query import FilterOp, LogicalOp
        f1 = FilterOp(field="type", operator="$eq", value="cat")
        f2 = FilterOp(field="type", operator="$eq", value="dog")
        lop = LogicalOp(op="$or", children=(f1, f2))
        assert lop.op == "$or"

    def test_logical_not(self):
        """LogicalOp with $not wraps a single filter."""
        from foundry.query import FilterOp, LogicalOp
        f1 = FilterOp(field="type", operator="$eq", value="cat")
        lop = LogicalOp(op="$not", children=(f1,))
        assert lop.op == "$not"
        assert len(lop.children) == 1

    def test_nested_logical(self):
        """LogicalOps can be nested."""
        from foundry.query import FilterOp, LogicalOp
        f1 = FilterOp(field="type", operator="$eq", value="cat")
        f2 = FilterOp(field="age", operator="$gt", value=2)
        inner = LogicalOp(op="$and", children=(f1, f2))
        outer = LogicalOp(op="$not", children=(inner,))
        assert outer.op == "$not"
        assert isinstance(outer.children[0], LogicalOp)

    def test_logical_frozen(self):
        """LogicalOp is frozen (immutable)."""
        from foundry.query import FilterOp, LogicalOp
        f1 = FilterOp(field="type", operator="$eq", value="cat")
        lop = LogicalOp(op="$and", children=(f1,))
        with pytest.raises(AttributeError):
            lop.op = "$or"


# ---------------------------------------------------------------------------
# TraversalOp
# ---------------------------------------------------------------------------

class TestTraversalOp:
    """Tests for the TraversalOp data structure."""

    def test_traversal_defaults(self):
        """TraversalOp defaults: direction=both, max_hops=1, no filter."""
        from foundry.query import TraversalOp
        t = TraversalOp(link_predicate="owns", direction="forward")
        assert t.max_hops == 1
        assert t.filter_target is None

    def test_traversal_all_links(self):
        """Empty predicate traverses all link types."""
        from foundry.query import TraversalOp
        t = TraversalOp(link_predicate="", direction="both")
        assert t.link_predicate == ""

    def test_traversal_with_filter(self):
        """TraversalOp can have a filter on target entities."""
        from foundry.query import TraversalOp, FilterOp
        filt = FilterOp(field="type", operator="$eq", value="restaurant")
        t = TraversalOp(link_predicate="visited", direction="forward",
                        filter_target=filt)
        assert t.filter_target is not None
        assert t.filter_target.value == "restaurant"

    def test_traversal_max_hops_cap(self):
        """max_hops=3 is the proven useful maximum."""
        from foundry.query import TraversalOp
        t = TraversalOp(link_predicate="", direction="both", max_hops=3)
        assert t.max_hops == 3

    def test_traversal_frozen(self):
        """TraversalOp is frozen."""
        from foundry.query import TraversalOp
        t = TraversalOp(link_predicate="owns", direction="forward")
        with pytest.raises(AttributeError):
            t.max_hops = 5


# ---------------------------------------------------------------------------
# QueryPlan chaining
# ---------------------------------------------------------------------------

class TestQueryPlanChaining:
    """Tests for lazy, immutable query plan chaining."""

    def test_query_plan_type_returns_new_plan(self):
        """Each .type() call returns a NEW QueryPlan (immutable)."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan1 = QueryPlan(store)
        plan2 = plan1.type("cat")
        assert plan1 is not plan2

    def test_query_plan_where_returns_new_plan(self):
        """Each .where() call returns a new QueryPlan."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan1 = QueryPlan(store)
        plan2 = plan1.where("age", "$gt", 2)
        assert plan1 is not plan2

    def test_query_plan_related_returns_new_plan(self):
        """Each .related() call returns a new QueryPlan."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan1 = QueryPlan(store)
        plan2 = plan1.related("owns")
        assert plan1 is not plan2

    def test_chaining_multiple_operations(self):
        """Multiple operations can be chained."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = (
            QueryPlan(store)
            .type("restaurant")
            .where("where", "$eq", "Rome")
            .related("visited_by")
        )
        # Plan should exist and be a QueryPlan
        assert isinstance(plan, QueryPlan)

    def test_deep_chain(self):
        """.type().where().related().type().where() -- deep chains work."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = (
            QueryPlan(store)
            .type("person")
            .where("name", "$eq", "John")
            .related("works_at")
            .type("company")
            .where("location", "$eq", "NYC")
        )
        assert isinstance(plan, QueryPlan)

    def test_types_multi(self):
        """.types() accepts multiple type labels."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).types(["cat", "dog", "fish"])
        assert isinstance(plan, QueryPlan)


# ---------------------------------------------------------------------------
# QueryPlan temporal operations
# ---------------------------------------------------------------------------

class TestQueryPlanTemporal:
    """Tests for temporal filtering operations on QueryPlan."""

    def test_before_returns_new_plan(self):
        """.before() returns a new QueryPlan."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).before(datetime(2024, 6, 15))
        assert isinstance(plan, QueryPlan)

    def test_after_returns_new_plan(self):
        """.after() returns a new QueryPlan."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).after(datetime(2024, 1, 1))
        assert isinstance(plan, QueryPlan)

    def test_during_returns_new_plan(self):
        """.during() returns a new QueryPlan."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).during(
            datetime(2024, 6, 1), datetime(2024, 6, 30),
        )
        assert isinstance(plan, QueryPlan)

    def test_overlaps_returns_new_plan(self):
        """.overlaps() returns a new QueryPlan."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).overlaps(
            datetime(2024, 6, 1), datetime(2024, 6, 30),
        )
        assert isinstance(plan, QueryPlan)


# ---------------------------------------------------------------------------
# QueryPlan ordering and limit
# ---------------------------------------------------------------------------

class TestQueryPlanOrdering:
    """Tests for ordering and limiting."""

    def test_order_by(self):
        """.order_by() returns a new QueryPlan."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).order_by("name", "asc")
        assert isinstance(plan, QueryPlan)

    def test_limit(self):
        """.limit() returns a new QueryPlan."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).limit(10)
        assert isinstance(plan, QueryPlan)


# ---------------------------------------------------------------------------
# QueryPlan set operations
# ---------------------------------------------------------------------------

class TestQueryPlanSetOps:
    """Tests for set operations (union, intersection, difference)."""

    def test_union(self):
        """.union() combines two plans."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        p1 = QueryPlan(store).type("cat")
        p2 = QueryPlan(store).type("dog")
        combined = p1.union(p2)
        assert isinstance(combined, QueryPlan)

    def test_intersection(self):
        """.intersection() intersects two plans."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        p1 = QueryPlan(store).type("pet")
        p2 = QueryPlan(store).where("age", "$gt", 2)
        combined = p1.intersection(p2)
        assert isinstance(combined, QueryPlan)

    def test_difference(self):
        """.difference() removes entities in other plan."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        p1 = QueryPlan(store).type("pet")
        p2 = QueryPlan(store).type("fish")
        combined = p1.difference(p2)
        assert isinstance(combined, QueryPlan)


# ---------------------------------------------------------------------------
# QueryPlan terminal operations (on empty store)
# ---------------------------------------------------------------------------

class TestQueryPlanTerminals:
    """Tests for terminal operations on empty stores."""

    def test_execute_empty(self):
        """execute() on empty store returns empty list."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).execute()
        assert result == []

    def test_count_empty(self):
        """count() on empty store returns 0."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        assert QueryPlan(store).count() == 0

    def test_first_empty(self):
        """first() on empty store returns None."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        assert QueryPlan(store).first() is None

    def test_aggregate_count_empty(self):
        """aggregate('count') on empty store returns 0."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).aggregate("anything", "count")
        assert result == 0

    def test_aggregate_sum_empty(self):
        """aggregate('sum') on empty store returns 0.0."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).aggregate("cost", "sum")
        assert result == 0.0

    def test_facts_empty(self):
        """facts() on empty store returns empty list."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).facts()
        assert result == []

    def test_evidence_turns_empty(self):
        """evidence_turns() on empty store returns empty list."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        result = QueryPlan(store).evidence_turns()
        assert result == []


# ---------------------------------------------------------------------------
# EntityResolver
# ---------------------------------------------------------------------------

class TestEntityResolver:
    """Tests for the EntityResolver (spec section 3.4)."""

    def test_resolve_canonical_returns_self(self):
        """resolve(e.canonical) == e.canonical for all entities."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("luna", aliases={"Luna", "my cat Luna"})
        assert resolver.resolve("luna") == "luna"

    def test_resolve_alias(self):
        """resolve(alias) returns the canonical name."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("luna", aliases={"Luna", "my cat Luna"})
        assert resolver.resolve("Luna") == "luna"
        assert resolver.resolve("my cat Luna") == "luna"

    def test_resolve_exact_no_substring(self):
        """resolve_exact does NOT perform substring containment."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("plane", aliases={"plane"})
        # "la" should NOT match "plane" via resolve_exact
        assert resolver.resolve_exact("la") is None

    def test_resolve_determiner_strip(self):
        """Determiner stripped: 'the restaurant' -> 'restaurant'."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("restaurant", aliases={"restaurant"})
        assert resolver.resolve("the restaurant") == "restaurant"

    def test_resolve_possessive_strip(self):
        """Possessive stripped: \"user's cat\" tries 'user cat'."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("user_cat", aliases={"user cat"})
        result = resolver.resolve("user's cat")
        assert result == "user_cat"

    def test_resolve_none_on_unknown(self):
        """resolve returns None for unknown mentions."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        assert resolver.resolve("nonexistent_entity") is None

    def test_resolve_exact_returns_none_for_unknown(self):
        """resolve_exact returns None for unknown mentions."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        assert resolver.resolve_exact("anything") is None

    def test_resolve_or_create_always_returns(self):
        """resolve_or_create always returns a non-empty canonical."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create("new entity")
        assert result != ""
        assert isinstance(result, str)

    def test_resolve_or_create_idempotent(self):
        """Calling resolve_or_create twice for same mention returns same canonical."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        r1 = resolver.resolve_or_create("luna")
        r2 = resolver.resolve_or_create("luna")
        assert r1 == r2

    def test_resolve_typed(self):
        """resolve_typed uses type hint for disambiguation."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("mercury_planet", aliases={"mercury"}, type_hint="planet")
        resolver.register("mercury_element", aliases={"mercury"}, type_hint="element")
        result = resolver.resolve_typed("mercury", "planet")
        assert result == "mercury_planet"

    def test_normalize_idempotent(self):
        """normalize(x) is deterministic and idempotent."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        cases = ["Luna", "MY CAT", "the restaurant", " spaces "]
        for mention in cases:
            n1 = resolver.normalize(mention)
            n2 = resolver.normalize(n1)
            assert n1 == n2, f"Not idempotent: {mention!r} -> {n1!r} -> {n2!r}"
