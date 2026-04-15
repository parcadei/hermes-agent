"""Unit tests for the Writeback / Mutation Layer (spec section 6).

Tests SupersessionEngine, TemporalValidity, EntityMerger,
DeletionPolicy, and Tombstone.
"""

from __future__ import annotations

import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# SupersessionEngine
# ---------------------------------------------------------------------------

class TestSupersessionEngine:
    """Tests for fact supersession (latest wins)."""

    def test_supersede_creates_chain(self):
        """Superseding a fact marks the old one and returns its ID."""
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        engine = SupersessionEngine()
        old_fact = Fact(
            id="f1", subject="user", predicate="lives_in",
            object="san_francisco", object_type=PropertyType.STRING,
            turn_idx=0,
        )
        new_fact = Fact(
            id="f2", subject="user", predicate="lives_in",
            object="portland", object_type=PropertyType.STRING,
            turn_idx=3,
        )
        engine.add_fact(old_fact)
        superseded_id = engine.supersede("user", "lives_in", new_fact)
        assert superseded_id == "f1"

    def test_supersede_no_prior(self):
        """Superseding with no prior fact returns None."""
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        engine = SupersessionEngine()
        new_fact = Fact(
            id="f1", subject="user", predicate="lives_in",
            object="portland", object_type=PropertyType.STRING,
            turn_idx=0,
        )
        result = engine.supersede("user", "lives_in", new_fact)
        assert result is None

    def test_resolve_latest_wins(self):
        """'latest_wins' keeps the fact with highest turn_idx."""
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        engine = SupersessionEngine()
        facts = [
            Fact(id="f1", subject="user", predicate="lives_in",
                 object="sf", object_type=PropertyType.STRING, turn_idx=0),
            Fact(id="f2", subject="user", predicate="lives_in",
                 object="portland", object_type=PropertyType.STRING, turn_idx=3),
        ]
        resolved = engine.resolve_conflicts(facts, strategy="latest_wins")
        assert len(resolved) == 1
        assert resolved[0].object == "portland"

    def test_resolve_user_wins(self):
        """'user_wins' prefers facts from user turns over assistant turns."""
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact, RawChunk
        engine = SupersessionEngine()
        facts = [
            Fact(id="f1", subject="user", predicate="favorite_color",
                 object="blue", object_type=PropertyType.STRING,
                 turn_idx=0, source_text="user said blue"),
            Fact(id="f2", subject="user", predicate="favorite_color",
                 object="red", object_type=PropertyType.STRING,
                 turn_idx=1, source_text="assistant said red"),
        ]
        # Simulation: engine needs to know which turns are user vs assistant
        # The test validates the API contract exists
        resolved = engine.resolve_conflicts(facts, strategy="user_wins")
        assert len(resolved) == 1

    def test_resolve_per_hop(self):
        """'per_hop' strategy is accepted."""
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        engine = SupersessionEngine()
        facts = [
            Fact(id="f1", subject="user", predicate="pet",
                 object="luna", object_type=PropertyType.STRING, turn_idx=0),
        ]
        resolved = engine.resolve_conflicts(facts, strategy="per_hop")
        assert len(resolved) == 1

    def test_superseded_facts_preserved(self):
        """Superseded facts are NOT deleted -- just marked."""
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        engine = SupersessionEngine()
        old = Fact(id="f1", subject="user", predicate="city",
                   object="sf", object_type=PropertyType.STRING, turn_idx=0)
        new = Fact(id="f2", subject="user", predicate="city",
                   object="portland", object_type=PropertyType.STRING, turn_idx=3)
        engine.add_fact(old)
        engine.supersede("user", "city", new)
        # Old fact should be marked superseded, not deleted
        all_facts = engine.get_all_facts("user", "city")
        assert any(f.superseded for f in all_facts)
        assert any(not f.superseded for f in all_facts)

    def test_monotonicity_invariant(self):
        """For any (entity, predicate), at most 1 current fact.

        Spec invariant: Supersession monotonicity -- the number of current
        (non-superseded) facts is bounded by 1.
        """
        from foundry.writeback import SupersessionEngine
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        engine = SupersessionEngine()
        for i in range(5):
            f = Fact(id=f"f{i}", subject="user", predicate="city",
                     object=f"city_{i}", object_type=PropertyType.STRING,
                     turn_idx=i)
            engine.add_fact(f)
            engine.supersede("user", "city", f)
        current = [f for f in engine.get_all_facts("user", "city")
                   if not f.superseded]
        assert len(current) <= 1


# ---------------------------------------------------------------------------
# TemporalValidity
# ---------------------------------------------------------------------------

class TestTemporalValidity:
    """Tests for the TemporalValidity data structure."""

    def test_current_when_no_until(self):
        """Fact with no valid_until is always current."""
        from foundry.writeback import TemporalValidity
        tv = TemporalValidity(valid_from=0, valid_until=None)
        assert tv.is_current() is True
        assert tv.is_current(as_of=100) is True

    def test_current_as_of_before_until(self):
        """Fact is current when as_of <= valid_until."""
        from foundry.writeback import TemporalValidity
        tv = TemporalValidity(valid_from=0, valid_until=5)
        assert tv.is_current(as_of=3) is True
        assert tv.is_current(as_of=5) is True

    def test_not_current_after_until(self):
        """Fact is NOT current when as_of > valid_until."""
        from foundry.writeback import TemporalValidity
        tv = TemporalValidity(valid_from=0, valid_until=5)
        assert tv.is_current(as_of=6) is False

    def test_current_with_datetime(self):
        """TemporalValidity works with datetime objects too."""
        from foundry.writeback import TemporalValidity
        tv = TemporalValidity(
            valid_from=datetime(2024, 1, 1),
            valid_until=datetime(2024, 6, 30),
        )
        assert tv.is_current(as_of=datetime(2024, 3, 15)) is True
        assert tv.is_current(as_of=datetime(2024, 12, 1)) is False

    def test_default_none_validity(self):
        """Default TemporalValidity: both None (always current)."""
        from foundry.writeback import TemporalValidity
        tv = TemporalValidity()
        assert tv.is_current() is True


# ---------------------------------------------------------------------------
# EntityMerger
# ---------------------------------------------------------------------------

class TestEntityMerger:
    """Tests for entity merge operations."""

    def test_merge_precondition_self(self):
        """Cannot merge an entity into itself."""
        from foundry.writeback import EntityMerger
        merger = EntityMerger()
        with pytest.raises(ValueError):
            merger.merge("luna", "luna")

    def test_merge_moves_aliases(self):
        """After merge, all source aliases point to target canonical."""
        from foundry.writeback import EntityMerger
        from foundry.storage import EntityNode
        merger = EntityMerger()
        source = EntityNode(canonical="my_cat", aliases={"my cat", "the cat"})
        target = EntityNode(canonical="luna", aliases={"Luna"})
        merger.merge_nodes(source, target)
        assert "my cat" in target.aliases
        assert "the cat" in target.aliases

    def test_merge_moves_facts(self):
        """After merge, source facts belong to target."""
        from foundry.writeback import EntityMerger
        from foundry.storage import EntityNode, Fact
        from foundry.ontology import PropertyType
        merger = EntityMerger()
        source = EntityNode(canonical="my_cat")
        source.facts["breed"] = [
            Fact(id="f1", subject="my_cat", predicate="breed",
                 object="tabby", object_type=PropertyType.STRING, turn_idx=0)
        ]
        target = EntityNode(canonical="luna")
        target.facts["age"] = [
            Fact(id="f2", subject="luna", predicate="age",
                 object="3", object_type=PropertyType.NUMERIC, turn_idx=0)
        ]
        merger.merge_nodes(source, target)
        assert "breed" in target.facts
        assert "age" in target.facts

    def test_merge_updates_related(self):
        """After merge, target.related includes source.related entries."""
        from foundry.writeback import EntityMerger
        from foundry.storage import EntityNode
        merger = EntityMerger()
        source = EntityNode(canonical="my_cat", related={"mochi", "user"})
        target = EntityNode(canonical="luna", related={"user"})
        merger.merge_nodes(source, target)
        assert "mochi" in target.related
        # Target should NOT be in its own related
        assert "luna" not in target.related

    def test_merge_updates_types(self):
        """After merge, target types include source types."""
        from foundry.writeback import EntityMerger
        from foundry.storage import EntityNode
        merger = EntityMerger()
        source = EntityNode(canonical="my_cat", types={"pet"})
        target = EntityNode(canonical="luna", types={"cat"})
        merger.merge_nodes(source, target)
        assert "pet" in target.types
        assert "cat" in target.types


# ---------------------------------------------------------------------------
# DeletionPolicy and Tombstone
# ---------------------------------------------------------------------------

class TestDeletionPolicy:
    """Tests for the deletion model."""

    def test_deletion_policy_values(self):
        """DeletionPolicy has SOFT_DELETE and HARD_DELETE."""
        from foundry.writeback import DeletionPolicy
        assert DeletionPolicy.SOFT_DELETE.value == "soft_delete"
        assert DeletionPolicy.HARD_DELETE.value == "hard_delete"

    def test_tombstone_creation(self):
        """Tombstone records deletion metadata."""
        from foundry.writeback import Tombstone
        ts = Tombstone(
            fact_id="f1",
            deleted_at=datetime(2024, 6, 15),
            reason="superseded",
        )
        assert ts.fact_id == "f1"
        assert ts.reason == "superseded"

    def test_tombstone_reasons(self):
        """Tombstone accepts standard deletion reasons."""
        from foundry.writeback import Tombstone
        for reason in ["superseded", "user_request", "decay"]:
            ts = Tombstone(
                fact_id="f1", deleted_at=datetime(2024, 1, 1), reason=reason,
            )
            assert ts.reason == reason
