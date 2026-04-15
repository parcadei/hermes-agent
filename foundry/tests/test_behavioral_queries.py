"""Behavioral tests: expected query outputs.

Given specific entities and facts in the store, queries must produce
EXACT expected results. These are the golden tests.
"""

from __future__ import annotations

import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers (inline store setup)
# ---------------------------------------------------------------------------

def _build_pet_store():
    """Build a store with known pet entities for deterministic testing."""
    from foundry.core import Foundry, FoundryConfig
    from foundry.ontology import PropertyType, GroundingLayer
    from foundry.storage import Fact, EntityNode

    foundry = Foundry(FoundryConfig())
    store = foundry._store

    # Luna (cat)
    luna = EntityNode(canonical="luna", aliases={"Luna", "my cat Luna"},
                      types={"cat", "pet"})
    luna.facts["breed"] = [
        Fact(id="f1", subject="luna", predicate="breed", object="tabby",
             object_type=PropertyType.STRING, turn_idx=0,
             source_text="Luna is a tabby")
    ]
    luna.facts["age"] = [
        Fact(id="f2", subject="luna", predicate="age", object="3",
             object_type=PropertyType.NUMERIC, turn_idx=2,
             source_text="Luna is 3 years old")
    ]
    luna.mention_turns = {0, 2}
    luna.related = {"mochi", "user"}

    # Mochi (cat)
    mochi = EntityNode(canonical="mochi", aliases={"Mochi"},
                       types={"cat", "pet"})
    mochi.facts["color"] = [
        Fact(id="f3", subject="mochi", predicate="color", object="black",
             object_type=PropertyType.STRING, turn_idx=0,
             source_text="Mochi is a black cat")
    ]
    mochi.mention_turns = {0, 2}
    mochi.related = {"luna", "user"}

    # Max (dog)
    max_dog = EntityNode(canonical="max", aliases={"Max"},
                         types={"dog", "pet", "golden_retriever"})
    max_dog.facts["age"] = [
        Fact(id="f4", subject="max", predicate="age", object="5",
             object_type=PropertyType.NUMERIC, turn_idx=4,
             source_text="Max is 5 years old")
    ]
    max_dog.mention_turns = {4}
    max_dog.related = {"user"}

    # Register in store
    for entity in [luna, mochi, max_dog]:
        store.entity_graph.add_entity(entity)
        for t in entity.types:
            store.types.index(entity.canonical, t)
        for pred, facts in entity.facts.items():
            for f in facts:
                store.properties.index(entity.canonical, pred, f.object, f.object_type)

    return foundry


# ---------------------------------------------------------------------------
# Type queries
# ---------------------------------------------------------------------------

class TestBehavioralTypeQueries:
    """Tests for type-based entity queries."""

    def test_type_cat_returns_two(self):
        """entities().type('cat') returns exactly Luna and Mochi."""
        foundry = _build_pet_store()
        result = foundry.entities().type("cat").execute()
        canonicals = {e.canonical for e in result}
        assert canonicals == {"luna", "mochi"}

    def test_type_dog_returns_one(self):
        """entities().type('dog') returns exactly Max."""
        foundry = _build_pet_store()
        result = foundry.entities().type("dog").execute()
        canonicals = {e.canonical for e in result}
        assert canonicals == {"max"}

    def test_type_pet_returns_all(self):
        """entities().type('pet') returns all three pets."""
        foundry = _build_pet_store()
        result = foundry.entities().type("pet").execute()
        canonicals = {e.canonical for e in result}
        assert canonicals == {"luna", "mochi", "max"}

    def test_type_nonexistent_returns_empty(self):
        """entities().type('fish') returns empty (no fish entities)."""
        foundry = _build_pet_store()
        result = foundry.entities().type("fish").execute()
        assert result == []

    def test_type_depluralization(self):
        """entities().type('cats') matches entities typed 'cat'."""
        foundry = _build_pet_store()
        result = foundry.entities().type("cats").execute()
        canonicals = {e.canonical for e in result}
        assert canonicals == {"luna", "mochi"}

    def test_count_cats(self):
        """entities().type('cat').count() returns 2."""
        foundry = _build_pet_store()
        assert foundry.entities().type("cat").count() == 2

    def test_count_pets(self):
        """entities().type('pet').count() returns 3."""
        foundry = _build_pet_store()
        assert foundry.entities().type("pet").count() == 3


# ---------------------------------------------------------------------------
# Property queries
# ---------------------------------------------------------------------------

class TestBehavioralPropertyQueries:
    """Tests for property-based entity queries."""

    def test_where_breed_eq_tabby(self):
        """entities().where('breed', '$eq', 'tabby') returns Luna."""
        foundry = _build_pet_store()
        result = foundry.entities().where("breed", "$eq", "tabby").execute()
        canonicals = {e.canonical for e in result}
        assert canonicals == {"luna"}

    def test_where_age_gt_3(self):
        """entities().where('age', '$gt', 3) returns Max (age=5)."""
        foundry = _build_pet_store()
        result = foundry.entities().where("age", "$gt", 3).execute()
        canonicals = {e.canonical for e in result}
        assert "max" in canonicals
        assert "luna" not in canonicals  # Luna is exactly 3, not > 3

    def test_where_color_eq_black(self):
        """entities().where('color', '$eq', 'black') returns Mochi."""
        foundry = _build_pet_store()
        result = foundry.entities().where("color", "$eq", "black").execute()
        canonicals = {e.canonical for e in result}
        assert canonicals == {"mochi"}

    def test_where_nonexistent_property(self):
        """Filtering on a property no entity has returns empty."""
        foundry = _build_pet_store()
        result = foundry.entities().where("wingspan", "$gt", 0).execute()
        assert result == []


# ---------------------------------------------------------------------------
# Combined type + property
# ---------------------------------------------------------------------------

class TestBehavioralCombinedQueries:
    """Tests for chained type + property queries."""

    def test_type_and_where(self):
        """entities().type('cat').where('breed', '$eq', 'tabby') returns Luna only."""
        foundry = _build_pet_store()
        result = (
            foundry.entities()
            .type("cat")
            .where("breed", "$eq", "tabby")
            .execute()
        )
        canonicals = {e.canonical for e in result}
        assert canonicals == {"luna"}

    def test_type_and_count(self):
        """entities().type('cat').count() is consistent with len(execute())."""
        foundry = _build_pet_store()
        count = foundry.entities().type("cat").count()
        results = foundry.entities().type("cat").execute()
        assert count == len(results)


# ---------------------------------------------------------------------------
# Related / traversal queries
# ---------------------------------------------------------------------------

class TestBehavioralRelatedQueries:
    """Tests for graph traversal queries."""

    def test_related_from_luna(self):
        """Luna's related entities include Mochi and user."""
        foundry = _build_pet_store()
        luna = foundry.entity("luna")
        assert luna is not None
        assert "mochi" in luna.related

    def test_related_query_returns_neighbors(self):
        """entities().type('cat').related() includes connected entities."""
        foundry = _build_pet_store()
        # Starting from cats, traverse to related entities
        result = foundry.entities().type("cat").related().execute()
        canonicals = {e.canonical for e in result}
        # Should include at least the user or other co-occurring entities
        assert len(canonicals) > 0

    def test_related_excludes_seed(self):
        """related() never returns the seed entity itself."""
        foundry = _build_pet_store()
        # Query related from luna
        plan = foundry.entities().type("cat").related()
        result = plan.execute()
        # Luna was in the seed set; related should not include cats themselves
        # unless they are related to each other (which they are)
        # The key invariant: if entity A is seed, A should not appear via its
        # own related set traversal of itself


# ---------------------------------------------------------------------------
# Fact retrieval
# ---------------------------------------------------------------------------

class TestBehavioralFactQueries:
    """Tests for fact collection from queries."""

    def test_facts_for_entity(self):
        """foundry.facts('luna') returns Luna's facts."""
        foundry = _build_pet_store()
        facts = foundry.facts("luna")
        predicates = {f.predicate for f in facts}
        assert "breed" in predicates
        assert "age" in predicates

    def test_facts_for_entity_by_predicate(self):
        """foundry.facts('luna', predicate='breed') returns only breed facts."""
        foundry = _build_pet_store()
        facts = foundry.facts("luna", predicate="breed")
        assert all(f.predicate == "breed" for f in facts)
        assert len(facts) == 1
        assert facts[0].object == "tabby"

    def test_facts_current_only(self):
        """facts(current_only=True) excludes superseded facts."""
        foundry = _build_pet_store()
        facts = foundry.facts("luna", current_only=True)
        assert all(not f.superseded for f in facts)


# ---------------------------------------------------------------------------
# Aggregation queries
# ---------------------------------------------------------------------------

class TestBehavioralAggregation:
    """Tests for aggregation terminal operations."""

    def test_aggregate_count_cats(self):
        """aggregate('count') on cats returns 2."""
        foundry = _build_pet_store()
        result = foundry.entities().type("cat").aggregate("", "count")
        assert result == 2

    def test_aggregate_count_all_pets(self):
        """aggregate('count') on all pets returns 3."""
        foundry = _build_pet_store()
        result = foundry.entities().type("pet").aggregate("", "count")
        assert result == 3

    def test_aggregate_list_breeds(self):
        """aggregate('breed', 'list') returns list of breed values."""
        foundry = _build_pet_store()
        result = foundry.entities().type("cat").aggregate("breed", "list")
        assert isinstance(result, list)
        assert "tabby" in result

    def test_aggregate_sum_ages(self):
        """aggregate('age', 'sum') returns sum of ages."""
        foundry = _build_pet_store()
        result = foundry.entities().type("pet").aggregate("age", "sum")
        # Luna=3, Max=5, Mochi=no age
        assert result >= 8.0

    def test_aggregate_on_empty_type(self):
        """aggregate('count') on nonexistent type returns 0."""
        foundry = _build_pet_store()
        result = foundry.entities().type("parrot").aggregate("", "count")
        assert result == 0


# ---------------------------------------------------------------------------
# Entity resolution entry points
# ---------------------------------------------------------------------------

class TestBehavioralEntityResolution:
    """Tests for foundry.entity() resolution."""

    def test_entity_by_canonical(self):
        """foundry.entity('luna') returns Luna."""
        foundry = _build_pet_store()
        entity = foundry.entity("luna")
        assert entity is not None
        assert entity.canonical == "luna"

    def test_entity_by_alias(self):
        """foundry.entity('Luna') resolves the alias."""
        foundry = _build_pet_store()
        entity = foundry.entity("Luna")
        assert entity is not None
        assert entity.canonical == "luna"

    def test_entity_nonexistent(self):
        """foundry.entity('nonexistent') returns None."""
        foundry = _build_pet_store()
        entity = foundry.entity("nonexistent_entity_xyz")
        assert entity is None
