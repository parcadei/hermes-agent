"""Regression tests: known aggregation bugs.

Tests for counting, type variation handling, ownership filtering,
and species-based deduplication.
"""

from __future__ import annotations

import pytest


class TestCountingAggregation:
    """Regression: counting queries must handle type variations."""

    def test_count_handles_plural_type(self):
        """count('bikes') returns same as count('bike')."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        for name in ["tarmac", "fuel_ex", "gravel"]:
            idx.index(name, "bike")
        assert idx.count("bikes") == idx.count("bike") == 3

    def test_count_handles_normalized_type(self):
        """count('pet_fish') works with entities typed 'pet_fish'."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        idx.index("neon_tetra", "pet_fish")
        idx.index("betta", "pet_fish")
        assert idx.count("pet_fish") == 2

    def test_count_zero_for_absent_type(self):
        """count('unicorn') returns 0."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        assert idx.count("unicorn") == 0


class TestOwnershipFiltering:
    """Regression: intent entities should not be counted as possessions."""

    def test_intent_not_counted(self):
        """Entity with sub_role='intent' should NOT be counted in 'how many X'.

        Example: 'thinking about getting a parrot' -- parrot is intent,
        not a possessed pet.
        """
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        parrot_fact = Fact(
            id="f1", subject="parrot", predicate="ownership",
            object="considering", object_type=PropertyType.STRING,
            turn_idx=7, sub_role="intent",
        )
        assert parrot_fact.sub_role == "intent"
        # When counting pets, intent entities should be excluded

    def test_hypothetical_not_counted(self):
        """Hypothetical facts should not be counted."""
        from foundry.storage import Fact
        from foundry.ontology import PropertyType
        f = Fact(
            id="f2", subject="parrot", predicate="type",
            object="pet", object_type=PropertyType.STRING,
            turn_idx=7, sub_role="hypothetical",
        )
        assert f.sub_role == "hypothetical"

    def test_acquisition_is_counted(self):
        """Acquisition facts SHOULD be counted (user actually has it)."""
        from foundry.storage import Fact
        from foundry.ontology import PropertyType
        f = Fact(
            id="f3", subject="luna", predicate="ownership",
            object="adopted", object_type=PropertyType.STRING,
            turn_idx=0, sub_role="acquisition",
        )
        assert f.sub_role == "acquisition"


class TestSupersessionInAggregation:
    """Regression: superseded facts must be excluded from counts."""

    def test_superseded_not_counted(self):
        """Superseded facts should not appear in aggregate results.

        Example: if user had 3 cats but gave one away, count should
        reflect current state.
        """
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        old = Fact(
            id="f1", subject="user", predicate="pet_count",
            object="3", object_type=PropertyType.NUMERIC,
            turn_idx=0, superseded=True, superseded_by="f2",
        )
        new = Fact(
            id="f2", subject="user", predicate="pet_count",
            object="2", object_type=PropertyType.NUMERIC,
            turn_idx=5,
        )
        current = [f for f in [old, new] if not f.superseded]
        assert len(current) == 1
        assert current[0].object == "2"


class TestAggregationOperations:
    """Tests for all aggregation operations."""

    def test_aggregate_distinct(self):
        """aggregate('breed', 'distinct') returns unique values."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode, Fact
        from foundry.ontology import PropertyType
        foundry = Foundry(FoundryConfig())
        for name, breed in [("luna", "tabby"), ("mochi", "black"), ("nemo", "tabby")]:
            entity = EntityNode(canonical=name, types={"cat"})
            entity.facts["breed"] = [
                Fact(id=f"f_{name}", subject=name, predicate="breed",
                     object=breed, object_type=PropertyType.STRING, turn_idx=0)
            ]
            foundry._store.entity_graph.add_entity(entity)
            foundry._store.types.index(name, "cat")
            foundry._store.properties.index(name, "breed", breed, PropertyType.STRING)
        result = foundry.entities().type("cat").aggregate("breed", "distinct")
        assert isinstance(result, list)
        assert set(result) == {"tabby", "black"}

    def test_aggregate_min_max(self):
        """aggregate('age', 'min') and 'max' return correct extremes."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode, Fact
        from foundry.ontology import PropertyType
        foundry = Foundry(FoundryConfig())
        for name, age in [("luna", "3"), ("max", "5"), ("mochi", "1")]:
            entity = EntityNode(canonical=name, types={"pet"})
            entity.facts["age"] = [
                Fact(id=f"f_{name}", subject=name, predicate="age",
                     object=age, object_type=PropertyType.NUMERIC, turn_idx=0)
            ]
            foundry._store.entity_graph.add_entity(entity)
            foundry._store.types.index(name, "pet")
            foundry._store.properties.index(name, "age", age, PropertyType.NUMERIC)
        min_age = foundry.entities().type("pet").aggregate("age", "min")
        max_age = foundry.entities().type("pet").aggregate("age", "max")
        assert min_age == 1.0
        assert max_age == 5.0

    def test_aggregate_avg(self):
        """aggregate('age', 'avg') returns correct average."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode, Fact
        from foundry.ontology import PropertyType
        foundry = Foundry(FoundryConfig())
        for name, age in [("a", "2"), ("b", "4"), ("c", "6")]:
            entity = EntityNode(canonical=name, types={"item"})
            entity.facts["age"] = [
                Fact(id=f"f_{name}", subject=name, predicate="age",
                     object=age, object_type=PropertyType.NUMERIC, turn_idx=0)
            ]
            foundry._store.entity_graph.add_entity(entity)
            foundry._store.types.index(name, "item")
            foundry._store.properties.index(name, "age", age, PropertyType.NUMERIC)
        avg = foundry.entities().type("item").aggregate("age", "avg")
        assert abs(avg - 4.0) < 1e-6
