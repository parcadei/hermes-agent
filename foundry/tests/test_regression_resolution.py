"""Regression tests: known entity resolution bugs.

Each test documents a specific bug that was discovered during development
and must never regress.
"""

from __future__ import annotations

import pytest


class TestPluralNormalization:
    """Regression: plural forms must match singular entity types."""

    def test_bikes_matches_bike(self):
        """'bikes' must match entities typed as 'bike'."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        idx.index("tarmac", "bike")
        idx.index("fuel_ex", "bike")
        result = idx.lookup("bikes")
        assert set(result) == {"tarmac", "fuel_ex"}

    def test_cats_matches_cat(self):
        """'cats' must match entities typed as 'cat'."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        idx.index("luna", "cat")
        idx.index("mochi", "cat")
        result = idx.lookup("cats")
        assert set(result) == {"luna", "mochi"}

    def test_hobbies_matches_hobby(self):
        """'hobbies' must match entities typed as 'hobby'."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        idx.index("cycling", "hobby")
        idx.index("photography", "hobby")
        result = idx.lookup("hobbies")
        assert set(result) == {"cycling", "photography"}

    def test_luxury_items_matches_luxury_item(self):
        """'luxury items' must match entities typed as 'luxury_item'."""
        from foundry.storage import TypeIndex
        from foundry.ontology import EntityType
        idx = TypeIndex()
        idx.index("rolex", "luxury_item")
        # The query "luxury items" should be normalized to "luxury_item"
        normalized = EntityType.normalize("luxury items")
        result = idx.lookup(normalized)
        assert "rolex" in result

    def test_cities_matches_city(self):
        """'cities' -> 'city' (ies -> y)."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        idx.index("rome", "city")
        result = idx.lookup("cities")
        assert "rome" in result


class TestSubstringFalseMatches:
    """Regression: substring resolution must not produce false merges."""

    def test_la_does_not_match_plane(self):
        """'la' must NOT match 'plane' via substring.

        This is a known regression from the resolve_exact fix.
        resolve_exact stops at alias lookup -- no substring.
        """
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("plane", aliases={"plane"})
        assert resolver.resolve_exact("la") is None

    def test_plan_does_not_match_internet_plan(self):
        """'internet plan' must NOT resolve to entity 'plan'.

        Substring matching must be type-gated and length-constrained.
        """
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("plan", aliases={"plan"})
        result = resolver.resolve_exact("internet plan")
        # resolve_exact should not match "plan" within "internet plan"
        # because "internet plan" is not an alias of "plan"
        assert result is None

    def test_car_does_not_match_cargo(self):
        """'car' must NOT match entity 'cargo' via substring."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("cargo", aliases={"cargo"})
        result = resolver.resolve_exact("car")
        assert result is None

    def test_art_does_not_match_heart(self):
        """'art' must NOT match entity 'heart'."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("heart", aliases={"heart"})
        result = resolver.resolve_exact("art")
        assert result is None


class TestEntityMergeRegression:
    """Regression: entity resolution should merge when appropriate."""

    def test_my_doctor_and_dr_smith(self):
        """'my doctor' and 'Dr. Smith' should be candidates for merge.

        This requires embedding similarity (threshold >= 0.85) in the
        full resolution cascade, not in resolve_exact.
        """
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("dr_smith", aliases={"Dr. Smith", "dr smith"})
        # "my doctor" should NOT resolve via exact match
        assert resolver.resolve_exact("my doctor") is None
        # But full resolve might match via embedding (if embed_fn set)

    def test_user_cat_and_luna(self):
        """\"user's cat\" should resolve to 'luna' if 'user cat' is an alias."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("luna", aliases={"Luna", "user cat", "my cat"})
        result = resolver.resolve("user's cat")
        assert result == "luna"


class TestDeterminerStripping:
    """Regression: determiners must be stripped during resolution."""

    def test_the_restaurant(self):
        """'the restaurant' resolves to 'restaurant'."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("restaurant", aliases={"restaurant"})
        assert resolver.resolve("the restaurant") == "restaurant"

    def test_a_cat(self):
        """'a cat' resolves to 'cat'."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("cat", aliases={"cat"})
        result = resolver.resolve("a cat")
        assert result == "cat"

    def test_an_apple(self):
        """'an apple' resolves to 'apple'."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        resolver.register("apple", aliases={"apple"})
        result = resolver.resolve("an apple")
        assert result == "apple"


class TestGroundingRegression:
    """Regression: grounding must reject model hallucinations."""

    def test_airline_not_in_fly_delta(self):
        """'airline' must FAIL grounding against 'I fly Delta'.

        The graph stores user's words, not model's interpretation.
        """
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        f = Fact(
            id="g1", subject="delta", predicate="type", object="airline",
            object_type=PropertyType.STRING, turn_idx=0,
            source_text="I fly Delta",
        )
        assert verify_grounding(f) is False

    def test_pet_store_not_in_adopted_from_shelter(self):
        """'pet store' must FAIL grounding against 'adopted from the shelter'."""
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        f = Fact(
            id="g2", subject="luna", predicate="source",
            object="pet store",
            object_type=PropertyType.STRING, turn_idx=0,
            source_text="I adopted Luna from the shelter",
        )
        assert verify_grounding(f) is False

    def test_grounded_fact_passes(self):
        """'tabby' IS in 'Luna is a tabby cat' -- grounding passes."""
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        f = Fact(
            id="g3", subject="luna", predicate="breed", object="tabby",
            object_type=PropertyType.STRING, turn_idx=0,
            source_text="Luna is a tabby cat",
        )
        assert verify_grounding(f) is True
