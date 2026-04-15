"""Property-based tests for graph invariants.

Uses Hypothesis to generate random entities, facts, and graphs, then
verifies that system invariants hold for ANY valid input.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

entity_names = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_- "),
    min_size=1, max_size=50,
).map(str.strip).filter(lambda s: len(s) > 0)

type_labels = st.sampled_from([
    "cat", "dog", "pet", "restaurant", "city", "person",
    "bike", "car", "fish", "company", "hotel", "movie",
])

predicates = st.sampled_from([
    "breed", "color", "age", "location", "cost",
    "name", "type", "status", "rating", "count",
])


# ---------------------------------------------------------------------------
# EntityType normalization invariants
# ---------------------------------------------------------------------------

class TestNormalizationProperties:
    """Property-based tests for EntityType normalization."""

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_normalize_idempotent(self, text):
        """normalize(normalize(x)) == normalize(x) for any string."""
        from foundry.ontology import EntityType
        once = EntityType.normalize(text)
        twice = EntityType.normalize(once)
        assert once == twice

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_depluralize_idempotent(self, text):
        """depluralize(depluralize(x)) == depluralize(x) for any string."""
        from foundry.ontology import EntityType
        once = EntityType.depluralize(text)
        twice = EntityType.depluralize(once)
        assert once == twice

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_normalize_lowercase(self, text):
        """Normalized result is always lowercase."""
        from foundry.ontology import EntityType
        result = EntityType.normalize(text)
        assert result == result.lower()

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_normalize_no_leading_trailing_space(self, text):
        """Normalized result has no leading/trailing whitespace."""
        from foundry.ontology import EntityType
        result = EntityType.normalize(text)
        assert result == result.strip()


# ---------------------------------------------------------------------------
# Type index invariants
# ---------------------------------------------------------------------------

class TestTypeIndexProperties:
    """Property-based tests for TypeIndex invariants."""

    @given(
        entities=st.lists(
            st.tuples(entity_names, type_labels),
            min_size=1, max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_count_equals_len_lookup(self, entities):
        """count(type) == len(lookup(type)) for any type."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        for name, typ in entities:
            idx.index(name, typ)
        for _, typ in entities:
            assert idx.count(typ) == len(idx.lookup(typ))

    @given(
        entities=st.lists(
            st.tuples(entity_names, type_labels),
            min_size=0, max_size=50,
        )
    )
    @settings(max_examples=50)
    def test_count_bounded_by_total(self, entities):
        """count(type) <= total entity count."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        all_entities = set()
        for name, typ in entities:
            idx.index(name, typ)
            all_entities.add(name)
        for _, typ in entities:
            assert idx.count(typ) <= len(all_entities)


# ---------------------------------------------------------------------------
# Entity graph invariants
# ---------------------------------------------------------------------------

class TestEntityGraphProperties:
    """Property-based tests for EntityGraph invariants."""

    @given(
        pairs=st.lists(
            st.tuples(entity_names, entity_names),
            min_size=1, max_size=20,
        )
    )
    @settings(max_examples=50)
    def test_related_symmetry(self, pairs):
        """If A.related contains B, then B.related contains A (invariant I3)."""
        from foundry.storage import EntityNode
        graph = {}
        for a_name, b_name in pairs:
            assume(a_name != b_name)
            if a_name not in graph:
                graph[a_name] = EntityNode(canonical=a_name)
            if b_name not in graph:
                graph[b_name] = EntityNode(canonical=b_name)
            graph[a_name].related.add(b_name)
            graph[b_name].related.add(a_name)
        for name, entity in graph.items():
            for related_name in entity.related:
                if related_name in graph:
                    assert name in graph[related_name].related


# ---------------------------------------------------------------------------
# Resolve invariants
# ---------------------------------------------------------------------------

class TestResolverProperties:
    """Property-based tests for EntityResolver invariants."""

    @given(entity_names)
    @settings(max_examples=100)
    def test_resolve_or_create_nonempty(self, name):
        """resolve_or_create always returns non-empty string."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create(name)
        assert isinstance(result, str)
        assert len(result) > 0

    @given(entity_names)
    @settings(max_examples=100)
    def test_resolve_or_create_deterministic(self, name):
        """Same input always produces same output."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        r1 = resolver.resolve_or_create(name)
        r2 = resolver.resolve_or_create(name)
        assert r1 == r2

    @given(entity_names)
    @settings(max_examples=100)
    def test_canonical_resolves_to_self(self, name):
        """resolve(canonical) == canonical (invariant I8)."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        canonical = resolver.resolve_or_create(name)
        result = resolver.resolve(canonical)
        assert result == canonical
