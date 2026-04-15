"""Fuzz tests: random entity name generation.

Uses Hypothesis to generate random entity names, types, and predicates.
No operation should crash -- all should either return results or empty.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


random_text = st.text(min_size=0, max_size=200)
random_nonempty = st.text(min_size=1, max_size=200)
random_ascii = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1, max_size=100,
)


class TestFuzzEntityResolver:
    """Fuzz: EntityResolver must never crash on any input."""

    @given(random_nonempty)
    @settings(max_examples=300)
    def test_resolve_or_create_never_crashes(self, name):
        """resolve_or_create handles any non-empty string."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_or_create(name)
        assert isinstance(result, str)
        assert len(result) > 0

    @given(random_text)
    @settings(max_examples=300)
    def test_resolve_exact_never_crashes(self, name):
        """resolve_exact handles any string (returns None for unknown)."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve_exact(name)
        assert result is None or isinstance(result, str)

    @given(random_text)
    @settings(max_examples=300)
    def test_resolve_never_crashes(self, name):
        """resolve handles any string."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        result = resolver.resolve(name)
        assert result is None or isinstance(result, str)


class TestFuzzEntityType:
    """Fuzz: EntityType normalization must never crash."""

    @given(random_text)
    @settings(max_examples=300)
    def test_normalize_never_crashes(self, text):
        """normalize handles any string."""
        from foundry.ontology import EntityType
        result = EntityType.normalize(text)
        assert isinstance(result, str)

    @given(random_text)
    @settings(max_examples=300)
    def test_depluralize_never_crashes(self, text):
        """depluralize handles any string."""
        from foundry.ontology import EntityType
        result = EntityType.depluralize(text)
        assert isinstance(result, str)


class TestFuzzPropertyType:
    """Fuzz: Property type inference must never crash."""

    @given(random_text)
    @settings(max_examples=300)
    def test_infer_never_crashes(self, text):
        """infer_property_type handles any string."""
        from foundry.ontology import infer_property_type, PropertyType
        result = infer_property_type(text)
        assert isinstance(result, PropertyType)


class TestFuzzBM25:
    """Fuzz: BM25 index must never crash on any query."""

    @given(
        docs=st.lists(random_nonempty, min_size=0, max_size=20),
        query=random_text,
    )
    @settings(max_examples=100)
    def test_bm25_never_crashes(self, docs, query):
        """BM25 handles any documents and queries."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        for i, doc in enumerate(docs):
            idx.add(i, doc)
        results = idx.query(query, top_k=5)
        assert isinstance(results, list)
        for doc_id, score in results:
            assert isinstance(doc_id, int)
            assert isinstance(score, (int, float))
            assert score >= 0


class TestFuzzTypeIndex:
    """Fuzz: TypeIndex must never crash."""

    @given(
        entries=st.lists(
            st.tuples(random_nonempty, random_nonempty),
            min_size=0, max_size=50,
        ),
        query_type=random_text,
    )
    @settings(max_examples=100)
    def test_type_index_never_crashes(self, entries, query_type):
        """TypeIndex handles any entity names and type labels."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        for name, typ in entries:
            idx.index(name, typ)
        result = idx.lookup(query_type)
        assert isinstance(result, list)
        count = idx.count(query_type)
        assert isinstance(count, int)
        assert count >= 0


class TestFuzzGrounding:
    """Fuzz: Grounding verification must never crash."""

    @given(
        obj=random_text,
        source=random_text,
    )
    @settings(max_examples=200)
    def test_grounding_never_crashes(self, obj, source):
        """verify_grounding handles any object and source text."""
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        f = Fact(
            id="fuzz", subject="x", predicate="p",
            object=obj, object_type=PropertyType.STRING,
            turn_idx=0, source_text=source,
        )
        result = verify_grounding(f)
        assert isinstance(result, bool)
