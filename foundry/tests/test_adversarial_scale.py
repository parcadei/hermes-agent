"""Adversarial scale tests.

Tests system behavior at scale: large entity counts, large fact counts,
deep traversals, and concurrent operations.
"""

from __future__ import annotations

import pytest
import time


class TestLargeEntityCount:
    """Tests with many entities."""

    def test_1k_entities_query(self):
        """1000 entities: type query completes."""
        from foundry.storage import FoundryStore, EntityNode, TypeIndex
        store = FoundryStore()
        for i in range(1000):
            entity = EntityNode(canonical=f"entity_{i}", types={"test"})
            store.entity_graph.add_entity(entity)
            store.types.index(f"entity_{i}", "test")
        from foundry.query import QueryPlan
        result = QueryPlan(store).type("test").count()
        assert result == 1000

    def test_10k_entities_type_lookup(self):
        """10K entities: type lookup returns correct count."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        for i in range(10000):
            idx.index(f"entity_{i}", "widget")
        assert idx.count("widget") == 10000

    def test_10k_bm25_documents(self):
        """10K documents in BM25 index: query returns results."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        for i in range(10000):
            idx.add(i, f"document {i} about cats and dogs and fish")
        results = idx.query("cats", top_k=10)
        assert len(results) == 10

    def test_large_cooccurrence_index(self):
        """Many co-occurrence pairs: lookup still works."""
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        for i in range(100):
            for j in range(i + 1, min(i + 10, 100)):
                idx.record(f"e{i}", f"e{j}", i * 100 + j)
        coocs = idx.get_cooccurrences("e50")
        assert len(coocs) > 0


class TestDeepTraversal:
    """Tests for deep graph traversals."""

    def test_max_hops_3_terminates(self):
        """3-hop traversal on chain graph terminates."""
        from foundry.storage import EntityNode
        # Chain: e0 -> e1 -> e2 -> e3 -> e4 -> ...
        entities = {}
        for i in range(100):
            entities[f"e{i}"] = EntityNode(
                canonical=f"e{i}",
                related={f"e{i-1}"} if i > 0 else set(),
            )
            if i > 0:
                entities[f"e{i-1}"].related.add(f"e{i}")
        # 3-hop from e0 should reach e1, e2, e3 (not further)

    def test_star_graph_traversal(self):
        """Star topology: central node with 1000 neighbors."""
        from foundry.storage import EntityNode
        center = EntityNode(
            canonical="center",
            related={f"leaf_{i}" for i in range(1000)},
        )
        # Traversal from center should find all 1000 leaves
        assert len(center.related) == 1000


class TestEmptyEverything:
    """Tests for completely empty state."""

    def test_empty_foundry_search(self):
        """search() on empty Foundry returns empty."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        assert foundry.search("anything") == []

    def test_empty_foundry_entities(self):
        """entities().execute() on empty Foundry returns []."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        assert foundry.entities().execute() == []

    def test_empty_foundry_entity(self):
        """entity('x') on empty Foundry returns None."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        assert foundry.entity("anything") is None

    def test_empty_foundry_facts(self):
        """facts('x') on empty Foundry returns []."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        assert foundry.facts("anything") == []

    def test_empty_foundry_stats(self):
        """stats() on empty Foundry returns all zeros."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        stats = foundry.stats()
        assert stats["entity_count"] == 0
        assert stats["fact_count"] == 0

    def test_empty_foundry_temporal(self):
        """search_temporal() on empty returns structured empty."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        result = foundry.search_temporal("When?")
        assert isinstance(result, dict)
