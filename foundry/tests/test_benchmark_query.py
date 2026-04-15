"""Benchmark tests: query performance contracts.

Verifies that queries complete within specified time bounds.
Uses time.time() for portable assertions.
"""

from __future__ import annotations

import pytest
import time


class TestQueryPerformance:
    """Performance contracts for query operations."""

    def test_query_1k_entities_under_100ms(self):
        """Query over 1K entities completes in <100ms."""
        from foundry.storage import FoundryStore, EntityNode, TypeIndex
        from foundry.query import QueryPlan
        store = FoundryStore()
        for i in range(1000):
            entity = EntityNode(canonical=f"e{i}", types={"widget"})
            store.entity_graph.add_entity(entity)
            store.types.index(f"e{i}", "widget")
        start = time.time()
        result = QueryPlan(store).type("widget").execute()
        elapsed_ms = (time.time() - start) * 1000
        assert len(result) == 1000
        assert elapsed_ms < 100, f"Query took {elapsed_ms:.1f}ms (limit: 100ms)"

    def test_query_10k_entities_under_1s(self):
        """Query over 10K entities completes in <1s."""
        from foundry.storage import FoundryStore, EntityNode
        from foundry.query import QueryPlan
        store = FoundryStore()
        for i in range(10000):
            entity = EntityNode(canonical=f"e{i}", types={"item"})
            store.entity_graph.add_entity(entity)
            store.types.index(f"e{i}", "item")
        start = time.time()
        result = QueryPlan(store).type("item").count()
        elapsed_ms = (time.time() - start) * 1000
        assert result == 10000
        assert elapsed_ms < 1000, f"Query took {elapsed_ms:.1f}ms (limit: 1000ms)"

    def test_bm25_10k_chunks_under_10ms(self):
        """BM25 query over 10K chunks completes in <10ms."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        for i in range(10000):
            idx.add(i, f"document {i} about various topics including cats and dogs")
        start = time.time()
        results = idx.query("cats dogs", top_k=10)
        elapsed_ms = (time.time() - start) * 1000
        assert len(results) <= 10
        assert elapsed_ms < 10, f"BM25 query took {elapsed_ms:.1f}ms (limit: 10ms)"

    def test_type_index_1k_under_1ms(self):
        """Type index lookup with 1K entities completes in <1ms."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        for i in range(1000):
            idx.index(f"e{i}", "thing")
        start = time.time()
        result = idx.lookup("thing")
        elapsed_ms = (time.time() - start) * 1000
        assert len(result) == 1000
        assert elapsed_ms < 1, f"Type lookup took {elapsed_ms:.1f}ms (limit: 1ms)"

    def test_resolve_exact_under_1ms(self):
        """resolve_exact completes in <1ms per call."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        for i in range(1000):
            resolver.register(f"entity_{i}", aliases={f"alias_{i}"})
        start = time.time()
        for i in range(100):
            resolver.resolve_exact(f"alias_{i}")
        elapsed_ms = (time.time() - start) * 1000
        per_call = elapsed_ms / 100
        assert per_call < 1, f"resolve_exact took {per_call:.3f}ms/call (limit: 1ms)"

    def test_count_under_10ms_for_10k(self):
        """count() over 10K entities completes in <10ms."""
        from foundry.storage import FoundryStore, EntityNode
        from foundry.query import QueryPlan
        store = FoundryStore()
        for i in range(10000):
            entity = EntityNode(canonical=f"e{i}", types={"item"})
            store.entity_graph.add_entity(entity)
            store.types.index(f"e{i}", "item")
        start = time.time()
        count = QueryPlan(store).type("item").count()
        elapsed_ms = (time.time() - start) * 1000
        assert count == 10000
        assert elapsed_ms < 10, f"count took {elapsed_ms:.1f}ms (limit: 10ms)"
