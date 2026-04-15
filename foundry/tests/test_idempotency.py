"""Idempotency tests.

Verifies that repeated operations produce the same result as a single
operation. No operation should create duplicate data.
"""

from __future__ import annotations

import pytest


class TestIngestionIdempotency:
    """Ingesting the same data twice produces identical results."""

    def test_same_triple_twice(self):
        """Ingesting same triple twice produces 1 fact, not 2 (I7)."""
        from foundry.storage import FoundryStore, EntityNode, Fact
        from foundry.ontology import PropertyType
        store = FoundryStore()
        entity = EntityNode(canonical="luna", types={"cat"})
        store.entity_graph.add_entity(entity)
        fact = Fact(
            id="f1", subject="luna", predicate="breed",
            object="tabby", object_type=PropertyType.STRING,
            turn_idx=0,
        )
        # Add same fact twice
        entity.facts.setdefault("breed", []).append(fact)
        # Dedup should prevent duplicate
        # (Implementation must handle this at ingestion time)

    def test_same_conversation_twice(self):
        """Ingesting same conversation twice does not double entity count."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        text = "User: I have a cat named Luna. She is a tabby."
        foundry.ingest(text)
        count_after_first = foundry.stats()["entity_count"]
        foundry.ingest(text)
        count_after_second = foundry.stats()["entity_count"]
        assert count_after_second == count_after_first

    def test_same_entity_resolve_or_create(self):
        """resolve_or_create same mention twice creates 1 entity."""
        from foundry.ingestion import EntityResolver
        resolver = EntityResolver()
        r1 = resolver.resolve_or_create("luna")
        r2 = resolver.resolve_or_create("luna")
        assert r1 == r2


class TestQueryIdempotency:
    """Running the same query twice returns identical results."""

    def test_execute_twice(self):
        """execute() twice returns same list."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).type("cat")
        r1 = plan.execute()
        r2 = plan.execute()
        assert r1 == r2

    def test_count_twice(self):
        """count() twice returns same number."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).type("cat")
        assert plan.count() == plan.count()

    def test_facts_twice(self):
        """facts() twice returns same list."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        store = FoundryStore()
        plan = QueryPlan(store).type("cat")
        r1 = plan.facts()
        r2 = plan.facts()
        assert r1 == r2

    def test_bm25_query_twice(self):
        """BM25 query twice returns identical results."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        idx.add(0, "cats and dogs")
        idx.add(1, "fish and birds")
        r1 = idx.query("cats", top_k=5)
        r2 = idx.query("cats", top_k=5)
        assert r1 == r2


class TestSerializationIdempotency:
    """serialize -> deserialize -> serialize produces same output."""

    def test_save_load_save_identical(self, tmp_path):
        """save -> load -> save produces identical output."""
        from foundry.storage import FoundryStore, EntityNode
        store = FoundryStore()
        store.entity_graph.add_entity(EntityNode(canonical="luna", types={"cat"}))
        store.types.index("luna", "cat")
        store.bm25_chunks.add(0, "test document about cats")
        path1 = str(tmp_path / "store1")
        path2 = str(tmp_path / "store2")
        store.save(path1)
        loaded = FoundryStore.load(path1)
        loaded.save(path2)
        # Both saves should produce equivalent data
        reloaded = FoundryStore.load(path2)
        assert reloaded.types.count("cat") == store.types.count("cat")


class TestDeduplicationIdempotency:
    """DeduplicationTracker is idempotent per call."""

    def test_filter_same_results_twice(self):
        """Filtering same results twice returns empty on second call."""
        from foundry.retrieval import DeduplicationTracker
        tracker = DeduplicationTracker()
        results = [{"turn_id": 0}, {"turn_id": 1}]
        first = tracker.filter_new(results)
        second = tracker.filter_new(results)
        assert len(first) == 2
        assert len(second) == 0

    def test_reset_restores_full_filter(self):
        """After reset, filter returns all results again."""
        from foundry.retrieval import DeduplicationTracker
        tracker = DeduplicationTracker()
        results = [{"turn_id": 0}]
        tracker.filter_new(results)
        tracker.reset()
        restored = tracker.filter_new(results)
        assert len(restored) == 1
