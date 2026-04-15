"""Benchmark tests: ingestion performance contracts.

Verifies that ingestion and serialization complete within specified
time bounds.
"""

from __future__ import annotations

import pytest
import time


class TestIngestionPerformance:
    """Performance contracts for ingestion operations."""

    def test_1000_triples_under_5s(self):
        """Ingestion of 1000 triples completes in <5s."""
        from foundry.storage import FoundryStore, EntityNode, Fact
        from foundry.ontology import PropertyType
        store = FoundryStore()
        start = time.time()
        for i in range(1000):
            entity_name = f"entity_{i % 100}"
            if store.entity_graph.get_entity(entity_name) is None:
                store.entity_graph.add_entity(
                    EntityNode(canonical=entity_name, types={"test"})
                )
                store.types.index(entity_name, "test")
            fact = Fact(
                id=f"f{i}", subject=entity_name,
                predicate=f"prop_{i % 10}", object=f"value_{i}",
                object_type=PropertyType.STRING, turn_idx=i,
            )
            store.properties.index(
                entity_name, f"prop_{i % 10}", f"value_{i}", PropertyType.STRING
            )
        elapsed_s = time.time() - start
        assert elapsed_s < 5.0, f"Ingestion took {elapsed_s:.2f}s (limit: 5s)"

    def test_bm25_add_1000_docs_under_2s(self):
        """Adding 1000 documents to BM25 completes in <2s."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        start = time.time()
        for i in range(1000):
            idx.add(i, f"Document {i} contains text about topic {i % 50} "
                       f"with various words for BM25 indexing test")
        elapsed_s = time.time() - start
        assert elapsed_s < 2.0, f"BM25 add took {elapsed_s:.2f}s (limit: 2s)"


class TestSerializationPerformance:
    """Performance contracts for serialization round-trips."""

    def test_save_load_10k_entities_under_5s(self, tmp_path):
        """Save/load of 10K entities completes in <5s total."""
        from foundry.storage import FoundryStore, EntityNode
        store = FoundryStore()
        for i in range(10000):
            store.entity_graph.add_entity(
                EntityNode(canonical=f"e{i}", types={"item"})
            )
            store.types.index(f"e{i}", "item")
        save_path = str(tmp_path / "bench_store")
        start = time.time()
        store.save(save_path)
        loaded = FoundryStore.load(save_path)
        elapsed_s = time.time() - start
        assert elapsed_s < 5.0, f"Save/load took {elapsed_s:.2f}s (limit: 5s)"
        assert loaded.types.count("item") == 10000
