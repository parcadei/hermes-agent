"""Behavioral tests: expected retrieval outputs.

Tests that the retrieval system (BM25 search, entity search, temporal
search) returns expected results for known queries over known data.
"""

from __future__ import annotations

import pytest
from datetime import datetime


class TestBM25Retrieval:
    """Tests for BM25 keyword search over raw chunks."""

    def test_search_finds_matching_chunk(self):
        """BM25 search('cat Luna') finds the chunk mentioning Luna."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import RawChunk
        foundry = Foundry(FoundryConfig())
        foundry._store.chunks.store(
            RawChunk(chunk_idx=0, text="I have a cat named Luna", turn_idx=0, role="user")
        )
        foundry._store.bm25_chunks.add(0, "I have a cat named Luna")
        results = foundry.search("cat Luna", top_k=5)
        assert len(results) > 0
        assert results[0][0] == 0  # chunk_idx

    def test_search_empty_query(self):
        """Empty query returns no results or all zeros."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        results = foundry.search("", top_k=5)
        assert len(results) == 0 or all(s == 0 for _, s in results)

    def test_search_no_match(self):
        """Search for absent term returns empty."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        foundry._store.bm25_chunks.add(0, "I have a cat named Luna")
        results = foundry.search("xyzzyx nonexistent term", top_k=5)
        matching = [(idx, s) for idx, s in results if s > 0]
        assert len(matching) == 0


class TestEntityRetrieval:
    """Tests for entity BM25 search."""

    def test_search_entities_finds_luna(self):
        """search_entities('luna cat') returns Luna entity."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode
        foundry = Foundry(FoundryConfig())
        luna = EntityNode(canonical="luna", types={"cat"})
        foundry._store.entity_graph.add_entity(luna)
        foundry._store.bm25_entities.add(0, "luna cat tabby pet")
        results = foundry.search_entities("luna cat", top_k=5)
        assert len(results) > 0
        # First result should be luna
        assert results[0][0] == "luna"


class TestTemporalRetrieval:
    """Tests for temporal query retrieval."""

    def test_search_temporal_empty(self):
        """Temporal search on empty index returns structured empty result."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        result = foundry.search_temporal("When did user visit Rome?")
        assert isinstance(result, dict)
        assert result.get("matched_events", []) == []
        assert result.get("confidence", 0.0) == 0.0


class TestFoundryStats:
    """Tests for the statistics/introspection interface."""

    def test_stats_on_empty(self):
        """stats() on empty Foundry returns all-zero counts."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        stats = foundry.stats()
        assert stats["entity_count"] == 0
        assert stats["fact_count"] == 0
        assert stats["chunk_count"] == 0
        assert stats["temporal_event_count"] == 0

    def test_stats_structure(self):
        """stats() returns all required keys."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        stats = foundry.stats()
        required_keys = [
            "entity_count", "fact_count", "current_fact_count",
            "alias_count", "chunk_count", "temporal_event_count",
            "type_distribution", "avg_facts_per_entity",
            "avg_aliases_per_entity", "grounding_rate", "layer_sizes",
        ]
        for key in required_keys:
            assert key in stats, f"Missing stats key: {key}"
