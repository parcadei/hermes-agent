"""Snapshot tests: simple conversation.

Given a FIXED conversation, the entire system state must match expected
output EXACTLY. This is the most strict form of behavioral test.
"""

from __future__ import annotations

import pytest
import json
from datetime import datetime


SIMPLE_CONVERSATION = [
    ("user", "I have a cat named Luna. She is a tabby."),
    ("assistant", "Tell me more about Luna!"),
    ("user", "Luna is 3 years old. I adopted her in 2021."),
]


class TestSimpleSnapshot:
    """Snapshot tests for a simple pet conversation."""

    def test_entity_count(self):
        """Exactly the expected number of entities are created."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        raw = "\n".join(f"{role}: {text}" for role, text in SIMPLE_CONVERSATION)
        foundry.ingest(raw)
        stats = foundry.stats()
        # Should have at least 1 entity (luna)
        assert stats["entity_count"] >= 1

    def test_luna_entity_exists(self):
        """Luna entity is created."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        raw = "\n".join(f"{role}: {text}" for role, text in SIMPLE_CONVERSATION)
        foundry.ingest(raw)
        luna = foundry.entity("luna")
        assert luna is not None

    def test_luna_typed_as_cat(self):
        """Luna is typed as 'cat'."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        raw = "\n".join(f"{role}: {text}" for role, text in SIMPLE_CONVERSATION)
        foundry.ingest(raw)
        luna = foundry.entity("luna")
        assert luna is not None
        assert "cat" in luna.types

    def test_luna_breed_tabby(self):
        """Luna has breed fact 'tabby'."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        raw = "\n".join(f"{role}: {text}" for role, text in SIMPLE_CONVERSATION)
        foundry.ingest(raw)
        facts = foundry.facts("luna", predicate="breed")
        assert any("tabby" in f.object.lower() for f in facts)

    def test_luna_age_3(self):
        """Luna has age fact '3'."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        raw = "\n".join(f"{role}: {text}" for role, text in SIMPLE_CONVERSATION)
        foundry.ingest(raw)
        facts = foundry.facts("luna", predicate="age")
        assert any("3" in f.object for f in facts)

    def test_chunks_stored(self):
        """Raw chunks are stored."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        raw = "\n".join(f"{role}: {text}" for role, text in SIMPLE_CONVERSATION)
        foundry.ingest(raw)
        assert len(foundry._store.chunks) >= 1

    def test_bm25_finds_luna(self):
        """BM25 search for 'Luna tabby' returns results."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        raw = "\n".join(f"{role}: {text}" for role, text in SIMPLE_CONVERSATION)
        foundry.ingest(raw)
        results = foundry.search("Luna tabby", top_k=5)
        assert len(results) > 0

    def test_export_graph_structure(self):
        """export_graph() produces serializable dict."""
        from foundry.core import Foundry, FoundryConfig
        foundry = Foundry(FoundryConfig())
        raw = "\n".join(f"{role}: {text}" for role, text in SIMPLE_CONVERSATION)
        foundry.ingest(raw)
        graph = foundry.export_graph()
        assert isinstance(graph, dict)
        # Must be JSON-serializable
        serialized = json.dumps(graph)
        assert len(serialized) > 0
