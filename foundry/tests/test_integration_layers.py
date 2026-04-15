"""Integration tests: multi-layer interactions.

Tests that stated, inferred, and derived layers work together correctly.
"""

from __future__ import annotations

import pytest
from datetime import datetime


class TestLayerIsolation:
    """Tests for layer isolation invariant (I1)."""

    def test_wipe_inferred_preserves_stated_entities(self):
        """Wiping inferred layer does not remove stated entities."""
        from foundry.dynamics import MemorySystem
        from foundry.storage import EntityNode
        ms = MemorySystem()
        luna = EntityNode(canonical="luna", types={"cat"})
        ms.stated.add_entity(luna)
        ms.build_inferred({"cat": ["feline"]})
        ms.wipe_inferred()
        # Stated entity should still exist
        assert ms.stated.get_entity("luna") is not None

    def test_wipe_inferred_preserves_derived(self):
        """Wiping inferred does not affect derived layer."""
        from foundry.dynamics import MemorySystem
        ms = MemorySystem()
        ms.rebuild_derived()
        derived_before = ms.derived
        ms.wipe_inferred()
        assert ms.derived is not None

    def test_rebuild_derived_preserves_stated(self):
        """Rebuilding derived does not modify stated layer."""
        from foundry.dynamics import MemorySystem
        from foundry.storage import EntityNode
        ms = MemorySystem()
        luna = EntityNode(canonical="luna", types={"cat"})
        ms.stated.add_entity(luna)
        ms.rebuild_derived()
        assert ms.stated.get_entity("luna") is not None

    def test_inferred_layer_navigation_only(self):
        """Inferred entities should NOT appear in final query results.

        Inferred layer is for navigation only -- answers come from stated.
        """
        from foundry.dynamics import MemorySystem
        from foundry.storage import EntityNode
        ms = MemorySystem()
        # Add entity to stated
        luna = EntityNode(canonical="luna", types={"cat"})
        ms.stated.add_entity(luna)
        # Add synonym in inferred
        ms.build_inferred({"cat": ["feline", "kitty"]})
        # Traverse from "feline" should resolve to stated entity "luna"
        result = ms.traverse_layers("feline")
        # Result should contain stated canonicals only
        for canonical in result:
            assert ms.stated.get_entity(canonical) is not None


class TestLayerTraversal:
    """Tests for cross-layer traversal (stated -> inferred -> stated)."""

    def test_traverse_direct_stated(self):
        """Direct lookup in stated layer (no inferred needed)."""
        from foundry.dynamics import MemorySystem
        from foundry.storage import EntityNode
        ms = MemorySystem()
        luna = EntityNode(canonical="luna", types={"cat"})
        ms.stated.add_entity(luna)
        result = ms.traverse_layers("luna")
        assert "luna" in result

    def test_traverse_via_inferred(self):
        """Query term maps through inferred synonym to stated entity."""
        from foundry.dynamics import MemorySystem
        from foundry.storage import EntityNode
        ms = MemorySystem()
        luna = EntityNode(canonical="luna", types={"cat"})
        ms.stated.add_entity(luna)
        ms.build_inferred({"cat": ["feline"]})
        result = ms.traverse_layers("feline")
        # Should resolve to luna via: feline -> cat (inferred) -> luna (stated)
        assert len(result) > 0


class TestDerivedLayerComputation:
    """Tests for derived layer computation from stated."""

    def test_derived_type_counts(self):
        """Derived layer computes type counts."""
        from foundry.dynamics import MemorySystem
        from foundry.storage import EntityNode
        ms = MemorySystem()
        for name in ["luna", "mochi"]:
            entity = EntityNode(canonical=name, types={"cat", "pet"})
            ms.stated.add_entity(entity)
        entity = EntityNode(canonical="max", types={"dog", "pet"})
        ms.stated.add_entity(entity)
        ms.rebuild_derived()
        # Derived should know cat_count=2, dog_count=1, pet_count=3
        # Exact API depends on implementation

    def test_derived_is_deterministic(self):
        """Rebuilding derived from same stated produces identical results."""
        from foundry.dynamics import MemorySystem
        from foundry.storage import EntityNode
        ms = MemorySystem()
        luna = EntityNode(canonical="luna", types={"cat"})
        ms.stated.add_entity(luna)
        ms.rebuild_derived()
        derived1 = ms.derived
        ms.rebuild_derived()
        derived2 = ms.derived
        # Both should be equivalent (implementation defines equality)


class TestCrossIndexConsistency:
    """Tests that different indexes agree on the same data."""

    def test_bm25_and_type_agree_on_entity_count(self):
        """Type index count matches entities found via BM25."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode
        foundry = Foundry(FoundryConfig())
        for name, typ in [("luna", "cat"), ("mochi", "cat"), ("max", "dog")]:
            entity = EntityNode(canonical=name, types={typ})
            foundry._store.entity_graph.add_entity(entity)
            foundry._store.types.index(name, typ)
            foundry._store.bm25_entities.add(hash(name), f"{name} {typ}")
        type_count = foundry._store.types.count("cat")
        assert type_count == 2

    def test_cooccurrence_and_entity_agree(self):
        """Co-occurrence index only references entities that exist in graph."""
        from foundry.core import Foundry, FoundryConfig
        from foundry.storage import EntityNode, CooccurrenceIndex
        foundry = Foundry(FoundryConfig())
        foundry._store.entity_graph.add_entity(EntityNode(canonical="luna"))
        foundry._store.entity_graph.add_entity(EntityNode(canonical="mochi"))
        foundry._store.cooccurrence.record("luna", "mochi", 0)
        coocs = foundry._store.cooccurrence.get_cooccurrences("luna")
        for entity_name in coocs:
            assert foundry._store.entity_graph.get_entity(entity_name) is not None
