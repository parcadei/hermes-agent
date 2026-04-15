"""Serialization round-trip tests.

Tests that every data structure can be serialized and deserialized
without data loss.
"""

from __future__ import annotations

import pytest
import json
from datetime import datetime


class TestFactSerialization:
    """Round-trip serialization for Fact."""

    def test_fact_to_dict_from_dict(self):
        """Fact -> dict -> Fact preserves all fields."""
        from foundry.storage import Fact
        from foundry.ontology import PropertyType, GroundingLayer
        original = Fact(
            id="f1", subject="luna", predicate="breed",
            object="tabby", object_type=PropertyType.STRING,
            turn_idx=0, chunk_idx=2,
            timestamp=datetime(2024, 6, 15),
            session_time=datetime(2024, 6, 15, 10, 0, 0),
            source_text="Luna is a tabby",
            layer=GroundingLayer.STATED,
            superseded=False,
            superseded_by=None,
            when="since 2021",
            role="state",
            sub_role="maintenance",
        )
        d = original.to_dict()
        restored = Fact.from_dict(d)
        assert restored.id == original.id
        assert restored.subject == original.subject
        assert restored.predicate == original.predicate
        assert restored.object == original.object
        assert restored.object_type == original.object_type
        assert restored.turn_idx == original.turn_idx
        assert restored.chunk_idx == original.chunk_idx
        assert restored.source_text == original.source_text
        assert restored.layer == original.layer
        assert restored.superseded == original.superseded
        assert restored.when == original.when
        assert restored.role == original.role
        assert restored.sub_role == original.sub_role

    def test_fact_json_roundtrip(self):
        """Fact -> JSON -> Fact preserves data."""
        from foundry.storage import Fact
        from foundry.ontology import PropertyType
        original = Fact(
            id="f1", subject="luna", predicate="breed",
            object="tabby", object_type=PropertyType.STRING,
            turn_idx=0,
        )
        d = original.to_dict()
        json_str = json.dumps(d, default=str)
        loaded = json.loads(json_str)
        restored = Fact.from_dict(loaded)
        assert restored.subject == original.subject

    def test_fact_with_none_fields(self):
        """Fact with None timestamp serializes correctly."""
        from foundry.storage import Fact
        from foundry.ontology import PropertyType
        original = Fact(
            id="f1", subject="luna", predicate="age",
            object="3", object_type=PropertyType.NUMERIC,
            turn_idx=0,
        )
        d = original.to_dict()
        restored = Fact.from_dict(d)
        assert restored.timestamp is None
        assert restored.superseded_by is None


class TestEntityNodeSerialization:
    """Round-trip serialization for EntityNode."""

    def test_entity_to_dict_from_dict(self):
        """EntityNode -> dict -> EntityNode preserves all fields."""
        from foundry.storage import EntityNode
        original = EntityNode(
            canonical="luna",
            aliases={"Luna", "my cat"},
            types={"cat", "pet"},
            related={"mochi", "user"},
            mention_turns={0, 2, 5},
            mention_chunks={0, 1},
        )
        d = original.to_dict()
        restored = EntityNode.from_dict(d)
        assert restored.canonical == original.canonical
        assert restored.aliases == original.aliases
        assert restored.types == original.types
        assert restored.related == original.related
        assert restored.mention_turns == original.mention_turns

    def test_entity_empty_fields(self):
        """EntityNode with all empty fields serializes correctly."""
        from foundry.storage import EntityNode
        original = EntityNode(canonical="empty")
        d = original.to_dict()
        restored = EntityNode.from_dict(d)
        assert restored.canonical == "empty"
        assert restored.aliases == set()
        assert restored.types == set()
        assert restored.facts == {}


class TestRawChunkSerialization:
    """Round-trip serialization for RawChunk."""

    def test_chunk_to_dict_from_dict(self):
        """RawChunk -> dict -> RawChunk preserves all fields."""
        from foundry.storage import RawChunk
        original = RawChunk(
            chunk_idx=0, text="Hello world",
            turn_idx=0, role="user",
            session_time=datetime(2024, 6, 15),
            entities=["luna", "mochi"],
        )
        d = original.to_dict()
        restored = RawChunk.from_dict(d)
        assert restored.chunk_idx == original.chunk_idx
        assert restored.text == original.text
        assert restored.role == original.role
        assert restored.entities == original.entities


class TestTemporalEventSerialization:
    """Round-trip serialization for TemporalEvent."""

    def test_event_to_dict_from_dict(self):
        """TemporalEvent -> dict -> TemporalEvent preserves all fields."""
        from foundry.storage import TemporalEvent
        original = TemporalEvent(
            event="Visited Rome",
            start_dt=datetime(2024, 6, 1),
            end_dt=datetime(2024, 6, 14),
            source="dateparser",
            turn_id=3,
            confidence=0.95,
            subject="user",
            verb="visited",
            object="rome",
        )
        d = original.to_dict()
        restored = TemporalEvent.from_dict(d)
        assert restored.event == original.event
        assert restored.start_dt == original.start_dt
        assert restored.end_dt == original.end_dt
        assert restored.confidence == original.confidence
        assert restored.subject == original.subject

    def test_event_point_serializes(self):
        """Point event (start == end) serializes correctly."""
        from foundry.storage import TemporalEvent
        dt = datetime(2024, 6, 5)
        original = TemporalEvent(event="test", start_dt=dt, end_dt=dt, source="manual")
        d = original.to_dict()
        restored = TemporalEvent.from_dict(d)
        assert restored.start_dt == restored.end_dt


class TestFoundryStoreSerialization:
    """Round-trip serialization for the full FoundryStore."""

    def test_full_store_roundtrip(self, tmp_path):
        """FoundryStore save -> load preserves all data."""
        from foundry.storage import FoundryStore, EntityNode, RawChunk
        store = FoundryStore()
        # Add entity
        entity = EntityNode(canonical="luna", aliases={"Luna"}, types={"cat"})
        store.entity_graph.add_entity(entity)
        store.types.index("luna", "cat")
        # Add chunk
        store.chunks.store(
            RawChunk(chunk_idx=0, text="Luna is a cat", turn_idx=0, role="user")
        )
        store.bm25_chunks.add(0, "Luna is a cat")
        # Save and load
        path = str(tmp_path / "test_store")
        store.save(path)
        loaded = FoundryStore.load(path)
        # Verify
        assert loaded.types.count("cat") == 1
        assert loaded.entity_graph.get_entity("luna") is not None
        assert len(loaded.chunks) == 1
        results = loaded.bm25_chunks.query("Luna", top_k=1)
        assert len(results) > 0

    def test_empty_store_roundtrip(self, tmp_path):
        """Empty FoundryStore save -> load works."""
        from foundry.storage import FoundryStore
        store = FoundryStore()
        path = str(tmp_path / "empty_store")
        store.save(path)
        loaded = FoundryStore.load(path)
        assert loaded.types.count("anything") == 0
        assert len(loaded.chunks) == 0


class TestQueryPlanNotSerializable:
    """QueryPlan is not meant to be serialized (it holds a store reference)."""

    def test_queryplan_has_no_to_dict(self):
        """QueryPlan should not have to_dict (it's ephemeral)."""
        from foundry.query import QueryPlan
        from foundry.storage import FoundryStore
        plan = QueryPlan(FoundryStore())
        assert not hasattr(plan, "to_dict")


class TestCollectionEdgeCases:
    """Serialization edge cases for collections."""

    def test_empty_set_serializes(self):
        """Empty set fields serialize and deserialize correctly."""
        from foundry.storage import EntityNode
        original = EntityNode(canonical="empty")
        d = original.to_dict()
        restored = EntityNode.from_dict(d)
        assert restored.aliases == set()
        assert restored.types == set()

    def test_nested_facts_serialize(self):
        """Entity with nested fact lists serializes correctly."""
        from foundry.storage import EntityNode, Fact
        from foundry.ontology import PropertyType
        entity = EntityNode(canonical="luna")
        entity.facts["breed"] = [
            Fact(id="f1", subject="luna", predicate="breed",
                 object="tabby", object_type=PropertyType.STRING, turn_idx=0),
            Fact(id="f2", subject="luna", predicate="breed",
                 object="domestic shorthair", object_type=PropertyType.STRING,
                 turn_idx=5),
        ]
        d = entity.to_dict()
        restored = EntityNode.from_dict(d)
        assert len(restored.facts["breed"]) == 2
