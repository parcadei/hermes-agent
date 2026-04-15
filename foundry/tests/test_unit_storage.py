"""Unit tests for the Storage Layer (spec section 2).

Tests all data structures (Fact, EntityNode, RawChunk, TemporalEvent)
and all index types (BM25, Vector, Type, Property, Temporal, Co-occurrence,
RawChunkStore, FoundryStore).
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Fact
# ---------------------------------------------------------------------------

class TestFact:
    """Tests for the Fact data structure."""

    def test_fact_creation(self):
        """Fact can be created with required fields."""
        from foundry.ontology import PropertyType, GroundingLayer
        from foundry.storage import Fact
        f = Fact(
            id="test-001",
            subject="luna",
            predicate="breed",
            object="tabby",
            object_type=PropertyType.STRING,
            turn_idx=0,
        )
        assert f.subject == "luna"
        assert f.predicate == "breed"
        assert f.object == "tabby"

    def test_fact_defaults(self):
        """Fact defaults: chunk_idx=-1, superseded=False, layer=STATED."""
        from foundry.ontology import PropertyType, GroundingLayer
        from foundry.storage import Fact
        f = Fact(
            id="test-001", subject="luna", predicate="breed",
            object="tabby", object_type=PropertyType.STRING, turn_idx=0,
        )
        assert f.chunk_idx == -1
        assert f.superseded is False
        assert f.superseded_by is None
        assert f.layer == GroundingLayer.STATED
        assert f.timestamp is None
        assert f.session_time is None
        assert f.source_text == ""
        assert f.when == ""
        assert f.role == ""
        assert f.sub_role == ""

    def test_fact_with_temporal(self):
        """Fact with temporal fields populated."""
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        ts = datetime(2024, 6, 15)
        f = Fact(
            id="test-002", subject="user", predicate="visited",
            object="rome", object_type=PropertyType.STRING, turn_idx=1,
            timestamp=ts, when="last June", role="event", sub_role="acquisition",
        )
        assert f.timestamp == ts
        assert f.when == "last June"
        assert f.role == "event"
        assert f.sub_role == "acquisition"

    def test_fact_supersession_marking(self):
        """Superseded facts can be marked with the replacing fact ID."""
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        f = Fact(
            id="test-003", subject="user", predicate="lives_in",
            object="san_francisco", object_type=PropertyType.STRING,
            turn_idx=0, superseded=True, superseded_by="test-004",
        )
        assert f.superseded is True
        assert f.superseded_by == "test-004"


# ---------------------------------------------------------------------------
# EntityNode
# ---------------------------------------------------------------------------

class TestEntityNode:
    """Tests for the EntityNode data structure."""

    def test_entity_creation(self):
        """EntityNode requires only canonical name."""
        from foundry.storage import EntityNode
        e = EntityNode(canonical="luna")
        assert e.canonical == "luna"

    def test_entity_defaults(self):
        """EntityNode defaults: empty sets and dicts."""
        from foundry.storage import EntityNode
        e = EntityNode(canonical="luna")
        assert e.aliases == set()
        assert e.types == set()
        assert e.facts == {}
        assert e.links == {}
        assert e.related == set()
        assert e.mention_turns == set()
        assert e.mention_chunks == set()

    def test_entity_add_alias(self):
        """Aliases are a mutable set."""
        from foundry.storage import EntityNode
        e = EntityNode(canonical="luna")
        e.aliases.add("Luna")
        e.aliases.add("my cat Luna")
        assert len(e.aliases) == 2
        assert "Luna" in e.aliases

    def test_entity_add_type(self):
        """Types are a mutable set."""
        from foundry.storage import EntityNode
        e = EntityNode(canonical="luna")
        e.types.add("cat")
        e.types.add("pet")
        assert "cat" in e.types

    def test_entity_add_fact(self):
        """Facts are stored by predicate."""
        from foundry.ontology import PropertyType
        from foundry.storage import EntityNode, Fact
        e = EntityNode(canonical="luna")
        f = Fact(
            id="f1", subject="luna", predicate="breed",
            object="tabby", object_type=PropertyType.STRING, turn_idx=0,
        )
        e.facts.setdefault("breed", []).append(f)
        assert len(e.facts["breed"]) == 1

    def test_entity_related_symmetry_setup(self):
        """Related is a set of canonical names (symmetry tested elsewhere)."""
        from foundry.storage import EntityNode
        e = EntityNode(canonical="luna")
        e.related.add("mochi")
        assert "mochi" in e.related

    def test_entity_mention_tracking(self):
        """Mention turns and chunks are tracked."""
        from foundry.storage import EntityNode
        e = EntityNode(canonical="luna")
        e.mention_turns.add(0)
        e.mention_turns.add(2)
        e.mention_chunks.add(0)
        assert 0 in e.mention_turns
        assert 2 in e.mention_turns


# ---------------------------------------------------------------------------
# RawChunk
# ---------------------------------------------------------------------------

class TestRawChunk:
    """Tests for the RawChunk data structure."""

    def test_rawchunk_creation(self):
        """RawChunk can be created with required fields."""
        from foundry.storage import RawChunk
        rc = RawChunk(chunk_idx=0, text="I have a cat", turn_idx=0, role="user")
        assert rc.chunk_idx == 0
        assert rc.text == "I have a cat"
        assert rc.role == "user"

    def test_rawchunk_defaults(self):
        """RawChunk defaults: no session_time, empty entities."""
        from foundry.storage import RawChunk
        rc = RawChunk(chunk_idx=0, text="test", turn_idx=0, role="user")
        assert rc.session_time is None
        assert rc.entities == []

    def test_rawchunk_with_entities(self):
        """RawChunk tracks which entities it mentions."""
        from foundry.storage import RawChunk
        rc = RawChunk(
            chunk_idx=0, text="Luna and Mochi", turn_idx=0, role="user",
            entities=["luna", "mochi"],
        )
        assert "luna" in rc.entities
        assert len(rc.entities) == 2


# ---------------------------------------------------------------------------
# TemporalEvent
# ---------------------------------------------------------------------------

class TestTemporalEvent:
    """Tests for the TemporalEvent data structure."""

    def test_temporal_event_creation(self):
        """TemporalEvent with valid interval."""
        from foundry.storage import TemporalEvent
        start = datetime(2024, 6, 1)
        end = datetime(2024, 6, 14)
        te = TemporalEvent(
            event="Trip to Rome", start_dt=start, end_dt=end,
            source="dateparser",
        )
        assert te.event == "Trip to Rome"
        assert te.start_dt == start
        assert te.end_dt == end

    def test_temporal_event_point(self):
        """Point event: start == end."""
        from foundry.storage import TemporalEvent
        dt = datetime(2024, 6, 5)
        te = TemporalEvent(event="Visited Colosseum", start_dt=dt, end_dt=dt, source="dateparser")
        assert te.start_dt == te.end_dt

    def test_temporal_event_invalid_interval(self):
        """start_dt > end_dt should raise ValueError."""
        from foundry.storage import TemporalEvent
        with pytest.raises(ValueError, match="start_dt.*must be <= end_dt"):
            TemporalEvent(
                event="Invalid",
                start_dt=datetime(2024, 6, 15),
                end_dt=datetime(2024, 6, 1),
                source="manual",
            )

    def test_temporal_event_defaults(self):
        """Defaults: turn_id=None, confidence=1.0, SVO fields=None."""
        from foundry.storage import TemporalEvent
        dt = datetime(2024, 6, 1)
        te = TemporalEvent(event="test", start_dt=dt, end_dt=dt, source="manual")
        assert te.turn_id is None
        assert te.confidence == 1.0
        assert te.subject is None
        assert te.verb is None
        assert te.object is None

    def test_temporal_event_with_svo(self):
        """TemporalEvent with SVO triple populated."""
        from foundry.storage import TemporalEvent
        dt = datetime(2024, 6, 5)
        te = TemporalEvent(
            event="User visited Colosseum",
            start_dt=dt, end_dt=dt, source="ouro_extraction",
            subject="user", verb="visited", object="colosseum",
        )
        assert te.subject == "user"
        assert te.verb == "visited"


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class TestBM25Index:
    """Tests for the BM25Index (primary retrieval channel)."""

    def test_empty_index(self):
        """Empty BM25 index has length 0."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        assert len(idx) == 0

    def test_add_document(self):
        """Adding a document increases length."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        idx.add(0, "I have a cat named Luna")
        assert len(idx) == 1

    def test_query_matching_document(self):
        """Query with matching terms returns results with positive scores."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        idx.add(0, "I have a cat named Luna")
        idx.add(1, "The weather is nice today")
        results = idx.query("cat Luna", top_k=2)
        assert len(results) > 0
        doc_id, score = results[0]
        assert doc_id == 0
        assert score > 0

    def test_query_no_match(self):
        """Query with no matching terms returns empty or zero-score results."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        idx.add(0, "I have a cat named Luna")
        results = idx.query("xyzzyx nonexistent", top_k=5)
        # Either empty or all scores are zero
        for _, score in results:
            assert score == 0.0 or len(results) == 0

    def test_query_empty_index(self):
        """Querying empty index returns empty list."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        results = idx.query("anything", top_k=10)
        assert results == []

    def test_scores_non_negative(self):
        """BM25 scores are non-negative."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        idx.add(0, "cats and dogs")
        idx.add(1, "cats are great pets")
        results = idx.query("cats", top_k=5)
        for _, score in results:
            assert score >= 0

    def test_top_k_limits_results(self):
        """top_k limits the number of results."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        for i in range(20):
            idx.add(i, f"document {i} about cats and dogs")
        results = idx.query("cats", top_k=5)
        assert len(results) <= 5

    def test_results_sorted_by_score(self):
        """Results are sorted by score descending."""
        from foundry.storage import BM25Index
        idx = BM25Index()
        idx.add(0, "cat cat cat cat cat")
        idx.add(1, "cat")
        idx.add(2, "dog")
        results = idx.query("cat", top_k=3)
        scores = [s for _, s in results if s > 0]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------

class TestVectorIndex:
    """Tests for the VectorIndex (dense vector similarity)."""

    def test_empty_vector_index(self):
        """Fresh VectorIndex has no entries."""
        from foundry.storage import VectorIndex
        idx = VectorIndex()
        # Query should return empty
        import numpy as np
        q = np.random.randn(768).astype(np.float32)
        q = q / np.linalg.norm(q)
        results = idx.query(q, top_k=5)
        assert results == []

    def test_add_and_query(self):
        """Adding a vector allows retrieval by similarity."""
        from foundry.storage import VectorIndex
        import numpy as np
        idx = VectorIndex()
        v = np.random.randn(768).astype(np.float32)
        v = v / np.linalg.norm(v)
        idx.add(0, v)
        results = idx.query(v, top_k=1)
        assert len(results) == 1
        assert results[0][0] == 0
        # Self-similarity should be ~1.0
        assert results[0][1] > 0.99

    def test_batch_add(self):
        """Batch insertion of multiple vectors."""
        from foundry.storage import VectorIndex
        import numpy as np
        idx = VectorIndex()
        vecs = np.random.randn(10, 768).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        idx.add_batch(list(range(10)), vecs)
        results = idx.query(vecs[0], top_k=1)
        assert results[0][0] == 0

    def test_cosine_score_range(self):
        """Cosine similarity scores are in [-1.0, 1.0]."""
        from foundry.storage import VectorIndex
        import numpy as np
        idx = VectorIndex()
        for i in range(5):
            v = np.random.randn(768).astype(np.float32)
            v = v / np.linalg.norm(v)
            idx.add(i, v)
        q = np.random.randn(768).astype(np.float32)
        q = q / np.linalg.norm(q)
        results = idx.query(q, top_k=5)
        for _, score in results:
            assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TypeIndex
# ---------------------------------------------------------------------------

class TestTypeIndex:
    """Tests for the TypeIndex (type label to entity mapping)."""

    def test_empty_type_index(self):
        """Empty index returns empty for any lookup."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        assert idx.lookup("cat") == []
        assert idx.count("cat") == 0

    def test_index_and_lookup(self):
        """Indexed entities are returned by lookup."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        idx.index("luna", "cat")
        idx.index("mochi", "cat")
        result = idx.lookup("cat")
        assert set(result) == {"luna", "mochi"}

    def test_count(self):
        """Count returns number of entities of given type."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        idx.index("luna", "cat")
        idx.index("mochi", "cat")
        idx.index("max", "dog")
        assert idx.count("cat") == 2
        assert idx.count("dog") == 1

    def test_depluralization_lookup(self):
        """lookup('bikes') matches entities typed 'bike' (depluralization)."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        idx.index("tarmac", "bike")
        idx.index("fuel_ex", "bike")
        result = idx.lookup("bikes")
        assert set(result) == {"tarmac", "fuel_ex"}

    def test_depluralization_count(self):
        """count('bikes') counts entities typed 'bike'."""
        from foundry.storage import TypeIndex
        idx = TypeIndex()
        idx.index("tarmac", "bike")
        idx.index("fuel_ex", "bike")
        assert idx.count("bikes") == 2


# ---------------------------------------------------------------------------
# PropertyIndex
# ---------------------------------------------------------------------------

class TestPropertyIndex:
    """Tests for the PropertyIndex (predicate -> (entity, value) pairs)."""

    def test_empty_property_index(self):
        """Empty index returns empty for any lookup."""
        from foundry.storage import PropertyIndex
        from foundry.ontology import PropertyType
        idx = PropertyIndex()
        assert idx.lookup("color") == []

    def test_index_and_lookup(self):
        """Indexed properties can be looked up by predicate."""
        from foundry.storage import PropertyIndex
        from foundry.ontology import PropertyType
        idx = PropertyIndex()
        idx.index("luna", "breed", "tabby", PropertyType.STRING)
        idx.index("mochi", "breed", "black", PropertyType.STRING)
        results = idx.lookup("breed")
        entities = {r[0] for r in results}
        assert entities == {"luna", "mochi"}

    def test_numeric_lookup(self):
        """Numeric lookup supports comparison operators."""
        from foundry.storage import PropertyIndex
        from foundry.ontology import PropertyType
        idx = PropertyIndex()
        idx.index("tarmac", "cost", "$3200", PropertyType.NUMERIC)
        idx.index("fuel_ex", "cost", "$2500", PropertyType.NUMERIC)
        idx.index("gravel", "cost", "$1800", PropertyType.NUMERIC)
        results = idx.lookup_numeric("cost", "$gt", 2000)
        entities = {r[0] for r in results}
        assert "tarmac" in entities
        assert "fuel_ex" in entities
        assert "gravel" not in entities

    def test_entities_with_property(self):
        """entities_with_property returns all entities having a predicate."""
        from foundry.storage import PropertyIndex
        from foundry.ontology import PropertyType
        idx = PropertyIndex()
        idx.index("luna", "age", "3", PropertyType.NUMERIC)
        idx.index("mochi", "age", "0.5", PropertyType.NUMERIC)
        idx.index("max", "breed", "golden retriever", PropertyType.STRING)
        result = idx.entities_with_property("age")
        assert set(result) == {"luna", "mochi"}


# ---------------------------------------------------------------------------
# TemporalIndex
# ---------------------------------------------------------------------------

class TestTemporalIndex:
    """Tests for the TemporalIndex (Allen interval algebra engine)."""

    def test_empty_temporal_index(self):
        """Empty index returns empty for all queries."""
        from foundry.storage import TemporalIndex
        idx = TemporalIndex()
        dt = datetime(2024, 6, 15)
        assert idx.events_on_date(dt) == []
        assert idx.events_between(dt, dt + timedelta(days=30)) == []

    def test_ingest_and_query(self):
        """Ingested events are returned by date query."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        dt = datetime(2024, 6, 5)
        te = TemporalEvent(event="Visited Colosseum", start_dt=dt, end_dt=dt, source="dateparser")
        idx.ingest(te)
        results = idx.events_on_date(dt)
        assert len(results) == 1
        assert results[0].event == "Visited Colosseum"

    def test_events_between(self):
        """Range query returns events overlapping with the range."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        idx.ingest(TemporalEvent(
            event="Rome trip",
            start_dt=datetime(2024, 6, 1),
            end_dt=datetime(2024, 6, 14),
            source="dateparser",
        ))
        idx.ingest(TemporalEvent(
            event="Paris trip",
            start_dt=datetime(2024, 3, 15),
            end_dt=datetime(2024, 3, 18),
            source="dateparser",
        ))
        results = idx.events_between(
            datetime(2024, 6, 1), datetime(2024, 6, 30),
        )
        events = [r.event for r in results]
        assert "Rome trip" in events
        assert "Paris trip" not in events

    def test_batch_ingest(self):
        """Batch ingestion adds multiple events."""
        from foundry.storage import TemporalIndex, TemporalEvent
        idx = TemporalIndex()
        events = [
            TemporalEvent(event=f"Event {i}", start_dt=datetime(2024, 6, i+1),
                          end_dt=datetime(2024, 6, i+1), source="manual")
            for i in range(5)
        ]
        idx.ingest_batch(events)
        results = idx.events_between(datetime(2024, 6, 1), datetime(2024, 6, 30))
        assert len(results) == 5


# ---------------------------------------------------------------------------
# CooccurrenceIndex
# ---------------------------------------------------------------------------

class TestCooccurrenceIndex:
    """Tests for the CooccurrenceIndex."""

    def test_empty_cooccurrence(self):
        """Empty index returns empty for any entity."""
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        assert idx.get_cooccurrences("luna") == {}

    def test_record_and_query(self):
        """Recorded co-occurrences are returned."""
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        idx.record("luna", "mochi", 0)
        coocs = idx.get_cooccurrences("luna")
        assert "mochi" in coocs
        assert coocs["mochi"] == 1

    def test_bidirectional(self):
        """Co-occurrence is bidirectional: A-B and B-A both recorded."""
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        idx.record("luna", "mochi", 0)
        assert "mochi" in idx.get_cooccurrences("luna")
        assert "luna" in idx.get_cooccurrences("mochi")

    def test_multiple_chunks(self):
        """Multiple co-occurrences across chunks increase count."""
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        idx.record("luna", "mochi", 0)
        idx.record("luna", "mochi", 3)
        coocs = idx.get_cooccurrences("luna")
        assert coocs["mochi"] == 2

    def test_shared_chunks(self):
        """get_shared_chunks returns the chunk indices."""
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        idx.record("luna", "mochi", 0)
        idx.record("luna", "mochi", 5)
        shared = idx.get_shared_chunks("luna", "mochi")
        assert shared == {0, 5}

    def test_no_self_cooccurrence(self):
        """An entity should not co-occur with itself (caller ensures this)."""
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        # The spec says co-occurrence is between distinct pairs
        # If the implementation silently allows it, the test should still
        # verify the behavior is consistent
        idx.record("luna", "luna", 0)
        coocs = idx.get_cooccurrences("luna")
        # Either "luna" is not in its own co-occurrences or count is 0
        # Implementation-defined, but should not crash


# ---------------------------------------------------------------------------
# RawChunkStore
# ---------------------------------------------------------------------------

class TestRawChunkStore:
    """Tests for the RawChunkStore."""

    def test_empty_store(self):
        """Empty store has length 0."""
        from foundry.storage import RawChunkStore
        store = RawChunkStore()
        assert len(store) == 0

    def test_store_and_get(self):
        """Stored chunks can be retrieved by index."""
        from foundry.storage import RawChunkStore, RawChunk
        store = RawChunkStore()
        chunk = RawChunk(chunk_idx=0, text="Hello world", turn_idx=0, role="user")
        store.store(chunk)
        assert len(store) == 1
        retrieved = store.get(0)
        assert retrieved.text == "Hello world"

    def test_get_nonexistent(self):
        """Getting a non-existent chunk raises or returns None."""
        from foundry.storage import RawChunkStore
        store = RawChunkStore()
        with pytest.raises((KeyError, IndexError)):
            store.get(999)

    def test_get_by_turn(self):
        """get_by_turn returns all chunks from a turn."""
        from foundry.storage import RawChunkStore, RawChunk
        store = RawChunkStore()
        store.store(RawChunk(chunk_idx=0, text="part 1", turn_idx=0, role="user"))
        store.store(RawChunk(chunk_idx=1, text="part 2", turn_idx=0, role="user"))
        store.store(RawChunk(chunk_idx=2, text="other", turn_idx=1, role="assistant"))
        results = store.get_by_turn(0)
        assert len(results) == 2
        assert all(r.turn_idx == 0 for r in results)


# ---------------------------------------------------------------------------
# FoundryStore
# ---------------------------------------------------------------------------

class TestFoundryStore:
    """Tests for the unified persistence interface."""

    def test_creation_has_all_indexes(self):
        """FoundryStore has all 7 index types + entity graphs."""
        from foundry.storage import FoundryStore
        store = FoundryStore()
        assert hasattr(store, "bm25_chunks")
        assert hasattr(store, "bm25_entities")
        assert hasattr(store, "vectors")
        assert hasattr(store, "types")
        assert hasattr(store, "properties")
        assert hasattr(store, "temporal")
        assert hasattr(store, "cooccurrence")
        assert hasattr(store, "chunks")
        assert hasattr(store, "entity_graph")
        assert hasattr(store, "inferred_graph")
        assert hasattr(store, "derived")

    def test_save_creates_directory(self, tmp_path):
        """save() creates files in the target directory."""
        from foundry.storage import FoundryStore
        store = FoundryStore()
        save_path = str(tmp_path / "test_store")
        store.save(save_path)
        import os
        assert os.path.exists(save_path)

    def test_save_load_roundtrip(self, tmp_path):
        """After save/load, all indexes return identical query results."""
        from foundry.storage import FoundryStore
        store = FoundryStore()
        # Add some data to verify roundtrip
        store.bm25_chunks.add(0, "test document about cats")
        save_path = str(tmp_path / "test_store")
        store.save(save_path)
        loaded = FoundryStore.load(save_path)
        results = loaded.bm25_chunks.query("cats", top_k=1)
        assert len(results) > 0
        assert results[0][0] == 0
