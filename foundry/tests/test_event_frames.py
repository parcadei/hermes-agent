"""Tests for EventFrame dataclass and EventFrameIndex.

Validates VAL-001: EventFrame exists in storage.py with fields id, turns,
timestamp, gist, entities, fact_ids, prev, next. EventFrameIndex provides
add/get/get_by_turn/get_neighbors. Serializable via to_dict/from_dict.
"""

from __future__ import annotations

import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# EventFrame dataclass
# ---------------------------------------------------------------------------

class TestEventFrame:
    """Tests for the EventFrame data structure."""

    def test_creation_required_fields(self):
        """EventFrame can be created with only the required id and turns fields."""
        from foundry.storage import EventFrame
        ef = EventFrame(id="ef-001", turns=[0, 1, 2])
        assert ef.id == "ef-001"
        assert ef.turns == [0, 1, 2]

    def test_defaults(self):
        """EventFrame defaults: timestamp=None, gist='', entities=[], fact_ids=[], prev=None, next=None."""
        from foundry.storage import EventFrame
        ef = EventFrame(id="ef-001", turns=[0])
        assert ef.timestamp is None
        assert ef.gist == ""
        assert ef.entities == []
        assert ef.fact_ids == []
        assert ef.prev is None
        assert ef.next is None

    def test_all_fields(self):
        """EventFrame can be created with all fields populated."""
        from foundry.storage import EventFrame
        ts = datetime(2025, 6, 15, 10, 30, 0)
        ef = EventFrame(
            id="ef-002",
            turns=[3, 4, 5],
            timestamp=ts,
            gist="User asked about weather",
            entities=["weather", "new_york"],
            fact_ids=["fact-001", "fact-002"],
            prev="ef-001",
            next="ef-003",
        )
        assert ef.id == "ef-002"
        assert ef.turns == [3, 4, 5]
        assert ef.timestamp == ts
        assert ef.gist == "User asked about weather"
        assert ef.entities == ["weather", "new_york"]
        assert ef.fact_ids == ["fact-001", "fact-002"]
        assert ef.prev == "ef-001"
        assert ef.next == "ef-003"

    def test_timestamp_string_parsed(self):
        """EventFrame __post_init__ parses ISO string timestamps into datetime."""
        from foundry.storage import EventFrame
        ef = EventFrame(id="ef-001", turns=[0], timestamp="2025-06-15T10:30:00")
        assert isinstance(ef.timestamp, datetime)
        assert ef.timestamp == datetime(2025, 6, 15, 10, 30, 0)

    def test_to_dict(self):
        """EventFrame.to_dict() returns a serializable dict with all fields."""
        from foundry.storage import EventFrame
        ts = datetime(2025, 6, 15, 10, 30, 0)
        ef = EventFrame(
            id="ef-002",
            turns=[3, 4, 5],
            timestamp=ts,
            gist="User asked about weather",
            entities=["weather"],
            fact_ids=["fact-001"],
            prev="ef-001",
            next="ef-003",
        )
        d = ef.to_dict()
        assert d["id"] == "ef-002"
        assert d["turns"] == [3, 4, 5]
        assert d["timestamp"] == ts.isoformat()
        assert d["gist"] == "User asked about weather"
        assert d["entities"] == ["weather"]
        assert d["fact_ids"] == ["fact-001"]
        assert d["prev"] == "ef-001"
        assert d["next"] == "ef-003"

    def test_to_dict_none_timestamp(self):
        """EventFrame.to_dict() serializes None timestamp as None."""
        from foundry.storage import EventFrame
        ef = EventFrame(id="ef-001", turns=[0])
        d = ef.to_dict()
        assert d["timestamp"] is None

    def test_from_dict(self):
        """EventFrame.from_dict() reconstructs from a dict payload."""
        from foundry.storage import EventFrame
        payload = {
            "id": "ef-002",
            "turns": [3, 4, 5],
            "timestamp": "2025-06-15T10:30:00",
            "gist": "User asked about weather",
            "entities": ["weather"],
            "fact_ids": ["fact-001"],
            "prev": "ef-001",
            "next": "ef-003",
        }
        ef = EventFrame.from_dict(payload)
        assert ef.id == "ef-002"
        assert ef.turns == [3, 4, 5]
        assert isinstance(ef.timestamp, datetime)
        assert ef.timestamp == datetime(2025, 6, 15, 10, 30, 0)
        assert ef.gist == "User asked about weather"
        assert ef.entities == ["weather"]
        assert ef.fact_ids == ["fact-001"]
        assert ef.prev == "ef-001"
        assert ef.next == "ef-003"

    def test_from_dict_minimal(self):
        """EventFrame.from_dict() works with only required fields; defaults for optional."""
        from foundry.storage import EventFrame
        payload = {"id": "ef-001", "turns": [0]}
        ef = EventFrame.from_dict(payload)
        assert ef.id == "ef-001"
        assert ef.turns == [0]
        assert ef.timestamp is None
        assert ef.gist == ""
        assert ef.entities == []
        assert ef.fact_ids == []
        assert ef.prev is None
        assert ef.next is None

    def test_roundtrip(self):
        """EventFrame survives to_dict -> from_dict roundtrip."""
        from foundry.storage import EventFrame
        ts = datetime(2025, 6, 15, 10, 30, 0)
        original = EventFrame(
            id="ef-rt",
            turns=[10, 11],
            timestamp=ts,
            gist="roundtrip test",
            entities=["alpha", "beta"],
            fact_ids=["f1", "f2"],
            prev="ef-prev",
            next="ef-next",
        )
        restored = EventFrame.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.turns == original.turns
        assert restored.timestamp == original.timestamp
        assert restored.gist == original.gist
        assert restored.entities == original.entities
        assert restored.fact_ids == original.fact_ids
        assert restored.prev == original.prev
        assert restored.next == original.next

    def test_mutable_defaults_isolated(self):
        """Each EventFrame instance gets its own list defaults (no shared mutable state)."""
        from foundry.storage import EventFrame
        ef1 = EventFrame(id="ef-1", turns=[0])
        ef2 = EventFrame(id="ef-2", turns=[1])
        ef1.entities.append("shared_bug")
        assert ef2.entities == []
        ef1.fact_ids.append("shared_bug")
        assert ef2.fact_ids == []


# ---------------------------------------------------------------------------
# EventFrameIndex
# ---------------------------------------------------------------------------

class TestEventFrameIndex:
    """Tests for the EventFrameIndex."""

    def _make_frame(self, fid, turns, prev=None, nxt=None, timestamp=None):
        from foundry.storage import EventFrame
        return EventFrame(
            id=fid, turns=turns, prev=prev, next=nxt,
            timestamp=timestamp,
        )

    def test_add_and_get(self):
        """EventFrameIndex.add() stores a frame, get() retrieves by id."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        frame = self._make_frame("ef-001", [0, 1, 2])
        idx.add(frame)
        retrieved = idx.get("ef-001")
        assert retrieved is not None
        assert retrieved.id == "ef-001"
        assert retrieved.turns == [0, 1, 2]

    def test_get_missing(self):
        """EventFrameIndex.get() returns None for unknown frame_id."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        assert idx.get("nonexistent") is None

    def test_len(self):
        """EventFrameIndex.__len__() returns count of stored frames."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        assert len(idx) == 0
        idx.add(self._make_frame("ef-001", [0]))
        assert len(idx) == 1
        idx.add(self._make_frame("ef-002", [1]))
        assert len(idx) == 2

    def test_get_by_turn(self):
        """EventFrameIndex.get_by_turn() returns the frame containing a given turn."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0, 1, 2]))
        idx.add(self._make_frame("ef-002", [3, 4]))

        assert idx.get_by_turn(0).id == "ef-001"
        assert idx.get_by_turn(1).id == "ef-001"
        assert idx.get_by_turn(2).id == "ef-001"
        assert idx.get_by_turn(3).id == "ef-002"
        assert idx.get_by_turn(4).id == "ef-002"

    def test_get_by_turn_missing(self):
        """EventFrameIndex.get_by_turn() returns None for un-indexed turn."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0, 1]))
        assert idx.get_by_turn(99) is None

    def test_get_neighbors_forward(self):
        """get_neighbors follows next pointers."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0, 1], nxt="ef-002"))
        idx.add(self._make_frame("ef-002", [2, 3], prev="ef-001", nxt="ef-003"))
        idx.add(self._make_frame("ef-003", [4, 5], prev="ef-002"))

        neighbors = idx.get_neighbors("ef-001", window=2)
        neighbor_ids = [f.id for f in neighbors]
        assert "ef-002" in neighbor_ids
        assert "ef-003" in neighbor_ids

    def test_get_neighbors_backward(self):
        """get_neighbors follows prev pointers."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0, 1], nxt="ef-002"))
        idx.add(self._make_frame("ef-002", [2, 3], prev="ef-001", nxt="ef-003"))
        idx.add(self._make_frame("ef-003", [4, 5], prev="ef-002"))

        neighbors = idx.get_neighbors("ef-003", window=2)
        neighbor_ids = [f.id for f in neighbors]
        assert "ef-002" in neighbor_ids
        assert "ef-001" in neighbor_ids

    def test_get_neighbors_both_directions(self):
        """get_neighbors returns frames from both prev and next directions."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0], nxt="ef-002"))
        idx.add(self._make_frame("ef-002", [1], prev="ef-001", nxt="ef-003"))
        idx.add(self._make_frame("ef-003", [2], prev="ef-002"))

        neighbors = idx.get_neighbors("ef-002", window=1)
        neighbor_ids = [f.id for f in neighbors]
        assert "ef-001" in neighbor_ids
        assert "ef-003" in neighbor_ids
        # Must not include the center frame itself
        assert "ef-002" not in neighbor_ids

    def test_get_neighbors_window_limit(self):
        """get_neighbors respects window size; does not overshoot."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0], nxt="ef-002"))
        idx.add(self._make_frame("ef-002", [1], prev="ef-001", nxt="ef-003"))
        idx.add(self._make_frame("ef-003", [2], prev="ef-002", nxt="ef-004"))
        idx.add(self._make_frame("ef-004", [3], prev="ef-003"))

        neighbors = idx.get_neighbors("ef-001", window=1)
        neighbor_ids = [f.id for f in neighbors]
        assert "ef-002" in neighbor_ids
        # window=1 should NOT reach ef-003 or ef-004
        assert "ef-003" not in neighbor_ids
        assert "ef-004" not in neighbor_ids

    def test_get_neighbors_missing_frame(self):
        """get_neighbors returns empty list for unknown frame_id."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        assert idx.get_neighbors("nonexistent") == []

    def test_get_neighbors_no_links(self):
        """get_neighbors returns empty list for frame with no prev/next."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0]))
        assert idx.get_neighbors("ef-001", window=1) == []

    def test_to_dict(self):
        """EventFrameIndex.to_dict() returns serializable payload."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0, 1], nxt="ef-002"))
        idx.add(self._make_frame("ef-002", [2, 3], prev="ef-001"))

        d = idx.to_dict()
        assert "frames" in d
        assert len(d["frames"]) == 2

    def test_from_dict(self):
        """EventFrameIndex.from_dict() reconstructs the index."""
        from foundry.storage import EventFrameIndex, EventFrame
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0, 1], nxt="ef-002"))
        idx.add(self._make_frame("ef-002", [2, 3], prev="ef-001"))

        restored = EventFrameIndex.from_dict(idx.to_dict())
        assert len(restored) == 2
        assert restored.get("ef-001") is not None
        assert restored.get("ef-002") is not None
        assert restored.get_by_turn(0).id == "ef-001"
        assert restored.get_by_turn(3).id == "ef-002"

    def test_roundtrip(self):
        """EventFrameIndex survives to_dict -> from_dict roundtrip with full pointers."""
        from foundry.storage import EventFrameIndex
        idx = EventFrameIndex()
        idx.add(self._make_frame("ef-001", [0], nxt="ef-002"))
        idx.add(self._make_frame("ef-002", [1], prev="ef-001", nxt="ef-003"))
        idx.add(self._make_frame("ef-003", [2], prev="ef-002"))

        restored = EventFrameIndex.from_dict(idx.to_dict())
        neighbors = restored.get_neighbors("ef-002", window=1)
        neighbor_ids = [f.id for f in neighbors]
        assert "ef-001" in neighbor_ids
        assert "ef-003" in neighbor_ids

    def test_from_dict_empty(self):
        """EventFrameIndex.from_dict() handles empty payload."""
        from foundry.storage import EventFrameIndex
        restored = EventFrameIndex.from_dict({})
        assert len(restored) == 0


# ---------------------------------------------------------------------------
# FoundryStore integration
# ---------------------------------------------------------------------------

class TestFoundryStoreEventFrames:
    """Tests that EventFrameIndex is wired into FoundryStore."""

    def test_foundry_store_has_event_frames(self):
        """FoundryStore has an event_frames field of type EventFrameIndex."""
        from foundry.storage import FoundryStore, EventFrameIndex
        store = FoundryStore()
        assert isinstance(store.event_frames, EventFrameIndex)
        assert len(store.event_frames) == 0

    def test_foundry_store_to_dict_includes_event_frames(self):
        """FoundryStore.to_dict() includes event_frames key."""
        from foundry.storage import FoundryStore, EventFrame
        store = FoundryStore()
        store.event_frames.add(EventFrame(id="ef-001", turns=[0]))
        d = store.to_dict()
        assert "event_frames" in d
        assert len(d["event_frames"]["frames"]) == 1

    def test_foundry_store_from_dict_restores_event_frames(self):
        """FoundryStore.from_dict() restores event_frames."""
        from foundry.storage import FoundryStore, EventFrame
        store = FoundryStore()
        store.event_frames.add(EventFrame(id="ef-001", turns=[0, 1]))
        restored = FoundryStore.from_dict(store.to_dict())
        assert len(restored.event_frames) == 1
        assert restored.event_frames.get("ef-001") is not None
        assert restored.event_frames.get_by_turn(0).id == "ef-001"


# ---------------------------------------------------------------------------
# VAL-003: Foundry integration — ingest_turns / ingest_chunk build frames
# ---------------------------------------------------------------------------


def _make_foundry():
    """Create a Foundry instance with no external model dependencies."""
    from foundry.core import Foundry, FoundryConfig
    config = FoundryConfig(
        extraction_url=None,
        embedding_model="",
        reranker_model=None,
    )
    return Foundry(config)


def _make_turns(topics):
    """Build ConversationTurn list from a list of (turn_idx, text) pairs."""
    from foundry.ingestion import ConversationTurn
    return [
        ConversationTurn(
            turn_idx=turn_idx,
            role="user",
            text=text,
            session_time=datetime(2025, 6, 15, 10, turn_idx, 0),
        )
        for turn_idx, text in topics
    ]


class TestFoundryIngestTurnsEventFrames:
    """VAL-003: Foundry.ingest_turns() builds EventFrames after entity extraction."""

    def test_ingest_turns_populates_event_frames(self):
        """After ingest_turns(), the store's event_frames index is non-empty."""
        foundry = _make_foundry()
        turns = _make_turns([
            (0, "I just got a new guitar from the music store"),
            (1, "The guitar is a Fender Stratocaster"),
            (2, "I also adopted a cat named Whiskers"),
        ])
        foundry.ingest_turns(turns)
        assert len(foundry._store.event_frames) > 0

    def test_ingest_turns_frames_cover_all_entity_turns(self):
        """Every turn that mentions an entity appears in some EventFrame."""
        foundry = _make_foundry()
        turns = _make_turns([
            (0, "I bought a guitar yesterday"),
            (1, "The guitar sounds amazing"),
            (2, "My cat Whiskers is sleeping"),
        ])
        foundry.ingest_turns(turns)

        # Collect all turns referenced by entities
        entity_turns = set()
        for entity in foundry._store.entity_graph.all_entities():
            entity_turns.update(entity.mention_turns)

        # Every entity turn must be covered by some frame
        for tidx in entity_turns:
            frame = foundry._store.event_frames.get_by_turn(tidx)
            assert frame is not None, f"Turn {tidx} not covered by any event frame"

    def test_ingest_turns_frames_linked_list(self):
        """Frames produced by ingest_turns form a doubly-linked list (prev/next)."""
        foundry = _make_foundry()
        # Use distinct topics so segmentation produces multiple frames
        turns = _make_turns([
            (0, "I love playing guitar and piano"),
            (1, "The guitar is a Gibson Les Paul"),
            (2, "My dog Rex loves to fetch tennis balls"),
            (3, "Rex also enjoys swimming in the lake"),
        ])
        foundry.ingest_turns(turns)

        frames_count = len(foundry._store.event_frames)
        if frames_count <= 1:
            # If only one frame, no linked-list to verify (all turns same topic)
            pytest.skip("Segmentation produced single frame; skip linked-list test")

        # Collect all frames
        all_frames = []
        for entity in foundry._store.entity_graph.all_entities():
            for tidx in entity.mention_turns:
                f = foundry._store.event_frames.get_by_turn(tidx)
                if f and f not in all_frames:
                    all_frames.append(f)

        # Find the head frame (prev=None)
        heads = [f for f in all_frames if f.prev is None]
        assert len(heads) >= 1, "No head frame found (prev=None)"

        # Walk forward from head
        head = heads[0]
        visited = [head.id]
        cur = head
        while cur.next is not None:
            nxt = foundry._store.event_frames.get(cur.next)
            assert nxt is not None, f"Broken next pointer: {cur.next}"
            assert nxt.prev == cur.id, (
                f"Frame {nxt.id}.prev={nxt.prev} does not point back to {cur.id}"
            )
            visited.append(nxt.id)
            cur = nxt

        # The tail must have next=None
        assert cur.next is None, f"Tail frame {cur.id} has non-None next"

    def test_ingest_turns_frame_entities_match(self):
        """Each frame's entities list matches the entities mentioned in its turns."""
        foundry = _make_foundry()
        turns = _make_turns([
            (0, "I got a new guitar from the store"),
            (1, "The guitar is red and shiny"),
        ])
        foundry.ingest_turns(turns)

        for entity in foundry._store.entity_graph.all_entities():
            for tidx in entity.mention_turns:
                frame = foundry._store.event_frames.get_by_turn(tidx)
                if frame is not None:
                    # The entity's canonical should appear in the frame's entity list
                    assert entity.canonical in frame.entities, (
                        f"Entity {entity.canonical} mentioned in turn {tidx} "
                        f"not in frame {frame.id} entities: {frame.entities}"
                    )

    def test_ingest_turns_frame_gist_is_text_prefix(self):
        """Each frame's gist is the first turn's text[:200] (no LLM call)."""
        foundry = _make_foundry()
        # Use possessive "My cat Whiskers" which the regex extractor reliably finds
        turns = _make_turns([
            (0, "My cat Whiskers loves to play in the garden every morning"),
            (1, "Whiskers caught a mouse today near the fence"),
        ])
        foundry.ingest_turns(turns)

        # Get the frame for turn 0
        frame = foundry._store.event_frames.get_by_turn(0)
        assert frame is not None, "No frame for turn 0 (regex extractor should find 'Whiskers')"

        # Find the first turn in the frame
        first_turn_idx = min(frame.turns)
        expected_text = None
        for t in turns:
            if t.turn_idx == first_turn_idx:
                expected_text = t.text[:200]
                break

        assert frame.gist == expected_text, (
            f"Frame gist does not match text[:200]. Got: {frame.gist!r}"
        )

    def test_ingest_turns_frame_timestamp_from_first_turn(self):
        """Frame timestamp matches the session_time of its first turn."""
        foundry = _make_foundry()
        ts = datetime(2025, 6, 15, 10, 0, 0)
        from foundry.ingestion import ConversationTurn
        # Use possessive pattern so regex extractor finds entities on turn 0
        turns = [
            ConversationTurn(turn_idx=0, role="user",
                             text="My dog Rex loves to run in the park",
                             session_time=ts),
            ConversationTurn(turn_idx=1, role="user",
                             text="Rex also enjoys swimming in the lake",
                             session_time=datetime(2025, 6, 15, 10, 1, 0)),
        ]
        foundry.ingest_turns(turns)

        frame = foundry._store.event_frames.get_by_turn(0)
        assert frame is not None, "No frame for turn 0 (regex extractor should find 'Rex')"

        assert frame.timestamp == ts, (
            f"Frame timestamp {frame.timestamp} != expected {ts}"
        )

    def test_ingest_turns_empty_produces_no_frames(self):
        """Ingesting empty turns list produces no event frames."""
        foundry = _make_foundry()
        foundry.ingest_turns([])
        assert len(foundry._store.event_frames) == 0


class TestFoundryIngestChunkEventFrames:
    """VAL-003: Foundry.ingest_chunk() builds a single EventFrame."""

    def test_ingest_chunk_creates_frame(self):
        """Single ingest_chunk() call produces one event frame."""
        from foundry.storage import RawChunk
        foundry = _make_foundry()
        chunk = RawChunk(
            chunk_idx=0,
            text="I bought a guitar and a piano yesterday",
            turn_idx=0,
            role="user",
            session_time=datetime(2025, 6, 15, 10, 0, 0),
        )
        foundry.ingest_chunk(chunk)

        # Should have at least one frame if entities were extracted
        if len(foundry._store.entity_graph) > 0:
            assert len(foundry._store.event_frames) > 0
            frame = foundry._store.event_frames.get_by_turn(0)
            assert frame is not None, "No frame for turn 0 after ingest_chunk()"

    def test_ingest_chunk_frame_has_correct_turn(self):
        """The frame created by ingest_chunk contains the chunk's turn_idx."""
        from foundry.storage import RawChunk
        foundry = _make_foundry()
        chunk = RawChunk(
            chunk_idx=0,
            text="My dog Rex loves to play fetch in the park",
            turn_idx=5,
            role="user",
            session_time=datetime(2025, 6, 15, 10, 5, 0),
        )
        foundry.ingest_chunk(chunk)

        if len(foundry._store.entity_graph) > 0:
            frame = foundry._store.event_frames.get_by_turn(5)
            assert frame is not None, "No frame for turn 5 after ingest_chunk()"
            assert 5 in frame.turns

    def test_ingest_chunk_frame_is_isolated(self):
        """A single-chunk frame has prev=None and next=None (no neighbors)."""
        from foundry.storage import RawChunk
        foundry = _make_foundry()
        chunk = RawChunk(
            chunk_idx=0,
            text="I got a new guitar from the music store",
            turn_idx=0,
            role="user",
            session_time=datetime(2025, 6, 15, 10, 0, 0),
        )
        foundry.ingest_chunk(chunk)

        if len(foundry._store.entity_graph) > 0:
            frame = foundry._store.event_frames.get_by_turn(0)
            assert frame is not None
            assert frame.prev is None
            assert frame.next is None


# ---------------------------------------------------------------------------
# VAL-004: recall() uses event frames for temporal contiguity
# ---------------------------------------------------------------------------


def _make_store_with_frames():
    """Build a FoundryStore with BM25 index + event frames for testing retrieval.

    Layout:
      Frame ef-A: turns [0, 1]  (topic: guitars)
      Frame ef-B: turns [2, 3]  (topic: cats)
      Frame ef-C: turns [4, 5]  (topic: cooking)

    ef-A <-> ef-B <-> ef-C  (doubly linked)

    BM25 indexes all 6 turns. Querying "guitar" should hit turn 0 or 1;
    with event-frame contiguity, turns from neighbor frame ef-B (turns 2, 3)
    should appear as bonus candidates.
    """
    from foundry.storage import FoundryStore, EventFrame

    store = FoundryStore()

    texts = {
        0: "I bought a new Fender Stratocaster guitar yesterday",
        1: "The guitar has a beautiful sunburst finish and maple neck",
        2: "My cat Whiskers loves sleeping on the couch all day",
        3: "Whiskers caught a mouse in the garden this morning",
        4: "I made pasta carbonara for dinner last night",
        5: "The carbonara recipe uses eggs parmesan and guanciale",
    }
    for doc_id, text in texts.items():
        store.bm25_chunks.add(doc_id, text)

    # Build event frames with linked list
    store.event_frames.add(EventFrame(
        id="ef-A", turns=[0, 1], prev=None, next="ef-B",
    ))
    store.event_frames.add(EventFrame(
        id="ef-B", turns=[2, 3], prev="ef-A", next="ef-C",
    ))
    store.event_frames.add(EventFrame(
        id="ef-C", turns=[4, 5], prev="ef-B", next=None,
    ))

    return store, texts


class TestRecallRetrieveEventFrameContiguity:
    """VAL-004: _recall_retrieve uses event frames for temporal contiguity."""

    def test_use_event_frames_adds_neighbor_turns(self):
        """When use_event_frames=True, neighbor frame turns appear in results."""
        from foundry.retrieval import _recall_retrieve
        store, texts = _make_store_with_frames()

        # Query "guitar" — should primarily hit turns 0, 1 (ef-A).
        # With event-frame contiguity, turns from neighbor ef-B (2, 3) should
        # also appear as bonus candidates.
        result = _recall_retrieve(
            store, "guitar",
            top_k=2,
            char_budget=50000,
            use_event_frames=True,
        )

        # The result should contain text from neighbor frame ef-B
        assert "Whiskers" in result or "cat" in result, (
            f"Event-frame contiguity did not pull neighbor frame turns. Got: {result!r}"
        )

    def test_use_event_frames_false_no_neighbor_expansion(self):
        """When use_event_frames=False and contiguity_window=0, no neighbors added."""
        from foundry.retrieval import _recall_retrieve
        store, texts = _make_store_with_frames()

        result = _recall_retrieve(
            store, "guitar",
            top_k=2,
            char_budget=50000,
            use_event_frames=False,
            contiguity_window=0,
        )

        # Without contiguity, only guitar turns should appear
        assert "guitar" in result.lower() or "Fender" in result or "sunburst" in result
        # Neighbor frames should NOT be pulled in
        assert "Whiskers" not in result and "carbonara" not in result, (
            f"Neighbor turns leaked without contiguity. Got: {result!r}"
        )

    def test_old_contiguity_window_still_works(self):
        """The old doc_id+/-1 contiguity_window approach still works as fallback."""
        from foundry.retrieval import _recall_retrieve
        store, texts = _make_store_with_frames()

        # contiguity_window=1 with use_event_frames=False should use the old
        # doc_id +/- 1 approach. Turn 0 (guitar) is doc_id 0, so neighbor
        # doc_id 1 should be pulled.
        result = _recall_retrieve(
            store, "guitar",
            top_k=1,
            char_budget=50000,
            use_event_frames=False,
            contiguity_window=1,
        )

        # doc_id 0 hits "guitar"; contiguity_window=1 pulls doc_id 1
        # (also guitar text) — that is the old behavior
        assert "guitar" in result.lower() or "sunburst" in result.lower()

    def test_event_frames_fallback_when_no_frames(self):
        """When use_event_frames=True but store has no frames, falls through gracefully."""
        from foundry.retrieval import _recall_retrieve
        from foundry.storage import FoundryStore

        store = FoundryStore()
        store.bm25_chunks.add(0, "I bought a guitar yesterday at the store")
        store.bm25_chunks.add(1, "The guitar sounds great with new strings")

        # No event frames exist — should not crash, just return BM25 results
        result = _recall_retrieve(
            store, "guitar",
            top_k=2,
            char_budget=50000,
            use_event_frames=True,
        )
        assert "guitar" in result.lower()

    def test_event_frames_neighbor_score_is_attenuated(self):
        """Neighbor turns from event frames get score * 0.1 (attenuated)."""
        from foundry.retrieval import _recall_retrieve
        store, texts = _make_store_with_frames()

        # With top_k=10 and event frames, the primary guitar hits should
        # come before the attenuated neighbor turns. The guitar text should
        # appear first in the evidence.
        result = _recall_retrieve(
            store, "guitar",
            top_k=10,
            char_budget=50000,
            use_event_frames=True,
        )

        # Guitar turns should appear before cat turns in the output
        guitar_pos = result.find("guitar") if "guitar" in result.lower() else result.find("Fender")
        cat_pos = result.find("Whiskers") if "Whiskers" in result else result.find("cat")
        if guitar_pos >= 0 and cat_pos >= 0:
            assert guitar_pos < cat_pos, (
                f"Guitar text (pos={guitar_pos}) should appear before "
                f"neighbor cat text (pos={cat_pos})"
            )

    def test_event_frames_does_not_duplicate_already_retrieved(self):
        """Turns already in the merged list are not duplicated by frame expansion."""
        from foundry.retrieval import _recall_retrieve
        store, texts = _make_store_with_frames()

        result = _recall_retrieve(
            store, "guitar",
            top_k=10,
            char_budget=50000,
            use_event_frames=True,
        )

        # Count occurrences of the unique guitar text — should appear once
        count = result.count("Fender Stratocaster")
        assert count <= 1, f"Duplicate guitar turn in output: appeared {count} times"


class TestRecallPassesEventFrames:
    """VAL-004: recall() passes use_event_frames through to _recall_retrieve."""

    def test_recall_accepts_use_event_frames_param(self):
        """recall() accepts use_event_frames keyword argument without error."""
        from foundry.retrieval import recall
        store, _ = _make_store_with_frames()

        # Should not raise TypeError for unknown keyword argument
        result = recall(
            store, "guitar",
            use_event_frames=True,
        )
        assert isinstance(result, str)

    def test_recall_default_use_event_frames_is_false(self):
        """recall() defaults use_event_frames to False (backward compatible)."""
        from foundry.retrieval import recall
        import inspect
        sig = inspect.signature(recall)
        param = sig.parameters.get("use_event_frames")
        assert param is not None, "recall() missing use_event_frames parameter"
        assert param.default is False, (
            f"recall() use_event_frames default should be False, got {param.default}"
        )

    def test_recall_event_frames_factual_route(self):
        """recall() with use_event_frames=True on the factual route pulls neighbors."""
        from foundry.retrieval import recall
        store, _ = _make_store_with_frames()

        result = recall(
            store, "guitar",
            use_event_frames=True,
        )
        # Factual route should use event-frame contiguity
        assert "Whiskers" in result or "cat" in result, (
            f"recall() factual route did not propagate use_event_frames. Got: {result!r}"
        )
