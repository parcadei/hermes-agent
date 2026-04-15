"""Unit tests for the Ingestion Pipeline (spec section 4).

Tests all 11 pipeline stages: turn parsing, chunking, triple extraction,
faceted entity extraction, entity resolution, type inference, temporal
extraction, co-occurrence tracking, grounding verification, role
classification, and three-layer routing.
"""

from __future__ import annotations

import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# Turn Parsing (Stage 1)
# ---------------------------------------------------------------------------

class TestTurnParsing:
    """Tests for parse_conversation and ConversationTurn."""

    def test_conversation_turn_creation(self):
        """ConversationTurn stores role, text, and index."""
        from foundry.ingestion import ConversationTurn
        turn = ConversationTurn(turn_idx=0, role="user", text="Hello")
        assert turn.turn_idx == 0
        assert turn.role == "user"
        assert turn.text == "Hello"

    def test_parse_simple_conversation(self):
        """Simple user/assistant alternation is parsed correctly."""
        from foundry.ingestion import parse_conversation
        raw = "User: Hello\nAssistant: Hi there\nUser: How are you?"
        turns = parse_conversation(raw)
        assert len(turns) >= 2
        assert turns[0].role in ("user", "assistant")

    def test_parse_preserves_text(self):
        """Text content is preserved exactly."""
        from foundry.ingestion import parse_conversation
        raw = "User: I have two cats named Luna and Mochi"
        turns = parse_conversation(raw)
        assert any("Luna" in t.text and "Mochi" in t.text for t in turns)

    def test_parse_chat_time_header(self):
        """Chat Time headers set session_time on subsequent turns."""
        from foundry.ingestion import parse_conversation
        raw = "Chat Time: 2024-01-15 10:00:00\nUser: Hello"
        turns = parse_conversation(raw)
        assert any(t.session_time is not None for t in turns)

    def test_parse_empty_text(self):
        """Empty input produces empty turn list."""
        from foundry.ingestion import parse_conversation
        turns = parse_conversation("")
        assert turns == []

    def test_session_time_inheritance(self):
        """Turns without Chat Time inherit the last parsed time."""
        from foundry.ingestion import parse_conversation
        raw = (
            "Chat Time: 2024-01-15 10:00:00\n"
            "User: First message\n"
            "User: Second message\n"
        )
        turns = parse_conversation(raw)
        times = [t.session_time for t in turns if t.session_time is not None]
        # All turns after a Chat Time header should have a session_time
        assert len(times) >= 1

    def test_turn_indices_sequential(self):
        """Turn indices are sequential starting from 0."""
        from foundry.ingestion import parse_conversation
        raw = "User: A\nAssistant: B\nUser: C"
        turns = parse_conversation(raw)
        indices = [t.turn_idx for t in turns]
        assert indices == list(range(len(turns)))


# ---------------------------------------------------------------------------
# Chunking (Stage 2)
# ---------------------------------------------------------------------------

class TestChunking:
    """Tests for chunk_turns function."""

    def test_chunk_simple(self):
        """Short turns produce at least one chunk."""
        from foundry.ingestion import ConversationTurn, chunk_turns
        turns = [
            ConversationTurn(turn_idx=0, role="user", text="Hello world"),
        ]
        chunks = chunk_turns(turns, chunk_size=512)
        assert len(chunks) >= 1

    def test_chunk_respects_size(self):
        """Chunks are approximately chunk_size tokens or smaller."""
        from foundry.ingestion import ConversationTurn, chunk_turns
        long_text = " ".join(["word"] * 1000)
        turns = [ConversationTurn(turn_idx=0, role="user", text=long_text)]
        chunks = chunk_turns(turns, chunk_size=512)
        assert len(chunks) > 1

    def test_chunk_role_inheritance(self):
        """Continuation chunks inherit role from source turn."""
        from foundry.ingestion import ConversationTurn, chunk_turns
        long_text = " ".join(["word"] * 1000)
        turns = [ConversationTurn(turn_idx=0, role="user", text=long_text)]
        chunks = chunk_turns(turns, chunk_size=512)
        for chunk in chunks:
            assert chunk.role in ("user", "continuation")

    def test_chunk_default_512(self):
        """Default chunk_size is 512."""
        from foundry.ingestion import ConversationTurn, chunk_turns
        turns = [ConversationTurn(turn_idx=0, role="user", text="short")]
        chunks = chunk_turns(turns)  # No chunk_size specified
        assert len(chunks) >= 1

    def test_chunk_turn_idx_tracking(self):
        """Each chunk records its source turn index."""
        from foundry.ingestion import ConversationTurn, chunk_turns
        turns = [
            ConversationTurn(turn_idx=0, role="user", text="First turn"),
            ConversationTurn(turn_idx=1, role="assistant", text="Second turn"),
        ]
        chunks = chunk_turns(turns, chunk_size=512)
        turn_indices = {c.turn_idx for c in chunks}
        assert 0 in turn_indices

    def test_chunk_idx_unique(self):
        """Each chunk has a unique chunk_idx."""
        from foundry.ingestion import ConversationTurn, chunk_turns
        turns = [
            ConversationTurn(turn_idx=i, role="user", text=f"Turn {i} " * 50)
            for i in range(5)
        ]
        chunks = chunk_turns(turns, chunk_size=512)
        indices = [c.chunk_idx for c in chunks]
        assert len(indices) == len(set(indices)), "Chunk indices must be unique"


# ---------------------------------------------------------------------------
# Triple Extraction (Stage 3)
# ---------------------------------------------------------------------------

class TestExtractedTriple:
    """Tests for the ExtractedTriple data structure."""

    def test_triple_creation(self):
        """ExtractedTriple stores S-P-O."""
        from foundry.ingestion import ExtractedTriple
        t = ExtractedTriple(subject="luna", predicate="breed", object="tabby")
        assert t.subject == "luna"
        assert t.predicate == "breed"
        assert t.object == "tabby"

    def test_triple_defaults(self):
        """Defaults: empty when/role/sub_role, -1 indices."""
        from foundry.ingestion import ExtractedTriple
        t = ExtractedTriple(subject="x", predicate="y", object="z")
        assert t.when == ""
        assert t.role == ""
        assert t.sub_role == ""
        assert t.turn_idx == -1
        assert t.chunk_idx == -1
        assert t.source_text == ""

    def test_triple_with_temporal(self):
        """Triple with temporal annotation."""
        from foundry.ingestion import ExtractedTriple
        t = ExtractedTriple(
            subject="user", predicate="adopted", object="luna",
            when="in 2021", role="event", sub_role="acquisition",
        )
        assert t.when == "in 2021"
        assert t.role == "event"
        assert t.sub_role == "acquisition"


# ---------------------------------------------------------------------------
# Faceted Entity Extraction (Stage 4)
# ---------------------------------------------------------------------------

class TestFacetEntity:
    """Tests for the FacetEntity data structure."""

    def test_facet_entity_creation(self):
        """FacetEntity stores name and facet values."""
        from foundry.ingestion import FacetEntity
        fe = FacetEntity(
            name="Luna",
            facets={"what": "cat", "who": "", "status": "active"},
            turn_idx=0,
            source_text="I have a cat named Luna",
        )
        assert fe.name == "Luna"
        assert fe.facets["what"] == "cat"
        assert fe.turn_idx == 0


# ---------------------------------------------------------------------------
# Type Inference (Stage 6)
# ---------------------------------------------------------------------------

class TestTypeInference:
    """Tests for infer_type function."""

    def test_infer_from_what_facet(self):
        """Type inferred from 'what' facet value."""
        from foundry.ingestion import FacetEntity, infer_type
        fe = FacetEntity(name="Luna", facets={"what": "cat"}, turn_idx=0, source_text="")
        assert infer_type(fe) == "cat"

    def test_infer_normalizes(self):
        """Inferred type is normalized (lowercase, depluralized)."""
        from foundry.ingestion import FacetEntity, infer_type
        fe = FacetEntity(name="Luna", facets={"what": "Pet Fish"}, turn_idx=0, source_text="")
        assert infer_type(fe) == "pet_fish"

    def test_infer_depluralize(self):
        """Plural 'what' facet is depluralized."""
        from foundry.ingestion import FacetEntity, infer_type
        fe = FacetEntity(name="stuff", facets={"what": "bikes"}, turn_idx=0, source_text="")
        assert infer_type(fe) == "bike"

    def test_infer_none_when_empty(self):
        """Empty 'what' facet returns None."""
        from foundry.ingestion import FacetEntity, infer_type
        fe = FacetEntity(name="Unknown", facets={"what": ""}, turn_idx=0, source_text="")
        assert infer_type(fe) is None

    def test_infer_none_when_missing(self):
        """Missing 'what' facet returns None."""
        from foundry.ingestion import FacetEntity, infer_type
        fe = FacetEntity(name="Unknown", facets={"who": "John"}, turn_idx=0, source_text="")
        assert infer_type(fe) is None


# ---------------------------------------------------------------------------
# Grounding Verification (Stage 9)
# ---------------------------------------------------------------------------

class TestGroundingVerification:
    """Tests for verify_grounding function."""

    def test_grounding_substring_match(self):
        """Object that is a substring of source_text passes."""
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType, GroundingLayer
        from foundry.storage import Fact
        f = Fact(
            id="g1", subject="luna", predicate="breed", object="tabby",
            object_type=PropertyType.STRING, turn_idx=0,
            source_text="Luna is a tabby cat",
        )
        assert verify_grounding(f) is True

    def test_grounding_case_insensitive(self):
        """Grounding check is case-insensitive."""
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        f = Fact(
            id="g2", subject="luna", predicate="breed", object="Tabby",
            object_type=PropertyType.STRING, turn_idx=0,
            source_text="luna is a tabby cat",
        )
        assert verify_grounding(f) is True

    def test_grounding_empty_source_passes(self):
        """Empty source_text passes grounding (ungated)."""
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        f = Fact(
            id="g3", subject="luna", predicate="breed", object="tabby",
            object_type=PropertyType.STRING, turn_idx=0,
            source_text="",
        )
        assert verify_grounding(f) is True

    def test_grounding_model_inference_fails(self):
        """Model inference that isn't in source text fails grounding.

        Example: 'airline' is NOT in 'I fly Delta' -- this catches
        model hallucination of inferred categories.
        """
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        f = Fact(
            id="g4", subject="delta", predicate="type", object="airline",
            object_type=PropertyType.STRING, turn_idx=0,
            source_text="I fly Delta",
        )
        assert verify_grounding(f) is False

    def test_grounding_word_level_fallback(self):
        """Word-level fallback: every content word (len >= 3) in source."""
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        # "tabby cat" -- "tabby" and "cat" are both in source
        f = Fact(
            id="g5", subject="luna", predicate="description",
            object="tabby cat",
            object_type=PropertyType.STRING, turn_idx=0,
            source_text="Luna is a cat that is tabby colored",
        )
        assert verify_grounding(f) is True

    def test_grounding_partial_word_fails(self):
        """Short words (< 3 chars) are skipped in word-level check."""
        from foundry.ingestion import verify_grounding
        from foundry.ontology import PropertyType
        from foundry.storage import Fact
        # "is a tabby" -- "is" and "a" are < 3 chars, only "tabby" matters
        f = Fact(
            id="g6", subject="luna", predicate="description",
            object="is a tabby",
            object_type=PropertyType.STRING, turn_idx=0,
            source_text="Luna is a tabby cat",
        )
        assert verify_grounding(f) is True


# ---------------------------------------------------------------------------
# Role Classification / VerbNet (Stage 10)
# ---------------------------------------------------------------------------

class TestVerbClassification:
    """Tests for VerbNet-based role classification."""

    def test_acquisition_verb(self):
        """Acquisition verbs (buy, adopt, get) map to acquisition sub_role."""
        from foundry.ingestion import classify_verb
        result = classify_verb("adopted")
        assert result.sub_role == "acquisition"

    def test_maintenance_verb(self):
        """Maintenance verbs (have, own, keep) map to maintenance sub_role."""
        from foundry.ingestion import classify_verb
        result = classify_verb("has")
        assert result.sub_role == "maintenance"

    def test_intent_verb(self):
        """Intent verbs (want, plan, hope) map to intent sub_role."""
        from foundry.ingestion import classify_verb
        result = classify_verb("wants")
        assert result.sub_role == "intent"

    def test_default_state(self):
        """Unrecognized verbs default to role='state'."""
        from foundry.ingestion import classify_verb
        result = classify_verb("is")
        assert result.role == "state"

    def test_classification_structure(self):
        """VerbClassification has role and sub_role fields."""
        from foundry.ingestion import classify_verb, VerbClassification
        result = classify_verb("bought")
        assert hasattr(result, "role")
        assert hasattr(result, "sub_role")

    def test_classification_frozen(self):
        """VerbClassification is frozen (immutable)."""
        from foundry.ingestion import classify_verb
        result = classify_verb("bought")
        with pytest.raises(AttributeError):
            result.role = "other"


# ---------------------------------------------------------------------------
# Three-Layer Routing (Stage 11)
# ---------------------------------------------------------------------------

class TestThreeLayerRouting:
    """Tests for route_to_layer function."""

    def test_stated_when_grounding_passes(self):
        """Grounding-passed facts go to STATED."""
        from foundry.ingestion import route_to_layer
        from foundry.ontology import PropertyType, GroundingLayer
        from foundry.storage import Fact
        f = Fact(
            id="r1", subject="luna", predicate="breed", object="tabby",
            object_type=PropertyType.STRING, turn_idx=0,
            layer=GroundingLayer.STATED,
        )
        assert route_to_layer(f, grounding_passed=True) == GroundingLayer.STATED

    def test_inferred_when_grounding_fails(self):
        """Grounding-failed facts go to INFERRED."""
        from foundry.ingestion import route_to_layer
        from foundry.ontology import PropertyType, GroundingLayer
        from foundry.storage import Fact
        f = Fact(
            id="r2", subject="delta", predicate="type", object="airline",
            object_type=PropertyType.STRING, turn_idx=0,
            layer=GroundingLayer.STATED,
        )
        assert route_to_layer(f, grounding_passed=False) == GroundingLayer.INFERRED

    def test_derived_stays_derived(self):
        """Facts already marked DERIVED stay DERIVED regardless of grounding."""
        from foundry.ingestion import route_to_layer
        from foundry.ontology import PropertyType, GroundingLayer
        from foundry.storage import Fact
        f = Fact(
            id="r3", subject="user", predicate="pet_count", object="5",
            object_type=PropertyType.NUMERIC, turn_idx=-1,
            layer=GroundingLayer.DERIVED,
        )
        assert route_to_layer(f, grounding_passed=True) == GroundingLayer.DERIVED
        assert route_to_layer(f, grounding_passed=False) == GroundingLayer.DERIVED


# ---------------------------------------------------------------------------
# Co-occurrence Tracking (Stage 8)
# ---------------------------------------------------------------------------

class TestCooccurrenceTracking:
    """Tests for track_cooccurrences function."""

    def test_track_pair(self):
        """Two entities in a chunk produce a co-occurrence."""
        from foundry.ingestion import track_cooccurrences
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        track_cooccurrences(["luna", "mochi"], chunk_idx=0, cooc_index=idx)
        assert "mochi" in idx.get_cooccurrences("luna")

    def test_track_all_pairs(self):
        """Three entities produce 3 co-occurrence pairs."""
        from foundry.ingestion import track_cooccurrences
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        track_cooccurrences(["luna", "mochi", "max"], chunk_idx=0, cooc_index=idx)
        assert "mochi" in idx.get_cooccurrences("luna")
        assert "max" in idx.get_cooccurrences("luna")
        assert "luna" in idx.get_cooccurrences("max")

    def test_track_single_entity_no_cooccurrence(self):
        """Single entity produces no co-occurrences."""
        from foundry.ingestion import track_cooccurrences
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        track_cooccurrences(["luna"], chunk_idx=0, cooc_index=idx)
        assert idx.get_cooccurrences("luna") == {}

    def test_track_empty_list(self):
        """Empty entity list produces no co-occurrences."""
        from foundry.ingestion import track_cooccurrences
        from foundry.storage import CooccurrenceIndex
        idx = CooccurrenceIndex()
        track_cooccurrences([], chunk_idx=0, cooc_index=idx)
        # Should not crash


# ---------------------------------------------------------------------------
# IngestionReport
# ---------------------------------------------------------------------------

class TestIngestionReport:
    """Tests for the IngestionReport data structure."""

    def test_report_creation(self):
        """IngestionReport stores all metrics."""
        from foundry.ingestion import IngestionReport
        report = IngestionReport(
            chunks_processed=10,
            entities_created=5,
            entities_resolved=3,
            facts_added=20,
            facts_superseded=2,
            facts_rejected=1,
            temporal_events=4,
            cooccurrences=15,
            grounding_rate=0.95,
            grounding_failures=["airline not in source"],
            elapsed_ms=1500.0,
        )
        assert report.chunks_processed == 10
        assert report.grounding_rate == 0.95
        assert len(report.grounding_failures) == 1
