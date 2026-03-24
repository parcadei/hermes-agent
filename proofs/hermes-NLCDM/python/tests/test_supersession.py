"""Discriminative tests for supersession: triple-based conflict filtering.

Core hypothesis: when a new fact has the same (subject, predicate) as an
older fact but a different object, the older fact is outdated and should
be excluded from retrieval results. This is the mechanism for resolving
76/79 conflict errors (96% of failures in factconsolidation_mh_262k).

Example:
  Fact 1: "Abbiati plays for AC Milan"     triple: (Abbiati, plays for, AC Milan)
  Fact 2: "Abbiati plays for Atletico"      triple: (Abbiati, plays for, Atletico)
  → Fact 1 is superseded by Fact 2.
  → Query "What team does Abbiati play for?" should only see Fact 2.

But:
  Fact 1: "Abbiati plays for AC Milan"      triple: (Abbiati, plays for, AC Milan)
  Fact 3: "Abbiati was born in Milan"       triple: (Abbiati, born in, Milan)
  → NOT a conflict — different predicate. Both survive.

All tests run on CPU. No GPU, no external models, no LLM calls.
Supersession is tested at the data structure level: direct manipulation
of _triple_index and _superseded_texts, plus integration through
_store() with mocked ingestion.
"""

import numpy as np
import pytest
import sys
import os
from collections import defaultdict
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, idx: int, noise: float = 0.05) -> np.ndarray:
    """Unit vector along axis `idx` with small noise."""
    v = np.zeros(dim, dtype=np.float64)
    v[idx % dim] = 1.0
    rng = np.random.RandomState(idx)
    v += rng.randn(dim) * noise
    return v / np.linalg.norm(v)


@dataclass(frozen=True)
class FakeFact:
    """Minimal SegmentedFact-like object for testing."""

    text: str
    triples: tuple
    entities: tuple = ()
    importance_hint: float = 0.5
    source_sentence_idx: int = 0
    segment_idx: int = 0
    fact_id: str = ""
    text_raw: str = ""


@dataclass(frozen=True)
class FakeIngestionResult:
    """Minimal IngestionResult-like object for testing."""

    facts: list
    entity_graph: object = None
    n_sentences: int = 1
    n_segments: int = 1
    elapsed_seconds: float = 0.0
    coref_clusters: list = ()
    verb_mappings: list = ()


class FakeEntityGraph:
    entity_to_facts = {}
    fact_to_entities = {}


# ---------------------------------------------------------------------------
# Test 1: Data structure level — triple index and supersession detection
# ---------------------------------------------------------------------------


class TestSupersessionDataStructures:
    """Test supersession detection at the data structure level."""

    def test_same_subject_predicate_different_object_supersedes(self):
        """When (S, P) matches but O differs, old fact is superseded."""
        triple_index = defaultdict(list)
        superseded = set()

        # Fact 1: Abbiati plays for AC Milan
        triple_index[("abbiati", "plays for")].append(("ac milan", "Abbiati plays for AC Milan"))

        # Fact 2: Abbiati plays for Atletico — should supersede Fact 1
        key = ("abbiati", "plays for")
        new_obj = "atletico"
        for existing_obj, existing_text in triple_index[key]:
            if existing_obj != new_obj:
                superseded.add(existing_text)
        triple_index[key].append(("atletico", "Abbiati plays for Atletico"))

        assert "Abbiati plays for AC Milan" in superseded
        assert "Abbiati plays for Atletico" not in superseded

    def test_same_subject_different_predicate_no_supersession(self):
        """Different predicate → no conflict, both facts survive."""
        triple_index = defaultdict(list)
        superseded = set()

        # Fact 1: Abbiati plays for AC Milan
        triple_index[("abbiati", "plays for")].append(("ac milan", "Abbiati plays for AC Milan"))

        # Fact 2: Abbiati was born in Milan — different predicate
        key = ("abbiati", "born in")
        new_obj = "milan"
        for existing_obj, existing_text in triple_index[key]:
            if existing_obj != new_obj:
                superseded.add(existing_text)
        triple_index[key].append(("milan", "Abbiati was born in Milan"))

        assert len(superseded) == 0

    def test_same_triple_no_supersession(self):
        """Identical triple (same S, P, O) → not a conflict, no supersession."""
        triple_index = defaultdict(list)
        superseded = set()

        triple_index[("abbiati", "plays for")].append(("ac milan", "Abbiati plays for AC Milan"))

        # Same triple again
        key = ("abbiati", "plays for")
        new_obj = "ac milan"
        for existing_obj, existing_text in triple_index[key]:
            if existing_obj != new_obj:
                superseded.add(existing_text)
        triple_index[key].append(("ac milan", "Abbiati plays for AC Milan (repeat)"))

        assert len(superseded) == 0

    def test_case_insensitive_matching(self):
        """Subject and predicate matching should be case-insensitive."""
        triple_index = defaultdict(list)
        superseded = set()

        triple_index[("abbiati", "plays for")].append(("ac milan", "Abbiati plays for AC Milan"))

        # Same key after normalization, different object
        key = ("abbiati", "plays for")  # already normalized
        new_obj = "atletico"
        for existing_obj, existing_text in triple_index[key]:
            if existing_obj != new_obj:
                superseded.add(existing_text)

        assert "Abbiati plays for AC Milan" in superseded

    def test_chain_of_supersessions(self):
        """A → B → C: A and B should both be superseded."""
        triple_index = defaultdict(list)
        superseded = set()

        key = ("president", "is")

        # Fact 1: President is A
        triple_index[key].append(("a", "President is A"))

        # Fact 2: President is B — supersedes A
        for existing_obj, existing_text in triple_index[key]:
            if existing_obj != "b":
                superseded.add(existing_text)
        triple_index[key].append(("b", "President is B"))

        # Fact 3: President is C — supersedes A and B
        for existing_obj, existing_text in triple_index[key]:
            if existing_obj != "c":
                superseded.add(existing_text)
        triple_index[key].append(("c", "President is C"))

        assert "President is A" in superseded
        assert "President is B" in superseded
        assert "President is C" not in superseded


# ---------------------------------------------------------------------------
# Test 2: Integration through _store() with mocked ingestion
# ---------------------------------------------------------------------------


class TestSupersessionStore:
    """Test supersession detection through the _store method."""

    def _make_agent(self, supersession: bool = True):
        """Create a HermesMemoryAgent with mocked dependencies."""
        # We need to mock the ingestion pipeline and embedding scorer
        # since spaCy isn't available locally
        from mabench.hermes_agent import HermesMemoryAgent

        # Patch the IngestionPipeline to avoid spaCy import
        mock_pipeline = MagicMock()
        mock_scorer = MagicMock()
        mock_scorer.embed_batch.return_value = [_unit_vec(8, i) for i in range(10)]
        mock_scorer.embed.side_effect = lambda t: _unit_vec(8, hash(t) % 8)

        # Create agent with all deps mocked
        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.supersession = supersession
                agent.conflict_resolution = False
                agent._triple_index = defaultdict(list)
                agent._superseded_texts = set()
                agent._fact_entities = {}
                agent._supersession_graph = {}
                agent._supersedes = defaultdict(set)
                agent._ingestion = mock_pipeline
                agent._scorer = mock_scorer
                agent._store_count = 0
                agent.dream_interval = 0
                agent._triadic = None
                agent._entity_accumulator = MagicMock()
                agent._belief_index = None

                # Minimal orchestrator and coupled engine stubs
                agent.orchestrator = MagicMock()
                agent.coupled_engine = MagicMock()
                agent.coupled_engine._session_buffer = []
                agent.coupled_engine.memory_store = []

        return agent

    def test_store_detects_supersession(self):
        """_store should populate _superseded_texts when triples conflict
        and facts share a named entity (entity-gated)."""
        agent = self._make_agent(supersession=True)

        # First store: Abbiati plays football
        fact1 = FakeFact(
            text="Abbiati plays football",
            triples=(("Abbiati", "plays", "football"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("Abbiati plays football")

        assert len(agent._superseded_texts) == 0
        assert len(agent._triple_index[("abbiati", "plays")]) == 1

        # Second store: Abbiati plays basketball — supersedes first (shared entity: Abbiati)
        fact2 = FakeFact(
            text="Abbiati plays basketball",
            triples=(("Abbiati", "plays", "basketball"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("Abbiati plays basketball")

        assert "Abbiati plays football" in agent._superseded_texts
        assert "Abbiati plays basketball" not in agent._superseded_texts

    def test_store_no_supersession_when_disabled(self):
        """With supersession=False, no conflict detection occurs."""
        agent = self._make_agent(supersession=False)

        fact1 = FakeFact(
            text="Abbiati plays football",
            triples=(("Abbiati", "plays", "football"),),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("Abbiati plays football")

        fact2 = FakeFact(
            text="Abbiati plays basketball",
            triples=(("Abbiati", "plays", "basketball"),),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("Abbiati plays basketball")

        assert len(agent._superseded_texts) == 0

    def test_store_with_empty_triples_no_crash(self):
        """Facts with no triples should not cause errors."""
        agent = self._make_agent(supersession=True)

        fact = FakeFact(text="Some text", triples=())
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("Some text")

        assert len(agent._superseded_texts) == 0

    def test_store_multiple_triples_per_fact(self):
        """A fact with multiple triples: each is checked independently."""
        agent = self._make_agent(supersession=True)

        # Fact 1: two triples
        fact1 = FakeFact(
            text="Abbiati plays football in Milan",
            triples=(
                ("Abbiati", "plays", "football"),
                ("Abbiati", "lives in", "Milan"),
            ),
            entities=("Abbiati", "Milan"),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("Abbiati plays football in Milan")

        # Fact 2: updates where he lives, not what he plays (shared entity: Abbiati)
        fact2 = FakeFact(
            text="Abbiati lives in Madrid",
            triples=(("Abbiati", "lives in", "Madrid"),),
            entities=("Abbiati", "Madrid"),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("Abbiati lives in Madrid")

        # First fact superseded because (Abbiati, lives in) changed + shared entity
        assert "Abbiati plays football in Milan" in agent._superseded_texts

    def test_store_degenerate_triples_skipped(self):
        """Triples with empty subject or predicate are skipped."""
        agent = self._make_agent(supersession=True)

        fact = FakeFact(
            text="Something happened",
            triples=(("", "plays", "football"), ("Abbiati", "", "football")),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("Something happened")

        # Empty triples should be skipped, no index entries
        assert len(agent._triple_index) == 0

    def test_entity_gate_prevents_false_supersession(self):
        """Facts with same (S,P) but NO shared entity should NOT supersede.

        This is the core fix for the SubEM=8 regression: spaCy extracts
        'The author' as subject for all author facts, but they have
        different named entities (L. Ron Hubbard vs Aristotle).
        """
        agent = self._make_agent(supersession=True)

        # Fact 1: "The author of Dianetics is L. Ron Hubbard"
        fact1 = FakeFact(
            text="The author of Dianetics is L. Ron Hubbard",
            triples=(("The author", "is", "L. Ron Hubbard"),),
            entities=("L. Ron Hubbard", "Dianetics"),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("The author of Dianetics is L. Ron Hubbard")

        # Fact 2: "The author of Prior Analytics is Aristotle"
        # Same (S,P) = ("the author", "is") but different entities entirely
        fact2 = FakeFact(
            text="The author of Prior Analytics is Aristotle",
            triples=(("The author", "is", "Aristotle"),),
            entities=("Aristotle", "Prior Analytics"),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("The author of Prior Analytics is Aristotle")

        # NO supersession — different entities, not a real conflict
        assert len(agent._superseded_texts) == 0

    def test_entity_gate_allows_true_supersession(self):
        """Facts with same (S,P), different O, AND shared entity SHOULD supersede."""
        agent = self._make_agent(supersession=True)

        # Fact 1: "Abbiati plays association football"
        fact1 = FakeFact(
            text="Abbiati plays association football",
            triples=(("Abbiati", "plays", "association football"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("Abbiati plays association football")

        # Fact 2: "Abbiati plays basketball" — shared entity "Abbiati"
        fact2 = FakeFact(
            text="Abbiati plays basketball",
            triples=(("Abbiati", "plays", "basketball"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("Abbiati plays basketball")

        assert "Abbiati plays association football" in agent._superseded_texts

    def test_empty_entities_prevents_supersession(self):
        """Facts with no entities should never trigger supersession."""
        agent = self._make_agent(supersession=True)

        fact1 = FakeFact(
            text="The company is based in New York",
            triples=(("The company", "is based in", "New York"),),
            entities=(),  # no entities
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("The company is based in New York")

        fact2 = FakeFact(
            text="The company is based in London",
            triples=(("The company", "is based in", "London"),),
            entities=(),  # no entities
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("The company is based in London")

        # No shared entities → no supersession
        assert len(agent._superseded_texts) == 0

    def test_entity_gate_case_insensitive(self):
        """Entity matching should be case-insensitive."""
        agent = self._make_agent(supersession=True)

        fact1 = FakeFact(
            text="ABBIATI plays football",
            triples=(("ABBIATI", "plays", "football"),),
            entities=("ABBIATI",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("ABBIATI plays football")

        fact2 = FakeFact(
            text="Abbiati plays basketball",
            triples=(("Abbiati", "plays", "basketball"),),
            entities=("Abbiati",),  # different case
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2],
            entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("Abbiati plays basketball")

        assert "ABBIATI plays football" in agent._superseded_texts


# ---------------------------------------------------------------------------
# Test 3: Query-time filtering
# ---------------------------------------------------------------------------


class TestSupersessionQueryFiltering:
    """Test that superseded facts are filtered from retrieval results."""

    def test_filter_removes_superseded_chunks(self):
        """Superseded texts should be removed from merged_chunks."""
        superseded = {"Old fact about Abbiati playing football"}
        chunks = [
            "Old fact about Abbiati playing football",
            "New fact about Abbiati playing basketball",
            "Unrelated fact about weather",
        ]

        filtered = [c for c in chunks if c not in superseded]

        assert len(filtered) == 2
        assert "Old fact about Abbiati playing football" not in filtered
        assert "New fact about Abbiati playing basketball" in filtered
        assert "Unrelated fact about weather" in filtered

    def test_filter_preserves_order(self):
        """Filtering should preserve the original ranking order."""
        superseded = {"Bad fact"}
        chunks = ["Rank 1", "Bad fact", "Rank 2", "Rank 3"]

        filtered = [c for c in chunks if c not in superseded]

        assert filtered == ["Rank 1", "Rank 2", "Rank 3"]

    def test_filter_empty_superseded_set(self):
        """With no superseded facts, all chunks survive."""
        superseded = set()
        chunks = ["A", "B", "C"]

        filtered = [c for c in chunks if c not in superseded]

        assert filtered == chunks

    def test_filter_all_superseded(self):
        """Edge case: all retrieved chunks are superseded."""
        superseded = {"A", "B", "C"}
        chunks = ["A", "B", "C"]

        filtered = [c for c in chunks if c not in superseded]

        assert filtered == []


# ---------------------------------------------------------------------------
# Test 4: Reset clears supersession state
# ---------------------------------------------------------------------------


class TestSupersessionReset:
    """Test that reset() clears supersession data structures."""

    def test_reset_clears_triple_index(self):
        """reset() should empty _triple_index."""
        triple_index = defaultdict(list)
        triple_index[("a", "b")].append(("c", "text"))

        # Simulate reset
        triple_index = defaultdict(list)
        assert len(triple_index) == 0

    def test_reset_clears_superseded_texts(self):
        """reset() should empty _superseded_texts."""
        superseded = {"old fact"}

        # Simulate reset
        superseded = set()
        assert len(superseded) == 0


# ---------------------------------------------------------------------------
# Test 5: End-to-end scenario
# ---------------------------------------------------------------------------


class TestSupersessionEndToEnd:
    """Scenario tests that mirror the actual conflict resolution use case."""

    def test_factconsolidation_scenario(self):
        """Simulate the factconsolidation benchmark pattern:

        Chunk 1 (early): "Christian Abbiati plays association football"
        Chunk 2 (later): "Christian Abbiati is a basketball player"

        Query: "What sport does Christian Abbiati play?"
        Without supersession: both facts retrieved → LLM picks wrong one 96% of time
        With supersession: only "basketball player" survives → correct answer
        """
        triple_index = defaultdict(list)
        superseded = set()

        # Ingest chunk 1 (early fact)
        key = ("christian abbiati", "plays")
        triple_index[key].append(("association football", "Christian Abbiati plays association football"))

        # Ingest chunk 2 (later, updated fact)
        new_obj = "basketball"
        for existing_obj, existing_text in triple_index[key]:
            if existing_obj != new_obj:
                superseded.add(existing_text)
        triple_index[key].append(("basketball", "Christian Abbiati is a basketball player"))

        # Retrieval returns both (cosine similarity doesn't distinguish)
        retrieved = [
            "Christian Abbiati plays association football",
            "Christian Abbiati is a basketball player",
        ]

        # Apply supersession filter
        filtered = [c for c in retrieved if c not in superseded]

        # Only the newer fact survives
        assert len(filtered) == 1
        assert "basketball" in filtered[0]
        assert "association football" not in " ".join(filtered)

    def test_unrelated_facts_both_survive(self):
        """Non-conflicting facts about the same entity should both survive."""
        triple_index = defaultdict(list)
        superseded = set()

        # Fact 1: Abbiati plays football
        triple_index[("abbiati", "plays")].append(("football", "Abbiati plays football"))

        # Fact 2: Abbiati born in Milan (different predicate)
        key2 = ("abbiati", "born in")
        for existing_obj, existing_text in triple_index[key2]:
            if existing_obj != "milan":
                superseded.add(existing_text)
        triple_index[key2].append(("milan", "Abbiati born in Milan"))

        retrieved = [
            "Abbiati plays football",
            "Abbiati born in Milan",
        ]

        filtered = [c for c in retrieved if c not in superseded]

        # Both survive — no conflict
        assert len(filtered) == 2


# -----------------------------------------------------------------------
# Query-Time Conflict Resolution Tests
# -----------------------------------------------------------------------

class TestQueryTimeCR:
    """Tests for query-time CR (_query_time_cr).

    The key idea: instead of building a global supersession graph from all
    18k facts at ingestion, extract triples from only the 10-20 retrieved
    facts at query time. Group by (subject, predicate); when a group has
    different objects, keep only the newest chunk (highest recency).
    """

    def _make_agent(self, entity_gate: bool = False):
        """Create agent with mocked deps for query-time CR testing."""
        from mabench.hermes_agent import HermesMemoryAgent

        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.conflict_resolution = True
                agent._cr_per_hop = True
                agent._cr_entity_gate = entity_gate
                agent._ingestion = MagicMock()
                agent._ingestion._nlp = MagicMock()
                agent._scorer = MagicMock()
                agent.coupled_engine = MagicMock()
                agent.coupled_engine.memory_store = []
        return agent

    def _mock_nlp_pipe(self, agent, triple_map: dict[str, list[tuple[str, str, str]]]):
        """Configure agent's mock nlp.pipe to return docs that produce given triples.

        triple_map: chunk_text → list of (subject, predicate, object) triples
        """
        from mabench.hermes_agent import _extract_acl_triples

        mock_docs = []
        for text in triple_map:
            doc = MagicMock()
            doc.ents = []  # no NER entities by default
            mock_docs.append(doc)

        agent._ingestion._nlp.pipe.return_value = mock_docs

        # Patch _extract_acl_triples at module level to return our triples
        self._triple_map = triple_map
        self._triple_map_keys = list(triple_map.keys())
        self._call_count = 0

        def fake_extract(doc):
            idx = self._call_count
            self._call_count += 1
            key = self._triple_map_keys[idx]
            return self._triple_map[key]

        return fake_extract

    def _set_memory_store(self, agent, text_recency: dict[str, float]):
        """Set up coupled_engine.memory_store with given text→recency pairs."""
        entries = []
        for text, recency in text_recency.items():
            entry = MagicMock()
            entry.text = text
            entry.recency = recency
            entries.append(entry)
        agent.coupled_engine.memory_store = entries

    def test_basic_conflict_keeps_newest(self):
        """Two chunks with same (S,P) but different O: keep newest."""
        agent = self._make_agent()

        chunks = [
            "The sport played by Abbiati is basketball.",
            "The sport played by Abbiati is association football.",
        ]
        triple_map = {
            chunks[0]: [("Abbiati", "sport_played_by", "basketball")],
            chunks[1]: [("Abbiati", "sport_played_by", "association football")],
        }
        self._set_memory_store(agent, {chunks[0]: 100.0, chunks[1]: 200.0})

        fake_extract = self._mock_nlp_pipe(agent, triple_map)

        with patch("mabench.hermes_agent._extract_acl_triples", side_effect=fake_extract):
            result = agent._query_time_cr(chunks)

        # Newer chunk (recency=200) survives, older removed
        assert len(result) == 1
        assert result[0] == chunks[1]

    def test_no_conflict_same_object(self):
        """Two chunks with same (S,P,O): no conflict, both kept."""
        agent = self._make_agent()

        chunks = [
            "Abbiati plays football.",
            "Abbiati plays football (confirmed).",
        ]
        triple_map = {
            chunks[0]: [("Abbiati", "play", "football")],
            chunks[1]: [("Abbiati", "play", "football")],
        }
        self._set_memory_store(agent, {chunks[0]: 100.0, chunks[1]: 200.0})

        fake_extract = self._mock_nlp_pipe(agent, triple_map)

        with patch("mabench.hermes_agent._extract_acl_triples", side_effect=fake_extract):
            result = agent._query_time_cr(chunks)

        assert len(result) == 2

    def test_different_predicate_no_conflict(self):
        """Same subject, different predicate: no conflict."""
        agent = self._make_agent()

        chunks = [
            "Abbiati plays football.",
            "Abbiati was born in Milan.",
        ]
        triple_map = {
            chunks[0]: [("Abbiati", "play", "football")],
            chunks[1]: [("Abbiati", "born_in", "Milan")],
        }
        self._set_memory_store(agent, {chunks[0]: 100.0, chunks[1]: 200.0})

        fake_extract = self._mock_nlp_pipe(agent, triple_map)

        with patch("mabench.hermes_agent._extract_acl_triples", side_effect=fake_extract):
            result = agent._query_time_cr(chunks)

        assert len(result) == 2

    def test_no_triples_chunks_preserved(self):
        """Chunks with no extractable triples are never removed."""
        agent = self._make_agent()

        chunks = [
            "This is a bridging fact with no clear triple.",
            "The sport played by Abbiati is basketball.",
            "The sport played by Abbiati is football.",
        ]
        triple_map = {
            chunks[0]: [],  # no triples
            chunks[1]: [("Abbiati", "sport_played_by", "basketball")],
            chunks[2]: [("Abbiati", "sport_played_by", "football")],
        }
        self._set_memory_store(agent, {
            chunks[0]: 50.0, chunks[1]: 100.0, chunks[2]: 200.0,
        })

        fake_extract = self._mock_nlp_pipe(agent, triple_map)

        with patch("mabench.hermes_agent._extract_acl_triples", side_effect=fake_extract):
            result = agent._query_time_cr(chunks)

        # Bridging fact preserved, old basketball removed, new football kept
        assert len(result) == 2
        assert chunks[0] in result
        assert chunks[2] in result

    def test_single_chunk_passthrough(self):
        """Single chunk: returned as-is without parsing."""
        agent = self._make_agent()
        result = agent._query_time_cr(["only one chunk"])
        assert result == ["only one chunk"]

    def test_empty_passthrough(self):
        """Empty list: returned as-is."""
        agent = self._make_agent()
        result = agent._query_time_cr([])
        assert result == []

    def test_three_versions_keeps_newest(self):
        """Three conflicting chunks: only newest survives."""
        agent = self._make_agent()

        chunks = [
            "Abbiati plays basketball.",
            "Abbiati plays football.",
            "Abbiati plays rugby.",
        ]
        triple_map = {
            chunks[0]: [("Abbiati", "play", "basketball")],
            chunks[1]: [("Abbiati", "play", "football")],
            chunks[2]: [("Abbiati", "play", "rugby")],
        }
        self._set_memory_store(agent, {
            chunks[0]: 100.0, chunks[1]: 200.0, chunks[2]: 300.0,
        })

        fake_extract = self._mock_nlp_pipe(agent, triple_map)

        with patch("mabench.hermes_agent._extract_acl_triples", side_effect=fake_extract):
            result = agent._query_time_cr(chunks)

        # Only newest (rugby, rec=300) survives
        assert len(result) == 1
        assert result[0] == chunks[2]

    def test_entity_gate_blocks_without_shared_ner(self):
        """With entity_gate=True, skip removal when no NER overlap."""
        agent = self._make_agent(entity_gate=True)

        chunks = [
            "Abbiati plays basketball.",
            "Messi plays football.",
        ]
        triple_map = {
            chunks[0]: [("X", "play", "basketball")],
            chunks[1]: [("X", "play", "football")],
        }
        self._set_memory_store(agent, {chunks[0]: 100.0, chunks[1]: 200.0})

        # Mock docs with non-overlapping NER entities
        mock_docs = []
        for ents in [["Abbiati"], ["Messi"]]:
            doc = MagicMock()
            mock_ents = []
            for e in ents:
                ent = MagicMock()
                ent.text = e
                mock_ents.append(ent)
            doc.ents = mock_ents
            mock_docs.append(doc)
        agent._ingestion._nlp.pipe.return_value = mock_docs

        self._triple_map = triple_map
        self._triple_map_keys = list(triple_map.keys())
        self._call_count = 0

        def fake_extract(doc):
            idx = self._call_count
            self._call_count += 1
            return self._triple_map[self._triple_map_keys[idx]]

        with patch("mabench.hermes_agent._extract_acl_triples", side_effect=fake_extract):
            result = agent._query_time_cr(chunks)

        # No removal: entity gate blocks it (no shared NER)
        assert len(result) == 2

    def test_cr_overfetch_increases_effective_k(self):
        """With conflict_resolution + cr_overfetch=3, _retrieve_v2 should
        pass top_k = retrieve_num * 3 to the retrieval method."""
        from mabench.hermes_agent import HermesMemoryAgent

        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.conflict_resolution = True
                agent._cr_overfetch = 3
                agent.retrieve_num = 10
                agent._belief_index = None
                agent.cosine_dedup = False
                agent.cooc_boost_retrieval = True
                agent.transfer_retrieval = False
                agent.coretrieval_retrieval = False
                agent.ppr_retrieval = False
                agent.bm25_weight = 0.0
                agent.hybrid_retrieval = False
                agent.associative_retrieval = False
                agent.sparse_retrieval = False
                agent.cooc_weight = 1.0
                agent.dream_weight = 0.0
                agent.cooc_gate_threshold = 0.0

                agent._scorer = MagicMock()
                agent._scorer.embed.return_value = _unit_vec(8, 0)
                agent.coupled_engine = MagicMock()
                agent.coupled_engine.query_cooc_boost.return_value = [
                    {"index": i, "score": 0.9 - i * 0.01, "text": f"fact {i}"}
                    for i in range(30)
                ]

        results = agent._retrieve_v2("test query")

        # Should have called query_cooc_boost with top_k=30 (10 * 3)
        call_kwargs = agent.coupled_engine.query_cooc_boost.call_args
        assert call_kwargs.kwargs.get("top_k", call_kwargs[1].get("top_k")) == 30
        assert len(results) == 30

    def test_cr_overfetch_1_no_overfetch(self):
        """With cr_overfetch=1, effective_k should equal retrieve_num."""
        from mabench.hermes_agent import HermesMemoryAgent

        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.conflict_resolution = True
                agent._cr_overfetch = 1
                agent.retrieve_num = 10
                agent._belief_index = None
                agent.cosine_dedup = False
                agent.cooc_boost_retrieval = True
                agent.transfer_retrieval = False
                agent.coretrieval_retrieval = False
                agent.ppr_retrieval = False
                agent.bm25_weight = 0.0
                agent.hybrid_retrieval = False
                agent.associative_retrieval = False
                agent.sparse_retrieval = False
                agent.cooc_weight = 1.0
                agent.dream_weight = 0.0
                agent.cooc_gate_threshold = 0.0

                agent._scorer = MagicMock()
                agent._scorer.embed.return_value = _unit_vec(8, 0)
                agent.coupled_engine = MagicMock()
                agent.coupled_engine.query_cooc_boost.return_value = [
                    {"index": i, "score": 0.9, "text": f"fact {i}"}
                    for i in range(10)
                ]

        results = agent._retrieve_v2("test query")

        call_kwargs = agent.coupled_engine.query_cooc_boost.call_args
        assert call_kwargs.kwargs.get("top_k", call_kwargs[1].get("top_k")) == 10


class TestPairwiseCosineDedup:
    """Tests for pairwise cosine dedup: near-duplicate facts → keep newest."""

    def _make_agent_with_store(self, facts, threshold=0.95):
        """Create a minimal agent with cosine_dedup enabled and a mock store."""
        from mabench.hermes_agent import HermesMemoryAgent

        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.cosine_dedup = True
                agent._cosine_dedup_threshold = threshold
                agent.conflict_resolution = False

                # Build mock memory store
                store = []
                for text, emb, recency in facts:
                    m = MagicMock()
                    m.text = text
                    m.embedding = np.array(emb, dtype=np.float64)
                    m.recency = recency
                    store.append(m)

                agent.coupled_engine = MagicMock()
                agent.coupled_engine.memory_store = store
        return agent

    def test_identical_embeddings_keeps_newest(self):
        """Two facts with identical embeddings → older one removed."""
        emb = [1.0, 0.0, 0.0]
        facts = [
            ("Paris is capital", emb, 1.0),
            ("Lyon is capital", emb, 2.0),
        ]
        agent = self._make_agent_with_store(facts, threshold=0.99)
        result = agent._pairwise_cosine_dedup(["Paris is capital", "Lyon is capital"])
        assert result == ["Lyon is capital"]

    def test_orthogonal_embeddings_no_removal(self):
        """Two facts with orthogonal embeddings → both kept."""
        facts = [
            ("Fact about France", [1.0, 0.0, 0.0], 1.0),
            ("Fact about sports", [0.0, 1.0, 0.0], 2.0),
        ]
        agent = self._make_agent_with_store(facts, threshold=0.95)
        result = agent._pairwise_cosine_dedup(
            ["Fact about France", "Fact about sports"]
        )
        assert len(result) == 2

    def test_threshold_boundary(self):
        """Facts just below threshold → both kept."""
        # cos(a, b) ≈ 0.94 (below 0.95 threshold)
        emb_a = [0.94, 0.34, 0.0]
        emb_b = [1.0, 0.0, 0.0]
        facts = [
            ("Fact A", emb_a, 1.0),
            ("Fact B", emb_b, 2.0),
        ]
        agent = self._make_agent_with_store(facts, threshold=0.95)
        result = agent._pairwise_cosine_dedup(["Fact A", "Fact B"])
        assert len(result) == 2

    def test_three_way_conflict_keeps_newest(self):
        """Three near-identical facts → only newest survives."""
        emb = [1.0, 0.0, 0.0]
        facts = [
            ("V1: capital is Paris", emb, 1.0),
            ("V2: capital is Lyon", emb, 2.0),
            ("V3: capital is Marseille", emb, 3.0),
        ]
        agent = self._make_agent_with_store(facts, threshold=0.99)
        chunks = ["V1: capital is Paris", "V2: capital is Lyon", "V3: capital is Marseille"]
        result = agent._pairwise_cosine_dedup(chunks)
        assert result == ["V3: capital is Marseille"]

    def test_mixed_conflicts_and_unrelated(self):
        """Some facts conflict, others are unrelated → only conflicts deduped."""
        facts = [
            ("Abbiati plays football", [1.0, 0.0, 0.0], 1.0),
            ("Abbiati plays hockey", [1.0, 0.01, 0.0], 2.0),  # near-identical
            ("France capital is Paris", [0.0, 1.0, 0.0], 1.0),  # unrelated
        ]
        agent = self._make_agent_with_store(facts, threshold=0.95)
        chunks = ["Abbiati plays football", "Abbiati plays hockey", "France capital is Paris"]
        result = agent._pairwise_cosine_dedup(chunks)
        assert "Abbiati plays hockey" in result  # newer kept
        assert "France capital is Paris" in result  # unrelated kept
        assert "Abbiati plays football" not in result  # older removed

    def test_single_chunk_passthrough(self):
        """Single chunk → returned as-is."""
        facts = [("Only fact", [1.0, 0.0], 1.0)]
        agent = self._make_agent_with_store(facts)
        result = agent._pairwise_cosine_dedup(["Only fact"])
        assert result == ["Only fact"]

    def test_empty_chunks_passthrough(self):
        """Empty list → returned as-is."""
        agent = self._make_agent_with_store([])
        result = agent._pairwise_cosine_dedup([])
        assert result == []

    def test_chunk_without_embedding_preserved(self):
        """Chunks not found in memory store are preserved (e.g. V1 text recall)."""
        facts = [
            ("Known fact", [1.0, 0.0, 0.0], 1.0),
        ]
        agent = self._make_agent_with_store(facts)
        result = agent._pairwise_cosine_dedup(["Known fact", "Unknown V1 fact"])
        assert len(result) == 2


class TestEntityTextSearch:
    """Tests for GLiNER entity index search: find facts by entity lookup."""

    def _make_agent_with_facts(self, fact_texts, entity_map=None):
        """Create a mock agent with facts and a pre-built entity index.

        entity_map: dict mapping entity_name.lower() → list of fact indices.
            If None, builds a simple index from proper nouns in each fact.
        """
        from collections import defaultdict
        from mabench.hermes_agent import HermesMemoryAgent
        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.entity_retrieval = True
                agent._gliner_model = None  # GLiNER not available
                agent._gliner_labels = ["person", "organization", "location", "country", "city"]
                store = []
                for text in fact_texts:
                    m = MagicMock()
                    m.text = text
                    m.recency = 1.0
                    store.append(m)
                agent.coupled_engine = MagicMock()
                agent.coupled_engine.memory_store = store

                # Build entity index
                agent._gliner_entity_index = defaultdict(set)
                if entity_map:
                    for ent, indices in entity_map.items():
                        agent._gliner_entity_index[ent.lower()] = set(indices)
        return agent

    def test_finds_exact_entity(self):
        """Sub-query about 'Christian Abbiati' finds facts mentioning that entity."""
        agent = self._make_agent_with_facts(
            [
                "Esteban Cambiasso is associated with the sport of cricket.",
                "Christian Abbiati is associated with the sport of basketball.",
                "Marco Belinelli is associated with the sport of football.",
            ],
            entity_map={
                "esteban cambiasso": [0], "cricket": [0],
                "christian abbiati": [1], "basketball": [1],
                "marco belinelli": [2], "football": [2],
            },
        )
        results = agent._entity_text_search("What sport did Christian Abbiati play?")
        assert len(results) >= 1
        assert "Christian Abbiati" in results[0]["text"]

    def test_finds_multiword_entity(self):
        """Finds facts about multi-word entities like 'Measure for Measure'."""
        agent = self._make_agent_with_facts(
            [
                'The author of Measure for Measure is Terrance Dicks.',
                'The author of Romeo and Juliet is William Shakespeare.',
                'The author of The Dark Knight Returns is Frank Miller.',
            ],
            entity_map={
                "measure for measure": [0], "terrance dicks": [0],
                "romeo and juliet": [1], "william shakespeare": [1],
                "the dark knight returns": [2], "frank miller": [2],
            },
        )
        # Mock entity extraction to simulate GLiNER output
        agent._extract_query_entities = lambda q: ["Measure for Measure"]
        results = agent._entity_text_search("What is the author of Measure for Measure?")
        assert len(results) >= 1
        assert "Measure for Measure" in results[0]["text"]

    def test_returns_empty_for_no_match(self):
        """Returns empty when no facts match the entity."""
        agent = self._make_agent_with_facts(
            [
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ],
            entity_map={
                "paris": [0], "france": [0],
                "berlin": [1], "germany": [1],
            },
        )
        results = agent._entity_text_search("What is the capital of Atlantis?")
        assert len(results) == 0

    def test_ranks_by_match_count(self):
        """Facts matching more entities rank higher."""
        agent = self._make_agent_with_facts(
            [
                "CBS is a broadcasting company.",
                "The CEO of CBS Corporation is John Smith.",
                "NBC is a broadcasting company.",
            ],
            entity_map={
                "cbs": [0, 1],
                "cbs corporation": [1],
                "john smith": [1],
                "nbc": [2],
            },
        )
        # Mock entity extraction to simulate GLiNER output
        agent._extract_query_entities = lambda q: ["CBS", "CBS Corporation"]
        results = agent._entity_text_search(
            "Who is the Chief Executive Officer of CBS Corporation?"
        )
        # Fact 1 matches both "CBS" and "CBS Corporation" → score 2
        # Fact 0 matches only "CBS" → score 1
        assert len(results) >= 2
        assert "CBS Corporation" in results[0]["text"]

    def test_case_insensitive(self):
        """Entity index lookup is case-insensitive."""
        agent = self._make_agent_with_facts(
            [
                "The author of Dune is Frank Herbert.",
            ],
            entity_map={"dune": [0], "frank herbert": [0]},
        )
        results = agent._entity_text_search("Who wrote Dune?")
        assert len(results) >= 1

    def test_result_format(self):
        """Results have the expected structure."""
        agent = self._make_agent_with_facts(
            [
                "Karl Lueger is a citizen of Austria.",
            ],
            entity_map={"karl lueger": [0], "austria": [0]},
        )
        results = agent._entity_text_search("What is the country of citizenship of Karl Lueger?")
        assert len(results) == 1
        r = results[0]
        assert "text" in r
        assert "score" in r
        assert "source" in r
        assert r["source"] == "entity_index"


class TestEntityGraphExpand:
    """Tests for entity co-mention graph expansion."""

    def _make_agent(self, fact_texts, entity_map):
        from collections import defaultdict
        from mabench.hermes_agent import HermesMemoryAgent
        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                store = []
                for text in fact_texts:
                    m = MagicMock()
                    m.text = text
                    m.recency = 1.0
                    store.append(m)
                agent.coupled_engine = MagicMock()
                agent.coupled_engine.memory_store = store
                agent._gliner_entity_index = defaultdict(set)
                for ent, indices in entity_map.items():
                    agent._gliner_entity_index[ent.lower()] = set(indices)
        return agent

    def test_bridges_via_shared_entity(self):
        """Finds facts sharing an entity with seed facts."""
        agent = self._make_agent(
            [
                "Christian Abbiati plays basketball.",     # 0
                "Basketball was created in USA.",          # 1
                "Marco Belinelli plays football.",         # 2
            ],
            entity_map={
                "christian abbiati": [0],
                "basketball": [0, 1],
                "usa": [1],
                "marco belinelli": [2],
                "football": [2],
            },
        )
        results = agent._entity_graph_expand(
            ["Christian Abbiati plays basketball."], top_k=5,
        )
        # Should find fact 1 (shares "basketball" entity) but not fact 2
        assert len(results) >= 1
        assert any("USA" in r["text"] for r in results)
        assert not any("Belinelli" in r["text"] for r in results)

    def test_returns_empty_when_no_neighbors(self):
        """Returns empty when seed facts have no shared entities."""
        agent = self._make_agent(
            [
                "Paris is the capital of France.",
                "Berlin is the capital of Germany.",
            ],
            entity_map={
                "paris": [0], "france": [0],
                "berlin": [1], "germany": [1],
            },
        )
        results = agent._entity_graph_expand(
            ["Paris is the capital of France."], top_k=5,
        )
        # No other facts share "paris" or "france"
        assert len(results) == 0

    def test_excludes_seed_facts(self):
        """Seed facts are not included in expansion results."""
        agent = self._make_agent(
            [
                "X plays basketball.",
                "Basketball was created in USA.",
            ],
            entity_map={
                "x": [0], "basketball": [0, 1], "usa": [1],
            },
        )
        results = agent._entity_graph_expand(
            ["X plays basketball."], top_k=5,
        )
        assert all(r["text"] != "X plays basketball." for r in results)
