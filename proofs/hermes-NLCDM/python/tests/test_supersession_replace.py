"""Tests for entity-aware supersession with in-place replacement.

Core hypothesis: when two facts share the same (subject, predicate) and at least
one named entity (substring match, len > 2), the newer fact REPLACES the older
one in-place in coupled_engine.memory_store instead of appending. This is O(1)
and keeps memory_store length constant for updates.

Test cases:
  a) In-place replacement: same entity+predicate, different object => replace
  b) Entity gate blocks cross-entity: different entities, same predicate => append
  c) Audit log preserved across multiple supersessions
  d) reset() clears all supersession state

All tests run on CPU. No GPU, no LLM calls, no spaCy. Ingestion and scoring
are mocked to produce controlled facts with known triples and entities.
"""

import numpy as np
import time
import pytest
import sys
import os
from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import MagicMock

# Prevent OpenMP conflict between torch and FAISS on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from coupled_engine import CoupledEngine


# ---------------------------------------------------------------------------
# Fake types (mimic ingestion.SegmentedFact and IngestionResult)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeFact:
    """Minimal SegmentedFact-like object for testing."""
    text: str
    triples: tuple  # tuple of (subj, pred, obj) tuples
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
# Helpers
# ---------------------------------------------------------------------------

DIM = 64
RNG = np.random.RandomState(42)


def _random_emb(dim: int = DIM, seed: int | None = None) -> np.ndarray:
    """Generate a random L2-normalized embedding."""
    rng = np.random.RandomState(seed) if seed is not None else RNG
    v = rng.randn(dim).astype(np.float64)
    return v / np.linalg.norm(v)


def _make_agent(dim: int = DIM):
    """Create a HermesMemoryAgent with mocked ingestion/scoring and a REAL
    CoupledEngine for testing supersession in-place replacement.

    The ingestion pipeline and scorer are mocked so _store() can be called
    with controlled facts. The coupled_engine is real so we can verify
    memory_store mutations.
    """
    from mabench.hermes_agent import HermesMemoryAgent

    mock_pipeline = MagicMock()
    mock_scorer = MagicMock()
    # reset() calls self._scorer._cache.clear()
    mock_scorer._cache = {}

    # Create agent bypassing __init__ (which loads spaCy, models, etc.)
    agent = HermesMemoryAgent.__new__(HermesMemoryAgent)

    # Core config
    agent.model = "gpt-4o-mini"
    agent.dim = dim
    agent.contradiction_threshold = 0.95
    agent.retrieve_num = 10
    agent.dream_interval = 0
    agent.temperature = 0.0
    agent._max_gen_tokens = 64
    agent.dedup_threshold = 0.0
    agent.dream_params = None  # needed by reset()

    # Retrieval flags (all off for store-only testing)
    agent.cooc_boost_retrieval = True
    agent.decompose_query = False
    agent.conflict_resolution = True
    agent._cr_per_hop = False
    agent._cr_entity_gate = False
    agent._cr_overfetch = 1
    agent.cosine_dedup = False
    agent.entity_retrieval = False
    agent.entity_first = False
    agent.entity_first_aggressive = False
    agent.bm25_hops = False
    agent.chain_cache = False
    agent.rebel_kg = False
    agent.associative_retrieval = False
    agent.sparse_retrieval = False
    agent.hybrid_retrieval = False
    agent.ppr_retrieval = False
    agent.coretrieval_retrieval = False
    agent.transfer_retrieval = False
    agent.triadic_retrieval = False
    agent.bm25_weight = 0.0
    agent.iterative_query = False
    agent.decompose_rrf = False
    agent.decompose_coverage = False
    agent.decompose_max_hops = 3

    # Supersession: enabled
    agent.supersession = True
    agent._triple_index = defaultdict(list)
    agent._superseded_texts = set()
    agent._fact_entities = {}
    agent._supersession_graph = {}
    agent._supersedes = defaultdict(set)
    agent._text_to_store_idx = {}
    agent._supersession_log = []

    # Dependencies
    agent._ingestion = mock_pipeline
    agent._scorer = mock_scorer
    agent._store_count = 0
    agent._triadic = None
    agent._bm25_index = None
    agent._chain_cache = None
    agent._rebel_kg = None
    agent._belief_index = None
    agent._gliner_entity_index = defaultdict(set)
    agent._entity_accumulator = MagicMock()

    # V1 orchestrator (mocked — not used when cooc_boost=True)
    mock_orchestrator = MagicMock()
    # reset() accesses self.orchestrator.params
    mock_orchestrator.params = MagicMock()
    agent.orchestrator = mock_orchestrator

    # V2 coupled engine (REAL — so we can verify in-place replacement)
    agent.coupled_engine = CoupledEngine(
        dim=dim,
        contradiction_aware=False,
        emotional_tagging=False,
    )

    return agent, mock_pipeline, mock_scorer


def _setup_store_call(mock_pipeline, mock_scorer, facts, dim=DIM):
    """Configure mocked ingestion and scorer to return the given facts.

    Each call to _store() will consume one set of facts from the queue.
    """
    result = FakeIngestionResult(
        facts=facts,
        entity_graph=FakeEntityGraph(),
    )
    mock_pipeline.segment.return_value = result
    # Return a distinct random embedding for each fact
    embeddings = [_random_emb(dim, seed=hash(f.text) % 10000) for f in facts]
    mock_scorer.embed_batch.return_value = embeddings
    return embeddings


# ---------------------------------------------------------------------------
# Test a: In-place replacement
# ---------------------------------------------------------------------------


class TestSupersessionReplacesInPlace:
    """When a new fact has the same (subject, predicate) as an existing fact,
    AND they share a named entity (substring match, len > 2), the old entry
    should be replaced in-place: memory_store length stays the same, the entry
    at the old index now has the new text.
    """

    def test_supersession_replaces_in_place(self):
        agent, mock_pipeline, mock_scorer = _make_agent()

        # Chunk 1: "The capital of France is Lyon"
        fact1 = FakeFact(
            text="The capital of France is Lyon",
            triples=(("France", "capital of", "Lyon"),),
            entities=("France", "Lyon"),
        )
        embs1 = _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("The capital of France is Lyon")

        assert len(agent.coupled_engine.memory_store) == 1
        assert agent.coupled_engine.memory_store[0].text == "The capital of France is Lyon"
        assert agent._text_to_store_idx["The capital of France is Lyon"] == 0

        # Chunk 2: "The capital of France is Paris" (same subject+predicate, different object)
        fact2 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        embs2 = _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("The capital of France is Paris")

        # Memory store should NOT have grown — in-place replacement
        assert len(agent.coupled_engine.memory_store) == 1, (
            f"Expected 1 entry (in-place replacement), got {len(agent.coupled_engine.memory_store)}"
        )

        # The entry at index 0 now has the NEW text
        assert agent.coupled_engine.memory_store[0].text == "The capital of France is Paris"

        # Audit log should have exactly one supersession entry
        assert len(agent._supersession_log) == 1
        log_entry = agent._supersession_log[0]
        assert log_entry["old_text"] == "The capital of France is Lyon"
        assert log_entry["new_text"] == "The capital of France is Paris"
        assert "old_recency" in log_entry
        assert "new_recency" in log_entry
        assert "timestamp" in log_entry

        # _text_to_store_idx should map new text to index 0, old text gone
        assert "The capital of France is Paris" in agent._text_to_store_idx
        assert "The capital of France is Lyon" not in agent._text_to_store_idx
        assert agent._text_to_store_idx["The capital of France is Paris"] == 0

        # _superseded_texts should contain the old text
        assert "The capital of France is Lyon" in agent._superseded_texts

        # _supersession_graph should have old->new mapping
        assert agent._supersession_graph["The capital of France is Lyon"] == "The capital of France is Paris"

    def test_embedding_updated_in_place(self):
        """The replaced entry's embedding should match the NEW fact's embedding,
        not the old one.
        """
        agent, mock_pipeline, mock_scorer = _make_agent()

        fact1 = FakeFact(
            text="The capital of France is Lyon",
            triples=(("France", "capital of", "Lyon"),),
            entities=("France", "Lyon"),
        )
        embs1 = _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        old_emb = agent.coupled_engine.memory_store[0].embedding.copy()

        fact2 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        embs2 = _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        new_emb = agent.coupled_engine.memory_store[0].embedding
        # The embedding should have changed (different seed => different vector)
        assert not np.allclose(old_emb, new_emb), "Embedding should be updated to the new fact's embedding"
        # It should match the embedding we provided for fact2
        assert np.allclose(new_emb, embs2[0]), "Embedding should match the replacement embedding"

    def test_norm_embs_updated_in_place(self):
        """The _norm_embs_f32 cache should be updated at the replaced index."""
        agent, mock_pipeline, mock_scorer = _make_agent()

        fact1 = FakeFact(
            text="The capital of France is Lyon",
            triples=(("France", "capital of", "Lyon"),),
            entities=("France", "Lyon"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        old_norm = agent.coupled_engine._norm_embs_f32[0].copy()

        fact2 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        embs2 = _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        new_norm = agent.coupled_engine._norm_embs_f32[0]
        # Should have changed
        assert not np.allclose(old_norm, new_norm), "_norm_embs_f32 should be updated"
        # Should be the normalized version of the new embedding
        expected = embs2[0] / (np.linalg.norm(embs2[0]) + 1e-12)
        assert np.allclose(new_norm, expected.astype(np.float32), atol=1e-5)

    def test_faiss_dirty_after_replacement(self):
        """After supersession replacement, _faiss_dirty should be True, then
        cleared by rebuild_faiss_if_dirty() at end of _store().
        """
        agent, mock_pipeline, mock_scorer = _make_agent()

        fact1 = FakeFact(
            text="The capital of France is Lyon",
            triples=(("France", "capital of", "Lyon"),),
            entities=("France", "Lyon"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        fact2 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        # After _store completes, rebuild_faiss_if_dirty() should have been called
        # so _faiss_dirty should be False
        assert not agent.coupled_engine._faiss_dirty, (
            "_faiss_dirty should be False after _store (deferred rebuild runs at end)"
        )


# ---------------------------------------------------------------------------
# Test b: Entity gate blocks cross-entity supersession
# ---------------------------------------------------------------------------


class TestEntityGateBlocksCrossEntity:
    """When facts have the same predicate structure but DIFFERENT entities,
    the entity gate should prevent supersession. The store should grow normally.
    """

    def test_different_entities_not_superseded(self):
        agent, mock_pipeline, mock_scorer = _make_agent()

        # Fact 1: France's capital
        fact1 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        assert len(agent.coupled_engine.memory_store) == 1

        # Fact 2: Germany's capital — same predicate "capital of" but different subject
        # Entity gate: "Germany" not in "The capital of France is Paris",
        #              "France" not in "The capital of Germany is Berlin"
        fact2 = FakeFact(
            text="The capital of Germany is Berlin",
            triples=(("Germany", "capital of", "Berlin"),),
            entities=("Germany", "Berlin"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        # Should have GROWN (no replacement — different entities)
        assert len(agent.coupled_engine.memory_store) == 2, (
            f"Expected 2 entries (different entities, no supersession), "
            f"got {len(agent.coupled_engine.memory_store)}"
        )

        # No supersession log entries
        assert len(agent._supersession_log) == 0, (
            f"Expected empty supersession log for cross-entity facts, "
            f"got {len(agent._supersession_log)} entries"
        )

        # Both texts should be in the store
        texts = [e.text for e in agent.coupled_engine.memory_store]
        assert "The capital of France is Paris" in texts
        assert "The capital of Germany is Berlin" in texts

    def test_short_entity_names_blocked(self):
        """Entities with len <= 2 should NOT trigger entity gate matching.
        The len > 2 check prevents false matches on 'US', 'UK', etc.
        """
        agent, mock_pipeline, mock_scorer = _make_agent()

        # Both facts mention "US" (len=2) — should NOT match
        fact1 = FakeFact(
            text="The leader of US is Biden",
            triples=(("US", "leader of", "Biden"),),
            entities=("US",),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        fact2 = FakeFact(
            text="The leader of US is Trump",
            triples=(("US", "leader of", "Trump"),),
            entities=("US",),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        # "US" has len=2, so entity gate should block the match.
        # Both entries should be in the store.
        assert len(agent.coupled_engine.memory_store) == 2, (
            f"Expected 2 entries (entity name 'US' too short for gate), "
            f"got {len(agent.coupled_engine.memory_store)}"
        )

    def test_same_subject_different_predicate_not_superseded(self):
        """Same entity but DIFFERENT predicate should NOT trigger supersession."""
        agent, mock_pipeline, mock_scorer = _make_agent()

        fact1 = FakeFact(
            text="France has a population of 67 million",
            triples=(("France", "population of", "67 million"),),
            entities=("France",),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        fact2 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France",),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        # Different predicates — no supersession
        assert len(agent.coupled_engine.memory_store) == 2
        assert len(agent._supersession_log) == 0


# ---------------------------------------------------------------------------
# Test c: Audit log preserved across multiple supersessions
# ---------------------------------------------------------------------------


class TestSupersessionAuditLog:
    """The _supersession_log should accumulate entries across multiple
    supersessions, ordered by occurrence, with full bitemporal metadata.
    """

    def test_multiple_supersessions_ordered(self):
        agent, mock_pipeline, mock_scorer = _make_agent()

        # Store initial: France capital = Lyon
        fact1 = FakeFact(
            text="The capital of France is Lyon",
            triples=(("France", "capital of", "Lyon"),),
            entities=("France", "Lyon"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        # Update 1: France capital = Marseille (supersedes Lyon)
        fact2 = FakeFact(
            text="The capital of France is Marseille",
            triples=(("France", "capital of", "Marseille"),),
            entities=("France", "Marseille"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        # Update 2: France capital = Paris (supersedes Marseille)
        fact3 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact3])
        agent._store("chunk3")

        # Should have exactly 2 supersession events
        assert len(agent._supersession_log) == 2, (
            f"Expected 2 supersession events, got {len(agent._supersession_log)}"
        )

        # First event: Lyon -> Marseille
        assert agent._supersession_log[0]["old_text"] == "The capital of France is Lyon"
        assert agent._supersession_log[0]["new_text"] == "The capital of France is Marseille"

        # Second event: Marseille -> Paris
        assert agent._supersession_log[1]["old_text"] == "The capital of France is Marseille"
        assert agent._supersession_log[1]["new_text"] == "The capital of France is Paris"

        # Timestamps should be monotonically increasing
        assert agent._supersession_log[0]["timestamp"] <= agent._supersession_log[1]["timestamp"]

        # Memory store still has exactly 1 entry (all replacements in-place)
        assert len(agent.coupled_engine.memory_store) == 1
        assert agent.coupled_engine.memory_store[0].text == "The capital of France is Paris"

    def test_audit_log_has_all_required_fields(self):
        agent, mock_pipeline, mock_scorer = _make_agent()

        fact1 = FakeFact(
            text="The capital of France is Lyon",
            triples=(("France", "capital of", "Lyon"),),
            entities=("France", "Lyon"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        fact2 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        entry = agent._supersession_log[0]
        required_fields = {"old_text", "new_text", "old_recency", "new_recency", "timestamp"}
        actual_fields = set(entry.keys())
        assert required_fields <= actual_fields, (
            f"Missing fields: {required_fields - actual_fields}. Got: {actual_fields}"
        )

    def test_mixed_supersession_and_normal_stores(self):
        """Interleaved supersession and normal stores should produce
        the correct audit log and memory_store count.
        """
        agent, mock_pipeline, mock_scorer = _make_agent()

        # Fact 1: France capital (normal store)
        fact1 = FakeFact(
            text="The capital of France is Lyon",
            triples=(("France", "capital of", "Lyon"),),
            entities=("France", "Lyon"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        # Fact 2: Germany capital (normal store, different entity)
        fact2 = FakeFact(
            text="The capital of Germany is Berlin",
            triples=(("Germany", "capital of", "Berlin"),),
            entities=("Germany", "Berlin"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        # Fact 3: France capital update (supersession)
        fact3 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact3])
        agent._store("chunk3")

        # memory_store: 2 entries (France replaced, Germany added normally)
        assert len(agent.coupled_engine.memory_store) == 2
        # Supersession log: 1 entry (France Lyon -> Paris)
        assert len(agent._supersession_log) == 1
        assert agent._supersession_log[0]["old_text"] == "The capital of France is Lyon"
        assert agent._supersession_log[0]["new_text"] == "The capital of France is Paris"


# ---------------------------------------------------------------------------
# Test d: reset() clears supersession state
# ---------------------------------------------------------------------------


class TestSupersessionResetClearsState:
    """After calling reset(), all supersession tracking state should be empty."""

    def test_reset_clears_all_supersession_state(self):
        agent, mock_pipeline, mock_scorer = _make_agent()

        # Perform a supersession
        fact1 = FakeFact(
            text="The capital of France is Lyon",
            triples=(("France", "capital of", "Lyon"),),
            entities=("France", "Lyon"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        fact2 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        # Sanity: state is populated before reset
        assert len(agent._supersession_log) == 1
        assert len(agent._text_to_store_idx) > 0
        assert len(agent._superseded_texts) > 0
        assert len(agent._supersession_graph) > 0
        assert len(agent._triple_index) > 0
        assert len(agent._fact_entities) > 0

        # Reset
        agent.reset()

        # All supersession state should be cleared
        assert len(agent._text_to_store_idx) == 0, (
            f"_text_to_store_idx should be empty after reset, has {len(agent._text_to_store_idx)} entries"
        )
        assert len(agent._supersession_log) == 0, (
            f"_supersession_log should be empty after reset, has {len(agent._supersession_log)} entries"
        )
        assert len(agent._superseded_texts) == 0, (
            f"_superseded_texts should be empty after reset, has {len(agent._superseded_texts)} entries"
        )
        assert len(agent._supersession_graph) == 0, (
            f"_supersession_graph should be empty after reset, has {len(agent._supersession_graph)} entries"
        )
        assert len(agent._supersedes) == 0, (
            f"_supersedes should be empty after reset, has {len(agent._supersedes)} entries"
        )
        assert len(agent._triple_index) == 0, (
            f"_triple_index should be empty after reset, has {len(agent._triple_index)} entries"
        )
        assert len(agent._fact_entities) == 0, (
            f"_fact_entities should be empty after reset, has {len(agent._fact_entities)} entries"
        )


# ---------------------------------------------------------------------------
# Test: triple_index updated correctly for supersession
# ---------------------------------------------------------------------------


class TestTripleIndexUpdated:
    """The _triple_index should be updated correctly when a fact is superseded:
    the old (obj, text) entry should be removed and replaced with the new one.
    """

    def test_triple_index_old_removed_new_added(self):
        agent, mock_pipeline, mock_scorer = _make_agent()

        fact1 = FakeFact(
            text="The capital of France is Lyon",
            triples=(("France", "capital of", "Lyon"),),
            entities=("France", "Lyon"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact1])
        agent._store("chunk1")

        key = ("france", "capital of")
        entries_before = agent._triple_index[key]
        assert len(entries_before) == 1
        assert entries_before[0] == ("lyon", "The capital of France is Lyon")

        fact2 = FakeFact(
            text="The capital of France is Paris",
            triples=(("France", "capital of", "Paris"),),
            entities=("France", "Paris"),
        )
        _setup_store_call(mock_pipeline, mock_scorer, [fact2])
        agent._store("chunk2")

        entries_after = agent._triple_index[key]
        # Should have exactly 1 entry (old removed, new added)
        assert len(entries_after) == 1, (
            f"Expected 1 triple_index entry after supersession, got {len(entries_after)}: {entries_after}"
        )
        assert entries_after[0] == ("paris", "The capital of France is Paris")
