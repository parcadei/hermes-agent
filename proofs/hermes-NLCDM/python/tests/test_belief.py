"""Discriminative tests for Bayesian belief system: soft fact currency scoring.

Two-layer system:
  Layer 1 — Beta-Bernoulli per fact: P(current) = α / (α + β)
  Layer 2 — Belief propagation via entity graph: degraded facts penalize neighbors

Tests follow the same FakeFact/FakeIngestionResult/mock pattern from
test_supersession.py. All tests run on CPU, no GPU, no external models.

Test groups:
  1. Layer 1 — Beta-Bernoulli unit tests (~8)
  2. Layer 2 — Propagation tests (~4)
  3. Integration through _store() (~5)
  4. Retrieval reranking (~3)
  5. Backward compatibility (~2)
  6. Serialization (~2)
"""

import numpy as np
import pytest
import sys
import os
from collections import defaultdict
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mabench.belief import FactBelief, BeliefIndex


# ---------------------------------------------------------------------------
# Helpers (reused from test_supersession.py)
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


@dataclass
class FakeMemoryEntry:
    """Minimal memory store entry with a .text attribute."""
    text: str


# ===========================================================================
# Test 1: Layer 1 — Beta-Bernoulli unit tests
# ===========================================================================

class TestFactBelief:
    """Test the FactBelief dataclass."""

    def test_initial_prior(self):
        """Beta(2, 1) → P(current) = 2/3 ≈ 0.667."""
        b = FactBelief(alpha=2.0, beta=1.0)
        assert abs(b.p_current - 2.0 / 3.0) < 1e-9

    def test_uniform_prior(self):
        """Beta(1, 1) → P(current) = 0.5."""
        b = FactBelief(alpha=1.0, beta=1.0)
        assert abs(b.p_current - 0.5) < 1e-9

    def test_strong_belief(self):
        """Beta(10, 1) → P(current) ≈ 0.909."""
        b = FactBelief(alpha=10.0, beta=1.0)
        assert abs(b.p_current - 10.0 / 11.0) < 1e-9

    def test_weak_belief(self):
        """Beta(1, 10) → P(current) ≈ 0.091."""
        b = FactBelief(alpha=1.0, beta=10.0)
        assert abs(b.p_current - 1.0 / 11.0) < 1e-9


class TestBeliefIndexLayer1:
    """Test Layer 1: Beta-Bernoulli per-fact belief updates."""

    def test_register_fact_default_prior(self):
        """Registered fact gets Beta(prior_alpha, prior_beta)."""
        idx = BeliefIndex(prior_alpha=2.0, prior_beta=1.0)
        idx.register_fact("Abbiati plays football")
        assert abs(idx.score("Abbiati plays football") - 2.0 / 3.0) < 1e-9

    def test_register_fact_idempotent(self):
        """Registering the same fact twice doesn't reset its belief."""
        idx = BeliefIndex(prior_alpha=2.0, prior_beta=1.0)
        idx.register_fact("fact A")
        idx.on_conflict("fact A", "fact B")
        # fact A now has beta=2, P=2/4=0.5
        idx.register_fact("fact A")  # should be no-op
        assert abs(idx.score("fact A") - 0.5) < 1e-9

    def test_conflict_lowers_old_belief(self):
        """Contradiction: old fact beta += 1, lowering P(current)."""
        idx = BeliefIndex(prior_alpha=2.0, prior_beta=1.0)
        idx.register_fact("old fact")
        idx.register_fact("new fact")
        idx.on_conflict("old fact", "new fact")

        # old: Beta(2, 2) → P = 0.5
        assert abs(idx.score("old fact") - 0.5) < 1e-9

    def test_conflict_raises_new_belief(self):
        """Contradiction: new fact alpha += 1, raising P(current)."""
        idx = BeliefIndex(prior_alpha=2.0, prior_beta=1.0)
        idx.register_fact("old fact")
        idx.register_fact("new fact")
        idx.on_conflict("old fact", "new fact")

        # new: Beta(3, 1) → P = 0.75
        assert abs(idx.score("new fact") - 0.75) < 1e-9

    def test_chain_of_conflicts(self):
        """A → B → C: A gets lowest belief, C gets highest."""
        idx = BeliefIndex(prior_alpha=2.0, prior_beta=1.0)
        idx.register_fact("fact A")
        idx.register_fact("fact B")
        idx.register_fact("fact C")

        idx.on_conflict("fact A", "fact B")
        # A: Beta(2,2)=0.5, B: Beta(3,1)=0.75
        idx.on_conflict("fact B", "fact C")
        # B: Beta(3,2)=0.6, C: Beta(3,1)=0.75

        # A also gets penalized again as B's predecessor was A,
        # but on_conflict(B,C) only touches B and C
        assert idx.score("fact A") < idx.score("fact B")
        assert idx.score("fact B") < idx.score("fact C")

    def test_hard_floor_exclusion(self):
        """Fact below hard_floor should be excluded."""
        idx = BeliefIndex(prior_alpha=1.0, prior_beta=1.0, hard_floor=0.15)
        idx.register_fact("stale fact")
        # Drive belief down: multiple contradictions
        for i in range(10):
            idx.on_conflict("stale fact", f"newer_{i}")
        # stale: Beta(1, 11) → P = 1/12 ≈ 0.083 < 0.15
        assert idx.is_excluded("stale fact")

    def test_above_hard_floor_not_excluded(self):
        """Fact above hard_floor should NOT be excluded."""
        idx = BeliefIndex(prior_alpha=2.0, prior_beta=1.0, hard_floor=0.05)
        idx.register_fact("healthy fact")
        idx.on_conflict("healthy fact", "newer")
        # healthy: Beta(2, 2) → P = 0.5 > 0.05
        assert not idx.is_excluded("healthy fact")

    def test_unregistered_fact_scores_one(self):
        """Unknown facts return P(current) = 1.0."""
        idx = BeliefIndex()
        assert idx.score("never seen") == 1.0

    def test_unregistered_fact_not_excluded(self):
        """Unknown facts are never excluded."""
        idx = BeliefIndex()
        assert not idx.is_excluded("never seen")


# ===========================================================================
# Test 2: Layer 2 — Belief propagation
# ===========================================================================

class TestBeliefPropagation:
    """Test Layer 2: entity-graph belief propagation."""

    def _make_entity_graph(self, entity_to_facts, fact_to_entities):
        """Create a mock entity accumulator."""
        graph = MagicMock()
        graph.entity_to_facts = entity_to_facts
        graph.fact_to_entities = fact_to_entities
        return graph

    def test_degraded_fact_penalizes_connected(self):
        """A fact with P < 0.5 should penalize connected facts."""
        idx = BeliefIndex(prior_alpha=2.0, prior_beta=1.0, propagation_damping=0.3)
        idx.register_fact("fact A")  # will be degraded
        idx.register_fact("fact B")  # connected to A

        # Degrade fact A
        idx.on_conflict("fact A", "fact C")
        # A: Beta(2,2) → P = 0.5 — right at boundary

        # Push A below 0.5 with another conflict
        idx._beliefs["fact A"].beta += 1.0
        # A: Beta(2,3) → P = 0.4

        # Entity graph: "entity_x" connects fact A (idx 0) and fact B (idx 1)
        memory_store = [FakeMemoryEntry("fact A"), FakeMemoryEntry("fact B")]
        entity_acc = self._make_entity_graph(
            entity_to_facts={"entity_x": {0, 1}},
            fact_to_entities={0: {"entity_x"}, 1: {"entity_x"}},
        )

        p_b_before = idx.score("fact B")
        idx.propagate(entity_acc, memory_store)
        p_b_after = idx.score("fact B")

        # fact B should have lower P after propagation
        assert p_b_after < p_b_before

    def test_healthy_fact_no_propagation(self):
        """Facts with P >= 0.5 should NOT propagate penalties."""
        idx = BeliefIndex(prior_alpha=2.0, prior_beta=1.0, propagation_damping=0.3)
        idx.register_fact("fact A")  # P = 2/3 > 0.5
        idx.register_fact("fact B")

        memory_store = [FakeMemoryEntry("fact A"), FakeMemoryEntry("fact B")]
        entity_acc = self._make_entity_graph(
            entity_to_facts={"entity_x": {0, 1}},
            fact_to_entities={0: {"entity_x"}, 1: {"entity_x"}},
        )

        p_b_before = idx.score("fact B")
        idx.propagate(entity_acc, memory_store)
        p_b_after = idx.score("fact B")

        assert abs(p_b_after - p_b_before) < 1e-9

    def test_damping_controls_penalty(self):
        """Higher damping → larger penalty on connected facts."""
        for damping in [0.1, 0.5, 0.9]:
            idx = BeliefIndex(
                prior_alpha=2.0, prior_beta=1.0, propagation_damping=damping,
            )
            idx.register_fact("degraded")
            idx.register_fact("connected")

            # Force degraded below 0.5
            idx._beliefs["degraded"].beta = 5.0
            # degraded: Beta(2, 5) → P = 2/7 ≈ 0.286

            memory_store = [
                FakeMemoryEntry("degraded"),
                FakeMemoryEntry("connected"),
            ]
            entity_acc = self._make_entity_graph(
                entity_to_facts={"ent": {0, 1}},
                fact_to_entities={0: {"ent"}, 1: {"ent"}},
            )

            idx.propagate(entity_acc, memory_store)

            # penalty = damping * (1 - 2/7) = damping * 5/7
            expected_beta = 1.0 + damping * (5.0 / 7.0)
            actual_beta = idx._beliefs["connected"].beta
            assert abs(actual_beta - expected_beta) < 1e-9

    def test_propagation_single_pass_no_cycles(self):
        """Propagation is single-pass: A degrades B, but B doesn't
        re-degrade A in the same call."""
        idx = BeliefIndex(prior_alpha=1.0, prior_beta=1.0, propagation_damping=0.5)
        idx.register_fact("A")
        idx.register_fact("B")

        # Force A below 0.5
        idx._beliefs["A"].beta = 3.0  # Beta(1,3) → P = 0.25

        memory_store = [FakeMemoryEntry("A"), FakeMemoryEntry("B")]
        entity_acc = self._make_entity_graph(
            entity_to_facts={"ent": {0, 1}},
            fact_to_entities={0: {"ent"}, 1: {"ent"}},
        )

        a_alpha_before = idx._beliefs["A"].alpha
        a_beta_before = idx._beliefs["A"].beta
        idx.propagate(entity_acc, memory_store)
        # A should NOT get additional penalty from B (single pass only penalizes
        # neighbors of *already* degraded facts at the time of the call)
        assert idx._beliefs["A"].alpha == a_alpha_before
        assert idx._beliefs["A"].beta == a_beta_before


# ===========================================================================
# Test 3: Integration through _store()
# ===========================================================================

class TestBeliefStore:
    """Test belief updates through the _store method."""

    def _make_agent(self, belief: bool = True, supersession: bool = False):
        """Create a HermesMemoryAgent with mocked deps and belief enabled."""
        from mabench.hermes_agent import HermesMemoryAgent

        mock_pipeline = MagicMock()
        mock_scorer = MagicMock()
        mock_scorer.embed_batch.return_value = [_unit_vec(8, i) for i in range(10)]
        mock_scorer.embed.side_effect = lambda t: _unit_vec(8, hash(t) % 8)

        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.supersession = supersession
                agent._triple_index = defaultdict(list)
                agent._superseded_texts = set()
                agent._fact_entities = {}
                agent._ingestion = mock_pipeline
                agent._scorer = mock_scorer
                agent._store_count = 0
                agent.dream_interval = 0
                agent._triadic = None
                agent._entity_accumulator = MagicMock()

                # Minimal orchestrator and coupled engine stubs
                agent.orchestrator = MagicMock()
                agent.coupled_engine = MagicMock()
                agent.coupled_engine._session_buffer = []
                agent.coupled_engine.memory_store = []

                # Belief index
                if belief:
                    agent._belief_index = BeliefIndex(
                        prior_alpha=2.0, prior_beta=1.0,
                        propagation_damping=0.3, hard_floor=0.05,
                    )
                else:
                    agent._belief_index = None

        return agent

    def test_store_registers_facts_in_belief(self):
        """_store should register each fact in the belief index."""
        agent = self._make_agent(belief=True)

        fact = FakeFact(
            text="Abbiati plays football",
            triples=(("Abbiati", "plays", "football"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact], entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("Abbiati plays football")

        assert agent._belief_index.score("Abbiati plays football") == pytest.approx(2.0 / 3.0)

    def test_store_updates_belief_on_conflict(self):
        """_store should call on_conflict when triples conflict (entity-gated)."""
        agent = self._make_agent(belief=True, supersession=False)

        # First fact
        fact1 = FakeFact(
            text="Abbiati plays football",
            triples=(("Abbiati", "plays", "football"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1], entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("Abbiati plays football")

        # Second fact — conflict
        fact2 = FakeFact(
            text="Abbiati plays basketball",
            triples=(("Abbiati", "plays", "basketball"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2], entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("Abbiati plays basketball")

        # Old fact belief should be lower than new
        assert agent._belief_index.score("Abbiati plays football") < \
               agent._belief_index.score("Abbiati plays basketball")

    def test_belief_works_without_supersession(self):
        """Belief should detect conflicts even when --supersession is off."""
        agent = self._make_agent(belief=True, supersession=False)

        fact1 = FakeFact(
            text="Abbiati plays football",
            triples=(("Abbiati", "plays", "football"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1], entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("Abbiati plays football")

        fact2 = FakeFact(
            text="Abbiati plays basketball",
            triples=(("Abbiati", "plays", "basketball"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2], entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("Abbiati plays basketball")

        # Belief updated despite supersession=False
        assert agent._belief_index.score("Abbiati plays football") < 2.0 / 3.0
        # But _superseded_texts should be empty (supersession off)
        assert len(agent._superseded_texts) == 0

    def test_entity_gate_prevents_false_belief_update(self):
        """No shared entity → no belief conflict update."""
        agent = self._make_agent(belief=True, supersession=False)

        fact1 = FakeFact(
            text="The author of Dianetics is Hubbard",
            triples=(("The author", "is", "Hubbard"),),
            entities=("Hubbard", "Dianetics"),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1], entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]
        agent._store("The author of Dianetics is Hubbard")

        fact2 = FakeFact(
            text="The author of Prior Analytics is Aristotle",
            triples=(("The author", "is", "Aristotle"),),
            entities=("Aristotle", "Prior Analytics"),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2], entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("The author of Prior Analytics is Aristotle")

        # No conflict — different entities, both at default prior
        assert agent._belief_index.score("The author of Dianetics is Hubbard") == \
               pytest.approx(2.0 / 3.0)

    def test_propagation_fires_after_merge(self):
        """Propagation should run after entity_accumulator.merge in _store."""
        agent = self._make_agent(belief=True)

        # Set up entity accumulator with real data so propagation can run
        agent._entity_accumulator = MagicMock()
        agent._entity_accumulator.fact_to_entities = {}
        agent._entity_accumulator.entity_to_facts = {}

        fact = FakeFact(
            text="some fact",
            triples=(),
            entities=(),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact], entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 0)]

        # This should not crash — propagation runs but with empty graph
        agent._store("some fact")
        assert agent._belief_index.score("some fact") == pytest.approx(2.0 / 3.0)


# ===========================================================================
# Test 4: Retrieval reranking
# ===========================================================================

class TestBeliefRerank:
    """Test belief-based retrieval reranking."""

    def test_rerank_promotes_newer_facts(self):
        """Belief rerank should promote high-P facts over high-cosine stale facts."""
        idx = BeliefIndex(prior_alpha=2.0, prior_beta=1.0)
        idx.register_fact("old fact")
        idx.register_fact("new fact")
        idx.on_conflict("old fact", "new fact")

        # Simulate retrieval results: old fact has higher cosine
        results = [
            {"text": "old fact", "score": 0.85},
            {"text": "new fact", "score": 0.80},
        ]

        # Apply belief rerank
        reranked = []
        for r in results:
            p = idx.score(r["text"])
            reranked.append({**r, "adjusted_score": r["score"] * p})
        reranked.sort(key=lambda x: x["adjusted_score"], reverse=True)

        # new fact should now rank first
        assert reranked[0]["text"] == "new fact"

    def test_hard_floor_excludes_from_results(self):
        """Facts below hard_floor should be excluded entirely."""
        idx = BeliefIndex(prior_alpha=1.0, prior_beta=1.0, hard_floor=0.15)
        idx.register_fact("stale")
        # Drive belief very low
        for i in range(10):
            idx.on_conflict("stale", f"newer_{i}")

        results = [
            {"text": "stale", "score": 0.90},
            {"text": "fresh", "score": 0.70},
        ]

        filtered = [r for r in results if not idx.is_excluded(r["text"])]
        assert len(filtered) == 1
        assert filtered[0]["text"] == "fresh"

    def test_unknown_facts_pass_through(self):
        """Facts not in the belief index should score 1.0 and pass through."""
        idx = BeliefIndex()
        results = [
            {"text": "unknown fact", "score": 0.80},
        ]

        reranked = []
        for r in results:
            p = idx.score(r["text"])
            reranked.append({**r, "adjusted_score": r["score"] * p})

        assert reranked[0]["adjusted_score"] == pytest.approx(0.80)


# ===========================================================================
# Test 5: Backward compatibility
# ===========================================================================

class TestBeliefBackwardCompat:
    """Test that disabling belief preserves existing behavior."""

    def test_no_belief_flag_no_changes(self):
        """With _belief_index=None, _store should behave identically to before."""
        from mabench.hermes_agent import HermesMemoryAgent

        mock_pipeline = MagicMock()
        mock_scorer = MagicMock()
        mock_scorer.embed_batch.return_value = [_unit_vec(8, 0)]

        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.supersession = False
                agent._triple_index = defaultdict(list)
                agent._superseded_texts = set()
                agent._fact_entities = {}
                agent._ingestion = mock_pipeline
                agent._scorer = mock_scorer
                agent._store_count = 0
                agent.dream_interval = 0
                agent._triadic = None
                agent._entity_accumulator = MagicMock()
                agent.orchestrator = MagicMock()
                agent.coupled_engine = MagicMock()
                agent.coupled_engine._session_buffer = []
                agent.coupled_engine.memory_store = []
                agent._belief_index = None

        fact = FakeFact(
            text="some text",
            triples=(("A", "B", "C"),),
            entities=("A",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact], entity_graph=FakeEntityGraph(),
        )
        agent._store("some text")

        # No belief operations should have occurred
        assert agent._belief_index is None
        assert len(agent._superseded_texts) == 0

    def test_supersession_alone_unchanged(self):
        """With supersession=True but belief=None, supersession works as before."""
        from mabench.hermes_agent import HermesMemoryAgent

        mock_pipeline = MagicMock()
        mock_scorer = MagicMock()
        mock_scorer.embed_batch.return_value = [_unit_vec(8, 0)]

        with patch("mabench.hermes_agent.IngestionPipeline"):
            with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
                agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
                agent.supersession = True
                agent._triple_index = defaultdict(list)
                agent._superseded_texts = set()
                agent._fact_entities = {}
                agent._ingestion = mock_pipeline
                agent._scorer = mock_scorer
                agent._store_count = 0
                agent.dream_interval = 0
                agent._triadic = None
                agent._entity_accumulator = MagicMock()
                agent.orchestrator = MagicMock()
                agent.coupled_engine = MagicMock()
                agent.coupled_engine._session_buffer = []
                agent.coupled_engine.memory_store = []
                agent._belief_index = None

        # Store two conflicting facts
        fact1 = FakeFact(
            text="Abbiati plays football",
            triples=(("Abbiati", "plays", "football"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact1], entity_graph=FakeEntityGraph(),
        )
        agent._store("Abbiati plays football")

        fact2 = FakeFact(
            text="Abbiati plays basketball",
            triples=(("Abbiati", "plays", "basketball"),),
            entities=("Abbiati",),
        )
        agent._ingestion.segment.return_value = FakeIngestionResult(
            facts=[fact2], entity_graph=FakeEntityGraph(),
        )
        agent._scorer.embed_batch.return_value = [_unit_vec(8, 1)]
        agent._store("Abbiati plays basketball")

        # Supersession should still work
        assert "Abbiati plays football" in agent._superseded_texts
        # But no belief index
        assert agent._belief_index is None


# ===========================================================================
# Test 6: Serialization
# ===========================================================================

class TestBeliefSerialization:
    """Test state_dict / from_state_dict round-trip."""

    def test_round_trip(self):
        """Serialize and deserialize should preserve all state."""
        idx = BeliefIndex(prior_alpha=3.0, prior_beta=2.0,
                          propagation_damping=0.5, hard_floor=0.1)
        idx.register_fact("fact A")
        idx.register_fact("fact B")
        idx.on_conflict("fact A", "fact B")

        state = idx.state_dict()
        restored = BeliefIndex.from_state_dict(state)

        assert restored.score("fact A") == pytest.approx(idx.score("fact A"))
        assert restored.score("fact B") == pytest.approx(idx.score("fact B"))
        assert restored._prior_alpha == 3.0
        assert restored._prior_beta == 2.0
        assert restored._propagation_damping == 0.5
        assert restored._hard_floor == 0.1

    def test_empty_round_trip(self):
        """Empty BeliefIndex should round-trip cleanly."""
        idx = BeliefIndex()
        state = idx.state_dict()
        restored = BeliefIndex.from_state_dict(state)
        assert len(restored._beliefs) == 0
        assert restored.score("unknown") == 1.0
