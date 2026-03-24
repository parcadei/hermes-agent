"""Tests for _retrieve_v2 modular decomposition.

Verifies that extracting _retrieve_v2 into _embed_query + _retrieve_candidates
+ _apply_post_retrieval preserves identical behavior.

The extraction is purely mechanical: no logic changes, just method boundaries.
All tests run on CPU with mocked dependencies (no GPU, no LLM, no spaCy).
"""

import numpy as np
import pytest
import sys
import os
from collections import defaultdict
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mabench.belief import BeliefIndex


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


def _make_agent(
    belief: bool = False,
    conflict_resolution: bool = False,
    cr_overfetch: int = 3,
    cosine_dedup: bool = False,
    retrieve_num: int = 5,
    cooc_boost: bool = False,
    ppr: bool = False,
    bm25_weight: float = 0.0,
):
    """Create a HermesMemoryAgent with mocked dependencies for retrieval tests.

    Sets up:
      - mock _scorer with embed() returning deterministic vectors
      - mock coupled_engine with query() returning fake results
      - all retrieval flags defaulting to the plain cosine path
    """
    from mabench.hermes_agent import HermesMemoryAgent

    mock_scorer = MagicMock()
    mock_scorer.embed.side_effect = lambda t: _unit_vec(8, hash(t) % 8)

    # Fake retrieval results: 10 results with descending scores
    fake_results = [
        {"text": f"fact_{i}", "score": 1.0 - i * 0.05}
        for i in range(10)
    ]

    mock_engine = MagicMock()
    mock_engine.query.return_value = list(fake_results)
    mock_engine.query_cooc_boost.return_value = list(fake_results)
    mock_engine.query_cooc_ppr.return_value = list(fake_results)
    mock_engine.query_cooc_bm25.return_value = list(fake_results)
    mock_engine.query_ppr.return_value = list(fake_results)
    mock_engine.query_hybrid.return_value = list(fake_results)
    mock_engine.query_associative.return_value = list(fake_results)
    mock_engine.query_transfer.return_value = list(fake_results)
    mock_engine.query_coretrieval.return_value = list(fake_results)
    mock_engine.memory_store = []
    mock_engine._session_buffer = []

    with patch("mabench.hermes_agent.IngestionPipeline"):
        with patch("mabench.hermes_agent.EmbeddingRelevanceScorer"):
            agent = HermesMemoryAgent.__new__(HermesMemoryAgent)
            agent._scorer = mock_scorer
            agent.coupled_engine = mock_engine
            agent.retrieve_num = retrieve_num
            agent.conflict_resolution = conflict_resolution
            agent._cr_overfetch = cr_overfetch
            agent.cosine_dedup = cosine_dedup
            agent._cosine_dedup_threshold = 0.95
            agent.cooc_boost_retrieval = cooc_boost
            agent.cooc_weight = 0.3
            agent.cooc_gate_threshold = 0.0
            agent.dream_weight = 0.0
            agent.bm25_weight = bm25_weight
            agent.ppr_retrieval = ppr
            agent.ppr_weight = 0.5
            agent.ppr_damping = 0.85
            agent.transfer_retrieval = False
            agent.transfer_k = None
            agent.coretrieval_retrieval = False
            agent.coretrieval_bonus = 0.3
            agent.coretrieval_min_count = 2.0
            agent.hybrid_retrieval = False
            agent.associative_retrieval = False
            agent.sparse_retrieval = False
            agent._belief_overfetch = 3

            if belief:
                agent._belief_index = BeliefIndex(
                    prior_alpha=2.0,
                    prior_beta=1.0,
                    propagation_damping=0.3,
                    hard_floor=0.05,
                )
                # Register all fake facts so belief scoring works
                for r in fake_results:
                    agent._belief_index.register_fact(r["text"])
            else:
                agent._belief_index = None

    return agent


# ===========================================================================
# Test Group 1: Extracted methods exist with correct signatures
# ===========================================================================


class TestRetrieveModularAPI:
    """Verify the extracted methods exist and have correct signatures."""

    def test_embed_query_exists(self):
        """HermesAgent instance has _embed_query method."""
        agent = _make_agent()
        assert hasattr(agent, "_embed_query"), (
            "_embed_query method not found on HermesMemoryAgent"
        )
        assert callable(agent._embed_query)

    def test_retrieve_candidates_exists(self):
        """HermesAgent instance has _retrieve_candidates method."""
        agent = _make_agent()
        assert hasattr(agent, "_retrieve_candidates"), (
            "_retrieve_candidates method not found on HermesMemoryAgent"
        )
        assert callable(agent._retrieve_candidates)

    def test_apply_post_retrieval_exists(self):
        """HermesAgent instance has _apply_post_retrieval method."""
        agent = _make_agent()
        assert hasattr(agent, "_apply_post_retrieval"), (
            "_apply_post_retrieval method not found on HermesMemoryAgent"
        )
        assert callable(agent._apply_post_retrieval)


# ===========================================================================
# Test Group 2: _embed_query behavior
# ===========================================================================


class TestEmbedQuery:
    """Verify _embed_query returns correct types and delegates to _scorer."""

    def test_embed_query_returns_numpy(self):
        """_embed_query returns np.ndarray."""
        agent = _make_agent()
        result = agent._embed_query("test query")
        assert isinstance(result, np.ndarray), (
            f"Expected np.ndarray, got {type(result)}"
        )

    def test_embed_query_calls_scorer(self):
        """_embed_query delegates to self._scorer.embed."""
        agent = _make_agent()
        agent._embed_query("hello world")
        agent._scorer.embed.assert_called_once_with("hello world")

    def test_embed_query_same_as_scorer_embed(self):
        """_embed_query returns exactly what _scorer.embed returns."""
        agent = _make_agent()
        direct = agent._scorer.embed("test")
        via_method = agent._embed_query("test")
        np.testing.assert_array_equal(direct, via_method)


# ===========================================================================
# Test Group 3: _retrieve_candidates behavior
# ===========================================================================


class TestRetrieveCandidates:
    """Verify _retrieve_candidates runs the correct backend and respects top_k."""

    def test_retrieve_candidates_returns_list(self):
        """_retrieve_candidates returns a list of dicts."""
        agent = _make_agent()
        q_emb = agent._scorer.embed("test")
        results = agent._retrieve_candidates(q_emb, 5)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_retrieve_candidates_uses_plain_query_by_default(self):
        """With no retrieval flags, _retrieve_candidates calls coupled_engine.query."""
        agent = _make_agent()
        q_emb = agent._scorer.embed("test")
        agent._retrieve_candidates(q_emb, 5)
        agent.coupled_engine.query.assert_called_once()

    def test_retrieve_candidates_uses_cooc_boost(self):
        """With cooc_boost flag, _retrieve_candidates calls query_cooc_boost."""
        agent = _make_agent(cooc_boost=True)
        q_emb = agent._scorer.embed("test")
        agent._retrieve_candidates(q_emb, 5)
        agent.coupled_engine.query_cooc_boost.assert_called_once()

    def test_retrieve_candidates_uses_cooc_ppr(self):
        """With cooc_boost + ppr flags, calls query_cooc_ppr."""
        agent = _make_agent(cooc_boost=True, ppr=True)
        q_emb = agent._scorer.embed("test")
        agent._retrieve_candidates(q_emb, 5)
        agent.coupled_engine.query_cooc_ppr.assert_called_once()

    def test_retrieve_candidates_uses_cooc_bm25(self):
        """With cooc_boost + bm25_weight > 0, calls query_cooc_bm25."""
        agent = _make_agent(cooc_boost=True, bm25_weight=0.3)
        q_emb = agent._scorer.embed("test")
        agent._retrieve_candidates(q_emb, 5)
        agent.coupled_engine.query_cooc_bm25.assert_called_once()

    def test_retrieve_candidates_cr_overfetch(self):
        """With conflict_resolution, effective_k is multiplied by cr_overfetch."""
        agent = _make_agent(conflict_resolution=True, cr_overfetch=3, retrieve_num=5)
        q_emb = agent._scorer.embed("test")
        agent._retrieve_candidates(q_emb, 5)
        # Should have called query with effective_k = 5 * 3 = 15
        call_kwargs = agent.coupled_engine.query.call_args
        assert call_kwargs[1]["top_k"] == 15 or call_kwargs.kwargs["top_k"] == 15

    def test_retrieve_candidates_belief_overfetch(self):
        """With belief active, effective_k is multiplied by belief_overfetch."""
        agent = _make_agent(belief=True, retrieve_num=5)
        q_emb = agent._scorer.embed("test")
        agent._retrieve_candidates(q_emb, 5)
        # effective_k = max(5, 5 * 3) = 15 (belief_overfetch=3)
        call_kwargs = agent.coupled_engine.query.call_args
        assert call_kwargs[1]["top_k"] == 15 or call_kwargs.kwargs["top_k"] == 15

    def test_retrieve_candidates_respects_top_k(self):
        """Passing different top_k values changes the effective_k passed to engine."""
        agent = _make_agent(retrieve_num=5)
        q_emb = agent._scorer.embed("test")

        agent._retrieve_candidates(q_emb, 3)
        call_kwargs_3 = agent.coupled_engine.query.call_args

        agent.coupled_engine.query.reset_mock()

        agent._retrieve_candidates(q_emb, 7)
        call_kwargs_7 = agent.coupled_engine.query.call_args

        # top_k should differ between calls
        k3 = call_kwargs_3[1].get("top_k") or call_kwargs_3.kwargs.get("top_k")
        k7 = call_kwargs_7[1].get("top_k") or call_kwargs_7.kwargs.get("top_k")
        assert k3 != k7, f"top_k should differ: {k3} vs {k7}"


# ===========================================================================
# Test Group 4: _apply_post_retrieval behavior
# ===========================================================================


class TestApplyPostRetrieval:
    """Verify _apply_post_retrieval handles reranking and truncation."""

    def test_apply_post_retrieval_handles_empty(self):
        """Empty list in -> empty list out."""
        agent = _make_agent()
        result = agent._apply_post_retrieval([])
        assert result == []

    def test_apply_post_retrieval_no_belief_passthrough(self):
        """Without belief, results pass through unchanged."""
        agent = _make_agent(belief=False)
        raw = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.8}]
        result = agent._apply_post_retrieval(raw)
        assert result == raw

    def test_apply_post_retrieval_truncates_with_belief(self):
        """With belief active, results are truncated to retrieve_num."""
        agent = _make_agent(belief=True, retrieve_num=3)
        raw = [
            {"text": f"fact_{i}", "score": 1.0 - i * 0.05}
            for i in range(10)
        ]
        # Register facts in belief so they aren't excluded
        for r in raw:
            agent._belief_index.register_fact(r["text"])
        result = agent._apply_post_retrieval(raw)
        assert len(result) <= 3

    def test_apply_post_retrieval_returns_list(self):
        """_apply_post_retrieval always returns a list of dicts."""
        agent = _make_agent()
        raw = [{"text": "x", "score": 0.5}]
        result = agent._apply_post_retrieval(raw)
        assert isinstance(result, list)


# ===========================================================================
# Test Group 5: Composition equivalence (MOST IMPORTANT)
# ===========================================================================


class TestCompositionEquivalence:
    """Verify _retrieve_v2(q) == _apply_post_retrieval(_retrieve_candidates(_embed_query(q), k))."""

    def test_retrieve_v2_composition_matches_plain(self):
        """Plain cosine path: composed stages match monolithic _retrieve_v2."""
        agent = _make_agent(retrieve_num=5)
        query = "What team does Abbiati play for?"

        # Monolithic
        result_mono = agent._retrieve_v2(query)

        # Reset mocks so the composed call gets fresh state
        agent.coupled_engine.query.reset_mock()
        agent._scorer.embed.reset_mock()

        # Composed
        q_emb = agent._embed_query(query)
        candidates = agent._retrieve_candidates(q_emb, agent.retrieve_num)
        result_composed = agent._apply_post_retrieval(candidates)

        assert result_mono == result_composed, (
            f"Monolithic and composed results differ:\n"
            f"  mono={result_mono}\n"
            f"  composed={result_composed}"
        )

    def test_retrieve_v2_composition_matches_cooc(self):
        """Cooc boost path: composed stages match monolithic _retrieve_v2."""
        agent = _make_agent(cooc_boost=True, retrieve_num=5)
        query = "What country is Milan in?"

        result_mono = agent._retrieve_v2(query)

        agent.coupled_engine.query_cooc_boost.reset_mock()
        agent._scorer.embed.reset_mock()

        q_emb = agent._embed_query(query)
        candidates = agent._retrieve_candidates(q_emb, agent.retrieve_num)
        result_composed = agent._apply_post_retrieval(candidates)

        assert result_mono == result_composed

    def test_retrieve_v2_composition_matches_with_belief(self):
        """Belief path: composed stages match monolithic _retrieve_v2."""
        agent = _make_agent(belief=True, retrieve_num=5)
        query = "Who is the CEO?"

        result_mono = agent._retrieve_v2(query)

        agent.coupled_engine.query.reset_mock()
        agent._scorer.embed.reset_mock()

        q_emb = agent._embed_query(query)
        candidates = agent._retrieve_candidates(q_emb, agent.retrieve_num)
        result_composed = agent._apply_post_retrieval(candidates)

        assert result_mono == result_composed

    def test_retrieve_v2_composition_matches_with_cr(self):
        """CR overfetch path: composed stages match monolithic _retrieve_v2."""
        agent = _make_agent(
            conflict_resolution=True, cr_overfetch=3, retrieve_num=5
        )
        query = "Where does Abbiati play?"

        result_mono = agent._retrieve_v2(query)

        agent.coupled_engine.query.reset_mock()
        agent._scorer.embed.reset_mock()

        q_emb = agent._embed_query(query)
        candidates = agent._retrieve_candidates(q_emb, agent.retrieve_num)
        result_composed = agent._apply_post_retrieval(candidates)

        assert result_mono == result_composed
