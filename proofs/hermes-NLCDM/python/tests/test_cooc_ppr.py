"""Tests for query_cooc_boost and query_ppr retrieval methods."""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

_NLCDM_PYTHON = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_NLCDM_PYTHON))

from coupled_engine import CoupledEngine


def _random_engine(
    n_facts: int = 20,
    dim: int = 32,
    n_sessions: int = 4,
    seed: int = 42,
    hebbian_epsilon: float = 0.05,
) -> CoupledEngine:
    """Create an engine with random facts grouped into sessions."""
    rng = np.random.RandomState(seed)
    engine = CoupledEngine(
        dim=dim,
        hebbian_epsilon=hebbian_epsilon,
        recency_alpha=0.1,
    )
    facts_per_session = n_facts // n_sessions
    idx = 0
    for s in range(n_sessions):
        for _ in range(facts_per_session):
            emb = rng.randn(dim)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"fact_{idx}", embedding=emb, recency=float(idx))
            idx += 1
        engine.flush_session()
    return engine


def _bridge_engine(dim: int = 32, seed: int = 42) -> tuple[CoupledEngine, dict]:
    """Create engine with two co-stored bridge facts and background noise."""
    rng = np.random.RandomState(seed)
    engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05, recency_alpha=0.1)

    # Background noise: 50 random facts in 5 sessions
    for s in range(5):
        for i in range(10):
            emb = rng.randn(dim)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"bg_{s}_{i}", embedding=emb, recency=float(s * 10 + i))
        engine.flush_session()

    # Bridge pair: orthogonal embeddings stored in same session
    emb_a = rng.randn(dim)
    emb_a /= np.linalg.norm(emb_a)
    emb_b = rng.randn(dim)
    # Make B orthogonal to A
    emb_b -= np.dot(emb_b, emb_a) * emb_a
    emb_b /= np.linalg.norm(emb_b)

    idx_a = engine.store(text="bridge_A_target", embedding=emb_a, recency=60.0)
    idx_b = engine.store(text="bridge_B_target", embedding=emb_b, recency=61.0)
    engine.flush_session()

    return engine, {
        "idx_a": idx_a,
        "idx_b": idx_b,
        "emb_a": emb_a,
        "emb_b": emb_b,
    }


# =====================================================================
# query_cooc_boost tests
# =====================================================================


class TestQueryCoocBoost:
    def test_returns_results(self):
        engine = _random_engine()
        q = np.random.randn(32)
        results = engine.query_cooc_boost(embedding=q, top_k=5)
        assert len(results) == 5
        for r in results:
            assert "index" in r
            assert "score" in r
            assert "text" in r

    def test_empty_store(self):
        engine = CoupledEngine(dim=32)
        q = np.random.randn(32)
        results = engine.query_cooc_boost(embedding=q, top_k=5)
        assert results == []

    def test_dimension_mismatch_raises(self):
        engine = _random_engine(dim=32)
        with pytest.raises(ValueError, match="dimension"):
            engine.query_cooc_boost(embedding=np.random.randn(64), top_k=5)

    def test_cooc_boost_promotes_partner(self):
        """Co-occurrence partner should rank higher with boost than without."""
        engine, info = _bridge_engine()

        # Query near fact_A — fact_B should be promoted by co-occurrence
        q = info["emb_a"] + 0.1 * np.random.randn(32)
        q /= np.linalg.norm(q)

        # Without boost
        results_no_boost = engine.query_cooc_boost(
            embedding=q, top_k=engine.n_memories, cooc_weight=0.0
        )
        # With boost
        results_with_boost = engine.query_cooc_boost(
            embedding=q, top_k=engine.n_memories, cooc_weight=1.0
        )

        # Find rank of bridge_B in each
        rank_no_boost = None
        rank_with_boost = None
        for i, r in enumerate(results_no_boost):
            if r["text"] == "bridge_B_target":
                rank_no_boost = i
                break
        for i, r in enumerate(results_with_boost):
            if r["text"] == "bridge_B_target":
                rank_with_boost = i
                break

        assert rank_no_boost is not None
        assert rank_with_boost is not None
        assert rank_with_boost < rank_no_boost, (
            f"Co-occurrence boost should promote partner: "
            f"rank {rank_with_boost} vs {rank_no_boost}"
        )

    def test_cooc_weight_zero_equals_cosine(self):
        """With cooc_weight=0, results should match cosine-only query."""
        engine = _random_engine()
        q = np.random.randn(32)

        results_cooc = engine.query_cooc_boost(
            embedding=q, top_k=10, cooc_weight=0.0
        )
        results_cosine = engine.query(embedding=q, top_k=10)

        # Same indices (order might differ slightly due to recency)
        cooc_indices = {r["index"] for r in results_cooc}
        cosine_indices = {r["index"] for r in results_cosine}
        # At least 8/10 should overlap (recency weighting may cause minor differences)
        overlap = len(cooc_indices & cosine_indices)
        assert overlap >= 8, f"Only {overlap}/10 overlap with cosine"

    def test_no_cooccurrence_links(self):
        """Without co-occurrence data, should still return results."""
        engine = CoupledEngine(dim=32)
        rng = np.random.RandomState(42)
        for i in range(10):
            emb = rng.randn(32)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"fact_{i}", embedding=emb)
        # No flush_session — no co-occurrence links

        q = rng.randn(32)
        results = engine.query_cooc_boost(embedding=q, top_k=5, cooc_weight=1.0)
        assert len(results) == 5

    def test_scores_sorted_descending(self):
        engine = _random_engine()
        q = np.random.randn(32)
        results = engine.query_cooc_boost(embedding=q, top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# =====================================================================
# query_ppr tests
# =====================================================================


class TestQueryPPR:
    def test_returns_results(self):
        engine = _random_engine()
        q = np.random.randn(32)
        results = engine.query_ppr(embedding=q, top_k=5)
        assert len(results) == 5
        for r in results:
            assert "index" in r
            assert "score" in r
            assert "text" in r

    def test_empty_store(self):
        engine = CoupledEngine(dim=32)
        q = np.random.randn(32)
        results = engine.query_ppr(embedding=q, top_k=5)
        assert results == []

    def test_dimension_mismatch_raises(self):
        engine = _random_engine(dim=32)
        with pytest.raises(ValueError, match="dimension"):
            engine.query_ppr(embedding=np.random.randn(64), top_k=5)

    def test_ppr_promotes_partner(self):
        """PPR should promote co-occurrence partners."""
        engine, info = _bridge_engine()
        q = info["emb_a"] + 0.1 * np.random.randn(32)
        q /= np.linalg.norm(q)

        # Without PPR signal
        results_no_ppr = engine.query_ppr(
            embedding=q, top_k=engine.n_memories, ppr_weight=0.0
        )
        # With PPR signal
        results_with_ppr = engine.query_ppr(
            embedding=q, top_k=engine.n_memories, ppr_weight=0.5
        )

        rank_no_ppr = None
        rank_with_ppr = None
        for i, r in enumerate(results_no_ppr):
            if r["text"] == "bridge_B_target":
                rank_no_ppr = i
                break
        for i, r in enumerate(results_with_ppr):
            if r["text"] == "bridge_B_target":
                rank_with_ppr = i
                break

        assert rank_no_ppr is not None
        assert rank_with_ppr is not None
        assert rank_with_ppr < rank_no_ppr, (
            f"PPR should promote partner: "
            f"rank {rank_with_ppr} vs {rank_no_ppr}"
        )

    def test_ppr_weight_zero_matches_cosine(self):
        """With ppr_weight=0, results should match cosine-only query."""
        engine = _random_engine()
        q = np.random.randn(32)

        results_ppr = engine.query_ppr(
            embedding=q, top_k=10, ppr_weight=0.0
        )
        results_cosine = engine.query(embedding=q, top_k=10)

        ppr_indices = {r["index"] for r in results_ppr}
        cosine_indices = {r["index"] for r in results_cosine}
        overlap = len(ppr_indices & cosine_indices)
        assert overlap >= 8, f"Only {overlap}/10 overlap with cosine"

    def test_damping_factor_range(self):
        """Different damping factors should all produce valid results."""
        engine = _random_engine()
        q = np.random.randn(32)
        for d in [0.1, 0.5, 0.85, 0.99]:
            results = engine.query_ppr(embedding=q, top_k=5, damping=d)
            assert len(results) == 5
            assert all(np.isfinite(r["score"]) for r in results)

    def test_ppr_steps_convergence(self):
        """More PPR steps should converge (similar results at 5 vs 20 steps)."""
        engine = _random_engine()
        q = np.random.randn(32)

        results_5 = engine.query_ppr(embedding=q, top_k=10, ppr_steps=5)
        results_20 = engine.query_ppr(embedding=q, top_k=10, ppr_steps=20)

        indices_5 = [r["index"] for r in results_5]
        indices_20 = [r["index"] for r in results_20]
        overlap = len(set(indices_5) & set(indices_20))
        assert overlap >= 7, f"PPR not converging: only {overlap}/10 overlap"

    def test_scores_sorted_descending(self):
        engine = _random_engine()
        q = np.random.randn(32)
        results = engine.query_ppr(embedding=q, top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_cooccurrence_links(self):
        """Without co-occurrence data, PPR should still return results."""
        engine = CoupledEngine(dim=32)
        rng = np.random.RandomState(42)
        for i in range(10):
            emb = rng.randn(32)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"fact_{i}", embedding=emb)

        q = rng.randn(32)
        results = engine.query_ppr(embedding=q, top_k=5)
        assert len(results) == 5


# =====================================================================
# Comparative tests
# =====================================================================


class TestCoocVsPPR:
    def test_both_promote_partner_over_cosine(self):
        """Both methods should promote co-occurrence partners vs pure cosine."""
        engine, info = _bridge_engine(seed=123)
        q = info["emb_a"] + 0.05 * np.random.RandomState(123).randn(32)
        q /= np.linalg.norm(q)

        results_cosine = engine.query(embedding=q, top_k=engine.n_memories)
        results_cooc = engine.query_cooc_boost(
            embedding=q, top_k=engine.n_memories, cooc_weight=0.5
        )
        results_ppr = engine.query_ppr(
            embedding=q, top_k=engine.n_memories, ppr_weight=0.5
        )

        def rank_of_b(results):
            for i, r in enumerate(results):
                if r["text"] == "bridge_B_target":
                    return i
            return len(results)

        rank_cos = rank_of_b(results_cosine)
        rank_cooc = rank_of_b(results_cooc)
        rank_ppr = rank_of_b(results_ppr)

        assert rank_cooc < rank_cos, (
            f"cooc_boost should beat cosine: {rank_cooc} vs {rank_cos}"
        )
        assert rank_ppr < rank_cos, (
            f"PPR should beat cosine: {rank_ppr} vs {rank_cos}"
        )

    def test_both_handle_large_store(self):
        """Both methods should handle 100+ facts without error."""
        engine = _random_engine(n_facts=100, n_sessions=10)
        q = np.random.randn(32)

        results_cooc = engine.query_cooc_boost(embedding=q, top_k=10)
        results_ppr = engine.query_ppr(embedding=q, top_k=10)

        assert len(results_cooc) == 10
        assert len(results_ppr) == 10
