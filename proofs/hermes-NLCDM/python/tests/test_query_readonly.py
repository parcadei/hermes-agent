"""Tests for CoupledEngine.query_readonly() — read-only retrieval.

Defines the behavioral contract for query_readonly() BEFORE implementation.
All tests should FAIL with AttributeError until query_readonly() is implemented.

Contract:
  - Returns identical results (top-k indices, scores, text) as query()
  - DOES log co-retrieval edges (_co_retrieval graph)
  - DOES increment _co_retrieval_query_count
  - Does NOT mutate access_count, importance, last_access_time
  - Does NOT trigger reconsolidation (no embedding drift)
  - Does NOT invalidate caches
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
sys.path.insert(0, str(_NLCDM_PYTHON))

from coupled_engine import CoupledEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(dim: int = 128, n: int = 5, seed: int = 42) -> CoupledEngine:
    """Create a CoupledEngine with n random unit-norm patterns stored."""
    rng = np.random.default_rng(seed)
    engine = CoupledEngine(dim=dim)
    for i in range(n):
        emb = rng.standard_normal(dim)
        emb /= np.linalg.norm(emb)
        engine.store(text=f"Fact {i}", embedding=emb)
    return engine


def _random_query(dim: int = 128, seed: int = 99) -> np.ndarray:
    """Generate a random unit-norm query embedding."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim)
    q /= np.linalg.norm(q)
    return q


# ---------------------------------------------------------------------------
# Test: query_readonly returns same results as query
# ---------------------------------------------------------------------------


class TestQueryReadonlyResultEquivalence:
    """query_readonly() must return the same top-k indices, scores, and text
    as query() on the same engine state."""

    def test_same_results_as_query(self):
        """query_readonly and query return identical top-k results.

        Strategy: deep-copy engine, call query_readonly on original,
        call query on copy. Compare index/score/text.
        NOTE: query_readonly called first because query() mutates state.
        """
        engine = _make_engine(dim=128, n=10, seed=42)
        query_emb = _random_query(dim=128, seed=99)

        # Deep-copy engine for the mutating query() call
        engine_copy = copy.deepcopy(engine)

        # Call query_readonly FIRST (no side effects on scoring state)
        results_readonly = engine.query_readonly(query_emb, top_k=5)

        # Call regular query on the copy (which mutates the copy)
        results_regular = engine_copy.query(query_emb, top_k=5)

        assert len(results_readonly) == len(results_regular)
        for r_ro, r_rg in zip(results_readonly, results_regular):
            assert r_ro["index"] == r_rg["index"], (
                f"Index mismatch: readonly={r_ro['index']} vs query={r_rg['index']}"
            )
            assert abs(r_ro["score"] - r_rg["score"]) < 1e-9, (
                f"Score mismatch: readonly={r_ro['score']} vs query={r_rg['score']}"
            )
            assert r_ro["text"] == r_rg["text"], (
                f"Text mismatch: readonly={r_ro['text']!r} vs query={r_rg['text']!r}"
            )

    def test_result_dict_keys(self):
        """Each result dict has 'index', 'score', 'text' keys."""
        engine = _make_engine(dim=64, n=3, seed=10)
        q = _random_query(dim=64, seed=20)
        results = engine.query_readonly(q, top_k=3)
        assert len(results) > 0
        for r in results:
            assert "index" in r
            assert "score" in r
            assert "text" in r
            assert isinstance(r["index"], int)
            assert isinstance(r["score"], float)
            assert isinstance(r["text"], str)


# ---------------------------------------------------------------------------
# Test: query_readonly does NOT mutate access_count
# ---------------------------------------------------------------------------


class TestQueryReadonlyNoAccessCountMutation:
    """query_readonly() must NOT increment access_count on any pattern."""

    def test_access_count_unchanged(self):
        engine = _make_engine(dim=128, n=5, seed=42)
        q = _random_query(dim=128, seed=99)

        access_counts_before = [m.access_count for m in engine.memory_store]

        engine.query_readonly(q, top_k=5)

        access_counts_after = [m.access_count for m in engine.memory_store]
        assert access_counts_after == access_counts_before, (
            f"access_count changed: before={access_counts_before}, "
            f"after={access_counts_after}"
        )

    def test_access_count_unchanged_multiple_calls(self):
        """Multiple query_readonly calls should not accumulate access_count."""
        engine = _make_engine(dim=64, n=8, seed=55)
        access_counts_before = [m.access_count for m in engine.memory_store]

        for seed in range(10):
            q = _random_query(dim=64, seed=seed)
            engine.query_readonly(q, top_k=4)

        access_counts_after = [m.access_count for m in engine.memory_store]
        assert access_counts_after == access_counts_before


# ---------------------------------------------------------------------------
# Test: query_readonly does NOT mutate importance
# ---------------------------------------------------------------------------


class TestQueryReadonlyNoImportanceMutation:
    """query_readonly() must NOT update importance on any pattern."""

    def test_importance_unchanged(self):
        engine = _make_engine(dim=128, n=5, seed=42)
        q = _random_query(dim=128, seed=99)

        importances_before = [m.importance for m in engine.memory_store]

        engine.query_readonly(q, top_k=5)

        importances_after = [m.importance for m in engine.memory_store]
        assert importances_after == importances_before, (
            f"importance changed: before={importances_before}, "
            f"after={importances_after}"
        )


# ---------------------------------------------------------------------------
# Test: query_readonly does NOT mutate last_access_time
# ---------------------------------------------------------------------------


class TestQueryReadonlyNoLastAccessTimeMutation:
    """query_readonly() must NOT update last_access_time on any pattern."""

    def test_last_access_time_unchanged(self):
        engine = _make_engine(dim=128, n=5, seed=42)
        q = _random_query(dim=128, seed=99)

        # Allow store-time timestamps to settle
        last_access_before = [m.last_access_time for m in engine.memory_store]

        # Small sleep to ensure any time.time() call would produce a different value
        time.sleep(0.01)

        engine.query_readonly(q, top_k=5)

        last_access_after = [m.last_access_time for m in engine.memory_store]
        assert last_access_after == last_access_before, (
            f"last_access_time changed: before={last_access_before}, "
            f"after={last_access_after}"
        )


# ---------------------------------------------------------------------------
# Test: query_readonly DOES log co-retrieval edges
# ---------------------------------------------------------------------------


class TestQueryReadonlyLogsCoRetrieval:
    """query_readonly() MUST log co-retrieval edges (the whole point)."""

    def test_co_retrieval_edges_created(self):
        """After query_readonly with top_k >= 2, co-retrieval edges should exist."""
        engine = _make_engine(dim=128, n=5, seed=42)

        # Verify co_retrieval graph is empty initially
        assert len(engine._co_retrieval) == 0, (
            "co_retrieval should be empty before any queries"
        )

        # Use a query embedding that is the mean of first two patterns
        # to ensure both are retrieved
        emb0 = engine.memory_store[0].embedding
        emb1 = engine.memory_store[1].embedding
        q = (emb0 + emb1) / 2.0
        q /= np.linalg.norm(q)

        results = engine.query_readonly(q, top_k=5)
        assert len(results) >= 2, "Should retrieve at least 2 results"

        # Extract top-k indices from results
        result_indices = [r["index"] for r in results]

        # co_retrieval should now have edges between the retrieved indices
        assert len(engine._co_retrieval) > 0, (
            "co_retrieval should have edges after query_readonly"
        )

        # Verify bidirectional edge exists between first two retrieved indices
        a, b = result_indices[0], result_indices[1]
        assert a in engine._co_retrieval, f"Node {a} missing from co_retrieval"
        assert b in engine._co_retrieval[a], f"Edge {a}->{b} missing"
        assert a in engine._co_retrieval[b], f"Edge {b}->{a} missing"

    def test_co_retrieval_query_count_incremented(self):
        """_co_retrieval_query_count should increment by 1 per call."""
        engine = _make_engine(dim=64, n=5, seed=42)

        assert engine._co_retrieval_query_count == 0

        q = _random_query(dim=64, seed=99)
        engine.query_readonly(q, top_k=3)
        assert engine._co_retrieval_query_count == 1

        engine.query_readonly(q, top_k=3)
        assert engine._co_retrieval_query_count == 2

    def test_co_retrieval_weights_accumulate(self):
        """Repeated query_readonly calls should accumulate edge weights."""
        engine = _make_engine(dim=128, n=2, seed=42)

        # Query that retrieves both patterns
        emb0 = engine.memory_store[0].embedding
        emb1 = engine.memory_store[1].embedding
        q = (emb0 + emb1) / 2.0
        q /= np.linalg.norm(q)

        engine.query_readonly(q, top_k=2)
        assert engine._co_retrieval[0][1] == 1.0

        engine.query_readonly(q, top_k=2)
        assert engine._co_retrieval[0][1] == 2.0

        engine.query_readonly(q, top_k=2)
        assert engine._co_retrieval[0][1] == 3.0


# ---------------------------------------------------------------------------
# Test: query_readonly does NOT trigger reconsolidation (no embedding drift)
# ---------------------------------------------------------------------------


class TestQueryReadonlyNoReconsolidation:
    """query_readonly() must NOT trigger reconsolidation (embedding drift),
    even when the engine has reconsolidation=True."""

    def test_embeddings_unchanged(self):
        """Pattern embeddings must not change after query_readonly."""
        engine = _make_engine(dim=128, n=5, seed=42)

        # Copy all embeddings before
        embeddings_before = [m.embedding.copy() for m in engine.memory_store]

        q = _random_query(dim=128, seed=99)
        engine.query_readonly(q, top_k=5)

        # Verify embeddings are identical
        for i, m in enumerate(engine.memory_store):
            np.testing.assert_array_equal(
                m.embedding, embeddings_before[i],
                err_msg=f"Pattern {i} embedding changed after query_readonly"
            )

    def test_embeddings_unchanged_with_reconsolidation_enabled(self):
        """Even with reconsolidation=True, query_readonly must not drift embeddings."""
        engine = CoupledEngine(dim=64, reconsolidation=True, reconsolidation_eta=0.1)
        rng = np.random.default_rng(42)
        for i in range(5):
            emb = rng.standard_normal(64)
            emb /= np.linalg.norm(emb)
            engine.store(text=f"Fact {i}", embedding=emb)

        embeddings_before = [m.embedding.copy() for m in engine.memory_store]

        q = _random_query(dim=64, seed=99)
        engine.query_readonly(q, top_k=5)

        for i, m in enumerate(engine.memory_store):
            np.testing.assert_allclose(
                m.embedding, embeddings_before[i], atol=1e-15,
                err_msg=(
                    f"Pattern {i} embedding drifted after query_readonly "
                    f"with reconsolidation=True"
                )
            )

    def test_caches_not_invalidated(self):
        """query_readonly should NOT invalidate _embeddings_cache or _W_cache.

        Since no embeddings change, cache invalidation is unnecessary and wasteful.
        """
        engine = _make_engine(dim=64, n=5, seed=42)

        # Force cache computation
        _ = engine._embeddings_matrix()
        _ = engine.W
        cache_id_emb = id(engine._embeddings_cache)
        cache_id_W = id(engine._W_cache)

        q = _random_query(dim=64, seed=99)
        engine.query_readonly(q, top_k=3)

        # Caches should be the same objects (not re-computed)
        assert id(engine._embeddings_cache) == cache_id_emb, (
            "_embeddings_cache was invalidated by query_readonly"
        )
        assert id(engine._W_cache) == cache_id_W, (
            "_W_cache was invalidated by query_readonly"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestQueryReadonlyEdgeCases:
    """Edge cases from the spec: empty store, single pattern, zero-norm query,
    dimension mismatch."""

    def test_empty_store_returns_empty(self):
        """query_readonly on empty engine returns []."""
        engine = CoupledEngine(dim=128)
        q = _random_query(dim=128)
        results = engine.query_readonly(q, top_k=10)
        assert results == []
        assert len(engine._co_retrieval) == 0

    def test_single_pattern_no_edges(self):
        """With 1 stored pattern, no co-retrieval edges (need >= 2 nodes)."""
        engine = CoupledEngine(dim=64)
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(64)
        emb /= np.linalg.norm(emb)
        engine.store("Only fact", emb)

        results = engine.query_readonly(emb, top_k=10)
        assert len(results) == 1
        assert results[0]["text"] == "Only fact"
        # No co-retrieval edges with only 1 result
        assert len(engine._co_retrieval) == 0

    def test_zero_norm_query_returns_empty(self):
        """Zero-norm query embedding should return []."""
        engine = _make_engine(dim=64, n=3, seed=42)
        zero_q = np.zeros(64)
        results = engine.query_readonly(zero_q, top_k=5)
        assert results == []

    def test_dimension_mismatch_raises_valueerror(self):
        """Query with wrong dimension should raise ValueError."""
        engine = _make_engine(dim=64, n=3, seed=42)
        wrong_dim_q = np.ones(32)
        with pytest.raises(ValueError, match="dimension"):
            engine.query_readonly(wrong_dim_q, top_k=5)

    def test_top_k_exceeds_n_memories(self):
        """top_k > n_memories should return all memories without error."""
        engine = _make_engine(dim=64, n=3, seed=42)
        q = _random_query(dim=64)
        results = engine.query_readonly(q, top_k=100)
        assert len(results) == 3

    def test_default_beta_used_when_none(self):
        """When beta=None, should use engine's default beta."""
        engine = _make_engine(dim=64, n=5, seed=42)
        q = _random_query(dim=64)

        # Explicit beta=None
        results_none = engine.query_readonly(q, beta=None, top_k=3)

        # Explicit beta matching engine default
        results_explicit = engine.query_readonly(q, beta=engine.beta, top_k=3)

        assert len(results_none) == len(results_explicit)
        for r1, r2 in zip(results_none, results_explicit):
            assert r1["index"] == r2["index"]
            assert abs(r1["score"] - r2["score"]) < 1e-12
