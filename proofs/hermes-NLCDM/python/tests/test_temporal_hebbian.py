"""Behavioral contracts for temporal Hebbian binding in CoupledEngine.

Tests define expected behavior BEFORE implementation. All 12 contracts from
the temporal-hebbian-spec plus 2 backward-compatibility tests for the biased
dream_ops functions.

Expected failures on first run:
  - AttributeError for missing _session_buffer, _W_temporal, hebbian_epsilon
  - AttributeError for missing flush_session(), reset_temporal()
  - ImportError for missing hopfield_update_biased, spreading_activation_biased
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

# Ensure the parent directory (where coupled_engine.py lives) is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from coupled_engine import CoupledEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_unit_vector(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def make_orthogonal_patterns(n: int, dim: int, seed: int) -> np.ndarray:
    """Create n nearly-orthogonal unit vectors in R^dim."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim))
    # QR gives orthonormal columns
    Q, _ = np.linalg.qr(vecs.T)
    return Q.T[:n]  # (n, dim) orthonormal rows


# ---------------------------------------------------------------------------
# Contract 1: Session buffering accumulates without side effects
# ---------------------------------------------------------------------------


class TestContract1SessionBuffering:
    """store() appends to _session_buffer without touching _W_temporal."""

    def test_session_buffer_accumulates(self):
        engine = CoupledEngine(dim=8)
        a = random_unit_vector(8, seed=0)
        b = random_unit_vector(8, seed=1)

        engine.store("fact_a", a)
        engine.store("fact_b", b)

        # Session buffer has 2 embeddings
        assert len(engine._session_buffer) == 2
        # W_temporal is still zero -- no flush yet
        assert np.allclose(engine._W_temporal, 0.0)
        # W (auto-derived) is unaffected by session buffer
        W_before = engine.W.copy()
        # W should just be outer(a,a) + outer(b,b) with zeroed diagonal
        expected = np.outer(a, a) + np.outer(b, b)
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(engine.W, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Contract 2: Flush creates symmetric cross-links
# ---------------------------------------------------------------------------


class TestContract2FlushCrossLinks:
    """flush_session() creates epsilon*(outer(a,b)+outer(b,a)) in _W_temporal."""

    def test_flush_creates_symmetric_cross_links(self):
        engine = CoupledEngine(dim=8, hebbian_epsilon=0.1)
        a = random_unit_vector(8, seed=0)
        b = random_unit_vector(8, seed=1)

        engine.store("fact_a", a)
        engine.store("fact_b", b)
        n_pairs = engine.flush_session()

        assert n_pairs == 1  # C(2,2) = 1 pair
        expected = 0.1 * (np.outer(a, b) + np.outer(b, a))
        np.testing.assert_allclose(engine._W_temporal, expected, atol=1e-12)
        # Symmetry
        np.testing.assert_allclose(
            engine._W_temporal, engine._W_temporal.T, atol=1e-12
        )


# ---------------------------------------------------------------------------
# Contract 3: Flush clears buffer
# ---------------------------------------------------------------------------


class TestContract3FlushClearsBuffer:
    """After flush, _session_buffer is empty."""

    def test_flush_clears_buffer(self):
        engine = CoupledEngine(dim=8)
        engine.store("a", random_unit_vector(8, seed=0))
        engine.store("b", random_unit_vector(8, seed=1))
        assert len(engine._session_buffer) == 2

        engine.flush_session()
        assert len(engine._session_buffer) == 0


# ---------------------------------------------------------------------------
# Contract 4: Additive accumulation across flushes
# ---------------------------------------------------------------------------


class TestContract4AdditiveAccumulation:
    """Multiple flushes stack in _W_temporal."""

    def test_multiple_flushes_accumulate(self):
        engine = CoupledEngine(dim=8, hebbian_epsilon=0.1)

        # Session 1: facts a, b
        a = random_unit_vector(8, seed=0)
        b = random_unit_vector(8, seed=1)
        engine.store("a", a)
        engine.store("b", b)
        engine.flush_session()
        W_after_1 = engine._W_temporal.copy()

        # Session 2: facts c, d
        c = random_unit_vector(8, seed=2)
        d = random_unit_vector(8, seed=3)
        engine.store("c", c)
        engine.store("d", d)
        engine.flush_session()
        W_after_2 = engine._W_temporal.copy()

        # W_temporal grew
        assert np.linalg.norm(W_after_2) > np.linalg.norm(W_after_1)
        # First session's contribution is still present
        expected_1 = 0.1 * (np.outer(a, b) + np.outer(b, a))
        expected_2 = expected_1 + 0.1 * (np.outer(c, d) + np.outer(d, c))
        np.testing.assert_allclose(engine._W_temporal, expected_2, atol=1e-12)


# ---------------------------------------------------------------------------
# Contract 5: Epsilon linearity
# ---------------------------------------------------------------------------


class TestContract5EpsilonLinearity:
    """5x epsilon produces 5x Frobenius norm."""

    def test_epsilon_scales_linearly(self):
        dim = 8
        a = random_unit_vector(dim, seed=0)
        b = random_unit_vector(dim, seed=1)

        engine_1x = CoupledEngine(dim=dim, hebbian_epsilon=0.01)
        engine_1x.store("a", a)
        engine_1x.store("b", b)
        engine_1x.flush_session()

        engine_5x = CoupledEngine(dim=dim, hebbian_epsilon=0.05)
        engine_5x.store("a", a)
        engine_5x.store("b", b)
        engine_5x.flush_session()

        ratio = np.linalg.norm(engine_5x._W_temporal) / np.linalg.norm(
            engine_1x._W_temporal
        )
        np.testing.assert_allclose(ratio, 5.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Contract 6: Temporal link discovery via spreading activation
# ---------------------------------------------------------------------------


class TestContract6TemporalLinkDiscovery:
    """query_associative finds co-stored orthogonal facts."""

    def test_query_associative_finds_temporal_link(self):
        """Two facts with low cosine similarity but co-stored should be linked."""
        dim = 32
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

        # Create two nearly orthogonal vectors (low cosine similarity)
        patterns = make_orthogonal_patterns(4, dim, seed=42)
        a, b, c, d = patterns[0], patterns[1], patterns[2], patterns[3]

        # Store a and b in the same session (co-occurring)
        engine.store("capital of France is Paris", a)
        engine.store("France population is 67 million", b)
        engine.flush_session()

        # Store c and d in a different session
        engine.store("Japan GDP is high", c)
        engine.store("Brazil has rainforests", d)
        engine.flush_session()

        # Query with a: without temporal, b has ~0 cosine to a (orthogonal)
        # With temporal binding, b should be boosted in associative results
        results_assoc = engine.query_associative(a, top_k=4, sparse=True)
        result_indices = [r["index"] for r in results_assoc]

        # b (index 1) should appear in top results due to temporal link
        assert 1 in result_indices[:2], (
            f"Temporal co-occurrence should surface b; got indices {result_indices}"
        )


# ---------------------------------------------------------------------------
# Contract 7: Cosine query unchanged by W_temporal
# ---------------------------------------------------------------------------


class TestContract7CosineQueryUnchanged:
    """query() scores identical with/without _W_temporal."""

    def test_cosine_query_unaffected(self):
        """query() uses direct cosine, not spreading activation -- W_temporal irrelevant."""
        dim = 16
        a = random_unit_vector(dim, seed=0)
        b = random_unit_vector(dim, seed=1)

        # Engine without temporal
        engine_base = CoupledEngine(dim=dim)
        engine_base.store("a", a)
        engine_base.store("b", b)

        # Engine with temporal bindings
        engine_temporal = CoupledEngine(dim=dim, hebbian_epsilon=0.1)
        engine_temporal.store("a", a)
        engine_temporal.store("b", b)
        engine_temporal.flush_session()

        query = random_unit_vector(dim, seed=99)
        results_base = engine_base.query(query, top_k=2)
        results_temporal = engine_temporal.query(query, top_k=2)

        # Same scores (query() does NOT use W or W_temporal)
        for rb, rt in zip(results_base, results_temporal):
            np.testing.assert_allclose(rb["score"], rt["score"], atol=1e-12)


# ---------------------------------------------------------------------------
# Contract 8: Dream cycle does not corrupt W_temporal
# ---------------------------------------------------------------------------


class TestContract8DreamCompatibility:
    """_W_temporal unchanged after dream()."""

    def test_dream_preserves_W_temporal(self):
        dim = 16
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

        # Store and flush 3 sessions
        for session in range(3):
            for i in range(4):
                emb = random_unit_vector(dim, seed=session * 100 + i)
                engine.store(f"s{session}_f{i}", emb)
            engine.flush_session()

        W_temporal_before = engine._W_temporal.copy()
        engine.dream()
        W_temporal_after = engine._W_temporal

        # W_temporal is in embedding space (d,d), not index space -- dream does not touch it
        np.testing.assert_array_equal(W_temporal_before, W_temporal_after)


# ---------------------------------------------------------------------------
# Contract 9: Save/load roundtrip preserves W_temporal
# ---------------------------------------------------------------------------


class TestContract9SaveLoadRoundtrip:
    """_W_temporal and hebbian_epsilon survive save/load."""

    def test_save_load_roundtrip(self, tmp_path):
        dim = 16
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

        a = random_unit_vector(dim, seed=0)
        b = random_unit_vector(dim, seed=1)
        engine.store("a", a)
        engine.store("b", b)
        engine.flush_session()

        save_path = tmp_path / "engine_test"
        engine.save(save_path)
        loaded = CoupledEngine.load(save_path)

        np.testing.assert_allclose(
            loaded._W_temporal, engine._W_temporal, atol=1e-12
        )
        assert loaded.hebbian_epsilon == engine.hebbian_epsilon
        assert len(loaded._session_buffer) == 0  # buffer is transient, not saved


# ---------------------------------------------------------------------------
# Contract 10: Edge cases
# ---------------------------------------------------------------------------


class TestContract10EdgeCases:
    """0 facts -> 0 pairs, 1 fact -> 0 pairs, 3 facts -> 3 pairs, N facts -> N*(N-1)/2 pairs."""

    def test_flush_empty_buffer_is_noop(self):
        engine = CoupledEngine(dim=8)
        n = engine.flush_session()
        assert n == 0
        assert np.allclose(engine._W_temporal, 0.0)

    def test_flush_single_fact_is_noop(self):
        engine = CoupledEngine(dim=8)
        engine.store("solo", random_unit_vector(8, seed=0))
        n = engine.flush_session()
        assert n == 0
        assert np.allclose(engine._W_temporal, 0.0)
        assert len(engine._session_buffer) == 0

    def test_flush_three_facts_creates_three_pairs(self):
        engine = CoupledEngine(dim=8, hebbian_epsilon=0.1)
        a = random_unit_vector(8, seed=0)
        b = random_unit_vector(8, seed=1)
        c = random_unit_vector(8, seed=2)
        engine.store("a", a)
        engine.store("b", b)
        engine.store("c", c)
        n = engine.flush_session()
        assert n == 3  # C(3,2) = 3 pairs: (a,b), (a,c), (b,c)

        expected = 0.1 * (
            np.outer(a, b)
            + np.outer(b, a)
            + np.outer(a, c)
            + np.outer(c, a)
            + np.outer(b, c)
            + np.outer(c, b)
        )
        np.testing.assert_allclose(engine._W_temporal, expected, atol=1e-12)

    def test_flush_n_facts_creates_n_choose_2_pairs(self):
        """Verify combinatorial count for larger sessions."""
        engine = CoupledEngine(dim=8, hebbian_epsilon=0.01)
        N = 10
        for i in range(N):
            engine.store(f"fact_{i}", random_unit_vector(8, seed=i))
        n = engine.flush_session()
        assert n == N * (N - 1) // 2  # 45 pairs


# ---------------------------------------------------------------------------
# Contract 11: Contradiction replacement uses new embedding in session buffer
# ---------------------------------------------------------------------------


class TestContract11ContradictionReplacement:
    """NEW embedding (not old) enters session buffer on contradiction replacement."""

    def test_contradiction_replacement_buffers_new_embedding(self):
        """When store() replaces a contradicting entry, the NEW embedding enters the buffer."""
        dim = 16
        engine = CoupledEngine(
            dim=dim,
            hebbian_epsilon=0.1,
            contradiction_aware=True,
            contradiction_threshold=0.8,
        )
        # Store initial fact
        a = random_unit_vector(dim, seed=0)
        engine.store("Paris is in France", a)
        engine.flush_session()  # clear session 1

        # Store a very similar embedding (above contradiction threshold)
        # that should trigger replacement
        a_updated = a + 0.05 * random_unit_vector(dim, seed=99)
        a_updated /= np.linalg.norm(a_updated)

        b = random_unit_vector(dim, seed=1)
        engine.store("Paris is the capital of France", a_updated)  # replaces index 0
        engine.store("Berlin is in Germany", b)
        engine.flush_session()

        # Session buffer should have contained a_updated (not a) and b
        # So W_temporal should encode outer(a_updated, b) + outer(b, a_updated)
        expected_temporal = 0.1 * (np.outer(a_updated, b) + np.outer(b, a_updated))
        # Plus the session-1 contribution (which was just a single fact -> no pairs)
        np.testing.assert_allclose(
            engine._W_temporal, expected_temporal, atol=1e-10
        )


# ---------------------------------------------------------------------------
# Contract 12: reset_temporal clears everything
# ---------------------------------------------------------------------------


class TestContract12ResetTemporal:
    """reset_temporal() zeros _W_temporal and clears buffer."""

    def test_reset_temporal(self):
        engine = CoupledEngine(dim=8, hebbian_epsilon=0.1)
        engine.store("a", random_unit_vector(8, seed=0))
        engine.store("b", random_unit_vector(8, seed=1))
        engine.flush_session()

        assert not np.allclose(engine._W_temporal, 0.0)  # has bindings

        engine.reset_temporal()
        assert np.allclose(engine._W_temporal, 0.0)
        assert len(engine._session_buffer) == 0


# ---------------------------------------------------------------------------
# Contract 13: hopfield_update_biased backward compat (zero W_temporal)
# ---------------------------------------------------------------------------


class TestContract13HopfieldBiasedBackwardCompat:
    """hopfield_update_biased with zero W_temporal equals hopfield_update."""

    def test_biased_with_zero_temporal_equals_unbiased(self):
        from dream_ops import (
            hopfield_update,
            hopfield_update_biased,
        )

        dim = 16
        rng = np.random.default_rng(42)
        patterns = rng.standard_normal((5, dim))
        # Normalize rows
        patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
        xi = rng.standard_normal(dim)
        xi = xi / np.linalg.norm(xi)

        W_zero = np.zeros((dim, dim), dtype=np.float64)
        beta = 5.0

        result_base = hopfield_update(beta, patterns, xi)
        result_biased = hopfield_update_biased(
            beta, patterns, xi, W_temporal=W_zero
        )

        np.testing.assert_allclose(result_biased, result_base, atol=1e-12)


# ---------------------------------------------------------------------------
# Contract 14: spreading_activation_biased backward compat (zero W_temporal)
# ---------------------------------------------------------------------------


class TestContract14SpreadingActivationBiasedBackwardCompat:
    """spreading_activation_biased with zero W_temporal equals spreading_activation."""

    def test_biased_with_zero_temporal_equals_unbiased(self):
        from dream_ops import (
            spreading_activation,
            spreading_activation_biased,
        )

        dim = 16
        rng = np.random.default_rng(42)
        patterns = rng.standard_normal((5, dim))
        patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
        xi = rng.standard_normal(dim)
        xi = xi / np.linalg.norm(xi)

        W_zero = np.zeros((dim, dim), dtype=np.float64)
        beta = 5.0

        result_base = spreading_activation(
            beta, patterns, xi, max_steps=50, tol=1e-6
        )
        result_biased = spreading_activation_biased(
            beta, patterns, xi, W_temporal=W_zero, max_steps=50, tol=1e-6
        )

        np.testing.assert_allclose(result_biased, result_base, atol=1e-12)


# ---------------------------------------------------------------------------
# Two-Hop Co-occurrence Retrieval
# ---------------------------------------------------------------------------


class TestTwoHopRetrieval:
    """Two-hop retrieval: cosine -> co-occurrence expansion -> score union."""

    def test_twohop_finds_co_stored_facts(self):
        """Two-hop should find facts that were co-stored even if cosine is low."""
        dim = 32
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

        # Create orthogonal patterns
        patterns = make_orthogonal_patterns(6, dim, seed=42)

        # Session 1: store a, b together (co-occurring)
        engine.store("fact_a", patterns[0])
        engine.store("fact_b", patterns[1])
        engine.flush_session()

        # Session 2: store c, d together
        engine.store("fact_c", patterns[2])
        engine.store("fact_d", patterns[3])
        engine.flush_session()

        # Session 3: standalone facts
        engine.store("fact_e", patterns[4])
        engine.store("fact_f", patterns[5])
        engine.flush_session()

        # Query with pattern close to a -- should find b via co-occurrence
        query = patterns[0] + 0.1 * np.random.default_rng(42).standard_normal(dim)
        query /= np.linalg.norm(query)

        # Cosine: should find a (close) but not b (orthogonal)
        cos_results = engine.query(embedding=query, top_k=3)
        cos_indices = [r["index"] for r in cos_results]
        assert 0 in cos_indices  # a is found

        # Two-hop: should find both a AND b
        twohop_results = engine.query_twohop(embedding=query, top_k=6)
        twohop_indices = [r["index"] for r in twohop_results]
        assert 0 in twohop_indices  # a found (cosine)
        assert 1 in twohop_indices  # b found (co-occurrence expansion)

    def test_twohop_without_co_occurrence_equals_cosine(self):
        """Without any flush_session calls, twohop is same as cosine."""
        dim = 16
        engine = CoupledEngine(dim=dim)
        a = random_unit_vector(dim, seed=0)
        b = random_unit_vector(dim, seed=1)
        engine.store("a", a)
        engine.store("b", b)
        # Don't flush -- no co-occurrence links

        query = random_unit_vector(dim, seed=99)
        cos_r = engine.query(embedding=query, top_k=2)
        twohop_r = engine.query_twohop(embedding=query, top_k=2)

        # Same results since no expansion happens
        for cr, tr in zip(cos_r, twohop_r):
            assert cr["index"] == tr["index"]

    def test_co_occurrence_survives_save_load(self, tmp_path):
        """Co-occurrence links persist through save/load cycle."""
        dim = 16
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)
        engine.store("a", random_unit_vector(dim, seed=0))
        engine.store("b", random_unit_vector(dim, seed=1))
        engine.flush_session()

        assert 0 in engine._co_occurrence
        assert 1 in engine._co_occurrence[0]

        path = tmp_path / "test_engine"
        engine.save(path)
        loaded = CoupledEngine.load(path)

        assert 0 in loaded._co_occurrence
        assert 1 in loaded._co_occurrence[0]

    def test_session_indices_tracked_correctly(self):
        """Each store() appends to _session_indices."""
        dim = 8
        engine = CoupledEngine(dim=dim)
        engine.store("a", random_unit_vector(dim, seed=0))
        engine.store("b", random_unit_vector(dim, seed=1))
        assert len(engine._session_indices) == 2
        engine.flush_session()
        assert len(engine._session_indices) == 0

    def test_reset_temporal_clears_co_occurrence(self):
        """reset_temporal clears co-occurrence links."""
        dim = 8
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)
        engine.store("a", random_unit_vector(dim, seed=0))
        engine.store("b", random_unit_vector(dim, seed=1))
        engine.flush_session()
        assert len(engine._co_occurrence) > 0
        engine.reset_temporal()
        assert len(engine._co_occurrence) == 0

    def test_twohop_empty_engine_returns_empty(self):
        """query_twohop on empty engine returns empty list."""
        dim = 8
        engine = CoupledEngine(dim=dim)
        query = random_unit_vector(dim, seed=0)
        results = engine.query_twohop(embedding=query, top_k=5)
        assert results == []

    def test_twohop_dimension_mismatch_raises(self):
        """query_twohop with wrong dimension raises ValueError."""
        dim = 8
        engine = CoupledEngine(dim=dim)
        engine.store("a", random_unit_vector(dim, seed=0))
        wrong_dim_query = random_unit_vector(dim + 2, seed=1)
        with pytest.raises(ValueError, match="Query dimension"):
            engine.query_twohop(embedding=wrong_dim_query, top_k=5)

    def test_twohop_recency_weighting(self):
        """query_twohop respects recency_alpha weighting."""
        dim = 16
        engine = CoupledEngine(dim=dim, recency_alpha=1.0)

        # Store two similar vectors with different recencies
        base = random_unit_vector(dim, seed=0)
        a = base + 0.01 * random_unit_vector(dim, seed=10)
        a /= np.linalg.norm(a)
        b = base + 0.01 * random_unit_vector(dim, seed=11)
        b /= np.linalg.norm(b)

        engine.store("old", a, recency=1.0)
        engine.store("new", b, recency=100.0)

        # Query with base -- both a and b are very similar
        results = engine.query_twohop(embedding=base, top_k=2)
        # With high recency_alpha, the newer one should rank first
        assert results[0]["index"] == 1  # "new" has higher recency

    def test_contradiction_replacement_tracks_session_indices(self):
        """Contradiction replacement should track the replacement index."""
        dim = 16
        engine = CoupledEngine(
            dim=dim,
            hebbian_epsilon=0.05,
            contradiction_aware=True,
            contradiction_threshold=0.8,
        )
        a = random_unit_vector(dim, seed=0)
        engine.store("original", a)

        # Store a very similar embedding that triggers contradiction replacement
        a_updated = a + 0.05 * random_unit_vector(dim, seed=99)
        a_updated /= np.linalg.norm(a_updated)
        engine.store("updated", a_updated)  # replaces index 0

        # Session indices should have [0, 0] -- both stores mapped to index 0
        assert 0 in engine._session_indices

    def test_twohop_co_occurrence_cross_session_isolation(self):
        """Facts in different sessions should NOT be co-occurrence linked."""
        dim = 16
        engine = CoupledEngine(dim=dim, hebbian_epsilon=0.05)

        a = random_unit_vector(dim, seed=0)
        b = random_unit_vector(dim, seed=1)

        # Session 1: only a
        engine.store("a", a)
        engine.flush_session()

        # Session 2: only b
        engine.store("b", b)
        engine.flush_session()

        # a and b should NOT be linked
        assert 0 not in engine._co_occurrence or 1 not in engine._co_occurrence.get(0, set())
        assert 1 not in engine._co_occurrence or 0 not in engine._co_occurrence.get(1, set())

    def test_flush_single_fact_clears_session_indices(self):
        """Flushing a single-fact session should clear _session_indices."""
        dim = 8
        engine = CoupledEngine(dim=dim)
        engine.store("solo", random_unit_vector(dim, seed=0))
        assert len(engine._session_indices) == 1
        engine.flush_session()
        assert len(engine._session_indices) == 0
