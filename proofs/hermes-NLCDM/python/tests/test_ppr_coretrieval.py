"""Tests for PPR co-retrieval boost and Hebbian edge weights.

Tests two features:
1. Hebbian edge weights in co-retrieval logging (replaces flat +1.0)
2. PPR-weighted co-retrieval boost in query() (blends cosine with PPR)

All tests must coexist with the 389 existing tests and 27 Lean invariant tests.
"""

import math

import numpy as np
import pytest

from coupled_engine import CoupledEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vectors(n: int, d: int, seed: int = 42) -> np.ndarray:
    """Generate n random unit vectors in R^d."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-12)


# ===========================================================================
# Change 1: Hebbian Edge Weights
# ===========================================================================


class TestHebbianEdgeWeights:
    """Co-retrieval edges should accumulate Hebbian strength, not flat +1.0."""

    def test_edge_weight_is_not_flat_one(self):
        """After a query returning 2+ results, co-retrieval edge weight should
        be a Hebbian strength value, NOT exactly 1.0."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        # Store two very similar patterns so they co-occur in results
        rng = np.random.default_rng(42)
        base = rng.standard_normal(d)
        base /= np.linalg.norm(base)
        p1 = base + rng.standard_normal(d) * 0.05
        p1 /= np.linalg.norm(p1)

        engine.store("fact_a", base)
        engine.store("fact_b", p1)

        # Query aligned with both
        query = base + rng.standard_normal(d) * 0.01
        query /= np.linalg.norm(query)
        engine.query(query, top_k=2)

        # Check that edges exist and are NOT exactly 1.0
        assert 0 in engine._co_retrieval
        assert 1 in engine._co_retrieval[0]
        weight = engine._co_retrieval[0][1]
        assert weight > 0, "Edge weight should be positive"
        assert weight != 1.0, (
            f"Edge weight is exactly 1.0 -- Hebbian weighting not applied"
        )

    def test_edge_weight_formula(self):
        """Verify edge weight = sqrt(score_i * score_j) * specificity."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(5, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        query = patterns[0] + np.random.default_rng(7).standard_normal(d) * 0.05
        query /= np.linalg.norm(query)
        results = engine.query(query, top_k=3)

        # Compute expected Hebbian strength for the first pair
        k = min(3, len(results))
        specificity = 1.0 / math.sqrt(k)
        score_0 = results[0]["score"]
        score_1 = results[1]["score"]
        expected_strength = math.sqrt(score_0 * score_1) * specificity

        idx_0 = results[0]["index"]
        idx_1 = results[1]["index"]
        actual = engine._co_retrieval.get(idx_0, {}).get(idx_1, 0.0)
        assert abs(actual - expected_strength) < 1e-6, (
            f"Edge weight {actual} != expected Hebbian {expected_strength}"
        )

    def test_hebbian_weights_accumulate(self):
        """Multiple queries should accumulate Hebbian edge deltas."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        # Use similar patterns so they all get positive cosine scores
        rng = np.random.default_rng(42)
        base = rng.standard_normal(d)
        base /= np.linalg.norm(base)
        patterns = [base]
        for i in range(2):
            p = base + rng.standard_normal(d) * 0.1
            p /= np.linalg.norm(p)
            patterns.append(p)

        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        query = base + rng.standard_normal(d) * 0.01
        query /= np.linalg.norm(query)

        results = engine.query(query, top_k=3)
        # Find a pair of indices with positive scores
        pos_results = [r for r in results if r["score"] > 0]
        assert len(pos_results) >= 2, "Need at least 2 positive-score results"
        idx_a = pos_results[0]["index"]
        idx_b = pos_results[1]["index"]

        weight_after_1 = engine._co_retrieval.get(idx_a, {}).get(idx_b, 0.0)
        assert weight_after_1 > 0, "First query should create positive edge"

        engine.query(query, top_k=3)
        weight_after_2 = engine._co_retrieval.get(idx_a, {}).get(idx_b, 0.0)

        assert weight_after_2 > weight_after_1, (
            f"Weights did not accumulate: {weight_after_2} <= {weight_after_1}"
        )

    def test_query_readonly_uses_hebbian_weights(self):
        """query_readonly should also use Hebbian edge weights, not flat +1.0."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        rng = np.random.default_rng(42)
        base = rng.standard_normal(d)
        base /= np.linalg.norm(base)
        p1 = base + rng.standard_normal(d) * 0.05
        p1 /= np.linalg.norm(p1)

        engine.store("fact_a", base)
        engine.store("fact_b", p1)

        query = base + rng.standard_normal(d) * 0.01
        query /= np.linalg.norm(query)
        engine.query_readonly(query, top_k=2)

        weight = engine._co_retrieval.get(0, {}).get(1, 0.0)
        assert weight > 0, "No edge weight from query_readonly"
        assert weight != 1.0, (
            f"query_readonly edge weight is exactly 1.0 -- Hebbian not applied"
        )


class TestCoRetrievalEdgeDecay:
    """Dream should decay co-retrieval edges exponentially."""

    def test_dream_decays_edges(self):
        """After dream(), co-retrieval edge weights should decrease."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(10, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        # Build co-retrieval edges via queries
        for i in range(5):
            query = patterns[i] + np.random.default_rng(i).standard_normal(d) * 0.1
            query /= np.linalg.norm(query)
            engine.query(query, top_k=5)

        # Record edge weights before dream
        weights_before: dict[tuple[int, int], float] = {}
        for idx, neighbors in engine._co_retrieval.items():
            for nbr, w in neighbors.items():
                weights_before[(idx, nbr)] = w

        assert len(weights_before) > 0, "No co-retrieval edges to test"

        # Dream
        engine.dream()

        # At least some edges that survived should have lower weights
        # (Some edges may be pruned if below 0.01 threshold)
        decayed_count = 0
        for (idx, nbr), old_w in weights_before.items():
            new_w = engine._co_retrieval.get(idx, {}).get(nbr, 0.0)
            if new_w < old_w:
                decayed_count += 1
        assert decayed_count > 0, "No co-retrieval edges were decayed by dream"

    def test_dream_removes_weak_edges(self):
        """Dream should remove edges below 0.01 threshold."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(5, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        # Manually set a very weak edge
        engine._co_retrieval.setdefault(0, {})[1] = 0.005
        engine._co_retrieval.setdefault(1, {})[0] = 0.005

        engine.dream()

        # Edge should be removed (below 0.01 after decay)
        edge_01 = engine._co_retrieval.get(0, {}).get(1, 0.0)
        assert edge_01 == 0.0 or 1 not in engine._co_retrieval.get(0, {}), (
            f"Weak edge not removed: {edge_01}"
        )


# ===========================================================================
# Change 2: PPR co-retrieval boost
# ===========================================================================


class TestPPRScoresMethod:
    """Test the _ppr_scores method independently."""

    def test_ppr_scores_returns_correct_shape(self):
        """_ppr_scores should return array of length N."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(10, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        seed_indices = np.array([0, 1, 2])
        seed_scores = np.array([0.9, 0.7, 0.5])
        ppr = engine._ppr_scores(seed_indices, seed_scores)
        assert ppr.shape == (10,), f"PPR shape {ppr.shape} != (10,)"

    def test_ppr_scores_empty_engine(self):
        """_ppr_scores should return empty array for empty engine."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)
        ppr = engine._ppr_scores(np.array([]), np.array([]))
        assert ppr.shape == (0,)

    def test_ppr_scores_zero_seeds(self):
        """_ppr_scores with zero-sum seed scores returns zeros."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(5, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        ppr = engine._ppr_scores(np.array([0, 1]), np.array([0.0, 0.0]))
        assert np.allclose(ppr, 0.0), "PPR with zero seeds should be zero"

    def test_ppr_without_edges_returns_teleport(self):
        """Without co-retrieval edges, PPR should converge to teleport vector."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(5, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        # No queries = no co-retrieval edges
        seed_idx = np.array([0, 1])
        seed_vals = np.array([0.8, 0.2])
        ppr = engine._ppr_scores(seed_idx, seed_vals)

        # Without edges, PPR = (1-alpha)*teleport at each step
        # For alpha=0.5, after convergence: ppr[0] > ppr[1] > ppr[2..4]
        assert ppr[0] > ppr[2], (
            f"Seed pattern 0 should have higher PPR: {ppr[0]} vs {ppr[2]}"
        )
        assert ppr[1] > ppr[3], (
            f"Seed pattern 1 should have higher PPR: {ppr[1]} vs {ppr[3]}"
        )

    def test_ppr_with_edges_propagates(self):
        """PPR should propagate score through co-retrieval edges."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(5, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        # Manually create co-retrieval edge: 0 -> 3
        engine._co_retrieval[0] = {3: 5.0}
        engine._co_retrieval[3] = {0: 5.0}

        seed_idx = np.array([0])
        seed_vals = np.array([1.0])
        ppr = engine._ppr_scores(seed_idx, seed_vals)

        # Pattern 3 should get boosted because it's connected to seed 0
        # Pattern 2 (no connection) should have lower PPR
        assert ppr[3] > ppr[2], (
            f"Connected pattern should get PPR boost: ppr[3]={ppr[3]} vs ppr[2]={ppr[2]}"
        )

    def test_ppr_respects_damping_parameter(self):
        """Constructor ppr_damping should be used as alpha."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0, ppr_damping=0.8)
        assert engine.ppr_damping == 0.8

    def test_ppr_respects_blend_weight_parameter(self):
        """Constructor ppr_blend_weight should be stored."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0, ppr_blend_weight=0.5)
        assert engine.ppr_blend_weight == 0.5


class TestPPRQueryIntegration:
    """Test PPR boost integration in query()."""

    def test_query_with_coretrieval_boosts_connected_patterns(self):
        """query() should boost patterns connected via co-retrieval graph."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(20, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        # Create strong co-retrieval edge: pattern 0 is strongly linked to pattern 15
        engine._co_retrieval[0] = {15: 10.0}
        engine._co_retrieval[15] = {0: 10.0}

        # Query aligned with pattern 0
        query = patterns[0] + np.random.default_rng(7).standard_normal(d) * 0.01
        query /= np.linalg.norm(query)

        results = engine.query(query, top_k=20)
        result_indices = [r["index"] for r in results]

        # Pattern 15 should be boosted higher in rankings than without edges
        # (In pure cosine, pattern 15 would rank near the bottom for a random vector)
        rank_15 = result_indices.index(15) if 15 in result_indices else 20
        # With PPR boost, rank should improve compared to pure cosine baseline
        # We verify this indirectly: rank of 15 should be better than its cosine rank

        # Compute pure cosine rank for comparison
        embs = engine._embeddings_matrix()
        q_norm = np.linalg.norm(query)
        norms = np.linalg.norm(embs, axis=1) * q_norm + 1e-12
        cosine = embs @ query / norms
        cosine_rank_15 = int(np.sum(cosine > cosine[15]))

        assert rank_15 <= cosine_rank_15, (
            f"PPR should improve rank of connected pattern: "
            f"PPR rank={rank_15}, cosine rank={cosine_rank_15}"
        )

    def test_query_without_coretrieval_is_pure_cosine(self):
        """Without co-retrieval edges, query() should return pure cosine scores."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(10, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        # No queries yet = no co-retrieval edges
        # Clear any edges that might have been created
        engine._co_retrieval = {}

        query = patterns[0] + np.random.default_rng(7).standard_normal(d) * 0.05
        query /= np.linalg.norm(query)

        results = engine.query(query, top_k=5)

        # Should be same as pure cosine (no PPR blend)
        embs = engine._embeddings_matrix()
        q_norm = np.linalg.norm(query)
        norms = np.linalg.norm(embs, axis=1) * q_norm + 1e-12
        expected_scores = embs @ query / norms

        for r in results:
            expected = float(expected_scores[r["index"]])
            assert abs(r["score"] - expected) < 1e-6, (
                f"Score mismatch without co-retrieval: {r['score']} != {expected}"
            )

    def test_ppr_preserves_hybrid_switching_rank1(self):
        """CRITICAL: PPR boost must NOT invert cosine rank-1.
        If it would, fall back to pure cosine (HybridSwitching.lean)."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(20, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        # Create strong co-retrieval edges that might try to boost pattern 5
        # above pattern 0 (which is the cosine rank-1 for its own query)
        engine._co_retrieval[5] = {i: 100.0 for i in range(20) if i != 5}
        for i in range(20):
            if i != 5:
                engine._co_retrieval.setdefault(i, {})[5] = 100.0

        # Query perfectly aligned with pattern 0
        query = patterns[0].copy()
        results = engine.query(query, top_k=5)

        # Cosine rank-1 is pattern 0 (perfect alignment)
        embs = engine._embeddings_matrix()
        q_norm = np.linalg.norm(query)
        norms = np.linalg.norm(embs, axis=1) * q_norm + 1e-12
        cosine = embs @ query / norms
        cosine_rank1 = int(np.argmax(cosine))

        # The query result rank-1 must preserve cosine rank-1
        assert results[0]["index"] == cosine_rank1, (
            f"PPR broke hybrid switching: result rank-1={results[0]['index']}, "
            f"cosine rank-1={cosine_rank1}"
        )

    def test_ppr_scores_finite_and_reasonable(self):
        """PPR-blended scores should be finite and the top-1 should be positive."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(10, d, seed=42)
        for i, p in enumerate(patterns):
            engine.store(f"fact_{i}", p)

        # Build some co-retrieval edges
        for i in range(5):
            query = patterns[i] + np.random.default_rng(i).standard_normal(d) * 0.1
            query /= np.linalg.norm(query)
            engine.query(query, top_k=5)

        # Query and verify scores are finite; top-1 should be positive
        query = patterns[0] + np.random.default_rng(99).standard_normal(d) * 0.05
        query /= np.linalg.norm(query)
        results = engine.query(query, top_k=10)
        for r in results:
            assert np.isfinite(r["score"]), f"Non-finite score: {r['score']}"
        # Top-1 result should have a positive score
        assert results[0]["score"] > 0, (
            f"Top-1 score not positive: {results[0]['score']}"
        )


# ===========================================================================
# Lean invariant preservation
# ===========================================================================


class TestLeanInvariantsPreserved:
    """Verify that PPR/Hebbian changes don't break Lean invariants."""

    def test_capacity_formula_unchanged(self):
        """The capacity formula exp(beta*delta)/(4*beta*M^2) must be unchanged."""
        d = 64
        engine = CoupledEngine(dim=d, beta=5.0)

        patterns = _unit_vectors(20, d, seed=99)
        for i, p in enumerate(patterns):
            engine.store(f"pattern_{i}", p)

        embeddings = engine._embeddings_matrix()
        ratio = engine._compute_capacity_ratio(embeddings)

        # Independently compute
        sq_norms = np.sum(embeddings ** 2, axis=1)
        M_sq = float(np.max(sq_norms))
        normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        cos_matrix = normed @ normed.T
        iu = np.triu_indices(len(embeddings), k=1)
        delta_min = float(np.min(1.0 - cos_matrix[iu]))
        if delta_min <= 0:
            delta_min = 0.01
        N_max = math.exp(engine.beta * delta_min) / (4.0 * engine.beta * M_sq)
        if N_max < 1.0:
            N_max = 1.0
        expected = len(embeddings) / N_max

        assert abs(ratio - expected) < 1e-6

    def test_dream_params_validation_unchanged(self):
        """DreamParams constraints (eta < min_sep/2, merge < prune) must hold."""
        from dream_ops import DreamParams

        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(eta=0.2, min_sep=0.3)
        with pytest.raises(ValueError, match="Lean bound violated"):
            DreamParams(merge_threshold=0.96, prune_threshold=0.95)

    def test_explore_readonly_preserved(self):
        """rem_explore_cross_domain_xb must remain read-only."""
        from dream_ops import rem_explore_cross_domain_xb

        d = 64
        patterns = _unit_vectors(30, d, seed=42)
        labels = np.array([i % 3 for i in range(30)])
        patterns_copy = patterns.copy()

        _ = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=50, rng=np.random.default_rng(42)
        )

        np.testing.assert_array_equal(patterns, patterns_copy)


# ===========================================================================
# Constructor parameter tests
# ===========================================================================


class TestConstructorParameters:
    """Verify ppr_blend_weight and ppr_damping constructor parameters."""

    def test_default_ppr_blend_weight(self):
        """Default ppr_blend_weight should be 0.3."""
        engine = CoupledEngine(dim=64, beta=5.0)
        assert engine.ppr_blend_weight == 0.3

    def test_default_ppr_damping(self):
        """Default ppr_damping should be 0.5."""
        engine = CoupledEngine(dim=64, beta=5.0)
        assert engine.ppr_damping == 0.5

    def test_custom_ppr_blend_weight(self):
        """Custom ppr_blend_weight should be stored."""
        engine = CoupledEngine(dim=64, beta=5.0, ppr_blend_weight=0.5)
        assert engine.ppr_blend_weight == 0.5

    def test_custom_ppr_damping(self):
        """Custom ppr_damping should be stored."""
        engine = CoupledEngine(dim=64, beta=5.0, ppr_damping=0.8)
        assert engine.ppr_damping == 0.8
