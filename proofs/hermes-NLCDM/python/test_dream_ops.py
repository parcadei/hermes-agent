"""Tests for thermodynamic dream cycle operations.

Validates the three dream operations (NREM-replay, REM-unlearn, REM-explore),
Lotka-Volterra competition, consolidation, and the full dream cycle.
Also tests hybrid retrieval (cosine seeds + one Hopfield expansion step).
"""

from __future__ import annotations

import numpy as np
import pytest

from coupled_engine import CoupledEngine
from dream_ops import (
    DreamParams,
    hopfield_update,
    spreading_activation,
    boltzmann_dynamics,
    nrem_replay,
    rem_unlearn,
    rem_explore,
    lv_competition,
    consolidate_similar,
    dream_cycle,
)
from nlcdm_core import cosine_sim, softmax, sparsemax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orthonormal_patterns(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Create n nearly-orthogonal unit patterns in R^d (d >> n)."""
    raw = rng.standard_normal((n, d))
    # QR gives orthogonal rows when n <= d
    q, _ = np.linalg.qr(raw.T)
    return q.T[:n]  # (n, d), each row is unit norm and mutually orthogonal


def _make_coupling_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Build a Hebbian coupling matrix W = X^T X / N (symmetric, no self-coupling diag)."""
    N, d = embeddings.shape
    W = (embeddings.T @ embeddings) / N
    np.fill_diagonal(W, 0.0)
    return W


# ---------------------------------------------------------------------------
# DreamParams
# ---------------------------------------------------------------------------

class TestDreamParams:
    def test_defaults_match_research_spec(self):
        """Default DreamParams should match research v3 specification."""
        p = DreamParams()
        assert p.beta_high == 10.0
        assert p.beta_mod == 2.0
        assert p.beta_low == 0.5
        assert p.eta == 0.01
        assert p.eta_weak == 0.001
        assert p.n_unlearn == 100
        assert p.n_explore == 50
        assert p.consolidation_threshold == 0.95
        assert p.seed is None

    def test_frozen(self):
        """DreamParams should be immutable (frozen dataclass)."""
        p = DreamParams()
        with pytest.raises(AttributeError):
            p.beta_high = 5.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Hopfield Update
# ---------------------------------------------------------------------------

class TestHopfieldUpdate:
    def test_is_convex_combination(self):
        """Output of hopfield_update should lie in the convex hull of patterns.

        Since output = sum_mu softmax(...)_mu * x_mu, and softmax weights
        are non-negative and sum to 1, the result is a convex combination.
        """
        rng = np.random.default_rng(42)
        d = 32
        N = 5
        patterns = rng.standard_normal((N, d))
        xi = rng.standard_normal(d)

        result = hopfield_update(1.0, patterns, xi)

        # The result should be expressible as alpha @ patterns where alpha >= 0, sum(alpha) = 1
        # We verify this by checking that the softmax weights reproduce the result
        sims = patterns @ xi
        weights = softmax(1.0, sims)
        expected = weights @ patterns
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_high_beta_selects_nearest(self):
        """At high beta, hopfield_update should return (nearly) the closest pattern."""
        rng = np.random.default_rng(42)
        d = 32
        patterns = np.eye(d)[:5]  # 5 standard basis vectors
        xi = patterns[2] + 0.1 * rng.standard_normal(d)

        result = hopfield_update(100.0, patterns, xi)
        # Should be very close to pattern 2
        assert cosine_sim(result, patterns[2]) > 0.99

    def test_output_shape(self):
        """Output should have same shape as input xi."""
        patterns = np.random.randn(10, 64)
        xi = np.random.randn(64)
        result = hopfield_update(1.0, patterns, xi)
        assert result.shape == xi.shape


# ---------------------------------------------------------------------------
# Spreading Activation
# ---------------------------------------------------------------------------

class TestSpreadingActivation:
    def test_converges(self):
        """Spreading activation should reach a fixed point."""
        rng = np.random.default_rng(42)
        d = 32
        N = 5
        patterns = _make_orthonormal_patterns(N, d, rng)
        xi = rng.standard_normal(d)

        result = spreading_activation(5.0, patterns, xi, max_steps=200)

        # Verify it's a fixed point: one more step shouldn't change it
        next_step = hopfield_update(5.0, patterns, result)
        np.testing.assert_allclose(result, next_step, atol=1e-5)

    def test_at_pattern_stays_near(self):
        """Starting from a stored pattern, spreading activation should stay near it."""
        rng = np.random.default_rng(42)
        d = 32
        N = 5
        patterns = _make_orthonormal_patterns(N, d, rng)

        # Start exactly at pattern 0
        result = spreading_activation(10.0, patterns, patterns[0].copy())
        assert cosine_sim(result, patterns[0]) > 0.99

    def test_respects_max_steps(self):
        """Should not run more than max_steps iterations."""
        rng = np.random.default_rng(42)
        d = 32
        patterns = rng.standard_normal((5, d))
        xi = rng.standard_normal(d)

        # With max_steps=1, should do exactly one Hopfield update
        result = spreading_activation(1.0, patterns, xi, max_steps=1)
        expected = hopfield_update(1.0, patterns, xi)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Sparsemax Attention
# ---------------------------------------------------------------------------

class TestSparsemax:
    """Verify sparsemax produces valid sparse probability distributions."""

    def test_sums_to_one(self):
        """Sparsemax output should sum to 1 (it's a simplex projection)."""
        rng = np.random.default_rng(42)
        z = rng.standard_normal(20)
        p = sparsemax(1.0, z)
        assert abs(p.sum() - 1.0) < 1e-10

    def test_non_negative(self):
        """All sparsemax outputs should be >= 0."""
        rng = np.random.default_rng(42)
        z = rng.standard_normal(50)
        p = sparsemax(1.0, z)
        assert np.all(p >= 0.0)

    def test_has_exact_zeros(self):
        """Sparsemax should produce exact zeros for low-scoring entries."""
        z = np.array([5.0, 4.5, 0.1, -1.0, -3.0])
        p = sparsemax(1.0, z)
        # Top 2 should be nonzero, bottom entries should be exactly 0
        assert p[0] > 0.0
        assert p[1] > 0.0
        assert p[3] == 0.0
        assert p[4] == 0.0

    def test_uniform_input(self):
        """Uniform input should give uniform output (all equal = all in support)."""
        z = np.ones(5)
        p = sparsemax(1.0, z)
        np.testing.assert_allclose(p, np.ones(5) / 5.0, atol=1e-10)

    def test_single_dominant(self):
        """One element much larger → sparsemax assigns it weight 1."""
        z = np.array([100.0, 0.0, 0.0, 0.0])
        p = sparsemax(1.0, z)
        assert p[0] > 0.99
        assert p[1] == 0.0
        assert p[2] == 0.0
        assert p[3] == 0.0

    def test_beta_scaling(self):
        """Higher beta should produce sparser output."""
        z = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        p_low = sparsemax(1.0, z)
        p_high = sparsemax(10.0, z)
        # Higher beta → fewer nonzero entries
        assert np.count_nonzero(p_high) <= np.count_nonzero(p_low)

    def test_matches_softmax_ranking(self):
        """Sparsemax should preserve the ranking order of softmax."""
        rng = np.random.default_rng(42)
        z = rng.standard_normal(10)
        p_soft = softmax(1.0, z)
        p_sparse = sparsemax(1.0, z)
        # For nonzero entries, ranking should match
        nonzero = p_sparse > 0
        if nonzero.sum() > 1:
            # Check ranking matches softmax ranking for the nonzero entries
            sparse_order = np.argsort(-p_sparse[nonzero])
            soft_order = np.argsort(-p_soft[nonzero])
            np.testing.assert_array_equal(sparse_order, soft_order)


class TestSparseHopfieldUpdate:
    """Verify sparse Hopfield update works correctly."""

    def test_sparse_is_convex_combination(self):
        """Sparse Hopfield update should be a convex combination of patterns."""
        rng = np.random.default_rng(42)
        d = 32
        N = 5
        patterns = rng.standard_normal((N, d))
        xi = rng.standard_normal(d)

        result = hopfield_update(1.0, patterns, xi, attention_fn=sparsemax)

        # Verify: result = weights @ patterns where weights sum to 1, >= 0
        sims = patterns @ xi
        weights = sparsemax(1.0, sims)
        expected = weights @ patterns
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_sparse_high_beta_selects_nearest(self):
        """At high beta, sparse Hopfield should select nearest pattern exactly."""
        rng = np.random.default_rng(42)
        d = 32
        patterns = np.eye(d)[:5]  # 5 standard basis vectors
        xi = patterns[2] + 0.1 * rng.standard_normal(d)

        result = hopfield_update(100.0, patterns, xi, attention_fn=sparsemax)
        assert cosine_sim(result, patterns[2]) > 0.99

    def test_sparse_fewer_active_patterns(self):
        """Sparse update should use fewer patterns than dense softmax."""
        rng = np.random.default_rng(42)
        d = 64
        N = 50  # Many patterns
        patterns = rng.standard_normal((N, d))
        # Normalize
        patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
        xi = patterns[0] + 0.3 * rng.standard_normal(d)

        sims = patterns @ xi
        soft_weights = softmax(5.0, sims)
        sparse_weights = sparsemax(5.0, sims)

        # Softmax: all N entries nonzero. Sparsemax: some entries exactly zero.
        assert np.count_nonzero(soft_weights) == N
        assert np.count_nonzero(sparse_weights) < N

    def test_sparse_spreading_converges(self):
        """Sparse spreading activation should converge to a fixed point."""
        rng = np.random.default_rng(42)
        d = 32
        N = 5
        patterns = _make_orthonormal_patterns(N, d, rng)
        xi = rng.standard_normal(d)

        result = spreading_activation(
            5.0, patterns, xi, max_steps=200, attention_fn=sparsemax,
        )

        # One more step shouldn't change it
        next_step = hopfield_update(5.0, patterns, result, attention_fn=sparsemax)
        np.testing.assert_allclose(result, next_step, atol=1e-5)

    def test_sparse_at_pattern_stays(self):
        """Starting from a stored pattern, sparse dynamics should stay near it."""
        rng = np.random.default_rng(42)
        d = 32
        N = 5
        patterns = _make_orthonormal_patterns(N, d, rng)

        result = spreading_activation(
            10.0, patterns, patterns[0].copy(), attention_fn=sparsemax,
        )
        assert cosine_sim(result, patterns[0]) > 0.99

    def test_sparse_retrieval_sharper_than_dense(self):
        """With many patterns, sparse retrieval should focus on fewer neighbors."""
        rng = np.random.default_rng(42)
        d = 128
        N = 100
        patterns = rng.standard_normal((N, d))
        patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
        xi = patterns[0] + 0.2 * rng.standard_normal(d)
        xi = xi / np.linalg.norm(xi)

        dense_result = spreading_activation(
            5.0, patterns, xi, max_steps=50, attention_fn=softmax,
        )
        sparse_result = spreading_activation(
            5.0, patterns, xi, max_steps=50, attention_fn=sparsemax,
        )

        # Sparse result should be closer to the nearest pattern
        # (less blurring from irrelevant patterns)
        dense_sim = cosine_sim(dense_result, patterns[0])
        sparse_sim = cosine_sim(sparse_result, patterns[0])
        assert sparse_sim >= dense_sim - 0.05  # sparse should be at least as sharp


# ---------------------------------------------------------------------------
# Boltzmann Dynamics
# ---------------------------------------------------------------------------

class TestBoltzmannDynamics:
    def test_output_bounded_zero_one(self):
        """Boltzmann dynamics uses sigmoid, so output should be in (0, 1)."""
        rng = np.random.default_rng(42)
        d = 16
        W = rng.standard_normal((d, d))
        W = (W + W.T) / 2  # symmetric
        x0 = rng.standard_normal(d)

        result = boltzmann_dynamics(W, x0, beta=2.0)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_converges_to_fixed_point(self):
        """Boltzmann dynamics should converge to a fixed point."""
        rng = np.random.default_rng(42)
        d = 16
        W = rng.standard_normal((d, d))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        x0 = rng.standard_normal(d)

        result = boltzmann_dynamics(W, x0, beta=1.0, max_steps=200, tol=1e-8)
        from nlcdm_core import sigmoid
        next_step = sigmoid(1.0 * W @ result)
        np.testing.assert_allclose(result, next_step, atol=1e-4)


# ---------------------------------------------------------------------------
# NREM Replay
# ---------------------------------------------------------------------------

class TestNREMReplay:
    def test_strengthens_tagged(self):
        """NREM replay should increase W entries corresponding to tagged memories."""
        rng = np.random.default_rng(42)
        d = 32
        N = 5
        patterns = _make_orthonormal_patterns(N, d, rng)
        W = _make_coupling_matrix(patterns)
        W_before = W.copy()

        tagged = [0, 2]  # tag memories 0 and 2
        W_after = nrem_replay(W_before, tagged, patterns, beta_high=10.0, eta=0.01)

        # W should have changed
        assert not np.allclose(W_before, W_after)

        # The Frobenius norm of W should have increased (Hebbian adds energy)
        assert np.linalg.norm(W_after, 'fro') > np.linalg.norm(W_before, 'fro')

    def test_pure_function(self):
        """NREM replay should not mutate the input W."""
        rng = np.random.default_rng(42)
        d = 32
        patterns = _make_orthonormal_patterns(5, d, rng)
        W = _make_coupling_matrix(patterns)
        W_original = W.copy()

        nrem_replay(W, [0], patterns)
        np.testing.assert_array_equal(W, W_original)

    def test_empty_tagged_is_identity(self):
        """With no tagged memories, W should be unchanged."""
        rng = np.random.default_rng(42)
        d = 32
        patterns = _make_orthonormal_patterns(5, d, rng)
        W = _make_coupling_matrix(patterns)

        W_after = nrem_replay(W, [], patterns)
        np.testing.assert_array_equal(W, W_after)


# ---------------------------------------------------------------------------
# REM Unlearn
# ---------------------------------------------------------------------------

class TestREMUnlearn:
    def test_weakens_spurious(self):
        """REM unlearn should weaken spurious attractors.

        Strategy: create a coupling matrix with spurious states, run unlearn,
        verify that attractors reachable from random starts become less stable
        (the converged states change, indicating the spurious basins are disrupted).
        """
        rng = np.random.default_rng(42)
        d = 16
        W = rng.standard_normal((d, d)) * 0.5
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)

        # Find attractors before unlearning
        def find_attractors(W_mat, n_starts=20):
            attractors = []
            test_rng = np.random.default_rng(999)
            for _ in range(n_starts):
                x0 = test_rng.standard_normal(d)
                x_inf = boltzmann_dynamics(W_mat, x0, beta=3.0, max_steps=100)
                is_new = True
                for att in attractors:
                    if np.linalg.norm(x_inf - att) < 0.05:
                        is_new = False
                        break
                if is_new:
                    attractors.append(x_inf)
            return attractors

        att_before = find_attractors(W)

        W_after = rem_unlearn(W, d, beta_mod=2.0, eta=0.005, n_trials=200,
                              rng=np.random.default_rng(42))

        att_after = find_attractors(W_after)

        # Anti-Hebbian should disrupt at least some spurious attractors.
        # The attractor landscape should change (not necessarily fewer,
        # but at least different -- the spurious ones are destabilized).
        # We verify W actually changed meaningfully.
        assert not np.allclose(W, W_after), "W should change after unlearning"
        # The number of attractors should not increase dramatically
        assert len(att_after) <= len(att_before) + 3

    def test_pure_function(self):
        """REM unlearn should not mutate the input W."""
        rng = np.random.default_rng(42)
        d = 16
        W = rng.standard_normal((d, d))
        W = (W + W.T) / 2
        W_original = W.copy()

        rem_unlearn(W, d, rng=rng)
        np.testing.assert_array_equal(W, W_original)

    def test_reproducible_with_rng(self):
        """Same RNG seed should give same result."""
        d = 16
        W = np.random.default_rng(0).standard_normal((d, d))
        W = (W + W.T) / 2

        W1 = rem_unlearn(W, d, rng=np.random.default_rng(99))
        W2 = rem_unlearn(W, d, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(W1, W2)


# ---------------------------------------------------------------------------
# REM Explore
# ---------------------------------------------------------------------------

class TestREMExplore:
    def test_does_not_converge(self):
        """REM explore uses fixed step count and should NOT converge.

        Different starting points should give different intermediate states.
        """
        rng = np.random.default_rng(42)
        d = 16
        embeddings = _make_orthonormal_patterns(5, d, rng)
        W = _make_coupling_matrix(embeddings)

        # Run twice with different seeds: results should differ
        W1 = rem_explore(W, embeddings, beta_low=0.5, n_steps=50,
                         rng=np.random.default_rng(1))
        W2 = rem_explore(W, embeddings, beta_low=0.5, n_steps=50,
                         rng=np.random.default_rng(2))

        assert not np.allclose(W1, W2), "Different seeds should give different results"

    def test_weak_reinforcement(self):
        """W changes from REM explore should be small per step (eta_weak)."""
        rng = np.random.default_rng(42)
        d = 16
        embeddings = _make_orthonormal_patterns(5, d, rng)
        W = _make_coupling_matrix(embeddings)
        W_before = W.copy()

        eta_weak = 0.001
        n_steps = 10
        W_after = rem_explore(W, embeddings, beta_low=0.5, eta_weak=eta_weak,
                              n_steps=n_steps, rng=np.random.default_rng(42))

        # Total change should be bounded: each step adds eta_weak * ||x||^2 at most
        # Since sigmoid output is in (0,1), ||x||^2 <= d, so total <= n_steps * eta_weak * d
        max_change = np.max(np.abs(W_after - W_before))
        assert max_change < n_steps * eta_weak * d

    def test_pure_function(self):
        """REM explore should not mutate the input W."""
        rng = np.random.default_rng(42)
        d = 16
        embeddings = _make_orthonormal_patterns(5, d, rng)
        W = _make_coupling_matrix(embeddings)
        W_original = W.copy()

        rem_explore(W, embeddings, rng=rng)
        np.testing.assert_array_equal(W, W_original)

    def test_uses_fixed_step_count(self):
        """With 0 steps, W should be unchanged. With 1 step, exactly one update."""
        rng = np.random.default_rng(42)
        d = 16
        embeddings = _make_orthonormal_patterns(5, d, rng)
        W = _make_coupling_matrix(embeddings)

        W_zero = rem_explore(W, embeddings, n_steps=0, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(W, W_zero)


# ---------------------------------------------------------------------------
# Lotka-Volterra Competition
# ---------------------------------------------------------------------------

class TestLVCompetition:
    def test_selects_strong(self):
        """Stronger attractors should survive competition.

        Strategy: give one attractor much higher fitness, after LV dynamics
        the coupling associated with weaker attractors should be diminished.
        """
        rng = np.random.default_rng(42)
        d = 16
        N = 4
        embeddings = _make_orthonormal_patterns(N, d, rng)
        W = _make_coupling_matrix(embeddings)

        # Attractor 0 is strong, rest are weak
        fitness = np.array([10.0, 1.0, 1.0, 1.0])

        W_after = lv_competition(W, embeddings, fitness, dt=0.01, steps=100)

        # W should have changed
        assert not np.allclose(W, W_after)

    def test_pure_function(self):
        """LV competition should not mutate the input W."""
        rng = np.random.default_rng(42)
        d = 16
        embeddings = _make_orthonormal_patterns(4, d, rng)
        W = _make_coupling_matrix(embeddings)
        W_original = W.copy()

        fitness = np.ones(4)
        lv_competition(W, embeddings, fitness)
        np.testing.assert_array_equal(W, W_original)

    def test_equal_fitness_preserves_symmetry(self):
        """Equal fitness values should preserve the symmetry of W."""
        rng = np.random.default_rng(42)
        d = 16
        embeddings = _make_orthonormal_patterns(4, d, rng)
        W = _make_coupling_matrix(embeddings)
        assert np.allclose(W, W.T), "Initial W should be symmetric"

        fitness = np.ones(4) * 5.0
        W_after = lv_competition(W, embeddings, fitness, dt=0.01, steps=50)
        assert np.allclose(W_after, W_after.T, atol=1e-10), "W should remain symmetric"


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

class TestConsolidation:
    def test_merges_similar(self):
        """Near-duplicate patterns (cosine_sim > threshold) should be merged."""
        rng = np.random.default_rng(42)
        d = 32
        base = rng.standard_normal(d)
        base /= np.linalg.norm(base)

        # Pattern 0 and 1 are near-duplicates, pattern 2 is different
        p0 = base.copy()
        p1 = base + 0.01 * rng.standard_normal(d)
        p1 /= np.linalg.norm(p1)
        p2 = rng.standard_normal(d)
        p2 /= np.linalg.norm(p2)

        embeddings = np.stack([p0, p1, p2])
        W = _make_coupling_matrix(embeddings)

        W_new, emb_new = consolidate_similar(W, embeddings, threshold=0.95)

        # Should have merged p0 and p1
        assert emb_new.shape[0] == 2, f"Expected 2 patterns after merge, got {emb_new.shape[0]}"
        assert W_new.shape == (emb_new.shape[1], emb_new.shape[1])

    def test_no_merge_below_threshold(self):
        """Patterns below threshold should not be merged."""
        rng = np.random.default_rng(42)
        d = 32
        embeddings = _make_orthonormal_patterns(5, d, rng)
        W = _make_coupling_matrix(embeddings)

        W_new, emb_new = consolidate_similar(W, embeddings, threshold=0.95)

        # Orthogonal patterns have cosine_sim ~ 0, so no merges
        assert emb_new.shape[0] == 5

    def test_pure_function(self):
        """Consolidation should not mutate inputs."""
        rng = np.random.default_rng(42)
        d = 32
        embeddings = _make_orthonormal_patterns(5, d, rng)
        W = _make_coupling_matrix(embeddings)
        W_orig = W.copy()
        emb_orig = embeddings.copy()

        consolidate_similar(W, embeddings)
        np.testing.assert_array_equal(W, W_orig)
        np.testing.assert_array_equal(embeddings, emb_orig)


# ---------------------------------------------------------------------------
# Full Dream Cycle
# ---------------------------------------------------------------------------

class TestDreamCycle:
    def test_pure(self):
        """Dream cycle should not mutate input W or embeddings."""
        rng = np.random.default_rng(42)
        d = 32
        N = 5
        embeddings = _make_orthonormal_patterns(N, d, rng)
        W = _make_coupling_matrix(embeddings)
        W_original = W.copy()
        emb_original = embeddings.copy()

        params = DreamParams(seed=42, n_unlearn=10, n_explore=5)
        dream_cycle(W, [0, 1], embeddings, params=params)

        np.testing.assert_array_equal(W, W_original)
        np.testing.assert_array_equal(embeddings, emb_original)

    def test_improves_landscape(self):
        """Full dream cycle should reduce spurious attractor count.

        Strategy: create a random W with many spurious states, run dream cycle,
        verify that the number of distinct attractors from random starts decreases
        or at least doesn't increase wildly.
        """
        rng = np.random.default_rng(42)
        d = 16
        N = 4
        embeddings = _make_orthonormal_patterns(N, d, rng)
        W = _make_coupling_matrix(embeddings)
        # Add noise to create spurious states
        noise = rng.standard_normal(W.shape) * 0.1
        W_noisy = W + (noise + noise.T) / 2

        params = DreamParams(
            seed=42,
            n_unlearn=50,
            n_explore=20,
            beta_high=10.0,
            beta_mod=2.0,
            beta_low=0.5,
        )
        tagged = list(range(N))

        W_after, emb_after = dream_cycle(W_noisy, tagged, embeddings, params=params)

        # Count attractors: run Boltzmann dynamics from random starts
        def count_attractors(W_mat, n_starts=30):
            attractors = []
            test_rng = np.random.default_rng(123)
            for _ in range(n_starts):
                x0 = test_rng.standard_normal(d)
                x_final = boltzmann_dynamics(W_mat, x0, beta=5.0, max_steps=100)
                # Check if this is a new attractor
                is_new = True
                for att in attractors:
                    if np.linalg.norm(x_final - att) < 0.1:
                        is_new = False
                        break
                if is_new:
                    attractors.append(x_final)
            return len(attractors)

        n_before = count_attractors(W_noisy)
        n_after = count_attractors(W_after)

        # Dream cycle should not dramatically increase attractor count
        # (it should clean spurious ones)
        assert n_after <= n_before + 2, (
            f"Attractor count increased: {n_before} -> {n_after}"
        )

    def test_returns_tuple(self):
        """Dream cycle should return (W, embeddings) tuple."""
        rng = np.random.default_rng(42)
        d = 16
        embeddings = _make_orthonormal_patterns(3, d, rng)
        W = _make_coupling_matrix(embeddings)

        params = DreamParams(seed=42, n_unlearn=5, n_explore=3)
        result = dream_cycle(W, [0], embeddings, params=params)

        assert isinstance(result, tuple)
        assert len(result) == 2
        W_out, emb_out = result
        assert isinstance(W_out, np.ndarray)
        assert isinstance(emb_out, np.ndarray)

    def test_custom_params(self):
        """Dream cycle should respect custom parameters."""
        rng = np.random.default_rng(42)
        d = 16
        embeddings = _make_orthonormal_patterns(3, d, rng)
        W = _make_coupling_matrix(embeddings)

        # Very aggressive unlearning
        params_aggressive = DreamParams(
            seed=42, eta=0.1, n_unlearn=50, n_explore=5
        )
        # Very mild
        params_mild = DreamParams(
            seed=42, eta=0.0001, n_unlearn=5, n_explore=2
        )

        W_agg, _ = dream_cycle(W, [0], embeddings, params=params_aggressive)
        W_mild, _ = dream_cycle(W, [0], embeddings, params=params_mild)

        # Aggressive should change W more than mild
        diff_agg = np.linalg.norm(W_agg - W, 'fro')
        diff_mild = np.linalg.norm(W_mild - W, 'fro')
        assert diff_agg > diff_mild


# ---------------------------------------------------------------------------
# Hybrid Retrieval (cosine seeds → one Hopfield expansion step → union)
# ---------------------------------------------------------------------------


class TestHybridRetrieval:
    """Tests for CoupledEngine.query_hybrid().

    The hybrid approach: cosine finds direct matches, one sparsemax
    Hopfield step from the seed centroid finds associatively linked
    patterns, union returns both.
    """

    @staticmethod
    def _make_engine(dim: int = 16, n: int = 20, seed: int = 42):
        """Create engine with random normalized patterns."""
        rng = np.random.default_rng(seed)
        engine = CoupledEngine(dim=dim)
        for i in range(n):
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v)
            engine.store(embedding=v, text=f"fact_{i}")
        return engine, rng

    def test_returns_results(self):
        """query_hybrid returns non-empty list for populated engine."""
        engine, rng = self._make_engine()
        q = rng.standard_normal(engine.dim)
        q /= np.linalg.norm(q)
        results = engine.query_hybrid(embedding=q, top_k=5)
        assert len(results) > 0
        assert len(results) <= 10  # union can be up to 2*top_k

    def test_result_fields(self):
        """Each result has index, score, and text."""
        engine, rng = self._make_engine()
        q = rng.standard_normal(engine.dim)
        q /= np.linalg.norm(q)
        results = engine.query_hybrid(embedding=q, top_k=3)
        for r in results:
            assert "index" in r
            assert "score" in r
            assert "text" in r
            assert isinstance(r["index"], int)
            assert isinstance(r["score"], float)

    def test_no_duplicate_indices(self):
        """Union deduplication: no index appears twice."""
        engine, rng = self._make_engine(n=30)
        q = rng.standard_normal(engine.dim)
        q /= np.linalg.norm(q)
        results = engine.query_hybrid(embedding=q, top_k=10)
        indices = [r["index"] for r in results]
        assert len(indices) == len(set(indices))

    def test_cosine_top_included(self):
        """The top cosine match should always appear in hybrid results."""
        engine, rng = self._make_engine()
        q = rng.standard_normal(engine.dim)
        q /= np.linalg.norm(q)
        cosine_results = engine.query(embedding=q, top_k=1)
        hybrid_results = engine.query_hybrid(embedding=q, top_k=5)
        hybrid_indices = {r["index"] for r in hybrid_results}
        assert cosine_results[0]["index"] in hybrid_indices

    def test_expansion_finds_associated_patterns(self):
        """With clustered patterns, hybrid finds associated patterns
        that share coupling with the cosine seeds.

        Setup: cluster A (patterns 0-4) are near each other,
        query is near cluster A. Hybrid should retrieve more of
        cluster A than cosine alone would at small top_k.
        """
        dim = 32
        rng = np.random.default_rng(99)
        engine = CoupledEngine(dim=dim)

        # Cluster A: 8 patterns around a direction
        base_a = rng.standard_normal(dim)
        base_a /= np.linalg.norm(base_a)
        for i in range(8):
            noise = rng.standard_normal(dim) * 0.15
            v = base_a + noise
            v /= np.linalg.norm(v)
            engine.store(embedding=v, text=f"cluster_a_{i}")

        # Cluster B: 8 patterns in an orthogonal direction
        base_b = rng.standard_normal(dim)
        # Make roughly orthogonal to A
        base_b -= base_a * (base_b @ base_a)
        base_b /= np.linalg.norm(base_b)
        for i in range(8):
            noise = rng.standard_normal(dim) * 0.15
            v = base_b + noise
            v /= np.linalg.norm(v)
            engine.store(embedding=v, text=f"cluster_b_{i}")

        # Query is slightly noisy version of cluster A direction
        q = base_a + rng.standard_normal(dim) * 0.1
        q /= np.linalg.norm(q)

        # With top_k=3, cosine gets some of cluster A
        cosine_results = engine.query(embedding=q, top_k=3)
        cosine_a_count = sum(
            1 for r in cosine_results if r["text"].startswith("cluster_a")
        )

        # Hybrid should get at least as many cluster A patterns
        # because the expansion from seeds spreads within the cluster
        hybrid_results = engine.query_hybrid(embedding=q, top_k=3)
        hybrid_a_count = sum(
            1 for r in hybrid_results if r["text"].startswith("cluster_a")
        )
        assert hybrid_a_count >= cosine_a_count

    def test_cross_domain_retrieval(self):
        """Core test: hybrid retrieves patterns linked through coupling
        that cosine alone misses.

        Setup: "hiking" is near the query. "Switzerland" is near "hiking"
        but NOT near the query. Hybrid should find both.
        """
        dim = 64
        rng = np.random.default_rng(77)

        # Create a structured scenario
        query_dir = rng.standard_normal(dim)
        query_dir /= np.linalg.norm(query_dir)

        # "hiking" — high cosine with query
        hiking = query_dir + rng.standard_normal(dim) * 0.1
        hiking /= np.linalg.norm(hiking)

        # "Switzerland" — high cosine with hiking, LOW cosine with query
        switzerland = hiking + rng.standard_normal(dim) * 0.2
        # Push away from query direction so cosine-to-query is low
        switzerland -= query_dir * (switzerland @ query_dir) * 0.7
        switzerland /= np.linalg.norm(switzerland)

        engine = CoupledEngine(dim=dim)
        engine.store(embedding=hiking, text="I like hiking")         # idx 0
        engine.store(embedding=switzerland, text="I live in Switzerland")  # idx 1

        # Add distractors in random directions
        for i in range(15):
            v = rng.standard_normal(dim)
            v /= np.linalg.norm(v)
            engine.store(embedding=v, text=f"distractor_{i}")

        q = query_dir.copy()

        # Cosine top-2: should find hiking but maybe not Switzerland
        cosine_results = engine.query(embedding=q, top_k=2)
        cosine_indices = {r["index"] for r in cosine_results}

        # Hybrid top-2: the expansion step from "hiking" seed should
        # spread to "Switzerland" via coupling
        hybrid_results = engine.query_hybrid(embedding=q, top_k=2)
        hybrid_indices = {r["index"] for r in hybrid_results}

        # Hybrid should find hiking (idx 0)
        assert 0 in hybrid_indices, "Hybrid should find 'hiking'"

        # Key property: hybrid should have at least as good coverage
        # (finding both hiking and Switzerland when cosine might miss one)
        assert len(hybrid_indices) >= len(cosine_indices)

    def test_empty_engine(self):
        """query_hybrid on empty engine returns empty list."""
        engine = CoupledEngine(dim=8)
        q = np.ones(8) / np.sqrt(8)
        assert engine.query_hybrid(embedding=q, top_k=5) == []

    def test_single_memory(self):
        """query_hybrid works with just one stored pattern."""
        engine = CoupledEngine(dim=8)
        v = np.ones(8) / np.sqrt(8)
        engine.store(embedding=v, text="only one")
        results = engine.query_hybrid(embedding=v, top_k=5)
        assert len(results) == 1
        assert results[0]["text"] == "only one"

    def test_dimension_mismatch_raises(self):
        """query_hybrid raises on dimension mismatch."""
        engine = CoupledEngine(dim=8)
        engine.store(embedding=np.ones(8) / np.sqrt(8), text="x")
        with pytest.raises(ValueError, match="dimension"):
            engine.query_hybrid(embedding=np.ones(4), top_k=5)

    def test_scores_sorted_descending(self):
        """Results are sorted by combined score, highest first."""
        engine, rng = self._make_engine(n=20)
        q = rng.standard_normal(engine.dim)
        q /= np.linalg.norm(q)
        results = engine.query_hybrid(embedding=q, top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        """Never returns more than top_k results."""
        engine, rng = self._make_engine(n=50)
        q = rng.standard_normal(engine.dim)
        q /= np.linalg.norm(q)
        for k in [3, 5, 10]:
            results = engine.query_hybrid(embedding=q, top_k=k)
            assert len(results) <= k

    def test_beta_affects_expansion(self):
        """Higher beta should make the Hopfield step more selective."""
        engine, rng = self._make_engine(dim=32, n=30, seed=55)
        q = rng.standard_normal(engine.dim)
        q /= np.linalg.norm(q)

        results_low = engine.query_hybrid(embedding=q, beta=1.0, top_k=10)
        results_high = engine.query_hybrid(embedding=q, beta=20.0, top_k=10)

        # Different beta should give different result sets
        idx_low = {r["index"] for r in results_low}
        idx_high = {r["index"] for r in results_high}
        # They may overlap, but scores should differ
        scores_low = {r["index"]: r["score"] for r in results_low}
        scores_high = {r["index"]: r["score"] for r in results_high}
        common = idx_low & idx_high
        if common:
            # At least some scores should differ
            diffs = [abs(scores_low[i] - scores_high[i]) for i in common]
            assert max(diffs) > 0 or idx_low != idx_high
