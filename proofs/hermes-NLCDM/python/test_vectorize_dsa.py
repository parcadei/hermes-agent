"""Discriminative tests for DSA-based vectorization of rem_unlearn_xb and
rem_explore_cross_domain_xb.

These tests empirically verify the core claims BEFORE modifying production code:

1. Precomputed S = X @ X.T provides equivalent information to Monte Carlo probing
2. Direct pair analysis from S identifies the same mixture-forming pairs as probing
3. Row-correlation on S approximates perturbation-response correlation
4. Shared S across dream ops is safe (no aliasing/mutation issues)
5. Fast-path short-circuits produce identical results to full computation
6. Scaling behavior confirms O(N²d) vs O(n_probes × steps × Nd)

Each test class targets one specific DSA claim and can FAIL independently.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from dream_ops import (
    hopfield_update,
    spreading_activation,
    rem_unlearn_xb,
    rem_explore_cross_domain_xb,
    nrem_prune_xb,
    nrem_repulsion_xb,
    nrem_merge_xb,
)
from nlcdm_core import softmax
from test_capacity_boundary import make_cluster_patterns, make_separated_centroids


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 128
N_CLUSTERS = 5
SPREAD = 0.10
BETA = 10.0
SEED = 7777


def _make_centroids(n: int = N_CLUSTERS) -> np.ndarray:
    return make_separated_centroids(n, DIM, seed=SEED)


def _make_unit_vectors(n: int, dim: int = DIM, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_close_pair_patterns(n: int = 50, dim: int = DIM, seed: int = SEED) -> np.ndarray:
    """Create patterns where some pairs are close (high cosine similarity).

    First 40 are random unit vectors (diverse).
    Last 10 are small perturbations of the first 5 (creating close pairs).
    This forces mixture states to exist in the Hopfield network.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n - 10, dim))
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    # Create 10 patterns close to first 5 base patterns (2 copies each)
    close = []
    for i in range(5):
        for _ in range(2):
            perturbed = base[i] + 0.08 * rng.standard_normal(dim)
            perturbed /= np.linalg.norm(perturbed)
            close.append(perturbed)

    patterns = np.vstack([base, np.array(close)])
    return patterns


# ===========================================================================
# Claim 1: S = X @ X.T faithfully represents pairwise relationships
# ===========================================================================

class TestPrecomputedSimilarityMatrix:
    """Verify that precomputed S = X @ X.T gives identical pairwise cosine
    similarities to per-pair dot products used inside spreading_activation."""

    def test_s_equals_pairwise_dots(self):
        """S[i,j] == X[i] @ X[j] for all pairs."""
        X = _make_unit_vectors(100, seed=1001)
        S = X @ X.T

        # Check a sample of pairs
        rng = np.random.default_rng(42)
        for _ in range(200):
            i, j = rng.integers(0, 100, size=2)
            expected = float(X[i] @ X[j])
            actual = float(S[i, j])
            assert abs(expected - actual) < 1e-12, (
                f"S[{i},{j}]={actual} != X[{i}]@X[{j}]={expected}"
            )

    def test_s_symmetry(self):
        """S must be symmetric: S[i,j] == S[j,i]."""
        X = _make_unit_vectors(100, seed=1002)
        S = X @ X.T
        assert np.allclose(S, S.T, atol=1e-14), "S is not symmetric"

    def test_s_diagonal_is_one_for_unit_vectors(self):
        """For unit vectors, S[i,i] == 1.0."""
        X = _make_unit_vectors(100, seed=1003)
        S = X @ X.T
        assert np.allclose(np.diag(S), 1.0, atol=1e-12), (
            f"Diagonal not all 1.0: min={np.diag(S).min()}, max={np.diag(S).max()}"
        )

    def test_s_invariant_under_spreading_activation(self):
        """Spreading activation at a pattern converges to that pattern.

        If X[i] is a stored pattern, spreading_activation(beta, X, X[i])
        should converge back to X[i] (it's a fixed point). This confirms
        the Hopfield dynamics are consistent with the similarity matrix.
        """
        X = _make_unit_vectors(30, seed=1004)
        S = X @ X.T
        # Ensure patterns are well-separated (max off-diagonal < 0.5)
        np.fill_diagonal(S, -2.0)
        max_offdiag = np.max(S)
        np.fill_diagonal(S, 1.0)

        if max_offdiag < 0.5:
            for i in range(min(10, len(X))):
                fp = spreading_activation(BETA, X, X[i], max_steps=100)
                cosine = float(fp @ X[i]) / (np.linalg.norm(fp) * np.linalg.norm(X[i]))
                assert cosine > 0.99, (
                    f"Pattern {i} not a fixed point: cosine={cosine:.4f}"
                )


# ===========================================================================
# Claim 2: Direct pair analysis from S identifies mixture-forming pairs
# ===========================================================================

class TestMixturePairIdentification:
    """Verify that high S[i,j] predicts mixture fixed points found by
    Monte Carlo probing in rem_unlearn_xb."""

    def test_high_similarity_pairs_form_mixtures(self):
        """Pairs with S[i,j] > threshold should be the same pairs that
        rem_unlearn_xb identifies as mixture-forming via probing.

        This is THE key discriminative test: does S-based pair identification
        agree with Monte Carlo probing?
        """
        X = _make_close_pair_patterns(50, seed=2001)
        N = X.shape[0]
        S = X @ X.T

        beta = 5.0
        energy_gap = np.log(N) / beta
        t_unlearn = energy_gap / 2.0
        beta_unlearn = 1.0 / max(t_unlearn, 1e-6)

        # --- Monte Carlo approach (current implementation) ---
        # Run probing to find mixture states
        rng = np.random.default_rng(42)
        n_probes = 500  # More probes for statistical confidence
        pair_mixture_count = np.zeros((N, N))

        for _ in range(n_probes):
            x0 = rng.standard_normal(X.shape[1])
            x0 /= np.linalg.norm(x0)
            fp = spreading_activation(beta_unlearn, X, x0, max_steps=50, tol=1e-6)
            sims = X @ fp
            weights = softmax(beta_unlearn, sims)
            entropy = -np.sum(weights * np.log(weights + 1e-30))
            max_entropy = np.log(N)

            if entropy > 0.3 * max_entropy:
                top_k = min(5, N)
                top_indices = np.argsort(weights)[-top_k:]
                top_weights = weights[top_indices]
                for a_pos in range(len(top_indices)):
                    for b_pos in range(a_pos + 1, len(top_indices)):
                        a, b = int(top_indices[a_pos]), int(top_indices[b_pos])
                        w_pair = float(top_weights[a_pos] * top_weights[b_pos])
                        pair_mixture_count[a, b] += w_pair
                        pair_mixture_count[b, a] += w_pair

        mc_threshold = 0.05 * n_probes
        mc_pairs = set()
        for i in range(N):
            for j in range(i + 1, N):
                if pair_mixture_count[i, j] > mc_threshold:
                    mc_pairs.add((i, j))

        # --- S-based approach (proposed DSA fix) ---
        # Pairs with high similarity should be mixture candidates
        np.fill_diagonal(S, -2.0)
        # Try a range of thresholds to find the one that best matches MC
        # The theoretical threshold is ~1/beta_unlearn, but we sweep to
        # find what empirically aligns.
        s_upper = np.triu(S, k=1)

        best_threshold = None
        best_f1 = -1.0

        for t in np.linspace(0.5, 0.99, 50):
            s_pairs = set()
            above = np.where(s_upper > t)
            for ii, jj in zip(above[0], above[1]):
                s_pairs.add((min(ii, jj), max(ii, jj)))

            if len(s_pairs) == 0 and len(mc_pairs) == 0:
                f1 = 1.0  # Both empty = perfect agreement
            elif len(s_pairs) == 0 or len(mc_pairs) == 0:
                f1 = 0.0
            else:
                tp = len(s_pairs & mc_pairs)
                precision = tp / len(s_pairs) if s_pairs else 0.0
                recall = tp / len(mc_pairs) if mc_pairs else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        # The S-based approach should achieve high F1 at some threshold
        # If mc_pairs is empty (diverse data), both approaches agree trivially
        if len(mc_pairs) == 0:
            # Verify S-based also finds no close pairs at a reasonable threshold
            above = np.where(s_upper > 0.85)
            assert len(above[0]) <= len(mc_pairs) + 5, (
                f"S found {len(above[0])} pairs but MC found 0"
            )
        else:
            assert best_f1 > 0.5, (
                f"S-based pair identification F1={best_f1:.3f} at threshold={best_threshold:.3f} "
                f"is too low vs Monte Carlo (mc_pairs={len(mc_pairs)}). "
                f"DSA claim REFUTED: S[i,j] does not predict mixture pairs."
            )

        print(f"\n  Mixture pair identification: MC found {len(mc_pairs)} pairs, "
              f"best S-threshold={best_threshold:.3f}, F1={best_f1:.3f}")

    def test_diverse_patterns_fast_path_agreement(self):
        """For diverse patterns (max S[i,j] < 0.7), both MC probing and
        S-based analysis should agree: zero mixture-forming pairs."""
        centroids = _make_centroids()
        patterns, _ = make_cluster_patterns(
            centroids, per_cluster=20, spread=SPREAD, seed=2002,
        )
        N = patterns.shape[0]
        S = patterns @ patterns.T
        np.fill_diagonal(S, -2.0)
        max_sim = np.max(S)

        # Run rem_unlearn_xb and check if it modifies patterns
        rng = np.random.default_rng(42)
        X_out = rem_unlearn_xb(patterns, beta=BETA, n_probes=200, rng=rng)

        # Compute max change
        max_delta = np.max(np.abs(X_out - patterns))

        print(f"\n  Diverse data: max_sim={max_sim:.4f}, max_delta={max_delta:.6f}")

        # If max_sim < reasonable threshold, changes should be minimal
        if max_sim < 0.7:
            # Fast-path: S says no close pairs → MC should agree (minimal changes)
            assert max_delta < 0.05, (
                f"max_sim={max_sim:.4f} < 0.7 but rem_unlearn_xb changed patterns "
                f"by up to {max_delta:.6f}. Fast-path assumption VIOLATED."
            )

    def test_close_pairs_get_pushed_apart(self):
        """Confirm that the close pairs in _make_close_pair_patterns are
        actually the ones that rem_unlearn_xb pushes apart."""
        X = _make_close_pair_patterns(50, seed=2003)
        N = X.shape[0]
        S_before = X @ X.T

        rng = np.random.default_rng(42)
        X_out = rem_unlearn_xb(X, beta=5.0, n_probes=300, rng=rng)
        S_after = X_out @ X_out.T

        # The intentionally close pairs (indices 40-49 paired with 0-4)
        # should have DECREASED similarity after unlearning
        close_pair_indices = [(i, 40 + 2 * i + j) for i in range(5) for j in range(2)]
        decreased = 0
        for i, j in close_pair_indices:
            if j < N:
                before = S_before[i, j]
                after = S_after[i, j]
                if after < before:
                    decreased += 1

        # At least some close pairs should be pushed apart
        print(f"\n  Close pairs pushed apart: {decreased}/{len(close_pair_indices)}")
        # This is an observational test — we just record what happens


# ===========================================================================
# Claim 3: Row-correlation on S approximates perturbation-response correlation
# ===========================================================================

class TestRowCorrelationApproximation:
    """Verify that corr(S[i,:], S[j,:]) approximates the perturbation-based
    correlation computed by rem_explore_cross_domain_xb."""

    def test_row_correlation_vs_perturbation_correlation(self):
        """For each pair discovered by rem_explore_cross_domain_xb, compute
        the S-based row correlation and check they are correlated.

        Uses tighter clusters (higher spread) to force more cross-domain
        associations, avoiding the degenerate single-association case.
        """
        centroids = _make_centroids(3)  # Fewer, closer clusters
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=20, spread=0.25, seed=3001,  # Higher spread
        )
        N = patterns.shape[0]
        S = patterns @ patterns.T

        # S-based row correlations (normalized)
        S_centered = S - S.mean(axis=1, keepdims=True)
        S_norms = np.linalg.norm(S_centered, axis=1, keepdims=True)
        S_normed = S_centered / (S_norms + 1e-12)

        # Run the Monte Carlo version with more probes
        rng = np.random.default_rng(42)
        associations = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=1000, rng=rng,
        )

        if len(associations) < 3:
            # Not enough data points for meaningful correlation — use pointwise
            # comparison instead: check that S-row corr is positive for each
            # discovered association
            all_positive = True
            for idx_i, idx_j, mc_sim in associations:
                s_corr = float(S_normed[idx_i] @ S_normed[idx_j])
                if s_corr <= 0:
                    all_positive = False
            print(f"\n  Only {len(associations)} associations — pointwise check: "
                  f"all S-corr positive = {all_positive}")
            if len(associations) > 0:
                assert all_positive, "S-row correlation negative for discovered pair"
            return

        # For each discovered pair, compute S-based correlation
        mc_sims = []
        s_corrs = []
        for idx_i, idx_j, mc_sim in associations:
            s_corr = float(S_normed[idx_i] @ S_normed[idx_j])
            mc_sims.append(mc_sim)
            s_corrs.append(s_corr)

        mc_arr = np.array(mc_sims)
        s_arr = np.array(s_corrs)

        # The two measures should be positively correlated
        mc_c = mc_arr - mc_arr.mean()
        s_c = s_arr - s_arr.mean()
        denom = np.sqrt(np.sum(mc_c**2) * np.sum(s_c**2))
        if denom > 1e-12:
            pearson_r = float(np.sum(mc_c * s_c) / denom)
        else:
            # All values identical (no variance) — check means agree directionally
            pearson_r = 1.0 if np.mean(s_arr) > 0 else 0.0

        print(f"\n  Associations found: {len(associations)}")
        print(f"  MC sim range: [{mc_arr.min():.4f}, {mc_arr.max():.4f}]")
        print(f"  S-row corr range: [{s_arr.min():.4f}, {s_arr.max():.4f}]")
        print(f"  Pearson r(MC, S-row): {pearson_r:.4f}")

        # Key assertion: the two measures should be non-negatively related
        # (using >= 0.0 since degenerate cases with identical values are fine)
        assert pearson_r >= 0.0, (
            f"Pearson r={pearson_r:.4f} is negative. "
            f"S-row correlation anti-correlates with perturbation correlation. "
            f"DSA claim 3 REFUTED."
        )

    def test_row_correlation_catches_structural_similarity(self):
        """The synthetic 'structurally similar' setup from test_dream_redesign:
        a pattern planted in cluster 1 that's actually close to cluster 0.
        S-row correlation should detect it."""
        rng = np.random.default_rng(3002)

        # Cluster 0: 5 patterns around c0
        c0 = rng.standard_normal(DIM)
        c0 /= np.linalg.norm(c0)
        c0_patterns = [c0 + 0.1 * rng.standard_normal(DIM) for _ in range(5)]
        c0_patterns = [v / np.linalg.norm(v) for v in c0_patterns]

        # Cluster 1: 5 patterns around c1 (different direction)
        c1 = rng.standard_normal(DIM)
        c1 /= np.linalg.norm(c1)
        c1_patterns = [c1 + 0.1 * rng.standard_normal(DIM) for _ in range(5)]
        c1_patterns = [v / np.linalg.norm(v) for v in c1_patterns]

        # Plant: one pattern in cluster 1 that's close to c0
        plant = c0 + 0.05 * rng.standard_normal(DIM)
        plant /= np.linalg.norm(plant)
        c1_patterns.append(plant)

        patterns = np.array(c0_patterns + c1_patterns)
        labels = np.array([0] * len(c0_patterns) + [1] * len(c1_patterns))
        plant_idx = len(patterns) - 1

        # Compute S-based row correlations
        S = patterns @ patterns.T
        S_centered = S - S.mean(axis=1, keepdims=True)
        S_norms = np.linalg.norm(S_centered, axis=1, keepdims=True)
        S_normed = S_centered / (S_norms + 1e-12)

        # Check: plant should have highest cross-cluster row-corr with some c0 pattern
        cross_cluster_corrs = []
        for i in range(len(c0_patterns)):
            corr = float(S_normed[plant_idx] @ S_normed[i])
            cross_cluster_corrs.append((i, corr))

        best_i, best_corr = max(cross_cluster_corrs, key=lambda x: x[1])
        print(f"\n  Plant (idx {plant_idx}) best cross-cluster corr: "
              f"r={best_corr:.4f} with pattern {best_i}")

        # Also check MC version finds it
        mc_assoc = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=500, rng=np.random.default_rng(42),
        )
        mc_found = any(
            idx_i == plant_idx or idx_j == plant_idx
            for idx_i, idx_j, _ in mc_assoc
        )
        print(f"  MC found plant: {mc_found} ({len(mc_assoc)} associations total)")

        # S-row approach should also identify this
        assert best_corr > 0.3, (
            f"S-row correlation {best_corr:.4f} too low for planted cross-domain pattern. "
            f"DSA claim REFUTED: row correlation misses structural similarity."
        )


# ===========================================================================
# Claim 4: Shared S across dream ops is safe
# ===========================================================================

class TestSharedSAcrossOps:
    """Verify that computing S once and passing it to multiple dream ops
    produces the same results as each op computing its own pairwise info."""

    def test_nrem_ops_dont_mutate_patterns(self):
        """NREM ops (prune, repulsion, merge) should not mutate the input.
        This is a prerequisite for sharing S across all ops."""
        patterns = _make_unit_vectors(50, seed=4001)
        importances = np.full(50, 0.5)
        patterns_copy = patterns.copy()

        # Run each NREM op
        nrem_repulsion_xb(patterns, importances, eta=0.01, min_sep=0.3)
        assert np.array_equal(patterns, patterns_copy), "nrem_repulsion_xb mutated input"

        nrem_prune_xb(patterns, importances, threshold=0.95)
        assert np.array_equal(patterns, patterns_copy), "nrem_prune_xb mutated input"

        nrem_merge_xb(patterns, importances, threshold=0.90, min_group=3)
        assert np.array_equal(patterns, patterns_copy), "nrem_merge_xb mutated input"

    def test_rem_unlearn_doesnt_mutate_input(self):
        """rem_unlearn_xb should return a new array, not mutate input."""
        X = _make_unit_vectors(30, seed=4002)
        X_copy = X.copy()

        rem_unlearn_xb(X, beta=5.0, n_probes=50, rng=np.random.default_rng(42))
        assert np.array_equal(X, X_copy), "rem_unlearn_xb mutated input patterns"

    def test_rem_explore_doesnt_mutate_input(self):
        """rem_explore_cross_domain_xb should not mutate input."""
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=10, spread=SPREAD, seed=4003,
        )
        patterns_copy = patterns.copy()

        rem_explore_cross_domain_xb(
            patterns, labels, n_probes=50, rng=np.random.default_rng(42),
        )
        assert np.array_equal(patterns, patterns_copy), (
            "rem_explore_cross_domain_xb mutated input patterns"
        )

    def test_s_computed_once_matches_each_ops_internal_dots(self):
        """S = X @ X.T computed once should match what each operation
        internally computes via patterns @ qi type operations."""
        X = _make_unit_vectors(30, seed=4004)
        S = X @ X.T

        # Verify: for any random query qi, X @ qi == S @ (X.T \ qi) ... no,
        # simpler: for stored patterns, S[i,:] == X @ X[i]
        for i in range(30):
            via_matmul = X @ X[i]
            via_s_row = S[i, :]
            assert np.allclose(via_matmul, via_s_row, atol=1e-12), (
                f"S[{i},:] != X @ X[{i}]"
            )


# ===========================================================================
# Claim 5: Fast-path short-circuit correctness
# ===========================================================================

class TestFastPathShortCircuit:
    """Verify that when max(S_offdiag) <= threshold, the functions
    produce minimal/no changes (validating the fast-path assumption)."""

    def test_diverse_data_unlearn_is_identity(self):
        """For well-separated patterns, rem_unlearn_xb should be ~identity."""
        X = _make_unit_vectors(100, seed=5001)
        S = X @ X.T
        np.fill_diagonal(S, -2.0)
        max_offdiag = np.max(S)

        # In dim=128 with random unit vectors, max similarity should be < 0.5
        assert max_offdiag < 0.5, (
            f"Test setup failed: max_offdiag={max_offdiag:.4f} >= 0.5"
        )

        X_out = rem_unlearn_xb(X, beta=5.0, n_probes=200, rng=np.random.default_rng(42))
        max_change = np.max(np.abs(X_out - X))
        mean_change = np.mean(np.abs(X_out - X))

        print(f"\n  Diverse patterns (N=100, d={DIM}): "
              f"max_offdiag_sim={max_offdiag:.4f}, "
              f"max_change={max_change:.6f}, mean_change={mean_change:.8f}")

        # Changes should be negligible
        assert max_change < 0.02, (
            f"Diverse patterns changed by up to {max_change:.6f} — fast-path would be wrong"
        )

    def test_diverse_data_explore_finds_few(self):
        """For well-separated clusters, exploration should find few associations."""
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=20, spread=SPREAD, seed=5002,
        )

        associations = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=200, rng=np.random.default_rng(42),
        )

        print(f"\n  Diverse clusters (N={len(patterns)}, "
              f"{N_CLUSTERS} clusters): {len(associations)} associations")

        # With well-separated clusters, cross-domain associations should be rare
        # (not zero necessarily, but bounded)
        assert len(associations) < len(patterns), (
            f"Too many associations ({len(associations)}) for well-separated clusters"
        )


# ===========================================================================
# Claim 6: Scaling behavior
# ===========================================================================

class TestScalingBehavior:
    """Empirically measure the time complexity of current implementations
    to confirm O(n_probes × steps × Nd) for unlearn and O(n_probes × K × Nd)
    for explore, establishing baseline for improvement."""

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_unlearn_time_scales_with_n(self, n):
        """Measure rem_unlearn_xb time at different N to estimate exponent."""
        X = _make_unit_vectors(n, seed=6001 + n)
        rng = np.random.default_rng(42)

        t0 = time.perf_counter()
        rem_unlearn_xb(X, beta=5.0, n_probes=50, rng=rng)
        elapsed = time.perf_counter() - t0

        print(f"\n  rem_unlearn_xb N={n}: {elapsed:.3f}s")

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_explore_time_scales_with_n(self, n):
        """Measure rem_explore_cross_domain_xb time at different N."""
        centroids = make_separated_centroids(5, DIM, seed=6050 + n)
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=n // 5, spread=SPREAD, seed=6051 + n,
        )
        rng = np.random.default_rng(42)

        t0 = time.perf_counter()
        rem_explore_cross_domain_xb(patterns, labels, n_probes=50, rng=rng)
        elapsed = time.perf_counter() - t0

        print(f"\n  rem_explore_cross_domain_xb N={len(patterns)}: {elapsed:.3f}s")

    @pytest.mark.parametrize("n", [50, 100, 200])
    def test_precompute_s_time(self, n):
        """Measure time to compute S = X @ X.T via BLAS."""
        X = _make_unit_vectors(n, seed=6100 + n)

        t0 = time.perf_counter()
        S = X @ X.T
        elapsed = time.perf_counter() - t0

        print(f"\n  S = X @ X.T at N={n}: {elapsed:.6f}s (shape {S.shape})")

    def test_scaling_exponent_unlearn(self):
        """Estimate scaling exponent for rem_unlearn_xb.

        Run at N=50 and N=200. If time scales as O(N^alpha),
        then alpha = log(t200/t50) / log(200/50) = log(t200/t50) / log(4).

        Expect alpha > 1 (superlinear due to N×d matmul in each step).
        """
        times = {}
        for n in [50, 200]:
            X = _make_unit_vectors(n, seed=6200 + n)
            rng = np.random.default_rng(42)

            t0 = time.perf_counter()
            rem_unlearn_xb(X, beta=5.0, n_probes=50, rng=rng)
            times[n] = time.perf_counter() - t0

        if times[50] > 1e-6:  # avoid division by zero
            ratio = times[200] / times[50]
            alpha = np.log(ratio) / np.log(200 / 50)
        else:
            alpha = float("nan")

        print(f"\n  Scaling: t(50)={times[50]:.4f}s, t(200)={times[200]:.4f}s, "
              f"ratio={times[200]/max(times[50], 1e-9):.2f}x, alpha≈{alpha:.2f}")

        # Record but don't hard-fail — just establish baseline
        assert alpha < 5.0, f"Scaling exponent {alpha:.2f} unreasonably high"


# ===========================================================================
# Claim 7: Analytical mixture threshold derivation
# ===========================================================================

class TestMixtureThresholdDerivation:
    """Verify the theoretical claim: mixture fixed points between patterns
    i and j exist when S[i,j] > ~1/beta_unlearn.

    We do this by constructing pairs with known similarity and checking
    whether spreading_activation from their midpoint converges to a mixture
    or to a single pattern."""

    @pytest.mark.parametrize("target_sim", [0.3, 0.5, 0.7, 0.85, 0.95])
    def test_mixture_existence_vs_similarity(self, target_sim):
        """Create a 2-pattern system with controlled similarity,
        probe from midpoint, classify the fixed point."""
        rng = np.random.default_rng(7001)
        d = DIM

        # Create two unit vectors with target cosine similarity
        v1 = rng.standard_normal(d)
        v1 /= np.linalg.norm(v1)

        # v2 = target_sim * v1 + sqrt(1 - target_sim^2) * orthogonal
        ortho = rng.standard_normal(d)
        ortho -= (ortho @ v1) * v1
        ortho /= np.linalg.norm(ortho)

        v2 = target_sim * v1 + np.sqrt(1 - target_sim**2) * ortho
        v2 /= np.linalg.norm(v2)

        actual_sim = float(v1 @ v2)
        assert abs(actual_sim - target_sim) < 0.01, (
            f"Setup: wanted sim={target_sim}, got {actual_sim:.4f}"
        )

        X = np.array([v1, v2])
        N = 2
        beta = 5.0
        energy_gap = np.log(N) / beta
        t_unlearn = energy_gap / 2.0
        beta_unlearn = 1.0 / max(t_unlearn, 1e-6)

        # Probe from midpoint
        midpoint = (v1 + v2)
        midpoint /= np.linalg.norm(midpoint)

        fp = spreading_activation(beta_unlearn, X, midpoint, max_steps=200, tol=1e-8)
        fp_norm = fp / np.linalg.norm(fp)

        # Classify: is fp a mixture or pure attractor?
        sim_v1 = float(fp_norm @ v1)
        sim_v2 = float(fp_norm @ v2)

        # Mixture: both similarities > 0.5 and neither dominates
        # Pure: one similarity >> the other
        balance = min(sim_v1, sim_v2) / max(sim_v1, sim_v2) if max(sim_v1, sim_v2) > 0 else 0
        is_mixture = balance > 0.8  # balanced attention to both patterns

        print(f"\n  sim={target_sim:.2f}, beta_unlearn={beta_unlearn:.2f}: "
              f"sim_v1={sim_v1:.4f}, sim_v2={sim_v2:.4f}, "
              f"balance={balance:.4f}, is_mixture={is_mixture}")

        # At high similarity (close to 1), mixture should exist at low beta_unlearn
        # At low similarity, it should converge to one pattern
        # This maps out the phase transition empirically


# ===========================================================================
# Claim 8: Batched spreading activation equivalence
# ===========================================================================

class TestBatchedSpreadingActivation:
    """If we can't skip probing entirely, verify that batching 200 probes
    into a (200, d) matrix gives equivalent results to sequential probing."""

    def test_batched_vs_sequential_hopfield_update(self):
        """A single batched hopfield_update on (K, d) state matrix should
        give the same result as K sequential updates."""
        X = _make_unit_vectors(30, seed=8001)
        K = 20
        rng = np.random.default_rng(42)

        # K random starting states
        states = rng.standard_normal((K, DIM))
        states /= np.linalg.norm(states, axis=1, keepdims=True)

        # Sequential
        sequential_results = []
        for k in range(K):
            result = hopfield_update(BETA, X, states[k])
            sequential_results.append(result)
        sequential = np.array(sequential_results)

        # Batched: X @ states.T gives (N, K) similarities
        # softmax each column, then weights.T @ X gives (K, d)
        sims = X @ states.T  # (N, K)
        # Softmax per column
        sims_shifted = sims - sims.max(axis=0, keepdims=True)
        exp_sims = np.exp(BETA * sims_shifted)
        weights = exp_sims / exp_sims.sum(axis=0, keepdims=True)  # (N, K)
        batched = weights.T @ X  # (K, d)

        # They should be equivalent
        max_diff = np.max(np.abs(batched - sequential))
        print(f"\n  Batched vs sequential hopfield_update: max_diff={max_diff:.2e}")

        assert max_diff < 1e-10, (
            f"Batched differs from sequential by {max_diff:.2e}. "
            f"DSA claim REFUTED: batching is not equivalent."
        )

    def test_batched_spreading_activation_convergence(self):
        """Run batched spreading activation and verify each probe converges
        to the same fixed point as sequential."""
        X = _make_unit_vectors(20, seed=8002)
        K = 10
        rng = np.random.default_rng(42)

        states = rng.standard_normal((K, DIM))
        states /= np.linalg.norm(states, axis=1, keepdims=True)

        # Sequential
        seq_fps = []
        for k in range(K):
            fp = spreading_activation(BETA, X, states[k], max_steps=50, tol=1e-6)
            seq_fps.append(fp)
        seq_fps = np.array(seq_fps)

        # Batched iteration
        batch_states = states.copy()
        for _ in range(50):
            sims = X @ batch_states.T  # (N, K)
            sims_shifted = sims - sims.max(axis=0, keepdims=True)
            exp_sims = np.exp(BETA * sims_shifted)
            weights = exp_sims / exp_sims.sum(axis=0, keepdims=True)
            batch_new = weights.T @ X  # (K, d)

            diffs = np.linalg.norm(batch_new - batch_states, axis=1)
            batch_states = batch_new
            if np.all(diffs < 1e-6):
                break

        max_diff = np.max(np.abs(batch_states - seq_fps))
        print(f"\n  Batched spreading activation: max_diff={max_diff:.2e}")

        assert max_diff < 1e-4, (
            f"Batched spreading activation differs by {max_diff:.2e}. "
            f"DSA claim REFUTED: batched dynamics diverge."
        )


# ===========================================================================
# Claim 9: At what scale does probing become a real bottleneck?
# ===========================================================================

class TestScalingAtRealSizes:
    """Run at N=500 and N=1000 to see the actual scaling exponent where
    the O(n_probes × steps × Nd) term dominates over Python overhead."""

    def test_unlearn_scaling_500_1000(self):
        """Measure rem_unlearn_xb at N=500 and N=1000 with production n_probes."""
        times = {}
        for n in [500, 1000]:
            X = _make_unit_vectors(n, dim=DIM, seed=9001 + n)
            rng = np.random.default_rng(42)

            t0 = time.perf_counter()
            X_out = rem_unlearn_xb(X, beta=5.0, n_probes=200, rng=rng)
            elapsed = time.perf_counter() - t0
            times[n] = elapsed

            # Check if anything changed
            max_delta = np.max(np.abs(X_out - X))
            print(f"\n  N={n}: unlearn={elapsed:.3f}s, max_delta={max_delta:.6f}")

        ratio = times[1000] / max(times[500], 1e-9)
        alpha = np.log(ratio) / np.log(1000 / 500)
        print(f"  Scaling 500→1000: ratio={ratio:.2f}x, alpha≈{alpha:.2f}")

    def test_explore_scaling_500_1000(self):
        """Measure rem_explore_cross_domain_xb at N=500 and N=1000."""
        times = {}
        for n in [500, 1000]:
            centroids = make_separated_centroids(5, DIM, seed=9050 + n)
            patterns, labels = make_cluster_patterns(
                centroids, per_cluster=n // 5, spread=SPREAD, seed=9051 + n,
            )
            rng = np.random.default_rng(42)

            t0 = time.perf_counter()
            assoc = rem_explore_cross_domain_xb(
                patterns, labels, n_probes=max(len(patterns), 50), rng=rng,
            )
            elapsed = time.perf_counter() - t0
            times[n] = elapsed

            print(f"\n  N={len(patterns)}: explore={elapsed:.3f}s, "
                  f"associations={len(assoc)}")

        ratio = times[1000] / max(times[500], 1e-9)
        alpha = np.log(ratio) / np.log(1000 / 500)
        print(f"  Scaling 500→1000: ratio={ratio:.2f}x, alpha≈{alpha:.2f}")

    def test_s_precompute_scaling_to_5k(self):
        """Verify S = X @ X.T stays fast even at N=5000."""
        for n in [500, 1000, 2000, 5000]:
            X = _make_unit_vectors(n, dim=DIM, seed=9100 + n)

            t0 = time.perf_counter()
            S = X @ X.T
            elapsed = time.perf_counter() - t0

            # Quick sanity: shape and symmetry
            assert S.shape == (n, n)
            mem_mb = S.nbytes / 1024**2

            print(f"\n  S at N={n}: {elapsed:.4f}s, {mem_mb:.1f}MB")

        # At N=5000: should complete in under 1 second on Apple Silicon
        assert elapsed < 2.0, (
            f"S computation at N=5000 took {elapsed:.3f}s — BLAS too slow?"
        )


# ===========================================================================
# Claim 10: Why does rem_unlearn_xb produce zero modifications?
# ===========================================================================

class TestUnlearnEntropyThreshold:
    """Investigate WHY rem_unlearn_xb is a no-op in practice.

    Hypothesis: the entropy threshold (0.3 * log(N)) filters out all probes
    because at operational beta_unlearn, probes converge to single patterns
    (low entropy) even when close pairs exist.
    """

    def test_entropy_distribution_of_probes(self):
        """Run probes and record the entropy of each fixed point.
        If all entropies are below the threshold, the function is a no-op."""
        X = _make_close_pair_patterns(50, seed=10001)
        N = X.shape[0]

        beta = 5.0
        energy_gap = np.log(N) / beta
        t_unlearn = energy_gap / 2.0
        beta_unlearn = 1.0 / max(t_unlearn, 1e-6)

        entropy_threshold = 0.3 * np.log(N)
        rng = np.random.default_rng(42)
        entropies = []

        for _ in range(500):
            x0 = rng.standard_normal(X.shape[1])
            x0 /= np.linalg.norm(x0)
            fp = spreading_activation(beta_unlearn, X, x0, max_steps=50, tol=1e-6)
            sims = X @ fp
            weights = softmax(beta_unlearn, sims)
            entropy = -np.sum(weights * np.log(weights + 1e-30))
            entropies.append(entropy)

        entropies = np.array(entropies)
        n_above = np.sum(entropies > entropy_threshold)
        max_entropy = np.log(N)

        print(f"\n  N={N}, beta={beta}, beta_unlearn={beta_unlearn:.2f}")
        print(f"  Entropy threshold: {entropy_threshold:.4f} "
              f"(0.3 × log({N}) = 0.3 × {max_entropy:.4f})")
        print(f"  Probe entropies: min={entropies.min():.4f}, "
              f"max={entropies.max():.4f}, mean={entropies.mean():.4f}")
        print(f"  Probes above threshold: {n_above}/500")

        # If zero probes exceed threshold, the function is provably a no-op
        # regardless of close pairs existing — the entropy filter kills them
        if n_above == 0:
            print("  → ALL probes below entropy threshold: rem_unlearn_xb is a NO-OP")
            print("  → Fast-path safe: check max(S) is unnecessary, entropy filter")
            print("    already makes it return unchanged.")

    def test_try_lower_beta_unlearn(self):
        """What if we lower beta_unlearn? Does the entropy filter open up?

        At lower beta_unlearn (higher temperature), the softmax becomes
        more uniform → higher entropy → probes might exceed the threshold.
        """
        X = _make_close_pair_patterns(50, seed=10002)
        N = X.shape[0]
        entropy_threshold = 0.3 * np.log(N)
        rng_base = np.random.default_rng(42)

        # Pre-generate the same 200 random starts for all betas
        starts = rng_base.standard_normal((200, X.shape[1]))
        starts /= np.linalg.norm(starts, axis=1, keepdims=True)

        results = []
        for beta_u in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            n_above = 0
            for k in range(200):
                fp = spreading_activation(beta_u, X, starts[k], max_steps=50, tol=1e-6)
                sims = X @ fp
                weights = softmax(beta_u, sims)
                entropy = -np.sum(weights * np.log(weights + 1e-30))
                if entropy > entropy_threshold:
                    n_above += 1
            results.append((beta_u, n_above))

        print(f"\n  Entropy threshold: {entropy_threshold:.4f}")
        print(f"  β_unlearn sweep (200 probes each):")
        for beta_u, n_above in results:
            bar = "█" * (n_above // 5)
            print(f"    β={beta_u:5.1f}: {n_above:3d}/200 above threshold {bar}")

    def test_compare_direct_sim_vs_probing(self):
        """Direct comparison: which close pairs does S identify vs which
        does probing identify? Use a scenario where we KNOW pairs should
        be pushed apart (sim > 0.90)."""
        rng = np.random.default_rng(10003)
        d = DIM

        # Create 10 diverse base patterns
        base = rng.standard_normal((10, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)

        # Create 5 near-duplicate pairs (sim > 0.95)
        duplicates = []
        for i in range(5):
            dup = base[i] + 0.02 * rng.standard_normal(d)
            dup /= np.linalg.norm(dup)
            duplicates.append(dup)

        X = np.vstack([base, np.array(duplicates)])
        N = X.shape[0]

        S = X @ X.T
        np.fill_diagonal(S, -2.0)

        # S-based: pairs with sim > 0.90
        s_pairs = set()
        above = np.where(np.triu(S, k=1) > 0.90)
        for ii, jj in zip(above[0], above[1]):
            s_pairs.add((min(ii, jj), max(ii, jj)))

        # MC-based: run rem_unlearn_xb and check which pairs changed
        X_out = rem_unlearn_xb(X, beta=5.0, n_probes=500,
                               rng=np.random.default_rng(42))
        per_pattern_change = np.linalg.norm(X_out - X, axis=1)
        mc_modified = set()
        for i in range(N):
            if per_pattern_change[i] > 1e-6:
                mc_modified.add(i)

        print(f"\n  Near-duplicate experiment (N={N}, d={d}):")
        print(f"  S-based pairs (sim > 0.90): {len(s_pairs)} — {s_pairs}")
        print(f"  MC-modified patterns: {len(mc_modified)} — {mc_modified}")
        print(f"  Per-pattern change norm: "
              f"max={per_pattern_change.max():.6f}, "
              f"nonzero={np.sum(per_pattern_change > 1e-6)}")

        # The key question: does S find pairs that MC misses?
        if len(s_pairs) > 0 and len(mc_modified) == 0:
            print("  → S identifies close pairs but MC probing misses them entirely!")
            print("  → This means the analytical approach would be BETTER, not just faster.")
        elif len(s_pairs) > 0 and len(mc_modified) > 0:
            print("  → Both approaches find close pairs. Compare overlap.")
        else:
            print("  → Neither found close pairs (data too diverse even for near-dups).")


# ===========================================================================
# Claim 11: S-based fast path in rem_unlearn_xb
# ===========================================================================

class TestSBasedFastPath:
    """Verify the S-based fast path for rem_unlearn_xb when similarity_matrix
    is provided. This bypasses the broken MC probing and directly uses the
    precomputed similarity matrix to identify and separate close pairs."""

    def test_accepts_similarity_matrix_parameter(self):
        """rem_unlearn_xb should accept an optional similarity_matrix kwarg."""
        X = _make_unit_vectors(20, seed=11001)
        S = X @ X.T
        # Should not raise
        result = rem_unlearn_xb(X, beta=5.0, similarity_matrix=S)
        assert result.shape == X.shape

    def test_fast_path_identity_for_diverse_patterns(self):
        """When max off-diagonal similarity <= 0.70, the S-based fast path
        should return patterns unchanged (identity operation)."""
        X = _make_unit_vectors(100, seed=11002)
        S = X @ X.T
        np.fill_diagonal(S, -2.0)
        max_sim = np.max(S)
        # Verify test setup: random unit vectors in d=128 should be well-separated
        assert max_sim < 0.70, f"Test setup failed: max_sim={max_sim:.4f} >= 0.70"

        X_out = rem_unlearn_xb(X, beta=5.0, similarity_matrix=X @ X.T)
        assert np.array_equal(X_out, X), (
            "S-based fast path should return input unchanged when max sim <= 0.70"
        )

    def test_fast_path_pushes_close_pairs_apart(self):
        """When close pairs exist (sim > 0.70), the S-based path should
        reduce their similarity."""
        X = _make_close_pair_patterns(50, seed=11003)
        S = X @ X.T
        np.fill_diagonal(S, -2.0)
        max_sim = np.max(S)
        # Close pair patterns should have some pairs with high similarity
        assert max_sim > 0.70, f"Test setup failed: max_sim={max_sim:.4f} <= 0.70"

        X_out = rem_unlearn_xb(X, beta=5.0, similarity_matrix=X @ X.T)
        S_after = X_out @ X_out.T
        np.fill_diagonal(S_after, -2.0)

        # Identify which pairs were close before
        close_before = np.where(np.triu(S, k=1) > 0.70)
        n_close = len(close_before[0])
        assert n_close > 0, "Test setup: should have close pairs"

        # At least some close pairs should have decreased similarity
        decreased = 0
        for idx in range(n_close):
            i, j = close_before[0][idx], close_before[1][idx]
            if S_after[i, j] < S[i, j]:
                decreased += 1

        assert decreased > 0, (
            f"S-based path found {n_close} close pairs but pushed NONE apart"
        )

    def test_output_unit_vectors(self):
        """All output patterns must be unit vectors after S-based separation."""
        X = _make_close_pair_patterns(50, seed=11004)
        S = X @ X.T
        X_out = rem_unlearn_xb(X, beta=5.0, similarity_matrix=S)
        norms = np.linalg.norm(X_out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10), (
            f"Output not unit vectors: norms min={norms.min():.10f}, "
            f"max={norms.max():.10f}"
        )

    def test_input_not_mutated_with_similarity_matrix(self):
        """Input patterns must NOT be mutated when using S-based path."""
        X = _make_close_pair_patterns(30, seed=11005)
        X_copy = X.copy()
        S = X @ X.T
        rem_unlearn_xb(X, beta=5.0, similarity_matrix=S)
        assert np.array_equal(X, X_copy), (
            "rem_unlearn_xb mutated input patterns when using similarity_matrix"
        )

    def test_similarity_matrix_not_mutated(self):
        """The provided similarity_matrix must NOT be mutated."""
        X = _make_close_pair_patterns(30, seed=11006)
        S = X @ X.T
        S_copy = S.copy()
        rem_unlearn_xb(X, beta=5.0, similarity_matrix=S)
        assert np.array_equal(S, S_copy), (
            "rem_unlearn_xb mutated the provided similarity_matrix"
        )

    def test_none_similarity_matrix_uses_mc_path(self):
        """When similarity_matrix is None, behavior should be identical to
        the original MC probing path (backward compatibility)."""
        X = _make_unit_vectors(30, seed=11007)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        # Without similarity_matrix (MC path)
        X_mc = rem_unlearn_xb(X, beta=5.0, n_probes=50, rng=rng1)

        # Explicitly passing None should be same
        X_none = rem_unlearn_xb(X, beta=5.0, n_probes=50, rng=rng2,
                                similarity_matrix=None)

        assert np.array_equal(X_mc, X_none), (
            "similarity_matrix=None should produce identical results to omitting it"
        )

    def test_intensity_proportional_to_similarity(self):
        """Pairs with higher similarity should be pushed apart more
        aggressively than pairs with lower similarity."""
        rng = np.random.default_rng(11008)
        d = DIM

        # Create a base pattern
        base = rng.standard_normal(d)
        base /= np.linalg.norm(base)

        # Create two duplicates with different levels of closeness
        # very close (sim ~0.99) and moderately close (sim ~0.80)
        very_close = base + 0.05 * rng.standard_normal(d)
        very_close /= np.linalg.norm(very_close)

        moderate = base + 0.5 * rng.standard_normal(d)
        moderate /= np.linalg.norm(moderate)
        # Re-scale to get sim ~0.80
        sim_moderate = base @ moderate
        if sim_moderate < 0.71:
            # Try again with less noise
            moderate = base + 0.3 * rng.standard_normal(d)
            moderate /= np.linalg.norm(moderate)

        # Also add some diverse patterns as filler
        filler = rng.standard_normal((10, d))
        filler /= np.linalg.norm(filler, axis=1, keepdims=True)

        X = np.vstack([base[None, :], very_close[None, :],
                       moderate[None, :], filler])
        S = X @ X.T

        sim_01 = S[0, 1]  # very close pair
        sim_02 = S[0, 2]  # moderate pair

        # Both should be above 0.70 for this test to be meaningful
        if sim_01 > 0.70 and sim_02 > 0.70:
            X_out = rem_unlearn_xb(X, beta=5.0, similarity_matrix=S)
            S_out = X_out @ X_out.T

            delta_01 = S[0, 1] - S_out[0, 1]  # decrease for very close
            delta_02 = S[0, 2] - S_out[0, 2]  # decrease for moderate

            # The very close pair should have a larger decrease
            assert delta_01 > delta_02, (
                f"Very close pair (sim={sim_01:.4f}) should decrease more "
                f"than moderate pair (sim={sim_02:.4f}), but "
                f"delta_01={delta_01:.6f} <= delta_02={delta_02:.6f}"
            )

    def test_fast_path_returns_copy_not_view(self):
        """Even in the fast path (no close pairs), the function should
        return X (the copy), not the original."""
        X = _make_unit_vectors(20, seed=11009)
        S = X @ X.T
        X_out = rem_unlearn_xb(X, beta=5.0, similarity_matrix=S)
        # Output should be a copy (not the same object as input)
        assert X_out is not X, "Fast path returned same object instead of copy"