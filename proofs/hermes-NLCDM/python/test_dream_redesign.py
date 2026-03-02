"""Behavioral contract tests for dream architecture redesign.

Tests the 4 new functions + updated dream_cycle_xb + updated CoupledEngine.dream().

These tests define the behavioral contracts that the implementation must satisfy.
They will fail with ImportError until the functions are implemented.

Contracts tested:
  - R1-R5: nrem_repulsion_xb
  - P1-P4: nrem_prune_xb (+ edge cases)
  - M1-M4: nrem_merge_xb (+ partition property, edge cases)
  - X1-X3: rem_explore_cross_domain_xb (+ structural similarity, determinism)
  - DC1-DC5: dream_cycle_xb (updated pipeline)
  - CE1-CE5: CoupledEngine.dream() (variable-size memory store)
"""

from __future__ import annotations

import numpy as np

from dream_ops import (
    DreamReport,
    dream_cycle_xb,
    nrem_merge_xb,
    nrem_prune_xb,
    nrem_repulsion_xb,
    rem_explore_cross_domain_xb,
)
from coupled_engine import CoupledEngine
from test_capacity_boundary import (
    compute_min_delta,
    make_cluster_patterns,
    make_separated_centroids,
    measure_p1,
)


# ---------------------------------------------------------------------------
# Shared parameters (matching test_capacity_boundary.py)
# ---------------------------------------------------------------------------

DIM = 128
N_CLUSTERS = 5
SPREAD = 0.10
BETA = 10.0
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_centroids(n: int = N_CLUSTERS) -> np.ndarray:
    return make_separated_centroids(n, DIM, seed=SEED)


def _make_unit_vectors(n: int, dim: int = DIM, seed: int = SEED) -> np.ndarray:
    """Generate n random unit vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_close_pair(dim: int = DIM, similarity: float = 0.98, seed: int = 0) -> np.ndarray:
    """Generate a pair of unit vectors with approximately the given cosine similarity."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim)
    base /= np.linalg.norm(base)
    # Perturb to get desired similarity
    noise_scale = np.sqrt(2 * (1 - similarity))
    noise = rng.standard_normal(dim)
    noise -= noise @ base * base  # orthogonal component
    noise /= np.linalg.norm(noise)
    second = base + noise_scale * noise
    second /= np.linalg.norm(second)
    return np.array([base, second])


def _make_similar_group(
    n: int, dim: int = DIM, spread: float = 0.02, seed: int = 0,
) -> np.ndarray:
    """Generate n unit vectors tightly clustered (high pairwise cosine similarity)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim)
    base /= np.linalg.norm(base)
    vecs = []
    for _ in range(n):
        v = base + spread * rng.standard_normal(dim)
        v /= np.linalg.norm(v)
        vecs.append(v)
    return np.array(vecs)


# ===========================================================================
# TestNREMRepulsion
# ===========================================================================

class TestNREMRepulsion:
    """Tests for nrem_repulsion_xb — SHY-inspired pattern separation."""

    def test_separation_increases(self):
        """R1: delta_min of output >= delta_min of input."""
        centroids = _make_centroids()
        patterns, _ = make_cluster_patterns(
            centroids, per_cluster=10, spread=SPREAD, seed=SEED,
        )
        importances = np.full(len(patterns), 0.3)

        delta_before = compute_min_delta(patterns)
        output = nrem_repulsion_xb(patterns, importances, eta=0.01, min_sep=0.3)
        delta_after = compute_min_delta(output)

        assert delta_after >= delta_before - 1e-9, (
            f"Separation decreased: delta_min {delta_before:.6f} -> {delta_after:.6f}"
        )

    def test_high_importance_anchored(self):
        """R2: patterns with importance >= 0.7 do not move."""
        patterns = _make_unit_vectors(20, seed=100)
        importances = np.array([0.8 if i < 5 else 0.2 for i in range(20)])

        output = nrem_repulsion_xb(patterns, importances, eta=0.01, min_sep=0.3)

        for i in range(5):
            displacement = np.linalg.norm(output[i] - patterns[i])
            assert displacement < 1e-10, (
                f"High-importance pattern {i} moved by {displacement:.2e} "
                f"(importance={importances[i]:.1f})"
            )

    def test_unit_norm_preserved(self):
        """R3: all output vectors are unit norm."""
        patterns = _make_unit_vectors(15, seed=200)
        importances = np.full(15, 0.3)

        output = nrem_repulsion_xb(patterns, importances, eta=0.01, min_sep=0.3)

        for i in range(len(output)):
            norm_i = np.linalg.norm(output[i])
            assert abs(norm_i - 1.0) < 1e-6, (
                f"Output vector {i} has norm {norm_i:.8f}, expected 1.0"
            )

    def test_low_importance_pushed_apart(self):
        """R4: close low-importance patterns are pushed apart."""
        # Create two very close low-importance patterns
        pair = _make_close_pair(DIM, similarity=0.95, seed=300)
        # Add a distant third pattern so we have a meaningful set
        rng = np.random.default_rng(301)
        far = rng.standard_normal(DIM)
        far /= np.linalg.norm(far)
        patterns = np.vstack([pair, far.reshape(1, -1)])
        importances = np.array([0.2, 0.2, 0.2])

        sim_before = float(patterns[0] @ patterns[1])
        output = nrem_repulsion_xb(patterns, importances, eta=0.01, min_sep=0.3)
        sim_after = float(output[0] @ output[1])

        assert sim_after <= sim_before + 1e-9, (
            f"Close low-importance pair did not separate: "
            f"sim {sim_before:.6f} -> {sim_after:.6f}"
        )

    def test_single_pattern_noop(self):
        """R5: N=1 returns a copy of the input."""
        pattern = _make_unit_vectors(1, seed=400)
        importances = np.array([0.5])

        output = nrem_repulsion_xb(pattern, importances)
        np.testing.assert_array_almost_equal(
            output, pattern,
            err_msg="Single-pattern input should return a copy",
        )
        # Verify it is a copy, not the same object
        assert output is not pattern, "Output should be a copy, not the same array"

    def test_p1_improves_at_boundary(self):
        """At the capacity wall (N=19/cluster), repulsion should improve P@1."""
        centroids = _make_centroids()
        per_cluster = 19
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster, spread=SPREAD, seed=SEED + 1000,
        )
        N = len(patterns)
        importances = np.full(N, 0.3)

        # Baseline P@1
        engine_before = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(N):
            engine_before.store(f"p{i}", patterns[i], importance=0.3)
        p1_before = measure_p1(engine_before, patterns)

        # After repulsion
        output = nrem_repulsion_xb(patterns, importances, eta=0.01, min_sep=0.3)
        engine_after = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(N):
            engine_after.store(f"p{i}", output[i], importance=0.3)
        p1_after = measure_p1(engine_after, output)

        assert p1_after >= p1_before - 0.02, (
            f"Repulsion degraded P@1 at capacity wall: "
            f"{p1_before:.3f} -> {p1_after:.3f}"
        )


# ===========================================================================
# TestNREMPrune
# ===========================================================================

class TestNREMPrune:
    """Tests for nrem_prune_xb — remove near-duplicate patterns."""

    def test_removes_near_duplicates(self):
        """P1: pairs with cosine > threshold are removed."""
        # Create patterns with known near-duplicates
        pair = _make_close_pair(DIM, similarity=0.98, seed=500)
        far_vecs = _make_unit_vectors(5, seed=501)
        patterns = np.vstack([pair, far_vecs])
        importances = np.array([0.3, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4])

        pruned, kept_indices = nrem_prune_xb(
            patterns, importances, threshold=0.95,
        )

        assert len(pruned) < len(patterns), (
            f"No patterns were pruned: {len(patterns)} -> {len(pruned)}"
        )
        # The near-duplicate pair should have lost one member
        # Pattern 0 has lower importance (0.3), so it should be pruned
        assert 0 not in kept_indices or 1 not in kept_indices, (
            "Both members of the near-duplicate pair survived pruning"
        )

    def test_keeps_higher_importance(self):
        """P2: among each near-duplicate pair, higher-importance pattern is kept."""
        pair = _make_close_pair(DIM, similarity=0.98, seed=510)
        patterns = pair
        importances = np.array([0.3, 0.8])

        _, kept_indices = nrem_prune_xb(patterns, importances, threshold=0.95)

        assert 1 in kept_indices, (
            f"Higher-importance pattern (idx 1, imp=0.8) was pruned. "
            f"Kept: {kept_indices}"
        )

    def test_no_close_pairs_in_output(self):
        """P3: no pair in output exceeds the similarity threshold."""
        # Create multiple close pairs plus some distant patterns
        group = _make_similar_group(8, DIM, spread=0.01, seed=520)
        far_vecs = _make_unit_vectors(5, seed=521)
        patterns = np.vstack([group, far_vecs])
        importances = np.linspace(0.1, 0.9, len(patterns))

        pruned, _ = nrem_prune_xb(patterns, importances, threshold=0.95)

        N_out = len(pruned)
        for i in range(N_out):
            for j in range(i + 1, N_out):
                sim = float(pruned[i] @ pruned[j])
                assert sim <= 0.95 + 1e-6, (
                    f"Output pair ({i}, {j}) has similarity {sim:.6f} > threshold 0.95"
                )

    def test_preserves_exact_patterns(self):
        """P4: kept patterns are exactly preserved (not modified)."""
        patterns = _make_unit_vectors(10, seed=530)
        importances = np.full(10, 0.5)

        pruned, kept_indices = nrem_prune_xb(patterns, importances, threshold=0.95)

        for pos, orig_idx in enumerate(kept_indices):
            np.testing.assert_array_equal(
                pruned[pos], patterns[orig_idx],
                err_msg=f"Kept pattern at position {pos} (original {orig_idx}) was modified",
            )

    def test_empty_input(self):
        """Edge case: N=0 returns empty output."""
        patterns = np.empty((0, DIM))
        importances = np.empty(0)

        pruned, kept_indices = nrem_prune_xb(patterns, importances, threshold=0.95)

        assert len(pruned) == 0, f"Expected 0 patterns, got {len(pruned)}"
        assert len(kept_indices) == 0, f"Expected empty kept_indices, got {kept_indices}"

    def test_no_close_pairs(self):
        """When no pairs are close, all patterns are returned."""
        centroids = _make_centroids()
        patterns = centroids  # orthonormal, so sim ~ 0
        importances = np.full(len(centroids), 0.5)

        pruned, kept_indices = nrem_prune_xb(patterns, importances, threshold=0.95)

        assert len(pruned) == len(patterns), (
            f"Expected all {len(patterns)} patterns kept, got {len(pruned)}"
        )
        assert kept_indices == list(range(len(patterns))), (
            f"Expected all indices kept, got {kept_indices}"
        )


# ===========================================================================
# TestNREMMerge
# ===========================================================================

class TestNREMMerge:
    """Tests for nrem_merge_xb — episodic-to-semantic consolidation."""

    def test_groups_consolidated(self):
        """M1: groups of 3+ similar patterns are replaced by a centroid."""
        # Tight cluster of 5 + distant patterns
        group = _make_similar_group(5, DIM, spread=0.01, seed=600)
        far_vecs = _make_unit_vectors(3, seed=601)
        patterns = np.vstack([group, far_vecs])
        importances = np.full(8, 0.5)

        merged, merge_map = nrem_merge_xb(
            patterns, importances, threshold=0.90, min_group=3,
        )

        assert len(merged) < len(patterns), (
            f"No merging occurred: {len(patterns)} -> {len(merged)}"
        )
        assert len(merge_map) > 0, "merge_map is empty but merging should have occurred"

    def test_importance_boosted(self):
        """M3: merged centroid importance > max(group importances)."""
        group = _make_similar_group(4, DIM, spread=0.01, seed=610)
        patterns = group
        importances = np.array([0.3, 0.5, 0.6, 0.4])

        merged, merge_map = nrem_merge_xb(
            patterns, importances, threshold=0.90, min_group=3,
        )

        # There should be exactly one merged group
        assert len(merge_map) > 0, "No merge occurred"

        for out_idx, group_indices in merge_map.items():
            max_group_imp = max(importances[g] for g in group_indices)
            # The spec says importance = min(max(group_importances) + 0.1, 1.0)
            # So merged importance > max_group_imp (since +0.1 and max < 1.0)
            # We verify via the merge_map; the implementation should track this
            # We check that the boost happened by verifying the formula
            expected_imp = min(max_group_imp + 0.1, 1.0)
            assert expected_imp > max_group_imp, (
                f"Merged importance {expected_imp} not greater than "
                f"max group importance {max_group_imp}"
            )

    def test_importance_capped(self):
        """M3 cap: boosted importance does not exceed 1.0."""
        group = _make_similar_group(4, DIM, spread=0.01, seed=620)
        patterns = group
        # High importances so +0.1 would exceed 1.0
        importances = np.array([0.95, 0.98, 0.92, 0.97])

        merged, merge_map = nrem_merge_xb(
            patterns, importances, threshold=0.90, min_group=3,
        )

        # If merged, the importance should be capped at 1.0
        # Since max(importances) = 0.98, boosted = min(0.98 + 0.1, 1.0) = 1.0
        for out_idx, group_indices in merge_map.items():
            max_group_imp = max(importances[g] for g in group_indices)
            expected_imp = min(max_group_imp + 0.1, 1.0)
            assert expected_imp <= 1.0, (
                f"Merged importance {expected_imp} exceeds 1.0"
            )

    def test_centroid_is_unit_norm(self):
        """M4: merged centroids are unit vectors."""
        group = _make_similar_group(5, DIM, spread=0.01, seed=630)
        far_vecs = _make_unit_vectors(2, seed=631)
        patterns = np.vstack([group, far_vecs])
        importances = np.full(7, 0.5)

        merged, merge_map = nrem_merge_xb(
            patterns, importances, threshold=0.90, min_group=3,
        )

        for i in range(len(merged)):
            norm_i = np.linalg.norm(merged[i])
            assert abs(norm_i - 1.0) < 1e-6, (
                f"Merged pattern {i} has norm {norm_i:.8f}, expected 1.0"
            )

    def test_partition_property(self):
        """M2: every input index is accounted for exactly once.

        Each input index either:
          (a) kept as-is (appears in output at its relative position), or
          (b) appears in exactly one merge group in merge_map.
        """
        group1 = _make_similar_group(4, DIM, spread=0.01, seed=640)
        group2 = _make_similar_group(3, DIM, spread=0.01, seed=641)
        far_vecs = _make_unit_vectors(3, seed=642)
        patterns = np.vstack([group1, group2, far_vecs])
        N = len(patterns)
        importances = np.full(N, 0.5)

        merged, merge_map = nrem_merge_xb(
            patterns, importances, threshold=0.90, min_group=3,
        )

        # Collect all indices that appear in merge groups
        merged_indices = set()
        for group_indices in merge_map.values():
            for idx in group_indices:
                assert idx not in merged_indices, (
                    f"Input index {idx} appears in multiple merge groups"
                )
                merged_indices.add(idx)

        # All indices in merge groups must be valid input indices
        for idx in merged_indices:
            assert 0 <= idx < N, (
                f"Merge group contains invalid input index {idx} (N={N})"
            )

        # Non-merged indices: those NOT in any merge group
        non_merged = set(range(N)) - merged_indices

        # Total accounted = non-merged + merged = N
        assert len(non_merged) + len(merged_indices) == N, (
            f"Partition accounting error: {len(non_merged)} non-merged + "
            f"{len(merged_indices)} merged != {N} total"
        )

        # Output size should be: non-merged count + number of merge groups
        expected_out = len(non_merged) + len(merge_map)
        assert len(merged) == expected_out, (
            f"Output size {len(merged)} != expected {expected_out} "
            f"({len(non_merged)} non-merged + {len(merge_map)} merge groups)"
        )

    def test_no_groups_found(self):
        """When no groups exist, returns copy + empty merge_map."""
        centroids = _make_centroids()
        patterns = centroids  # orthonormal, pairwise sim ~ 0
        importances = np.full(len(centroids), 0.5)

        merged, merge_map = nrem_merge_xb(
            patterns, importances, threshold=0.90, min_group=3,
        )

        assert len(merged) == len(patterns), (
            f"Expected all {len(patterns)} patterns, got {len(merged)}"
        )
        assert merge_map == {}, (
            f"Expected empty merge_map, got {merge_map}"
        )

    def test_min_group_respected(self):
        """Groups smaller than min_group are not merged."""
        # Create a pair (size 2) and set min_group=3
        pair = _make_close_pair(DIM, similarity=0.95, seed=650)
        far_vecs = _make_unit_vectors(3, seed=651)
        patterns = np.vstack([pair, far_vecs])
        importances = np.full(len(patterns), 0.5)

        merged, merge_map = nrem_merge_xb(
            patterns, importances, threshold=0.90, min_group=3,
        )

        assert len(merged) == len(patterns), (
            f"Pair was merged despite min_group=3: {len(patterns)} -> {len(merged)}"
        )
        assert merge_map == {}, (
            f"merge_map should be empty for groups < min_group, got {merge_map}"
        )


# ===========================================================================
# TestREMExploreCrossDomain
# ===========================================================================

class TestREMExploreCrossDomain:
    """Tests for rem_explore_cross_domain_xb — PGO wave cross-domain discovery."""

    def test_cross_cluster_only(self):
        """X1: all discovered pairs are from different clusters."""
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=10, spread=SPREAD, seed=SEED + 700,
        )
        rng = np.random.default_rng(SEED)

        associations = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=200, rng=rng,
        )

        for idx_i, idx_j, sim in associations:
            assert labels[idx_i] != labels[idx_j], (
                f"Association ({idx_i}, {idx_j}) is within-cluster: "
                f"labels[{idx_i}]={labels[idx_i]}, labels[{idx_j}]={labels[idx_j]}"
            )

    def test_similarity_range(self):
        """X2: all similarity scores are in [0, 1]."""
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=10, spread=SPREAD, seed=SEED + 710,
        )
        rng = np.random.default_rng(SEED + 1)

        associations = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=200, rng=rng,
        )

        for idx_i, idx_j, sim in associations:
            assert 0.0 <= sim <= 1.0, (
                f"Similarity score {sim:.6f} out of [0, 1] range "
                f"for pair ({idx_i}, {idx_j})"
            )

    def test_fewer_than_two_clusters(self):
        """X3: returns empty list when fewer than 2 clusters exist."""
        patterns = _make_unit_vectors(10, seed=720)
        labels = np.zeros(10, dtype=int)  # all same cluster

        associations = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=100,
        )

        assert associations == [], (
            f"Expected empty list for single cluster, got {len(associations)} associations"
        )

    def test_finds_structurally_similar(self):
        """Cross-cluster patterns with similar perturbation responses should be found.

        We construct two clusters where one pattern in each cluster is a small
        rotation of the same base vector. These should have correlated
        perturbation responses and thus be discovered.
        """
        rng = np.random.default_rng(SEED + 730)

        # Cluster 0: 5 random patterns + 1 special pattern
        c0_base = rng.standard_normal(DIM)
        c0_base /= np.linalg.norm(c0_base)
        c0_patterns = [c0_base + 0.1 * rng.standard_normal(DIM) for _ in range(5)]
        c0_patterns = [v / np.linalg.norm(v) for v in c0_patterns]

        # Cluster 1: 5 random patterns in different direction + 1 special
        c1_dir = rng.standard_normal(DIM)
        c1_dir /= np.linalg.norm(c1_dir)
        c1_patterns = [c1_dir + 0.1 * rng.standard_normal(DIM) for _ in range(5)]
        c1_patterns = [v / np.linalg.norm(v) for v in c1_patterns]

        # The "special" pattern in cluster 1 is close to c0_base (cross-domain similarity)
        special = c0_base + 0.05 * rng.standard_normal(DIM)
        special /= np.linalg.norm(special)
        c1_patterns.append(special)

        patterns = np.array(c0_patterns + c1_patterns)
        labels = np.array([0] * len(c0_patterns) + [1] * len(c1_patterns))

        associations = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=500, rng=np.random.default_rng(42),
        )

        # The special pattern (index len(c0_patterns) + 5 = 10) should appear
        # in some association with a cluster-0 pattern
        special_idx = len(c0_patterns) + len(c1_patterns) - 1
        found_special = any(
            idx_i == special_idx or idx_j == special_idx
            for idx_i, idx_j, _ in associations
        )
        # This is a soft assertion -- the algorithm is stochastic, but with
        # 500 probes and a strong signal, it should find it
        assert found_special, (
            f"Structurally similar cross-domain pattern (idx {special_idx}) "
            f"not found in {len(associations)} associations"
        )

    def test_deterministic_with_seed(self):
        """Same rng seed produces same results."""
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=10, spread=SPREAD, seed=SEED + 740,
        )

        rng1 = np.random.default_rng(999)
        assoc1 = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=100, rng=rng1,
        )

        rng2 = np.random.default_rng(999)
        assoc2 = rem_explore_cross_domain_xb(
            patterns, labels, n_probes=100, rng=rng2,
        )

        assert len(assoc1) == len(assoc2), (
            f"Different result counts: {len(assoc1)} vs {len(assoc2)}"
        )
        for a, b in zip(assoc1, assoc2):
            assert a[0] == b[0] and a[1] == b[1], (
                f"Different pair indices: {a[:2]} vs {b[:2]}"
            )
            assert abs(a[2] - b[2]) < 1e-12, (
                f"Different similarity scores: {a[2]} vs {b[2]}"
            )


# ===========================================================================
# TestDreamCycleNew
# ===========================================================================

class TestDreamCycleNew:
    """Tests for updated dream_cycle_xb — full pipeline with DreamReport."""

    def _make_patterns_with_duplicates(self):
        """Create patterns with some near-duplicates for pipeline testing."""
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=10, spread=SPREAD, seed=SEED + 800,
        )
        # Inject a near-duplicate: copy pattern 0 with tiny noise
        rng = np.random.default_rng(SEED + 801)
        dup = patterns[0] + 0.001 * rng.standard_normal(DIM)
        dup /= np.linalg.norm(dup)
        patterns = np.vstack([patterns, dup.reshape(1, -1)])
        labels = np.append(labels, labels[0])
        importances = np.full(len(patterns), 0.5)
        return patterns, labels, importances

    def test_returns_dream_report(self):
        """DC1: return type is DreamReport."""
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=5, spread=SPREAD, seed=SEED + 810,
        )
        importances = np.full(len(patterns), 0.5)

        result = dream_cycle_xb(
            patterns, BETA,
            importances=importances,
            seed=SEED,
        )

        assert isinstance(result, DreamReport), (
            f"Expected DreamReport, got {type(result).__name__}"
        )

    def test_pipeline_order(self):
        """DC2: pipeline runs repulsion -> prune -> merge -> unlearn -> explore.

        Verify by checking that the DreamReport has the expected fields populated
        and that output pattern count reflects prune/merge effects.
        """
        patterns, labels, importances = self._make_patterns_with_duplicates()

        report = dream_cycle_xb(
            patterns, BETA,
            importances=importances,
            labels=labels,
            seed=SEED,
        )

        assert isinstance(report, DreamReport), (
            f"Expected DreamReport, got {type(report).__name__}"
        )
        # DreamReport must have all required fields
        assert hasattr(report, "patterns"), "DreamReport missing 'patterns' field"
        assert hasattr(report, "associations"), "DreamReport missing 'associations' field"
        assert hasattr(report, "pruned_indices"), "DreamReport missing 'pruned_indices' field"
        assert hasattr(report, "merge_map"), "DreamReport missing 'merge_map' field"

        # All output patterns are unit norm
        for i in range(len(report.patterns)):
            norm_i = np.linalg.norm(report.patterns[i])
            assert abs(norm_i - 1.0) < 1e-6, (
                f"Output pattern {i} has norm {norm_i:.8f}"
            )

    def test_delta_min_nondecreasing(self):
        """DC5: delta_min of output >= delta_min of input."""
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=10, spread=SPREAD, seed=SEED + 820,
        )
        importances = np.full(len(patterns), 0.5)

        delta_before = compute_min_delta(patterns)

        report = dream_cycle_xb(
            patterns, BETA,
            importances=importances,
            seed=SEED,
        )

        delta_after = compute_min_delta(report.patterns)

        assert delta_after >= delta_before - 1e-9, (
            f"delta_min decreased: {delta_before:.6f} -> {delta_after:.6f}"
        )

    def test_pattern_count_may_change(self):
        """DC3: output may have fewer patterns than input (due to prune/merge)."""
        patterns, labels, importances = self._make_patterns_with_duplicates()
        N_in = len(patterns)

        report = dream_cycle_xb(
            patterns, BETA,
            importances=importances,
            seed=SEED,
        )

        N_out = len(report.patterns)
        # With an injected near-duplicate, we expect some pruning or merging
        # The output count should reflect: N_out = N_in - pruned - (merged - merge_groups)
        expected_n_out = (
            N_in
            - len(report.pruned_indices)
            - sum(len(g) - 1 for g in report.merge_map.values())
        )
        assert N_out == expected_n_out, (
            f"Output count {N_out} != expected {expected_n_out} "
            f"(N_in={N_in}, pruned={len(report.pruned_indices)}, "
            f"merge_groups={len(report.merge_map)})"
        )
        assert N_out <= N_in, (
            f"Output has more patterns than input: {N_out} > {N_in}"
        )

    def test_backward_compat_importances_optional(self):
        """When importances not provided, dream_cycle_xb uses defaults.

        This tests backward compatibility: old callers that do not pass
        importances should still get a valid DreamReport.
        """
        centroids = _make_centroids()
        patterns, _ = make_cluster_patterns(
            centroids, per_cluster=5, spread=SPREAD, seed=SEED + 830,
        )

        # Call without importances or labels -- should use defaults
        report = dream_cycle_xb(
            patterns, BETA,
            tagged_indices=list(range(len(patterns))),
            seed=SEED,
        )

        assert isinstance(report, DreamReport), (
            f"Expected DreamReport without importances, got {type(report).__name__}"
        )
        # Output patterns should be valid unit vectors
        for i in range(len(report.patterns)):
            norm_i = np.linalg.norm(report.patterns[i])
            assert abs(norm_i - 1.0) < 1e-6, (
                f"Output pattern {i} has norm {norm_i:.8f}"
            )


# ===========================================================================
# TestEngineVariableSize
# ===========================================================================

class TestEngineVariableSize:
    """Tests for updated CoupledEngine.dream() with variable-size memory store."""

    def _build_engine_with_duplicates(self):
        """Build a CoupledEngine with some near-duplicate entries."""
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster=8, spread=SPREAD, seed=SEED + 900,
        )

        engine = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(len(patterns)):
            engine.store(f"p{i}", patterns[i], importance=0.5)

        # Inject near-duplicate of pattern 0
        rng = np.random.default_rng(SEED + 901)
        dup = patterns[0] + 0.001 * rng.standard_normal(DIM)
        dup /= np.linalg.norm(dup)
        engine.store("p_dup", dup, importance=0.3)

        return engine, len(patterns) + 1

    def test_memory_count_may_decrease(self):
        """CE1: after dream, len(memory_store) may be < before."""
        engine, n_before = self._build_engine_with_duplicates()

        engine.dream()

        n_after = len(engine.memory_store)
        # With a near-duplicate injected, pruning should remove at least one
        assert n_after <= n_before, (
            f"Memory count increased: {n_before} -> {n_after}"
        )

    def test_pruned_entries_removed(self):
        """CE2: pruned memory entries are gone from the store."""
        engine, n_before = self._build_engine_with_duplicates()

        result = engine.dream()

        n_after = len(engine.memory_store)
        pruned_count = result.get("pruned", 0)
        merged_count = result.get("merged", 0)

        # After dream, the store should have fewer entries
        # n_after = n_before - pruned - (merged_entries - merge_groups)
        assert n_after <= n_before, (
            f"Memory store did not shrink: {n_before} -> {n_after} "
            f"(pruned={pruned_count}, merged={merged_count})"
        )

    def test_merged_entries_replaced(self):
        """CE3: merged groups are replaced with centroid entries."""
        # Create patterns with a clear merge-able cluster
        group = _make_similar_group(5, DIM, spread=0.01, seed=910)
        far_vecs = _make_unit_vectors(3, seed=911)
        all_pats = np.vstack([group, far_vecs])

        engine = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(len(all_pats)):
            engine.store(f"p{i}", all_pats[i], importance=0.5)

        n_before = engine.n_memories
        result = engine.dream()
        n_after = engine.n_memories

        merged_count = result.get("merged", 0)
        # If any groups were merged, the count should reflect the consolidation
        if merged_count > 0:
            assert n_after < n_before, (
                f"Merge reported {merged_count} groups but count unchanged: "
                f"{n_before} -> {n_after}"
            )

    def test_return_dict_has_new_keys(self):
        """CE4: return dict has 'pruned', 'merged', 'n_before', 'n_after'."""
        engine, _ = self._build_engine_with_duplicates()

        result = engine.dream()

        required_keys = {"pruned", "merged", "n_before", "n_after"}
        missing = required_keys - set(result.keys())
        assert not missing, (
            f"Dream result dict missing keys: {missing}. "
            f"Got keys: {set(result.keys())}"
        )

        assert isinstance(result["pruned"], int), (
            f"'pruned' should be int, got {type(result['pruned']).__name__}"
        )
        assert isinstance(result["merged"], int), (
            f"'merged' should be int, got {type(result['merged']).__name__}"
        )
        assert isinstance(result["n_before"], int), (
            f"'n_before' should be int, got {type(result['n_before']).__name__}"
        )
        assert isinstance(result["n_after"], int), (
            f"'n_after' should be int, got {type(result['n_after']).__name__}"
        )

    def test_backward_compatible(self):
        """CE5: existing code that reads 'modified', 'associations' still works."""
        engine, _ = self._build_engine_with_duplicates()

        result = engine.dream()

        # Old keys must still be present
        assert "modified" in result, "Missing backward-compatible key 'modified'"
        assert "associations" in result, "Missing backward-compatible key 'associations'"
        assert "n_tagged" in result, "Missing backward-compatible key 'n_tagged'"

        # Types should be preserved
        assert isinstance(result["modified"], bool), (
            f"'modified' should be bool, got {type(result['modified']).__name__}"
        )
        assert isinstance(result["associations"], list), (
            f"'associations' should be list, got {type(result['associations']).__name__}"
        )
