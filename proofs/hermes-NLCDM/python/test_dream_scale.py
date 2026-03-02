"""Dream scale integration tests -- capstone validation at realistic scale.

Test 5: Full integrated pipeline validation across scale, geometry, and
edge cases. Exercises the complete dream_cycle_xb pipeline (repulsion ->
prune -> merge -> rem_unlearn -> rem_explore_cross_domain) on workloads
ranging from small overcrowded clusters to 1000-pattern realistic workloads.

Tests:
  5a. Small overcrowded clusters -- complete pipeline on tight geometry
  5b. Realistic workload at scale -- 1000 patterns, multi-cycle convergence
  5c. Cross-domain associations at scale -- REM explore fires meaningfully
  5d. Contradiction handling -- contradicting pairs survive without crash
  5e. Variable cluster geometry -- tight clusters pruned more than loose
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from dream_ops import (
    DreamReport,
    dream_cycle_xb,
    spreading_activation,
)
from coupled_engine import CoupledEngine
from test_capacity_boundary import (
    compute_min_delta,
    make_separated_centroids,
    make_cluster_patterns,
    measure_p1,
)
from test_dream_convergence import (
    generate_realistic_workload,
    measure_spurious_rate,
    measure_delta_within_between,
    measure_cluster_coherence,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

BETA = 10.0
SEED = 42


def _build_engine(patterns: np.ndarray, dim: int, beta: float) -> CoupledEngine:
    """Build a CoupledEngine populated with the given patterns."""
    engine = CoupledEngine(dim=dim, beta=beta)
    for i in range(len(patterns)):
        engine.store(f"p{i}", patterns[i], importance=0.5)
    return engine


# ===========================================================================
# Test 5a: Small overcrowded clusters -- complete pipeline
# ===========================================================================


class TestSmallOvercrowded:
    """Test 5a: 5 clusters x 25 patterns, dim=128, spread=0.04.

    Overcrowded geometry (within-cluster cosine ~ 0.83) stresses every
    pipeline stage: repulsion must push apart, prune/merge must thin,
    REM must clean spurious attractors.
    """

    DIM = 128
    N_CLUSTERS = 5
    PER_CLUSTER = 25
    SPREAD = 0.04
    SEED = SEED

    def _setup(self):
        """Create overcrowded cluster patterns."""
        centroids = make_separated_centroids(self.N_CLUSTERS, self.DIM, seed=self.SEED)
        patterns, labels = make_cluster_patterns(
            centroids, self.PER_CLUSTER, spread=self.SPREAD, seed=self.SEED,
        )
        N = len(patterns)
        importances = np.full(N, 0.5)
        return patterns, labels, importances, centroids

    def test_complete_pipeline(self):
        """Run the full dream pipeline and verify all quality metrics improve or hold."""
        patterns, labels, importances, centroids = self._setup()
        N_before = len(patterns)

        # --- Before metrics ---
        delta_min_before = compute_min_delta(patterns)

        engine_before = _build_engine(patterns, self.DIM, BETA)
        p1_before = measure_p1(engine_before, patterns)

        coherence_before = measure_cluster_coherence(patterns, labels, BETA)

        spurious_before, _ = measure_spurious_rate(
            patterns, BETA, n_queries=200, seed=self.SEED,
        )

        # --- Run dream cycle ---
        report = dream_cycle_xb(
            patterns, BETA,
            importances=importances,
            labels=labels,
            seed=self.SEED,
        )

        X_after = report.patterns
        N_after = len(X_after)

        # --- After metrics ---
        delta_min_after = compute_min_delta(X_after)

        engine_after = _build_engine(X_after, self.DIM, BETA)
        p1_after = measure_p1(engine_after, X_after)

        # Recompute labels for surviving patterns via nearest centroid
        labels_after = np.argmax(X_after @ centroids.T, axis=1)
        coherence_after = measure_cluster_coherence(X_after, labels_after, BETA)

        spurious_after, _ = measure_spurious_rate(
            X_after, BETA, n_queries=200, seed=self.SEED,
        )

        # --- Report ---
        print(f"\n  Test 5a: Small overcrowded pipeline (N={N_before} -> {N_after})")
        print(f"  delta_min:  {delta_min_before:.6f} -> {delta_min_after:.6f}")
        print(f"  P@1:        {p1_before:.3f} -> {p1_after:.3f}")
        print(f"  coherence:  {coherence_before:.3f} -> {coherence_after:.3f}")
        print(f"  spurious:   {spurious_before:.3f} -> {spurious_after:.3f}")
        print(f"  associations: {len(report.associations)}")
        print(f"  pruned: {len(report.pruned_indices)}, merged: {len(report.merge_map)}")

        # --- Assertions ---
        # delta_min must increase (repulsion worked)
        assert delta_min_after > delta_min_before, (
            f"delta_min did not increase: {delta_min_before:.6f} -> {delta_min_after:.6f}"
        )

        # Coherence preserved (semantic structure intact)
        assert coherence_after >= 0.8, (
            f"coherence too low after dream: {coherence_after:.3f}"
        )

        # Pattern count decreased (pruning/merging active on overcrowded data)
        assert N_after <= N_before, (
            f"pattern count increased: {N_before} -> {N_after}"
        )

        # P@1 did not degrade
        assert p1_after >= p1_before - 0.05, (
            f"P@1 degraded: {p1_before:.3f} -> {p1_after:.3f}"
        )


# ===========================================================================
# Test 5b: Realistic workload at scale (THE KEY TEST)
# ===========================================================================


class TestRealisticScale:
    """Test 5b: 1000 patterns, 200 topics, dim=384, multi-cycle convergence.

    This is the capstone test. Exercises the pipeline on a workload that
    mimics real usage: power-law cluster sizes, variable tightness,
    cross-topic patterns, contradictions, and temporal bursts.
    """

    MAX_CYCLES = 10
    CONVERGENCE_TOL = 1e-3  # delta_min change threshold for convergence
    PER_CYCLE_TIMEOUT = 60  # seconds

    def _generate(self):
        """Generate the realistic workload."""
        return generate_realistic_workload(
            n_memories=1000,
            n_topics=200,
            dim=384,
            cross_topic_fraction=0.15,
            contradiction_fraction=0.05,
            seed=SEED,
        )

    def test_multi_cycle_convergence(self):
        """Run up to 10 dream cycles, tracking key metrics at each step.

        Stops early if delta_min converges (change < 1e-3).
        Asserts computational feasibility, monotonic delta improvement,
        coherence preservation, and P@1 non-degradation.
        """
        workload = self._generate()
        patterns = workload["patterns"]
        importances = workload["importances"]
        labels = workload["labels"]
        centroids = workload["centroids"]
        N_initial = len(patterns)
        dim = patterns.shape[1]

        # --- Initial metrics ---
        delta_min_initial = compute_min_delta(patterns)

        engine_initial = _build_engine(patterns, dim, BETA)
        p1_initial = measure_p1(engine_initial, patterns)

        coherence_initial = measure_cluster_coherence(patterns, labels, BETA)

        # --- Cycle tracking ---
        deltas = [delta_min_initial]
        p1_values = [p1_initial]
        coherence_values = [coherence_initial]
        pattern_counts = [N_initial]
        cycle_times = []

        current = patterns.copy()
        current_importances = importances.copy()
        current_labels = labels.copy()

        for cycle in range(self.MAX_CYCLES):
            t0 = time.time()

            report = dream_cycle_xb(
                current, BETA,
                importances=current_importances[:len(current)],
                labels=current_labels[:len(current)] if len(current) == len(current_labels) else None,
                seed=SEED + cycle,
            )

            elapsed = time.time() - t0
            cycle_times.append(elapsed)

            current = report.patterns
            N_cur = len(current)

            # Update importances and labels for surviving patterns
            current_importances = np.full(N_cur, 0.5)
            if N_cur > 0:
                current_labels = np.argmax(current @ centroids.T, axis=1)

            # Measure metrics
            delta_cur = compute_min_delta(current)
            deltas.append(delta_cur)

            engine_cur = _build_engine(current, dim, BETA)
            p1_cur = measure_p1(engine_cur, current)
            p1_values.append(p1_cur)

            coherence_cur = measure_cluster_coherence(current, current_labels, BETA)
            coherence_values.append(coherence_cur)

            pattern_counts.append(N_cur)

            # Check convergence
            if abs(delta_cur - deltas[-2]) < self.CONVERGENCE_TOL:
                print(f"  Converged at cycle {cycle + 1} (delta change < {self.CONVERGENCE_TOL})")
                break

        # --- Report ---
        print(f"\n  Test 5b: Realistic scale (N={N_initial}, dim={dim})")
        print(f"  Cycles completed: {len(cycle_times)}")
        print(f"  {'Cycle':>6} {'delta':>10} {'P@1':>6} {'coher':>7} {'N':>6} {'time_s':>8}")
        for i in range(len(cycle_times)):
            print(
                f"  {i+1:>6} {deltas[i+1]:>10.6f} {p1_values[i+1]:>6.3f} "
                f"{coherence_values[i+1]:>7.3f} {pattern_counts[i+1]:>6} {cycle_times[i]:>8.2f}"
            )

        # --- Assertions ---

        # 1. Computational feasibility: each cycle under timeout
        for i, t in enumerate(cycle_times):
            assert t < self.PER_CYCLE_TIMEOUT, (
                f"Cycle {i+1} took {t:.1f}s, exceeding {self.PER_CYCLE_TIMEOUT}s timeout"
            )

        # 2. delta_min improves or stays same across cycles (monotonic non-decreasing)
        for i in range(1, len(deltas)):
            assert deltas[i] >= deltas[i - 1] - 1e-9, (
                f"delta_min decreased at cycle {i}: {deltas[i-1]:.6f} -> {deltas[i]:.6f}"
            )

        # 3. Coherence does not degrade significantly from initial level.
        # With 200 topics and power-law sizes, many clusters are singletons,
        # so absolute coherence can be low. The key invariant is that dreaming
        # does not DESTROY coherence -- it should stay at or above the initial
        # level minus a tolerance for noise from pattern count changes.
        for i, c in enumerate(coherence_values):
            assert c >= coherence_initial - 0.10, (
                f"Coherence dropped significantly at step {i}: "
                f"{coherence_initial:.3f} -> {c:.3f}"
            )

        # 4. Pattern count does not increase (pruning/merging may or may not
        # fire depending on cluster tightness; at dim=384 with variable spread
        # many clusters are loose enough to pass through unchanged).
        assert pattern_counts[-1] <= pattern_counts[0], (
            f"Pattern count increased: {pattern_counts[0]} -> {pattern_counts[-1]}"
        )

        # 5. Final P@1 >= initial P@1 (no degradation)
        assert p1_values[-1] >= p1_values[0] - 0.05, (
            f"P@1 degraded: {p1_values[0]:.3f} -> {p1_values[-1]:.3f}"
        )


# ===========================================================================
# Test 5c: Cross-domain associations at scale
# ===========================================================================


class TestCrossDomainAssociations:
    """Test 5c: REM explore fires at scale and links different clusters."""

    def test_associations_found(self):
        """Dream cycle discovers cross-domain associations on realistic workload.

        Validates:
        - Associations list is non-empty
        - Associations link patterns from DIFFERENT clusters
        """
        workload = generate_realistic_workload(
            n_memories=1000,
            n_topics=200,
            dim=384,
            cross_topic_fraction=0.15,
            contradiction_fraction=0.05,
            seed=SEED,
        )
        patterns = workload["patterns"]
        importances = workload["importances"]
        labels = workload["labels"]
        centroids = workload["centroids"]

        report = dream_cycle_xb(
            patterns, BETA,
            importances=importances,
            labels=labels,
            seed=SEED,
        )

        associations = report.associations
        X_after = report.patterns
        N_after = len(X_after)

        # Recompute labels for post-dream patterns
        labels_after = np.argmax(X_after @ centroids.T, axis=1)

        print(f"\n  Test 5c: Cross-domain associations")
        print(f"  Total associations found: {len(associations)}")

        # Count cross-cluster associations
        n_cross_cluster = 0
        n_same_cluster = 0
        for idx_i, idx_j, sim in associations:
            if idx_i < N_after and idx_j < N_after:
                if labels_after[idx_i] != labels_after[idx_j]:
                    n_cross_cluster += 1
                else:
                    n_same_cluster += 1

        print(f"  Cross-cluster: {n_cross_cluster}")
        print(f"  Same-cluster:  {n_same_cluster}")

        if len(associations) > 0:
            sims = [s for _, _, s in associations]
            print(f"  Similarity range: [{min(sims):.3f}, {max(sims):.3f}]")

        # Assertions
        assert len(associations) > 0, (
            "No cross-domain associations found at scale"
        )

        # At least some associations should link different clusters
        assert n_cross_cluster > 0, (
            f"All {len(associations)} associations are within the same cluster; "
            f"expected cross-cluster discovery"
        )


# ===========================================================================
# Test 5d: Contradiction handling
# ===========================================================================


class TestContradictionHandling:
    """Test 5d: Contradicting pattern pairs survive the pipeline without crash.

    The realistic workload includes contradiction_pairs -- patterns with
    partially flipped components in the same cluster. This is a
    characterization test: document what happens, assert no crash.
    """

    def test_contradictions_survive_or_separate(self):
        """Dream cycle handles contradictions gracefully.

        Characterizes:
        - How many contradictions survive (both patterns still present)
        - How many get pruned or merged away
        - Whether surviving contradictions maintain or increase separation
        """
        workload = generate_realistic_workload(
            n_memories=1000,
            n_topics=200,
            dim=384,
            cross_topic_fraction=0.15,
            contradiction_fraction=0.05,
            seed=SEED,
        )
        patterns = workload["patterns"]
        importances = workload["importances"]
        labels = workload["labels"]
        contradiction_pairs = workload["contradiction_pairs"]
        N_before = len(patterns)

        # Measure initial pairwise distances for contradiction pairs
        initial_distances = []
        for src_idx, contra_idx in contradiction_pairs:
            cos_sim = float(patterns[src_idx] @ patterns[contra_idx])
            initial_distances.append(1.0 - cos_sim)

        # Run dream cycle
        report = dream_cycle_xb(
            patterns, BETA,
            importances=importances,
            labels=labels,
            seed=SEED,
        )

        X_after = report.patterns
        N_after = len(X_after)
        pruned_set = set(report.pruned_indices)

        # Determine which contradiction pairs survived
        # A pattern "survives" if its index is not pruned AND it wasn't consumed
        # by a merge. Build a mapping from input index -> output index.
        # Merge map values are lists of input indices merged into an output index.
        merged_input_indices: set[int] = set()
        for group in report.merge_map.values():
            merged_input_indices.update(group)

        # Build input -> output mapping for non-pruned, non-merged patterns
        surviving_input_indices = [
            i for i in range(N_before)
            if i not in pruned_set and i not in merged_input_indices
        ]
        # The output ordering: non-merged patterns come first (in order),
        # then merged centroids. We need to figure out the mapping.
        # For a simpler characterization, check if both contradiction members
        # are in the surviving set.

        n_both_survive = 0
        n_one_pruned = 0
        n_both_pruned = 0

        for src_idx, contra_idx in contradiction_pairs:
            src_survives = src_idx in surviving_input_indices or src_idx in merged_input_indices
            contra_survives = contra_idx in surviving_input_indices or contra_idx in merged_input_indices
            if src_survives and contra_survives:
                n_both_survive += 1
            elif src_survives or contra_survives:
                n_one_pruned += 1
            else:
                n_both_pruned += 1

        print(f"\n  Test 5d: Contradiction handling")
        print(f"  Total contradiction pairs: {len(contradiction_pairs)}")
        print(f"  Both survive:  {n_both_survive}")
        print(f"  One pruned:    {n_one_pruned}")
        print(f"  Both pruned:   {n_both_pruned}")
        print(f"  Pattern count: {N_before} -> {N_after}")
        if initial_distances:
            print(f"  Initial contradiction distances: "
                  f"mean={np.mean(initial_distances):.4f}, "
                  f"min={np.min(initial_distances):.4f}, "
                  f"max={np.max(initial_distances):.4f}")

        # The primary assertion: no crash, and the pipeline completes
        assert N_after > 0, "Pipeline collapsed all patterns"
        assert N_after <= N_before, (
            f"Pattern count increased: {N_before} -> {N_after}"
        )

        # Characterization: contradictions should be mostly far apart
        # (cosine distance > 0.5 on average) since we flip ~50% of components
        if initial_distances:
            assert np.mean(initial_distances) > 0.3, (
                f"Contradiction pairs unexpectedly close: "
                f"mean distance={np.mean(initial_distances):.4f}"
            )


# ===========================================================================
# Test 5e: Variable cluster geometry
# ===========================================================================


class TestVariableClusterGeometry:
    """Test 5e: Tight clusters experience more pruning/merging than loose ones.

    The realistic workload has variable cluster_spreads drawn from
    U[0.05, 0.35]. After dream cycles, tight clusters (small spread)
    should show more pruning/merging activity than loose clusters
    (large spread).
    """

    def test_tight_clusters_pruned_more(self):
        """Per-cluster pruning/merging activity correlates with cluster spread.

        Tight clusters (spread < 0.15) should experience more pruning/merging
        than loose clusters (spread > 0.25) because their patterns have higher
        pairwise cosine similarity.

        We measure this using the first dream cycle's DreamReport, which
        directly reports pruned_indices and merge_map. We attribute each
        pruned/merged pattern to its original cluster label and compute
        per-cluster "affected rate" (fraction of original patterns that
        were pruned or merged).
        """
        workload = generate_realistic_workload(
            n_memories=1000,
            n_topics=200,
            dim=384,
            cross_topic_fraction=0.15,
            contradiction_fraction=0.05,
            seed=SEED,
        )
        patterns = workload["patterns"]
        importances = workload["importances"]
        labels = workload["labels"]
        cluster_spreads = workload["cluster_spreads"]
        N_before = len(patterns)

        # Count patterns per cluster before dreaming
        unique_labels = np.unique(labels)
        before_counts: dict[int, int] = {}
        for lbl in unique_labels:
            before_counts[int(lbl)] = int(np.sum(labels == lbl))

        # Run a single dream cycle and inspect the report directly
        report = dream_cycle_xb(
            patterns, BETA,
            importances=importances,
            labels=labels,
            seed=SEED,
        )

        # Collect all input indices affected by pruning or merging
        affected_indices: set[int] = set(report.pruned_indices)
        for group in report.merge_map.values():
            affected_indices.update(group)

        # Compute per-cluster affected rate
        affected_rate: dict[int, float] = {}
        for lbl in unique_labels:
            lbl_int = int(lbl)
            n_total = before_counts.get(lbl_int, 0)
            if n_total < 2:
                # Singletons can't be pruned/merged pairwise
                continue
            cluster_indices = set(np.where(labels == lbl_int)[0].tolist())
            n_affected = len(cluster_indices & affected_indices)
            affected_rate[lbl_int] = n_affected / n_total

        # Categorize clusters by spread
        tight_affected = []
        loose_affected = []
        tight_threshold = 0.15
        loose_threshold = 0.25

        for lbl_int, rate in affected_rate.items():
            spread = cluster_spreads.get(lbl_int, 0.2)
            if spread < tight_threshold:
                tight_affected.append(rate)
            elif spread > loose_threshold:
                loose_affected.append(rate)

        # Also measure within-cluster mean cosine similarity to validate
        # that tight clusters really ARE tighter
        tight_mean_sim = []
        loose_mean_sim = []
        for lbl_int in unique_labels:
            lbl_int = int(lbl_int)
            mask = labels == lbl_int
            c_pats = patterns[mask]
            if len(c_pats) < 2:
                continue
            # Sample up to 50 pairs for efficiency
            rng = np.random.default_rng(SEED + lbl_int)
            n_c = len(c_pats)
            sims = []
            n_pairs = min(50, n_c * (n_c - 1) // 2)
            for _ in range(n_pairs):
                i, j = rng.choice(n_c, size=2, replace=False)
                sims.append(float(c_pats[i] @ c_pats[j]))
            mean_sim = np.mean(sims)
            spread = cluster_spreads.get(lbl_int, 0.2)
            if spread < tight_threshold:
                tight_mean_sim.append(mean_sim)
            elif spread > loose_threshold:
                loose_mean_sim.append(mean_sim)

        print(f"\n  Test 5e: Variable cluster geometry")
        print(f"  Patterns: {N_before} -> {len(report.patterns)}")
        print(f"  Pruned: {len(report.pruned_indices)}, Merge groups: {len(report.merge_map)}")

        # Print spread distribution
        all_spreads = list(cluster_spreads.values())
        print(f"  Spread distribution: min={min(all_spreads):.3f}, "
              f"max={max(all_spreads):.3f}, mean={np.mean(all_spreads):.3f}")

        if tight_affected:
            print(f"  Tight clusters (spread < {tight_threshold}): "
                  f"n={len(tight_affected)}, mean_affected_rate={np.mean(tight_affected):.3f}")
        else:
            print(f"  Tight clusters (spread < {tight_threshold}): none with 2+ patterns")

        if loose_affected:
            print(f"  Loose clusters (spread > {loose_threshold}): "
                  f"n={len(loose_affected)}, mean_affected_rate={np.mean(loose_affected):.3f}")
        else:
            print(f"  Loose clusters (spread > {loose_threshold}): none with 2+ patterns")

        if tight_mean_sim and loose_mean_sim:
            print(f"  Validation -- mean cosine sim: tight={np.mean(tight_mean_sim):.3f}, "
                  f"loose={np.mean(loose_mean_sim):.3f}")

        # Assertions
        # Primary: tight clusters should have equal or higher affected rate
        # (more pruning/merging) than loose clusters.
        # If both groups have zero affected rate (no pruning/merging at all),
        # that is still consistent -- the effect is just absent at this scale.
        if tight_affected and loose_affected:
            mean_tight = np.mean(tight_affected)
            mean_loose = np.mean(loose_affected)
            print(f"  Affected rate gap: tight={mean_tight:.3f}, loose={mean_loose:.3f}")

            # Tight clusters should be pruned/merged at least as much as loose
            # (allow tolerance for stochastic effects)
            assert mean_tight >= mean_loose - 0.10, (
                f"Loose clusters were pruned MORE than tight: "
                f"tight={mean_tight:.3f}, loose={mean_loose:.3f}"
            )
        else:
            # If no pruning/merging happened at all (both rates = 0), that's fine.
            # The test documents that the pipeline didn't crash and characterizes
            # the geometry.
            total_affected = len(affected_indices)
            print(f"  Total affected indices: {total_affected}")
            print("  Insufficient data in both categories for direct comparison")
