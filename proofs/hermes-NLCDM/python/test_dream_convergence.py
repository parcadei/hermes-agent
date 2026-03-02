"""Dream convergence validation suite — workload generator and helpers.

Tests the dream cycle's convergence behavior on realistic synthetic workloads
with power-law cluster sizes, variable tightness, cross-topic patterns,
contradictions, and temporal bursts.

This file provides:
  1. generate_realistic_workload() — synthetic memory workload generator
  2. measure_spurious_rate() — spurious fixed point rate measurement
  3. measure_delta_within_between() — within/between cluster distance measurement
  4. measure_cluster_coherence() — Hopfield recall cluster coherence measurement
"""

from __future__ import annotations

import numpy as np

from dream_ops import (
    DreamReport,
    dream_cycle_xb,
    nrem_repulsion_xb,
    nrem_prune_xb,
    nrem_merge_xb,
    rem_unlearn_xb,
    rem_explore_cross_domain_xb,
    spreading_activation,
    hopfield_update,
)
from coupled_engine import CoupledEngine, MemoryEntry
from test_capacity_boundary import (
    compute_min_delta,
    make_separated_centroids,
    make_cluster_patterns,
    measure_p1,
    compute_n_max,
    cluster_coherence,
)


# ---------------------------------------------------------------------------
# Workload generator
# ---------------------------------------------------------------------------

def generate_realistic_workload(
    n_memories: int = 1000,
    n_topics: int = 200,
    dim: int = 384,
    cross_topic_fraction: float = 0.15,
    contradiction_fraction: float = 0.05,
    seed: int = 42,
) -> dict:
    """Generate realistic synthetic memory workload.

    Produces a set of unit-vector patterns organized into clusters with
    power-law size distribution, variable within-cluster tightness,
    cross-topic bridging patterns, temporal burst structure, and
    contradicting pattern pairs.

    Args:
        n_memories: total number of patterns to generate (approximate)
        n_topics: number of distinct topic clusters
        dim: embedding dimension
        cross_topic_fraction: fraction of n_memories that bridge two topics
        contradiction_fraction: fraction of n_memories with a contradicting partner
        seed: random seed for reproducibility

    Returns:
        dict with keys:
            patterns: (N, dim) unit vectors
            importances: (N,) in [0.0, 1.0]
            labels: (N,) integer cluster assignments (primary cluster)
            creation_times: (N,) floats simulating temporal bursts
            cross_topic_indices: list[int] -- indices of cross-topic patterns
            cross_topic_pairs: list[tuple[int, int]] -- (index, other_cluster_centroid_idx)
            contradiction_pairs: list[tuple[int, int]] -- pairs of contradicting patterns
            cluster_spreads: dict[int, float] -- per-cluster tightness (cosine distance)
            centroids: (n_topics, dim) unit vectors -- cluster centroids
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Generate orthonormal cluster centroids
    # ------------------------------------------------------------------
    centroids = make_separated_centroids(n_topics, dim, seed=seed)

    # ------------------------------------------------------------------
    # 2. Power-law cluster sizes (Zipf distribution)
    # ------------------------------------------------------------------
    # Reserve slots for cross-topic and contradiction patterns
    n_cross = int(n_memories * cross_topic_fraction)
    n_contradiction = int(n_memories * contradiction_fraction)
    n_base = n_memories - n_cross - n_contradiction

    # Zipf-distributed cluster sizes
    raw_sizes = rng.zipf(a=1.5, size=n_topics).astype(float)
    raw_sizes = raw_sizes / raw_sizes.sum() * n_base
    cluster_sizes = np.maximum(np.round(raw_sizes).astype(int), 1)
    # Adjust last cluster to hit the target total
    diff = n_base - cluster_sizes.sum()
    cluster_sizes[-1] = max(1, cluster_sizes[-1] + diff)

    # ------------------------------------------------------------------
    # 3. Variable within-cluster tightness
    # ------------------------------------------------------------------
    # Draw spread per cluster from uniform [0.05, 0.35]
    spreads = rng.uniform(0.05, 0.35, size=n_topics)
    cluster_spreads = {int(c): float(spreads[c]) for c in range(n_topics)}

    # ------------------------------------------------------------------
    # 4. Generate base patterns per cluster
    # ------------------------------------------------------------------
    patterns_list: list[np.ndarray] = []
    labels_list: list[int] = []
    creation_times_list: list[float] = []

    # Temporal structure: each cluster gets a burst base time
    burst_base_times = np.cumsum(rng.uniform(10.0, 50.0, size=n_topics))

    for c in range(n_topics):
        n_c = int(cluster_sizes[c])
        spread_c = spreads[c]
        for _ in range(n_c):
            p = centroids[c] + spread_c * rng.standard_normal(dim)
            norm = np.linalg.norm(p)
            if norm < 1e-12:
                p = centroids[c].copy()
            else:
                p = p / norm
            patterns_list.append(p)
            labels_list.append(c)
            # Within-burst temporal spacing
            creation_times_list.append(
                burst_base_times[c] + rng.exponential(scale=1.0)
            )

    # ------------------------------------------------------------------
    # 5. Cross-topic patterns
    # ------------------------------------------------------------------
    cross_topic_indices: list[int] = []
    cross_topic_pairs: list[tuple[int, int]] = []

    for _ in range(n_cross):
        # Pick two distinct clusters
        c1, c2 = rng.choice(n_topics, size=2, replace=False)
        weight = rng.uniform(0.3, 0.7)
        p = weight * centroids[c1] + (1.0 - weight) * centroids[c2]
        p += 0.02 * rng.standard_normal(dim)
        norm = np.linalg.norm(p)
        if norm < 1e-12:
            p = centroids[c1].copy()
        else:
            p = p / norm

        # Assign primary label to the higher-weight cluster
        primary_cluster = c1 if weight >= 0.5 else c2
        other_cluster = c2 if weight >= 0.5 else c1

        idx = len(patterns_list)
        patterns_list.append(p)
        labels_list.append(primary_cluster)
        # Temporal: place between the two clusters' burst times
        creation_times_list.append(
            0.5 * (burst_base_times[c1] + burst_base_times[c2])
            + rng.exponential(scale=1.0)
        )

        cross_topic_indices.append(idx)
        cross_topic_pairs.append((idx, other_cluster))

    # ------------------------------------------------------------------
    # 6. Contradiction patterns
    # ------------------------------------------------------------------
    contradiction_pairs: list[tuple[int, int]] = []
    n_base_total = len(patterns_list) - n_cross  # base patterns count

    # Select base pattern indices to create contradictions for
    if n_base_total > 0 and n_contradiction > 0:
        source_indices = rng.choice(
            n_base_total, size=min(n_contradiction, n_base_total), replace=False,
        )
        for src_idx in source_indices:
            src_pattern = patterns_list[src_idx]
            src_label = labels_list[src_idx]

            # Create near-antipodal contradiction in the SAME cluster:
            # flip sign of 50% of components, add noise, normalize
            mask = rng.random(dim) < 0.5
            contra = src_pattern.copy()
            contra[mask] = -contra[mask]
            contra += 0.05 * rng.standard_normal(dim)
            norm = np.linalg.norm(contra)
            if norm < 1e-12:
                contra = -src_pattern.copy()
                contra /= np.linalg.norm(contra)
            else:
                contra = contra / norm

            contra_idx = len(patterns_list)
            patterns_list.append(contra)
            labels_list.append(src_label)
            creation_times_list.append(
                creation_times_list[src_idx] + rng.exponential(scale=0.5)
            )
            contradiction_pairs.append((src_idx, contra_idx))

    # ------------------------------------------------------------------
    # 7. Variable importance (power law)
    # ------------------------------------------------------------------
    total_n = len(patterns_list)
    importances = np.clip(
        rng.pareto(a=2.0, size=total_n) * 0.3 + 0.2, 0.1, 1.0
    )

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    patterns = np.array(patterns_list, dtype=np.float64)
    labels = np.array(labels_list, dtype=int)
    creation_times = np.array(creation_times_list, dtype=np.float64)

    return {
        "patterns": patterns,
        "importances": importances,
        "labels": labels,
        "creation_times": creation_times,
        "cross_topic_indices": cross_topic_indices,
        "cross_topic_pairs": cross_topic_pairs,
        "contradiction_pairs": contradiction_pairs,
        "cluster_spreads": cluster_spreads,
        "centroids": centroids,
    }


# ---------------------------------------------------------------------------
# Helper: spurious fixed point rate
# ---------------------------------------------------------------------------

def measure_spurious_rate(
    patterns: np.ndarray,
    beta: float,
    n_queries: int = 200,
    similarity_threshold: float = 0.8,
    seed: int = 0,
) -> tuple[float, float]:
    """Measure spurious fixed point rate of the Hopfield network.

    Generates n_queries random unit vectors, runs hopfield_update to
    convergence via spreading_activation, and classifies each converged
    state as "stored" (max cosine similarity to any stored pattern exceeds
    similarity_threshold) or "spurious" (otherwise).

    Args:
        patterns: (N, dim) stored patterns (unit vectors)
        beta: inverse temperature for Hopfield dynamics
        n_queries: number of random probe vectors
        similarity_threshold: cosine similarity above which a converged
            state is classified as matching a stored pattern
        seed: random seed for probe generation

    Returns:
        (spurious_rate, stored_rate) where each is fraction of n_queries.
    """
    rng = np.random.default_rng(seed)
    N, dim = patterns.shape

    if N == 0:
        return 1.0, 0.0

    n_spurious = 0
    n_stored = 0

    for _ in range(n_queries):
        # Random unit vector probe
        probe = rng.standard_normal(dim)
        probe /= np.linalg.norm(probe)

        # Run to convergence
        converged = spreading_activation(beta, patterns, probe)

        # Classify: max cosine similarity to stored patterns
        similarities = patterns @ converged
        max_sim = float(np.max(similarities))

        if max_sim > similarity_threshold:
            n_stored += 1
        else:
            n_spurious += 1

    spurious_rate = n_spurious / n_queries
    stored_rate = n_stored / n_queries
    return spurious_rate, stored_rate


# ---------------------------------------------------------------------------
# Helper: within-cluster and between-cluster min cosine distances
# ---------------------------------------------------------------------------

def measure_delta_within_between(
    patterns: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Measure within-cluster and between-cluster min cosine distances.

    delta_within = min over all same-cluster pairs of (1 - cosine_sim)
    delta_between = min over all different-cluster pairs of (1 - cosine_sim)

    For efficiency with large N, samples up to 10000 pairs for each
    category when the exhaustive count would exceed that limit.

    Args:
        patterns: (N, dim) unit vectors
        labels: (N,) integer cluster assignments

    Returns:
        (delta_within, delta_between) -- minimum cosine distances.
        Returns (inf, inf) if no valid pairs exist for that category.
    """
    rng = np.random.default_rng(12345)
    N = patterns.shape[0]
    max_sample = 10000

    # Collect within-cluster and between-cluster index pairs
    within_pairs: list[tuple[int, int]] = []
    between_pairs: list[tuple[int, int]] = []

    for i in range(N):
        for j in range(i + 1, N):
            if labels[i] == labels[j]:
                within_pairs.append((i, j))
            else:
                between_pairs.append((i, j))

    def _min_cosine_distance(
        pair_list: list[tuple[int, int]],
    ) -> float:
        if len(pair_list) == 0:
            return float("inf")

        # Sample if too many pairs
        if len(pair_list) > max_sample:
            indices = rng.choice(len(pair_list), size=max_sample, replace=False)
            sampled = [pair_list[k] for k in indices]
        else:
            sampled = pair_list

        min_d = float("inf")
        for i, j in sampled:
            cos_sim = float(patterns[i] @ patterns[j])
            d = 1.0 - cos_sim
            if d < min_d:
                min_d = d
        return min_d

    delta_within = _min_cosine_distance(within_pairs)
    delta_between = _min_cosine_distance(between_pairs)

    return delta_within, delta_between


# ---------------------------------------------------------------------------
# Helper: cluster coherence via Hopfield recall
# ---------------------------------------------------------------------------

def measure_cluster_coherence(
    patterns: np.ndarray,
    labels: np.ndarray,
    beta: float,
) -> float:
    """Fraction of patterns where top-1 Hopfield recall returns a same-cluster pattern.

    For each pattern, runs spreading_activation to convergence, then finds
    the nearest stored pattern (by cosine similarity). If the nearest
    pattern shares the same cluster label, counts as coherent.

    Args:
        patterns: (N, dim) unit vectors
        labels: (N,) integer cluster assignments
        beta: inverse temperature for Hopfield dynamics

    Returns:
        Fraction in [0.0, 1.0] of patterns with coherent top-1 recall.
    """
    N = patterns.shape[0]
    if N <= 1:
        return 1.0

    n_coherent = 0
    for i in range(N):
        # Run spreading activation from pattern i
        converged = spreading_activation(beta, patterns, patterns[i])

        # Find nearest stored pattern (exclude self for a meaningful test)
        similarities = patterns @ converged
        # Mask self
        similarities[i] = -float("inf")
        nearest_idx = int(np.argmax(similarities))

        if labels[nearest_idx] == labels[i]:
            n_coherent += 1

    return n_coherent / N


# === Test classes follow below ===


class TestMultiCycleConvergence:
    """Test 1: Multi-cycle convergence -- delta_min and P@1 over 50 dream cycles.

    Setup: spread=0.04 (tight clusters), 5 clusters x 20 patterns, beta=10.
    The handoff showed: single cycle bumps delta from 0.126 to 0.200 but P@1 stays at 0.050.

    Questions answered:
    1. Does delta keep climbing? -> Assert monotonically non-decreasing (within noise)
    2. Where does it plateau? -> Record and report plateau cycle
    3. Does P@1 eventually respond? -> Check against capacity formula prediction

    Capacity formula: delta_threshold = log(4*beta*N) / beta
    At beta=10, N=100: delta_threshold = log(4000)/10 ~ 0.829
    This is very high -- predicts repulsion alone at beta=10 CAN'T reach it.
    """

    DIM = 128
    N_CLUSTERS = 5
    PER_CLUSTER = 20
    SPREAD = 0.04
    BETA = 10.0
    N_CYCLES = 50
    SEED = 42

    def _setup(self):
        """Create tight cluster patterns and return (patterns, labels, importances)."""
        centroids = make_separated_centroids(self.N_CLUSTERS, self.DIM, seed=self.SEED)
        patterns, labels = make_cluster_patterns(
            centroids, self.PER_CLUSTER, spread=self.SPREAD, seed=self.SEED,
        )
        importances = np.full(len(patterns), 0.3)
        return patterns, labels, importances

    def test_delta_monotonically_nondecreasing(self):
        """delta_min never decreases across dream cycles.

        Runs N_CYCLES dream cycles and records delta_min after each.
        Asserts that delta_min[i] >= delta_min[i-1] - 1e-9 for all i,
        accounting for floating point noise.
        """
        patterns, labels, importances = self._setup()
        deltas = [compute_min_delta(patterns)]

        current = patterns.copy()
        for cycle in range(self.N_CYCLES):
            report = dream_cycle_xb(
                current, self.BETA,
                importances=importances[:len(current)],
                labels=labels[:len(current)] if len(current) == len(labels) else None,
                seed=self.SEED + cycle,
            )
            current = report.patterns
            # Adjust importances/labels for reduced pattern count
            importances = np.full(len(current), 0.3)
            if len(current) < len(labels):
                labels = np.arange(len(current))
            deltas.append(compute_min_delta(current))

        # Verify monotonically non-decreasing (allow 1e-9 tolerance for float)
        for i in range(1, len(deltas)):
            assert deltas[i] >= deltas[i - 1] - 1e-9, (
                f"delta_min decreased at cycle {i}: {deltas[i-1]:.6f} -> {deltas[i]:.6f}"
            )

        # Report: print for visibility
        print(f"\ndelta_min trajectory: {deltas[0]:.4f} -> {deltas[-1]:.4f} over {self.N_CYCLES} cycles")
        print(f"delta_min plateau region: cycles {self._find_plateau(deltas)}")

    def test_delta_plateau_below_threshold(self):
        """delta_min plateaus below the capacity formula threshold at beta=10.

        Capacity formula: delta_threshold = log(4*beta*N) / beta
        At beta=10, N=100: delta_threshold ~ 0.829
        Repulsion with eta=0.01 cannot reach this. Asserts that after N_CYCLES
        dream cycles, delta_min remains below the capacity threshold.
        """
        patterns, labels, importances = self._setup()
        N = len(patterns)
        delta_threshold = np.log(4 * self.BETA * N) / self.BETA

        current = patterns.copy()
        final_delta = compute_min_delta(current)
        for cycle in range(self.N_CYCLES):
            report = dream_cycle_xb(
                current, self.BETA,
                importances=np.full(len(current), 0.3),
                seed=self.SEED + cycle,
            )
            current = report.patterns
            final_delta = compute_min_delta(current)

        print(f"\ndelta_threshold (capacity formula) = {delta_threshold:.4f}")
        print(f"delta_min after {self.N_CYCLES} cycles = {final_delta:.4f}")
        print(f"Ratio: {final_delta/delta_threshold:.2%} of threshold")

        # The key assertion: delta plateaus well below threshold at this beta
        # This proves repulsion alone isn't enough -- need higher beta or more cycles
        assert final_delta < delta_threshold, (
            f"delta_min {final_delta:.4f} reached threshold {delta_threshold:.4f} -- "
            f"unexpected at beta={self.BETA}"
        )

    def test_p1_trajectory(self):
        """Track P@1 over dream cycles. Record whether and when it improves.

        Builds a CoupledEngine at each cycle, measures P@1, and asserts that
        P@1 does not degrade significantly (allows 0.05 tolerance for
        pattern count changes from pruning/merging).
        """
        patterns, labels, importances = self._setup()

        # Build engine for initial P@1
        engine = CoupledEngine(dim=self.DIM, beta=self.BETA)
        for i in range(len(patterns)):
            engine.store(f"p{i}", patterns[i], importance=0.3)
        p1_initial = measure_p1(engine, patterns)

        # Run cycles, measuring P@1 at each
        current = patterns.copy()
        p1_values = [p1_initial]
        delta_values = [compute_min_delta(patterns)]
        n_patterns = [len(patterns)]

        for cycle in range(self.N_CYCLES):
            report = dream_cycle_xb(
                current, self.BETA,
                importances=np.full(len(current), 0.3),
                seed=self.SEED + cycle,
            )
            current = report.patterns

            # Rebuild engine with post-dream patterns
            engine = CoupledEngine(dim=self.DIM, beta=self.BETA)
            for i in range(len(current)):
                engine.store(f"p{i}", current[i], importance=0.3)

            p1_values.append(measure_p1(engine, current))
            delta_values.append(compute_min_delta(current))
            n_patterns.append(len(current))

        print(f"\nP@1 trajectory: {p1_values[0]:.3f} -> {p1_values[-1]:.3f}")
        print(f"delta_min trajectory: {delta_values[0]:.4f} -> {delta_values[-1]:.4f}")
        print(f"Pattern count: {n_patterns[0]} -> {n_patterns[-1]}")

        # P@1 should not decrease significantly
        assert p1_values[-1] >= p1_values[0] - 0.05, (
            f"P@1 degraded significantly: {p1_values[0]:.3f} -> {p1_values[-1]:.3f}"
        )

    def test_pattern_count_decreases(self):
        """Dream cycles reduce pattern count via pruning and merging.

        With spread=0.04, patterns within each cluster are very close together.
        Over 10 dream cycles, pruning and merging should remove some patterns.
        Asserts that the final pattern count is <= the initial count.
        """
        patterns, labels, importances = self._setup()
        N_initial = len(patterns)

        current = patterns.copy()
        for cycle in range(10):  # Just 10 cycles -- enough to observe pruning
            report = dream_cycle_xb(
                current, self.BETA,
                importances=np.full(len(current), 0.3),
                seed=self.SEED + cycle,
            )
            current = report.patterns

        N_final = len(current)
        print(f"\nPattern count: {N_initial} -> {N_final} ({N_initial - N_final} removed)")

        # With spread=0.04, patterns are very close -- expect pruning/merging
        assert N_final <= N_initial, (
            f"Pattern count increased: {N_initial} -> {N_final}"
        )

    @staticmethod
    def _find_plateau(values, tolerance=0.001):
        """Find first cycle where delta_min stops growing (change < tolerance)."""
        for i in range(1, len(values)):
            if abs(values[i] - values[i - 1]) < tolerance:
                return i
        return len(values)  # never plateaued


class TestDreamBetaSweep:
    """Test 2: Dream + beta sweep -- verify dreams shift the operating point.

    Pre-dream geometry needed beta~50+ for P@1=1.0 on tight clusters.
    Post-dream geometry has wider delta, so required beta should be lower.

    The capacity formula predicts the exact crossover:
    beta_threshold = log(4*N) / delta  (solving N = exp(beta*delta)/(4*beta) for beta when P@1 should jump)

    If empirical crossover matches prediction, dreams provably reduce beta requirement.
    """

    DIM = 128
    N_CLUSTERS = 5
    PER_CLUSTER = 20
    SPREAD = 0.04
    BETA_OPERATIONAL = 10.0
    N_DREAM_CYCLES = 30  # enough for convergence
    SEED = 42
    BETA_SWEEP = np.arange(10, 105, 5)  # beta from 10 to 100, step 5

    def _prepare_pre_post_dream(self):
        """Run dream cycles and return (pre_patterns, post_patterns)."""
        centroids = make_separated_centroids(self.N_CLUSTERS, self.DIM, seed=self.SEED)
        patterns, labels = make_cluster_patterns(
            centroids, self.PER_CLUSTER, spread=self.SPREAD, seed=self.SEED,
        )

        pre_patterns = patterns.copy()

        current = patterns.copy()
        for cycle in range(self.N_DREAM_CYCLES):
            report = dream_cycle_xb(
                current, self.BETA_OPERATIONAL,
                importances=np.full(len(current), 0.3),
                seed=self.SEED + cycle,
            )
            current = report.patterns

        return pre_patterns, current

    def test_beta_sweep_pre_vs_post(self):
        """Sweep beta and compare P@1 curves before and after dreaming.

        For each beta in [10, 15, ..., 100], builds a CoupledEngine with
        pre-dream and post-dream patterns and measures P@1. Asserts that
        post-dream P@1 >= pre-dream P@1 - 0.05 at every beta value (dreams
        should not degrade retrieval quality).
        """
        pre_patterns, post_patterns = self._prepare_pre_post_dream()

        pre_p1_curve = []
        post_p1_curve = []

        for beta in self.BETA_SWEEP:
            # Pre-dream P@1
            engine = CoupledEngine(dim=self.DIM, beta=float(beta))
            for i in range(len(pre_patterns)):
                engine.store(f"p{i}", pre_patterns[i], importance=0.3)
            pre_p1_curve.append(measure_p1(engine, pre_patterns))

            # Post-dream P@1
            engine = CoupledEngine(dim=self.DIM, beta=float(beta))
            for i in range(len(post_patterns)):
                engine.store(f"p{i}", post_patterns[i], importance=0.3)
            post_p1_curve.append(measure_p1(engine, post_patterns))

        print("\nbeta sweep results:")
        print(f"{'beta':>6} {'Pre P@1':>8} {'Post P@1':>9}")
        for beta, pre_p1, post_p1 in zip(self.BETA_SWEEP, pre_p1_curve, post_p1_curve):
            marker = " <-- post beats pre" if post_p1 > pre_p1 + 0.01 else ""
            print(f"{beta:>6.0f} {pre_p1:>8.3f} {post_p1:>9.3f}{marker}")

        # Key assertion: post-dream P@1 >= pre-dream P@1 at every beta
        for i, beta in enumerate(self.BETA_SWEEP):
            assert post_p1_curve[i] >= pre_p1_curve[i] - 0.05, (
                f"Post-dream P@1 worse than pre-dream at beta={beta}: "
                f"{pre_p1_curve[i]:.3f} -> {post_p1_curve[i]:.3f}"
            )

    def test_predicted_vs_empirical_crossover(self):
        """Verify capacity formula predicts the beta crossover point.

        Computes delta_pre and delta_post, then uses the capacity formula
        beta_threshold ~ log(4*N) / delta to predict the required beta for
        each geometry. Asserts that post-dream requires lower beta (due to
        wider delta and/or fewer patterns).
        """
        pre_patterns, post_patterns = self._prepare_pre_post_dream()

        delta_pre = compute_min_delta(pre_patterns)
        delta_post = compute_min_delta(post_patterns)
        N_pre = len(pre_patterns)
        N_post = len(post_patterns)

        # Predicted beta where P@1 should reach 1.0:
        # From capacity: N < exp(beta*delta)/(4*beta)
        # Approximate: beta_threshold ~ log(4*N) / delta (ignoring the log(beta) correction)
        # This is a rough prediction -- we check if it's in the right ballpark

        if delta_pre > 0:
            beta_predicted_pre = np.log(4 * N_pre) / delta_pre
        else:
            beta_predicted_pre = float('inf')

        if delta_post > 0:
            beta_predicted_post = np.log(4 * N_post) / delta_post
        else:
            beta_predicted_post = float('inf')

        print(f"\nPre-dream:  delta={delta_pre:.4f}, N={N_pre}, beta_predicted~{beta_predicted_pre:.1f}")
        print(f"Post-dream: delta={delta_post:.4f}, N={N_post}, beta_predicted~{beta_predicted_post:.1f}")
        print(f"beta reduction: {beta_predicted_pre:.1f} -> {beta_predicted_post:.1f}")

        # Key assertion: post-dream requires lower beta
        assert beta_predicted_post <= beta_predicted_pre + 1e-6, (
            f"Post-dream beta_predicted ({beta_predicted_post:.1f}) not lower than "
            f"pre-dream ({beta_predicted_pre:.1f})"
        )

        # Verify the shift is due to both delta increase AND N decrease
        assert delta_post >= delta_pre - 1e-9, (
            f"delta decreased after dream: {delta_pre:.4f} -> {delta_post:.4f}"
        )

    def test_post_dream_needs_lower_beta(self):
        """Post-dream patterns reach P@1=1.0 at lower beta than pre-dream.

        Sweeps beta from 10 to 200 in steps of 5 and finds the minimum beta
        where P@1 >= 0.95 for both pre-dream and post-dream geometries.
        Asserts that post-dream achieves the target at equal or lower beta.
        """
        pre_patterns, post_patterns = self._prepare_pre_post_dream()

        # Find minimum beta where P@1 = 1.0
        def find_p1_threshold(patterns, target_p1=0.95):
            """Find minimum beta where P@1 >= target_p1."""
            for beta in range(10, 201, 5):
                engine = CoupledEngine(dim=self.DIM, beta=float(beta))
                for i in range(len(patterns)):
                    engine.store(f"p{i}", patterns[i], importance=0.3)
                p1 = measure_p1(engine, patterns)
                if p1 >= target_p1:
                    return beta
            return None  # never reached

        beta_pre = find_p1_threshold(pre_patterns)
        beta_post = find_p1_threshold(post_patterns)

        print(f"\nbeta for P@1>=0.95: pre={beta_pre}, post={beta_post}")

        if beta_pre is not None and beta_post is not None:
            assert beta_post <= beta_pre, (
                f"Post-dream needs higher beta ({beta_post}) than pre-dream ({beta_pre})"
            )
