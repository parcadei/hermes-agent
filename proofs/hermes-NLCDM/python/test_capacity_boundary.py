"""Capacity boundary test — characterizes dream cycle behavior at the critical region.

Validated findings:
  1. Capacity wall is sharp: P@1 drops from 1.0 to <0.1 over a narrow N range
  2. Dream (with default β_nrem = 2×β) is a no-op for P@1 — the NREM attractor
     IS the pattern itself at high β, so pulling toward it does nothing
  3. NREM at low β_nrem causes centroid collapse (patterns merge toward
     cluster average), which HURTS P@1 but increases cluster coherence
  4. The capacity formula N_max = exp(β·δ)/(4·β·M²) is extremely conservative
     — empirical capacity is 10-20x higher
  5. Dream's value is in cluster-level coherence on ambiguous queries,
     not in pattern-level discrimination (P@1)

Setup:
  dim=128, 5 clusters, orthogonal centroids (cosine ≈ 0),
  within-cluster spread=0.10 (cosine similarity ≈ 0.60-0.80, δ ≈ 0.20-0.40),
  operational β=10.

Phase 1: Pack to boundary — find empirical capacity wall
Phase 2: Dream at boundary — verify no degradation, measure coherence
Phase 3: Overpack past boundary — characterize degradation curve
Phase 4: NREM β sweep — map compression vs preservation tradeoff
"""

from __future__ import annotations

import numpy as np
import pytest

from coupled_engine import CoupledEngine
from dream_ops import nrem_replay_xb, spreading_activation


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_separated_centroids(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Generate n orthonormal centroids (pairwise cosine ≈ 0)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, max(n, dim)))
    Q, _ = np.linalg.qr(M)
    return Q[:, :n].T


def make_cluster_patterns(
    centroids: np.ndarray,
    per_cluster: int,
    spread: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate clustered unit vectors around centroids."""
    rng = np.random.default_rng(seed)
    n_clusters, dim = centroids.shape
    patterns, labels = [], []
    for c in range(n_clusters):
        for _ in range(per_cluster):
            p = centroids[c] + spread * rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            patterns.append(p)
            labels.append(c)
    return np.array(patterns), np.array(labels)


def compute_n_max(beta: float, delta: float, M_sq: float = 1.0) -> float:
    """Capacity formula from Capacity.lean: N_max = exp(β·δ)/(4·β·M²)."""
    return np.exp(beta * delta) / (4.0 * beta * M_sq)


def compute_min_delta(patterns: np.ndarray) -> float:
    """Minimum pairwise cosine distance."""
    N = patterns.shape[0]
    min_d = 2.0
    for i in range(N):
        for j in range(i + 1, N):
            min_d = min(min_d, 1.0 - float(patterns[i] @ patterns[j]))
    return min_d


def measure_p1(engine: CoupledEngine, patterns: np.ndarray) -> float:
    """Fraction where exact query returns itself as top-1."""
    N = len(patterns)
    if N == 0:
        return 1.0
    return sum(
        1 for i in range(N)
        if engine.query(patterns[i], top_k=1)[0]["index"] == i
    ) / N


def cluster_coherence(engine: CoupledEngine, query: np.ndarray, labels: np.ndarray, top_k: int = 5) -> float:
    """Fraction of top-k results sharing the plurality cluster label."""
    hits = engine.query(query, top_k=top_k)
    if not hits:
        return 0.0
    retrieved_labels = [int(labels[h["index"]]) for h in hits]
    from collections import Counter
    return Counter(retrieved_labels).most_common(1)[0][1] / len(retrieved_labels)


def intra_cluster_spread(patterns: np.ndarray, labels: np.ndarray, cluster: int) -> float:
    """Mean pairwise cosine distance within a cluster."""
    mask = labels == cluster
    c_pats = patterns[mask]
    n = len(c_pats)
    if n < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - float(c_pats[i] @ c_pats[j])
            count += 1
    return total / count


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

DIM = 128
N_CLUSTERS = 5
SPREAD = 0.10
BETA = 10.0
SEED = 42


def _make_centroids():
    return make_separated_centroids(N_CLUSTERS, DIM, seed=SEED)


# ===========================================================================
# Phase 1: Pack to boundary
# ===========================================================================

class TestPackToBoundary:
    """Incrementally add patterns per cluster, find the empirical P@1 wall."""

    def test_capacity_wall(self):
        """Find where P@1 drops below 0.9 and compare to formula."""
        centroids = _make_centroids()
        rng = np.random.default_rng(SEED + 100)
        max_per_cluster = 35

        # Baseline: 5 patterns in each of 4 other clusters
        baseline_pats, baseline_lbls = [], []
        for c in range(1, N_CLUSTERS):
            for _ in range(5):
                p = centroids[c] + SPREAD * rng.standard_normal(DIM)
                p /= np.linalg.norm(p)
                baseline_pats.append(p)
                baseline_lbls.append(c)

        c0_pats = []
        p1_curve = []
        wall_n = None

        for n_add in range(1, max_per_cluster + 1):
            p = centroids[0] + SPREAD * rng.standard_normal(DIM)
            p /= np.linalg.norm(p)
            c0_pats.append(p)

            all_pats = c0_pats + baseline_pats
            all_lbls = [0] * len(c0_pats) + baseline_lbls

            engine = CoupledEngine(dim=DIM, beta=BETA)
            for i, pat in enumerate(all_pats):
                engine.store(f"p{i}", pat, importance=0.5)

            c0_idx = [i for i, l in enumerate(all_lbls) if l == 0]
            correct = sum(
                1 for i in c0_idx
                if engine.query(all_pats[i], top_k=1)[0]["index"] == i
            )
            p1 = correct / len(c0_idx)
            p1_curve.append(p1)

            if wall_n is None and p1 < 0.9:
                wall_n = n_add

        delta_within = compute_min_delta(np.array(c0_pats))
        n_max_formula = compute_n_max(BETA, delta_within)

        print(f"\n  Phase 1: Capacity wall (cluster 0, β={BETA}, dim={DIM}, spread={SPREAD})")
        print(f"  δ_min (within cluster) = {delta_within:.4f}")
        print(f"  N_max (formula) = {n_max_formula:.1f}")
        print(f"  Empirical wall (P@1 < 0.9) at N = {wall_n}")
        print(f"  {'N':>4} {'P@1':>6}")
        for i, p1 in enumerate(p1_curve):
            marker = " <-- wall" if (i + 1) == wall_n else ""
            print(f"  {i+1:>4} {p1:>6.3f}{marker}")

        # Assertions
        assert wall_n is not None, "P@1 never dropped below 0.9 — increase max_per_cluster"
        assert wall_n >= 5, f"Capacity wall too low at N={wall_n} — check parameters"
        assert min(p1_curve) < 0.2, "Capacity cliff not steep enough"

        # The formula gives a very conservative bound (N_max ≈ 1 vs empirical ~17)
        # Document the gap — this is expected for a sufficient condition
        print(f"\n  Formula gap: empirical wall={wall_n}, formula N_max={n_max_formula:.1f}")
        print(f"  Ratio: {wall_n / max(n_max_formula, 0.01):.0f}x")


# ===========================================================================
# Phase 2: Dream at boundary
# ===========================================================================

class TestDreamAtBoundary:
    """At the capacity boundary, verify dream doesn't degrade P@1.
    Also measure coherence on ambiguous queries.
    """

    def test_dream_preserves_p1_at_boundary(self):
        """Pack near the capacity wall, dream, verify P@1 is preserved."""
        centroids = _make_centroids()
        per_cluster = 18  # just below empirical wall (P@1 ≈ 0.90-0.95)
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster, spread=SPREAD, seed=SEED + 200,
        )
        N = len(patterns)

        # No dream
        eng_nd = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(N):
            eng_nd.store(f"p{i}", patterns[i], importance=0.5)
        p1_nd = measure_p1(eng_nd, patterns)

        # With dream
        eng_d = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(N):
            eng_d.store(f"p{i}", patterns[i], importance=0.5)
        result = eng_d.dream()

        p1_d = sum(
            1 for i in range(N)
            if eng_d.query(patterns[i], top_k=1)[0]["index"] == i
        ) / N

        print(f"\n  Phase 2: Dream at boundary (N={N}, β={BETA})")
        print(f"  P@1 no-dream: {p1_nd:.3f}")
        print(f"  P@1 dream:    {p1_d:.3f} (Δ={p1_d - p1_nd:+.3f})")

        # Dream should not degrade
        assert p1_d >= p1_nd - 0.05, (
            f"Dream degraded P@1: {p1_nd:.3f} → {p1_d:.3f}"
        )

    def test_dream_coherence_at_boundary(self):
        """Ambiguous queries at the boundary: dream should preserve or improve coherence."""
        centroids = _make_centroids()
        per_cluster = 18
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster, spread=SPREAD, seed=SEED + 200,
        )
        N = len(patterns)

        eng_nd = CoupledEngine(dim=DIM, beta=BETA)
        eng_d = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(N):
            eng_nd.store(f"p{i}", patterns[i], importance=0.5)
            eng_d.store(f"p{i}", patterns[i], importance=0.5)
        eng_d.dream()

        rng = np.random.default_rng(SEED + 300)
        n_trials = 15
        coh_nd, coh_d = [], []

        for _ in range(n_trials):
            c1, c2 = rng.choice(N_CLUSTERS, size=2, replace=False)
            midpoint = centroids[c1] + centroids[c2]
            midpoint /= np.linalg.norm(midpoint)

            coh_nd.append(cluster_coherence(eng_nd, midpoint, labels))
            coh_d.append(cluster_coherence(eng_d, midpoint, labels))

        mean_nd = float(np.mean(coh_nd))
        mean_d = float(np.mean(coh_d))

        print(f"\n  Phase 2b: Coherence on ambiguous queries at boundary")
        print(f"  Coherence no-dream: {mean_nd:.3f}")
        print(f"  Coherence dream:    {mean_d:.3f} (Δ={mean_d - mean_nd:+.3f})")

        # Dream should not degrade coherence
        assert mean_d >= mean_nd - 0.1, (
            f"Dream degraded coherence: {mean_nd:.3f} → {mean_d:.3f}"
        )


# ===========================================================================
# Phase 3: Overpack past boundary
# ===========================================================================

class TestOverpackPastBoundary:
    """Pack well past the capacity wall. Characterize degradation."""

    def test_overpack_degradation_curve(self):
        """Sweep from below wall to 2x past wall, measure P@1 at each level."""
        centroids = _make_centroids()
        sweep = [10, 15, 18, 20, 22, 25, 30]

        print(f"\n  Phase 3: Overpack degradation curve (β={BETA}, spread={SPREAD})")
        print(f"  {'N/c':>5} {'Total':>6} {'P@1_nd':>7} {'P@1_d':>6} {'Δ':>7}")

        for n_per in sweep:
            patterns, labels = make_cluster_patterns(
                centroids, n_per, spread=SPREAD, seed=SEED + 300,
            )
            N = len(patterns)

            eng_nd = CoupledEngine(dim=DIM, beta=BETA)
            eng_d = CoupledEngine(dim=DIM, beta=BETA)
            for i in range(N):
                eng_nd.store(f"p{i}", patterns[i], importance=0.5)
                eng_d.store(f"p{i}", patterns[i], importance=0.5)
            eng_d.dream()

            p1_nd = measure_p1(eng_nd, patterns)
            p1_d = sum(
                1 for i in range(N)
                if eng_d.query(patterns[i], top_k=1)[0]["index"] == i
            ) / N
            delta = p1_d - p1_nd

            print(f"  {n_per:>5} {N:>6} {p1_nd:>7.3f} {p1_d:>6.3f} {delta:>+7.3f}")

        # Assertions:
        # 1. Below wall should have high P@1
        pats_low, _ = make_cluster_patterns(centroids, 10, spread=SPREAD, seed=SEED + 300)
        eng_low = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(len(pats_low)):
            eng_low.store(f"p{i}", pats_low[i], importance=0.5)
        assert measure_p1(eng_low, pats_low) >= 0.9, "P@1 should be high below capacity wall"

        # 2. Well past wall should have low P@1
        pats_hi, _ = make_cluster_patterns(centroids, 30, spread=SPREAD, seed=SEED + 300)
        eng_hi = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(len(pats_hi)):
            eng_hi.store(f"p{i}", pats_hi[i], importance=0.5)
        assert measure_p1(eng_hi, pats_hi) < 0.2, "P@1 should be low past 2x capacity wall"

    def test_dream_at_overpack_no_degradation(self):
        """Dream should not make retrieval worse even when overpacked."""
        centroids = _make_centroids()
        per_cluster = 25  # well past wall
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster, spread=SPREAD, seed=SEED + 400,
        )
        N = len(patterns)

        eng_nd = CoupledEngine(dim=DIM, beta=BETA)
        eng_d = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(N):
            eng_nd.store(f"p{i}", patterns[i], importance=0.5)
            eng_d.store(f"p{i}", patterns[i], importance=0.5)
        eng_d.dream()

        p1_nd = measure_p1(eng_nd, patterns)
        p1_d = sum(
            1 for i in range(N)
            if eng_d.query(patterns[i], top_k=1)[0]["index"] == i
        ) / N

        print(f"\n  Phase 3b: Overpack dream non-degradation (N/c={per_cluster})")
        print(f"  P@1 no-dream: {p1_nd:.3f}")
        print(f"  P@1 dream:    {p1_d:.3f}")

        assert p1_d >= p1_nd - 0.05, (
            f"Dream degraded P@1 when overpacked: {p1_nd:.3f} → {p1_d:.3f}"
        )


# ===========================================================================
# Phase 4: NREM β sweep
# ===========================================================================

class TestNREMBetaSweep:
    """Map the compression-preservation tradeoff as a function of β_nrem.

    Key physics:
      - Low β_nrem → neighbors have significant softmax weight → attractor
        is the cluster centroid → patterns compress (centroid collapse)
      - High β_nrem → self dominates softmax → attractor ≈ self → no change
      - The transition β depends on within-cluster cosine similarity
    """

    def _build_patterns(self, per_cluster: int = 20, seed: int = SEED + 500):
        centroids = _make_centroids()
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster, spread=SPREAD, seed=seed,
        )
        return centroids, patterns, labels

    def test_beta_nrem_sweep(self):
        """Sweep β_nrem, measure displacement and spread compression."""
        centroids, patterns, labels = self._build_patterns()
        N = len(patterns)
        tagged = list(range(N))

        orig_spreads = [intra_cluster_spread(patterns, labels, c) for c in range(N_CLUSTERS)]
        orig_mean = float(np.mean(orig_spreads))

        beta_values = [2, 5, 8, 10, 15, 20, 30, 50, 100]
        results = {}

        for beta_nrem in beta_values:
            X_after = nrem_replay_xb(patterns, tagged, beta_high=beta_nrem, pull_strength=0.05)
            disp = float(np.mean([np.linalg.norm(X_after[i] - patterns[i]) for i in range(N)]))
            spreads = [intra_cluster_spread(X_after, labels, c) for c in range(N_CLUSTERS)]
            mean_spread = float(np.mean(spreads))
            results[beta_nrem] = {"displacement": disp, "mean_spread": mean_spread}

        print(f"\n  Phase 4: NREM β sweep (operational β={BETA}, spread={SPREAD})")
        print(f"  Original mean intra-cluster spread: {orig_mean:.4f}")
        print(f"  {'β_nrem':>8} {'displace':>9} {'spread':>8} {'Δ_spread':>10} {'compress%':>10}")
        for bv in beta_values:
            r = results[bv]
            ds = r["mean_spread"] - orig_mean
            compress = (1.0 - r["mean_spread"] / orig_mean) * 100
            print(f"  {bv:>8} {r['displacement']:>9.6f} {r['mean_spread']:>8.4f} "
                  f"{ds:>+10.4f} {compress:>9.1f}%")

        # Assertions:
        # 1. Low β should compress more than high β (monotonic trend)
        assert results[2]["mean_spread"] <= results[50]["mean_spread"], (
            f"Low β didn't compress more: β=2 spread={results[2]['mean_spread']:.4f}, "
            f"β=50 spread={results[50]['mean_spread']:.4f}"
        )

        # 2. Low β should have larger displacement than high β
        assert results[2]["displacement"] > results[100]["displacement"], (
            f"Low β didn't displace more: β=2 disp={results[2]['displacement']:.6f}, "
            f"β=100 disp={results[100]['displacement']:.6f}"
        )

        # 3. Very high β should be near-zero displacement
        assert results[100]["displacement"] < 0.001, (
            f"β_nrem=100 should be a no-op: displacement={results[100]['displacement']:.6f}"
        )

    def test_centroid_collapse_at_low_beta(self):
        """At low β_nrem with multiple cycles, patterns should collapse
        toward cluster centroids (spread decreases substantially).
        """
        centroids, patterns, labels = self._build_patterns()
        N = len(patterns)
        tagged = list(range(N))
        beta_nrem = 5  # low enough for neighbor mixing

        X = patterns.copy()
        orig_spread = float(np.mean(
            [intra_cluster_spread(patterns, labels, c) for c in range(N_CLUSTERS)]
        ))

        cycle_spreads = [orig_spread]
        for cycle in range(10):
            X = nrem_replay_xb(X, tagged, beta_high=beta_nrem, pull_strength=0.10)
            spread = float(np.mean(
                [intra_cluster_spread(X, labels, c) for c in range(N_CLUSTERS)]
            ))
            cycle_spreads.append(spread)

        print(f"\n  Phase 4b: Centroid collapse (β_nrem={beta_nrem}, pull=0.10, 10 cycles)")
        print(f"  {'Cycle':>6} {'Spread':>8} {'Compress%':>10}")
        for i, s in enumerate(cycle_spreads):
            compress = (1.0 - s / orig_spread) * 100
            print(f"  {i:>6} {s:>8.4f} {compress:>9.1f}%")

        final_compress = (1.0 - cycle_spreads[-1] / orig_spread) * 100
        # After 10 cycles at low β, should see meaningful compression
        assert cycle_spreads[-1] < orig_spread, (
            f"No compression after 10 cycles: {orig_spread:.4f} → {cycle_spreads[-1]:.4f}"
        )
        assert final_compress > 5.0, (
            f"Compression too weak: {final_compress:.1f}%"
        )

    def test_high_beta_preserves_patterns(self):
        """At high β_nrem, NREM should be a near-no-op (patterns unchanged)."""
        centroids, patterns, labels = self._build_patterns()
        N = len(patterns)
        tagged = list(range(N))
        beta_nrem = 50  # high enough that attractor = self

        X_after = nrem_replay_xb(patterns, tagged, beta_high=beta_nrem, pull_strength=0.05)

        displacements = [np.linalg.norm(X_after[i] - patterns[i]) for i in range(N)]
        mean_disp = float(np.mean(displacements))
        max_disp = float(np.max(displacements))

        print(f"\n  Phase 4c: Pattern preservation (β_nrem={beta_nrem})")
        print(f"  Mean displacement: {mean_disp:.8f}")
        print(f"  Max displacement:  {max_disp:.8f}")

        assert mean_disp < 0.001, (
            f"High β_nrem should preserve patterns: mean displacement={mean_disp:.6f}"
        )

    def test_p1_impact_of_nrem_beta(self):
        """Document how different β_nrem values affect retrieval P@1.

        Key finding: low β_nrem HURTS P@1 because centroid collapse
        makes patterns less distinguishable. High β_nrem is a no-op.
        """
        centroids = _make_centroids()
        per_cluster = 18  # near wall
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster, spread=SPREAD, seed=SEED + 600,
        )
        N = len(patterns)
        tagged = list(range(N))

        # Baseline P@1
        eng_base = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(N):
            eng_base.store(f"p{i}", patterns[i], importance=0.5)
        p1_base = measure_p1(eng_base, patterns)

        print(f"\n  Phase 4d: P@1 impact of NREM β (N/c={per_cluster}, β_op={BETA})")
        print(f"  Baseline P@1: {p1_base:.3f}")
        print(f"  {'β_nrem':>8} {'P@1':>6} {'Δ':>7}")

        for beta_nrem in [5, 10, 15, 20, 30, 60]:
            X_after = nrem_replay_xb(
                patterns, tagged, beta_high=beta_nrem, pull_strength=0.05,
            )
            eng = CoupledEngine(dim=DIM, beta=BETA)
            for i in range(N):
                eng.store(f"p{i}", X_after[i], importance=0.5)
            p1 = measure_p1(eng, X_after)
            print(f"  {beta_nrem:>8} {p1:>6.3f} {p1 - p1_base:>+7.3f}")

        # Low β_nrem should hurt P@1 (centroid collapse = less distinguishable)
        X_low = nrem_replay_xb(patterns, tagged, beta_high=5, pull_strength=0.05)
        eng_low = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(N):
            eng_low.store(f"p{i}", X_low[i], importance=0.5)
        p1_low = measure_p1(eng_low, X_low)

        assert p1_low <= p1_base + 0.05, (
            f"Low β_nrem unexpectedly improved P@1: base={p1_base:.3f}, low={p1_low:.3f}"
        )


# ===========================================================================
# Combined diagnostic report
# ===========================================================================

class TestCapacityReport:
    """End-to-end diagnostic report. Always passes — purely informational."""

    def test_full_report(self):
        centroids = _make_centroids()
        per_cluster = 18
        patterns, labels = make_cluster_patterns(
            centroids, per_cluster, spread=SPREAD, seed=SEED + 700,
        )
        N = len(patterns)

        # Capacity formula
        delta = compute_min_delta(patterns)
        n_max = compute_n_max(BETA, delta)

        # Pre/post dream
        eng_nd = CoupledEngine(dim=DIM, beta=BETA)
        eng_d = CoupledEngine(dim=DIM, beta=BETA)
        for i in range(N):
            eng_nd.store(f"p{i}", patterns[i], importance=0.5)
            eng_d.store(f"p{i}", patterns[i], importance=0.5)
        eng_d.dream()

        p1_nd = measure_p1(eng_nd, patterns)
        p1_d = sum(
            1 for i in range(N)
            if eng_d.query(patterns[i], top_k=1)[0]["index"] == i
        ) / N

        # Coherence on ambiguous queries
        rng = np.random.default_rng(SEED + 800)
        coh_nd, coh_d = [], []
        for _ in range(15):
            c1, c2 = rng.choice(N_CLUSTERS, size=2, replace=False)
            mid = centroids[c1] + centroids[c2]
            mid /= np.linalg.norm(mid)
            coh_nd.append(cluster_coherence(eng_nd, mid, labels))
            coh_d.append(cluster_coherence(eng_d, mid, labels))

        print(f"\n{'='*70}")
        print(f"  CAPACITY BOUNDARY REPORT")
        print(f"  dim={DIM}, clusters={N_CLUSTERS}, β={BETA}, spread={SPREAD}")
        print(f"  δ_min = {delta:.4f}, N_max (formula) = {n_max:.1f}")
        print(f"  N = {N} ({per_cluster}/cluster)")
        print(f"{'='*70}")
        print(f"  P@1:       no-dream={p1_nd:.3f}, dream={p1_d:.3f} (Δ={p1_d - p1_nd:+.3f})")
        print(f"  Coherence: no-dream={np.mean(coh_nd):.3f}, dream={np.mean(coh_d):.3f} "
              f"(Δ={np.mean(coh_d) - np.mean(coh_nd):+.3f})")
        print(f"\n  NREM physics: β_nrem = 2×{BETA} = {2*BETA}")
        print(f"  At β_nrem={2*BETA}, self-weight in softmax ≈ 1.0")
        print(f"  → Attractor = pattern itself → NREM is a no-op")
        print(f"  → Dream value comes from REM phases (unlearn + explore)")

        # Per-cluster P@1
        print(f"\n  Per-cluster P@1:")
        print(f"  {'Cluster':>8} {'P@1_nd':>7} {'P@1_d':>6}")
        for c in range(N_CLUSTERS):
            mask = labels == c
            c_idx = np.where(mask)[0]
            c_nd = sum(1 for i in c_idx if eng_nd.query(patterns[i], top_k=1)[0]["index"] == i) / len(c_idx)
            c_d = sum(1 for i in c_idx if eng_d.query(patterns[i], top_k=1)[0]["index"] == i) / len(c_idx)
            print(f"  {c:>8} {c_nd:>7.3f} {c_d:>6.3f}")
        print(f"{'='*70}")
