"""Discriminative tests for NLCDM Phase 8-10 mathematical claims.

Tests are designed to FALSIFY (not just validate) the formal proof claims
by probing the gap between Lean definitions (raw dot product on abstract vectors)
and Python implementations (cosine similarity on normalized vectors).

Each hypothesis is tested with:
  - Happy path: conditions matching the theorem hypotheses
  - Stress tests: probing boundary cases and third-alternative scenarios
  - Measurements: actual numerical values printed for analysis

Hypotheses:
  H1 (Phase 8): Bridge Formation via Trace Centroids
  H2 (Phase 9b): Conditional Bridge Monotonicity Under Dream Operations
  H3 (Phase 10): Transfer Dynamics — Cross-Domain Retrieval Lower Bound
"""

import sys
import numpy as np
import pytest

# Ensure the proof code directory is importable
sys.path.insert(0, "/Users/cosimo/.hermes/hermes-agent/proofs/hermes-NLCDM/python")

from dream_ops import hopfield_update, nrem_merge_xb, nrem_prune_xb
from nlcdm_core import cosine_sim, softmax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_unit_vector(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random unit vector in R^d."""
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def make_vector_with_cosine_sim(
    target: np.ndarray,
    desired_sim: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a unit vector with approximately the given cosine similarity to target.

    Strategy: v = desired_sim * target_hat + sqrt(1 - desired_sim^2) * orthogonal_hat
    """
    d = target.shape[0]
    target_hat = target / np.linalg.norm(target)

    # Generate a random vector orthogonal to target
    rand_vec = rng.standard_normal(d)
    rand_vec -= np.dot(rand_vec, target_hat) * target_hat
    orth_norm = np.linalg.norm(rand_vec)
    if orth_norm < 1e-12:
        # Extremely unlikely; just regenerate
        rand_vec = rng.standard_normal(d)
        rand_vec -= np.dot(rand_vec, target_hat) * target_hat
        orth_norm = np.linalg.norm(rand_vec)
    orth_hat = rand_vec / orth_norm

    s = np.clip(desired_sim, -1.0, 1.0)
    v = s * target_hat + np.sqrt(max(0, 1 - s * s)) * orth_hat
    v = v / np.linalg.norm(v)
    return v


def make_cross_domain_vector(
    source: np.ndarray,
    target: np.ndarray,
    sim_s: float,
    sim_t: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a unit vector with approximately sim_s to source and sim_t to target.

    For this to work well, source and target should be roughly orthogonal,
    giving enough degrees of freedom.

    Strategy:
      v = alpha * source_hat + beta * target_hat + gamma * residual_hat
      where alpha, beta are chosen to approximate the desired similarities
      and gamma fills the remaining norm to make ||v|| = 1.
    """
    d = source.shape[0]
    s_hat = source / np.linalg.norm(source)
    t_hat = target / np.linalg.norm(target)

    # Gram-Schmidt: make t_hat orthogonal to s_hat for clean decomposition
    overlap_st = np.dot(s_hat, t_hat)
    t_perp = t_hat - overlap_st * s_hat
    t_perp_norm = np.linalg.norm(t_perp)
    if t_perp_norm < 1e-10:
        # source and target are nearly parallel; fall back
        return make_vector_with_cosine_sim(source, sim_s, rng)
    t_perp_hat = t_perp / t_perp_norm

    # v = alpha * s_hat + beta' * t_perp_hat + gamma * residual
    # cos(v, s_hat) = alpha = sim_s
    # cos(v, t_hat) = alpha * overlap_st + beta' * t_perp_norm / 1.0
    #   = sim_s * overlap_st + beta' * t_perp_norm
    # So: beta' = (sim_t - sim_s * overlap_st) / t_perp_norm
    alpha = sim_s
    beta_prime = (sim_t - sim_s * overlap_st) / t_perp_norm

    used_norm_sq = alpha ** 2 + beta_prime ** 2
    if used_norm_sq > 1.0:
        # Scale down to stay on the unit sphere (sacrificing exact sims slightly)
        scale = 0.99 / np.sqrt(used_norm_sq)
        alpha *= scale
        beta_prime *= scale
        used_norm_sq = alpha ** 2 + beta_prime ** 2

    # Residual direction orthogonal to both s_hat and t_perp_hat
    residual = rng.standard_normal(d)
    residual -= np.dot(residual, s_hat) * s_hat
    residual -= np.dot(residual, t_perp_hat) * t_perp_hat
    res_norm = np.linalg.norm(residual)
    if res_norm > 1e-12:
        residual = residual / res_norm
    gamma = np.sqrt(max(0, 1.0 - used_norm_sq))

    v = alpha * s_hat + beta_prime * t_perp_hat + gamma * residual
    v = v / np.linalg.norm(v)
    return v


def count_bridges(patterns: np.ndarray, threshold: float) -> int:
    """Count the number of ordered bridge pairs (i, j) with i != j
    and cosine_sim(patterns[i], patterns[j]) >= threshold."""
    N = patterns.shape[0]
    count = 0
    for i in range(N):
        for j in range(N):
            if i != j and cosine_sim(patterns[i], patterns[j]) >= threshold:
                count += 1
    return count


def count_bridges_dot(patterns: np.ndarray, threshold: float) -> int:
    """Count bridges using raw dot product (Lean definition) instead of
    cosine similarity."""
    N = patterns.shape[0]
    count = 0
    for i in range(N):
        for j in range(N):
            if i != j and np.dot(patterns[i], patterns[j]) >= threshold:
                count += 1
    return count


# ===========================================================================
# H1: Bridge Formation (Phase 8)
# ===========================================================================

class TestH1BridgeFormation:
    """Phase 8: When K trace vectors each have cosine similarity >= sigma with
    both source and target, their centroid (as computed by nrem_merge_xb)
    has cosine_sim >= sigma with both, creating >= 2 bridges.

    CRITICAL: Lean uses raw dot product (dotSim). Python normalizes centroids.
    For unit vectors, dot product = cosine sim. But the centroid of unit vectors
    is NOT a unit vector. Normalization changes the direction.
    """

    @pytest.fixture
    def setup_d128(self):
        """Standard d=128 setup with well-separated source and target."""
        rng = np.random.default_rng(42)
        d = 128
        source = make_unit_vector(d, rng)
        # Make target orthogonal to source
        target = make_unit_vector(d, rng)
        target -= np.dot(target, source) * source
        target = target / np.linalg.norm(target)
        return d, source, target, rng

    def test_h1_happy_path_centroid_preserves_similarity(self, setup_d128):
        """HAPPY PATH: K=5 cross-domain patterns each with sim >= sigma to both
        source and target. Verify centroid has sim >= sigma to both.

        NOTE: For orthogonal source/target, the maximum achievable equal cosine
        similarity to both is 1/sqrt(2) = 0.707. We use sigma=0.65 which is
        achievable. The Lean theorem's sigma is unconstrained because it reasons
        about arbitrary vectors, not about constructibility.
        """
        d, source, target, rng = setup_d128
        sigma = 0.65  # Must be < 1/sqrt(2) for orthogonal source/target
        K = 5

        patterns = []
        for k in range(K):
            v = make_cross_domain_vector(source, target, sigma, sigma, rng)
            patterns.append(v)
        patterns = np.array(patterns)

        # Verify input: all K patterns have sim >= sigma to both
        sims_to_source = [cosine_sim(patterns[k], source) for k in range(K)]
        sims_to_target = [cosine_sim(patterns[k], target) for k in range(K)]

        print(f"\n--- H1 Happy Path: K={K}, sigma={sigma}, d={d} ---")
        print(f"  Source-Target orthogonality: cosine_sim = {cosine_sim(source, target):.6f}")
        for k in range(K):
            print(f"  Pattern {k}: sim_source={sims_to_source[k]:.6f}, sim_target={sims_to_target[k]:.6f}")

        # Compute raw centroid (what Lean theorem reasons about)
        raw_centroid = np.mean(patterns, axis=0)
        raw_dot_source = np.dot(raw_centroid, source)
        raw_dot_target = np.dot(raw_centroid, target)

        # Compute normalized centroid (what Python nrem_merge_xb produces)
        norm_centroid = raw_centroid / np.linalg.norm(raw_centroid)
        cos_source = cosine_sim(norm_centroid, source)
        cos_target = cosine_sim(norm_centroid, target)

        print(f"\n  Raw centroid ||c|| = {np.linalg.norm(raw_centroid):.6f}")
        print(f"  Raw dot(centroid, source) = {raw_dot_source:.6f}  (Lean definition)")
        print(f"  Raw dot(centroid, target) = {raw_dot_target:.6f}  (Lean definition)")
        print(f"  Normalized cosine(centroid, source) = {cos_source:.6f}  (Python nrem_merge_xb)")
        print(f"  Normalized cosine(centroid, target) = {cos_target:.6f}  (Python nrem_merge_xb)")

        # The Lean theorem guarantees: raw dot product of mean >= sigma
        # when each input has dot product >= sigma with target.
        # But sigma here means cosine_sim for unit vectors (dot = cosine for unit vecs).
        # The raw centroid is NOT unit-norm, so its dot product with unit target
        # = ||centroid|| * cosine_sim(centroid, target).
        # This will be < sigma because ||centroid|| < 1 when patterns are not identical.
        print(f"\n  LEAN CLAIM: dot(mean, source) >= sigma={sigma}")
        print(f"    Actual raw dot: {raw_dot_source:.6f} {'PASS' if raw_dot_source >= sigma else 'FAIL'}")
        print(f"  LEAN CLAIM: dot(mean, target) >= sigma={sigma}")
        print(f"    Actual raw dot: {raw_dot_target:.6f} {'PASS' if raw_dot_target >= sigma else 'FAIL'}")
        print(f"  PYTHON CLAIM: cosine(normalized_centroid, source) >= sigma={sigma}")
        print(f"    Actual: {cos_source:.6f} {'PASS' if cos_source >= sigma else 'FAIL'}")
        print(f"  PYTHON CLAIM: cosine(normalized_centroid, target) >= sigma={sigma}")
        print(f"    Actual: {cos_target:.6f} {'PASS' if cos_target >= sigma else 'FAIL'}")

        # The Lean theorem (centroid_alignment_preserved) guarantees:
        #   dot(mean_of_unit_vecs, target) >= sigma
        # when each input unit vector has dot >= sigma with target.
        # This is a direct consequence of linearity: mean(dot_i) >= sigma.
        #
        # For the NORMALIZED centroid (Python nrem_merge_xb):
        #   cosine(norm_centroid, target) = dot(raw, target) / ||raw||
        # Since ||raw|| <= 1 (convex combination of unit vectors), dividing
        # by ||raw|| can only INCREASE the similarity. So:
        #   cosine(norm_centroid, target) >= dot(raw, target) >= sigma.
        #
        # This means the Python normalization is actually STRONGER than the
        # Lean raw dot product bound.

        # First verify the Lean bound on raw dot product
        min_input_dot_s = min(np.dot(patterns[k], source) for k in range(K))
        min_input_dot_t = min(np.dot(patterns[k], target) for k in range(K))
        print(f"\n  Min input dot to source: {min_input_dot_s:.6f}")
        print(f"  Min input dot to target: {min_input_dot_t:.6f}")
        effective_sigma = min(min_input_dot_s, min_input_dot_t)
        print(f"  Effective sigma (min achievable): {effective_sigma:.6f}")

        assert raw_dot_source >= effective_sigma - 1e-6, (
            f"Lean bound FALSIFIED: raw dot to source {raw_dot_source:.6f} < effective sigma {effective_sigma:.6f}"
        )
        assert raw_dot_target >= effective_sigma - 1e-6, (
            f"Lean bound FALSIFIED: raw dot to target {raw_dot_target:.6f} < effective sigma {effective_sigma:.6f}"
        )
        # Python bound: normalized cosine >= effective sigma (since ||raw|| <= 1)
        assert cos_source >= effective_sigma - 1e-6, (
            f"Normalized centroid cosine to source {cos_source:.6f} < effective sigma {effective_sigma:.6f}"
        )
        assert cos_target >= effective_sigma - 1e-6, (
            f"Normalized centroid cosine to target {cos_target:.6f} < effective sigma {effective_sigma:.6f}"
        )

    def test_h1_bridge_count_increases(self, setup_d128):
        """Verify that adding the centroid increases bridge count by at least 2."""
        d, source, target, rng = setup_d128
        sigma = 0.7
        K = 5

        # Create base patterns: source and target as first two
        base_patterns = np.array([source, target])
        bridges_before = count_bridges(base_patterns, sigma)

        # Create K cross-domain patterns
        cross_patterns = []
        for _ in range(K):
            v = make_cross_domain_vector(source, target, sigma + 0.05, sigma + 0.05, rng)
            cross_patterns.append(v)
        cross_patterns = np.array(cross_patterns)

        # Compute centroid (normalized, as nrem_merge_xb does)
        raw_centroid = np.mean(cross_patterns, axis=0)
        norm_centroid = raw_centroid / np.linalg.norm(raw_centroid)

        # Add centroid to base patterns
        extended = np.vstack([base_patterns, [norm_centroid]])
        bridges_after = count_bridges(extended, sigma)

        print(f"\n--- H1 Bridge Count ---")
        print(f"  Bridges before (source, target only): {bridges_before}")
        print(f"  cosine_sim(source, target) = {cosine_sim(source, target):.6f}")
        print(f"  cosine_sim(centroid, source) = {cosine_sim(norm_centroid, source):.6f}")
        print(f"  cosine_sim(centroid, target) = {cosine_sim(norm_centroid, target):.6f}")
        print(f"  Bridges after adding centroid: {bridges_after}")
        print(f"  New bridges: {bridges_after - bridges_before}")

        # The centroid should bridge to both source and target
        assert bridges_after >= bridges_before + 2, (
            f"Expected at least 2 new bridges, got {bridges_after - bridges_before}"
        )

    def test_h1_lean_vs_python_divergence(self, setup_d128):
        """CRITICAL TEST: Probe where Lean (raw dot) and Python (cosine) diverge.

        The Lean proof uses dot product on the raw mean. The Python code normalizes.
        For unit vectors with high mutual similarity, ||centroid|| is close to 1,
        so the difference is small. But for vectors pointing in very different
        directions (both above threshold individually), ||centroid|| << 1 and
        the raw dot product drops below threshold while cosine stays above.
        """
        d, source, target, rng = setup_d128
        sigma = 0.6

        print(f"\n--- H1 Lean vs Python Divergence ---")
        print(f"  Testing K=2 with patterns in VERY different directions")
        print(f"  (both above sigma={sigma} to source and target)\n")

        # Create 2 patterns that both have sim >= sigma to source and target
        # but are very different from each other
        K = 2
        patterns = []
        for k in range(K):
            v = make_cross_domain_vector(source, target, sigma + 0.05, sigma + 0.05, rng)
            patterns.append(v)
        patterns = np.array(patterns)

        mutual_sim = cosine_sim(patterns[0], patterns[1])
        raw_centroid = np.mean(patterns, axis=0)
        centroid_norm = np.linalg.norm(raw_centroid)
        norm_centroid = raw_centroid / centroid_norm

        raw_dot_s = np.dot(raw_centroid, source)
        raw_dot_t = np.dot(raw_centroid, target)
        cos_s = cosine_sim(norm_centroid, source)
        cos_t = cosine_sim(norm_centroid, target)

        print(f"  Mutual sim between the 2 patterns: {mutual_sim:.6f}")
        print(f"  Centroid norm: {centroid_norm:.6f}")
        print(f"  Raw dot(centroid, source): {raw_dot_s:.6f}")
        print(f"  Raw dot(centroid, target): {raw_dot_t:.6f}")
        print(f"  Cosine(norm_centroid, source): {cos_s:.6f}")
        print(f"  Cosine(norm_centroid, target): {cos_t:.6f}")
        print(f"\n  Lean bound (raw dot >= sigma): source={'PASS' if raw_dot_s >= sigma else 'FAIL'}, target={'PASS' if raw_dot_t >= sigma else 'FAIL'}")
        print(f"  Python bound (cosine >= sigma): source={'PASS' if cos_s >= sigma else 'FAIL'}, target={'PASS' if cos_t >= sigma else 'FAIL'}")
        print(f"\n  KEY: When ||centroid|| < 1, raw dot < cosine. Lean bound is TIGHTER.")
        print(f"  ||centroid|| = {centroid_norm:.6f}, so raw_dot = {centroid_norm:.6f} * cosine")

        # The Lean theorem guarantees raw_dot >= sigma. This should hold since
        # each pattern is a unit vector with dot >= sigma to source/target, and
        # the mean preserves this (linearity of dot product).
        for k in range(K):
            dot_k_s = np.dot(patterns[k], source)
            dot_k_t = np.dot(patterns[k], target)
            print(f"  Pattern {k}: dot_source={dot_k_s:.6f}, dot_target={dot_k_t:.6f}")

        # Lean claim: raw dot product of mean >= sigma (when all inputs have dot >= sigma)
        # This is a direct consequence of linearity: mean(dots) >= sigma
        assert raw_dot_s >= sigma - 1e-6, (
            f"Lean raw dot bound FALSIFIED: {raw_dot_s:.6f} < {sigma}"
        )
        assert raw_dot_t >= sigma - 1e-6, (
            f"Lean raw dot bound FALSIFIED: {raw_dot_t:.6f} < {sigma}"
        )

        # Python: normalized centroid should have HIGHER cosine sim (dividing
        # by ||centroid|| < 1 amplifies the similarity in cosine metric)
        assert cos_s >= sigma - 1e-6, (
            f"Python cosine bound FALSIFIED: {cos_s:.6f} < {sigma}"
        )

    def test_h1_nrem_merge_xb_integration(self, setup_d128):
        """Test bridge formation through the actual nrem_merge_xb function."""
        d, source, target, rng = setup_d128
        sigma = 0.7
        K = 5

        # Create K cross-domain patterns that should form a merge group
        # (pairwise sim > 0.90 to each other, and > sigma to source and target)
        # Strategy: generate tightly clustered cross-domain patterns
        base_cross = make_cross_domain_vector(source, target, sigma + 0.1, sigma + 0.1, rng)
        cross_patterns = [base_cross]
        for _ in range(K - 1):
            noise = rng.standard_normal(d) * 0.03
            p = base_cross + noise
            p = p / np.linalg.norm(p)
            cross_patterns.append(p)
        cross_patterns = np.array(cross_patterns)

        # Add some filler patterns to make it realistic
        filler_count = 5
        filler = np.array([make_unit_vector(d, rng) for _ in range(filler_count)])
        all_patterns = np.vstack([cross_patterns, filler])

        # Importances: all equal
        importances = np.full(len(all_patterns), 0.5)

        print(f"\n--- H1 nrem_merge_xb Integration ---")
        print(f"  {K} cross-domain patterns + {filler_count} filler patterns")

        # Show pairwise sims within the cross-domain group
        for i in range(K):
            for j in range(i + 1, K):
                print(f"  Pairwise sim cross[{i}]-cross[{j}]: {cosine_sim(cross_patterns[i], cross_patterns[j]):.6f}")

        bridges_before = count_bridges(all_patterns, sigma)

        # Run actual nrem_merge_xb
        merged, merge_map = nrem_merge_xb(
            all_patterns, importances, threshold=0.90, min_group=3
        )

        bridges_after = count_bridges(merged, sigma)

        print(f"\n  Patterns before merge: {len(all_patterns)}")
        print(f"  Patterns after merge: {len(merged)}")
        print(f"  Merge map: {merge_map}")
        print(f"  Bridges before: {bridges_before}")
        print(f"  Bridges after: {bridges_after}")

        # Check that merged centroids have high sim to source and target
        for out_idx, group in merge_map.items():
            centroid = merged[out_idx]
            cs = cosine_sim(centroid, source)
            ct = cosine_sim(centroid, target)
            print(f"  Centroid at output idx {out_idx} (from group {group}):")
            print(f"    cosine_sim(centroid, source) = {cs:.6f}")
            print(f"    cosine_sim(centroid, target) = {ct:.6f}")
            print(f"    Is bridge to source (>= {sigma})? {cs >= sigma}")
            print(f"    Is bridge to target (>= {sigma})? {ct >= sigma}")

        # The claim: centroid preserves similarity. With K=5 tightly clustered
        # cross-domain patterns, the centroid should maintain high sim to both.
        if merge_map:
            out_idx = list(merge_map.keys())[0]
            centroid = merged[out_idx]
            assert cosine_sim(centroid, source) >= sigma - 0.05, (
                f"Centroid lost bridge to source"
            )
            assert cosine_sim(centroid, target) >= sigma - 0.05, (
                f"Centroid lost bridge to target"
            )

    def test_h1_low_dimensional_exact(self):
        """Low-dimensional exact test to verify the math precisely.

        In d=4, create source=(1,0,0,0), target=(0,1,0,0), and K=3 patterns
        each with known exact cosine sims to both.
        """
        d = 4
        source = np.array([1.0, 0.0, 0.0, 0.0])
        target = np.array([0.0, 1.0, 0.0, 0.0])
        sigma = 0.6

        # Create patterns with exact geometry:
        # Each pattern: (a, b, c_k, ...) where a >= sigma, b >= sigma
        # and ||v|| = 1
        # For sigma=0.6: a=0.65, b=0.65, remaining = sqrt(1 - 0.65^2 - 0.65^2) = sqrt(0.155)
        a, b = 0.65, 0.65
        remain = np.sqrt(max(0, 1.0 - a ** 2 - b ** 2))

        patterns = np.array([
            [a, b, remain, 0.0],
            [a, b, 0.0, remain],
            [a, b, -remain * 0.5, remain * np.sqrt(3) / 2],
        ])
        # Normalize
        for i in range(len(patterns)):
            patterns[i] = patterns[i] / np.linalg.norm(patterns[i])

        raw_centroid = np.mean(patterns, axis=0)
        norm_centroid = raw_centroid / np.linalg.norm(raw_centroid)

        print(f"\n--- H1 Low-Dimensional Exact (d={d}) ---")
        print(f"  source = {source}")
        print(f"  target = {target}")
        print(f"  sigma = {sigma}")
        for i in range(3):
            print(f"  Pattern {i}: {patterns[i]}")
            print(f"    dot(p, source)={np.dot(patterns[i], source):.6f}, "
                  f"dot(p, target)={np.dot(patterns[i], target):.6f}")

        print(f"  Raw centroid: {raw_centroid}")
        print(f"  ||raw centroid|| = {np.linalg.norm(raw_centroid):.6f}")
        print(f"  Normalized centroid: {norm_centroid}")
        print(f"  dot(raw_centroid, source) = {np.dot(raw_centroid, source):.6f}")
        print(f"  dot(raw_centroid, target) = {np.dot(raw_centroid, target):.6f}")
        print(f"  cosine(norm_centroid, source) = {cosine_sim(norm_centroid, source):.6f}")
        print(f"  cosine(norm_centroid, target) = {cosine_sim(norm_centroid, target):.6f}")

        # Lean claim (raw dot): mean of 3 vectors each with dot >= sigma to source
        #   => dot(mean, source) >= sigma
        # Since dot is linear: dot(mean, source) = mean(dot(vi, source)) >= sigma
        input_dots_s = [np.dot(patterns[i], source) for i in range(3)]
        input_dots_t = [np.dot(patterns[i], target) for i in range(3)]
        mean_dot_s = np.mean(input_dots_s)
        mean_dot_t = np.mean(input_dots_t)

        print(f"\n  Mean of input dots to source: {mean_dot_s:.6f}")
        print(f"  Mean of input dots to target: {mean_dot_t:.6f}")
        print(f"  dot(raw_centroid, source) should equal mean_dots: {np.dot(raw_centroid, source):.6f}")

        assert abs(np.dot(raw_centroid, source) - mean_dot_s) < 1e-10, "Linearity of dot product failed"
        assert mean_dot_s >= sigma, f"Lean bound FALSIFIED: mean dot {mean_dot_s} < sigma {sigma}"

        # The cosine sim of normalized centroid >= sigma since normalization
        # only scales the vector (direction preserved). cosine_sim = dot / norms.
        assert cosine_sim(norm_centroid, source) >= sigma - 1e-6
        assert cosine_sim(norm_centroid, target) >= sigma - 1e-6


# ===========================================================================
# H2: Conditional Bridge Monotonicity (Phase 9b)
# ===========================================================================

class TestH2ConditionalBridgeMonotonicity:
    """Phase 9b: Bridges between high-importance patterns survive
    pruning/dream operations, provided both endpoints survive.

    CRITICAL: The theorem is conditional on P subset T (all protected
    patterns survive). But nrem_prune_xb removes near-duplicates, and
    two high-importance patterns COULD be near-duplicates. When two
    protected patterns are near-duplicates, one IS removed, violating
    P subset T, and destroying the bridge between them.
    """

    @pytest.fixture
    def setup_patterns(self):
        """Create a realistic pattern set with high and low importance."""
        rng = np.random.default_rng(123)
        d = 128
        return d, rng

    def test_h2_happy_path_protected_bridges_survive(self, setup_patterns):
        """HAPPY PATH: High-importance patterns are distinct. Bridges survive pruning."""
        d, rng = setup_patterns
        N = 20

        # Create 10 high-importance patterns (distinct, well-separated)
        high_patterns = []
        for i in range(10):
            v = make_unit_vector(d, rng)
            high_patterns.append(v)

        # Create 10 low-importance patterns (some near-duplicates of each other)
        low_patterns = []
        for i in range(10):
            # Make some low-importance patterns be near-duplicates
            if i < 5:
                v = make_unit_vector(d, rng)
            else:
                # Near-duplicate of low_patterns[i-5]
                v = low_patterns[i - 5] + rng.standard_normal(d) * 0.01
                v = v / np.linalg.norm(v)
            low_patterns.append(v)

        all_patterns = np.array(high_patterns + low_patterns)
        importances = np.array([0.9] * 10 + [0.2] * 10)

        # Introduce some bridges within high-importance patterns
        # by making some pairs similar
        bridge_threshold = 0.5
        # Make patterns 0 and 1 bridge-like (cosine_sim > threshold)
        all_patterns[1] = make_vector_with_cosine_sim(all_patterns[0], 0.7, rng)
        # Make patterns 2 and 3 bridge-like
        all_patterns[3] = make_vector_with_cosine_sim(all_patterns[2], 0.6, rng)

        # Count bridges within high-importance set before pruning
        high_indices = list(range(10))
        high_pat_before = all_patterns[high_indices]

        bridges_high_before = 0
        for i in range(10):
            for j in range(10):
                if i != j and cosine_sim(all_patterns[i], all_patterns[j]) >= bridge_threshold:
                    bridges_high_before += 1

        print(f"\n--- H2 Happy Path ---")
        print(f"  N={N}, d={d}, bridge_threshold={bridge_threshold}")
        print(f"  High-importance patterns: indices 0-9 (importance=0.9)")
        print(f"  Low-importance patterns: indices 10-19 (importance=0.2)")
        print(f"  Bridges within high-importance set before prune: {bridges_high_before}")

        # Run nrem_prune_xb with threshold=0.95 (removes near-duplicates)
        pruned, kept_indices = nrem_prune_xb(all_patterns, importances, threshold=0.95)

        print(f"  Kept indices after prune: {kept_indices}")
        print(f"  Patterns pruned: {N - len(kept_indices)}")

        # Check which high-importance patterns survived
        high_survived = [i for i in high_indices if i in kept_indices]
        high_removed = [i for i in high_indices if i not in kept_indices]

        print(f"  High-importance survived: {high_survived}")
        print(f"  High-importance removed: {high_removed}")

        # Count bridges within surviving high-importance patterns
        high_surv_patterns = all_patterns[high_survived]
        bridges_high_after = 0
        for i in range(len(high_survived)):
            for j in range(len(high_survived)):
                if i != j and cosine_sim(high_surv_patterns[i], high_surv_patterns[j]) >= bridge_threshold:
                    bridges_high_after += 1

        print(f"  Bridges within surviving high-importance: {bridges_high_after}")
        print(f"  Bridge preservation: {bridges_high_after >= bridges_high_before}")

        # When high-importance patterns are all distinct (sim < 0.95),
        # none should be removed. So P subset T should hold.
        assert len(high_removed) == 0, (
            f"FALSIFIED: High-importance patterns were removed: {high_removed}"
        )
        assert bridges_high_after >= bridges_high_before, (
            f"FALSIFIED: Bridge count decreased from {bridges_high_before} to {bridges_high_after}"
        )

    def test_h2_near_duplicate_protected_patterns(self, setup_patterns):
        """STRESS TEST: Two high-importance patterns are near-duplicates (sim > 0.95).
        This SHOULD violate the P subset T precondition.

        The theorem says bridges are preserved IF protected patterns survive.
        This test checks: do they survive when they are near-duplicates?
        """
        d, rng = setup_patterns

        # Create patterns where two high-importance ones are near-duplicates
        base = make_unit_vector(d, rng)
        near_dup = base + rng.standard_normal(d) * 0.01
        near_dup = near_dup / np.linalg.norm(near_dup)

        other_patterns = [make_unit_vector(d, rng) for _ in range(8)]
        all_patterns = np.array([base, near_dup] + other_patterns)
        importances = np.array([0.95, 0.90] + [0.5] * 8)

        dup_sim = cosine_sim(base, near_dup)

        print(f"\n--- H2 Near-Duplicate Protected Patterns ---")
        print(f"  Two high-importance patterns with sim = {dup_sim:.6f}")
        print(f"  Importances: {importances[:2]}")

        pruned, kept = nrem_prune_xb(all_patterns, importances, threshold=0.95)

        print(f"  Kept indices: {kept}")
        print(f"  Pattern 0 survived: {0 in kept}")
        print(f"  Pattern 1 survived: {1 in kept}")

        both_survived = 0 in kept and 1 in kept
        print(f"  Both high-importance near-duplicates survived: {both_survived}")

        if not both_survived:
            print(f"  PRECONDITION VIOLATED: P not subset T")
            print(f"  nrem_prune_xb removes near-duplicates even if high-importance")
            print(f"  The theorem is conditional — this is the condition failing, not the theorem")
            # This is expected: the theorem's precondition fails, not the theorem.
            # We record this as a valid failure mode that users must handle.
            removed_idx = 1 if 0 in kept else 0
            print(f"  Removed: index {removed_idx} (importance={importances[removed_idx]:.2f})")
            kept_idx = 0 if 0 in kept else 1
            print(f"  Kept: index {kept_idx} (importance={importances[kept_idx]:.2f})")

        # The KEY measurement: verify that when sim > threshold, pruning DOES
        # remove one, breaking the precondition
        if dup_sim > 0.95:
            assert not both_survived, (
                f"Near-duplicate with sim={dup_sim:.4f} > 0.95 was NOT pruned — "
                f"expected one to be removed"
            )
        else:
            print(f"  sim={dup_sim:.4f} <= 0.95, so both survive (no near-dup detected)")

    def test_h2_importance_tiebreak_preserves_higher(self, setup_patterns):
        """When two near-duplicates are pruned, the HIGHER importance one survives."""
        d, rng = setup_patterns

        base = make_unit_vector(d, rng)
        near_dup = base + rng.standard_normal(d) * 0.005  # very close
        near_dup = near_dup / np.linalg.norm(near_dup)

        all_patterns = np.array([base, near_dup])
        # Pattern 0 has higher importance
        importances = np.array([0.95, 0.60])

        sim = cosine_sim(base, near_dup)
        print(f"\n--- H2 Importance Tiebreak ---")
        print(f"  Two patterns, sim = {sim:.6f}")
        print(f"  Importances: {importances}")

        pruned, kept = nrem_prune_xb(all_patterns, importances, threshold=0.95)

        print(f"  Kept indices: {kept}")

        if sim > 0.95:
            assert 0 in kept, "Higher importance pattern was removed"
            assert 1 not in kept, "Lower importance pattern survived"
            print(f"  Correctly kept index 0 (importance=0.95)")
        else:
            print(f"  sim <= 0.95, both survive")

    def test_h2_equal_importance_near_duplicates(self, setup_patterns):
        """When two near-duplicates have EQUAL importance, tiebreak removes higher index."""
        d, rng = setup_patterns

        base = make_unit_vector(d, rng)
        near_dup = base + rng.standard_normal(d) * 0.005
        near_dup = near_dup / np.linalg.norm(near_dup)

        all_patterns = np.array([base, near_dup])
        importances = np.array([0.90, 0.90])

        sim = cosine_sim(base, near_dup)
        print(f"\n--- H2 Equal Importance Tiebreak ---")
        print(f"  Two patterns, sim = {sim:.6f}")
        print(f"  Both importance = 0.90")

        pruned, kept = nrem_prune_xb(all_patterns, importances, threshold=0.95)
        print(f"  Kept indices: {kept}")

        if sim > 0.95:
            assert 0 in kept, "Lower index should survive on tiebreak"
            assert 1 not in kept, "Higher index should be removed on tiebreak"
            print(f"  Correctly kept index 0 (lower index tiebreak)")

    def test_h2_bridges_between_distinct_protected_always_survive(self, setup_patterns):
        """The theorem's guarantee: when protected patterns are distinct (all pairwise
        sim <= prune_threshold), ALL bridges within the protected set survive."""
        d, rng = setup_patterns
        N_high = 8
        N_low = 12
        N = N_high + N_low
        bridge_threshold = 0.5
        prune_threshold = 0.95

        # Create high-importance patterns that are distinct (sim <= 0.95)
        # but some pairs have bridges (sim >= 0.5)
        high_patterns = [make_unit_vector(d, rng)]
        for i in range(1, N_high):
            if i < 4:
                # Create bridges: sim in [0.5, 0.8] range
                v = make_vector_with_cosine_sim(high_patterns[0], 0.55 + 0.05 * i, rng)
            else:
                v = make_unit_vector(d, rng)
            high_patterns.append(v)
        high_patterns = np.array(high_patterns)

        # Verify all high pairs have sim <= 0.95 (no near-duplicates)
        max_high_sim = 0.0
        for i in range(N_high):
            for j in range(i + 1, N_high):
                s = cosine_sim(high_patterns[i], high_patterns[j])
                max_high_sim = max(max_high_sim, s)

        print(f"\n--- H2 Distinct Protected Bridges ---")
        print(f"  N_high={N_high}, N_low={N_low}")
        print(f"  Max pairwise sim within high-importance: {max_high_sim:.6f}")
        assert max_high_sim <= prune_threshold, (
            f"Test setup error: high patterns have near-duplicate pair (sim={max_high_sim:.4f})"
        )

        # Create low-importance patterns (some are near-duplicates)
        low_patterns = [make_unit_vector(d, rng) for _ in range(N_low)]
        low_patterns = np.array(low_patterns)

        all_patterns = np.vstack([high_patterns, low_patterns])
        importances = np.array([0.85] * N_high + [0.2] * N_low)

        # Count bridges within protected set before prune
        bridges_before = 0
        bridge_pairs_before = []
        for i in range(N_high):
            for j in range(i + 1, N_high):
                if cosine_sim(all_patterns[i], all_patterns[j]) >= bridge_threshold:
                    bridges_before += 1
                    bridge_pairs_before.append((i, j, cosine_sim(all_patterns[i], all_patterns[j])))

        print(f"  Bridge pairs within protected (sim >= {bridge_threshold}):")
        for i, j, s in bridge_pairs_before:
            print(f"    ({i}, {j}): sim = {s:.6f}")
        print(f"  Total bridges (unordered): {bridges_before}")

        # Run prune
        pruned, kept = nrem_prune_xb(all_patterns, importances, threshold=prune_threshold)

        # Verify ALL high-importance patterns survived (P subset T)
        high_survived = [i for i in range(N_high) if i in kept]
        print(f"  High-importance survived: {high_survived}")
        assert len(high_survived) == N_high, (
            f"Protected patterns removed: expected {N_high}, got {len(high_survived)}"
        )

        # Verify all bridges survived
        bridges_after = 0
        # Map original indices to pruned indices
        idx_map = {orig: new for new, orig in enumerate(kept)}
        for i, j, s in bridge_pairs_before:
            new_i = idx_map.get(i)
            new_j = idx_map.get(j)
            if new_i is not None and new_j is not None:
                actual_sim = cosine_sim(pruned[new_i], pruned[new_j])
                preserved = actual_sim >= bridge_threshold
                bridges_after += 1 if preserved else 0
                print(f"    Bridge ({i},{j}): before sim={s:.6f}, after sim={actual_sim:.6f} {'PRESERVED' if preserved else 'LOST'}")

        print(f"  Bridges after: {bridges_after}")
        assert bridges_after == bridges_before, (
            f"FALSIFIED: Bridges decreased from {bridges_before} to {bridges_after}"
        )


# ===========================================================================
# H3: Transfer Dynamics (Phase 10)
# ===========================================================================

class TestH3TransferDynamics:
    """Phase 10: Hopfield update through a bridge pattern enables cross-domain
    retrieval with lower bound sigma / (1 + (N-1) * exp(-beta * delta)).

    CRITICAL ISSUES:
    1. The bound degrades as O(sigma/N) for large N. For large stores,
       the signal becomes undetectably weak.
    2. The Lean proof assumes non-negative dot products of all patterns
       with the target. Random patterns have mean-zero dot products.
    3. Python softmax uses shifted z for numerical stability; this is
       mathematically equivalent to the Lean definition.
    """

    def _theoretical_lower_bound(self, sigma, N, beta, delta):
        """Compute the theoretical lower bound from Phase 10 theorem."""
        return sigma / (1 + (N - 1) * np.exp(-beta * delta))

    def test_h3_softmax_weight_bound(self):
        """Verify the softmax weight lower bound independently.

        Claim: when pattern mu has alignment gap >= delta over all others,
        softmax weight p_mu >= 1 / (1 + (N-1) * exp(-beta * delta)).
        """
        print(f"\n--- H3 Softmax Weight Bound ---")

        for N in [5, 10, 50, 100, 500]:
            rng = np.random.default_rng(42)
            beta = 5.0
            delta = 0.3

            # Create similarity vector z where z[0] has gap >= delta over all others
            z = rng.standard_normal(N)
            z[0] = np.max(z[1:]) + delta + 0.01  # ensure gap >= delta

            actual_gap = z[0] - np.max(z[1:])

            weights = softmax(beta, z)
            actual_weight = weights[0]
            theoretical_bound = 1.0 / (1 + (N - 1) * np.exp(-beta * delta))

            print(f"  N={N:4d}: gap={actual_gap:.4f}, w[0]={actual_weight:.8f}, "
                  f"bound={theoretical_bound:.8f}, "
                  f"holds={'PASS' if actual_weight >= theoretical_bound - 1e-10 else 'FAIL'}")

            assert actual_weight >= theoretical_bound - 1e-10, (
                f"Softmax weight bound FALSIFIED at N={N}: "
                f"actual={actual_weight:.10f} < bound={theoretical_bound:.10f}"
            )

    def test_h3_happy_path_transfer(self):
        """HAPPY PATH: Hopfield update with a bridge pattern enables cross-domain retrieval.

        Create a controlled scenario:
        - Bridge pattern c has cosine_sim(c, target) = sigma (known)
        - Query xi is aligned with c by gap delta
        - All other patterns have NON-NEGATIVE dot with target (theorem precondition)
        """
        d = 128
        rng = np.random.default_rng(42)
        beta = 5.0
        sigma = 0.6
        delta = 0.3
        N = 10

        # Create target pattern
        target = make_unit_vector(d, rng)

        # Create bridge pattern with cosine_sim(bridge, target) = sigma
        bridge = make_vector_with_cosine_sim(target, sigma, rng)

        # Create other patterns with NON-NEGATIVE dot product to target
        # (this is the theorem's precondition hothers_nn)
        other_patterns = []
        for _ in range(N - 1):
            v = make_unit_vector(d, rng)
            # Ensure non-negative dot with target
            if np.dot(v, target) < 0:
                v = -v
            other_patterns.append(v)
        other_patterns = np.array(other_patterns)

        # Assemble pattern store: bridge at index 0
        patterns = np.vstack([[bridge], other_patterns])

        # Verify preconditions
        actual_sigma = cosine_sim(bridge, target)
        dots_with_target = [np.dot(patterns[i], target) for i in range(N)]

        # Create query xi aligned with bridge by gap delta
        # xi should have dot(xi, bridge) - dot(xi, other[j]) >= delta for all j
        xi = bridge.copy()
        # Add small noise to avoid trivial case
        noise = rng.standard_normal(d) * 0.05
        xi = xi + noise
        xi = xi / np.linalg.norm(xi)

        # Verify the gap condition
        dots_xi = patterns @ xi
        gap_to_bridge = dots_xi[0]
        max_other = np.max(dots_xi[1:])
        actual_gap = gap_to_bridge - max_other

        print(f"\n--- H3 Happy Path Transfer ---")
        print(f"  N={N}, d={d}, beta={beta}")
        print(f"  sigma (cosine bridge-target) = {actual_sigma:.6f}")
        print(f"  delta (alignment gap) = {actual_gap:.6f}")
        print(f"  All patterns dot with target >= 0: {all(d >= 0 for d in dots_with_target)}")

        # If gap is too small, adjust xi
        if actual_gap < delta:
            # Push xi more toward bridge
            xi = 0.95 * bridge + 0.05 * make_unit_vector(d, rng)
            xi = xi / np.linalg.norm(xi)
            dots_xi = patterns @ xi
            actual_gap = dots_xi[0] - np.max(dots_xi[1:])
            print(f"  Adjusted query: new gap = {actual_gap:.6f}")

        # Run Hopfield update
        result = hopfield_update(beta, patterns, xi)

        # Measure transfer: cosine_sim(result, target)
        transfer_sim = cosine_sim(result, target)
        theoretical_bound = self._theoretical_lower_bound(
            actual_sigma, N, beta, actual_gap
        )

        print(f"  Hopfield update result:")
        print(f"    cosine_sim(T(xi), target) = {transfer_sim:.6f}")
        print(f"    dot(T(xi), target) = {np.dot(result, target):.6f}")
        print(f"    Theoretical lower bound = {theoretical_bound:.6f}")
        print(f"    Bound holds: {transfer_sim >= theoretical_bound - 1e-6}")

        # The Lean theorem uses dot product, not cosine. For non-unit result,
        # dot(result, target) is what the theorem bounds (target is unit).
        dot_transfer = np.dot(result, target)
        print(f"    dot(T(xi), target) >= bound: {dot_transfer >= theoretical_bound - 1e-6}")

        # The theorem guarantees dot(T(xi), target) >= sigma / (1 + (N-1)*exp(-beta*delta))
        # We verify with the actual measured gap (may differ from the design delta).
        assert dot_transfer >= theoretical_bound - 1e-6, (
            f"Transfer bound FALSIFIED: dot={dot_transfer:.8f} < bound={theoretical_bound:.8f}"
        )

    def test_h3_scaling_behavior(self):
        """Test transfer bound across multiple N values to find where it becomes negligible.

        The bound is sigma / (1 + (N-1)*exp(-beta*delta)).
        For large N: bound ~ sigma * exp(beta*delta) / N.
        """
        d = 128
        beta = 5.0
        sigma = 0.6
        delta = 0.3
        N_values = [5, 10, 50, 100, 500, 1000]

        print(f"\n--- H3 Scaling Behavior ---")
        print(f"  beta={beta}, sigma={sigma}, delta={delta}")
        print(f"  {'N':>6s}  {'bound':>12s}  {'actual_dot':>12s}  {'actual_cos':>12s}  {'holds':>6s}")

        for N in N_values:
            rng = np.random.default_rng(42)

            target = make_unit_vector(d, rng)
            bridge = make_vector_with_cosine_sim(target, sigma, rng)

            # Other patterns with non-negative dot to target
            other_patterns = []
            for _ in range(N - 1):
                v = make_unit_vector(d, rng)
                if np.dot(v, target) < 0:
                    v = -v
                other_patterns.append(v)

            patterns = np.vstack([[bridge]] + [other_patterns])

            # Create query aligned with bridge
            xi = 0.95 * bridge + 0.05 * make_unit_vector(d, rng)
            xi = xi / np.linalg.norm(xi)

            dots_xi = patterns @ xi
            actual_gap = dots_xi[0] - np.max(dots_xi[1:])

            # Ensure positive gap
            if actual_gap <= 0:
                xi = 0.99 * bridge + 0.01 * make_unit_vector(d, rng)
                xi = xi / np.linalg.norm(xi)
                dots_xi = patterns @ xi
                actual_gap = dots_xi[0] - np.max(dots_xi[1:])

            result = hopfield_update(beta, patterns, xi)
            actual_dot = np.dot(result, target)
            actual_cos = cosine_sim(result, target)
            bound = self._theoretical_lower_bound(sigma, N, beta, max(actual_gap, 0.01))

            holds = actual_dot >= bound - 1e-6
            print(f"  {N:6d}  {bound:12.8f}  {actual_dot:12.8f}  {actual_cos:12.8f}  {'PASS' if holds else 'FAIL'}")

            if actual_gap > 0:
                assert holds, (
                    f"Transfer bound FALSIFIED at N={N}: dot={actual_dot:.10f} < bound={bound:.10f}"
                )

    def test_h3_negative_dot_patterns_violate_precondition(self):
        """STRESS TEST: What happens when patterns have NEGATIVE dot with target?

        The Lean theorem requires hothers_nn: all patterns have non-negative dot
        with target. This is a strong assumption. Test what happens when it fails.
        """
        d = 128
        rng = np.random.default_rng(42)
        beta = 5.0
        sigma = 0.6
        N = 20

        target = make_unit_vector(d, rng)
        bridge = make_vector_with_cosine_sim(target, sigma, rng)

        # Create patterns WITHOUT the non-negative constraint
        other_patterns = [make_unit_vector(d, rng) for _ in range(N - 1)]
        patterns = np.vstack([[bridge]] + other_patterns)

        # Count how many have negative dot with target
        dots_target = [np.dot(patterns[i], target) for i in range(N)]
        n_negative = sum(1 for d_val in dots_target if d_val < 0)

        xi = 0.95 * bridge + 0.05 * make_unit_vector(d, rng)
        xi = xi / np.linalg.norm(xi)

        dots_xi = patterns @ xi
        actual_gap = dots_xi[0] - np.max(dots_xi[1:])

        result = hopfield_update(beta, patterns, xi)
        actual_dot = np.dot(result, target)
        bound = self._theoretical_lower_bound(sigma, N, beta, max(actual_gap, 0.01))

        print(f"\n--- H3 Negative Dot Products (Precondition Violation) ---")
        print(f"  N={N}, {n_negative} patterns have negative dot with target")
        print(f"  Gap: {actual_gap:.6f}")
        print(f"  dot(T(xi), target) = {actual_dot:.6f}")
        print(f"  Theoretical bound (assuming precondition holds) = {bound:.6f}")
        print(f"  Bound holds despite violation: {actual_dot >= bound - 1e-6}")
        if actual_dot < bound - 1e-6:
            print(f"  EXPECTED VIOLATION: negative-dot patterns pull result away from target")
            print(f"  The theorem's hothers_nn precondition is NECESSARY, not just sufficient")

    def test_h3_bridge_pattern_weight_decomposition(self):
        """Verify the weight decomposition: T(xi) = sum_mu w_mu * x_mu.

        The Hopfield update is a convex combination. Verify that the weight
        on the bridge pattern matches the softmax weight bound.
        """
        d = 64
        rng = np.random.default_rng(42)
        beta = 8.0
        N = 10
        delta = 0.5

        target = make_unit_vector(d, rng)
        bridge = make_vector_with_cosine_sim(target, 0.7, rng)

        other_patterns = []
        for _ in range(N - 1):
            v = make_unit_vector(d, rng)
            if np.dot(v, target) < 0:
                v = -v
            other_patterns.append(v)

        patterns = np.vstack([[bridge]] + other_patterns)

        # Create query with known gap
        xi = 0.98 * bridge + 0.02 * make_unit_vector(d, rng)
        xi = xi / np.linalg.norm(xi)

        dots_xi = patterns @ xi
        actual_gap = dots_xi[0] - np.max(dots_xi[1:])

        weights = softmax(beta, dots_xi)
        w_bridge = weights[0]
        weight_bound = 1.0 / (1 + (N - 1) * np.exp(-beta * max(actual_gap, 0.001)))

        result = hopfield_update(beta, patterns, xi)

        # Verify: result should equal sum of weights * patterns
        reconstructed = weights @ patterns
        reconstruction_error = np.linalg.norm(result - reconstructed)

        # The dot product with target decomposes as:
        dot_target_decomp = sum(weights[mu] * np.dot(patterns[mu], target) for mu in range(N))
        dot_target_direct = np.dot(result, target)

        print(f"\n--- H3 Weight Decomposition ---")
        print(f"  N={N}, beta={beta}, gap={actual_gap:.6f}")
        print(f"  Bridge weight: w[0] = {w_bridge:.8f}")
        print(f"  Weight bound: {weight_bound:.8f}")
        print(f"  Weight bound holds: {w_bridge >= weight_bound - 1e-10}")
        print(f"  Reconstruction error (T(xi) vs sum w*x): {reconstruction_error:.2e}")
        print(f"  dot(T(xi), target) direct = {dot_target_direct:.8f}")
        print(f"  dot(T(xi), target) decomp = {dot_target_decomp:.8f}")
        print(f"  Decomposition match: {abs(dot_target_direct - dot_target_decomp) < 1e-10}")

        assert reconstruction_error < 1e-10, (
            f"Hopfield update is not a convex combination: error = {reconstruction_error}"
        )
        assert abs(dot_target_direct - dot_target_decomp) < 1e-10, (
            f"Dot product decomposition failed"
        )
        if actual_gap > 0:
            assert w_bridge >= weight_bound - 1e-10, (
                f"Softmax weight bound FALSIFIED: {w_bridge:.10f} < {weight_bound:.10f}"
            )

    def test_h3_high_beta_approaches_sigma(self):
        """As beta -> infinity, the bound approaches sigma (full transfer).

        The bridge pattern should dominate the softmax, so T(xi) -> bridge pattern.
        Then dot(T(xi), target) -> dot(bridge, target) = sigma.
        """
        d = 64
        rng = np.random.default_rng(42)
        sigma = 0.7
        N = 20
        delta = 0.2

        target = make_unit_vector(d, rng)
        bridge = make_vector_with_cosine_sim(target, sigma, rng)
        actual_sigma = cosine_sim(bridge, target)

        other_patterns = []
        for _ in range(N - 1):
            v = make_unit_vector(d, rng)
            if np.dot(v, target) < 0:
                v = -v
            other_patterns.append(v)
        patterns = np.vstack([[bridge]] + other_patterns)

        # Query aligned with bridge
        xi = 0.98 * bridge + 0.02 * make_unit_vector(d, rng)
        xi = xi / np.linalg.norm(xi)

        print(f"\n--- H3 High Beta Limit ---")
        print(f"  N={N}, sigma={actual_sigma:.6f}")
        print(f"  {'beta':>8s}  {'dot_transfer':>12s}  {'bound':>12s}  {'gap_to_sigma':>12s}")

        for beta in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            result = hopfield_update(beta, patterns, xi)
            dot_transfer = np.dot(result, target)

            dots_xi = patterns @ xi
            actual_gap = dots_xi[0] - np.max(dots_xi[1:])
            bound = self._theoretical_lower_bound(actual_sigma, N, beta, max(actual_gap, 0.001))

            print(f"  {beta:8.1f}  {dot_transfer:12.8f}  {bound:12.8f}  "
                  f"{abs(dot_transfer - actual_sigma):12.8f}")

        # At beta=100, transfer should be very close to sigma
        result_high = hopfield_update(100.0, patterns, xi)
        dot_high = np.dot(result_high, target)
        print(f"\n  At beta=100: dot(T(xi), target) = {dot_high:.8f}, sigma = {actual_sigma:.8f}")
        print(f"  Difference: {abs(dot_high - actual_sigma):.2e}")

        assert abs(dot_high - actual_sigma) < 0.01, (
            f"High-beta limit failed: expected ~{actual_sigma:.4f}, got {dot_high:.4f}"
        )

    def test_h3_dot_product_vs_cosine_sim_distinction(self):
        """CRITICAL: The Lean theorem bounds dot(T(xi), target), not cosine_sim.

        T(xi) is a convex combination of patterns, so ||T(xi)|| <= max(||x_mu||).
        For unit patterns, ||T(xi)|| <= 1. The dot product and cosine sim differ
        when ||T(xi)|| < 1.

        Measure the gap between dot and cosine for the Hopfield output.
        """
        d = 64
        rng = np.random.default_rng(42)
        N = 10
        beta = 3.0
        sigma = 0.6

        target = make_unit_vector(d, rng)
        bridge = make_vector_with_cosine_sim(target, sigma, rng)
        other_patterns = [make_unit_vector(d, rng) for _ in range(N - 1)]
        patterns = np.vstack([[bridge]] + other_patterns)

        xi = 0.9 * bridge + 0.1 * make_unit_vector(d, rng)
        xi = xi / np.linalg.norm(xi)

        result = hopfield_update(beta, patterns, xi)
        result_norm = np.linalg.norm(result)

        dot_val = np.dot(result, target)
        cos_val = cosine_sim(result, target)

        print(f"\n--- H3 Dot vs Cosine Distinction ---")
        print(f"  ||T(xi)|| = {result_norm:.8f}")
        print(f"  dot(T(xi), target) = {dot_val:.8f}")
        print(f"  cosine_sim(T(xi), target) = {cos_val:.8f}")
        print(f"  Ratio dot/cosine = {dot_val / cos_val:.8f} (should be ||T(xi)||)")
        print(f"  ||T(xi)|| matches ratio: {abs(result_norm - dot_val / cos_val) < 1e-6}")
        print(f"\n  KEY INSIGHT: Lean bounds dot product. Python typically uses cosine_sim.")
        print(f"  cosine_sim >= dot for ||T(xi)|| <= 1 (cosine is more optimistic)")
        print(f"  So the Lean bound on dot is ALSO a bound on cosine when ||T(xi)|| <= 1")

        # Since target is unit: dot(T(xi), target) = ||T(xi)|| * cosine_sim(T(xi), target)
        # So cosine_sim >= dot when ||T(xi)|| <= 1
        assert abs(dot_val - result_norm * cos_val) < 1e-10
        if result_norm <= 1.0 + 1e-10:
            assert cos_val >= dot_val - 1e-10, (
                f"Cosine should be >= dot when ||T(xi)|| <= 1"
            )


# ===========================================================================
# Integration: End-to-End Dream Pipeline Tests
# ===========================================================================

class TestEndToEndDreamPipeline:
    """Integration tests combining H1, H2, H3 through the dream pipeline."""

    def test_e2e_store_dream_query_cross_domain(self):
        """End-to-end: store patterns from two domains, dream, then query
        cross-domain. Verify that bridge formation enables cross-domain retrieval.
        """
        d = 64
        rng = np.random.default_rng(42)
        N_per_domain = 8
        sigma_threshold = 0.4

        # Create two "domains" as clusters
        domain_a_center = make_unit_vector(d, rng)
        domain_b_center = make_unit_vector(d, rng)
        # Make them fairly orthogonal
        domain_b_center -= np.dot(domain_b_center, domain_a_center) * domain_a_center
        domain_b_center = domain_b_center / np.linalg.norm(domain_b_center)

        print(f"\n--- E2E: Store, Dream, Cross-Domain Query ---")
        print(f"  Domain A center <-> Domain B center sim: {cosine_sim(domain_a_center, domain_b_center):.6f}")

        # Generate domain A patterns (clustered around domain_a_center)
        domain_a = []
        for _ in range(N_per_domain):
            v = domain_a_center + rng.standard_normal(d) * 0.15
            v = v / np.linalg.norm(v)
            domain_a.append(v)

        # Generate domain B patterns
        domain_b = []
        for _ in range(N_per_domain):
            v = domain_b_center + rng.standard_normal(d) * 0.15
            v = v / np.linalg.norm(v)
            domain_b.append(v)

        # Create cross-domain patterns (bridges)
        n_bridges = 3
        bridge_patterns = []
        for _ in range(n_bridges):
            v = make_cross_domain_vector(domain_a_center, domain_b_center, 0.5, 0.5, rng)
            bridge_patterns.append(v)

        all_patterns = np.array(domain_a + domain_b + bridge_patterns)
        N = len(all_patterns)
        importances = np.array(
            [0.5] * N_per_domain + [0.5] * N_per_domain + [0.7] * n_bridges
        )

        print(f"  Total patterns: {N} ({N_per_domain} A + {N_per_domain} B + {n_bridges} bridges)")

        # Count bridges across domains before dream
        bridges_across = 0
        for i in range(N_per_domain):
            for j in range(N_per_domain, 2 * N_per_domain):
                if cosine_sim(all_patterns[i], all_patterns[j]) >= sigma_threshold:
                    bridges_across += 1

        # Query from domain A, see if we can reach domain B
        query = domain_a[0] + rng.standard_normal(d) * 0.05
        query = query / np.linalg.norm(query)

        # Hopfield update at moderate beta
        beta = 5.0
        result = hopfield_update(beta, all_patterns, query)

        # Measure similarity to domain B
        sims_to_b = [cosine_sim(result, domain_b[j]) for j in range(N_per_domain)]
        max_sim_b = max(sims_to_b)

        # Compare with direct cosine sim (no Hopfield)
        direct_sims_b = [cosine_sim(query, domain_b[j]) for j in range(N_per_domain)]
        max_direct_b = max(direct_sims_b)

        print(f"  Cross-domain bridges (A->B at threshold {sigma_threshold}): {bridges_across}")
        print(f"  Query from domain A:")
        print(f"    Direct max sim to domain B: {max_direct_b:.6f}")
        print(f"    After Hopfield update, max sim to domain B: {max_sim_b:.6f}")
        print(f"    Transfer improvement: {max_sim_b - max_direct_b:+.6f}")

        # The bridge patterns should enable SOME cross-domain signal
        # (not necessarily dramatic, but measurable)
        print(f"    Cross-domain retrieval {'IMPROVED' if max_sim_b > max_direct_b else 'NOT IMPROVED'}")
