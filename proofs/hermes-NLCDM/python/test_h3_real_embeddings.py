"""Real-embedding H3 test: Transfer Dynamics with realistic data.

Tests the Phase 10 theorem (transfer_via_bridge) using real 1024-dim
embeddings from Qwen3-Embedding-0.6B. The key questions:

1. Is delta > 0 for cross-domain queries in practice?
   (If not, the theorem's precondition fails and the bound is inapplicable.)

2. How does the dot-product bound compare to actual cosine similarity?
   (The tribunal found dot degrades as O(1/N) while cosine stays strong.)

3. At what N does the bound become vacuous for realistic parameters?

Uses:
  - CORPUS (10 domains x 20 sentences) for domain-labeled patterns
  - embeddings_170k.npy as background noise to test scaling
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure importable
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from dream_ops import hopfield_update
from nlcdm_core import cosine_sim, softmax
from test_real_embeddings import (
    CORPUS,
    CORPUS_METADATA,
    DOMAIN_NAMES,
    encode_sentences,
    get_all_sentences,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus_embeddings():
    """Embed all 200 CORPUS sentences, return with domain labels."""
    sentences, domain_indices, domain_names = get_all_sentences()
    embeddings = encode_sentences(sentences)
    return {
        "embeddings": embeddings,  # (200, 1024)
        "sentences": sentences,
        "domain_indices": np.array(domain_indices),
        "domain_names": domain_names,
    }


@pytest.fixture(scope="module")
def background_embeddings():
    """Load 170k background embeddings for scaling tests."""
    path = _THIS_DIR / "embeddings_170k.npy"
    if not path.exists():
        pytest.skip("embeddings_170k.npy not found")
    emb = np.load(path).astype(np.float64)
    return emb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _domain_mask(domain_indices: np.ndarray, domain_idx: int) -> np.ndarray:
    """Boolean mask for patterns belonging to a domain."""
    return domain_indices == domain_idx


def _theoretical_bound(sigma: float, N: int, beta: float, delta: float) -> float:
    """Phase 10 lower bound: sigma / (1 + (N-1)*exp(-beta*delta))."""
    if delta <= 0:
        return 0.0  # precondition fails
    return sigma / (1 + (N - 1) * np.exp(-beta * delta))


# ===========================================================================
# Test 1: Measure alignment gap delta for cross-domain queries
# ===========================================================================

class TestAlignmentGapMeasurement:
    """Measure the alignment gap delta for real cross-domain queries.

    For each cross-domain sentence (indices 15-19 in each domain), use it
    as a query and measure:
      delta = dot(query, bridge_centroid) - max(dot(query, same_domain_pattern))

    If delta < 0 consistently, the transfer theorem is inapplicable.
    """

    def test_alignment_gap_per_cross_domain_sentence(self, corpus_embeddings):
        """For each of the 50 cross-domain sentences, measure delta.

        A cross-domain sentence from domain A (e.g., "The stress from watching
        my portfolio tank is giving me chest tightness" from finance->health)
        should ideally have higher alignment with a bridge centroid than with
        same-domain patterns. But the tribunal hypothesized delta < 0.
        """
        emb = corpus_embeddings["embeddings"]
        domain_indices = corpus_embeddings["domain_indices"]
        sentences = corpus_embeddings["sentences"]

        deltas = []
        delta_positive_count = 0
        total_cross_domain = 0

        print(f"\n{'='*80}")
        print("ALIGNMENT GAP (delta) FOR CROSS-DOMAIN QUERIES")
        print(f"{'='*80}")
        print(f"  delta = dot(query, bridge_centroid) - max_k(dot(query, same_domain[k]))")
        print(f"  Theorem requires delta > 0 for the bound to apply.\n")

        for src_domain_idx, src_domain in enumerate(DOMAIN_NAMES):
            # Cross-domain sentences are indices 15-19 within each domain's 20
            base_idx = src_domain_idx * 20
            cross_indices = list(range(base_idx + 15, base_idx + 20))

            # Same-domain patterns (indices 0-14, the non-cross-domain ones)
            same_domain_indices = list(range(base_idx, base_idx + 15))
            same_domain_patterns = emb[same_domain_indices]

            meta = CORPUS_METADATA[src_domain]

            for local_idx, global_idx in enumerate(cross_indices):
                total_cross_domain += 1
                query = emb[global_idx]
                target_domain = meta[15 + local_idx]["related_domain"]
                target_domain_idx = DOMAIN_NAMES.index(target_domain)

                # Target domain patterns (all 20)
                target_base = target_domain_idx * 20
                target_patterns = emb[target_base:target_base + 20]

                # Build a naive "bridge centroid": average of patterns from
                # both source and target domains that have above-median
                # similarity to both domain centroids
                src_centroid = np.mean(same_domain_patterns, axis=0)
                src_centroid = src_centroid / np.linalg.norm(src_centroid)
                tgt_centroid = np.mean(target_patterns, axis=0)
                tgt_centroid = tgt_centroid / np.linalg.norm(tgt_centroid)

                # Bridge centroid: average of source and target centroids
                bridge_centroid = (src_centroid + tgt_centroid) / 2
                bridge_centroid = bridge_centroid / np.linalg.norm(bridge_centroid)

                # Alignment of query with bridge vs same-domain patterns
                dot_bridge = np.dot(query, bridge_centroid)
                dots_same = same_domain_patterns @ query
                max_same = np.max(dots_same)
                delta = dot_bridge - max_same

                deltas.append(delta)
                if delta > 0:
                    delta_positive_count += 1

                print(f"  [{src_domain:>13s} -> {target_domain:<13s}] "
                      f"dot_bridge={dot_bridge:+.4f}  max_same={max_same:+.4f}  "
                      f"delta={delta:+.4f}  {'OK' if delta > 0 else 'FAIL'}")

        deltas = np.array(deltas)
        pct_positive = 100 * delta_positive_count / total_cross_domain

        print(f"\n{'─'*80}")
        print(f"SUMMARY: {delta_positive_count}/{total_cross_domain} queries have delta > 0 "
              f"({pct_positive:.1f}%)")
        print(f"  mean(delta) = {deltas.mean():+.4f}")
        print(f"  std(delta)  = {deltas.std():.4f}")
        print(f"  min(delta)  = {deltas.min():+.4f}")
        print(f"  max(delta)  = {deltas.max():+.4f}")
        print(f"  median      = {np.median(deltas):+.4f}")

        if pct_positive < 50:
            print(f"\n  VERDICT: delta < 0 for majority of cross-domain queries.")
            print(f"  The transfer theorem's precondition FAILS in practice.")
            print(f"  The bound is INAPPLICABLE for typical cross-domain retrieval.")
        else:
            print(f"\n  VERDICT: delta > 0 for majority — theorem may be applicable.")

        # Record measurement, don't assert pass/fail — this is empirical science
        assert total_cross_domain == 50, "Expected 50 cross-domain sentences"

    def test_alignment_gap_with_actual_cross_domain_pattern_as_bridge(self, corpus_embeddings):
        """Instead of a synthetic bridge centroid, use the actual cross-domain
        sentences as bridge patterns (they literally span two domains).

        This is the best-case scenario for the theorem: the bridge IS the
        cross-domain pattern itself.
        """
        emb = corpus_embeddings["embeddings"]
        domain_indices = corpus_embeddings["domain_indices"]

        print(f"\n{'='*80}")
        print("ALIGNMENT GAP USING CROSS-DOMAIN SENTENCES AS BRIDGES")
        print(f"{'='*80}")

        deltas = []
        for src_idx, src_domain in enumerate(DOMAIN_NAMES):
            base = src_idx * 20

            for tgt_idx, tgt_domain in enumerate(DOMAIN_NAMES):
                if src_idx == tgt_idx:
                    continue

                # Find cross-domain sentences from src that point to tgt
                meta = CORPUS_METADATA[src_domain]
                bridge_globals = []
                for local_i in range(15, 20):
                    if meta[local_i]["related_domain"] == tgt_domain:
                        bridge_globals.append(base + local_i)

                if not bridge_globals:
                    continue

                # Use a query from the source domain (varied sentences 5-14)
                query_idx = base + 7  # arbitrary varied sentence
                query = emb[query_idx]

                # All patterns in the store (full 200)
                all_dots = emb @ query

                for bg in bridge_globals:
                    bridge_dot = all_dots[bg]
                    # Max dot from same-domain (excluding the query itself)
                    same_mask = _domain_mask(domain_indices, src_idx)
                    same_dots = all_dots[same_mask]
                    # Exclude the query from same-domain max
                    same_dots_filtered = np.array([
                        all_dots[i] for i in range(len(emb))
                        if domain_indices[i] == src_idx and i != query_idx
                    ])
                    max_same = np.max(same_dots_filtered)
                    delta = bridge_dot - max_same
                    deltas.append(delta)

        deltas = np.array(deltas)
        pct_pos = 100 * np.sum(deltas > 0) / len(deltas)

        print(f"  Total measurements: {len(deltas)}")
        print(f"  delta > 0: {np.sum(deltas > 0)}/{len(deltas)} ({pct_pos:.1f}%)")
        print(f"  mean(delta) = {deltas.mean():+.4f}")
        print(f"  min(delta)  = {deltas.min():+.4f}")
        print(f"  max(delta)  = {deltas.max():+.4f}")

        if pct_pos < 10:
            print(f"\n  VERDICT: Even with ideal bridge patterns, delta < 0 almost always.")
            print(f"  Same-domain patterns dominate query alignment.")


# ===========================================================================
# Test 2: Dot-product bound vs cosine similarity at scale
# ===========================================================================

class TestDotVsCosineAtScale:
    """Test the key tribunal finding: dot bound degrades as O(1/N)
    but cosine stays strong because ||T(xi)|| shrinks proportionally."""

    def test_dot_vs_cosine_scaling(self, corpus_embeddings, background_embeddings):
        """Add background noise patterns and measure dot vs cosine at various N."""
        emb = corpus_embeddings["embeddings"]
        bg = background_embeddings

        # Pick a concrete cross-domain scenario:
        # Finance sentence about health stress as query
        query_idx = 0 * 20 + 15  # finance cross-domain sentence 0 -> health
        query = emb[query_idx]

        # Target: health domain centroid
        health_patterns = emb[3 * 20: 3 * 20 + 20]  # health is domain index 3
        target = np.mean(health_patterns, axis=0)
        target = target / np.linalg.norm(target)

        # Bridge: the query itself is a natural bridge (finance->health)
        bridge = query.copy()
        sigma = cosine_sim(bridge, target)

        beta = 5.0
        N_values = [200, 500, 1000, 2000, 5000]

        print(f"\n{'='*80}")
        print("DOT PRODUCT BOUND vs COSINE SIMILARITY AT SCALE")
        print(f"{'='*80}")
        print(f"  Query: '{corpus_embeddings['sentences'][query_idx][:60]}...'")
        print(f"  Target: health domain centroid")
        print(f"  sigma (bridge-target cosine) = {sigma:.4f}")
        print(f"  beta = {beta}")
        print()
        print(f"  {'N':>6s}  {'delta':>8s}  {'bound':>10s}  {'dot(T,tgt)':>10s}  "
              f"{'cos(T,tgt)':>10s}  {'||T(xi)||':>10s}  {'bound_ok':>8s}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

        for N in N_values:
            rng = np.random.default_rng(42)

            # Build pattern store: 200 corpus + (N-200) background
            n_bg = N - 200
            if n_bg > len(bg):
                print(f"  {N:6d}  SKIP (not enough background embeddings)")
                continue

            bg_sample = bg[rng.choice(len(bg), size=n_bg, replace=False)]
            patterns = np.vstack([emb, bg_sample])

            # Find bridge index (the query pattern in the store)
            bridge_idx = query_idx

            # Compute alignment gap
            dots_xi = patterns @ query
            bridge_dot = dots_xi[bridge_idx]
            # Max over all other patterns
            other_dots = np.concatenate([dots_xi[:bridge_idx], dots_xi[bridge_idx + 1:]])
            max_other = np.max(other_dots)
            delta = bridge_dot - max_other

            # Hopfield update
            result = hopfield_update(beta, patterns, query)
            dot_transfer = np.dot(result, target)
            cos_transfer = cosine_sim(result, target)
            result_norm = np.linalg.norm(result)

            bound = _theoretical_bound(sigma, N, beta, delta)
            bound_ok = dot_transfer >= bound - 1e-6 if delta > 0 else True

            print(f"  {N:6d}  {delta:+8.4f}  {bound:10.6f}  {dot_transfer:10.6f}  "
                  f"{cos_transfer:10.6f}  {result_norm:10.6f}  "
                  f"{'PASS' if bound_ok else 'FAIL'}")

        print(f"\n  KEY: Watch dot(T,tgt) shrink while cos(T,tgt) stays stable.")
        print(f"  The theorem bounds dot, but cosine (direction) is what matters.")

    def test_cosine_stability_detail(self, corpus_embeddings, background_embeddings):
        """Detailed measurement: for multiple cross-domain queries, measure
        how cosine_sim(T(xi), target) varies with N while dot collapses."""
        emb = corpus_embeddings["embeddings"]
        bg = background_embeddings
        beta = 5.0

        # Test 5 different cross-domain queries
        query_configs = [
            (0 * 20 + 15, 3, "finance->health"),
            (1 * 20 + 16, 2, "work->technology"),
            (4 * 20 + 15, 2, "family->technology"),
            (6 * 20 + 17, 2, "fitness->technology"),
            (8 * 20 + 18, 7, "travel->relationships"),
        ]

        N_values = [200, 1000, 5000]

        print(f"\n{'='*80}")
        print("COSINE STABILITY ACROSS QUERIES AND SCALES")
        print(f"{'='*80}")

        for qi, (query_idx, tgt_domain_idx, label) in enumerate(query_configs):
            query = emb[query_idx]
            tgt_base = tgt_domain_idx * 20
            target = np.mean(emb[tgt_base:tgt_base + 20], axis=0)
            target = target / np.linalg.norm(target)

            print(f"\n  Query {qi}: {label}")
            for N in N_values:
                rng = np.random.default_rng(42)
                n_bg = max(0, N - 200)
                if n_bg > len(bg):
                    continue
                if n_bg > 0:
                    bg_sample = bg[rng.choice(len(bg), size=n_bg, replace=False)]
                    patterns = np.vstack([emb, bg_sample])
                else:
                    patterns = emb

                result = hopfield_update(beta, patterns, query)
                dot_t = np.dot(result, target)
                cos_t = cosine_sim(result, target)
                norm_t = np.linalg.norm(result)

                print(f"    N={N:5d}: dot={dot_t:+.6f}  cos={cos_t:+.6f}  "
                      f"||T||={norm_t:.6f}  ratio=dot/cos={dot_t/cos_t:.6f}")


# ===========================================================================
# Test 3: Per-query precondition audit
# ===========================================================================

class TestPreconditionAudit:
    """Audit ALL preconditions of the Phase 10 theorem on real data.

    The theorem requires:
      (a) delta > 0: bridge has highest alignment with query
      (b) hothers_nn: all patterns have non-negative dot with target
      (c) sigma > 0: bridge has positive similarity to target
    """

    def test_precondition_satisfaction_rate(self, corpus_embeddings):
        """For each cross-domain query, check how many preconditions hold."""
        emb = corpus_embeddings["embeddings"]
        domain_indices = corpus_embeddings["domain_indices"]
        N = len(emb)

        results = {"all_hold": 0, "delta_fail": 0, "nn_fail": 0, "both_fail": 0}
        total = 0

        print(f"\n{'='*80}")
        print("PRECONDITION AUDIT (200-pattern store, all CORPUS embeddings)")
        print(f"{'='*80}")

        for src_idx, src_domain in enumerate(DOMAIN_NAMES):
            base = src_idx * 20
            meta = CORPUS_METADATA[src_domain]

            for local_i in range(15, 20):
                global_i = base + local_i
                query = emb[global_i]
                target_domain = meta[local_i]["related_domain"]
                tgt_idx = DOMAIN_NAMES.index(target_domain)
                tgt_base = tgt_idx * 20

                # Target centroid
                target = np.mean(emb[tgt_base:tgt_base + 20], axis=0)
                target = target / np.linalg.norm(target)

                # The query IS the bridge pattern in this test
                bridge_idx = global_i
                sigma = cosine_sim(query, target)

                # (a) delta check
                all_dots = emb @ query
                bridge_dot = all_dots[bridge_idx]
                other_dots = np.concatenate([all_dots[:bridge_idx], all_dots[bridge_idx + 1:]])
                delta = bridge_dot - np.max(other_dots)

                # (b) non-negativity check
                pattern_target_dots = emb @ target
                n_negative = np.sum(pattern_target_dots < 0)
                nn_holds = (n_negative == 0)

                total += 1
                if delta > 0 and nn_holds:
                    results["all_hold"] += 1
                elif delta <= 0 and not nn_holds:
                    results["both_fail"] += 1
                elif delta <= 0:
                    results["delta_fail"] += 1
                else:
                    results["nn_fail"] += 1

        print(f"  Total queries: {total}")
        print(f"  All preconditions hold:  {results['all_hold']:3d} ({100*results['all_hold']/total:.1f}%)")
        print(f"  delta <= 0 only:         {results['delta_fail']:3d} ({100*results['delta_fail']/total:.1f}%)")
        print(f"  non-negativity fail only: {results['nn_fail']:3d} ({100*results['nn_fail']/total:.1f}%)")
        print(f"  Both fail:               {results['both_fail']:3d} ({100*results['both_fail']/total:.1f}%)")

        pct_applicable = 100 * results["all_hold"] / total
        if pct_applicable < 10:
            print(f"\n  VERDICT: Theorem applicable in < 10% of cases.")
            print(f"  The transfer bound is effectively VACUOUS for real embeddings.")
        elif pct_applicable < 50:
            print(f"\n  VERDICT: Theorem applicable in minority of cases.")
        else:
            print(f"\n  VERDICT: Theorem applicable in majority of cases.")

    def test_non_negativity_fraction(self, corpus_embeddings):
        """For random target vectors, what fraction of patterns have
        non-negative dot product? (Theorem requires ALL to be non-negative.)"""
        emb = corpus_embeddings["embeddings"]
        N = len(emb)

        print(f"\n{'='*80}")
        print("NON-NEGATIVITY PRECONDITION (hothers_nn)")
        print(f"{'='*80}")

        fractions_negative = []
        for tgt_idx in range(len(DOMAIN_NAMES)):
            tgt_base = tgt_idx * 20
            target = np.mean(emb[tgt_base:tgt_base + 20], axis=0)
            target = target / np.linalg.norm(target)

            dots = emb @ target
            frac_neg = np.sum(dots < 0) / N
            fractions_negative.append(frac_neg)

            print(f"  Target=centroid({DOMAIN_NAMES[tgt_idx]:>13s}): "
                  f"{np.sum(dots < 0):3d}/{N} negative ({100*frac_neg:.1f}%)")

        mean_frac = np.mean(fractions_negative)
        print(f"\n  Mean fraction negative: {100*mean_frac:.1f}%")
        print(f"  For theorem to apply, need 0% negative.")

        if mean_frac > 0.1:
            print(f"  VERDICT: Non-negativity precondition FAILS for >{100*mean_frac:.0f}% of patterns.")
            print(f"  This is expected — real embeddings have near-zero mean dot products")
            print(f"  across unrelated domains, with ~half negative by symmetry.")


# ===========================================================================
# Test 4: What the theorem SHOULD bound — cosine-based analysis
# ===========================================================================

class TestCosineBoundProposal:
    """Explore what a cosine-based bound would look like.

    The tribunal found that ||T(xi)|| shrinks proportionally with N,
    preserving direction (cosine) while destroying magnitude (dot).

    Measure: cosine_sim(T(xi), target) vs theoretical predictions.
    """

    def test_cosine_vs_dot_ratio(self, corpus_embeddings, background_embeddings):
        """Measure ||T(xi)|| and the cosine/dot ratio at various N.

        If ||T(xi)|| ~ 1/sqrt(N) or similar, a cosine bound would be
        O(sigma) independent of N (much tighter than O(sigma/N)).
        """
        emb = corpus_embeddings["embeddings"]
        bg = background_embeddings
        beta = 5.0

        query_idx = 0 * 20 + 15  # finance->health
        query = emb[query_idx]
        target = np.mean(emb[3 * 20: 3 * 20 + 20], axis=0)
        target = target / np.linalg.norm(target)

        N_values = [200, 500, 1000, 2000, 5000, 10000]

        print(f"\n{'='*80}")
        print("||T(xi)|| SCALING AND COSINE/DOT RATIO")
        print(f"{'='*80}")
        print(f"  {'N':>6s}  {'||T(xi)||':>10s}  {'dot':>10s}  {'cosine':>10s}  "
              f"{'cos/dot':>8s}  {'1/||T||':>8s}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")

        for N in N_values:
            rng = np.random.default_rng(42)
            n_bg = max(0, N - 200)
            if n_bg > len(bg):
                continue

            if n_bg > 0:
                bg_sample = bg[rng.choice(len(bg), size=n_bg, replace=False)]
                patterns = np.vstack([emb, bg_sample])
            else:
                patterns = emb

            result = hopfield_update(beta, patterns, query)
            norm_t = np.linalg.norm(result)
            dot_t = np.dot(result, target)
            cos_t = cosine_sim(result, target)

            print(f"  {N:6d}  {norm_t:10.6f}  {dot_t:10.6f}  {cos_t:10.6f}  "
                  f"{cos_t/dot_t if abs(dot_t) > 1e-10 else float('inf'):8.4f}  "
                  f"{1/norm_t if norm_t > 1e-10 else float('inf'):8.4f}")

        print(f"\n  If cos/dot ≈ 1/||T(xi)||, then cosine = dot / ||T(xi)||")
        print(f"  and a cosine-based bound = dot_bound / ||T(xi)|| would be tight.")


# ===========================================================================
# Test 5: Mechanism comparison — Hopfield transfer vs what actually works
# ===========================================================================

class TestMechanismComparison:
    """Compare Hopfield single-step transfer with iterative convergence
    and measure whether the bridge mechanism helps cross-domain retrieval."""

    def test_single_step_vs_convergence(self, corpus_embeddings):
        """Compare single Hopfield step with iterating to convergence.

        The tribunal's H3-B hypothesis: iterated dynamics converge to
        same-domain patterns, not to bridges.
        """
        emb = corpus_embeddings["embeddings"]
        beta = 5.0
        max_steps = 50
        tol = 1e-6

        query_idx = 0 * 20 + 15  # finance->health
        query = emb[query_idx]

        # Health domain centroid as target
        target = np.mean(emb[3 * 20: 3 * 20 + 20], axis=0)
        target = target / np.linalg.norm(target)

        # Finance domain centroid
        src_centroid = np.mean(emb[0:15], axis=0)
        src_centroid = src_centroid / np.linalg.norm(src_centroid)

        print(f"\n{'='*80}")
        print("SINGLE STEP vs CONVERGENCE (Hopfield iteration)")
        print(f"{'='*80}")
        print(f"  Query: finance->health cross-domain sentence")
        print(f"  beta = {beta}\n")

        xi = query.copy()
        print(f"  {'step':>4s}  {'cos(xi,target)':>14s}  {'cos(xi,src)':>12s}  "
              f"{'||xi||':>8s}  {'delta_norm':>10s}")

        for step in range(max_steps):
            xi_new = hopfield_update(beta, emb, xi)
            delta_norm = np.linalg.norm(xi_new - xi)
            xi = xi_new

            cos_tgt = cosine_sim(xi, target)
            cos_src = cosine_sim(xi, src_centroid)

            if step < 10 or step % 10 == 0 or delta_norm < tol:
                print(f"  {step:4d}  {cos_tgt:14.6f}  {cos_src:12.6f}  "
                      f"{np.linalg.norm(xi):8.6f}  {delta_norm:10.2e}")

            if delta_norm < tol:
                print(f"  Converged at step {step}")
                break

        # Find which stored pattern the converged state is closest to
        final_sims = emb @ (xi / np.linalg.norm(xi))
        nearest_idx = np.argmax(final_sims)
        nearest_domain = DOMAIN_NAMES[nearest_idx // 20]
        nearest_local = nearest_idx % 20

        print(f"\n  Converged state nearest to: pattern {nearest_idx} "
              f"({nearest_domain}, local idx {nearest_local})")
        print(f"  Similarity to nearest: {final_sims[nearest_idx]:.6f}")
        print(f"  Final cos(xi, health_centroid): {cosine_sim(xi, target):.6f}")
        print(f"  Final cos(xi, finance_centroid): {cosine_sim(xi, src_centroid):.6f}")

        # Did it converge to target domain or source domain?
        if nearest_domain == "health":
            print(f"  OUTCOME: Converged to TARGET domain (cross-domain transfer worked)")
        elif nearest_domain == "finance":
            print(f"  OUTCOME: Converged to SOURCE domain (no cross-domain transfer)")
        else:
            print(f"  OUTCOME: Converged to {nearest_domain} domain (unexpected)")
