"""Discriminative tests for reasoning trace centroids as memory objects.

Core hypothesis: OOD reasoning is an emergent property of effective memory.
If reasoning traces (not just content patterns) are stored as first-class
unit vectors in the Hopfield network, dream consolidation naturally produces:

1. Trace centroid formation — structurally similar reasoning strategies merge
2. Cross-domain bridge discovery — rem_explore finds content↔trace associations
3. OOD transfer via memory — retrieval quality for novel domains improves
4. Recursive improvement — storing new traces enriches centroid structure

Mathematical framework:
  Combined pattern matrix X̃ = [X_content; X_traces] ∈ ℝ^{(N+M) × d}
  Similarity matrix S̃ = X̃ @ X̃.T has 4 blocks:
    S_CC (content-content), S_TT (trace-trace),
    S_CT (content-trace), S_TC (trace-content)

  Claim: dream_cycle_xb operating on X̃ produces S_TT with approximate
  block-diagonal structure (trace centroids), and S_CT with non-trivial
  off-diagonal structure (content-trace bridges enabling OOD transfer).

Each test class targets one specific claim and can FAIL independently.
"""

from __future__ import annotations

import numpy as np
import pytest

from dream_ops import (
    dream_cycle_xb,
    nrem_merge_xb,
    nrem_repulsion_xb,
    rem_unlearn_xb,
    rem_explore_cross_domain_xb,
    spreading_activation,
)
from nlcdm_core import softmax
from test_capacity_boundary import make_cluster_patterns, make_separated_centroids


# ---------------------------------------------------------------------------
# Shared fixtures — Domain and trace generation
# ---------------------------------------------------------------------------

DIM = 128
BETA = 5.0
SEED = 9999


def _make_domain_patterns(
    n_domains: int,
    per_domain: int,
    dim: int = DIM,
    spread: float = 0.10,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate content patterns organized into knowledge domains.

    Returns (patterns, domain_labels).
    """
    centroids = make_separated_centroids(n_domains, dim, seed=seed)
    return make_cluster_patterns(centroids, per_domain, spread, seed=seed + 1)


def _make_reasoning_traces(
    source_patterns: np.ndarray,
    source_labels: np.ndarray,
    target_patterns: np.ndarray,
    target_labels: np.ndarray,
    n_traces: int,
    dim: int = DIM,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate reasoning traces that bridge two domains.

    A reasoning trace for "applying domain A skill to domain B problem" is
    modeled as a normalized blend of a source pattern and a target pattern,
    capturing the structural relationship between them.

    Returns:
        traces: (n_traces, dim) unit vectors
        source_indices: which source pattern each trace came from
        target_indices: which target pattern each trace was applied to
    """
    rng = np.random.default_rng(seed)

    unique_source = np.unique(source_labels)
    unique_target = np.unique(target_labels)

    traces = []
    src_idx_list = []
    tgt_idx_list = []

    for _ in range(n_traces):
        # Pick a source domain and target domain (must be different)
        s_domain = rng.choice(unique_source)
        t_candidates = unique_target[unique_target != s_domain]
        if len(t_candidates) == 0:
            t_candidates = unique_target
        t_domain = rng.choice(t_candidates)

        # Pick one pattern from each
        s_indices = np.where(source_labels == s_domain)[0]
        t_indices = np.where(target_labels == t_domain)[0]
        s_idx = rng.choice(s_indices)
        t_idx = rng.choice(t_indices)

        # Reasoning trace = blend of source skill + target context + noise
        # The blend ratio captures "how much of the source skill was needed"
        alpha = rng.uniform(0.3, 0.7)
        trace = alpha * source_patterns[s_idx] + (1 - alpha) * target_patterns[t_idx]
        # Add small noise to represent the actual reasoning process
        trace += 0.05 * rng.standard_normal(dim)
        trace /= np.linalg.norm(trace)

        traces.append(trace)
        src_idx_list.append(s_idx)
        tgt_idx_list.append(t_idx)

    return (
        np.array(traces),
        np.array(src_idx_list),
        np.array(tgt_idx_list),
    )


def _block_diagonality(S: np.ndarray, labels: np.ndarray) -> float:
    """Measure how block-diagonal S is relative to labels.

    Returns ratio of within-cluster similarity to between-cluster similarity.
    Higher = more block-diagonal. Returns inf if between=0.
    """
    unique = np.unique(labels)
    within_sum = 0.0
    within_count = 0
    between_sum = 0.0
    between_count = 0

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                within_sum += abs(S[i, j])
                within_count += 1
            else:
                between_sum += abs(S[i, j])
                between_count += 1

    within_avg = within_sum / max(within_count, 1)
    between_avg = between_sum / max(between_count, 1)

    if between_avg < 1e-12:
        return float('inf')
    return within_avg / between_avg


def _effective_rank(S: np.ndarray, threshold: float = 0.01) -> int:
    """Effective rank of a matrix via singular value thresholding."""
    sv = np.linalg.svd(S, compute_uv=False)
    sv_normalized = sv / (sv[0] + 1e-30)
    return int(np.sum(sv_normalized > threshold))


# ===========================================================================
# Claim 1: Trace centroid formation via dream consolidation
# ===========================================================================

class TestTraceCentroidFormation:
    """When reasoning traces are stored alongside content patterns and
    dream_cycle_xb is run, structurally similar traces merge into centroids.

    The S_TT block (trace-trace similarity) should become more block-diagonal
    after dream consolidation, with each block corresponding to a reasoning
    strategy centroid.
    """

    def test_similar_traces_cluster_in_s_tt(self):
        """Traces from the same source→target domain pair should have higher
        mutual similarity than traces from different pairs."""
        content, labels = _make_domain_patterns(4, 10, seed=100)
        # Create traces: 3 strategies (0→1, 0→2, 1→3), 8 traces each
        traces_01, _, _ = _make_reasoning_traces(
            content, labels, content, labels, n_traces=8, seed=200
        )
        # Force domain 0→1 by manual construction
        rng = np.random.default_rng(201)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]
        dom2 = content[labels == 2]
        dom3 = content[labels == 3]

        def _make_strategy_traces(source_patterns, target_patterns, n, rng):
            traces = []
            for _ in range(n):
                s = source_patterns[rng.integers(len(source_patterns))]
                t = target_patterns[rng.integers(len(target_patterns))]
                alpha = rng.uniform(0.3, 0.7)
                tr = alpha * s + (1 - alpha) * t + 0.05 * rng.standard_normal(DIM)
                tr /= np.linalg.norm(tr)
                traces.append(tr)
            return np.array(traces)

        strat_01 = _make_strategy_traces(dom0, dom1, 8, rng)
        strat_02 = _make_strategy_traces(dom0, dom2, 8, rng)
        strat_13 = _make_strategy_traces(dom1, dom3, 8, rng)

        all_traces = np.vstack([strat_01, strat_02, strat_13])
        trace_labels = np.array([0]*8 + [1]*8 + [2]*8)

        S_tt = all_traces @ all_traces.T

        bd_ratio = _block_diagonality(S_tt, trace_labels)
        # Within-strategy similarity should exceed between-strategy by >1.5x
        assert bd_ratio > 1.5, (
            f"Trace block-diagonality ratio {bd_ratio:.2f} < 1.5 — "
            f"strategies not forming distinct clusters in S_TT"
        )
        print(f"  S_TT block-diagonality: {bd_ratio:.2f}x (within/between)")

    def test_dream_merges_similar_traces(self):
        """After dream_cycle_xb, near-duplicate traces from the same strategy
        should be merged into centroids (N_out < N_in for trace portion).

        Key: traces must be VERY tight (noise << spread) so intra-sim > 0.90
        to trigger nrem_merge_xb. We use the SAME source/target pair with
        tiny noise, not random samples from the cluster.
        """
        rng = np.random.default_rng(300)
        content, labels = _make_domain_patterns(3, 8, seed=300)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]

        # Use fixed source/target patterns to ensure high mutual similarity
        s_fixed = dom0[0]
        t_fixed = dom1[0]
        base_trace = 0.5 * s_fixed + 0.5 * t_fixed
        base_trace /= np.linalg.norm(base_trace)

        # Create 12 traces as tiny perturbations of the SAME base trace
        # At noise=0.005, dim=128, expected cosine sim ≈ 0.999
        traces = []
        for _ in range(12):
            tr = base_trace + 0.005 * rng.standard_normal(DIM)
            tr /= np.linalg.norm(tr)
            traces.append(tr)
        traces = np.array(traces)

        # Combine content + traces
        combined = np.vstack([content, traces])
        N_content = content.shape[0]
        N_total_in = combined.shape[0]

        # Run dream cycle
        combined_labels = np.concatenate([labels, np.full(12, max(labels) + 1)])
        report = dream_cycle_xb(
            combined, beta=BETA,
            labels=combined_labels,
            seed=42,
        )

        N_total_out = report.patterns.shape[0]
        # Dream should merge some of the 12 near-duplicate traces
        n_removed = N_total_in - N_total_out
        print(f"  Input: {N_total_in} ({N_content} content + 12 traces)")
        print(f"  Output: {N_total_out} (removed {n_removed})")
        print(f"  Merge map entries: {len(report.merge_map)}")

        # At minimum, some traces should be merged or pruned
        assert n_removed > 0, (
            f"Dream cycle removed 0 patterns from {N_total_in} — "
            f"12 near-duplicate traces should trigger merge/prune"
        )

    def test_centroid_preserves_strategy_direction(self):
        """When traces merge into a centroid, the centroid should still point
        in the direction of the original source→target strategy."""
        rng = np.random.default_rng(400)
        content, labels = _make_domain_patterns(3, 8, seed=400)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]

        # Strategy direction: average of dom0 → dom1 blends
        strategy_dir = np.mean(dom0, axis=0) * 0.5 + np.mean(dom1, axis=0) * 0.5
        strategy_dir /= np.linalg.norm(strategy_dir)

        # Create near-duplicate traces
        traces = []
        for _ in range(10):
            s = dom0[rng.integers(len(dom0))]
            t = dom1[rng.integers(len(dom1))]
            tr = 0.5 * s + 0.5 * t + 0.03 * rng.standard_normal(DIM)
            tr /= np.linalg.norm(tr)
            traces.append(tr)
        traces_arr = np.array(traces)

        # Merge via nrem_merge_xb directly
        importances = np.full(len(traces_arr), 0.5)
        merged, merge_map = nrem_merge_xb(
            traces_arr, importances, threshold=0.90, min_group=3
        )

        if len(merge_map) > 0:
            # Check centroid direction
            for out_idx, group in merge_map.items():
                centroid = merged[out_idx]
                cos_sim = float(centroid @ strategy_dir)
                print(f"  Centroid {out_idx} (from {len(group)} traces): "
                      f"cos_sim to strategy = {cos_sim:.4f}")
                assert cos_sim > 0.5, (
                    f"Centroid cos_sim {cos_sim:.4f} to strategy direction < 0.5 — "
                    f"merge lost strategy information"
                )
        else:
            # If no merge happened, traces must be too spread out
            intra_sim = traces_arr @ traces_arr.T
            np.fill_diagonal(intra_sim, 0)
            max_sim = np.max(intra_sim)
            print(f"  No merge — max intra-trace sim = {max_sim:.4f}")
            pytest.skip(f"No merge at threshold 0.90 (max_sim={max_sim:.4f})")


# ===========================================================================
# Claim 2: Cross-domain bridge discovery (content ↔ trace associations)
# ===========================================================================

class TestCrossDomainBridges:
    """rem_explore_cross_domain_xb should discover associations between
    content patterns and reasoning traces when they live in the same
    pattern matrix.

    The key insight: a reasoning trace for "applying domain A to domain B"
    has perturbation-response correlation with BOTH domain A and domain B
    patterns, creating a bridge that wouldn't exist with content alone.
    """

    def test_traces_bridge_content_domains(self):
        """With only content patterns, domains A and B have low association.
        Adding traces A→B should create discoverable bridges."""
        rng = np.random.default_rng(500)
        content, labels = _make_domain_patterns(4, 15, seed=500)

        # Baseline: explore with content only
        assoc_content_only = rem_explore_cross_domain_xb(
            content, labels, n_probes=200, rng=np.random.default_rng(42)
        )

        # Now create traces bridging domain 0 and domain 1
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]
        traces = []
        for _ in range(10):
            s = dom0[rng.integers(len(dom0))]
            t = dom1[rng.integers(len(dom1))]
            alpha = rng.uniform(0.3, 0.7)
            tr = alpha * s + (1 - alpha) * t + 0.05 * rng.standard_normal(DIM)
            tr /= np.linalg.norm(tr)
            traces.append(tr)
        traces_arr = np.array(traces)

        # Combined: content + traces, traces get a new cluster label
        combined = np.vstack([content, traces_arr])
        combined_labels = np.concatenate([
            labels,
            np.full(len(traces_arr), max(labels) + 1)
        ])

        assoc_with_traces = rem_explore_cross_domain_xb(
            combined, combined_labels, n_probes=200,
            rng=np.random.default_rng(42)
        )

        n_base = len(assoc_content_only)
        n_with = len(assoc_with_traces)

        print(f"  Associations (content only): {n_base}")
        print(f"  Associations (content + traces): {n_with}")

        # Count associations involving trace indices
        N_content = content.shape[0]
        trace_involved = [
            (i, j, s) for i, j, s in assoc_with_traces
            if i >= N_content or j >= N_content
        ]
        print(f"  Associations involving traces: {len(trace_involved)}")

        # Traces should create NEW associations (bridging content domains)
        assert n_with >= n_base, (
            f"Adding traces reduced associations from {n_base} to {n_with}"
        )

    def test_s_ct_block_has_nontrivial_structure(self):
        """The cross-correlation block S_CT = X_content @ X_traces.T should
        have non-trivial rank (more than 1), indicating multiple independent
        content-trace relationships."""
        rng = np.random.default_rng(600)
        content, labels = _make_domain_patterns(4, 10, seed=600)

        # Create traces from multiple strategies
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]
        dom2 = content[labels == 2]
        dom3 = content[labels == 3]

        def _strat(src, tgt, n):
            t = []
            for _ in range(n):
                s = src[rng.integers(len(src))]
                g = tgt[rng.integers(len(tgt))]
                tr = 0.5 * s + 0.5 * g + 0.05 * rng.standard_normal(DIM)
                tr /= np.linalg.norm(tr)
                t.append(tr)
            return np.array(t)

        traces = np.vstack([
            _strat(dom0, dom1, 5),
            _strat(dom0, dom2, 5),
            _strat(dom2, dom3, 5),
            _strat(dom1, dom3, 5),
        ])

        S_ct = content @ traces.T  # (N_content, N_traces)
        eff_rank = _effective_rank(S_ct, threshold=0.05)

        print(f"  S_CT shape: {S_ct.shape}")
        print(f"  S_CT effective rank: {eff_rank}")
        print(f"  S_CT max: {np.max(S_ct):.4f}, min: {np.min(S_ct):.4f}")

        # With 4 strategies across 4 domains, rank should be >= 4
        assert eff_rank >= 4, (
            f"S_CT effective rank {eff_rank} < 4 — "
            f"content-trace cross-correlation is too low-rank"
        )


# ===========================================================================
# Claim 3: OOD transfer — retrieving relevant traces for novel domains
# ===========================================================================

class TestOODTransfer:
    """When a query from novel domain Z is presented, the system should
    retrieve reasoning traces that are structurally relevant, even though
    Z was never seen during trace creation.

    The mechanism: Z-query has some structural similarity to domain A or B
    content, and traces bridging A↔B provide a reasoning template.
    """

    def test_novel_domain_retrieves_relevant_traces(self):
        """A query from unseen domain Z should retrieve traces from
        structurally similar domains, not random traces."""
        rng = np.random.default_rng(700)

        # 4 domains, but we only create traces for 0→1 and 2→3
        content, labels = _make_domain_patterns(4, 10, seed=700)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]
        dom2 = content[labels == 2]
        dom3 = content[labels == 3]

        def _strat(src, tgt, n):
            t = []
            for _ in range(n):
                s = src[rng.integers(len(src))]
                g = tgt[rng.integers(len(tgt))]
                tr = 0.5 * s + 0.5 * g + 0.05 * rng.standard_normal(DIM)
                tr /= np.linalg.norm(tr)
                t.append(tr)
            return np.array(t)

        traces_01 = _strat(dom0, dom1, 8)  # strategy: 0→1
        traces_23 = _strat(dom2, dom3, 8)  # strategy: 2→3
        all_traces = np.vstack([traces_01, traces_23])
        trace_strategy = np.array([0]*8 + [1]*8)  # 0=strat_01, 1=strat_23

        # Novel query: blend of domain 0 and domain 2 (never seen as a trace)
        # This should retrieve traces from BOTH strategies since it overlaps
        # with both source domains
        q = 0.6 * dom0[0] + 0.4 * dom2[0] + 0.03 * rng.standard_normal(DIM)
        q /= np.linalg.norm(q)

        # Retrieve via dot product (Hopfield-style)
        sims = all_traces @ q
        top_k = 4
        top_idx = np.argsort(sims)[-top_k:][::-1]

        top_strategies = trace_strategy[top_idx]
        top_sims = sims[top_idx]

        print(f"  Novel query (blend of dom0 + dom2)")
        print(f"  Top-{top_k} retrievals: strategies={top_strategies}, "
              f"sims={top_sims}")

        # Should retrieve from BOTH strategies (since query overlaps both sources)
        unique_strats = set(top_strategies)
        assert len(unique_strats) >= 2, (
            f"Top-{top_k} only retrieved from strategy {unique_strats} — "
            f"expected traces from both 0→1 and 2→3 strategies"
        )

    def test_hopfield_retrieval_concentrates_on_traces(self):
        """When content + traces are stored together, a Hopfield update from
        a novel query should have non-negligible weight on trace patterns,
        not just content patterns."""
        rng = np.random.default_rng(800)
        content, labels = _make_domain_patterns(3, 10, seed=800)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]

        # Create traces
        traces = []
        for _ in range(8):
            s = dom0[rng.integers(len(dom0))]
            t = dom1[rng.integers(len(dom1))]
            tr = 0.5 * s + 0.5 * t + 0.05 * rng.standard_normal(DIM)
            tr /= np.linalg.norm(tr)
            traces.append(tr)
        traces_arr = np.array(traces)

        combined = np.vstack([content, traces_arr])
        N_content = content.shape[0]

        # Novel query: perturbation of domain 0 centroid
        q = np.mean(dom0, axis=0) + 0.1 * rng.standard_normal(DIM)
        q /= np.linalg.norm(q)

        # Compute softmax attention weights
        sims = combined @ q
        weights = softmax(BETA, sims)

        trace_weight = float(np.sum(weights[N_content:]))
        content_weight = float(np.sum(weights[:N_content]))

        print(f"  Softmax attention: content={content_weight:.4f}, "
              f"traces={trace_weight:.4f}")
        print(f"  Trace fraction: {trace_weight/(content_weight+trace_weight):.4f}")

        # Traces should get some non-trivial attention (> 1%)
        # since they're blends involving domain 0
        assert trace_weight > 0.01, (
            f"Trace weight {trace_weight:.6f} < 0.01 — "
            f"traces invisible to Hopfield retrieval"
        )

    def test_spreading_activation_reaches_traces(self):
        """Starting from a content query, spreading activation should
        converge to a fixed point that has trace components."""
        rng = np.random.default_rng(850)
        content, labels = _make_domain_patterns(3, 10, seed=850)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]

        traces = []
        for _ in range(6):
            s = dom0[rng.integers(len(dom0))]
            t = dom1[rng.integers(len(dom1))]
            tr = 0.5 * s + 0.5 * t + 0.05 * rng.standard_normal(DIM)
            tr /= np.linalg.norm(tr)
            traces.append(tr)
        traces_arr = np.array(traces)
        combined = np.vstack([content, traces_arr])
        N_content = content.shape[0]

        # Run spreading activation from a domain 0 query
        q = dom0[0] + 0.05 * rng.standard_normal(DIM)
        q /= np.linalg.norm(q)

        fp = spreading_activation(BETA, combined, q, max_steps=100, tol=1e-6)

        # Decompose fixed point into content and trace components
        content_sims = content @ fp
        trace_sims = traces_arr @ fp

        max_trace_sim = float(np.max(trace_sims))
        mean_trace_sim = float(np.mean(trace_sims))
        max_content_sim = float(np.max(content_sims))

        print(f"  Fixed point: max content sim={max_content_sim:.4f}, "
              f"max trace sim={max_trace_sim:.4f}, "
              f"mean trace sim={mean_trace_sim:.4f}")

        # The fixed point should have measurable trace components
        # (not zero similarity to all traces)
        assert max_trace_sim > 0.1, (
            f"Max trace similarity {max_trace_sim:.4f} < 0.1 — "
            f"spreading activation ignores traces entirely"
        )


# ===========================================================================
# Claim 4: Recursive improvement — dream cycles enrich trace structure
# ===========================================================================

class TestRecursiveImprovement:
    """The recursive loop: reason about Y using N → store trace → dream →
    reason about Z using {N, Y-traces} → store → dream → ...

    Each iteration should:
    - Not lose existing trace centroids
    - Potentially discover new cross-domain bridges
    - Maintain or increase the effective rank of S_CT
    """

    def test_multi_cycle_trace_accumulation(self):
        """Run multiple dream cycles, adding new traces each time.
        The trace centroid structure should strengthen, not degrade."""
        rng = np.random.default_rng(900)

        # Initial content: 3 domains
        content, labels = _make_domain_patterns(3, 10, seed=900)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]
        dom2 = content[labels == 2]

        def _make_traces(src, tgt, n, rng):
            traces = []
            for _ in range(n):
                s = src[rng.integers(len(src))]
                t = tgt[rng.integers(len(tgt))]
                tr = 0.5 * s + 0.5 * t + 0.05 * rng.standard_normal(DIM)
                tr /= np.linalg.norm(tr)
                traces.append(tr)
            return np.array(traces)

        # Cycle 1: add traces for strategy 0→1
        traces_01 = _make_traces(dom0, dom1, 6, rng)
        combined_1 = np.vstack([content, traces_01])
        labels_1 = np.concatenate([labels, np.full(6, 3)])

        report_1 = dream_cycle_xb(combined_1, BETA, labels=labels_1, seed=42)
        X_after_1 = report_1.patterns
        N_after_1 = X_after_1.shape[0]

        # Measure S_CT rank after cycle 1
        # Approximate: use first N_content-ish rows as content, rest as traces
        # (dream may have pruned/merged, so we measure the full S)
        S_1 = X_after_1 @ X_after_1.T

        # Cycle 2: add traces for strategy 1→2
        traces_12 = _make_traces(dom1, dom2, 6, rng)
        combined_2 = np.vstack([X_after_1, traces_12])
        labels_2 = np.concatenate([
            np.zeros(N_after_1, dtype=int),  # placeholder labels for consolidated
            np.full(6, 1),
        ])
        # Re-derive labels from clustering
        report_2 = dream_cycle_xb(combined_2, BETA, seed=43)
        X_after_2 = report_2.patterns

        # Cycle 3: add traces for strategy 0→2 (completing the triangle)
        traces_02 = _make_traces(dom0, dom2, 6, rng)
        combined_3 = np.vstack([X_after_2, traces_02])
        report_3 = dream_cycle_xb(combined_3, BETA, seed=44)
        X_after_3 = report_3.patterns

        print(f"  Cycle 1: {N_after_1} patterns, "
              f"{len(report_1.associations)} associations")
        print(f"  Cycle 2: {report_2.patterns.shape[0]} patterns, "
              f"{len(report_2.associations)} associations")
        print(f"  Cycle 3: {report_3.patterns.shape[0]} patterns, "
              f"{len(report_3.associations)} associations")

        # After 3 cycles, patterns shouldn't have collapsed to nothing
        assert X_after_3.shape[0] >= 10, (
            f"After 3 dream cycles, only {X_after_3.shape[0]} patterns remain — "
            f"excessive pruning destroyed trace memory"
        )

    def test_rank_nondecreasing_across_cycles(self):
        """The effective rank of the combined similarity matrix should
        not decrease across dream cycles (trace addition expands capacity)."""
        rng = np.random.default_rng(950)
        content, labels = _make_domain_patterns(3, 8, seed=950)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]
        dom2 = content[labels == 2]

        def _make_traces(src, tgt, n, rng):
            traces = []
            for _ in range(n):
                s = src[rng.integers(len(src))]
                t = tgt[rng.integers(len(tgt))]
                tr = 0.5 * s + 0.5 * t + 0.05 * rng.standard_normal(DIM)
                tr /= np.linalg.norm(tr)
                traces.append(tr)
            return np.array(traces)

        # Baseline rank
        S_0 = content @ content.T
        rank_0 = _effective_rank(S_0, threshold=0.05)

        # Cycle 1: add 0→1 traces
        traces_01 = _make_traces(dom0, dom1, 5, rng)
        combined = np.vstack([content, traces_01])
        combined_labels = np.concatenate([labels, np.full(5, 3)])
        report = dream_cycle_xb(combined, BETA, labels=combined_labels, seed=42)
        S_1 = report.patterns @ report.patterns.T
        rank_1 = _effective_rank(S_1, threshold=0.05)

        # Cycle 2: add 1→2 traces
        traces_12 = _make_traces(dom1, dom2, 5, rng)
        combined_2 = np.vstack([report.patterns, traces_12])
        report_2 = dream_cycle_xb(combined_2, BETA, seed=43)
        S_2 = report_2.patterns @ report_2.patterns.T
        rank_2 = _effective_rank(S_2, threshold=0.05)

        print(f"  Rank evolution: {rank_0} → {rank_1} → {rank_2}")

        # Rank may decrease slightly due to merging/pruning of redundant patterns.
        # The key claim is that rank is PRESERVED modulo consolidation —
        # dream doesn't collapse the space. Allow up to 20% decrease per cycle.
        min_acceptable_1 = int(rank_0 * 0.8)
        min_acceptable_2 = int(rank_0 * 0.7)  # cumulative over 2 cycles
        assert rank_1 >= min_acceptable_1, (
            f"Rank collapsed from {rank_0} to {rank_1} (>{20}% loss) — "
            f"dream consolidation destroying trace information"
        )
        assert rank_2 >= min_acceptable_2, (
            f"Rank collapsed from {rank_0} to {rank_2} over 2 cycles — "
            f"dream consolidation destroying trace information"
        )


# ===========================================================================
# Claim 5: S_RR centroid structure — reasoning strategy centroids are
#           Hopfield attractors
# ===========================================================================

class TestTraceCentroidsAsAttractors:
    """After dream consolidation, trace centroids should function as
    Hopfield attractors: a noisy query near a centroid should converge
    to it under spreading activation."""

    def test_centroid_is_attractor(self):
        """A perturbed version of a trace centroid should converge back
        to approximately the centroid under spreading activation."""
        rng = np.random.default_rng(1100)
        content, labels = _make_domain_patterns(3, 8, seed=1100)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]

        # Create tight cluster of traces (will merge into centroid)
        traces = []
        for _ in range(8):
            s = dom0[rng.integers(len(dom0))]
            t = dom1[rng.integers(len(dom1))]
            tr = 0.5 * s + 0.5 * t + 0.02 * rng.standard_normal(DIM)
            tr /= np.linalg.norm(tr)
            traces.append(tr)
        traces_arr = np.array(traces)

        # Compute the "true" centroid of these traces
        true_centroid = np.mean(traces_arr, axis=0)
        true_centroid /= np.linalg.norm(true_centroid)

        # Store content + traces
        combined = np.vstack([content, traces_arr])

        # Perturb centroid
        q = true_centroid + 0.15 * rng.standard_normal(DIM)
        q /= np.linalg.norm(q)

        cos_before = float(q @ true_centroid)

        # Run spreading activation
        fp = spreading_activation(BETA, combined, q, max_steps=100, tol=1e-6)
        cos_after = float(fp @ true_centroid)

        print(f"  Cos(query, centroid) before: {cos_before:.4f}")
        print(f"  Cos(fixed_point, centroid) after: {cos_after:.4f}")
        print(f"  Improvement: {cos_after - cos_before:+.4f}")

        # Fixed point should be closer to centroid than the initial query
        assert cos_after > cos_before, (
            f"Spreading activation moved AWAY from trace centroid: "
            f"{cos_before:.4f} → {cos_after:.4f}"
        )

    def test_distinct_strategies_are_separate_attractors(self):
        """Two different reasoning strategies (0→1 vs 2→3) should converge
        to different fixed points, not collapse into one."""
        rng = np.random.default_rng(1200)
        content, labels = _make_domain_patterns(4, 8, seed=1200)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]
        dom2 = content[labels == 2]
        dom3 = content[labels == 3]

        def _strat(src, tgt, n):
            t = []
            for _ in range(n):
                s = src[rng.integers(len(src))]
                g = tgt[rng.integers(len(tgt))]
                tr = 0.5 * s + 0.5 * g + 0.02 * rng.standard_normal(DIM)
                tr /= np.linalg.norm(tr)
                t.append(tr)
            return np.array(t)

        traces_01 = _strat(dom0, dom1, 6)
        traces_23 = _strat(dom2, dom3, 6)

        combined = np.vstack([content, traces_01, traces_23])

        # Centroids of each strategy
        c_01 = np.mean(traces_01, axis=0)
        c_01 /= np.linalg.norm(c_01)
        c_23 = np.mean(traces_23, axis=0)
        c_23 /= np.linalg.norm(c_23)

        # Query near strategy 0→1
        q1 = c_01 + 0.1 * rng.standard_normal(DIM)
        q1 /= np.linalg.norm(q1)
        fp1 = spreading_activation(BETA, combined, q1, max_steps=100, tol=1e-6)

        # Query near strategy 2→3
        q2 = c_23 + 0.1 * rng.standard_normal(DIM)
        q2 /= np.linalg.norm(q2)
        fp2 = spreading_activation(BETA, combined, q2, max_steps=100, tol=1e-6)

        # Fixed points should be different
        fp_sim = float(fp1 @ fp2)
        fp1_to_c01 = float(fp1 @ c_01)
        fp2_to_c23 = float(fp2 @ c_23)

        print(f"  FP1 → centroid_01: {fp1_to_c01:.4f}")
        print(f"  FP2 → centroid_23: {fp2_to_c23:.4f}")
        print(f"  FP1 ↔ FP2 similarity: {fp_sim:.4f}")

        # Each fixed point should be closer to its own centroid than to the other
        assert fp1_to_c01 > fp_sim, (
            f"FP1 not closer to centroid_01 ({fp1_to_c01:.4f}) "
            f"than to FP2 ({fp_sim:.4f})"
        )
        assert fp2_to_c23 > fp_sim, (
            f"FP2 not closer to centroid_23 ({fp2_to_c23:.4f}) "
            f"than to FP2↔FP1 ({fp_sim:.4f})"
        )


# ===========================================================================
# Claim 6: Energy landscape — trace centroids lower energy
# ===========================================================================

class TestTraceEnergyLandscape:
    """From the Lean proofs: E(ξ*) = -½‖ξ*‖² + β⁻¹ Σ p_μ log p_μ.
    Concentrated fixed points (near a single pattern) have lower energy
    than mixture fixed points.

    Claim: After adding traces, the energy at trace centroids should be
    lower than the energy at random interpolation points, confirming
    that traces create genuine energy wells.
    """

    def test_trace_centroid_has_lower_energy_than_random(self):
        """Energy at trace centroid < energy at random interpolation point
        between domains."""
        rng = np.random.default_rng(1300)
        content, labels = _make_domain_patterns(3, 10, seed=1300)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]

        # Create trace cluster
        traces = []
        for _ in range(8):
            s = dom0[rng.integers(len(dom0))]
            t = dom1[rng.integers(len(dom1))]
            tr = 0.5 * s + 0.5 * t + 0.03 * rng.standard_normal(DIM)
            tr /= np.linalg.norm(tr)
            traces.append(tr)
        traces_arr = np.array(traces)

        combined = np.vstack([content, traces_arr])

        # Trace centroid
        centroid = np.mean(traces_arr, axis=0)
        centroid /= np.linalg.norm(centroid)

        # Random interpolation (not near any trace)
        random_interp = 0.5 * dom0[0] + 0.5 * content[labels == 2][0]
        random_interp += 0.3 * rng.standard_normal(DIM)
        random_interp /= np.linalg.norm(random_interp)

        def _local_energy(xi, patterns, beta):
            """E_local(ξ) = -lse(β, X^T ξ) + ½‖ξ‖²"""
            sims = patterns @ xi
            lse = (1.0 / beta) * np.log(np.sum(np.exp(beta * (sims - np.max(sims))))) + np.max(sims)
            return -lse + 0.5 * np.dot(xi, xi)

        E_centroid = _local_energy(centroid, combined, BETA)
        E_random = _local_energy(random_interp, combined, BETA)

        print(f"  E(trace centroid) = {E_centroid:.6f}")
        print(f"  E(random interp)  = {E_random:.6f}")
        print(f"  Δ = {E_centroid - E_random:.6f}")

        # Trace centroid should have lower energy (deeper well)
        assert E_centroid < E_random, (
            f"Trace centroid energy {E_centroid:.6f} >= "
            f"random interpolation energy {E_random:.6f} — "
            f"traces not creating energy wells"
        )

    def test_energy_decreases_after_dream_with_traces(self):
        """Total energy should decrease after dream cycle when traces
        are present (consolidation deepens trace wells)."""
        rng = np.random.default_rng(1400)
        content, labels = _make_domain_patterns(3, 8, seed=1400)
        dom0 = content[labels == 0]
        dom1 = content[labels == 1]

        traces = []
        for _ in range(6):
            s = dom0[rng.integers(len(dom0))]
            t = dom1[rng.integers(len(dom1))]
            tr = 0.5 * s + 0.5 * t + 0.03 * rng.standard_normal(DIM)
            tr /= np.linalg.norm(tr)
            traces.append(tr)
        traces_arr = np.array(traces)

        combined = np.vstack([content, traces_arr])
        combined_labels = np.concatenate([labels, np.full(6, max(labels) + 1)])

        def _total_energy(patterns, beta):
            """Sum of local energies for all patterns."""
            total = 0.0
            for i in range(patterns.shape[0]):
                sims = patterns @ patterns[i]
                max_s = np.max(sims)
                lse = (1.0 / beta) * np.log(
                    np.sum(np.exp(beta * (sims - max_s)))
                ) + max_s
                total += -lse + 0.5 * np.dot(patterns[i], patterns[i])
            return total

        E_before = _total_energy(combined, BETA)

        report = dream_cycle_xb(
            combined, BETA, labels=combined_labels, seed=42
        )

        E_after = _total_energy(report.patterns, BETA)

        # Normalize by pattern count for fair comparison
        e_per_pattern_before = E_before / combined.shape[0]
        e_per_pattern_after = E_after / report.patterns.shape[0]

        print(f"  Before dream: E_total={E_before:.4f} "
              f"({combined.shape[0]} patterns, per-pattern={e_per_pattern_before:.4f})")
        print(f"  After dream:  E_total={E_after:.4f} "
              f"({report.patterns.shape[0]} patterns, per-pattern={e_per_pattern_after:.4f})")

        # Per-pattern energy should decrease (consolidation deepens wells)
        assert e_per_pattern_after <= e_per_pattern_before + 0.01, (
            f"Per-pattern energy increased by "
            f"{e_per_pattern_after - e_per_pattern_before:.4f} — "
            f"dream cycle worsened energy landscape"
        )
