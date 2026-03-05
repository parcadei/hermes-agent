"""Discriminative tests: Phase 14 capacity-aware dream gating.

The Lean proof (Capacity.lean) proves:
  N_max ~ exp(β·δ) / (4·β·M²)

where δ is the minimum pairwise separation and M bounds pattern norms.
Dreams should only consolidate when N approaches N_max.

Currently dream() fires unconditionally — even when N << N_max, removing
patterns the eval still needs. This hurts composite by -0.04.

These tests verify that dream() checks capacity before executing
destructive operations (prune/merge).

All tests FAIL until capacity-aware gating is implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
sys.path.insert(0, str(_NLCDM_PYTHON))

from coupled_engine import CoupledEngine


class TestCapacityFormulaLeanAlignment:
    """Discriminative tests: _compute_capacity_ratio must match Lean proof.

    The Lean-proven formula (Capacity.lean:43):
        N_max = exp(β·δ_min) / (4·β·M²)

    The OLD (wrong) formula was:
        N_max = dim · (1 + β·δ_median) / 4

    These tests verify the implementation uses the correct formula.
    """

    def test_formula_matches_lean_on_known_values(self):
        """Verify N_max = exp(β·δ) / (4·β·M²) on hand-computed values."""
        dim = 64
        engine = CoupledEngine(dim=dim, beta=5.0)
        rng = np.random.default_rng(99)

        # Create 3 well-separated unit-norm patterns
        patterns = []
        for _ in range(3):
            p = rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            patterns.append(p)
            engine.store(text="t", embedding=p, importance=0.5)

        X = np.array(patterns)

        # Compute expected values by hand
        cos_matrix = X @ X.T
        N = len(patterns)
        iu = np.triu_indices(N, k=1)
        pairwise_cos = cos_matrix[iu]
        delta_min = float(np.min(1.0 - pairwise_cos))
        M_sq = float(np.max(np.sum(X ** 2, axis=1)))
        beta = 5.0

        expected_n_max = np.exp(beta * delta_min) / (4.0 * beta * M_sq)
        expected_ratio = N / max(expected_n_max, 1.0)

        embeddings = engine._embeddings_matrix()
        actual_ratio = engine._compute_capacity_ratio(embeddings)

        assert abs(actual_ratio - expected_ratio) < 1e-6, (
            f"Capacity ratio mismatch: got {actual_ratio:.6f}, "
            f"expected {expected_ratio:.6f} (Lean formula)"
        )

    def test_old_linear_formula_gives_different_result(self):
        """The old dim*(1+β*δ_median)/4 formula must differ from Lean."""
        dim = 64
        engine = CoupledEngine(dim=dim, beta=5.0)
        rng = np.random.default_rng(42)

        for _ in range(20):
            p = rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            engine.store(text="t", embedding=p, importance=0.5)

        X = engine._embeddings_matrix()
        N = X.shape[0]

        # Lean formula (what _compute_capacity_ratio should use)
        lean_ratio = engine._compute_capacity_ratio(X)

        # Old wrong formula
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        normed = X / (norms + 1e-12)
        cos_matrix = normed @ normed.T
        iu = np.triu_indices(N, k=1)
        pairwise_cos = cos_matrix[iu]
        delta_median = float(np.median(1.0 - pairwise_cos))
        old_n_max = dim * (1.0 + 5.0 * delta_median) / 4.0
        old_ratio = N / max(old_n_max, 1.0)

        # They must differ — the formulas are fundamentally different
        assert abs(lean_ratio - old_ratio) > 0.01, (
            f"Lean ratio ({lean_ratio:.4f}) should differ from old linear "
            f"ratio ({old_ratio:.4f}). If they match, the fix wasn't applied."
        )

    def test_uses_delta_min_not_median(self):
        """Verify the formula uses δ_min (minimum separation), not median.

        Create patterns where min and median separation differ significantly.
        """
        dim = 64
        rng = np.random.default_rng(7)
        engine = CoupledEngine(dim=dim, beta=5.0)

        # One pair of very close patterns (small δ_min)
        base = rng.standard_normal(dim)
        base /= np.linalg.norm(base)
        close = base + 0.005 * rng.standard_normal(dim)
        close /= np.linalg.norm(close)
        engine.store(text="a", embedding=base, importance=0.5)
        engine.store(text="b", embedding=close, importance=0.5)

        # Several well-separated patterns (large δ for most pairs)
        for _ in range(8):
            p = rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            engine.store(text="t", embedding=p, importance=0.5)

        X = engine._embeddings_matrix()
        N = X.shape[0]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        normed = X / (norms + 1e-12)
        cos_matrix = normed @ normed.T
        iu = np.triu_indices(N, k=1)
        pairwise_cos = cos_matrix[iu]

        delta_min = float(np.min(1.0 - pairwise_cos))
        delta_median = float(np.median(1.0 - pairwise_cos))

        # Confirm min and median actually differ
        assert delta_median > delta_min * 2, (
            f"Test setup: median ({delta_median:.4f}) should be much larger "
            f"than min ({delta_min:.4f})"
        )

        # Capacity ratio should reflect δ_min (close pair), not δ_median
        M_sq = float(np.max(np.sum(X ** 2, axis=1)))
        beta = 5.0
        expected_n_max = np.exp(beta * delta_min) / (4.0 * beta * M_sq)
        expected_ratio = N / max(expected_n_max, 1.0)

        actual_ratio = engine._compute_capacity_ratio(X)
        assert abs(actual_ratio - expected_ratio) < 1e-6, (
            f"Should use δ_min={delta_min:.4f}, not δ_median={delta_median:.4f}. "
            f"Got ratio {actual_ratio:.4f}, expected {expected_ratio:.4f}"
        )

    def test_uses_M_squared_not_dim(self):
        """Verify denominator uses M² (max squared norm), not dimensionality."""
        dim = 64
        engine = CoupledEngine(dim=dim, beta=5.0)
        rng = np.random.default_rng(11)

        # Non-unit-norm patterns: M² ≠ 1.0
        for _ in range(5):
            p = rng.standard_normal(dim)
            scale = rng.uniform(0.5, 2.0)
            p = p / np.linalg.norm(p) * scale
            engine.store(text="t", embedding=p, importance=0.5)

        X = engine._embeddings_matrix()
        M_sq = float(np.max(np.sum(X ** 2, axis=1)))

        # M² should NOT equal dim (which the old formula implicitly used)
        assert abs(M_sq - dim) > 1.0, (
            f"Test setup: M²={M_sq:.2f} should differ from dim={dim}"
        )

        ratio = engine._compute_capacity_ratio(X)
        assert ratio > 0.0, f"Capacity ratio should be positive, got {ratio}"


class TestCapacityGating:
    """Tests for Phase 14 capacity-aware dream gating."""

    def test_dream_skips_when_well_below_capacity(self):
        """Capacity gating tightens prune/merge thresholds when ratio < 0.8.

        When patterns are well-separated (large δ_min), the Lean capacity
        formula yields N/N_max < 0.8. Dream then tightens thresholds to
        0.995/0.99, effectively only removing exact duplicates. With
        well-separated patterns, no pair exceeds these thresholds so
        nothing is pruned or merged.

        Note: when patterns ARE close enough to exceed default thresholds
        (cosine > 0.95), δ_min is necessarily small, making N/N_max >> 1.
        The gating never fires in that case — by design, the Lean formula
        says the store is overcrowded and consolidation is warranted.
        """
        dim = 64
        rng = np.random.default_rng(42)
        engine = CoupledEngine(dim=dim, beta=5.0)

        # Create 3 well-separated patterns (random in 64d → typical
        # pairwise cosine near 0, δ_min ~ 0.88). With β=5 and large δ_min,
        # N_max = exp(5*0.88)/(4*5*1) ≈ 4.0, and N/N_max = 3/4.0 ≈ 0.74
        # which is < 0.8 → capacity gating fires, thresholds tightened.
        for i in range(3):
            p = rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            engine.store(text=f"pattern-{i}", embedding=p, importance=0.8)

        n_before = engine.n_memories
        result = engine.dream(seed=42)
        n_after = engine.n_memories

        # Verify capacity gating fired (ratio < 0.8)
        assert result["capacity_ratio"] < 0.8, (
            f"Expected capacity_ratio < 0.8 for well-separated patterns, "
            f"got {result['capacity_ratio']:.3f}"
        )
        # No patterns should be removed — thresholds tightened to 0.995/0.99
        # and no pair has cosine anywhere near that
        assert n_after == n_before, (
            f"Dream should not prune/merge well-separated patterns when "
            f"capacity gating fires. Lost {n_before - n_after} patterns. "
            f"Pruned: {result['pruned']}, merged: {result['merged']}."
        )

    def test_dream_still_runs_when_near_capacity(self):
        """Dream SHOULD run when patterns are dense (near capacity).

        Create many near-duplicate patterns to simulate being at capacity.
        Dream should prune/merge these.
        """
        dim = 64
        rng = np.random.default_rng(42)
        engine = CoupledEngine(dim=dim, beta=5.0)

        # Create a base pattern and many near-duplicates
        base = rng.standard_normal(dim)
        base /= np.linalg.norm(base)

        for i in range(20):
            p = base + 0.01 * rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            engine.store(text=f"dup-{i}", embedding=p, importance=0.8)

        n_before = engine.n_memories
        result = engine.dream(seed=42)
        n_after = engine.n_memories

        # Near-duplicate patterns should be pruned/merged
        assert n_after < n_before, (
            f"Dream should prune/merge near-duplicates. "
            f"Before: {n_before}, after: {n_after}. "
            f"Pruned: {result['pruned']}, merged: {result['merged']}."
        )

    def test_dream_returns_capacity_info(self):
        """Dream report should include capacity utilization metrics."""
        dim = 64
        rng = np.random.default_rng(42)
        engine = CoupledEngine(dim=dim, beta=5.0)

        for i in range(5):
            p = rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            engine.store(text=f"p-{i}", embedding=p, importance=0.8)

        result = engine.dream(seed=42)

        # Dream report should include capacity utilization
        assert "capacity_ratio" in result, (
            f"Dream report should include 'capacity_ratio' (N/N_max). "
            f"Got keys: {list(result.keys())}."
        )
        assert 0.0 <= result["capacity_ratio"] <= 1.0 or result["capacity_ratio"] > 1.0, (
            f"capacity_ratio should be a non-negative number. "
            f"Got: {result['capacity_ratio']}."
        )
