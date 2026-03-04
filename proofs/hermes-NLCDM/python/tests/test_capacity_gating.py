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


class TestCapacityGating:
    """Tests for Phase 14 capacity-aware dream gating."""

    def test_dream_skips_when_well_below_capacity(self):
        """THE DISCRIMINATIVE TEST: dream() should NOT prune/merge when
        N is well below N_max.

        Creates highly similar patterns (cosine ~0.97) that the default
        dream params (prune=0.95, merge=0.9) would prune/merge. But with
        only 15 patterns and β=5.0, we are well below Hopfield capacity.
        Dream should skip destructive ops due to capacity gating.
        """
        dim = 64
        rng = np.random.default_rng(42)
        engine = CoupledEngine(dim=dim, beta=5.0)

        # Create 3 clusters of 5 patterns each, with HIGH intra-cluster
        # similarity (~0.97). noise=0.02 in 64d gives cosine ~0.97, well
        # above prune_threshold=0.95. Without capacity gating, dream WILL
        # prune these. With gating, N=15 is far below N_max.
        for cluster in range(3):
            centroid = rng.standard_normal(dim)
            centroid /= np.linalg.norm(centroid)
            for _ in range(5):
                p = centroid + 0.02 * rng.standard_normal(dim)
                p /= np.linalg.norm(p)
                engine.store(text=f"c{cluster}", embedding=p, importance=0.8)

        n_before = engine.n_memories
        result = engine.dream(seed=42)
        n_after = engine.n_memories

        # With N=15 well below capacity, dream should not remove patterns
        assert n_after == n_before, (
            f"Dream should not prune/merge when N ({n_before}) is well "
            f"below capacity. Lost {n_before - n_after} patterns. "
            f"Pruned: {result['pruned']}, merged: {result['merged']}. "
            f"Capacity gating should prevent destructive operations."
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
