"""Discriminative tests: Phase 10 transfer retrieval via Hopfield update.

The Lean proof transfer_via_bridge (TransferDynamics.lean) proves:
  cosine(T(ξ), target) ≥ σ / (1 + (N-1)·exp(-β·δ))

where T(ξ) is the Hopfield update of query ξ. This gives a quantitative
cross-domain retrieval guarantee: if the query aligns with a bridge pattern
that connects to a target domain, the Hopfield update provably transfers
signal to that domain.

These tests verify that query_transfer() uses the Hopfield update as a
retrieval mechanism, boosting cross-domain patterns connected via bridges.

All tests FAIL until query_transfer() is implemented on CoupledEngine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
sys.path.insert(0, str(_NLCDM_PYTHON))

from coupled_engine import CoupledEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge_scenario(dim=64, n_per_cluster=3, noise=0.05, seed=42):
    """Create two clusters with a bridge pattern between them.

    Geometry (first 2 dimensions carry the signal, rest is noise):
      d_a = e_0, d_b = e_1
      bridge = (d_a + d_b) / √2  (equal connection to both domains)
      query = d_a  (pure domain-A direction)

    Returns (patterns, bridge_idx, cluster_a, cluster_b, query).
    """
    rng = np.random.default_rng(seed)

    d_a = np.zeros(dim)
    d_a[0] = 1.0
    d_b = np.zeros(dim)
    d_b[1] = 1.0

    patterns = []
    cluster_a, cluster_b = [], []

    # Cluster A: near d_a
    for _ in range(n_per_cluster):
        p = d_a + noise * rng.standard_normal(dim)
        p /= np.linalg.norm(p)
        cluster_a.append(len(patterns))
        patterns.append(p)

    # Bridge pattern at midpoint
    bridge = (d_a + d_b) / np.sqrt(2)
    bridge += (noise / 2) * rng.standard_normal(dim)
    bridge /= np.linalg.norm(bridge)
    bridge_idx = len(patterns)
    patterns.append(bridge)

    # Cluster B: near d_b
    for _ in range(n_per_cluster):
        p = d_b + noise * rng.standard_normal(dim)
        p /= np.linalg.norm(p)
        cluster_b.append(len(patterns))
        patterns.append(p)

    query = d_a.copy()
    return np.array(patterns), bridge_idx, cluster_a, cluster_b, query


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTransferRetrieval:
    """Tests for Phase 10 transfer retrieval via Hopfield update."""

    def test_query_transfer_returns_valid_results(self):
        """query_transfer returns results with same format as query()."""
        engine = CoupledEngine(dim=64)
        rng = np.random.default_rng(42)
        for i in range(10):
            p = rng.standard_normal(64)
            p /= np.linalg.norm(p)
            engine.store(text=f"p-{i}", embedding=p)

        query = rng.standard_normal(64)
        query /= np.linalg.norm(query)

        results = engine.query_transfer(embedding=query, top_k=5)

        assert isinstance(results, list)
        assert 0 < len(results) <= 5
        for r in results:
            assert "index" in r
            assert "score" in r
            assert "text" in r
            assert isinstance(r["index"], int)
            assert isinstance(r["score"], float)

    def test_transfer_empty_store(self):
        """Transfer on empty store returns empty list."""
        engine = CoupledEngine(dim=64)
        query = np.random.default_rng(42).standard_normal(64)
        assert engine.query_transfer(embedding=query) == []

    def test_transfer_boosts_bridge_connected_patterns(self):
        """Patterns connected via bridge get higher scores under transfer.

        Setup: cluster A near e_0, cluster B near e_1, bridge at midpoint.
        Query = e_0 (pure domain A).

        The Hopfield update T(ξ) incorporates the bridge component, giving
        T(ξ) a nonzero e_1 component. This boosts cosine(T(ξ), B_pattern)
        above cosine(query, B_pattern) for cluster B patterns.
        """
        dim = 64
        patterns, _bridge_idx, _cluster_a, cluster_b, query = (
            _make_bridge_scenario(dim=dim, n_per_cluster=3, noise=0.05, seed=42)
        )

        # Engine 1: pure cosine scoring
        engine_cos = CoupledEngine(dim=dim, beta=5.0)
        for i, p in enumerate(patterns):
            engine_cos.store(text=f"p-{i}", embedding=p)

        n = engine_cos.n_memories
        cosine_results = engine_cos.query(embedding=query, top_k=n)
        cosine_scores = {r["index"]: r["score"] for r in cosine_results}

        # Engine 2: transfer scoring (fresh engine, no side effects)
        engine_tr = CoupledEngine(dim=dim, beta=5.0)
        for i, p in enumerate(patterns):
            engine_tr.store(text=f"p-{i}", embedding=p)

        transfer_results = engine_tr.query_transfer(embedding=query, top_k=n)
        transfer_scores = {r["index"]: r["score"] for r in transfer_results}

        # At least one B pattern must have higher score under transfer
        improvements = []
        for idx in cluster_b:
            cos_s = cosine_scores.get(idx, 0.0)
            tr_s = transfer_scores.get(idx, 0.0)
            improvements.append(tr_s - cos_s)

        assert any(imp > 0.01 for imp in improvements), (
            f"Transfer should boost at least one B pattern. "
            f"Improvements: {[f'{x:.4f}' for x in improvements]}. "
            f"Cosine: {[f'{cosine_scores.get(i,0):.4f}' for i in cluster_b]}. "
            f"Transfer: {[f'{transfer_scores.get(i,0):.4f}' for i in cluster_b]}."
        )

    def test_transfer_scores_differ_from_cosine(self):
        """Transfer scores must differ from pure cosine (Hopfield is active).

        If query_transfer just returned cosine scores, this test would fail.
        """
        dim = 64
        patterns, _, _, _, query = _make_bridge_scenario(dim=dim, seed=99)

        engine_cos = CoupledEngine(dim=dim, beta=5.0)
        engine_tr = CoupledEngine(dim=dim, beta=5.0)
        for i, p in enumerate(patterns):
            engine_cos.store(text=f"p-{i}", embedding=p)
            engine_tr.store(text=f"p-{i}", embedding=p)

        n = engine_cos.n_memories
        cos_results = engine_cos.query(embedding=query, top_k=n)
        tr_results = engine_tr.query_transfer(embedding=query, top_k=n)

        cos_scores = sorted((r["index"], r["score"]) for r in cos_results)
        tr_scores = sorted((r["index"], r["score"]) for r in tr_results)

        # At least one pattern must have a different score
        diffs = [abs(c[1] - t[1]) for c, t in zip(cos_scores, tr_scores)]
        assert max(diffs) > 1e-6, (
            f"All scores identical — query_transfer is not using Hopfield. "
            f"Max diff: {max(diffs):.8f}"
        )
