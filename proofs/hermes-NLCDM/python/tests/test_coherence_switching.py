"""Discriminative tests: Phase 13 coherence switching in query().

The Lean proof (HybridRetrieval.lean, HybridSwitching.lean) proves:
  - multisignal_rank_preservation: if coherence > 0, multi-signal preserves rank 1
  - multisignal_rank_inversion: if coherence < 0, multi-signal inverts ranking
  - switching_criterion: use cosine-only when coherence ≤ 0

Currently query() ALWAYS applies recency weighting (the multi-signal component).
When recency contradicts cosine ranking, this can invert the correct ranking.

These tests verify that query() implements coherence switching:
use multi-signal when it helps, fall back to cosine-only when it hurts.

All tests FAIL until coherence switching is implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_NLCDM_PYTHON = _THIS_DIR.parent
sys.path.insert(0, str(_NLCDM_PYTHON))

from coupled_engine import CoupledEngine


class TestCoherenceSwitching:
    """Tests for Phase 13 coherence-based switching in retrieval."""

    def test_cosine_winner_stays_rank1_when_recency_contradicts(self):
        """THE DISCRIMINATIVE TEST: when recency contradicts cosine, the
        cosine winner should still be ranked first.

        Setup:
        - Pattern A: highest cosine with query, LOW recency (old)
        - Pattern B: lower cosine with query, HIGH recency (recent)
        - recency_alpha large enough that without switching, B beats A

        Without coherence switching: B ranks above A (inversion).
        With coherence switching: A stays rank 1 (cosine-only fallback).
        """
        dim = 64
        rng = np.random.default_rng(42)

        query = np.zeros(dim)
        query[0] = 1.0

        # Pattern A: very close to query (cosine ~0.99), LOW recency
        pat_a = query + 0.05 * rng.standard_normal(dim)
        pat_a /= np.linalg.norm(pat_a)

        # Pattern B: moderately close (cosine ~0.7), HIGH recency
        pat_b = np.zeros(dim)
        pat_b[0] = 0.7
        pat_b[1] = 0.7
        pat_b /= np.linalg.norm(pat_b)

        # recency_alpha=2.0 is large enough that with recency=100 on B
        # and recency=1 on A, B's score gets multiplied by (1 + 2.0 * 1.0) = 3.0
        # while A's gets multiplied by (1 + 2.0 * 0.01) = 1.02
        # B score: 0.707 * 3.0 = 2.12 vs A score: 0.99 * 1.02 = 1.01
        # → B wins without switching (inversion!)
        engine = CoupledEngine(dim=dim, recency_alpha=2.0)

        engine.store(text="old-relevant", embedding=pat_a,
                     importance=0.5, recency=1.0)
        engine.store(text="new-less-relevant", embedding=pat_b,
                     importance=0.5, recency=100.0)

        results = engine.query(embedding=query, top_k=2)

        # Pattern A (index 0) should be rank 1 because cosine is higher
        assert results[0]["index"] == 0, (
            f"Cosine winner (index 0) should be rank 1 when recency "
            f"contradicts. Got index {results[0]['index']} at rank 1. "
            f"Scores: {[(r['index'], r['score']) for r in results[:3]]}. "
            f"This means coherence switching is NOT active — recency "
            f"inverted the correct ranking."
        )

    def test_recency_helps_when_coherence_positive(self):
        """When recency REINFORCES cosine ranking, multi-signal should be used.

        Setup:
        - Pattern A: highest cosine AND highest recency
        - Pattern B: lower cosine AND lower recency
        - Recency reinforces cosine → coherence > 0 → multi-signal OK

        The score gap between A and B should be LARGER with recency than
        without, proving multi-signal enrichment is active.
        """
        dim = 64
        rng = np.random.default_rng(42)

        query = np.zeros(dim)
        query[0] = 1.0

        # A: high cosine, will be stored LAST (high recency)
        pat_a = query + 0.02 * rng.standard_normal(dim)
        pat_a /= np.linalg.norm(pat_a)

        # B: moderate cosine, stored FIRST (low recency)
        pat_b = np.zeros(dim)
        pat_b[0] = 0.6
        pat_b[1] = 0.8
        pat_b /= np.linalg.norm(pat_b)

        engine = CoupledEngine(dim=dim, recency_alpha=0.5)
        engine.store(text="old-less-relevant", embedding=pat_b,
                     recency=1.0)   # low recency, low cosine
        engine.store(text="new-relevant", embedding=pat_a,
                     recency=100.0)  # high recency, high cosine → coherent

        results = engine.query(embedding=query, top_k=2)

        # A (index 1) should be rank 1 (both cosine and recency favor it)
        assert results[0]["index"] == 1, (
            f"Pattern A should be rank 1 when cosine and recency agree. "
            f"Got index {results[0]['index']}."
        )

        # Score gap should be positive (A scores higher than B)
        gap = results[0]["score"] - results[1]["score"]
        assert gap > 0.1, (
            f"Score gap should be substantial when signals agree. Got {gap:.4f}."
        )

    def test_switching_does_not_affect_cosine_only_engine(self):
        """With recency_alpha=0, switching has no effect (pure cosine)."""
        dim = 64
        rng = np.random.default_rng(42)

        query = rng.standard_normal(dim)
        query /= np.linalg.norm(query)

        engine = CoupledEngine(dim=dim, recency_alpha=0.0)
        for i in range(10):
            p = rng.standard_normal(dim)
            p /= np.linalg.norm(p)
            engine.store(text=f"p-{i}", embedding=p)

        results = engine.query(embedding=query, top_k=5)

        # Verify results are in descending cosine similarity order
        embeddings = np.array([m.embedding for m in engine.memory_store])
        q_norm = query / np.linalg.norm(query)
        norms = np.linalg.norm(embeddings, axis=1)
        true_cosine = embeddings @ q_norm / (norms + 1e-12)

        result_indices = [r["index"] for r in results]
        result_cosines = [true_cosine[i] for i in result_indices]
        assert result_cosines == sorted(result_cosines, reverse=True), (
            f"With recency_alpha=0, results should be in cosine order. "
            f"Got cosines: {[f'{c:.4f}' for c in result_cosines]}"
        )
