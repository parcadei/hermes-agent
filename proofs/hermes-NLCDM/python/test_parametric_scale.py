"""Parametric scale test: sweep 100→10k with pre-cached embeddings.

Loads pre-embedded vectors from embeddings_170k.npy (run embed_170k.py first),
then runs the full proof-backed pipeline at varied scales to produce timing
and consolidation curves.

Scale points: 100, 200, 300, 500, 1k, 2k, 5k, 10k

At each scale N, tests:
  1. Store N patterns → timing
  2. Query 20 patterns → accuracy + timing
  3. Single dream cycle → timing, pruned/merged counts
  4. Streaming simulation → ingest at N/10 per day for 10 days
  5. Multi-dream convergence → 3 cycles, pattern count trajectory

Dataset: agentlans/high-quality-english-sentences (170k sentences)
Embeddings: Qwen3-Embedding-0.6B, 1024-dim, float32, pre-cached to .npy

Usage:
    .venv/bin/python -m pytest test_parametric_scale.py -v -s --tb=short
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from coupled_engine import CoupledEngine
from dream_ops import (
    DreamReport,
    dream_cycle_xb,
    nrem_prune_xb,
    nrem_repulsion_xb,
)
from nlcdm_core import cosine_sim


_THIS_DIR = Path(__file__).resolve().parent
_NPY_PATH = _THIS_DIR / "embeddings_170k.npy"
_TXT_PATH = _THIS_DIR / "sentences_170k.txt"


# ---------------------------------------------------------------------------
# Data loading (cached across all tests in the module)
# ---------------------------------------------------------------------------

_VECTORS: np.ndarray | None = None
_SENTENCES: list[str] | None = None


def _load_vectors() -> np.ndarray:
    global _VECTORS
    if _VECTORS is None:
        if not _NPY_PATH.exists():
            pytest.skip(f"Run embed_170k.py first to generate {_NPY_PATH.name}")
        _VECTORS = np.load(_NPY_PATH).astype(np.float64)
        norms = np.linalg.norm(_VECTORS, axis=1, keepdims=True)
        _VECTORS = _VECTORS / np.maximum(norms, 1e-8)
    return _VECTORS


def _load_sentences() -> list[str]:
    global _SENTENCES
    if _SENTENCES is None:
        if not _TXT_PATH.exists():
            pytest.skip(f"Run embed_170k.py first to generate {_TXT_PATH.name}")
        with open(_TXT_PATH, encoding="utf-8") as f:
            _SENTENCES = [line.strip() for line in f]
    return _SENTENCES


@pytest.fixture(scope="module")
def vectors():
    return _load_vectors()


@pytest.fixture(scope="module")
def sentences():
    return _load_sentences()


# ---------------------------------------------------------------------------
# Scale ladder
# ---------------------------------------------------------------------------

SCALE_POINTS = [100, 200, 300, 500, 1_000, 2_000, 5_000, 10_000]


# ---------------------------------------------------------------------------
# Timing/result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScalePoint:
    n: int
    store_s: float = 0.0
    query_s: float = 0.0
    query_mean_score: float = 0.0
    dream_s: float = 0.0
    dream_n_out: int = 0
    dream_pruned: int = 0
    dream_merged: int = 0


def _print_scale_table(points: list[ScalePoint], title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'N':>6}  {'Store':>7}  {'Query':>7}  {'Score':>6}  "
          f"{'Dream':>7}  {'N_out':>6}  {'Pruned':>6}  {'Merged':>6}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}")
    for p in points:
        print(f"  {p.n:>6}  {p.store_s:>6.2f}s  {p.query_s:>6.3f}s  "
              f"{p.query_mean_score:>5.3f}  {p.dream_s:>6.1f}s  "
              f"{p.dream_n_out:>6}  {p.dream_pruned:>6}  {p.dream_merged:>6}")


# ===========================================================================
# Test Class 1: Full pipeline sweep
# ===========================================================================

class TestPipelineSweep:
    """Store → Query → Dream at each scale point."""

    @pytest.mark.parametrize("n", SCALE_POINTS)
    def test_pipeline_at_scale(self, n, vectors, sentences):
        """Full store/query/dream pipeline at scale N."""
        vecs = vectors[:n]
        texts = sentences[:n]

        engine = CoupledEngine(dim=1024, beta=5.0)

        # --- Store ---
        t0 = time.time()
        for text, emb in zip(texts, vecs):
            engine.store(text, emb, importance=0.5)
        store_s = time.time() - t0
        assert engine.n_memories == n

        # --- Query ---
        rng = np.random.default_rng(42)
        query_indices = rng.choice(n, size=min(20, n), replace=False)
        scores = []
        t0 = time.time()
        for idx in query_indices:
            results = engine.query(vecs[idx], top_k=5)
            if results:
                scores.append(results[0]["score"])
        query_s = time.time() - t0
        mean_score = float(np.mean(scores)) if scores else 0.0

        # --- Dream ---
        importances = rng.uniform(0.2, 0.8, size=n)
        t0 = time.time()
        dream_result = engine.dream(seed=42)
        dream_s = time.time() - t0
        n_out = engine.n_memories

        print(f"\n  N={n:>5}: store={store_s:.2f}s  query={query_s:.3f}s  "
              f"score={mean_score:.3f}  dream={dream_s:.1f}s  "
              f"N_out={n_out}  pruned={dream_result['pruned']}  "
              f"merged={dream_result['merged']}")

        # Invariants
        assert n_out <= n, "Dream increased pattern count"
        assert mean_score > 0.2 or n <= 100, f"Query score {mean_score:.3f} too low at N={n}"


# ===========================================================================
# Test Class 2: Dream operation isolation (find the O(N²) knee)
# ===========================================================================

class TestDreamOpScaling:
    """Time individual dream operations across scale points to find the bottleneck."""

    @pytest.mark.parametrize("n", SCALE_POINTS)
    def test_nrem_prune_scaling(self, n, vectors):
        """nrem_prune_xb timing at each scale point."""
        vecs = vectors[:n]
        importances = np.full(n, 0.5)

        t0 = time.time()
        X_out, kept = nrem_prune_xb(vecs, importances, threshold=0.95)
        elapsed = time.time() - t0

        n_pruned = n - len(kept)
        print(f"\n  prune N={n:>5}: {elapsed:.2f}s  kept={len(kept)}  pruned={n_pruned}")

        assert len(kept) <= n
        assert kept == sorted(kept)  # Contract P4

    @pytest.mark.parametrize("n", [100, 200, 300, 500, 1_000, 2_000])
    def test_nrem_repulsion_scaling(self, n, vectors):
        """nrem_repulsion_xb timing — capped at 2k (O(N²) is brutal above that)."""
        vecs = vectors[:n]
        importances = np.full(n, 0.5)

        t0 = time.time()
        X_out = nrem_repulsion_xb(vecs, importances, eta=0.01, min_sep=0.3)
        elapsed = time.time() - t0

        norms = np.linalg.norm(X_out, axis=1)
        assert np.all(norms > 0.5), "Repulsion collapsed a pattern"
        print(f"\n  repulsion N={n:>5}: {elapsed:.2f}s")


# ===========================================================================
# Test Class 3: Streaming at varied daily rates
# ===========================================================================

class TestStreamingVariedRates:
    """Streaming simulation at each scale point with daily dreams."""

    @pytest.mark.parametrize("n", [500, 1_000, 2_000, 5_000])
    def test_streaming_tenth_per_day(self, n, vectors, sentences):
        """Ingest N total at N/10 per day (10 days), dream each night."""
        rate = max(n // 10, 10)
        n_days = n // rate
        vecs = vectors[:n]
        texts = sentences[:n]

        rng = np.random.default_rng(42)
        engine = CoupledEngine(dim=1024, beta=5.0)
        indices = rng.permutation(n)

        cursor = 0
        day_reports = []

        for day in range(n_days):
            day_slice = indices[cursor:cursor + rate]
            cursor += rate

            t0 = time.time()
            for idx in day_slice:
                engine.store(texts[idx], vecs[idx], importance=rng.uniform(0.3, 0.8))
            ingest_s = time.time() - t0

            n_before = engine.n_memories
            t0 = time.time()
            result = engine.dream(seed=42 + day)
            dream_s = time.time() - t0

            day_reports.append({
                "day": day,
                "ingested": cursor,
                "before": n_before,
                "after": engine.n_memories,
                "pruned": result["pruned"],
                "merged": result["merged"],
                "dream_s": dream_s,
            })

        final_n = engine.n_memories
        retention = final_n / n

        print(f"\n  Streaming N={n} at {rate}/day × {n_days} days:")
        print(f"    Final: {final_n} ({retention:.1%} retention)")

        # Print milestones
        milestones = [0, n_days // 2, n_days - 1]
        for m in milestones:
            r = day_reports[m]
            print(f"    Day {r['day']:>3}: ingested={r['ingested']:>5} "
                  f"N={r['before']:>5}→{r['after']:>5} "
                  f"pruned={r['pruned']} merged={r['merged']} "
                  f"dream={r['dream_s']:.1f}s")

        # Dream never increases count
        for r in day_reports:
            assert r["after"] <= r["before"], (
                f"Day {r['day']}: dream increased N {r['before']}→{r['after']}"
            )

    @pytest.mark.parametrize("daily_rate", [50, 100, 200, 500])
    def test_fixed_1k_varied_daily_rate(self, daily_rate, vectors, sentences):
        """Ingest 1k total at varied daily rates — compare consolidation."""
        n = 1_000
        n_days = n // daily_rate
        vecs = vectors[:n]
        texts = sentences[:n]

        rng = np.random.default_rng(42)
        engine = CoupledEngine(dim=1024, beta=5.0)
        indices = rng.permutation(n)

        cursor = 0
        total_dream_s = 0.0

        for day in range(n_days):
            day_slice = indices[cursor:cursor + daily_rate]
            cursor += daily_rate
            for idx in day_slice:
                engine.store(texts[idx], vecs[idx], importance=rng.uniform(0.3, 0.8))

            t0 = time.time()
            engine.dream(seed=42 + day)
            total_dream_s += time.time() - t0

        final_n = engine.n_memories
        retention = final_n / n
        print(f"\n  1k @ {daily_rate}/day × {n_days} days: "
              f"final={final_n} ({retention:.1%}) dream_total={total_dream_s:.1f}s")

        assert final_n <= n


# ===========================================================================
# Test Class 4: Multi-dream convergence at varied scales
# ===========================================================================

class TestConvergenceSweep:
    """Run 3 consecutive dream cycles at each scale, track pattern trajectory."""

    @pytest.mark.parametrize("n", [100, 300, 500, 1_000, 2_000, 5_000])
    def test_convergence_3_cycles(self, n, vectors):
        """3 consecutive dreams — pattern count should be monotonically non-increasing."""
        X = vectors[:n].copy()
        importances = np.full(n, 0.5)
        trajectory = [n]

        total_s = 0.0
        for cycle in range(3):
            t0 = time.time()
            report = dream_cycle_xb(
                X, beta=5.0,
                importances=importances[:X.shape[0]],
                seed=42 + cycle,
            )
            total_s += time.time() - t0

            X = report.patterns
            importances = np.full(X.shape[0], 0.5)
            trajectory.append(X.shape[0])

        print(f"\n  N={n:>5} → 3 cycles: {' → '.join(str(t) for t in trajectory)} "
              f"({total_s:.1f}s total)")

        # Monotonically non-increasing
        for i in range(len(trajectory) - 1):
            assert trajectory[i] >= trajectory[i + 1], (
                f"Pattern count increased: {trajectory[i]} → {trajectory[i + 1]}"
            )


# ===========================================================================
# Test Class 5: Embedding geometry sweep
# ===========================================================================

class TestGeometrySweep:
    """Validate geometric properties at each scale point."""

    @pytest.mark.parametrize("n", SCALE_POINTS)
    def test_unit_norm(self, n, vectors):
        """All vectors are unit norm at every scale."""
        vecs = vectors[:n]
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)

    @pytest.mark.parametrize("n", SCALE_POINTS)
    def test_pairwise_separation(self, n, vectors):
        """Estimate δ_min and max cosine at each scale.

        As N grows, the max pairwise cosine increases (tighter packing),
        so δ_min decreases. This directly affects capacity.
        """
        vecs = vectors[:n]
        rng = np.random.default_rng(42)
        n_samples = min(50_000, n * (n - 1) // 2)
        i_idx = rng.integers(0, n, size=n_samples)
        j_idx = rng.integers(0, n, size=n_samples)
        mask = i_idx != j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]

        sims = np.sum(vecs[i_idx] * vecs[j_idx], axis=1)
        max_sim = float(np.max(sims))
        mean_sim = float(np.mean(sims))
        delta_min = 1.0 - max_sim

        # Capacity formula: N_max ~ exp(β·δ) / (4·β·M²)
        beta = 5.0
        n_max = np.exp(beta * delta_min) / (4 * beta * 1.0)
        utilization = n / n_max if n_max > 0 else float("inf")

        print(f"\n  N={n:>5}: δ_min={delta_min:.4f}  max_cos={max_sim:.4f}  "
              f"mean_cos={mean_sim:.4f}  N_max={n_max:.1f}  util={utilization:.0%}")

        assert delta_min > 0, "Near-duplicate patterns found"


# ===========================================================================
# Test Class 6: Post-dream retrieval quality sweep
# ===========================================================================

class TestPostDreamRetrieval:
    """Query quality before and after dream at each scale."""

    @pytest.mark.parametrize("n", [200, 500, 1_000, 2_000, 5_000])
    def test_retrieval_survives_dream(self, n, vectors, sentences):
        """Retrieval still works after a dream cycle at each scale."""
        vecs = vectors[:n]
        texts = sentences[:n]

        engine = CoupledEngine(dim=1024, beta=5.0)
        for text, emb in zip(texts, vecs):
            engine.store(text, emb, importance=0.5)

        rng = np.random.default_rng(42)
        query_indices = rng.choice(n, size=min(10, n), replace=False)

        # Pre-dream query
        pre_scores = []
        for idx in query_indices:
            results = engine.query(vecs[idx], top_k=5)
            if results:
                pre_scores.append(results[0]["score"])

        # Dream
        engine.dream(seed=42)
        n_after = engine.n_memories

        # Post-dream query
        post_scores = []
        for idx in query_indices:
            results = engine.query(vecs[idx], top_k=5)
            if results:
                post_scores.append(results[0]["score"])

        pre_mean = float(np.mean(pre_scores)) if pre_scores else 0.0
        post_mean = float(np.mean(post_scores)) if post_scores else 0.0

        print(f"\n  N={n:>5}: pre_score={pre_mean:.3f}  post_score={post_mean:.3f}  "
              f"N_after={n_after}")

        # Post-dream retrieval should not completely collapse
        assert post_mean > 0.2, (
            f"Post-dream retrieval collapsed at N={n}: {post_mean:.3f}"
        )
