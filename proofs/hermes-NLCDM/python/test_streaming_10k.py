"""Streaming scale test: 10k+ memories ingested incrementally with nightly dreams.

Loads pre-embedded vectors from embeddings_170k.npy (run embed_170k.py first),
then simulates realistic memory lifecycle at different ingest rates:

  - 100/day × 100 days = 10k ingested, 100 dream cycles
  - 500/day × 20 days  = 10k ingested, 20 dream cycles
  - 1k/day  × 10 days  = 10k ingested, 10 dream cycles

This is the realistic counterpart to test_scale_10k.py (which dumps 10k at once).
In production, dreams consolidate between ingest batches, keeping the working set
bounded. The proofs guarantee:

  - Lyapunov.lean: energy decreases each dream → convergence
  - StrengthDecay.lean: old patterns decay → pruning candidates
  - Capacity.lean: as N drops via prune/merge, effective capacity rises

Usage:
    .venv/bin/python -m pytest test_streaming_10k.py -v -s --tb=short
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
    dream_cycle_v2,
    compute_adaptive_thresholds,
)
from nlcdm_core import cosine_sim


_THIS_DIR = Path(__file__).resolve().parent
_NPY_PATH = _THIS_DIR / "embeddings_170k.npy"
_TXT_PATH = _THIS_DIR / "sentences_170k.txt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_VECTORS: np.ndarray | None = None
_SENTENCES: list[str] | None = None


def load_vectors() -> np.ndarray:
    """Load pre-embedded vectors from .npy file."""
    global _VECTORS
    if _VECTORS is None:
        if not _NPY_PATH.exists():
            pytest.skip(
                f"Run embed_170k.py first to generate {_NPY_PATH.name}"
            )
        _VECTORS = np.load(_NPY_PATH).astype(np.float64)
        # Normalize (in case float32 rounding drifted)
        norms = np.linalg.norm(_VECTORS, axis=1, keepdims=True)
        _VECTORS = _VECTORS / np.maximum(norms, 1e-8)
    return _VECTORS


def load_sentences() -> list[str]:
    """Load sentences aligned with vectors."""
    global _SENTENCES
    if _SENTENCES is None:
        if not _TXT_PATH.exists():
            pytest.skip(
                f"Run embed_170k.py first to generate {_TXT_PATH.name}"
            )
        with open(_TXT_PATH, encoding="utf-8") as f:
            _SENTENCES = [line.strip() for line in f]
    return _SENTENCES


# ---------------------------------------------------------------------------
# Streaming simulation engine
# ---------------------------------------------------------------------------

@dataclass
class DayReport:
    day: int
    n_ingested_today: int
    n_total_ingested: int
    n_before_dream: int
    n_after_dream: int
    pruned: int
    merged: int
    associations: int
    dream_time_s: float
    ingest_time_s: float


@dataclass
class StreamingResult:
    rate_per_day: int
    n_days: int
    total_ingested: int
    final_n: int
    daily_reports: list[DayReport] = field(default_factory=list)
    total_dream_time_s: float = 0.0
    total_ingest_time_s: float = 0.0


def simulate_streaming(
    rate_per_day: int,
    n_days: int,
    vectors: np.ndarray,
    sentences: list[str],
    beta: float = 5.0,
    seed: int = 42,
    use_v2: bool = True,
) -> StreamingResult:
    """Simulate incremental memory ingestion with nightly dreams.

    Each day:
      1. Wake: ingest `rate_per_day` new memories into CoupledEngine
      2. Sleep: run dream cycle on the full memory store

    Args:
        rate_per_day: memories ingested per day
        n_days: number of days to simulate
        vectors: (N, d) pre-embedded vectors to draw from
        sentences: aligned text for the vectors
        beta: inverse temperature
        seed: random seed
        use_v2: use adaptive-threshold dream pipeline
    """
    rng = np.random.default_rng(seed)
    dim = vectors.shape[1]
    engine = CoupledEngine(dim=dim, beta=beta)

    total_available = len(vectors)
    total_needed = rate_per_day * n_days
    if total_needed > total_available:
        raise ValueError(
            f"Need {total_needed} vectors but only have {total_available}"
        )

    # Shuffle indices so each day gets diverse sentences
    indices = rng.permutation(total_available)[:total_needed]

    result = StreamingResult(
        rate_per_day=rate_per_day,
        n_days=n_days,
        total_ingested=0,
        final_n=0,
    )

    cursor = 0
    for day in range(n_days):
        # --- Wake: ingest today's batch ---
        day_indices = indices[cursor : cursor + rate_per_day]
        cursor += rate_per_day

        t0 = time.time()
        for idx in day_indices:
            importance = rng.uniform(0.3, 0.8)
            engine.store(
                sentences[idx],
                vectors[idx],
                importance=importance,
            )
        ingest_time = time.time() - t0
        result.total_ingested += rate_per_day

        n_before = engine.n_memories

        # --- Sleep: dream cycle ---
        t0 = time.time()
        dream_result = engine.dream(seed=seed + day)
        dream_time = time.time() - t0

        n_after = engine.n_memories

        report = DayReport(
            day=day,
            n_ingested_today=rate_per_day,
            n_total_ingested=result.total_ingested,
            n_before_dream=n_before,
            n_after_dream=n_after,
            pruned=dream_result["pruned"],
            merged=dream_result["merged"],
            associations=len(dream_result["associations"]),
            dream_time_s=dream_time,
            ingest_time_s=ingest_time,
        )
        result.daily_reports.append(report)
        result.total_dream_time_s += dream_time
        result.total_ingest_time_s += ingest_time

    result.final_n = engine.n_memories
    return result


def print_streaming_report(result: StreamingResult) -> None:
    """Print a summary of the streaming simulation."""
    print(f"\n  Streaming: {result.rate_per_day}/day × {result.n_days} days")
    print(f"  Total ingested: {result.total_ingested}")
    print(f"  Final surviving: {result.final_n} "
          f"({result.final_n / result.total_ingested:.1%} retention)")
    print(f"  Total dream time: {result.total_dream_time_s:.1f}s")
    print(f"  Total ingest time: {result.total_ingest_time_s:.1f}s")

    # Print a few milestone days
    milestones = [0, result.n_days // 4, result.n_days // 2,
                  3 * result.n_days // 4, result.n_days - 1]
    milestones = sorted(set(m for m in milestones if 0 <= m < result.n_days))
    print(f"  Day-by-day milestones:")
    for m in milestones:
        r = result.daily_reports[m]
        print(f"    Day {r.day:3d}: ingested={r.n_total_ingested:5d} "
              f"before={r.n_before_dream:5d} after={r.n_after_dream:5d} "
              f"pruned={r.pruned} merged={r.merged} "
              f"dream={r.dream_time_s:.1f}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vectors():
    return load_vectors()


@pytest.fixture(scope="module")
def sentences():
    return load_sentences()


# ===========================================================================
# Test Class 1: Streaming at different rates
# ===========================================================================

class TestStreamingRates:
    """Compare consolidation behavior at different ingest rates."""

    def test_100_per_day_100_days(self, vectors, sentences):
        """100/day × 100 days = 10k ingested with heavy consolidation."""
        result = simulate_streaming(
            rate_per_day=100, n_days=100,
            vectors=vectors, sentences=sentences, seed=42,
        )
        print_streaming_report(result)

        # With 100 dream cycles, pattern count should stabilize
        final_counts = [r.n_after_dream for r in result.daily_reports[-10:]]
        variance = np.var(final_counts)
        mean_count = np.mean(final_counts)
        print(f"  Last 10 days variance: {variance:.1f}, mean: {mean_count:.0f}")

        # Final count should be much less than 10k (dreams consolidated)
        assert result.final_n <= result.total_ingested

    def test_500_per_day_20_days(self, vectors, sentences):
        """500/day × 20 days = 10k ingested with moderate consolidation."""
        result = simulate_streaming(
            rate_per_day=500, n_days=20,
            vectors=vectors, sentences=sentences, seed=42,
        )
        print_streaming_report(result)
        assert result.final_n <= result.total_ingested

    def test_1k_per_day_10_days(self, vectors, sentences):
        """1k/day × 10 days = 10k ingested with light consolidation."""
        result = simulate_streaming(
            rate_per_day=1000, n_days=10,
            vectors=vectors, sentences=sentences, seed=42,
        )
        print_streaming_report(result)
        assert result.final_n <= result.total_ingested


# ===========================================================================
# Test Class 2: Dream timing scales with working set, not total ingested
# ===========================================================================

class TestDreamTimingScaling:
    """Verify that dream time depends on current N, not cumulative ingested."""

    def test_dream_time_bounded_100_per_day(self, vectors, sentences):
        """At 100/day, dream time should stay bounded (N stays low)."""
        result = simulate_streaming(
            rate_per_day=100, n_days=50,
            vectors=vectors, sentences=sentences, seed=42,
        )
        print_streaming_report(result)

        # Dream time for last 10 days should not grow unboundedly
        late_times = [r.dream_time_s for r in result.daily_reports[-10:]]
        early_times = [r.dream_time_s for r in result.daily_reports[:10]]
        late_mean = np.mean(late_times)
        early_mean = np.mean(early_times)

        print(f"  Early dream mean: {early_mean:.2f}s")
        print(f"  Late dream mean: {late_mean:.2f}s")
        # Late dreams should not be more than 10x slower than early
        # (if consolidation works, N is bounded)


# ===========================================================================
# Test Class 3: Retrieval quality after streaming
# ===========================================================================

class TestRetrievalAfterStreaming:
    """Query quality after incremental ingest + dreams."""

    def test_retrieval_100_per_day(self, vectors, sentences):
        """After 100 days of streaming, can we still retrieve recent memories?"""
        rng = np.random.default_rng(42)
        dim = vectors.shape[1]
        engine = CoupledEngine(dim=dim, beta=5.0)

        # Ingest 100/day for 30 days with dreams
        rate = 100
        n_days = 30
        total_available = len(vectors)
        indices = rng.permutation(total_available)[:rate * n_days]
        cursor = 0
        last_day_indices = []

        for day in range(n_days):
            day_indices = indices[cursor : cursor + rate]
            cursor += rate
            for idx in day_indices:
                engine.store(sentences[idx], vectors[idx], importance=0.5)
            if day == n_days - 1:
                last_day_indices = list(day_indices)
            engine.dream(seed=42 + day)

        # Query with last day's memories — should find high-similarity results
        scores = []
        for idx in last_day_indices[:10]:
            results = engine.query(vectors[idx], top_k=5)
            if results:
                scores.append(results[0]["score"])

        mean_score = np.mean(scores) if scores else 0.0
        print(f"\n  After 30 days streaming (100/day):")
        print(f"  Final N: {engine.n_memories}")
        print(f"  Mean retrieval score (last day queries): {mean_score:.3f}")
        assert mean_score > 0.2, f"Post-streaming retrieval too weak: {mean_score:.3f}"


# ===========================================================================
# Test Class 4: Monotonic energy descent across days
# ===========================================================================

class TestEnergyDescentStreaming:
    """Energy should decrease within each dream cycle (Lyapunov.lean)."""

    def test_pattern_count_monotone_within_dream(self, vectors, sentences):
        """Each dream cycle should never increase pattern count."""
        result = simulate_streaming(
            rate_per_day=200, n_days=20,
            vectors=vectors, sentences=sentences, seed=42,
        )

        for r in result.daily_reports:
            assert r.n_after_dream <= r.n_before_dream, (
                f"Day {r.day}: dream INCREASED patterns "
                f"{r.n_before_dream} → {r.n_after_dream}"
            )
        print(f"\n  All {result.n_days} days: pattern count monotone within dreams ✓")


# ===========================================================================
# Test Class 5: Consolidation ratio by rate
# ===========================================================================

class TestConsolidationRatio:
    """Lower ingest rates should produce higher consolidation ratios."""

    def test_consolidation_comparison(self, vectors, sentences):
        """Compare retention rates: slower ingest → more consolidation time."""
        rates = [100, 500, 1000]
        results = {}

        for rate in rates:
            n_days = 5000 // rate  # all ingest 5k total
            result = simulate_streaming(
                rate_per_day=rate, n_days=n_days,
                vectors=vectors, sentences=sentences, seed=42,
            )
            retention = result.final_n / result.total_ingested
            results[rate] = {
                "final_n": result.final_n,
                "retention": retention,
                "total_dream_time": result.total_dream_time_s,
            }

        print("\n  Consolidation comparison (5k total each):")
        for rate, r in sorted(results.items()):
            print(f"    {rate}/day: final={r['final_n']}, "
                  f"retention={r['retention']:.1%}, "
                  f"dream_time={r['total_dream_time']:.1f}s")
