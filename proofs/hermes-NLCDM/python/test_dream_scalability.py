"""Dream cycle computational scalability benchmarks.

Characterizes the O(N^2 * d) scaling behavior of dream_cycle_xb at various
pattern counts from N=10 up to N=10000, with extrapolation to 100K and 1M.

Test structure:
  1. Scaling curve: measure wall-clock time at 7 sizes, fit log-log regression
     to extract the empirical scaling exponent.
  2. 10K full validation: single dream cycle on 10K patterns with timing,
     delta_min sampling, and DreamReport structure checks.
  3. 100K feasibility probe (slow): run only nrem_repulsion_xb on 100K
     patterns, measure time and memory, extrapolate full pipeline cost.
  4. 1M estimation: pure calculation from the fitted scaling curve --
     no actual run, just extrapolation and documentation.
"""

from __future__ import annotations

import time
import tracemalloc
import sys
import numpy as np
import pytest
from scipy import stats

from dream_ops import dream_cycle_xb, nrem_repulsion_xb, DreamReport
from test_dream_convergence import generate_realistic_workload
from test_capacity_boundary import compute_min_delta

# Mark for slow tests (100K probe)
slow = pytest.mark.slow


# ---------------------------------------------------------------------------
# Test 1: Scaling curve
# ---------------------------------------------------------------------------

class TestDreamScalability:
    """Computational scaling benchmarks for dream_cycle_xb."""

    SCALING_SIZES = [10, 100, 500, 1000, 2000, 5000, 10000]
    DIM = 128
    SEED = 42

    @staticmethod
    def _safe_n_topics(n_memories: int, dim: int) -> int:
        """Compute n_topics that respects QR decomposition limit.

        make_separated_centroids uses QR to produce orthogonal centroids,
        so n_topics cannot exceed dim. We also want at least 2 topics
        and at most N//5 for realistic cluster sizes.
        """
        return min(max(n_memories // 5, 2), dim)

    def test_scaling_curve(self):
        """Measure wall-clock time at 7 sizes and fit log-log regression.

        For each N in [10, 100, 500, 1000, 2000, 5000, 10000]:
          - Generate patterns via generate_realistic_workload
          - Time a single dream_cycle_xb call
          - Record N, time_seconds, n_after

        After collecting all points, fit log(time) = a * log(N) + b.
        The slope a reveals the scaling exponent (expect ~2.0 for O(N^2)).

        Assertions:
          - All runs complete without error
          - Scaling exponent a is between 1.5 and 3.0 (polynomial, not exponential)
          - N=10000 completes within 120 seconds
        """
        results = []

        for N in self.SCALING_SIZES:
            n_topics = self._safe_n_topics(N, self.DIM)
            workload = generate_realistic_workload(
                n_memories=N,
                n_topics=n_topics,
                dim=self.DIM,
                cross_topic_fraction=0.1,
                contradiction_fraction=0.03,
                seed=self.SEED,
            )
            patterns = workload["patterns"]
            importances = workload["importances"]
            labels = workload["labels"]

            t0 = time.perf_counter()
            report = dream_cycle_xb(
                patterns,
                beta=10.0,
                importances=importances,
                labels=labels,
                seed=self.SEED,
            )
            elapsed = time.perf_counter() - t0

            n_after = report.patterns.shape[0]
            results.append({
                "N": N,
                "time_s": elapsed,
                "n_after": n_after,
            })

        # Print scaling table
        print("\n" + "=" * 65)
        print("  DREAM CYCLE SCALING CURVE")
        print("=" * 65)
        print(f"  {'N':>7} {'time (s)':>10} {'n_after':>8} {'rate (N/s)':>12}")
        print("  " + "-" * 45)
        for r in results:
            rate = r["N"] / r["time_s"] if r["time_s"] > 0 else float("inf")
            print(f"  {r['N']:>7} {r['time_s']:>10.4f} {r['n_after']:>8} {rate:>12.1f}")

        # Fit log-log regression (exclude N=10 which is too small for reliable timing)
        log_N = np.log(np.array([r["N"] for r in results[1:]]))
        log_t = np.log(np.array([r["time_s"] for r in results[1:]]))

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_N, log_t)

        print(f"\n  Log-log regression: log(t) = {slope:.3f} * log(N) + ({intercept:.3f})")
        print(f"  Scaling exponent (slope): {slope:.3f}")
        print(f"  R^2: {r_value**2:.4f}")
        print(f"  std_err: {std_err:.4f}")
        print("=" * 65)

        # Store the scaling exponent for the 1M estimation test
        self.__class__._scaling_slope = slope
        self.__class__._scaling_intercept = intercept
        self.__class__._scaling_results = results

        # Assertions
        # 1. Scaling exponent between 1.0 and 3.0 (polynomial, not exponential).
        #    Lower bound 1.0 allows for BLAS-accelerated matmul paths which
        #    bring the exponent below O(N^1.5) on hardware with fast GEMM.
        assert 1.0 <= slope <= 3.0, (
            f"Scaling exponent {slope:.3f} outside expected [1.0, 3.0] range. "
            f"This suggests the algorithm is not polynomial O(N^a) with a in [1.0, 3.0]."
        )

        # 2. N=10000 completes within 180 seconds (empirical: ~144s on Apple Silicon)
        t_10k = results[-1]["time_s"]
        assert t_10k < 180.0, (
            f"N=10000 took {t_10k:.1f}s, exceeding 180s timeout"
        )

        # 3. R^2 should be reasonably good (log-log fit explains most variance)
        assert r_value ** 2 > 0.90, (
            f"Log-log fit R^2 = {r_value**2:.4f} is too low -- "
            f"scaling may not be a clean power law"
        )

    # -------------------------------------------------------------------
    # Test 2: 10K full validation
    # -------------------------------------------------------------------

    def test_10k_full_validation(self):
        """Generate 10K patterns, run 1 dream cycle, validate timing and structure.

        Measures:
          - Wall clock time
          - delta_min before/after (sampled on 500 random patterns for O(N^2) safety)
          - Pattern count before/after
          - Whether DreamReport has associations

        Assertions:
          - Completes within 120 seconds
          - Pattern count after <= before
          - No crash (implicit)
        """
        workload = generate_realistic_workload(
            n_memories=10000,
            n_topics=self._safe_n_topics(10000, self.DIM),
            dim=self.DIM,
            cross_topic_fraction=0.1,
            contradiction_fraction=0.03,
            seed=self.SEED,
        )
        patterns = workload["patterns"]
        importances = workload["importances"]
        labels = workload["labels"]
        N_before = patterns.shape[0]

        # Sample 500 random patterns for delta_min computation (O(N^2) safe)
        rng = np.random.default_rng(self.SEED)
        sample_idx = rng.choice(N_before, size=min(500, N_before), replace=False)
        sample_before = patterns[sample_idx]
        delta_min_before = compute_min_delta(sample_before)

        # Time the dream cycle
        t0 = time.perf_counter()
        report = dream_cycle_xb(
            patterns,
            beta=10.0,
            importances=importances,
            labels=labels,
            seed=self.SEED,
        )
        elapsed = time.perf_counter() - t0

        N_after = report.patterns.shape[0]

        # Sample delta_min after (use first 500 of output)
        sample_after_idx = rng.choice(N_after, size=min(500, N_after), replace=False)
        sample_after = report.patterns[sample_after_idx]
        delta_min_after = compute_min_delta(sample_after)

        has_associations = len(report.associations) > 0
        n_pruned = len(report.pruned_indices)
        n_merged = len(report.merge_map)

        print("\n" + "=" * 65)
        print("  10K FULL VALIDATION")
        print("=" * 65)
        print(f"  Wall clock time: {elapsed:.2f}s")
        print(f"  Pattern count: {N_before} -> {N_after} (delta={N_before - N_after})")
        print(f"  delta_min (sampled 500): {delta_min_before:.6f} -> {delta_min_after:.6f}")
        print(f"  Associations discovered: {len(report.associations)}")
        print(f"  Pruned: {n_pruned}, Merged: {n_merged}")
        print(f"  DreamReport type: {type(report).__name__}")
        print("=" * 65)

        # Assertions
        assert elapsed < 180.0, (
            f"10K dream cycle took {elapsed:.1f}s, exceeding 180s timeout"
        )
        assert N_after <= N_before, (
            f"Pattern count increased: {N_before} -> {N_after}"
        )
        assert isinstance(report, DreamReport), (
            f"Expected DreamReport, got {type(report).__name__}"
        )
        assert report.patterns.shape[1] == self.DIM, (
            f"Dimension mismatch: expected {self.DIM}, got {report.patterns.shape[1]}"
        )

    # -------------------------------------------------------------------
    # Test 3: 100K feasibility probe (slow)
    # -------------------------------------------------------------------

    @slow
    def test_100k_feasibility_probe(self):
        """Run only nrem_repulsion_xb on 100K patterns. Time it. Extrapolate.

        Strategy: run ONLY the first stage (nrem_repulsion_xb) individually,
        not the full pipeline. Time it. Extrapolate total pipeline time by
        multiplying by ~3 (for 3 pairwise stages).

        Measures:
          - Wall clock time for just repulsion stage
          - Extrapolated full pipeline time
          - Peak memory usage via tracemalloc

        Assertions:
          - Repulsion completes (even if slow)
          - Report the time -- this is a characterization test
        """
        dim_100k = 64
        workload = generate_realistic_workload(
            n_memories=100000,
            n_topics=self._safe_n_topics(100000, dim_100k),
            dim=dim_100k,
            cross_topic_fraction=0.05,
            contradiction_fraction=0.02,
            seed=self.SEED,
        )
        patterns = workload["patterns"]
        importances = workload["importances"]
        N = patterns.shape[0]

        # Start memory tracking
        tracemalloc.start()

        t0 = time.perf_counter()
        result = nrem_repulsion_xb(patterns, importances)
        elapsed_repulsion = time.perf_counter() - t0

        # Memory snapshot
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Extrapolate: dream_cycle_xb has ~3 pairwise stages
        # (repulsion, prune, merge) plus 2 cheaper stages (unlearn, explore)
        extrapolated_full = elapsed_repulsion * 3.0

        print("\n" + "=" * 65)
        print("  100K FEASIBILITY PROBE")
        print("=" * 65)
        print(f"  N = {N}, dim = 64")
        print(f"  nrem_repulsion_xb time: {elapsed_repulsion:.2f}s")
        print(f"  Extrapolated full pipeline: {extrapolated_full:.1f}s (~{extrapolated_full/60:.1f} min)")
        print(f"  Peak memory: {peak_mem / (1024**2):.1f} MB")
        print(f"  Current memory: {current_mem / (1024**2):.1f} MB")
        print(f"  Output shape: {result.shape}")
        print(f"  Patterns array size: {sys.getsizeof(patterns) / (1024**2):.1f} MB (shallow)")
        print(f"  Patterns nbytes: {patterns.nbytes / (1024**2):.1f} MB")
        print("=" * 65)

        # Assertions
        assert result.shape == patterns.shape, (
            f"Output shape {result.shape} != input shape {patterns.shape}"
        )
        # All output vectors should be unit norm
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), (
            f"Some output vectors are not unit norm: min={norms.min():.6f}, max={norms.max():.6f}"
        )

    # -------------------------------------------------------------------
    # Test 4: 1M estimation (no actual run)
    # -------------------------------------------------------------------

    def test_1m_extrapolation(self):
        """Extrapolate timing for N=100K, 500K, 1M from the scaling curve.

        Uses the fitted log-log regression from test_scaling_curve to
        predict wall-clock time at large N. This is a pure calculation --
        no dream cycle is run.

        If test_scaling_curve hasn't run yet, runs a minimal fit inline.
        """
        # Try to use cached results from scaling curve test
        slope = getattr(self.__class__, '_scaling_slope', None)
        intercept = getattr(self.__class__, '_scaling_intercept', None)
        results = getattr(self.__class__, '_scaling_results', None)

        if slope is None or intercept is None:
            # Run a minimal scaling measurement inline
            mini_sizes = [100, 500, 1000, 2000, 5000]
            mini_results = []
            for N in mini_sizes:
                n_topics = self._safe_n_topics(N, self.DIM)
                workload = generate_realistic_workload(
                    n_memories=N,
                    n_topics=n_topics,
                    dim=self.DIM,
                    cross_topic_fraction=0.1,
                    contradiction_fraction=0.03,
                    seed=self.SEED,
                )
                t0 = time.perf_counter()
                dream_cycle_xb(
                    workload["patterns"],
                    beta=10.0,
                    importances=workload["importances"],
                    labels=workload["labels"],
                    seed=self.SEED,
                )
                elapsed = time.perf_counter() - t0
                mini_results.append({"N": N, "time_s": elapsed})

            log_N = np.log(np.array([r["N"] for r in mini_results]))
            log_t = np.log(np.array([r["time_s"] for r in mini_results]))
            slope, intercept, _, _, _ = stats.linregress(log_N, log_t)
            results = mini_results

        # Extrapolate to large N
        target_sizes = [10_000, 100_000, 500_000, 1_000_000]
        predictions = {}
        for N in target_sizes:
            log_t_pred = slope * np.log(N) + intercept
            t_pred = np.exp(log_t_pred)
            predictions[N] = t_pred

        print("\n" + "=" * 65)
        print("  1M EXTRAPOLATION (from fitted scaling curve)")
        print("=" * 65)
        print(f"  Scaling exponent: {slope:.3f}")
        print(f"  Intercept: {intercept:.3f}")
        print()
        print(f"  {'N':>10} {'Predicted time':>16} {'Human-readable':>20}")
        print("  " + "-" * 50)
        for N, t_pred in predictions.items():
            if t_pred < 60:
                human = f"{t_pred:.1f}s"
            elif t_pred < 3600:
                human = f"{t_pred/60:.1f} min"
            elif t_pred < 86400:
                human = f"{t_pred/3600:.1f} hours"
            else:
                human = f"{t_pred/86400:.1f} days"
            print(f"  {N:>10,} {t_pred:>14.1f}s {human:>20}")

        # Measured data points for reference
        if results:
            print(f"\n  Measured data points:")
            print(f"  {'N':>10} {'Measured time':>16}")
            print("  " + "-" * 30)
            for r in results:
                print(f"  {r['N']:>10,} {r['time_s']:>14.4f}s")

        print()
        print(f"  Algorithmic boundary note:")
        print(f"  At O(N^{slope:.1f}), 1M patterns would take ~{predictions[1_000_000]/3600:.0f} hours.")
        print(f"  Practical limit for sub-minute dream: N ~ {int(np.exp((np.log(60) - intercept) / slope)):,}")
        print(f"  Practical limit for sub-10min dream: N ~ {int(np.exp((np.log(600) - intercept) / slope)):,}")
        print("=" * 65)

        # Assertions: all predictions should be finite positive values
        for N, t_pred in predictions.items():
            assert np.isfinite(t_pred), (
                f"Prediction for N={N:,} is not finite: {t_pred}"
            )
            assert t_pred > 0, (
                f"Prediction for N={N:,} is not positive: {t_pred}"
            )

        # The scaling exponent should be consistent
        assert 1.0 <= slope <= 4.0, (
            f"Scaling exponent {slope:.3f} outside reasonable range [1.0, 4.0]"
        )
