"""Tests for CMA-ES dream parameter optimizer.

TDD Phase 1: These tests define expected behavior for cma_optimizer.py.
All should FAIL until implementation is written.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Imports — will fail until cma_optimizer.py exists
# ---------------------------------------------------------------------------
from cma_optimizer import (
    DreamOptimizer,
    ObjectiveFunction,
    grid_sweep,
    params_to_raw,
    raw_to_params,
)
from dream_ops import DreamParams


# ---------------------------------------------------------------------------
# 1. Sigmoid roundtrip: raw_to_params -> params_to_raw -> raw_to_params
# ---------------------------------------------------------------------------
class TestSigmoidRoundtrip:
    """Verify that the sigmoid mapping is invertible within tolerance."""

    def test_roundtrip_default_params(self):
        """Default DreamParams survive raw -> params -> raw -> params."""
        default_idle = 10.0
        default_params = DreamParams()

        raw = params_to_raw(default_idle, default_params)
        idle_back, params_back = raw_to_params(raw)
        raw_again = params_to_raw(idle_back, params_back)
        idle_final, params_final = raw_to_params(raw_again)

        assert abs(idle_back - idle_final) < 1e-6
        assert abs(params_back.eta - params_final.eta) < 1e-6
        assert abs(params_back.min_sep - params_final.min_sep) < 1e-6
        assert abs(params_back.prune_threshold - params_final.prune_threshold) < 1e-6
        assert abs(params_back.merge_threshold - params_final.merge_threshold) < 1e-6
        assert params_back.n_probes == params_final.n_probes
        assert abs(params_back.separation_rate - params_final.separation_rate) < 1e-6

    @pytest.mark.parametrize("seed", range(20))
    def test_roundtrip_random_raw(self, seed):
        """Random raw vectors roundtrip through params and back."""
        rng = np.random.default_rng(seed)
        raw = rng.normal(0, 2, size=7)

        idle1, params1 = raw_to_params(raw)
        raw_mid = params_to_raw(idle1, params1)
        idle2, params2 = raw_to_params(raw_mid)

        assert abs(idle1 - idle2) < 1e-4, f"idle mismatch: {idle1} vs {idle2}"
        assert abs(params1.eta - params2.eta) < 1e-4
        assert abs(params1.min_sep - params2.min_sep) < 1e-4
        assert abs(params1.prune_threshold - params2.prune_threshold) < 1e-4
        assert abs(params1.merge_threshold - params2.merge_threshold) < 1e-4
        assert params1.n_probes == params2.n_probes
        assert abs(params1.separation_rate - params2.separation_rate) < 1e-4

    def test_roundtrip_extreme_positive(self):
        """Large positive raw values map and roundtrip correctly."""
        raw = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        idle, params = raw_to_params(raw)
        raw_back = params_to_raw(idle, params)
        idle2, params2 = raw_to_params(raw_back)

        assert abs(idle - idle2) < 1e-3
        assert abs(params.eta - params2.eta) < 1e-3
        assert abs(params.min_sep - params2.min_sep) < 1e-3

    def test_roundtrip_extreme_negative(self):
        """Large negative raw values map and roundtrip correctly."""
        raw = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0])
        idle, params = raw_to_params(raw)
        raw_back = params_to_raw(idle, params)
        idle2, params2 = raw_to_params(raw_back)

        assert abs(idle - idle2) < 1e-3
        assert abs(params.eta - params2.eta) < 1e-3
        assert abs(params.min_sep - params2.min_sep) < 1e-3


# ---------------------------------------------------------------------------
# 2. Lean bounds always satisfied
# ---------------------------------------------------------------------------
class TestLeanBounds:
    """Any raw vector must produce DreamParams that satisfy Lean bounds."""

    @pytest.mark.parametrize("seed", range(50))
    def test_random_raw_satisfies_lean_bounds(self, seed):
        """Random raw vectors always produce valid DreamParams (no ValueError)."""
        rng = np.random.default_rng(seed)
        raw = rng.normal(0, 3, size=7)

        idle, params = raw_to_params(raw)

        # If we get here, validate() didn't raise — but check explicitly too
        assert 0 < params.eta < params.min_sep / 2, (
            f"eta={params.eta}, min_sep/2={params.min_sep / 2}"
        )
        assert 0 < params.min_sep <= 1, f"min_sep={params.min_sep}"
        assert 0 < params.merge_threshold < params.prune_threshold <= 1, (
            f"merge={params.merge_threshold}, prune={params.prune_threshold}"
        )
        assert 0.5 <= idle <= 30.0, f"idle={idle}"
        assert 10 <= params.n_probes <= 500, f"n_probes={params.n_probes}"
        assert 0.001 <= params.separation_rate <= 0.1, (
            f"sep_rate={params.separation_rate}"
        )

    def test_extreme_values_still_valid(self):
        """Extreme raw values (+-100) still produce valid params."""
        for extreme in [100.0, -100.0]:
            raw = np.full(7, extreme)
            idle, params = raw_to_params(raw)
            # validate() is called by __post_init__; if we're here, it passed
            params.validate()  # explicit double-check
            assert 0.5 <= idle <= 30.0

    def test_zero_raw_valid(self):
        """All-zero raw vector produces valid params (sigmoid(0) = 0.5)."""
        raw = np.zeros(7)
        idle, params = raw_to_params(raw)
        params.validate()
        assert 0.5 <= idle <= 30.0


# ---------------------------------------------------------------------------
# 3. Default params roundtrip
# ---------------------------------------------------------------------------
class TestDefaultParamsRoundtrip:
    """Default DreamParams with default idle threshold survives the mapping."""

    def test_default_params_values_preserved(self):
        """Default DreamParams values are recovered after raw roundtrip."""
        default_idle = 10.0
        dp = DreamParams()

        raw = params_to_raw(default_idle, dp)
        idle_out, dp_out = raw_to_params(raw)

        assert abs(idle_out - default_idle) < 0.01
        assert abs(dp_out.eta - dp.eta) < 1e-5
        assert abs(dp_out.min_sep - dp.min_sep) < 1e-5
        assert abs(dp_out.prune_threshold - dp.prune_threshold) < 1e-5
        assert abs(dp_out.merge_threshold - dp.merge_threshold) < 1e-5
        assert dp_out.n_probes == dp.n_probes
        assert abs(dp_out.separation_rate - dp.separation_rate) < 1e-5

    def test_raw_vector_is_7d(self):
        """params_to_raw returns a 7-element numpy array."""
        raw = params_to_raw(10.0, DreamParams())
        assert isinstance(raw, np.ndarray)
        assert raw.shape == (7,)

    def test_raw_values_are_finite(self):
        """All raw values are finite floats."""
        raw = params_to_raw(10.0, DreamParams())
        assert np.all(np.isfinite(raw))


# ---------------------------------------------------------------------------
# 4. CMA-ES optimizer runs with dummy objective
# ---------------------------------------------------------------------------
class TestOptimizerRuns:
    """DreamOptimizer runs CMA-ES with a dummy objective function."""

    @staticmethod
    def _dummy_objective(dream_idle_threshold: float, params: DreamParams) -> float:
        """Simple parabola: best score at defaults."""
        dp = DreamParams()
        penalty = (
            (dream_idle_threshold - 10.0) ** 2
            + (params.eta - dp.eta) ** 2
            + (params.min_sep - dp.min_sep) ** 2
            + (params.prune_threshold - dp.prune_threshold) ** 2
            + (params.merge_threshold - dp.merge_threshold) ** 2
            + (params.n_probes - dp.n_probes) ** 2 / 10000
            + (params.separation_rate - dp.separation_rate) ** 2
        )
        return -penalty  # Negative: higher is better, CMA-ES minimizes

    def test_optimizer_creates_and_runs(self):
        """DreamOptimizer completes 10 evaluations without error."""
        opt = DreamOptimizer(
            objective=self._dummy_objective,
            seed=42,
            max_evals=10,
            sigma0=0.5,
        )
        result = opt.optimize()

        assert "best_dream_idle" in result
        assert "best_params" in result
        assert "best_score" in result
        assert "n_evals" in result
        assert "history" in result
        assert result["n_evals"] >= 1

    def test_optimizer_result_params_are_valid(self):
        """Optimizer result contains valid DreamParams."""
        opt = DreamOptimizer(
            objective=self._dummy_objective,
            seed=42,
            max_evals=20,
            sigma0=0.5,
        )
        result = opt.optimize()

        bp = result["best_params"]
        assert isinstance(bp, DreamParams)
        bp.validate()  # must not raise

    def test_optimizer_history_grows(self):
        """History has at least one entry per generation."""
        opt = DreamOptimizer(
            objective=self._dummy_objective,
            seed=42,
            max_evals=30,
            sigma0=0.5,
        )
        result = opt.optimize()
        assert len(result["history"]) >= 1

    def test_optimizer_with_custom_popsize(self):
        """Custom popsize is respected."""
        opt = DreamOptimizer(
            objective=self._dummy_objective,
            seed=123,
            max_evals=20,
            sigma0=1.0,
            popsize=4,
        )
        result = opt.optimize()
        assert result["n_evals"] >= 1
        assert isinstance(result["best_params"], DreamParams)

    def test_optimizer_handles_exception_in_objective(self):
        """If objective raises, optimizer still completes (uses penalty)."""

        def bad_objective(idle: float, params: DreamParams) -> float:
            raise RuntimeError("intentional failure")

        opt = DreamOptimizer(
            objective=bad_objective,
            seed=42,
            max_evals=10,
            sigma0=0.5,
        )
        # Should not raise — exceptions get mapped to worst-case fitness
        result = opt.optimize()
        assert result["n_evals"] >= 1


# ---------------------------------------------------------------------------
# 5. Grid sweep with dummy objective
# ---------------------------------------------------------------------------
class TestGridSweep:
    """grid_sweep runs and returns sorted results."""

    @staticmethod
    def _dummy_objective(dream_idle_threshold: float, params: DreamParams) -> float:
        return -(dream_idle_threshold - 7.0) ** 2 - (params.eta - 0.01) ** 2

    def test_grid_sweep_returns_results(self):
        """Grid sweep returns non-empty list of dicts."""
        results = grid_sweep(
            objective=self._dummy_objective,
            dream_idle_values=[1, 7, 14],
            eta_values=[0.005, 0.01],
        )
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)

    def test_grid_sweep_sorted_descending(self):
        """Grid sweep results are sorted by score descending (best first)."""
        results = grid_sweep(
            objective=self._dummy_objective,
            dream_idle_values=[1, 7, 14, 30],
            eta_values=[0.005, 0.01, 0.02],
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not sorted descending: {scores}"
        )

    def test_grid_sweep_best_is_near_optimum(self):
        """Best grid result should be near the objective's optimum."""
        results = grid_sweep(
            objective=self._dummy_objective,
            dream_idle_values=[1, 3, 7, 14, 30],
            eta_values=[0.005, 0.01, 0.02],
        )
        best = results[0]
        # Best idle should be 7 (closest to optimum)
        assert best["dream_idle_threshold"] == 7

    def test_grid_sweep_result_dict_keys(self):
        """Each result dict has expected keys."""
        results = grid_sweep(
            objective=self._dummy_objective,
            dream_idle_values=[7],
            eta_values=[0.01],
        )
        r = results[0]
        assert "score" in r
        assert "dream_idle_threshold" in r
        assert "params" in r
        assert isinstance(r["params"], DreamParams)

    def test_grid_sweep_all_params_valid(self):
        """Every param set from grid sweep satisfies Lean bounds."""
        results = grid_sweep(
            objective=self._dummy_objective,
            dream_idle_values=[0.5, 1, 3, 7, 14, 30],
            eta_values=[0.005, 0.01, 0.02, 0.05],
        )
        for r in results:
            r["params"].validate()


# ---------------------------------------------------------------------------
# 6. Constraint projection edge cases
# ---------------------------------------------------------------------------
class TestConstraintProjection:
    """Edge cases near constraint boundaries are handled correctly."""

    def test_eta_clamped_below_min_sep_half(self):
        """When sigmoid would give eta >= min_sep/2, projection clamps it."""
        # Force a raw vector where eta sigmoid is high but min_sep sigmoid is low
        # raw[1] = large (eta -> 0.15), raw[2] = very negative (min_sep -> 0.05)
        raw = np.array([0.0, 10.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        idle, params = raw_to_params(raw)
        assert params.eta < params.min_sep / 2

    def test_merge_below_prune(self):
        """When sigmoid would give merge >= prune, projection fixes it."""
        # raw[3] = very negative (prune -> 0.5), raw[4] = large (merge -> 0.99)
        raw = np.array([0.0, 0.0, 0.0, -10.0, 10.0, 0.0, 0.0])
        idle, params = raw_to_params(raw)
        assert params.merge_threshold < params.prune_threshold

    def test_n_probes_is_integer(self):
        """n_probes is always an integer regardless of raw input."""
        for seed in range(30):
            rng = np.random.default_rng(seed)
            raw = rng.normal(0, 5, size=7)
            _, params = raw_to_params(raw)
            assert isinstance(params.n_probes, int), (
                f"n_probes is {type(params.n_probes)}: {params.n_probes}"
            )

    def test_projection_idempotent(self):
        """Projecting already-valid params doesn't change them."""
        dp = DreamParams()
        raw = params_to_raw(10.0, dp)
        idle, params = raw_to_params(raw)
        # Re-project
        raw2 = params_to_raw(idle, params)
        idle2, params2 = raw_to_params(raw2)
        assert abs(params.eta - params2.eta) < 1e-8
        assert abs(params.min_sep - params2.min_sep) < 1e-8
        assert abs(params.prune_threshold - params2.prune_threshold) < 1e-8
        assert abs(params.merge_threshold - params2.merge_threshold) < 1e-8

    def test_boundary_idle_min(self):
        """dream_idle_threshold never goes below 0.5."""
        raw = np.full(7, -100.0)
        idle, _ = raw_to_params(raw)
        assert idle >= 0.5

    def test_boundary_idle_max(self):
        """dream_idle_threshold never exceeds 30.0."""
        raw = np.full(7, 100.0)
        idle, _ = raw_to_params(raw)
        assert idle <= 30.0
