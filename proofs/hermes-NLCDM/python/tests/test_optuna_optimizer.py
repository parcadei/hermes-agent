"""Tests for Optuna GPSampler-based dream parameter optimizer.

TDD Phase 1: These tests define expected behavior for optuna_optimizer.py.
All should FAIL until implementation is written.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import numpy as np
import optuna
import pytest

# ---------------------------------------------------------------------------
# Imports -- will fail until optuna_optimizer.py exists
# ---------------------------------------------------------------------------
from optuna_optimizer import (
    OptunaOptimizer,
    params_to_trial_dict,
    trial_to_params,
)
from dream_ops import DreamParams


# ---------------------------------------------------------------------------
# Shared mock objective: quadratic with known optimum
# ---------------------------------------------------------------------------

def _quadratic_objective(dream_idle_threshold: float, params: DreamParams) -> float:
    """Simple quadratic: best score at idle=5.0, eta=0.01, min_sep=0.3,
    merge=0.90, prune=0.95, n_probes=200, sep_rate=0.02.

    Returns HIGHER = BETTER (negative quadratic penalty).
    """
    penalty = (
        (dream_idle_threshold - 5.0) ** 2
        + 100 * (params.eta - 0.01) ** 2
        + (params.min_sep - 0.3) ** 2
        + (params.merge_threshold - 0.90) ** 2
        + (params.prune_threshold - 0.95) ** 2
        + (params.n_probes - 200) ** 2 / 10000
        + (params.separation_rate - 0.02) ** 2
    )
    return -penalty


# ---------------------------------------------------------------------------
# 1. OptunaOptimizer core tests
# ---------------------------------------------------------------------------
class TestOptunaOptimizer:
    """Verify OptunaOptimizer creates and runs correctly."""

    def test_creates_study_with_gp_sampler(self):
        """The internal study uses GPSampler."""
        opt = OptunaOptimizer(
            objective=_quadratic_objective,
            seed=42,
        )
        assert isinstance(opt.study.sampler, optuna.samplers.GPSampler)

    def test_parameter_ranges(self):
        """All suggested params stay within Lean-proven bounds across many trials."""
        violations = []

        def checking_objective(dream_idle: float, params: DreamParams) -> float:
            # DreamParams.__post_init__ calls validate(), so if we get here
            # it already passed. But let's also verify the optimizer ranges.
            if not (0.1 <= dream_idle <= 50.0):
                violations.append(f"idle={dream_idle}")
            if not (0.001 <= params.eta <= 0.05):
                violations.append(f"eta={params.eta}")
            if not (0.01 <= params.min_sep <= 0.5):
                violations.append(f"min_sep={params.min_sep}")
            if not (0.7 <= params.merge_threshold <= 0.99):
                violations.append(f"merge={params.merge_threshold}")
            if not (0.8 <= params.prune_threshold <= 0.999):
                violations.append(f"prune={params.prune_threshold}")
            if not (1 <= params.n_probes <= 10):
                violations.append(f"n_probes={params.n_probes}")
            if not (0.01 <= params.separation_rate <= 0.5):
                violations.append(f"sep_rate={params.separation_rate}")
            return -dream_idle  # dummy

        opt = OptunaOptimizer(objective=checking_objective, seed=42)
        opt.optimize(n_trials=15)
        assert violations == [], f"Parameter range violations: {violations}"

    def test_seed_trial_enqueued(self):
        """When seed_params is given, the first trial uses those exact values."""
        seed_params = {
            "dream_idle_threshold": 5.0,
            "eta": 0.01,
            "min_sep": 0.3,
            "merge_threshold": 0.90,
            "prune_threshold": 0.95,
            "n_probes": 200,
            "separation_rate": 0.02,
        }
        captured = []

        def capture_objective(dream_idle: float, params: DreamParams) -> float:
            captured.append({
                "dream_idle_threshold": dream_idle,
                "eta": params.eta,
                "min_sep": params.min_sep,
                "merge_threshold": params.merge_threshold,
                "prune_threshold": params.prune_threshold,
                "n_probes": params.n_probes,
                "separation_rate": params.separation_rate,
            })
            return _quadratic_objective(dream_idle, params)

        opt = OptunaOptimizer(
            objective=capture_objective,
            seed=42,
            seed_params=seed_params,
        )
        opt.optimize(n_trials=3)

        assert len(captured) >= 1
        first = captured[0]
        assert abs(first["dream_idle_threshold"] - 5.0) < 1e-6
        assert abs(first["eta"] - 0.01) < 1e-6
        assert abs(first["min_sep"] - 0.3) < 1e-6
        assert abs(first["merge_threshold"] - 0.90) < 1e-6
        assert abs(first["prune_threshold"] - 0.95) < 1e-6
        assert first["n_probes"] == 200
        assert abs(first["separation_rate"] - 0.02) < 1e-6

    def test_optimize_returns_best_params(self):
        """Optimize with quadratic objective finds near-optimal params."""
        opt = OptunaOptimizer(
            objective=_quadratic_objective,
            seed=42,
            seed_params={
                "dream_idle_threshold": 5.0,
                "eta": 0.01,
                "min_sep": 0.3,
                "merge_threshold": 0.90,
                "prune_threshold": 0.95,
                "n_probes": 5,
                "separation_rate": 0.02,
            },
        )
        result = opt.optimize(n_trials=20)

        assert "best_dream_idle" in result
        assert "best_params" in result
        assert "best_score" in result
        assert "n_trials" in result
        assert "history" in result
        assert isinstance(result["best_params"], DreamParams)
        # GP sampler in 7D needs many trials; just verify it found something
        # better than random (which averages ~-10 on the quadratic)
        assert result["best_score"] > -10.0, (
            f"Expected better-than-random score, got {result['best_score']}"
        )

    def test_optimize_history_recorded(self):
        """Trial history is captured with per-trial data."""
        opt = OptunaOptimizer(
            objective=_quadratic_objective,
            seed=42,
        )
        result = opt.optimize(n_trials=10)

        assert len(result["history"]) == 10
        for entry in result["history"]:
            assert "trial" in entry
            assert "score" in entry
            assert "dream_idle_threshold" in entry
            assert "params" in entry

    def test_expanded_search_adds_beta(self):
        """When expand_search=True, beta is included in the search space."""
        captured_beta = []

        def obj_with_beta(dream_idle: float, params: DreamParams) -> float:
            # beta_high should vary if expanded search is on
            captured_beta.append(params.beta_high)
            return _quadratic_objective(dream_idle, params)

        opt = OptunaOptimizer(
            objective=obj_with_beta,
            seed=42,
            expand_search=True,
        )
        opt.optimize(n_trials=10)

        # With expanded search, beta should have varied (not all default 10.0)
        unique_betas = set(round(b, 2) for b in captured_beta)
        assert len(unique_betas) > 1, (
            f"beta should vary with expand_search=True, got only {unique_betas}"
        )

    def test_expanded_search_adds_capacity_threshold(self):
        """When expand_search=True, capacity_gating_low is in results."""
        opt = OptunaOptimizer(
            objective=_quadratic_objective,
            seed=42,
            expand_search=True,
        )
        result = opt.optimize(n_trials=5)

        # History entries should contain capacity_gating_low when expanded
        assert any(
            "capacity_gating_low" in entry
            for entry in result["history"]
        ), "Expected capacity_gating_low in history with expand_search=True"

    def test_revalidate_uses_median(self):
        """Revalidation calls objective N times and uses median score."""
        call_count = [0]

        def counting_objective(dream_idle: float, params: DreamParams) -> float:
            call_count[0] += 1
            return _quadratic_objective(dream_idle, params) + np.random.normal(0, 0.01)

        opt = OptunaOptimizer(
            objective=counting_objective,
            seed=42,
        )
        opt.optimize(n_trials=10)
        initial_calls = call_count[0]

        reval = opt.revalidate_top_k(k=3, n_repeats=5)
        extra_calls = call_count[0] - initial_calls

        # Should have called objective 3 * 5 = 15 more times
        assert extra_calls == 15, f"Expected 15 extra calls, got {extra_calls}"
        assert "revalidated" in reval
        assert len(reval["revalidated"]) == 3

        for entry in reval["revalidated"]:
            assert "median_score" in entry
            assert "std_score" in entry
            assert "n_repeats" in entry
            assert entry["n_repeats"] == 5

    def test_revalidate_reorders_by_median(self):
        """Revalidation can change the ranking of top candidates."""
        # Create an objective where trial noise can flip ordering
        eval_count = [0]

        def noisy_objective(dream_idle: float, params: DreamParams) -> float:
            eval_count[0] += 1
            base = _quadratic_objective(dream_idle, params)
            # First 10 evals (optimization phase) add large noise
            if eval_count[0] <= 10:
                return base + np.random.default_rng(eval_count[0]).normal(0, 2.0)
            # Revalidation phase: consistent
            return base

        opt = OptunaOptimizer(
            objective=noisy_objective,
            seed=42,
        )
        opt.optimize(n_trials=10)
        reval = opt.revalidate_top_k(k=5, n_repeats=5)

        # Verify results are sorted by median (best first)
        medians = [r["median_score"] for r in reval["revalidated"]]
        assert medians == sorted(medians, reverse=True), (
            f"Revalidated results not sorted by median: {medians}"
        )


# ---------------------------------------------------------------------------
# 2. Trial conversion tests
# ---------------------------------------------------------------------------
class TestTrialConversion:
    """Verify conversion between Optuna trials and DreamParams."""

    def test_trial_to_params_roundtrip(self):
        """Convert trial -> params -> dict -> verify fields match."""
        # Create a real study and run one trial to get a FrozenTrial
        sampler = optuna.samplers.GPSampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def obj(trial):
            idle = trial.suggest_float("dream_idle_threshold", 0.1, 50.0)
            eta = trial.suggest_float("eta", 0.001, 0.05, log=True)
            min_sep = trial.suggest_float("min_sep", 0.01, 0.5)
            merge = trial.suggest_float("merge_threshold", 0.7, 0.99)
            prune = trial.suggest_float("prune_threshold", 0.8, 0.999)
            n_probes = trial.suggest_int("n_probes", 1, 10)
            sep_rate = trial.suggest_float("separation_rate", 0.01, 0.5)
            return 0.0

        study.optimize(obj, n_trials=1)
        trial = study.best_trial

        idle, params = trial_to_params(trial)
        d = params_to_trial_dict(idle, params)

        # Roundtrip: dict values should match trial params
        assert abs(d["dream_idle_threshold"] - trial.params["dream_idle_threshold"]) < 1e-6
        assert abs(d["eta"] - params.eta) < 1e-6
        assert abs(d["min_sep"] - params.min_sep) < 1e-6
        assert abs(d["merge_threshold"] - params.merge_threshold) < 1e-6
        assert abs(d["prune_threshold"] - params.prune_threshold) < 1e-6
        assert d["n_probes"] == params.n_probes
        assert abs(d["separation_rate"] - params.separation_rate) < 1e-6

    def test_params_to_trial_dict_all_fields(self):
        """params_to_trial_dict produces a dict with all 7 required keys."""
        dp = DreamParams()
        d = params_to_trial_dict(10.0, dp)

        required_keys = {
            "dream_idle_threshold",
            "eta",
            "min_sep",
            "merge_threshold",
            "prune_threshold",
            "n_probes",
            "separation_rate",
        }
        assert set(d.keys()) >= required_keys, (
            f"Missing keys: {required_keys - set(d.keys())}"
        )

    def test_trial_to_params_produces_valid_dreamparams(self):
        """trial_to_params always produces DreamParams that pass validate()."""
        sampler = optuna.samplers.GPSampler(seed=123)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def obj(trial):
            idle = trial.suggest_float("dream_idle_threshold", 0.1, 50.0)
            eta = trial.suggest_float("eta", 0.001, 0.05, log=True)
            min_sep = trial.suggest_float("min_sep", 0.01, 0.5)
            merge = trial.suggest_float("merge_threshold", 0.7, 0.99)
            prune = trial.suggest_float("prune_threshold", 0.8, 0.999)
            n_probes = trial.suggest_int("n_probes", 1, 10)
            sep_rate = trial.suggest_float("separation_rate", 0.01, 0.5)
            return 0.0

        study.optimize(obj, n_trials=10)

        for trial in study.trials:
            idle, params = trial_to_params(trial)
            params.validate()  # Must not raise
            assert 0.1 <= idle <= 50.0
            assert isinstance(params, DreamParams)


# ---------------------------------------------------------------------------
# 3. Parameter importance test
# ---------------------------------------------------------------------------
class TestParamImportance:
    """Verify Optuna's importance analysis works on a completed study."""

    def test_importance_analysis_runs(self):
        """optuna.importance.get_param_importances works on a completed study."""
        opt = OptunaOptimizer(
            objective=_quadratic_objective,
            seed=42,
        )
        opt.optimize(n_trials=20)

        importances = optuna.importance.get_param_importances(opt.study)

        assert isinstance(importances, dict)
        assert len(importances) > 0
        # dream_idle_threshold should be among the parameters
        assert "dream_idle_threshold" in importances
        # All importance values should be non-negative
        for name, value in importances.items():
            assert value >= 0, f"Negative importance for {name}: {value}"
