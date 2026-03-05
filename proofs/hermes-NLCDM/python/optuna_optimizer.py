"""Optuna GPSampler-based optimizer for dream cycle hyperparameters.

Replaces CMA-ES with Bayesian optimization via Gaussian Process surrogate.
Handles bounded search directly (no sigmoid transforms needed), supports
seeding with known-good params, expanded search space (beta, capacity_gating),
and post-optimization revalidation with median aggregation for noise reduction.

Dependencies: optuna (>=4.0), numpy, dream_ops.DreamParams
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

import numpy as np
import optuna

from dream_ops import DreamParams

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Objective function protocol (same as CMA-ES version)
# ---------------------------------------------------------------------------


class ObjectiveFunction(Protocol):
    """Protocol for dream parameter evaluation functions.

    Implementations receive a dream idle threshold and DreamParams,
    and return a score where HIGHER IS BETTER.
    """

    def __call__(self, dream_idle_threshold: float, params: DreamParams) -> float: ...


# ---------------------------------------------------------------------------
# Parameter bounds (matching Lean-proven constraints)
# ---------------------------------------------------------------------------

PARAM_BOUNDS = {
    "dream_idle_threshold": (0.1, 50.0),
    "eta": (0.001, 0.05),
    "min_sep": (0.01, 0.5),
    "merge_threshold": (0.7, 0.99),
    "prune_threshold": (0.8, 0.999),
    "n_probes": (1, 10),
    "separation_rate": (0.01, 0.5),
}

EXPANDED_BOUNDS = {
    "beta": (1.0, 20.0),
    "capacity_gating_low": (0.1, 2.0),
}


# ---------------------------------------------------------------------------
# Trial <-> Params conversion helpers
# ---------------------------------------------------------------------------


def trial_to_params(trial: optuna.trial.FrozenTrial) -> tuple[float, DreamParams]:
    """Extract (dream_idle_threshold, DreamParams) from a completed Optuna trial.

    Applies Lean constraint projection to ensure validity:
    - eta < min_sep / 2
    - merge_threshold < prune_threshold
    """
    p = trial.params

    idle = p["dream_idle_threshold"]
    eta = p["eta"]
    min_sep = p["min_sep"]
    merge_t = p["merge_threshold"]
    prune_t = p["prune_threshold"]
    n_probes = p["n_probes"]
    sep_rate = p["separation_rate"]

    # Lean constraint projection
    eta = min(eta, min_sep / 2 - 0.001)
    eta = max(eta, 0.0001)
    if merge_t >= prune_t:
        merge_t = prune_t - 0.01
    merge_t = max(merge_t, 0.001)

    kwargs = dict(
        eta=eta,
        min_sep=min_sep,
        prune_threshold=prune_t,
        merge_threshold=merge_t,
        n_probes=n_probes,
        separation_rate=sep_rate,
    )

    # Handle expanded params if present
    if "beta" in p:
        kwargs["beta_high"] = p["beta"]
    if "capacity_gating_low" in p:
        # Store in the trial dict for downstream use; DreamParams doesn't
        # have this field natively, so we pass it through the result dict.
        pass

    params = DreamParams(**kwargs)
    return idle, params


def params_to_trial_dict(
    idle: float, params: DreamParams, capacity_gating_low: float | None = None
) -> dict:
    """Convert (dream_idle_threshold, DreamParams) to a dict suitable for
    study.enqueue_trial().

    Includes all 7 core parameters. Optionally includes expanded params.
    """
    d = {
        "dream_idle_threshold": idle,
        "eta": params.eta,
        "min_sep": params.min_sep,
        "merge_threshold": params.merge_threshold,
        "prune_threshold": params.prune_threshold,
        "n_probes": params.n_probes,
        "separation_rate": params.separation_rate,
    }
    if capacity_gating_low is not None:
        d["capacity_gating_low"] = capacity_gating_low
    return d


# ---------------------------------------------------------------------------
# OptunaOptimizer
# ---------------------------------------------------------------------------


class OptunaOptimizer:
    """Bayesian optimizer for dream cycle hyperparameters using Optuna GPSampler.

    Parameters
    ----------
    objective : ObjectiveFunction
        Scoring function. Returns HIGHER = BETTER.
    seed : int
        Random seed for reproducibility.
    seed_params : dict | None
        Known-good parameter dict to enqueue as the first trial.
    expand_search : bool
        If True, also optimize beta (inverse temperature) and
        capacity_gating_low threshold.
    """

    def __init__(
        self,
        objective: ObjectiveFunction,
        seed: int = 42,
        seed_params: dict | None = None,
        expand_search: bool = False,
    ):
        self.objective = objective
        self.seed = seed
        self.seed_params = seed_params
        self.expand_search = expand_search

        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.GPSampler(seed=seed)
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
        )

        # Enqueue seed trial if provided
        if seed_params is not None:
            self.study.enqueue_trial(seed_params)

    def _suggest_params(self, trial: optuna.Trial) -> tuple[float, DreamParams, dict]:
        """Suggest parameters from the trial, apply Lean projection, return
        (idle, DreamParams, extra_dict).

        extra_dict contains expanded params (capacity_gating_low) that are
        not part of DreamParams but need to be recorded.
        """
        idle = trial.suggest_float(
            "dream_idle_threshold",
            PARAM_BOUNDS["dream_idle_threshold"][0],
            PARAM_BOUNDS["dream_idle_threshold"][1],
        )
        eta = trial.suggest_float(
            "eta",
            PARAM_BOUNDS["eta"][0],
            PARAM_BOUNDS["eta"][1],
            log=True,
        )
        min_sep = trial.suggest_float(
            "min_sep",
            PARAM_BOUNDS["min_sep"][0],
            PARAM_BOUNDS["min_sep"][1],
        )
        merge_t = trial.suggest_float(
            "merge_threshold",
            PARAM_BOUNDS["merge_threshold"][0],
            PARAM_BOUNDS["merge_threshold"][1],
        )
        prune_t = trial.suggest_float(
            "prune_threshold",
            PARAM_BOUNDS["prune_threshold"][0],
            PARAM_BOUNDS["prune_threshold"][1],
        )
        n_probes = trial.suggest_int(
            "n_probes",
            PARAM_BOUNDS["n_probes"][0],
            PARAM_BOUNDS["n_probes"][1],
        )
        sep_rate = trial.suggest_float(
            "separation_rate",
            PARAM_BOUNDS["separation_rate"][0],
            PARAM_BOUNDS["separation_rate"][1],
        )

        # Lean constraint projection
        eta = min(eta, min_sep / 2 - 0.001)
        eta = max(eta, 0.0001)
        if merge_t >= prune_t:
            merge_t = prune_t - 0.01
        merge_t = max(merge_t, 0.001)

        kwargs = dict(
            eta=eta,
            min_sep=min_sep,
            prune_threshold=prune_t,
            merge_threshold=merge_t,
            n_probes=n_probes,
            separation_rate=sep_rate,
        )

        extra = {}

        if self.expand_search:
            beta = trial.suggest_float(
                "beta",
                EXPANDED_BOUNDS["beta"][0],
                EXPANDED_BOUNDS["beta"][1],
            )
            kwargs["beta_high"] = beta

            cap_low = trial.suggest_float(
                "capacity_gating_low",
                EXPANDED_BOUNDS["capacity_gating_low"][0],
                EXPANDED_BOUNDS["capacity_gating_low"][1],
            )
            extra["capacity_gating_low"] = cap_low

        params = DreamParams(**kwargs)
        return idle, params, extra

    def optimize(self, n_trials: int = 50) -> dict:
        """Run Bayesian optimization for n_trials.

        Returns
        -------
        dict with keys:
            best_dream_idle : float
            best_params : DreamParams
            best_score : float (higher is better)
            n_trials : int
            history : list[dict] (per-trial results)
        """
        history: list[dict] = []

        def optuna_objective(trial: optuna.Trial) -> float:
            idle, params, extra = self._suggest_params(trial)

            try:
                score = self.objective(idle, params)
            except Exception as e:
                logger.warning("Trial %d failed: %s", trial.number, e)
                return float("-inf")

            entry = {
                "trial": trial.number,
                "score": score,
                "dream_idle_threshold": idle,
                "params": params,
                "eta": params.eta,
                "min_sep": params.min_sep,
                "merge_threshold": params.merge_threshold,
                "prune_threshold": params.prune_threshold,
                "n_probes": params.n_probes,
                "separation_rate": params.separation_rate,
            }
            if self.expand_search:
                entry["beta"] = params.beta_high
                entry["capacity_gating_low"] = extra.get("capacity_gating_low")

            history.append(entry)
            return score

        self.study.optimize(optuna_objective, n_trials=n_trials)

        # Extract best
        best_trial = self.study.best_trial
        best_idle, best_params = trial_to_params(best_trial)

        self._history = history

        return {
            "best_dream_idle": best_idle,
            "best_params": best_params,
            "best_score": self.study.best_value,
            "n_trials": len(self.study.trials),
            "history": history,
        }

    def revalidate_top_k(self, k: int = 5, n_repeats: int = 5) -> dict:
        """Re-evaluate the top K trials n_repeats times each, using median.

        Returns
        -------
        dict with key 'revalidated': list of dicts sorted by median_score desc,
        each containing:
            trial_number, original_score, median_score, std_score,
            n_repeats, dream_idle_threshold, params
        """
        # Get top-k trials by value
        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        completed.sort(key=lambda t: t.value, reverse=True)
        top_trials = completed[:k]

        results = []
        for trial in top_trials:
            idle, params = trial_to_params(trial)
            scores = []
            for _ in range(n_repeats):
                try:
                    s = self.objective(idle, params)
                except Exception:
                    s = float("-inf")
                scores.append(s)

            median = float(np.median(scores))
            std = float(np.std(scores))

            results.append({
                "trial_number": trial.number,
                "original_score": trial.value,
                "median_score": median,
                "std_score": std,
                "n_repeats": n_repeats,
                "dream_idle_threshold": idle,
                "params": params,
            })

        # Sort by median score descending
        results.sort(key=lambda r: r["median_score"], reverse=True)

        return {"revalidated": results}
