"""CMA-ES optimizer for dream cycle hyperparameters.

Searches a 7D parameter space using CMA-ES with sigmoid transforms to
respect bounded intervals and Lean-proven inter-parameter constraints.

The optimizer is agnostic to the scoring function: any callable matching
the ObjectiveFunction protocol can be plugged in.

Dependencies: numpy, cma (pycma 4.x), dream_ops.DreamParams
"""

from __future__ import annotations

from typing import Protocol

import cma
import numpy as np

from dream_ops import DreamParams


# ---------------------------------------------------------------------------
# Objective function protocol
# ---------------------------------------------------------------------------


class ObjectiveFunction(Protocol):
    """Protocol for dream parameter evaluation functions.

    Implementations receive a dream idle threshold (days) and DreamParams,
    and return a score where HIGHER IS BETTER (the optimizer negates
    internally because CMA-ES minimizes).
    """

    def __call__(self, dream_idle_threshold: float, params: DreamParams) -> float: ...


# ---------------------------------------------------------------------------
# Parameter names and bounds (for reference / logging)
# ---------------------------------------------------------------------------

PARAM_NAMES = [
    "dream_idle_threshold",  # float, [0.5, 30.0] -- days of idle before dream
    "eta",                   # float, (0, min_sep/2) -- Lean bound
    "min_sep",               # float, (0, 1] -- Lean bound
    "prune_threshold",       # float, (merge_threshold, 1] -- Lean bound
    "merge_threshold",       # float, (0, prune_threshold) -- Lean bound
    "n_probes",              # int, [10, 500]
    "separation_rate",       # float, [0.001, 0.1]
]


# ---------------------------------------------------------------------------
# Sigmoid transforms: unconstrained <-> bounded
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Map (-inf, inf) -> (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _inverse_sigmoid(y: float) -> float:
    """Map (0, 1) -> (-inf, inf).  Clamps to avoid log(0)."""
    y = np.clip(y, 1e-7, 1 - 1e-7)
    return float(np.log(y / (1 - y)))


# ---------------------------------------------------------------------------
# Raw <-> Params mapping
# ---------------------------------------------------------------------------


def raw_to_params(raw: np.ndarray) -> tuple[float, DreamParams]:
    """Map CMA-ES raw 7D vector to (dream_idle_threshold, DreamParams).

    Uses sigmoid transforms to map each unbounded coordinate to its
    respective bounded interval, then projects to satisfy Lean inter-
    parameter constraints (eta < min_sep/2, merge < prune).
    """
    # Individual sigmoid mappings to bounded intervals
    dream_idle = 0.5 + 29.5 * _sigmoid(raw[0])      # [0.5, 30.0]
    eta = 0.001 + 0.149 * _sigmoid(raw[1])           # [0.001, 0.15]
    min_sep = 0.05 + 0.95 * _sigmoid(raw[2])         # [0.05, 1.0]
    prune_t = 0.5 + 0.5 * _sigmoid(raw[3])           # [0.5, 1.0]
    merge_t = 0.1 + 0.89 * _sigmoid(raw[4])          # [0.1, 0.99]
    n_probes = int(round(10 + 490 * _sigmoid(raw[5])))  # [10, 500]
    sep_rate = 0.001 + 0.099 * _sigmoid(raw[6])       # [0.001, 0.1]

    # ------------------------------------------------------------------
    # Lean constraint projection
    # ------------------------------------------------------------------
    # Constraint 1: 0 < eta < min_sep / 2
    eta = min(eta, min_sep / 2 - 0.001)
    # Safety: eta must be positive
    eta = max(eta, 0.0001)

    # Constraint 2: merge_threshold < prune_threshold
    if merge_t >= prune_t:
        merge_t = prune_t - 0.01
    # Safety: merge must be positive
    merge_t = max(merge_t, 0.001)

    params = DreamParams(
        eta=eta,
        min_sep=min_sep,
        prune_threshold=prune_t,
        merge_threshold=merge_t,
        n_probes=n_probes,
        separation_rate=sep_rate,
    )
    return dream_idle, params


def params_to_raw(dream_idle: float, params: DreamParams) -> np.ndarray:
    """Map (dream_idle_threshold, DreamParams) to CMA-ES raw 7D vector.

    Inverse of raw_to_params (up to floating-point precision and
    Lean constraint projection).
    """
    return np.array([
        _inverse_sigmoid((dream_idle - 0.5) / 29.5),
        _inverse_sigmoid((params.eta - 0.001) / 0.149),
        _inverse_sigmoid((params.min_sep - 0.05) / 0.95),
        _inverse_sigmoid((params.prune_threshold - 0.5) / 0.5),
        _inverse_sigmoid((params.merge_threshold - 0.1) / 0.89),
        _inverse_sigmoid((params.n_probes - 10) / 490),
        _inverse_sigmoid((params.separation_rate - 0.001) / 0.099),
    ])


# ---------------------------------------------------------------------------
# DreamOptimizer: CMA-ES runner
# ---------------------------------------------------------------------------


class DreamOptimizer:
    """CMA-ES optimizer for dream cycle hyperparameters.

    Parameters
    ----------
    objective : ObjectiveFunction
        Scoring function. Returns HIGHER = BETTER.
    seed : int
        Random seed for reproducibility.
    max_evals : int
        Maximum number of objective evaluations.
    sigma0 : float
        Initial step size for CMA-ES.
    popsize : int | None
        Population size per generation. None uses CMA-ES default.
    """

    def __init__(
        self,
        objective: ObjectiveFunction,
        seed: int = 42,
        max_evals: int = 100,
        sigma0: float = 1.0,
        popsize: int | None = None,
    ):
        self.objective = objective
        self.seed = seed
        self.max_evals = max_evals
        self.sigma0 = sigma0
        self.popsize = popsize

    def optimize(self) -> dict:
        """Run CMA-ES optimization.

        Returns
        -------
        dict with keys:
            best_dream_idle : float
            best_params : DreamParams
            best_score : float  (higher is better)
            n_evals : int
            history : list[dict]  (per-generation best)
        """
        # Initial point: map default params to raw space
        x0 = params_to_raw(10.0, DreamParams())

        opts: dict = {
            "seed": self.seed,
            "maxfevals": self.max_evals,
            "verbose": -1,
            "tolfun": 1e-4,
        }
        if self.popsize is not None:
            opts["popsize"] = self.popsize

        es = cma.CMAEvolutionStrategy(x0.tolist(), self.sigma0, opts)

        history: list[dict] = []
        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            for x in solutions:
                try:
                    dream_idle, params = raw_to_params(np.asarray(x))
                    score = self.objective(dream_idle, params)
                    fitness = -score  # CMA-ES minimizes
                except Exception:
                    fitness = 1e6  # worst case on any error
                fitnesses.append(fitness)
            es.tell(solutions, fitnesses)

            # Log best of this generation
            best_idx = int(np.argmin(fitnesses))
            try:
                best_idle, best_params = raw_to_params(np.asarray(solutions[best_idx]))
            except Exception:
                best_idle, best_params = 10.0, DreamParams()
            history.append(
                {
                    "generation": len(history),
                    "best_fitness": -fitnesses[best_idx],
                    "best_idle": best_idle,
                    "best_params": best_params,
                }
            )

        # Extract overall best
        best_raw = es.result.xbest
        try:
            best_idle, best_params = raw_to_params(np.asarray(best_raw))
        except Exception:
            best_idle, best_params = 10.0, DreamParams()

        return {
            "best_dream_idle": best_idle,
            "best_params": best_params,
            "best_score": -es.result.fbest,
            "n_evals": es.result.evaluations,
            "history": history,
        }


# ---------------------------------------------------------------------------
# Grid sweep helper
# ---------------------------------------------------------------------------


def grid_sweep(
    objective: ObjectiveFunction,
    dream_idle_values: list[float] | None = None,
    eta_values: list[float] | None = None,
) -> list[dict]:
    """Run a grid sweep over key dream parameters.

    Other DreamParams fields are kept at defaults.  Returns a list of
    result dicts sorted by score DESCENDING (best first).

    Parameters
    ----------
    objective : ObjectiveFunction
        Scoring function. Returns HIGHER = BETTER.
    dream_idle_values : list[float]
        Idle threshold values to try.
    eta_values : list[float]
        Hebbian learning rate values to try.

    Returns
    -------
    list[dict] with keys: score, dream_idle_threshold, params
    """
    if dream_idle_values is None:
        dream_idle_values = [1, 3, 7, 14, 30]
    if eta_values is None:
        eta_values = [0.005, 0.01, 0.02]

    defaults = DreamParams()
    results: list[dict] = []

    for idle_val in dream_idle_values:
        for eta_val in eta_values:
            # Ensure Lean bounds: eta < min_sep / 2
            safe_eta = min(eta_val, defaults.min_sep / 2 - 0.001)
            safe_eta = max(safe_eta, 0.0001)

            try:
                params = DreamParams(
                    eta=safe_eta,
                    min_sep=defaults.min_sep,
                    prune_threshold=defaults.prune_threshold,
                    merge_threshold=defaults.merge_threshold,
                    n_probes=defaults.n_probes,
                    separation_rate=defaults.separation_rate,
                )
            except ValueError:
                continue  # skip invalid combinations

            try:
                score = objective(idle_val, params)
            except Exception:
                score = float("-inf")

            results.append(
                {
                    "score": score,
                    "dream_idle_threshold": idle_val,
                    "params": params,
                }
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("CMA-ES Dream Optimizer — demo run")
    print("=" * 60)

    def demo_objective(dream_idle_threshold: float, params: DreamParams) -> float:
        """Parabola centered on defaults — for testing only."""
        dp = DreamParams()
        penalty = (
            (dream_idle_threshold - 10.0) ** 2
            + 100 * (params.eta - dp.eta) ** 2
            + 10 * (params.min_sep - dp.min_sep) ** 2
            + 10 * (params.prune_threshold - dp.prune_threshold) ** 2
            + 10 * (params.merge_threshold - dp.merge_threshold) ** 2
            + (params.n_probes - dp.n_probes) ** 2 / 10000
            + 100 * (params.separation_rate - dp.separation_rate) ** 2
        )
        return -penalty

    # Grid sweep first
    print("\n--- Grid Sweep ---")
    grid_results = grid_sweep(demo_objective)
    for i, r in enumerate(grid_results[:5]):
        print(f"  #{i + 1}: score={r['score']:.4f}  idle={r['dream_idle_threshold']}")

    # CMA-ES
    print("\n--- CMA-ES (50 evals) ---")
    optimizer = DreamOptimizer(
        objective=demo_objective,
        seed=42,
        max_evals=50,
        sigma0=0.5,
    )
    result = optimizer.optimize()
    print(f"  Best score: {result['best_score']:.6f}")
    print(f"  Best idle:  {result['best_dream_idle']:.2f}")
    print(f"  Best params: eta={result['best_params'].eta:.5f}, "
          f"min_sep={result['best_params'].min_sep:.4f}")
    print(f"  Evaluations: {result['n_evals']}")
    print(f"  Generations: {len(result['history'])}")
    print("\nDone.")
