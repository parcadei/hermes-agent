"""CMA-ES optimization of dream hyperparameters on the longitudinal eval.

Wires the DreamOptimizer to the LongitudinalEvaluator. Each CMA-ES evaluation
creates a fresh agent with the candidate DreamParams, runs 200 sessions + 100
questions, and returns the composite score.

The dataset is generated ONCE and reused across all evaluations.

Usage:
    .venv/bin/python run_cma_optimizer.py
    .venv/bin/python run_cma_optimizer.py --max-evals 30 --popsize 6
    .venv/bin/python run_cma_optimizer.py --output output/cma_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "mabench"))

from cma_optimizer import DreamOptimizer, params_to_raw
from dream_ops import DreamParams
from longitudinal_eval import LongitudinalEvaluator, generate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_objective(dataset, use_llm_judge: bool = True, capacity_gating: bool = False):
    """Create an objective function that evaluates dream params on the longitudinal eval.

    The returned callable matches the ObjectiveFunction protocol:
        (dream_idle_threshold: float, params: DreamParams) -> float

    Higher is better (composite score).
    """
    from mabench.hermes_agent import HermesMemoryAgent

    eval_count = [0]

    def objective(dream_idle_threshold: float, params: DreamParams) -> float:
        eval_count[0] += 1
        n = eval_count[0]

        logger.info(
            "  [eval %d] idle=%.2f eta=%.5f min_sep=%.4f merge=%.4f prune=%.4f n_probes=%d sep_rate=%.4f",
            n, dream_idle_threshold, params.eta, params.min_sep,
            params.merge_threshold, params.prune_threshold,
            params.n_probes, params.separation_rate,
        )

        def agent_factory():
            return HermesMemoryAgent(
                model="openai/gpt-4o-mini",
                dream_interval=0,
                dream_params=params,
                associative_retrieval=False,
            )

        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            dream_idle_threshold=dream_idle_threshold,
            use_llm_judge=use_llm_judge,
            capacity_gating=capacity_gating,
        )

        t0 = time.time()
        result = evaluator.evaluate(agent_factory)
        elapsed = time.time() - t0

        composite = result["composite"]
        cs = result["category_scores"]
        logger.info(
            "  [eval %d] composite=%.4f (cur=%.3f fgt=%.3f rein=%.3f xdom=%.3f) "
            "dreams=%d  %.1fs",
            n, composite,
            cs.get("current_fact", 0),
            cs.get("graceful_forgetting", 0),
            cs.get("reinforced_recall", 0),
            cs.get("cross_domain", 0),
            result["n_dreams"], elapsed,
        )
        return composite

    return objective


def run_cma(
    max_evals: int = 40,
    popsize: int = 6,
    sigma0: float = 0.5,
    seed: int = 42,
    use_llm_judge: bool = True,
    capacity_gating: bool = False,
    output_path: str | None = None,
) -> dict:
    """Run CMA-ES optimization and return results."""

    logger.info("Generating longitudinal dataset (seed=%d)...", seed)
    dataset = generate_dataset(seed=seed)
    logger.info(
        "Dataset: %d sessions, %d questions",
        len(dataset.sessions), len(dataset.questions),
    )

    objective = make_objective(dataset, use_llm_judge=use_llm_judge, capacity_gating=capacity_gating)

    # Grid sweep showed best at threshold ~0-5. Start CMA-ES near threshold=2.
    # Use default DreamParams as the starting point for other dimensions.
    x0_idle = 2.0
    x0_params = DreamParams()

    logger.info("━━━ CMA-ES Optimization ━━━")
    logger.info("  max_evals=%d  popsize=%d  sigma0=%.2f  seed=%d", max_evals, popsize, sigma0, seed)
    logger.info("  x0: idle=%.1f  eta=%.5f  min_sep=%.4f  merge=%.4f  prune=%.4f",
                x0_idle, x0_params.eta, x0_params.min_sep,
                x0_params.merge_threshold, x0_params.prune_threshold)

    # Override the optimizer's default x0 by constructing it manually
    optimizer = DreamOptimizer(
        objective=objective,
        seed=seed,
        max_evals=max_evals,
        sigma0=sigma0,
        popsize=popsize,
    )
    # Patch x0 to start near the grid sweep optimum
    import cma
    import numpy as np
    x0 = params_to_raw(x0_idle, x0_params)

    opts = {
        "seed": seed,
        "maxfevals": max_evals,
        "verbose": -1,
        "tolfun": 1e-4,
        "popsize": popsize,
    }

    t_start = time.time()
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    history = []
    best_so_far = float("-inf")

    while not es.stop():
        solutions = es.ask()
        fitnesses = []
        for x in solutions:
            try:
                from cma_optimizer import raw_to_params
                dream_idle, params = raw_to_params(np.asarray(x))
                score = objective(dream_idle, params)
                fitness = -score
            except Exception as e:
                logger.warning("  Eval failed: %s", e)
                fitness = 1e6
            fitnesses.append(fitness)
        es.tell(solutions, fitnesses)

        gen_best_score = -min(fitnesses)
        best_so_far = max(best_so_far, gen_best_score)
        gen = len(history)
        logger.info(
            "━━━ Generation %d: best_this_gen=%.4f  best_overall=%.4f  evals=%d ━━━",
            gen, gen_best_score, best_so_far, es.result.evaluations,
        )

        best_idx = int(np.argmin(fitnesses))
        try:
            from cma_optimizer import raw_to_params
            best_idle, best_params = raw_to_params(np.asarray(solutions[best_idx]))
        except Exception:
            best_idle, best_params = 2.0, DreamParams()

        history.append({
            "generation": gen,
            "best_score": gen_best_score,
            "best_overall": best_so_far,
            "best_idle": best_idle,
            "best_eta": best_params.eta,
            "best_merge": best_params.merge_threshold,
            "best_prune": best_params.prune_threshold,
            "best_min_sep": best_params.min_sep,
            "best_n_probes": best_params.n_probes,
            "best_sep_rate": best_params.separation_rate,
            "evaluations": es.result.evaluations,
        })

    total_time = time.time() - t_start

    # Extract overall best
    from cma_optimizer import raw_to_params
    best_raw = es.result.xbest
    best_idle, best_params = raw_to_params(np.asarray(best_raw))
    best_score = -es.result.fbest

    result = {
        "best_dream_idle": round(best_idle, 4),
        "best_score": round(best_score, 6),
        "best_params": {
            "eta": round(best_params.eta, 6),
            "min_sep": round(best_params.min_sep, 6),
            "prune_threshold": round(best_params.prune_threshold, 6),
            "merge_threshold": round(best_params.merge_threshold, 6),
            "n_probes": best_params.n_probes,
            "separation_rate": round(best_params.separation_rate, 6),
        },
        "n_evals": es.result.evaluations,
        "n_generations": len(history),
        "total_seconds": round(total_time, 1),
        "history": history,
    }

    # Print summary
    print(f"\n{'=' * 70}")
    print("CMA-ES OPTIMIZATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  Best composite score: {best_score:.4f}")
    print(f"  Dream idle threshold: {best_idle:.2f}")
    print(f"  DreamParams:")
    print(f"    eta              = {best_params.eta:.5f}")
    print(f"    min_sep          = {best_params.min_sep:.4f}")
    print(f"    merge_threshold  = {best_params.merge_threshold:.4f}")
    print(f"    prune_threshold  = {best_params.prune_threshold:.4f}")
    print(f"    n_probes         = {best_params.n_probes}")
    print(f"    separation_rate  = {best_params.separation_rate:.4f}")
    print(f"  Evaluations: {es.result.evaluations}")
    print(f"  Generations: {len(history)}")
    print(f"  Total time:  {total_time/60:.1f} min")
    print(f"{'=' * 70}")

    # Generation history
    print(f"\nGeneration History:")
    print(f"{'gen':>4s}  {'score':>8s}  {'best':>8s}  {'idle':>6s}  {'eta':>8s}  {'merge':>8s}")
    for h in history:
        print(
            f"{h['generation']:>4d}  {h['best_score']:>8.4f}  {h['best_overall']:>8.4f}  "
            f"{h['best_idle']:>6.2f}  {h['best_eta']:>8.5f}  {h['best_merge']:>8.4f}"
        )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CMA-ES dream parameter optimization")
    parser.add_argument("--max-evals", type=int, default=40,
                        help="Maximum objective evaluations (default: 40)")
    parser.add_argument("--popsize", type=int, default=6,
                        help="CMA-ES population size (default: 6)")
    parser.add_argument("--sigma0", type=float, default=0.5,
                        help="Initial step size (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Use SubEM only (no OpenRouter API calls)")
    parser.add_argument("--capacity-gating", action="store_true",
                        help="Enable capacity-gated dreaming (Lean formula)")
    parser.add_argument("--output", type=str, default="output/cma_results.json",
                        help="Path to save results JSON")
    args = parser.parse_args()

    run_cma(
        max_evals=args.max_evals,
        popsize=args.popsize,
        sigma0=args.sigma0,
        seed=args.seed,
        use_llm_judge=not args.no_llm_judge,
        capacity_gating=args.capacity_gating,
        output_path=args.output,
    )
