"""Optuna GPSampler optimization of dream hyperparameters on the longitudinal eval.

Replaces CMA-ES with Bayesian optimization. Supports capacity-gated dreaming,
seeding from prior results, expanded search (beta, gating threshold), and
post-optimization revalidation of top candidates.

Usage:
    .venv/bin/python run_optuna_optimizer.py --capacity-gating
    .venv/bin/python run_optuna_optimizer.py --capacity-gating --seed-from output/cma_gated_results.json
    .venv/bin/python run_optuna_optimizer.py --capacity-gating --expand-search --revalidate
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

from dream_ops import DreamParams
from optuna_optimizer import OptunaOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_objective(dataset, use_llm_judge: bool = True, capacity_gating: bool = False):
    """Create objective matching the OptunaOptimizer protocol."""
    from mabench.hermes_agent import HermesMemoryAgent
    from longitudinal_eval import LongitudinalEvaluator

    eval_count = [0]

    def objective(dream_idle_threshold: float, params: DreamParams) -> float:
        eval_count[0] += 1
        n = eval_count[0]

        logger.info(
            "  [eval %d] idle=%.2f eta=%.5f min_sep=%.4f merge=%.4f prune=%.4f",
            n, dream_idle_threshold, params.eta, params.min_sep,
            params.merge_threshold, params.prune_threshold,
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


def load_seed_params(path: str) -> dict | None:
    """Load best params from a previous results JSON (CMA-ES or Optuna)."""
    try:
        with open(path) as f:
            data = json.load(f)
        seed = {
            "dream_idle_threshold": data["best_dream_idle"],
        }
        bp = data["best_params"]
        if isinstance(bp, dict):
            seed.update({
                "eta": bp["eta"],
                "min_sep": bp["min_sep"],
                "merge_threshold": bp["merge_threshold"],
                "prune_threshold": bp["prune_threshold"],
                "n_probes": bp["n_probes"],
                "separation_rate": bp["separation_rate"],
            })
        logger.info("Seeded from %s: idle=%.2f", path, seed["dream_idle_threshold"])
        return seed
    except Exception as e:
        logger.warning("Could not load seed from %s: %s", path, e)
        return None


def run_optuna(
    n_trials: int = 50,
    seed: int = 42,
    use_llm_judge: bool = True,
    capacity_gating: bool = False,
    expand_search: bool = False,
    revalidate: bool = False,
    seed_from: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Run Optuna optimization and return results."""
    from longitudinal_eval import generate_dataset

    logger.info("Generating longitudinal dataset (seed=%d)...", seed)
    dataset = generate_dataset(seed=seed)
    logger.info("Dataset: %d sessions, %d questions", len(dataset.sessions), len(dataset.questions))

    objective = make_objective(dataset, use_llm_judge=use_llm_judge, capacity_gating=capacity_gating)

    seed_params = load_seed_params(seed_from) if seed_from else None

    logger.info("━━━ Optuna GPSampler Optimization ━━━")
    logger.info("  n_trials=%d  seed=%d  capacity_gating=%s  expand=%s  revalidate=%s",
                n_trials, seed, capacity_gating, expand_search, revalidate)

    optimizer = OptunaOptimizer(
        objective=objective,
        seed=seed,
        expand_search=expand_search,
        seed_params=seed_params,
    )

    t_start = time.time()
    result = optimizer.optimize(n_trials=n_trials)
    total_time = time.time() - t_start

    bp = result["best_params"]
    print(f"\n{'=' * 70}")
    print("OPTUNA GPSampler OPTIMIZATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  Best composite score: {result['best_score']:.4f}")
    print(f"  Dream idle threshold: {result['best_dream_idle']:.2f}")
    print(f"  DreamParams:")
    print(f"    eta              = {bp.eta:.5f}")
    print(f"    min_sep          = {bp.min_sep:.4f}")
    print(f"    merge_threshold  = {bp.merge_threshold:.4f}")
    print(f"    prune_threshold  = {bp.prune_threshold:.4f}")
    print(f"    n_probes         = {bp.n_probes}")
    print(f"    separation_rate  = {bp.separation_rate:.4f}")
    print(f"  Trials: {result['n_trials']}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"{'=' * 70}")

    # Revalidate top candidates
    if revalidate:
        logger.info("━━━ Revalidating top 5 candidates (5x each, median) ━━━")
        t_reval = time.time()
        reval_result = optimizer.revalidate_top_k(k=5, n_repeats=5)
        reval_time = time.time() - t_reval
        result["revalidation"] = reval_result
        print(f"\nRevalidation ({reval_time/60:.1f} min):")
        for i, r in enumerate(reval_result["revalidated"]):
            print(f"  #{i+1}: median={r['median_score']:.4f}  idle={r['dream_idle_threshold']:.2f}")
        total_time += reval_time

    # Serialize for JSON
    result_json = {
        "best_dream_idle": result["best_dream_idle"],
        "best_score": result["best_score"],
        "best_params": {
            "eta": bp.eta,
            "min_sep": bp.min_sep,
            "prune_threshold": bp.prune_threshold,
            "merge_threshold": bp.merge_threshold,
            "n_probes": bp.n_probes,
            "separation_rate": bp.separation_rate,
        },
        "n_trials": result["n_trials"],
        "total_seconds": round(total_time, 1),
        "capacity_gating": capacity_gating,
        "expand_search": expand_search,
        "history": result["history"],
    }
    if "revalidation" in result:
        # Serialize DreamParams in revalidation results
        reval_serialized = []
        for entry in result["revalidation"]["revalidated"]:
            p = entry["params"]
            reval_serialized.append({
                "trial_number": entry["trial_number"],
                "original_score": entry["original_score"],
                "median_score": entry["median_score"],
                "std_score": entry["std_score"],
                "n_repeats": entry["n_repeats"],
                "dream_idle_threshold": entry["dream_idle_threshold"],
                "params": {
                    "eta": p.eta,
                    "min_sep": p.min_sep,
                    "prune_threshold": p.prune_threshold,
                    "merge_threshold": p.merge_threshold,
                    "n_probes": p.n_probes,
                    "separation_rate": p.separation_rate,
                },
            })
        result_json["revalidation"] = reval_serialized

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result_json, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return result_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna GPSampler dream parameter optimization")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Use SubEM only (no OpenRouter API calls)")
    parser.add_argument("--capacity-gating", action="store_true",
                        help="Enable capacity-gated dreaming (Lean formula)")
    parser.add_argument("--expand-search", action="store_true",
                        help="Add beta + capacity_gating_low to search space")
    parser.add_argument("--revalidate", action="store_true",
                        help="Re-evaluate top 5 candidates 5x each after optimization")
    parser.add_argument("--seed-from", type=str, default=None,
                        help="Path to prior results JSON to seed from")
    parser.add_argument("--output", type=str, default="output/optuna_results.json",
                        help="Path to save results JSON")
    args = parser.parse_args()

    run_optuna(
        n_trials=args.n_trials,
        seed=args.seed,
        use_llm_judge=not args.no_llm_judge,
        capacity_gating=args.capacity_gating,
        expand_search=args.expand_search,
        revalidate=args.revalidate,
        seed_from=args.seed_from,
        output_path=args.output,
    )
