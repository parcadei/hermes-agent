"""Grid sweep and CMA-ES runner for the multi-domain longitudinal eval.

Tests the hypothesis: with diverse embeddings across 6 domains,
spreading activation outperforms direct cosine on cross-domain retrieval.

Usage:
    # Grid sweep: cosine vs associative at key thresholds
    .venv/bin/python run_multidomain_sweep.py --mode grid
    .venv/bin/python run_multidomain_sweep.py --mode grid --associative

    # CMA-ES optimization on best retrieval mode
    .venv/bin/python run_multidomain_sweep.py --mode cma --max-evals 30

    # Quick comparison: cosine vs associative at threshold=0
    .venv/bin/python run_multidomain_sweep.py --mode compare
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
from longitudinal_eval import LongitudinalEvaluator
from multidomain_eval import generate_multidomain_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_agent_factory(
    dream_params: DreamParams | None = None,
    associative: bool = False,
    sparse: bool = False,
    hybrid: bool = False,
):
    from mabench.hermes_agent import HermesMemoryAgent

    def factory():
        return HermesMemoryAgent(
            model="openai/gpt-4o-mini",
            dream_interval=0,
            dream_params=dream_params,
            associative_retrieval=associative,
            sparse_retrieval=sparse,
            hybrid_retrieval=hybrid,
        )

    return factory


def run_grid(
    idle_values: list[float],
    seed: int = 42,
    associative: bool = False,
    sparse: bool = False,
    hybrid: bool = False,
    output_path: str | None = None,
) -> list[dict]:
    """Run grid sweep on multi-domain dataset."""
    if hybrid:
        mode = "hybrid"
    elif sparse:
        mode = "sparse"
    elif associative:
        mode = "associative"
    else:
        mode = "cosine"
    logger.info(
        "Generating multi-domain dataset (seed=%d, mode=%s)...",
        seed, mode,
    )
    dataset = generate_multidomain_dataset(seed=seed)
    logger.info(
        "Dataset: %d sessions, %d questions",
        len(dataset.sessions), len(dataset.questions),
    )

    results = []
    default_params = DreamParams()

    for i, idle_val in enumerate(idle_values):
        logger.info(
            "━━━ [%d/%d] idle=%.1f mode=%s ━━━",
            i + 1, len(idle_values), idle_val, mode,
        )

        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            dream_idle_threshold=idle_val,
            use_llm_judge=True,
        )
        agent_factory = make_agent_factory(
            dream_params=default_params, associative=associative,
            sparse=sparse, hybrid=hybrid,
        )

        t0 = time.time()
        result = evaluator.evaluate(agent_factory)
        elapsed = time.time() - t0

        entry = {
            "dream_idle_threshold": idle_val,
            "composite": result["composite"],
            "category_scores": result["category_scores"],
            "n_dreams": result["n_dreams"],
            "sessions_processed": result["sessions_processed"],
            "scoring_method": result["scoring_method"],
            "associative": associative,
            "sparse": sparse,
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)

        logger.info(
            "  composite=%.4f  dreams=%d  method=%s  (%.1fs)",
            entry["composite"], entry["n_dreams"],
            entry["scoring_method"], elapsed,
        )
        for cat, score in result["category_scores"].items():
            logger.info("    %-25s %.4f", cat, score)

    results.sort(key=lambda r: r["composite"], reverse=True)

    # Print summary
    if hybrid:
        mode = "HYBRID"
    elif sparse:
        mode = "SPARSE"
    elif associative:
        mode = "ASSOCIATIVE"
    else:
        mode = "COSINE"
    print(f"\n{'=' * 80}")
    print(f"MULTI-DOMAIN GRID SWEEP — {mode} RETRIEVAL")
    print(f"{'=' * 80}")
    print(f"{'idle':>6s}  {'comp':>8s}  {'dreams':>6s}  "
          f"{'current':>8s}  {'forget':>8s}  {'reinforce':>8s}  {'cross_dom':>8s}")
    for r in results:
        cs = r["category_scores"]
        print(
            f"{r['dream_idle_threshold']:>6.1f}  "
            f"{r['composite']:>8.4f}  "
            f"{r['n_dreams']:>6d}  "
            f"{cs.get('current_fact', 0):>8.4f}  "
            f"{cs.get('graceful_forgetting', 0):>8.4f}  "
            f"{cs.get('reinforced_recall', 0):>8.4f}  "
            f"{cs.get('cross_domain', 0):>8.4f}"
        )
    print(f"{'=' * 80}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", output_path)

    return results


def run_compare(seed: int = 42, output_path: str | None = None) -> dict:
    """A/B/C/D comparison: cosine vs dense vs sparse vs hybrid."""
    logger.info("Generating multi-domain dataset (seed=%d)...", seed)
    dataset = generate_multidomain_dataset(seed=seed)

    # (name, associative, sparse, hybrid)
    modes = [
        ("cosine", False, False, False),
        ("associative", True, False, False),
        ("sparse", False, True, False),
        ("hybrid", False, False, True),
    ]

    results = {}
    for mode_name, assoc, sparse, hybrid in modes:
        logger.info("━━━ %s retrieval, idle=0 ━━━", mode_name.upper())

        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            dream_idle_threshold=0.0,
            use_llm_judge=True,
        )
        agent_factory = make_agent_factory(
            dream_params=DreamParams(), associative=assoc, sparse=sparse,
            hybrid=hybrid,
        )

        t0 = time.time()
        result = evaluator.evaluate(agent_factory)
        elapsed = time.time() - t0

        results[mode_name] = {
            "composite": result["composite"],
            "category_scores": result["category_scores"],
            "n_dreams": result["n_dreams"],
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info(
            "  %s: composite=%.4f  dreams=%d  (%.1fs)",
            mode_name, result["composite"], result["n_dreams"], elapsed,
        )

    # Print comparison
    print(f"\n{'=' * 90}")
    print("MULTI-DOMAIN: COSINE vs DENSE vs SPARSE vs HYBRID")
    print(f"{'=' * 90}")
    print(f"{'':>20s}  {'COSINE':>10s}  {'DENSE':>10s}  {'SPARSE':>10s}  "
          f"{'HYBRID':>10s}  {'Δ(H-C)':>8s}")
    print(f"{'─' * 90}")

    cos = results["cosine"]
    dense = results["associative"]
    sparse_r = results["sparse"]
    hybrid_r = results["hybrid"]

    for cat in ["current_fact", "graceful_forgetting", "reinforced_recall",
                "cross_domain"]:
        c = cos["category_scores"].get(cat, 0)
        d = dense["category_scores"].get(cat, 0)
        s = sparse_r["category_scores"].get(cat, 0)
        h = hybrid_r["category_scores"].get(cat, 0)
        delta = h - c
        sign = "+" if delta > 0 else ""
        print(f"  {cat:<20s}  {c:>10.4f}  {d:>10.4f}  {s:>10.4f}  "
              f"{h:>10.4f}  {sign}{delta:>7.4f}")

    c_comp = cos["composite"]
    d_comp = dense["composite"]
    s_comp = sparse_r["composite"]
    h_comp = hybrid_r["composite"]
    delta_hc = h_comp - c_comp
    sign = "+" if delta_hc > 0 else ""
    print(f"{'─' * 90}")
    print(f"  {'COMPOSITE':<20s}  {c_comp:>10.4f}  {d_comp:>10.4f}  "
          f"{s_comp:>10.4f}  {h_comp:>10.4f}  {sign}{delta_hc:>7.4f}")
    print(f"{'=' * 90}")

    scores = {"cosine": c_comp, "dense": d_comp, "sparse": s_comp,
              "hybrid": h_comp}
    winner = max(scores, key=scores.get)
    print(f"\nComposite winner: {winner.upper()} ({scores[winner]:.4f})")

    cd_scores = {
        "cosine": cos["category_scores"].get("cross_domain", 0),
        "dense": dense["category_scores"].get("cross_domain", 0),
        "sparse": sparse_r["category_scores"].get("cross_domain", 0),
        "hybrid": hybrid_r["category_scores"].get("cross_domain", 0),
    }
    cd_winner = max(cd_scores, key=cd_scores.get)
    print(f"Cross-domain winner: {cd_winner.upper()} "
          f"({cd_scores[cd_winner]:.4f})")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", output_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-domain longitudinal eval runner"
    )
    parser.add_argument(
        "--mode", choices=["grid", "compare", "cma"],
        default="compare",
        help="Run mode: grid sweep, quick compare, or CMA-ES (default: compare)",
    )
    parser.add_argument(
        "--idle-values", nargs="+", type=float, default=[0, 5, 10, 20, 50],
        help="Idle threshold values for grid mode (default: 0 5 10 20 50)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--associative", action="store_true")
    parser.add_argument(
        "--sparse", action="store_true",
        help="Use sparsemax Hopfield retrieval instead of dense softmax",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Use hybrid retrieval: cosine seeds + one Hopfield expansion step",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (auto-generated if not specified)",
    )
    parser.add_argument("--max-evals", type=int, default=30)
    parser.add_argument("--popsize", type=int, default=6)
    args = parser.parse_args()

    if args.output is None:
        if args.hybrid:
            suffix = "_hybrid"
        elif args.sparse:
            suffix = "_sparse"
        elif args.associative:
            suffix = "_assoc"
        else:
            suffix = ""
        args.output = f"output/multidomain_{args.mode}{suffix}.json"

    if args.mode == "grid":
        run_grid(
            idle_values=args.idle_values,
            seed=args.seed,
            associative=args.associative,
            sparse=args.sparse,
            hybrid=args.hybrid,
            output_path=args.output,
        )
    elif args.mode == "compare":
        run_compare(seed=args.seed, output_path=args.output)
    elif args.mode == "cma":
        # Import and run CMA-ES with multidomain objective
        from cma_optimizer import DreamOptimizer, params_to_raw, raw_to_params
        import cma
        import numpy as np

        logger.info("Generating multi-domain dataset (seed=%d)...", args.seed)
        dataset = generate_multidomain_dataset(seed=args.seed)

        from run_cma_optimizer import make_objective
        objective = make_objective(dataset, use_llm_judge=True)

        x0 = params_to_raw(2.0, DreamParams())
        opts = {
            "seed": args.seed,
            "maxfevals": args.max_evals,
            "verbose": -1,
            "tolfun": 1e-4,
            "popsize": args.popsize,
        }

        es = cma.CMAEvolutionStrategy(x0.tolist(), 0.5, opts)
        history = []
        best_so_far = float("-inf")

        while not es.stop():
            solutions = es.ask()
            fitnesses = []
            for x in solutions:
                try:
                    dream_idle, params = raw_to_params(np.asarray(x))
                    score = objective(dream_idle, params)
                    fitness = -score
                except Exception as e:
                    logger.warning("  Eval failed: %s", e)
                    fitness = 1e6
                fitnesses.append(fitness)
            es.tell(solutions, fitnesses)
            gen_best = -min(fitnesses)
            best_so_far = max(best_so_far, gen_best)
            logger.info(
                "━━━ Gen %d: best=%.4f overall=%.4f evals=%d ━━━",
                len(history), gen_best, best_so_far, es.result.evaluations,
            )
            history.append({
                "generation": len(history),
                "best_score": gen_best,
                "best_overall": best_so_far,
            })

        best_idle, best_params = raw_to_params(np.asarray(es.result.xbest))
        print(f"\n{'=' * 70}")
        print("CMA-ES RESULTS (MULTI-DOMAIN)")
        print(f"{'=' * 70}")
        print(f"  Best composite: {-es.result.fbest:.4f}")
        print(f"  Dream idle:     {best_idle:.2f}")
        print(f"  eta:            {best_params.eta:.5f}")
        print(f"  merge_thresh:   {best_params.merge_threshold:.4f}")
        print(f"  prune_thresh:   {best_params.prune_threshold:.4f}")
        print(f"  Evaluations:    {es.result.evaluations}")
        print(f"{'=' * 70}")

        result = {
            "best_score": round(-es.result.fbest, 6),
            "best_idle": round(best_idle, 4),
            "best_params": {
                "eta": round(best_params.eta, 6),
                "merge_threshold": round(best_params.merge_threshold, 6),
                "prune_threshold": round(best_params.prune_threshold, 6),
                "min_sep": round(best_params.min_sep, 6),
                "n_probes": best_params.n_probes,
                "separation_rate": round(best_params.separation_rate, 6),
            },
            "n_evals": es.result.evaluations,
            "history": history,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results saved to %s", args.output)
