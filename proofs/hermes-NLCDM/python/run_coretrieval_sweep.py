"""Co-retrieval graph sweep on multi-domain longitudinal eval.

Tests whether co-retrieval edges (built during retrieval practice)
improve cross-domain scores beyond the cosine baseline of 0.6125.

Sweep: min_coretrieval_count ∈ {1, 2, 3, 5}
Control: cosine baseline (no co-retrieval)
All runs: max_tokens=64, retrieval_practice=True

Usage:
    .venv/bin/python run_coretrieval_sweep.py
    .venv/bin/python run_coretrieval_sweep.py --seed 42 --output output/coretrieval_sweep.json
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
    coretrieval: bool = False,
    coretrieval_bonus: float = 0.3,
    coretrieval_min_count: float = 2.0,
):
    from mabench.hermes_agent import HermesMemoryAgent

    def factory():
        return HermesMemoryAgent(
            model="openai/gpt-4o-mini",
            dream_interval=0,
            dream_params=DreamParams(),
            coretrieval_retrieval=coretrieval,
            coretrieval_bonus=coretrieval_bonus,
            coretrieval_min_count=coretrieval_min_count,
        )

    return factory


def main():
    parser = argparse.ArgumentParser(
        description="Co-retrieval sweep on multi-domain longitudinal eval"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/coretrieval_sweep.json")
    parser.add_argument(
        "--min-counts", nargs="+", type=float, default=[1, 2, 3, 5],
        help="min_coretrieval_count values to sweep (default: 1 2 3 5)",
    )
    args = parser.parse_args()

    logger.info("Generating multi-domain dataset (seed=%d)...", args.seed)
    dataset = generate_multidomain_dataset(seed=args.seed)
    logger.info(
        "Dataset: %d sessions, %d questions",
        len(dataset.sessions), len(dataset.questions),
    )

    results = []

    # 1. Cosine baseline (no co-retrieval, no retrieval practice)
    logger.info("━━━ BASELINE: cosine (no retrieval practice) ━━━")
    evaluator = LongitudinalEvaluator(
        dataset=dataset,
        dream_idle_threshold=0.0,
        use_llm_judge=True,
        retrieval_practice=False,
    )
    agent_factory = make_agent_factory(coretrieval=False)
    t0 = time.time()
    result = evaluator.evaluate(agent_factory)
    elapsed = time.time() - t0

    entry = {
        "mode": "cosine_baseline",
        "coretrieval": False,
        "retrieval_practice": False,
        "min_count": None,
        "composite": result["composite"],
        "category_scores": result["category_scores"],
        "n_dreams": result["n_dreams"],
        "elapsed_seconds": round(elapsed, 1),
    }
    results.append(entry)
    logger.info("  composite=%.4f  (%.1fs)", entry["composite"], elapsed)
    for cat, score in result["category_scores"].items():
        logger.info("    %-25s %.4f", cat, score)

    # 2. Cosine with retrieval practice (no co-retrieval reranking)
    # This shows the effect of retrieval practice on cosine alone
    logger.info("━━━ COSINE + retrieval practice (no reranking) ━━━")
    evaluator = LongitudinalEvaluator(
        dataset=dataset,
        dream_idle_threshold=0.0,
        use_llm_judge=True,
        retrieval_practice=True,
    )
    agent_factory = make_agent_factory(coretrieval=False)
    t0 = time.time()
    result = evaluator.evaluate(agent_factory)
    elapsed = time.time() - t0

    entry = {
        "mode": "cosine_with_practice",
        "coretrieval": False,
        "retrieval_practice": True,
        "min_count": None,
        "composite": result["composite"],
        "category_scores": result["category_scores"],
        "n_dreams": result["n_dreams"],
        "elapsed_seconds": round(elapsed, 1),
    }
    results.append(entry)
    logger.info("  composite=%.4f  (%.1fs)", entry["composite"], elapsed)
    for cat, score in result["category_scores"].items():
        logger.info("    %-25s %.4f", cat, score)

    # 3. Co-retrieval sweep
    for mc in args.min_counts:
        logger.info("━━━ CO-RETRIEVAL: min_count=%.0f ━━━", mc)
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            dream_idle_threshold=0.0,
            use_llm_judge=True,
            retrieval_practice=True,
        )
        agent_factory = make_agent_factory(
            coretrieval=True,
            coretrieval_bonus=0.3,
            coretrieval_min_count=mc,
        )
        t0 = time.time()
        result = evaluator.evaluate(agent_factory)
        elapsed = time.time() - t0

        entry = {
            "mode": f"coretrieval_mc{int(mc)}",
            "coretrieval": True,
            "retrieval_practice": True,
            "min_count": mc,
            "composite": result["composite"],
            "category_scores": result["category_scores"],
            "n_dreams": result["n_dreams"],
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)
        logger.info("  composite=%.4f  (%.1fs)", entry["composite"], elapsed)
        for cat, score in result["category_scores"].items():
            logger.info("    %-25s %.4f", cat, score)

    # Print summary
    print(f"\n{'=' * 100}")
    print("CO-RETRIEVAL SWEEP — MULTI-DOMAIN LONGITUDINAL EVAL")
    print(f"{'=' * 100}")
    print(f"{'mode':<28s}  {'comp':>8s}  {'current':>8s}  "
          f"{'forget':>8s}  {'reinforce':>8s}  {'cross_dom':>8s}  {'Δ_cd':>8s}")
    print(f"{'─' * 100}")

    baseline_cd = results[0]["category_scores"].get("cross_domain", 0)
    for r in results:
        cs = r["category_scores"]
        cd = cs.get("cross_domain", 0)
        delta = cd - baseline_cd
        sign = "+" if delta > 0 else ""
        print(
            f"  {r['mode']:<26s}  "
            f"{r['composite']:>8.4f}  "
            f"{cs.get('current_fact', 0):>8.4f}  "
            f"{cs.get('graceful_forgetting', 0):>8.4f}  "
            f"{cs.get('reinforced_recall', 0):>8.4f}  "
            f"{cd:>8.4f}  "
            f"{sign}{delta:>7.4f}"
        )
    print(f"{'=' * 100}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
