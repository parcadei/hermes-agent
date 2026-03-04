"""Dream-driven co-retrieval sweep on multi-domain longitudinal eval.

Tests whether dream-generated co-retrieval edges improve cross-domain
retrieval compared to probe-generated edges and cosine baseline.

Dream cycle runs between session clusters (dream_idle_threshold=1.0).
After the dream-to-co_retrieval wiring, REM-explore associations are
logged as weighted edges, giving query_coretrieval() genuine cross-domain
expansion paths.

Sweep configurations:
  1. cosine_baseline              - No dreams, no co-retrieval
  2. dreams_only                  - Dreams fire, no co-retrieval reranking
  3. dreams_coretrieval_b004      - Dreams + co-retrieval bonus=0.04
  4. dreams_coretrieval_b003      - Dreams + co-retrieval bonus=0.03
  5. dreams_coretrieval_b005      - Dreams + co-retrieval bonus=0.05
  6. probes_coretrieval_b004      - Probes only (no dreams) + bonus=0.04

Usage:
    .venv/bin/python run_dream_coretrieval_sweep.py
    .venv/bin/python run_dream_coretrieval_sweep.py --seed 42 --output output/dream_coretrieval_sweep.json
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
    coretrieval_bonus: float = 0.04,
    coretrieval_min_count: float = 1.0,
):
    from mabench.hermes_agent import HermesMemoryAgent

    def factory():
        return HermesMemoryAgent(
            model="openai/gpt-4o-mini",
            dream_interval=0,  # evaluator controls dream timing
            dream_params=DreamParams(),
            coretrieval_retrieval=coretrieval,
            coretrieval_bonus=coretrieval_bonus,
            coretrieval_min_count=coretrieval_min_count,
        )

    return factory


configs = [
    {
        "name": "cosine_baseline",
        "dream_idle_threshold": 0.0,
        "retrieval_practice": False,
        "coretrieval": False,
        "bonus": 0.0,
    },
    {
        "name": "dreams_only",
        "dream_idle_threshold": 1.0,
        "retrieval_practice": False,
        "coretrieval": False,
        "bonus": 0.0,
    },
    {
        "name": "dreams_coretrieval_b004",
        "dream_idle_threshold": 1.0,
        "retrieval_practice": False,
        "coretrieval": True,
        "bonus": 0.04,
    },
    {
        "name": "dreams_coretrieval_b003",
        "dream_idle_threshold": 1.0,
        "retrieval_practice": False,
        "coretrieval": True,
        "bonus": 0.03,
    },
    {
        "name": "dreams_coretrieval_b005",
        "dream_idle_threshold": 1.0,
        "retrieval_practice": False,
        "coretrieval": True,
        "bonus": 0.05,
    },
    {
        "name": "probes_coretrieval_b004",
        "dream_idle_threshold": 0.0,
        "retrieval_practice": True,
        "coretrieval": True,
        "bonus": 0.04,
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Dream co-retrieval sweep on multi-domain longitudinal eval"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/dream_coretrieval_sweep.json")
    args = parser.parse_args()

    logger.info("Generating multi-domain dataset (seed=%d)...", args.seed)
    dataset = generate_multidomain_dataset(seed=args.seed)
    logger.info(
        "Dataset: %d sessions, %d questions",
        len(dataset.sessions), len(dataset.questions),
    )

    results = []

    for cfg in configs:
        logger.info("━━━ %s ━━━", cfg["name"])
        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            dream_idle_threshold=cfg["dream_idle_threshold"],
            use_llm_judge=True,
            retrieval_practice=cfg["retrieval_practice"],
        )
        agent_factory = make_agent_factory(
            coretrieval=cfg["coretrieval"],
            coretrieval_bonus=cfg["bonus"],
            coretrieval_min_count=1.0,
        )
        t0 = time.time()
        result = evaluator.evaluate(agent_factory)
        elapsed = time.time() - t0

        entry = {
            "name": cfg["name"],
            "config": cfg,
            "composite": result["composite"],
            "category_scores": result["category_scores"],
            "n_dreams": result["n_dreams"],
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)
        logger.info("  composite=%.4f  n_dreams=%d  (%.1fs)",
                     entry["composite"], entry["n_dreams"], elapsed)
        for cat, score in result["category_scores"].items():
            logger.info("    %-25s %.4f", cat, score)

    # Print summary
    print(f"\n{'=' * 110}")
    print("DREAM CO-RETRIEVAL SWEEP — MULTI-DOMAIN LONGITUDINAL EVAL")
    print(f"{'=' * 110}")
    print(f"{'config':<30s}  {'comp':>8s}  {'current':>8s}  "
          f"{'forget':>8s}  {'reinforce':>8s}  {'cross_dom':>8s}  "
          f"{'Δ_cd':>8s}  {'dreams':>6s}")
    print(f"{'─' * 110}")

    baseline_cd = results[0]["category_scores"].get("cross_domain", 0)
    for r in results:
        cs = r["category_scores"]
        cd = cs.get("cross_domain", 0)
        delta = cd - baseline_cd
        sign = "+" if delta > 0 else ""
        print(
            f"  {r['name']:<28s}  "
            f"{r['composite']:>8.4f}  "
            f"{cs.get('current_fact', 0):>8.4f}  "
            f"{cs.get('graceful_forgetting', 0):>8.4f}  "
            f"{cs.get('reinforced_recall', 0):>8.4f}  "
            f"{cd:>8.4f}  "
            f"{sign}{delta:>7.4f}  "
            f"{r['n_dreams']:>6d}"
        )
    print(f"{'=' * 110}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
