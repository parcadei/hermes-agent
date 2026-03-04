#!/usr/bin/env python3
"""Capacity-gated dream evaluation on multi-domain longitudinal dataset.

The critical experiment: does dream consolidation add value at 64 max_gen_tokens
when capacity gating prevents over-pruning?

Prior results:
  - 16 tokens, no gating: dream +8% (0.4675 → 0.5050)
  - 64 tokens, no dream:  baseline 0.6125
  - Question: does dream + capacity gating > 0.6125?

Runs 3 conditions:
  1. no_dream:          dream_idle_threshold=inf (no dreams fire)
  2. dream_ungated:     dream_idle_threshold=0 (dream every session gap, no gating)
  3. dream_gated:       dream_idle_threshold=0 + capacity_gating=True

All use max_gen_tokens=64 (HermesMemoryAgent default).
"""

import argparse
import json
import logging
import time
from pathlib import Path

from dream_ops import DreamParams
from longitudinal_eval import LongitudinalEvaluator
from multidomain_eval import generate_multidomain_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def make_agent_factory():
    """Return a zero-arg callable that creates a fresh HermesMemoryAgent."""
    from mabench.hermes_agent import HermesMemoryAgent

    def factory():
        return HermesMemoryAgent(
            model="openai/gpt-4o-mini",
            dream_interval=0,  # dreams controlled by evaluator, not store count
            dream_params=DreamParams(),
            max_gen_tokens=64,
        )

    return factory


CONFIGS = [
    {
        "name": "no_dream",
        "dream_idle_threshold": 1e9,  # effectively infinite — no dreams
        "capacity_gating": False,
    },
    {
        "name": "dream_ungated",
        "dream_idle_threshold": 0.0,  # dream every session gap
        "capacity_gating": False,
    },
    {
        "name": "dream_gated",
        "dream_idle_threshold": 0.0,  # dream every session gap, but gated
        "capacity_gating": True,
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Capacity-gated dream eval on multi-domain longitudinal dataset"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="output/capacity_gating_eval.json",
    )
    args = parser.parse_args()

    logger.info("Generating multi-domain dataset (seed=%d)...", args.seed)
    dataset = generate_multidomain_dataset(seed=args.seed)
    logger.info(
        "Dataset: %d sessions, %d questions",
        len(dataset.sessions), len(dataset.questions),
    )

    results = []
    agent_factory = make_agent_factory()

    for cfg in CONFIGS:
        name = cfg["name"]
        logger.info("━━━ %s ━━━", name.upper())

        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            dream_idle_threshold=cfg["dream_idle_threshold"],
            use_llm_judge=True,
            capacity_gating=cfg["capacity_gating"],
        )

        t0 = time.time()
        result = evaluator.evaluate(agent_factory)
        elapsed = time.time() - t0

        entry = {
            "mode": name,
            "capacity_gating": cfg["capacity_gating"],
            "dream_idle_threshold": cfg["dream_idle_threshold"],
            "composite": result["composite"],
            "category_scores": result["category_scores"],
            "n_dreams": result["n_dreams"],
            "sessions_processed": result["sessions_processed"],
            "scoring_method": result["scoring_method"],
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)

        logger.info("  composite=%.4f  n_dreams=%d  (%.1fs)",
                     entry["composite"], entry["n_dreams"], elapsed)
        for cat, score in result["category_scores"].items():
            logger.info("    %-25s %.4f", cat, score)

    # Summary table
    print(f"\n{'=' * 100}")
    print("CAPACITY-GATED DREAM EVAL — 64 tokens, multi-domain longitudinal")
    print(f"{'=' * 100}")
    print(
        f"  {'mode':<20s}  {'gated':>6s}  {'dreams':>6s}  {'comp':>8s}  "
        f"{'current':>8s}  {'forget':>8s}  {'reinf':>8s}  {'cross':>8s}  {'d_comp':>8s}"
    )
    print(f"  {'─' * 95}")

    baseline_comp = results[0]["composite"]

    for r in results:
        cs = r["category_scores"]
        delta = r["composite"] - baseline_comp
        sign = "+" if delta > 0 else ""
        gated_str = "YES" if r["capacity_gating"] else "no"
        print(
            f"  {r['mode']:<20s}  {gated_str:>6s}  {r['n_dreams']:>6d}  "
            f"{r['composite']:>8.4f}  "
            f"{cs.get('current_fact', 0):>8.4f}  "
            f"{cs.get('graceful_forgetting', 0):>8.4f}  "
            f"{cs.get('reinforced_recall', 0):>8.4f}  "
            f"{cs.get('cross_domain', 0):>8.4f}  "
            f"{sign}{delta:>7.4f}"
        )

    print(f"{'=' * 100}")

    # Verdict
    print("\nVERDICT:")
    no_dream = results[0]["composite"]
    ungated = results[1]["composite"]
    gated = results[2]["composite"]

    print(f"  no_dream:      {no_dream:.4f} (baseline)")
    print(f"  dream_ungated: {ungated:.4f} ({'+' if ungated > no_dream else ''}{ungated - no_dream:.4f})")
    print(f"  dream_gated:   {gated:.4f} ({'+' if gated > no_dream else ''}{gated - no_dream:.4f})")

    if gated > no_dream + 0.01:
        print("  → Capacity-gated dreams ADD VALUE at 64 tokens.")
    elif gated >= no_dream - 0.01:
        print("  → Capacity gating makes dreams NEUTRAL (no harm, no help).")
    else:
        print("  → Even with gating, dreams HURT at 64 tokens.")

    if gated > ungated + 0.01:
        print("  → Gating IMPROVES over ungated dreams.")
    elif ungated > gated + 0.01:
        print("  → Gating makes dreams WORSE (prevents useful consolidation).")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
