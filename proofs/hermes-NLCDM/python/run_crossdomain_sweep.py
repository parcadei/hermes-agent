"""Cross-domain probe sweep on multi-domain longitudinal eval.

Tests whether co-retrieval improves when given cross-domain edges injected
via query_readonly() probes. The probes build co-retrieval graph structure
without corrupting scoring signals (access_count, importance).

Sweep configurations:
  1. cosine_baseline           - No probes, no co-retrieval
  2. probes_only_no_coretrieval - Probes build edges, but retrieval ignores them (control)
  3. probes_coretrieval_b005   - Probes + co-retrieval bonus=0.05
  4. probes_coretrieval_b010   - Probes + co-retrieval bonus=0.10
  5. probes_coretrieval_b015   - Probes + co-retrieval bonus=0.15
  6. probes_freq5_b010         - Higher probe frequency (every 5 sessions) at bonus=0.10

Usage:
    .venv/bin/python run_crossdomain_sweep.py
    .venv/bin/python run_crossdomain_sweep.py --seed 42 --output output/crossdomain_sweep.json
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


# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

configs: list[dict] = [
    # Baseline: no co-retrieval, no probes
    {
        "name": "cosine_baseline",
        "coretrieval": False,
        "cross_domain_probes": False,
    },
    # Probes build edges, but eval queries don't use co-retrieval (control)
    {
        "name": "probes_only_no_coretrieval",
        "coretrieval": False,
        "cross_domain_probes": True,
        "probe_frequency": 10,
        "probes_per_session": 3,
    },
    # Probes build edges, eval uses co-retrieval at bonus=0.05
    {
        "name": "probes_coretrieval_b005",
        "coretrieval": True,
        "cross_domain_probes": True,
        "probe_frequency": 10,
        "probes_per_session": 3,
        "coretrieval_bonus": 0.05,
        "coretrieval_min_count": 1,
    },
    # Probes build edges, eval uses co-retrieval at bonus=0.10
    {
        "name": "probes_coretrieval_b010",
        "coretrieval": True,
        "cross_domain_probes": True,
        "probe_frequency": 10,
        "probes_per_session": 3,
        "coretrieval_bonus": 0.10,
        "coretrieval_min_count": 1,
    },
    # Probes build edges, eval uses co-retrieval at bonus=0.15
    {
        "name": "probes_coretrieval_b015",
        "coretrieval": True,
        "cross_domain_probes": True,
        "probe_frequency": 10,
        "probes_per_session": 3,
        "coretrieval_bonus": 0.15,
        "coretrieval_min_count": 1,
    },
    # Higher probe frequency (every 5 sessions) at best bonus
    {
        "name": "probes_freq5_b010",
        "coretrieval": True,
        "cross_domain_probes": True,
        "probe_frequency": 5,
        "probes_per_session": 3,
        "coretrieval_bonus": 0.10,
        "coretrieval_min_count": 1,
    },
]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def make_agent_factory(
    coretrieval: bool = False,
    coretrieval_bonus: float = 0.3,
    coretrieval_min_count: float = 2.0,
):
    """Return a zero-arg callable that creates a fresh HermesMemoryAgent."""
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


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-domain probe sweep on multi-domain longitudinal eval"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="output/crossdomain_sweep.json",
    )
    args = parser.parse_args()

    logger.info("Generating multi-domain dataset (seed=%d)...", args.seed)
    dataset = generate_multidomain_dataset(seed=args.seed)
    logger.info(
        "Dataset: %d sessions, %d questions",
        len(dataset.sessions), len(dataset.questions),
    )

    results = []

    for cfg in configs:
        name = cfg["name"]
        use_coretrieval = cfg["coretrieval"]
        use_probes = cfg["cross_domain_probes"]
        probe_frequency = cfg.get("probe_frequency", 10)
        probes_per_session = cfg.get("probes_per_session", 3)
        coretrieval_bonus = cfg.get("coretrieval_bonus", 0.3)
        coretrieval_min_count = cfg.get("coretrieval_min_count", 2.0)

        logger.info("━━━ %s ━━━", name.upper())
        logger.info(
            "  coretrieval=%s  probes=%s  freq=%d  per_sess=%d  bonus=%.2f  min_count=%.0f",
            use_coretrieval, use_probes, probe_frequency, probes_per_session,
            coretrieval_bonus, coretrieval_min_count,
        )

        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            dream_idle_threshold=0.0,
            use_llm_judge=True,
            retrieval_practice=True,
            cross_domain_probes=use_probes,
            probe_frequency=probe_frequency,
            probes_per_session=probes_per_session,
        )

        agent_factory = make_agent_factory(
            coretrieval=use_coretrieval,
            coretrieval_bonus=coretrieval_bonus,
            coretrieval_min_count=coretrieval_min_count,
        )

        t0 = time.time()
        result = evaluator.evaluate(agent_factory)
        elapsed = time.time() - t0

        entry = {
            "mode": name,
            "coretrieval": use_coretrieval,
            "cross_domain_probes": use_probes,
            "probe_frequency": probe_frequency,
            "probes_per_session": probes_per_session,
            "coretrieval_bonus": coretrieval_bonus,
            "coretrieval_min_count": coretrieval_min_count,
            "composite": result["composite"],
            "category_scores": result["category_scores"],
            "n_dreams": result["n_dreams"],
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)

        logger.info("  composite=%.4f  (%.1fs)", entry["composite"], elapsed)
        for cat, score in result["category_scores"].items():
            logger.info("    %-25s %.4f", cat, score)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------

    print(f"\n{'=' * 110}")
    print("CROSS-DOMAIN PROBE SWEEP — MULTI-DOMAIN LONGITUDINAL EVAL")
    print(f"{'=' * 110}")
    print(
        f"{'mode':<32s}  {'comp':>8s}  {'current':>8s}  "
        f"{'forget':>8s}  {'reinforce':>8s}  {'cross_dom':>8s}  "
        f"{'d_comp':>8s}  {'d_cd':>8s}"
    )
    print(f"{'─' * 110}")

    baseline_comp = results[0]["composite"]
    baseline_cd = results[0]["category_scores"].get("cross_domain", 0)

    for r in results:
        cs = r["category_scores"]
        cd = cs.get("cross_domain", 0)
        delta_comp = r["composite"] - baseline_comp
        delta_cd = cd - baseline_cd
        sign_comp = "+" if delta_comp > 0 else ""
        sign_cd = "+" if delta_cd > 0 else ""
        print(
            f"  {r['mode']:<30s}  "
            f"{r['composite']:>8.4f}  "
            f"{cs.get('current_fact', 0):>8.4f}  "
            f"{cs.get('graceful_forgetting', 0):>8.4f}  "
            f"{cs.get('reinforced_recall', 0):>8.4f}  "
            f"{cd:>8.4f}  "
            f"{sign_comp}{delta_comp:>7.4f}  "
            f"{sign_cd}{delta_cd:>7.4f}"
        )
    print(f"{'=' * 110}")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
