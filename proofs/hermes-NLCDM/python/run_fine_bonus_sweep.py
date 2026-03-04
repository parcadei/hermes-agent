"""Fine bonus sweep: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08].

Determines whether bonus=0.05 is a stable plateau or a lucky spike.
If there's a stable region (e.g. 0.02-0.07 all help), co-retrieval is real.
If only 0.05 helps and neighbors regress, it's noise — kill the mechanism.

Usage:
    .venv/bin/python run_fine_bonus_sweep.py
    .venv/bin/python run_fine_bonus_sweep.py --seed 42 --output output/fine_bonus_sweep.json
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

BONUS_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

configs: list[dict] = [
    # Baseline: no co-retrieval, no probes
    {
        "name": "cosine_baseline",
        "coretrieval": False,
        "cross_domain_probes": False,
    },
]

# Add one config per bonus value
for b in BONUS_VALUES:
    configs.append({
        "name": f"probes_coretrieval_b{b:.2f}".replace(".", ""),
        "coretrieval": True,
        "cross_domain_probes": True,
        "probe_frequency": 10,
        "probes_per_session": 3,
        "coretrieval_bonus": b,
        "coretrieval_min_count": 1,
    })


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
        description="Fine bonus sweep [0.01-0.08] on multi-domain longitudinal eval"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="output/fine_bonus_sweep.json",
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
        coretrieval_bonus = cfg.get("coretrieval_bonus", 0.0)
        coretrieval_min_count = cfg.get("coretrieval_min_count", 2.0)

        logger.info("━━━ %s ━━━", name.upper())
        logger.info(
            "  coretrieval=%s  probes=%s  bonus=%.3f  min_count=%.0f",
            use_coretrieval, use_probes, coretrieval_bonus, coretrieval_min_count,
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

    print(f"\n{'=' * 100}")
    print("FINE BONUS SWEEP [0.01 - 0.08] — MULTI-DOMAIN LONGITUDINAL EVAL")
    print(f"{'=' * 100}")
    print(
        f"  {'mode':<30s}  {'bonus':>6s}  {'comp':>8s}  "
        f"{'cross_dom':>8s}  {'d_comp':>8s}  {'d_cd':>8s}"
    )
    print(f"  {'─' * 90}")

    baseline_comp = results[0]["composite"]
    baseline_cd = results[0]["category_scores"].get("cross_domain", 0)

    for r in results:
        cs = r["category_scores"]
        cd = cs.get("cross_domain", 0)
        delta_comp = r["composite"] - baseline_comp
        delta_cd = cd - baseline_cd
        sign_comp = "+" if delta_comp > 0 else ""
        sign_cd = "+" if delta_cd > 0 else ""
        bonus_str = f"{r['coretrieval_bonus']:.3f}" if r["coretrieval"] else "—"
        print(
            f"  {r['mode']:<30s}  {bonus_str:>6s}  "
            f"{r['composite']:>8.4f}  "
            f"{cd:>8.4f}  "
            f"{sign_comp}{delta_comp:>7.4f}  "
            f"{sign_cd}{delta_cd:>7.4f}"
        )

    print(f"{'=' * 100}")

    # Verdict
    print("\nVERDICT:")
    bonus_results = [(r["coretrieval_bonus"], r["composite"],
                      r["category_scores"].get("cross_domain", 0))
                     for r in results if r["coretrieval"]]

    helping = [b for b, comp, cd in bonus_results
               if comp >= baseline_comp and cd > baseline_cd]
    hurting = [b for b, comp, cd in bonus_results
               if comp < baseline_comp - 0.02 or cd < baseline_cd - 0.02]

    if len(helping) >= 3:
        lo, hi = min(helping), max(helping)
        print(f"  PLATEAU: {len(helping)} bonus values help [{lo:.2f}-{hi:.2f}]")
        print(f"  → Co-retrieval is a real mechanism. Stable operating range exists.")
    elif len(helping) == 1:
        print(f"  SPIKE: Only bonus={helping[0]:.2f} helps. Likely noise.")
        print(f"  → Kill co-retrieval. Not robust enough to ship.")
    elif len(helping) == 0:
        print(f"  NO HELP: No bonus value improves over baseline.")
        print(f"  → Kill co-retrieval entirely.")
    else:
        print(f"  NARROW: {len(helping)} values help: {[f'{b:.2f}' for b in helping]}")
        print(f"  → Marginal mechanism. Decide based on implementation cost.")

    if hurting:
        print(f"  CAUTION: {len(hurting)} values hurt: {[f'{b:.2f}' for b in hurting]}")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
