"""Grid sweep over dream_idle_threshold to validate the optimization landscape.

Runs the longitudinal evaluation protocol with dream_idle_threshold ∈ {0, 5, 10, 20, 50}
and default DreamParams. Uses GPT-4o as LLM judge for open-ended categories.

Each evaluation:
  - Creates a fresh HermesMemoryAgent
  - Processes 200 sessions with dream cycles during idle periods
  - Scores 100 questions (30 SubEM + 70 LLM-judged)
  - Returns composite score

Usage:
    .venv/bin/python run_grid_sweep.py
    .venv/bin/python run_grid_sweep.py --idle-values 0 5 10 20 50
    .venv/bin/python run_grid_sweep.py --output output/grid_sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "mabench"))

from dream_ops import DreamParams
from longitudinal_eval import (
    LongitudinalEvaluator,
    generate_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def make_agent_factory(
    dream_params: DreamParams | None = None,
    associative: bool = False,
):
    """Return a factory that creates a fresh HermesMemoryAgent."""
    from mabench.hermes_agent import HermesMemoryAgent

    def factory():
        return HermesMemoryAgent(
            model="openai/gpt-4o-mini",
            dream_interval=0,  # evaluator controls dream timing
            dream_params=dream_params,
            associative_retrieval=associative,
        )

    return factory


def run_grid_sweep(
    idle_values: list[float],
    seed: int = 42,
    output_path: str | None = None,
    associative: bool = False,
) -> list[dict]:
    """Run grid sweep and return results sorted by composite score."""
    logger.info(
        "Generating longitudinal dataset (seed=%d, associative=%s)...",
        seed, associative,
    )
    dataset = generate_dataset(seed=seed)
    logger.info(
        "Dataset: %d sessions, %d questions",
        len(dataset.sessions),
        len(dataset.questions),
    )

    results = []
    default_params = DreamParams()

    for i, idle_val in enumerate(idle_values):
        logger.info(
            "━━━ [%d/%d] dream_idle_threshold=%.1f ━━━",
            i + 1, len(idle_values), idle_val,
        )

        evaluator = LongitudinalEvaluator(
            dataset=dataset,
            dream_idle_threshold=idle_val,
            use_llm_judge=True,
        )
        agent_factory = make_agent_factory(
            dream_params=default_params, associative=associative,
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
            "elapsed_seconds": round(elapsed, 1),
        }
        results.append(entry)

        logger.info(
            "  composite=%.4f  dreams=%d  method=%s  (%.1fs)",
            entry["composite"],
            entry["n_dreams"],
            entry["scoring_method"],
            elapsed,
        )
        for cat, score in result["category_scores"].items():
            logger.info("    %-25s %.4f", cat, score)

    # Sort by composite descending
    results.sort(key=lambda r: r["composite"], reverse=True)

    # Print summary
    print(f"\n{'=' * 70}")
    print("GRID SWEEP RESULTS (sorted by composite)")
    print(f"{'=' * 70}")
    print(f"{'idle_thresh':>12s}  {'composite':>10s}  {'dreams':>6s}  "
          f"{'current':>8s}  {'forget':>8s}  {'reinforce':>8s}  {'cross_dom':>8s}")
    print(f"{'─' * 70}")
    for r in results:
        cs = r["category_scores"]
        print(
            f"{r['dream_idle_threshold']:>12.1f}  "
            f"{r['composite']:>10.4f}  "
            f"{r['n_dreams']:>6d}  "
            f"{cs.get('current_fact', 0):>8.4f}  "
            f"{cs.get('graceful_forgetting', 0):>8.4f}  "
            f"{cs.get('reinforced_recall', 0):>8.4f}  "
            f"{cs.get('cross_domain', 0):>8.4f}"
        )
    print(f"{'=' * 70}")

    best = results[0]
    print(f"\nBest: dream_idle_threshold={best['dream_idle_threshold']:.1f} "
          f"→ composite={best['composite']:.4f}")

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", output_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid sweep over dream_idle_threshold"
    )
    parser.add_argument(
        "--idle-values",
        nargs="+",
        type=float,
        default=[0, 5, 10, 20, 50],
        help="dream_idle_threshold values to sweep (default: 0 5 10 20 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for dataset generation (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/grid_sweep_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--associative",
        action="store_true",
        help="Enable spreading activation for retrieval (Hopfield dynamics)",
    )
    args = parser.parse_args()

    run_grid_sweep(
        idle_values=args.idle_values,
        seed=args.seed,
        output_path=args.output,
        associative=args.associative,
    )
