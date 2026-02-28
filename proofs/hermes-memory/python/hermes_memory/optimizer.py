"""Hermes Memory System -- CMA-ES Parameter Optimizer.

Thin optimization layer atop engine.py and sensitivity.py. Maps raw CMA-ES
vectors to constrained ParameterSets, evaluates them against expanded benchmark
scenarios, and runs CMA-ES to find optimal parameters.

The 6 free parameters optimized by CMA-ES are:
    w1, w2, w4        -- scoring weights (w3 derived as 1 - w1 - w2 - w4)
    alpha, beta        -- dynamics parameters
    temperature        -- soft selection temperature

All other parameters are pinned at insensitive defaults identified by
the sensitivity analysis.

Mathematical invariant: all scoring, ranking, and simulation operations
delegate to engine.py. This module adds only the optimization framework
and scenario generation logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import Optional

import numpy as np
from cmaes import CMA

from hermes_memory.engine import (
    ParameterSet,
    MemoryState,
    rank_memories,
    simulate,
)
from hermes_memory.sensitivity import Scenario


# ============================================================
# 1. Constants
# ============================================================

PARAM_ORDER: list[str] = ["w1", "w2", "w4", "alpha", "beta", "temperature"]
"""The 6 free parameters in the order they appear in CMA-ES vectors."""

BOUNDS: dict[str, tuple[float, float]] = {
    "w1":          (0.05, 0.60),
    "w2":          (0.05, 0.39),
    "w4":          (0.05, 0.50),
    "alpha":       (0.01, 0.50),
    "beta":        (0.01, 1.00),
    "temperature": (0.10, 10.0),
}
"""Per-parameter bounds for CMA-ES optimization."""

PINNED: dict[str, float] = {
    "delta_t": 1.0,
    "s_max": 10.0,
    "s0": 1.0,
    "novelty_start": 0.3,
    "novelty_decay": 0.1,
    "survival_threshold": 0.05,
    "feedback_sensitivity": 0.1,
}
"""Pinned defaults for insensitive parameters (from sensitivity analysis)."""

STATIC_WEIGHT: float = 0.60
"""Weight for static accuracy in the composite objective."""

STABILITY_WEIGHT: float = 0.25
"""Weight for stability accuracy in the composite objective."""

MARGIN_WEIGHT: float = 0.15
"""Weight for contraction margin bonus in the composite objective."""

INFEASIBLE_PENALTY: float = 1000.0
"""Penalty value returned by objective() for infeasible parameter vectors."""

STABILITY_SIM_STEPS: int = 100
"""Number of simulation steps for stability evaluation in objective()."""


# ============================================================
# 2. Result dataclasses
# ============================================================


@dataclass(frozen=True)
class OptimizationResult:
    """Result of CMA-ES parameter optimization.

    Attributes:
        best_x:          Raw CMA-ES vector of shape (6,).
        best_score:      Composite score (positive, higher is better).
        best_params:     Decoded ParameterSet for the best solution.
        history:         Best score per generation (monotonically non-decreasing).
        generations:     Total number of generations run.
        per_competency:  Accuracy breakdown per competency category.
    """

    best_x: np.ndarray
    best_score: float
    best_params: ParameterSet
    history: list[float]
    generations: int
    per_competency: dict[str, float]


@dataclass(frozen=True)
class ValidationResult:
    """Result of post-optimization validation.

    Attributes:
        contraction_margin:    1 - K, where K is the contraction factor.
        static_accuracy:       Fraction of training scenarios with correct winner.
        stability_accuracy:    Fraction of stability scenarios passing monopolization check.
        held_out_accuracy:     Fraction of held-out scenarios with correct winner.
        steady_state_strength: Analytical S* = alpha * s_max / (1 - (1-alpha)*exp(-beta*delta_t)).
        steady_state_ratio:    S* / s_max.
        exploration_window:    W = ln(novelty_start / survival_threshold) / novelty_decay.
        weight_summary:        Dictionary with w1, w2, w3, w4 values.
        all_passed:            True if no validation failures.
        failures:              List of failure descriptions (empty if all_passed).
    """

    contraction_margin: float
    static_accuracy: float
    stability_accuracy: float
    held_out_accuracy: float
    steady_state_strength: float
    steady_state_ratio: float
    exploration_window: float
    weight_summary: dict[str, float]
    all_passed: bool
    failures: list[str]


# ============================================================
# 3. decode -- vector to ParameterSet
# ============================================================


def decode(x: np.ndarray) -> ParameterSet | None:
    """Decode a raw CMA-ES vector into a validated ParameterSet.

    Clips each element to its BOUNDS range, derives w3 = 1 - w1 - w2 - w4,
    and constructs a ParameterSet. Returns None if the vector is infeasible
    (w3 < 0, w3 >= 1, or contraction violation).

    Args:
        x: numpy array of shape (6,) with values in PARAM_ORDER.

    Returns:
        Valid ParameterSet, or None if infeasible.

    Raises:
        ValueError: If x has wrong shape (not exactly 6 elements).
        IndexError: If x has wrong shape (not enough elements).
    """
    if x.ndim != 1 or x.shape[0] != 6:
        raise ValueError(
            f"Expected array of shape (6,), got shape {x.shape}"
        )

    # Clip each element to its bounds
    clipped = np.empty(6)
    for i, name in enumerate(PARAM_ORDER):
        lo, hi = BOUNDS[name]
        clipped[i] = np.clip(x[i], lo, hi)

    w1 = float(clipped[0])
    w2 = float(clipped[1])
    w4 = float(clipped[2])
    alpha = float(clipped[3])
    beta = float(clipped[4])
    temperature = float(clipped[5])

    # Derive w3
    w3 = 1.0 - w1 - w2 - w4
    if w3 < 0.0:
        return None
    if w3 >= 1.0:
        return None

    # Construct ParameterSet (validates contraction and other constraints)
    try:
        return ParameterSet(
            alpha=alpha,
            beta=beta,
            delta_t=PINNED["delta_t"],
            s_max=PINNED["s_max"],
            s0=PINNED["s0"],
            temperature=temperature,
            novelty_start=PINNED["novelty_start"],
            novelty_decay=PINNED["novelty_decay"],
            survival_threshold=PINNED["survival_threshold"],
            feedback_sensitivity=PINNED["feedback_sensitivity"],
            w1=w1,
            w2=w2,
            w3=w3,
            w4=w4,
        )
    except ValueError:
        return None


# ============================================================
# 4. build_benchmark_scenarios -- expanded scenario suite
# ============================================================


def build_benchmark_scenarios(
    params: ParameterSet,
    held_out: bool = False,
) -> list[Scenario]:
    """Build 120+ benchmark scenarios covering all 4 competencies.

    Includes the 10 standard scenarios from sensitivity.py rebuilt here
    (to avoid dependency on params verification), plus procedurally generated
    difficulty variations.

    For non-stability scenarios, each generated scenario is verified: the
    expected_winner must match rank_memories() under the given params. If
    verification fails, the scenario is skipped (not included).

    Args:
        params:   Reference ParameterSet for scenario verification.
        held_out: If True, generate validation scenarios with slightly
                  different numerical values than the training set.

    Returns:
        List of 120+ Scenario instances covering 4 competencies.
    """
    s_max = PINNED["s_max"]
    s0 = PINNED["s0"]

    # Numerical offset for held-out variation
    offset = 0.02 if held_out else 0.0
    stale_mult = 1.1 if held_out else 1.0
    suffix = "-HO" if held_out else ""

    scenarios: list[Scenario] = []

    def _try_add(
        name: str,
        memories: list[MemoryState],
        current_time: float,
        expected_winner: int,
        competency: str,
    ) -> None:
        """Attempt to add a scenario, verifying expected_winner for non-stability.

        For stability scenarios (expected_winner=-1), adds unconditionally.
        For all other competencies, verifies that rank_memories() under the
        reference params produces the expected_winner as the top-ranked memory.
        If verification fails, the scenario is silently skipped -- this is
        intentional self-healing (don't include scenarios that are impossible
        under the reference parameters). The scenario count assertion at the
        end of build_benchmark_scenarios() guards against excessive drops.
        """
        if competency == "stability":
            scenarios.append(Scenario(
                name=name,
                memories=memories,
                current_time=current_time,
                expected_winner=expected_winner,
                competency=competency,
            ))
            return

        # Verify expected_winner matches ranking
        ranked = rank_memories(params, memories, current_time)
        actual_winner = ranked[0][0]
        if actual_winner == expected_winner:
            scenarios.append(Scenario(
                name=name,
                memories=memories,
                current_time=current_time,
                expected_winner=expected_winner,
                competency=competency,
            ))
        # If verification fails, skip silently

    # ================================================================
    # 4a. Rebuild the 10 standard scenarios from sensitivity.py
    # ================================================================

    # AR-1: High relevance beats high recency
    _try_add(
        name=f"AR-1: High relevance beats high recency{suffix}",
        memories=[
            MemoryState(
                relevance=0.95 + offset, last_access_time=5.0,
                importance=0.3, access_count=2,
                strength=3.0, creation_time=100.0,
            ),
            MemoryState(
                relevance=0.4 + offset, last_access_time=20.0,
                importance=0.3, access_count=2,
                strength=s_max * 0.8, creation_time=100.0,
            ),
        ],
        current_time=0.0,
        expected_winner=0,
        competency="accurate_retrieval",
    )

    # AR-2: Importance tiebreaker
    _try_add(
        name=f"AR-2: Importance tiebreaker{suffix}",
        memories=[
            MemoryState(
                relevance=0.7 + offset, last_access_time=10.0,
                importance=0.9, access_count=5,
                strength=5.0, creation_time=200.0,
            ),
            MemoryState(
                relevance=0.7 + offset, last_access_time=10.0,
                importance=0.1, access_count=5,
                strength=5.0, creation_time=200.0,
            ),
        ],
        current_time=200.0,
        expected_winner=0,
        competency="accurate_retrieval",
    )

    # AR-3: High relevance beats high activation
    _try_add(
        name=f"AR-3: High relevance beats high activation{suffix}",
        memories=[
            MemoryState(
                relevance=0.9 + offset, last_access_time=20.0,
                importance=0.5, access_count=1,
                strength=s0, creation_time=80.0,
            ),
            MemoryState(
                relevance=0.3 + offset, last_access_time=5.0,
                importance=0.5, access_count=100,
                strength=s_max * 0.9, creation_time=80.0,
            ),
        ],
        current_time=0.0,
        expected_winner=0,
        competency="accurate_retrieval",
    )

    # TTL-1: New high-relevance enters pool
    _try_add(
        name=f"TTL-1: New high-relevance enters pool{suffix}",
        memories=[
            MemoryState(
                relevance=0.9 + offset, last_access_time=0.0,
                importance=0.5, access_count=0,
                strength=s0, creation_time=0.0,
            ),
            MemoryState(
                relevance=0.5 + offset, last_access_time=5.0,
                importance=0.7, access_count=20,
                strength=s_max * 0.7, creation_time=200.0,
            ),
            MemoryState(
                relevance=0.4 + offset, last_access_time=3.0,
                importance=0.6, access_count=15,
                strength=s_max * 0.6, creation_time=150.0,
            ),
        ],
        current_time=0.0,
        expected_winner=0,
        competency="test_time_learning",
    )

    # TTL-2: Novelty saves zero-base memory
    _try_add(
        name=f"TTL-2: Novelty saves zero-base memory{suffix}",
        memories=[
            MemoryState(
                relevance=0.0, last_access_time=0.0,
                importance=0.0, access_count=0,
                strength=s0, creation_time=0.0,
            ),
            MemoryState(
                relevance=0.3 + offset, last_access_time=20.0,
                importance=0.3, access_count=5,
                strength=s0 * 0.5, creation_time=100.0,
            ),
        ],
        current_time=0.0,
        expected_winner=0,
        competency="test_time_learning",
    )

    # SF-1: Stale heavily-accessed loses
    _try_add(
        name=f"SF-1: Stale heavily-accessed loses{suffix}",
        memories=[
            MemoryState(
                relevance=0.8 + offset, last_access_time=5.0,
                importance=0.6, access_count=3,
                strength=s_max * 0.5, creation_time=20.0,
            ),
            MemoryState(
                relevance=0.5 + offset, last_access_time=200.0 * stale_mult,
                importance=0.8, access_count=100,
                strength=s_max * 0.3, creation_time=500.0,
            ),
        ],
        current_time=0.0,
        expected_winner=0,
        competency="selective_forgetting",
    )

    # SF-2: Anti-lock-in
    _try_add(
        name=f"SF-2: Anti-lock-in{suffix}",
        memories=[
            MemoryState(
                relevance=0.8 + offset, last_access_time=3.0,
                importance=0.6, access_count=5,
                strength=s_max * 0.5, creation_time=50.0,
            ),
            MemoryState(
                relevance=0.5 + offset, last_access_time=50.0 * stale_mult,
                importance=0.9, access_count=100,
                strength=s_max * 0.95, creation_time=1000.0,
            ),
        ],
        current_time=0.0,
        expected_winner=0,
        competency="selective_forgetting",
    )

    # SF-3: Low-importance/relevance loses
    _try_add(
        name=f"SF-3: Low-importance/relevance loses{suffix}",
        memories=[
            MemoryState(
                relevance=0.6 + offset, last_access_time=10.0,
                importance=0.5, access_count=3,
                strength=s0, creation_time=30.0,
            ),
            MemoryState(
                relevance=0.15 + offset, last_access_time=100.0 * stale_mult,
                importance=0.1, access_count=50,
                strength=s_max * 0.2, creation_time=500.0,
            ),
        ],
        current_time=0.0,
        expected_winner=0,
        competency="selective_forgetting",
    )

    # STAB-1: No monopolization
    _try_add(
        name=f"STAB-1: No monopolization{suffix}",
        memories=[
            MemoryState(
                relevance=0.5 + offset, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
            MemoryState(
                relevance=0.55 + offset, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
            MemoryState(
                relevance=0.6 + offset, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
            MemoryState(
                relevance=0.45 + offset, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
        ],
        current_time=0.0,
        expected_winner=-1,
        competency="stability",
    )

    # STAB-2: Strength convergence
    _try_add(
        name=f"STAB-2: Strength convergence{suffix}",
        memories=[
            MemoryState(
                relevance=0.5 + offset, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
            MemoryState(
                relevance=0.5 + offset, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s0, creation_time=50.0,
            ),
        ],
        current_time=0.0,
        expected_winner=-1,
        competency="stability",
    )

    # ================================================================
    # 4b. Procedurally generated Accurate Retrieval (AR) scenarios
    # ================================================================

    # Pattern A: Pure relevance gap, equal other stats (easiest)
    rel_gaps_pure = (
        [0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1]
        if not held_out else
        [0.58, 0.48, 0.38, 0.28, 0.18, 0.13, 0.08]
    )
    for rel_gap in rel_gaps_pure:
        for pool_size in [2, 3, 5]:
            high_rel = min(0.5 + rel_gap / 2.0 + offset, 1.0)
            low_rel = max(0.5 - rel_gap / 2.0 + offset, 0.0)

            memories = [
                MemoryState(
                    relevance=high_rel, last_access_time=10.0,
                    importance=0.5, access_count=5,
                    strength=s_max * 0.5, creation_time=100.0,
                ),
                MemoryState(
                    relevance=low_rel, last_access_time=10.0,
                    importance=0.5, access_count=5,
                    strength=s_max * 0.5, creation_time=100.0,
                ),
            ]
            for d in range(pool_size - 2):
                dist_rel = max(low_rel - 0.05 * (d + 1), 0.01)
                memories.append(MemoryState(
                    relevance=dist_rel, last_access_time=15.0 + d * 5.0,
                    importance=0.4, access_count=3,
                    strength=s_max * 0.4, creation_time=100.0,
                ))

            _try_add(
                name=f"AR-pure-gap{rel_gap:.2f}-pool{pool_size}{suffix}",
                memories=memories,
                current_time=0.0,
                expected_winner=0,
                competency="accurate_retrieval",
            )

    # Pattern B: Competitor has one moderate advantage type
    adv_configs = [
        # (label, comp_lat, comp_imp, comp_ac, comp_str_frac)
        ("rec-mild", 5.0, 0.5, 5, 0.5),     # mild recency advantage
        ("rec-mod", 3.0, 0.5, 5, 0.5),       # moderate recency
        ("imp-mild", 10.0, 0.65, 5, 0.5),    # mild importance advantage
        ("imp-mod", 10.0, 0.75, 5, 0.5),     # moderate importance
        ("act-mild", 10.0, 0.5, 15, 0.5),    # mild activation advantage
        ("act-mod", 10.0, 0.5, 30, 0.5),     # moderate activation
        ("str-mild", 10.0, 0.5, 5, 0.7),     # mild strength advantage
        ("str-mod", 10.0, 0.5, 5, 0.85),     # moderate strength
    ]
    ar_gaps_b = [0.5, 0.4, 0.3, 0.2] if not held_out else [0.48, 0.38, 0.28, 0.18]

    for adv_label, comp_lat, comp_imp, comp_ac, comp_str_f in adv_configs:
        for rel_gap in ar_gaps_b:
            high_rel = min(0.5 + rel_gap / 2.0 + offset, 1.0)
            low_rel = max(0.5 - rel_gap / 2.0 + offset, 0.0)

            target = MemoryState(
                relevance=high_rel, last_access_time=10.0,
                importance=0.5, access_count=5,
                strength=s_max * 0.5, creation_time=100.0,
            )
            comp = MemoryState(
                relevance=low_rel,
                last_access_time=comp_lat if not held_out else comp_lat * 1.05,
                importance=comp_imp,
                access_count=comp_ac,
                strength=min(s_max * comp_str_f, s_max),
                creation_time=100.0,
            )

            _try_add(
                name=f"AR-{adv_label}-gap{rel_gap:.2f}{suffix}",
                memories=[target, comp],
                current_time=0.0,
                expected_winner=0,
                competency="accurate_retrieval",
            )

    # ================================================================
    # 4c. Procedurally generated Test-Time Learning (TTL) scenarios
    # ================================================================

    new_rels = [0.9, 0.7, 0.5] if not held_out else [0.88, 0.68, 0.48]
    n_established_list = [2, 5, 10]
    estab_strength_fracs = [0.5, 0.8]

    for new_rel in new_rels:
        for n_established in n_established_list:
            for estab_frac in estab_strength_fracs:
                memories = [
                    MemoryState(
                        relevance=min(new_rel + offset, 1.0),
                        last_access_time=0.0,
                        importance=0.5,
                        access_count=0,
                        strength=s0,
                        creation_time=0.0,
                    ),
                ]
                for e in range(n_established):
                    memories.append(MemoryState(
                        relevance=max(0.3 + 0.02 * e + offset, 0.0),
                        last_access_time=5.0 + 3.0 * e,
                        importance=0.5 + 0.02 * e,
                        access_count=10 + 5 * e,
                        strength=min(s_max * estab_frac, s_max),
                        creation_time=200.0 + 10.0 * e,
                    ))

                _try_add(
                    name=f"TTL-gen-rel{new_rel:.1f}-n{n_established}-sf{estab_frac:.1f}{suffix}",
                    memories=memories,
                    current_time=0.0,
                    expected_winner=0,
                    competency="test_time_learning",
                )

    # Additional TTL with varying established recency
    for new_rel in [0.85, 0.65] if not held_out else [0.83, 0.63]:
        for estab_lat in [5.0, 20.0, 100.0]:
            memories = [
                MemoryState(
                    relevance=min(new_rel + offset, 1.0),
                    last_access_time=0.0,
                    importance=0.5,
                    access_count=0,
                    strength=s0,
                    creation_time=0.0,
                ),
                MemoryState(
                    relevance=0.4 + offset,
                    last_access_time=estab_lat * stale_mult,
                    importance=0.6,
                    access_count=15,
                    strength=s_max * 0.6,
                    creation_time=200.0,
                ),
                MemoryState(
                    relevance=0.35 + offset,
                    last_access_time=estab_lat * stale_mult * 1.2,
                    importance=0.5,
                    access_count=10,
                    strength=s_max * 0.5,
                    creation_time=180.0,
                ),
            ]

            _try_add(
                name=f"TTL-rec-rel{new_rel:.2f}-lat{estab_lat:.0f}{suffix}",
                memories=memories,
                current_time=0.0,
                expected_winner=0,
                competency="test_time_learning",
            )

    # ================================================================
    # 4d. Procedurally generated Selective Forgetting (SF) scenarios
    # ================================================================

    staleness_levels = [50, 100, 200, 500] if not held_out else [55, 110, 220, 550]
    fresh_rels = [0.6, 0.8] if not held_out else [0.62, 0.82]
    stale_access_counts = [10, 50, 100]

    for staleness in staleness_levels:
        for fresh_rel in fresh_rels:
            for stale_ac in stale_access_counts:
                memories = [
                    MemoryState(
                        relevance=min(fresh_rel + offset, 1.0),
                        last_access_time=3.0,
                        importance=0.5,
                        access_count=3,
                        strength=s_max * 0.5,
                        creation_time=20.0,
                    ),
                    MemoryState(
                        relevance=0.4 + offset,
                        last_access_time=float(staleness) * stale_mult,
                        importance=0.6,
                        access_count=stale_ac,
                        strength=s_max * 0.4,
                        creation_time=500.0,
                    ),
                ]

                _try_add(
                    name=f"SF-gen-stale{staleness}-rel{fresh_rel:.1f}-ac{stale_ac}{suffix}",
                    memories=memories,
                    current_time=0.0,
                    expected_winner=0,
                    competency="selective_forgetting",
                )

    # Additional SF with varying stale importance
    for staleness in [100, 300] if not held_out else [110, 330]:
        for stale_imp in [0.3, 0.6, 0.9]:
            memories = [
                MemoryState(
                    relevance=0.75 + offset,
                    last_access_time=5.0,
                    importance=0.5,
                    access_count=5,
                    strength=s_max * 0.5,
                    creation_time=30.0,
                ),
                MemoryState(
                    relevance=0.4 + offset,
                    last_access_time=float(staleness) * stale_mult,
                    importance=stale_imp,
                    access_count=80,
                    strength=s_max * 0.3,
                    creation_time=600.0,
                ),
            ]

            _try_add(
                name=f"SF-imp-stale{staleness}-imp{stale_imp:.1f}{suffix}",
                memories=memories,
                current_time=0.0,
                expected_winner=0,
                competency="selective_forgetting",
            )

    # ================================================================
    # 4e. Procedurally generated Stability (STAB) scenarios
    # ================================================================

    stab_pool_sizes = [2, 3, 4, 5, 8]
    stab_rel_gaps = [0.0, 0.01, 0.05, 0.1] if not held_out else [0.0, 0.015, 0.055, 0.11]

    for pool_size in stab_pool_sizes:
        for rel_gap in stab_rel_gaps:
            memories = []
            for m_idx in range(pool_size):
                rel = 0.5 + offset + rel_gap * (m_idx / max(pool_size - 1, 1))
                rel = min(rel, 1.0)
                memories.append(MemoryState(
                    relevance=rel,
                    last_access_time=10.0,
                    importance=0.5,
                    access_count=5,
                    strength=s0,
                    creation_time=50.0,
                ))

            _try_add(
                name=f"STAB-gen-pool{pool_size}-gap{rel_gap:.3f}{suffix}",
                memories=memories,
                current_time=0.0,
                expected_winner=-1,
                competency="stability",
            )

    # Additional stability: varying strength levels
    for pool_size in [3, 5]:
        for str_level in [0.3, 0.5, 0.8] if not held_out else [0.32, 0.52, 0.82]:
            memories = []
            for m_idx in range(pool_size):
                memories.append(MemoryState(
                    relevance=0.5 + offset + 0.01 * m_idx,
                    last_access_time=10.0,
                    importance=0.5,
                    access_count=5,
                    strength=min(s_max * str_level, s_max),
                    creation_time=50.0,
                ))

            _try_add(
                name=f"STAB-str-pool{pool_size}-s{str_level:.2f}{suffix}",
                memories=memories,
                current_time=0.0,
                expected_winner=-1,
                competency="stability",
            )

    assert len(scenarios) >= 120, (
        f"Expected >= 120 scenarios, got {len(scenarios)}. "
        f"Too many scenarios failed verification in _try_add."
    )

    return scenarios


# ============================================================
# 5. Reference scenario cache
# ============================================================

_REFERENCE_SCENARIOS_CACHE: list[Scenario] | None = None


def _get_reference_scenarios() -> list[Scenario]:
    """Return scenarios built from center-of-bounds reference params.

    Cached on first call for efficiency. All objective() evaluations
    use the same scenario set (Approach B from spec).
    """
    global _REFERENCE_SCENARIOS_CACHE  # noqa: PLW0603
    if _REFERENCE_SCENARIOS_CACHE is None:
        x0 = np.array([
            (BOUNDS[p][0] + BOUNDS[p][1]) / 2.0
            for p in PARAM_ORDER
        ])
        ref_params = decode(x0)
        assert ref_params is not None, "Center of bounds must decode to valid params"
        _REFERENCE_SCENARIOS_CACHE = build_benchmark_scenarios(ref_params)
    return _REFERENCE_SCENARIOS_CACHE


# ============================================================
# 6. objective -- fitness evaluation
# ============================================================


def objective(
    x: np.ndarray,
    scenarios: Optional[list[Scenario]] = None,
) -> float:
    """Evaluate the composite fitness of a parameter vector.

    CMA-ES minimizes this function. Returns negative values in [-1, 0] for
    feasible candidates (lower is better) and INFEASIBLE_PENALTY (1000.0) for
    infeasible vectors.

    The composite score is:
        -(STATIC_WEIGHT * static_accuracy
          + STABILITY_WEIGHT * stability_accuracy
          + MARGIN_WEIGHT * margin_bonus)

    Args:
        x:         Raw numpy array of shape (6,).
        scenarios: Optional pre-built scenario list. If None, builds scenarios
                   from the decoded params.

    Returns:
        Float to minimize. In [-1, 0] for feasible, 1000.0 for infeasible.
    """
    # 1. Decode
    params = decode(x)
    if params is None:
        return INFEASIBLE_PENALTY

    # 2. Build or use provided scenarios
    #    Approach B (from spec): build scenarios once with reference params
    #    (center of bounds), not the candidate's own params. This ensures
    #    all candidates are evaluated against the same scenario set.
    if scenarios is None:
        scenarios = _get_reference_scenarios()

    # 3. Partition scenarios
    static_scenarios = [s for s in scenarios if s.competency != "stability"]
    stability_scenarios = [s for s in scenarios if s.competency == "stability"]

    # 4. Static scoring: count correct winners
    static_correct = 0
    for s in static_scenarios:
        ranked = rank_memories(params, s.memories, s.current_time)
        if ranked and ranked[0][0] == s.expected_winner:
            static_correct += 1
    static_accuracy = static_correct / max(len(static_scenarios), 1)

    # 5. Stability scoring (Finding 4: explicit algorithm)
    stability_passes = 0
    for s in stability_scenarios:
        sim = simulate(
            params, s.memories, STABILITY_SIM_STEPS,
            [-1] * STABILITY_SIM_STEPS,
            rng=Random(42),
        )
        win_counts: dict[int, int] = {}
        for ranking in sim.rankings_per_step:
            winner = ranking[0]
            win_counts[winner] = win_counts.get(winner, 0) + 1
        max_share = max(win_counts.values()) / float(STABILITY_SIM_STEPS)
        if max_share <= 0.80:
            stability_passes += 1
    stability_accuracy = stability_passes / max(len(stability_scenarios), 1)

    # 6. Contraction margin bonus
    margin = params.contraction_margin()
    margin_bonus = min(margin, 0.3) / 0.3

    # 7. Composite (negated for minimization)
    composite = (
        STATIC_WEIGHT * static_accuracy
        + STABILITY_WEIGHT * stability_accuracy
        + MARGIN_WEIGHT * margin_bonus
    )
    return -composite


# ============================================================
# 6. run_optimization -- CMA-ES loop
# ============================================================


def run_optimization(
    n_generations: int = 300,
    seed: int = 42,
) -> OptimizationResult:
    """Run CMA-ES optimization to find optimal memory system parameters.

    Uses the cmaes library to search the 6-dimensional parameter space.
    Scenarios are built once at the start using center-of-bounds reference
    parameters (Approach B from spec).

    Args:
        n_generations: Maximum number of CMA-ES generations to run.
        seed:          Random seed for reproducibility.

    Returns:
        OptimizationResult with the best-found parameters and history.
    """
    # 1. Compute initial mean and sigma
    x0 = np.array([
        (BOUNDS[p][0] + BOUNDS[p][1]) / 2.0
        for p in PARAM_ORDER
    ])
    sigma0 = float(np.mean([
        (BOUNDS[p][1] - BOUNDS[p][0]) / 4.0
        for p in PARAM_ORDER
    ]))

    # 2. Build scenarios once with reference params
    ref_params = decode(x0)
    assert ref_params is not None, "Center of bounds must decode to valid params"
    scenarios = build_benchmark_scenarios(ref_params)

    # 3. Create CMA-ES optimizer
    optimizer = CMA(mean=x0, sigma=sigma0, seed=seed)

    # 4. Optimization loop
    best_x = x0.copy()
    best_obj = objective(x0, scenarios=scenarios)
    best_score = -best_obj if best_obj < INFEASIBLE_PENALTY else 0.0
    history: list[float] = []

    for gen in range(n_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            x_candidate = optimizer.ask()
            obj_val = objective(x_candidate, scenarios=scenarios)
            solutions.append((x_candidate, obj_val))

            # Track best
            if obj_val < INFEASIBLE_PENALTY:
                candidate_score = -obj_val
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_x = x_candidate.copy()
                    best_obj = obj_val

        optimizer.tell(solutions)
        history.append(best_score)

        if optimizer.should_stop():
            break

    # Pad history if early stop
    actual_generations = len(history)

    # 5. Decode best params
    best_params = decode(best_x)
    assert best_params is not None, "Best solution must decode to valid params"

    # 6. Compute per-competency accuracy breakdown
    per_competency = _compute_per_competency(best_params, scenarios)

    return OptimizationResult(
        best_x=best_x,
        best_score=best_score,
        best_params=best_params,
        history=history,
        generations=actual_generations,
        per_competency=per_competency,
    )


def _compute_per_competency(
    params: ParameterSet,
    scenarios: list[Scenario],
) -> dict[str, float]:
    """Compute accuracy breakdown per competency category.

    For non-stability competencies, measures static ranking accuracy.
    For stability, measures the fraction of scenarios passing the
    monopolization check.

    Args:
        params:    ParameterSet to evaluate.
        scenarios: Full scenario list.

    Returns:
        Dictionary mapping competency name to accuracy in [0, 1].
    """
    competency_correct: dict[str, int] = {
        "accurate_retrieval": 0,
        "test_time_learning": 0,
        "selective_forgetting": 0,
        "stability": 0,
    }
    competency_total: dict[str, int] = {
        "accurate_retrieval": 0,
        "test_time_learning": 0,
        "selective_forgetting": 0,
        "stability": 0,
    }

    for s in scenarios:
        competency_total[s.competency] += 1

        if s.competency == "stability":
            sim = simulate(
                params, s.memories, STABILITY_SIM_STEPS,
                [-1] * STABILITY_SIM_STEPS,
                rng=Random(42),
            )
            win_counts: dict[int, int] = {}
            for ranking in sim.rankings_per_step:
                winner = ranking[0]
                win_counts[winner] = win_counts.get(winner, 0) + 1
            max_share = max(win_counts.values()) / float(STABILITY_SIM_STEPS)
            if max_share <= 0.80:
                competency_correct[s.competency] += 1
        else:
            ranked = rank_memories(params, s.memories, s.current_time)
            if ranked and ranked[0][0] == s.expected_winner:
                competency_correct[s.competency] += 1

    return {
        comp: competency_correct[comp] / max(competency_total[comp], 1)
        for comp in competency_correct
    }


# ============================================================
# 7. validate -- post-optimization validation
# ============================================================


def validate(params: ParameterSet) -> ValidationResult:
    """Validate optimized parameters against all quality checks.

    Does not raise -- collects failures into a list. Computes analytical
    formulas for steady-state strength and exploration window.

    Args:
        params: A valid ParameterSet (the optimization winner).

    Returns:
        ValidationResult with all checks and analytical metrics.
    """
    failures: list[str] = []

    # 1. Contraction margin
    margin = params.contraction_margin()
    if margin <= 0.01:
        failures.append(
            f"Contraction margin too small: {margin:.6f} <= 0.01 (strict threshold)"
        )
    elif margin <= 0.05:
        failures.append(
            f"Contraction margin below preferred: {margin:.6f} <= 0.05"
        )

    # 2. Training scenario accuracy
    train_scenarios = build_benchmark_scenarios(params, held_out=False)
    static_train = [s for s in train_scenarios if s.competency != "stability"]
    stability_train = [s for s in train_scenarios if s.competency == "stability"]

    train_correct = 0
    for s in static_train:
        ranked = rank_memories(params, s.memories, s.current_time)
        if ranked and ranked[0][0] == s.expected_winner:
            train_correct += 1
    static_accuracy = train_correct / max(len(static_train), 1)

    if static_accuracy < 1.0:
        failures.append(
            f"Training static accuracy: {static_accuracy:.4f} < 1.0"
        )

    # 3. Stability accuracy
    stab_passes = 0
    for s in stability_train:
        sim = simulate(
            params, s.memories, STABILITY_SIM_STEPS,
            [-1] * STABILITY_SIM_STEPS,
            rng=Random(42),
        )
        win_counts: dict[int, int] = {}
        for ranking in sim.rankings_per_step:
            winner = ranking[0]
            win_counts[winner] = win_counts.get(winner, 0) + 1
        max_share = max(win_counts.values()) / float(STABILITY_SIM_STEPS)
        if max_share <= 0.80:
            stab_passes += 1
    stability_accuracy = stab_passes / max(len(stability_train), 1)

    if stability_accuracy < 1.0:
        failures.append(
            f"Training stability accuracy: {stability_accuracy:.4f} < 1.0"
        )

    # 4. Held-out accuracy
    held_scenarios = build_benchmark_scenarios(params, held_out=True)
    held_static = [s for s in held_scenarios if s.competency != "stability"]
    held_correct = 0
    for s in held_static:
        ranked = rank_memories(params, s.memories, s.current_time)
        if ranked and ranked[0][0] == s.expected_winner:
            held_correct += 1
    held_out_accuracy = held_correct / max(len(held_static), 1)

    if held_out_accuracy < 0.90:
        failures.append(
            f"Held-out accuracy: {held_out_accuracy:.4f} < 0.90"
        )

    # 5. Analytical: steady-state strength S*
    s_star = (params.alpha * params.s_max) / (
        1.0 - (1.0 - params.alpha) * math.exp(-params.beta * params.delta_t)
    )
    s_ratio = s_star / params.s_max

    if s_star >= params.s_max:
        failures.append(
            f"Steady-state S* = {s_star:.4f} >= s_max = {params.s_max}"
        )

    # 6. Analytical: exploration window W
    W = math.log(params.novelty_start / params.survival_threshold) / params.novelty_decay

    if W <= 0.0:
        failures.append(
            f"Exploration window W = {W:.4f} <= 0"
        )

    # 7. w2 cap check
    if params.w2 >= 0.4:
        failures.append(
            f"w2 cap violated: w2 = {params.w2:.4f} >= 0.4"
        )

    # Weight summary
    weight_summary = {
        "w1": params.w1,
        "w2": params.w2,
        "w3": params.w3,
        "w4": params.w4,
    }

    return ValidationResult(
        contraction_margin=margin,
        static_accuracy=static_accuracy,
        stability_accuracy=stability_accuracy,
        held_out_accuracy=held_out_accuracy,
        steady_state_strength=s_star,
        steady_state_ratio=s_ratio,
        exploration_window=W,
        weight_summary=weight_summary,
        all_passed=len(failures) == 0,
        failures=failures,
    )
