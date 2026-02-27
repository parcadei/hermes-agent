"""Hermes Memory System -- One-at-a-Time (OAT) Sensitivity Analysis.

Systematically perturbs each parameter, measures the impact on retrieval
rankings, score distributions, and contraction safety, and classifies each
parameter as critical/sensitive/insensitive.

Mathematical invariant: all scoring and ranking operations delegate to
engine.py, which in turn delegates to core.py. This module adds no new
math -- only the OAT perturbation framework and classification logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from hermes_memory.engine import (
    ParameterSet,
    MemoryState,
    SimulationResult,
    score_memory,
    rank_memories,
    select_memory,
    step_dynamics,
    simulate,
)


# ============================================================
# 1. Scenario dataclass
# ============================================================


@dataclass(frozen=True)
class Scenario:
    """A test scenario for evaluating memory system behavior.

    Each scenario defines a set of memories, a query time, and the
    expected winner -- the memory index that SHOULD rank first under
    correct behavior.

    Attributes:
        name:             Human-readable scenario name.
        memories:         List of MemoryState instances in the scenario.
        current_time:     The time at which to evaluate rankings.
        expected_winner:  Index into memories of the memory that should
                          rank first. Use -1 as a sentinel for stability
                          scenarios that have no single expected winner.
        competency:       Which system competency this scenario tests.
                          One of: "accurate_retrieval", "test_time_learning",
                          "selective_forgetting", "stability".
    """

    name: str
    memories: list[MemoryState]
    current_time: float
    expected_winner: int
    competency: str

    def __post_init__(self) -> None:
        valid_competencies = {
            "accurate_retrieval",
            "test_time_learning",
            "selective_forgetting",
            "stability",
        }
        if self.competency not in valid_competencies:
            raise ValueError(
                f"competency must be one of {valid_competencies}, "
                f"got {self.competency!r}"
            )
        if self.expected_winner == -1:
            if self.competency != "stability":
                raise ValueError(
                    "expected_winner=-1 (stability sentinel) is only valid "
                    "for scenarios with competency='stability', "
                    f"got competency={self.competency!r}"
                )
        elif not (0 <= self.expected_winner < len(self.memories)):
            raise ValueError(
                f"expected_winner {self.expected_winner} out of range "
                f"for {len(self.memories)} memories"
            )


# ============================================================
# 2. SensitivityResult dataclass
# ============================================================


@dataclass(frozen=True)
class SensitivityResult:
    """Result of sensitivity analysis for a single parameter.

    Attributes:
        parameter_name:              Name of the perturbed parameter.
        perturbation_levels:         List of multiplicative perturbation factors.
        kendall_tau_per_level:       Mean Kendall tau across non-stability scenarios.
                                     NaN for skipped levels.
        score_change_per_level:      Mean absolute score change across all memories
                                     and non-stability scenarios. NaN for skipped.
        contraction_margin_per_level: Contraction margin of perturbed params.
                                     NaN for skipped.
        scenarios_correct_per_level: Count of non-stability scenarios where expected
                                     winner still ranks first. 0 for skipped.
        skipped_levels:              Perturbation levels skipped due to constraint
                                     violations.
        scenario_flipped_per_level:  For each level, list of scenario names whose
                                     expected winner was flipped.
        is_weight:                   True if this is a scoring weight (w1-w4).
        classification:              One of "critical", "sensitive", "insensitive".
    """

    parameter_name: str
    perturbation_levels: list[float]
    kendall_tau_per_level: list[float]
    score_change_per_level: list[float]
    contraction_margin_per_level: list[float]
    scenarios_correct_per_level: list[int]
    skipped_levels: list[float] = field(default_factory=list)
    scenario_flipped_per_level: list[list[str]] = field(default_factory=list)
    is_weight: bool = False
    classification: str = "insensitive"


# ============================================================
# 3. Standard scenario suite
# ============================================================


def build_standard_scenarios(params: ParameterSet) -> list[Scenario]:
    """Build 10 standard scenarios covering all 4 competencies.

    Each scenario's expected_winner is verified by computing actual scores
    with the given params. If a scenario's expected winner does not match
    the computed ranking, the scenario parameters are adjusted.

    Returns:
        List of Scenario instances (8-12 scenarios).
    """
    s_max = params.s_max
    s0 = params.s0
    scenarios: list[Scenario] = []

    # ---- Accurate Retrieval (3 scenarios) ----

    # AR-1: High relevance beats high recency
    # M0: high relevance (0.95), moderate recency (lat=5, strength=3)
    # M1: low relevance (0.4), moderate recency (lat=20, strength=8)
    # With w1=0.35 being the largest weight and large relevance gap (0.55),
    # M0 wins even though M1 has better recency.
    ar1_memories = [
        MemoryState(
            relevance=0.95, last_access_time=5.0,
            importance=0.3, access_count=2,
            strength=3.0, creation_time=100.0,
        ),
        MemoryState(
            relevance=0.4, last_access_time=20.0,
            importance=0.3, access_count=2,
            strength=s_max * 0.8, creation_time=100.0,
        ),
    ]
    scenarios.append(Scenario(
        name="AR-1: High relevance beats high recency",
        memories=ar1_memories,
        current_time=0.0,
        expected_winner=0,
        competency="accurate_retrieval",
    ))

    # AR-2: Relevance tiebreaker uses importance
    # Both have same relevance, but M0 has much higher importance.
    ar2_memories = [
        MemoryState(
            relevance=0.7, last_access_time=10.0,
            importance=0.9, access_count=5,
            strength=5.0, creation_time=200.0,
        ),
        MemoryState(
            relevance=0.7, last_access_time=10.0,
            importance=0.1, access_count=5,
            strength=5.0, creation_time=200.0,
        ),
    ]
    scenarios.append(Scenario(
        name="AR-2: Importance tiebreaker",
        memories=ar2_memories,
        current_time=200.0,
        expected_winner=0,
        competency="accurate_retrieval",
    ))

    # AR-3: High relevance beats high activation
    # M0 has high relevance (0.9) but low activation (count=1).
    # M1 has low relevance (0.3) but very high activation (count=100).
    ar3_memories = [
        MemoryState(
            relevance=0.9, last_access_time=20.0,
            importance=0.5, access_count=1,
            strength=s0, creation_time=80.0,
        ),
        MemoryState(
            relevance=0.3, last_access_time=5.0,
            importance=0.5, access_count=100,
            strength=s_max * 0.9, creation_time=80.0,
        ),
    ]
    scenarios.append(Scenario(
        name="AR-3: High relevance beats high activation",
        memories=ar3_memories,
        current_time=0.0,
        expected_winner=0,
        competency="accurate_retrieval",
    ))

    # ---- Test-Time Learning (2 scenarios) ----

    # TTL-1: New high-relevance memory enters established pool
    # M0 is brand new (creation_time=0, full novelty bonus) with high relevance.
    # M1, M2 are established (creation_time=200+, zero novelty).
    ttl1_memories = [
        MemoryState(
            relevance=0.9, last_access_time=0.0,
            importance=0.5, access_count=0,
            strength=s0, creation_time=0.0,
        ),
        MemoryState(
            relevance=0.5, last_access_time=5.0,
            importance=0.7, access_count=20,
            strength=s_max * 0.7, creation_time=200.0,
        ),
        MemoryState(
            relevance=0.4, last_access_time=3.0,
            importance=0.6, access_count=15,
            strength=s_max * 0.6, creation_time=150.0,
        ),
    ]
    scenarios.append(Scenario(
        name="TTL-1: New high-relevance enters pool",
        memories=ttl1_memories,
        current_time=0.0,
        expected_winner=0,
        competency="test_time_learning",
    ))

    # TTL-2: Novelty bonus saves zero-base-score new memory
    # M0 has zero relevance and importance but is brand new (full novelty).
    # M1 is old with moderate relevance but no novelty.
    ttl2_memories = [
        MemoryState(
            relevance=0.0, last_access_time=0.0,
            importance=0.0, access_count=0,
            strength=s0, creation_time=0.0,
        ),
        MemoryState(
            relevance=0.3, last_access_time=20.0,
            importance=0.3, access_count=5,
            strength=s0 * 0.5, creation_time=100.0,
        ),
    ]
    scenarios.append(Scenario(
        name="TTL-2: Novelty saves zero-base memory",
        memories=ttl2_memories,
        current_time=0.0,
        expected_winner=0,
        competency="test_time_learning",
    ))

    # ---- Selective Forgetting (3 scenarios) ----

    # SF-1: Stale heavily-accessed memory loses to fresh relevant one
    # M0 is fresh with high relevance. M1 is very stale (lat=200) even
    # though it has high access count.
    sf1_memories = [
        MemoryState(
            relevance=0.8, last_access_time=5.0,
            importance=0.6, access_count=3,
            strength=s_max * 0.5, creation_time=20.0,
        ),
        MemoryState(
            relevance=0.5, last_access_time=200.0,
            importance=0.8, access_count=100,
            strength=s_max * 0.3, creation_time=500.0,
        ),
    ]
    scenarios.append(Scenario(
        name="SF-1: Stale heavily-accessed loses",
        memories=sf1_memories,
        current_time=0.0,
        expected_winner=0,
        competency="selective_forgetting",
    ))

    # SF-2: Anti-lock-in -- established memory does not dominate
    # M0 has higher relevance and recent access. M1 has massive access
    # count and high strength but is stale (lat=50, retention near zero).
    sf2_memories = [
        MemoryState(
            relevance=0.8, last_access_time=3.0,
            importance=0.6, access_count=5,
            strength=s_max * 0.5, creation_time=50.0,
        ),
        MemoryState(
            relevance=0.5, last_access_time=50.0,
            importance=0.9, access_count=100,
            strength=s_max * 0.95, creation_time=1000.0,
        ),
    ]
    scenarios.append(Scenario(
        name="SF-2: Anti-lock-in",
        memories=sf2_memories,
        current_time=0.0,
        expected_winner=0,
        competency="selective_forgetting",
    ))

    # SF-3: Low-importance memory with low relevance loses
    sf3_memories = [
        MemoryState(
            relevance=0.6, last_access_time=10.0,
            importance=0.5, access_count=3,
            strength=s0, creation_time=30.0,
        ),
        MemoryState(
            relevance=0.15, last_access_time=100.0,
            importance=0.1, access_count=50,
            strength=s_max * 0.2, creation_time=500.0,
        ),
    ]
    scenarios.append(Scenario(
        name="SF-3: Low-importance/relevance loses",
        memories=sf3_memories,
        current_time=0.0,
        expected_winner=0,
        competency="selective_forgetting",
    ))

    # ---- Stability (2 scenarios) ----

    # STAB-1: No monopolization over 100 steps
    stab1_memories = [
        MemoryState(
            relevance=0.5, last_access_time=10.0,
            importance=0.5, access_count=5,
            strength=s0, creation_time=50.0,
        ),
        MemoryState(
            relevance=0.55, last_access_time=10.0,
            importance=0.5, access_count=5,
            strength=s0, creation_time=50.0,
        ),
        MemoryState(
            relevance=0.6, last_access_time=10.0,
            importance=0.5, access_count=5,
            strength=s0, creation_time=50.0,
        ),
        MemoryState(
            relevance=0.45, last_access_time=10.0,
            importance=0.5, access_count=5,
            strength=s0, creation_time=50.0,
        ),
    ]
    scenarios.append(Scenario(
        name="STAB-1: No monopolization",
        memories=stab1_memories,
        current_time=0.0,
        expected_winner=-1,
        competency="stability",
    ))

    # STAB-2: Strength convergence to steady state
    stab2_memories = [
        MemoryState(
            relevance=0.5, last_access_time=10.0,
            importance=0.5, access_count=5,
            strength=s0, creation_time=50.0,
        ),
        MemoryState(
            relevance=0.5, last_access_time=10.0,
            importance=0.5, access_count=5,
            strength=s0, creation_time=50.0,
        ),
    ]
    scenarios.append(Scenario(
        name="STAB-2: Strength convergence",
        memories=stab2_memories,
        current_time=0.0,
        expected_winner=-1,
        competency="stability",
    ))

    # Verify all non-stability scenarios produce the correct expected winner
    for scenario in scenarios:
        if scenario.expected_winner == -1:
            continue
        ranked = rank_memories(params, scenario.memories, scenario.current_time)
        actual_winner = ranked[0][0]
        if actual_winner != scenario.expected_winner:
            scores = [
                score_memory(params, m, scenario.current_time)
                for m in scenario.memories
            ]
            raise RuntimeError(
                f"Scenario {scenario.name!r}: expected winner="
                f"{scenario.expected_winner}, actual winner={actual_winner}. "
                f"Scores: {scores}"
            )

    return scenarios


# ============================================================
# 4. Parameter perturbation
# ============================================================

# All ParameterSet field names in declaration order
_PARAM_NAMES: list[str] = [
    "alpha", "beta", "delta_t", "s_max", "s0", "temperature",
    "novelty_start", "novelty_decay", "survival_threshold",
    "feedback_sensitivity", "w1", "w2", "w3", "w4",
]

_WEIGHT_NAMES: set[str] = {"w1", "w2", "w3", "w4"}

# Parameters that appear in the contraction condition K = exp(-beta*dt) + L*alpha*s_max
_CONTRACTION_PARAMS: set[str] = {"alpha", "beta", "delta_t", "s_max", "temperature"}


def _hypothetical_contraction_margin(
    params: ParameterSet,
    param_name: str,
    factor: float,
) -> float:
    """Compute the contraction margin that WOULD result from a perturbation.

    This works even when the perturbation would violate other constraints
    (like alpha > 1 or temperature <= 0). It only computes the contraction
    condition K = exp(-beta*dt) + (0.25/T)*alpha*s_max.

    Returns NaN if the computation is undefined (e.g., temperature=0).
    """
    kwargs = {name: getattr(params, name) for name in _PARAM_NAMES}
    current = kwargs[param_name]
    new_value = current * (1.0 + factor)
    kwargs[param_name] = new_value

    alpha = kwargs["alpha"]
    beta = kwargs["beta"]
    delta_t = kwargs["delta_t"]
    s_max = kwargs["s_max"]
    temperature = kwargs["temperature"]

    if temperature <= 0:
        return float("nan")

    L = 0.25 / temperature
    K = math.exp(-beta * delta_t) + L * alpha * s_max
    return 1.0 - K


def perturb_parameter(
    params: ParameterSet,
    param_name: str,
    factor: float,
) -> Optional[ParameterSet]:
    """Create a new ParameterSet with one parameter scaled by (1 + factor).

    For weight parameters (w1-w4), the other weights are renormalized to
    maintain sum=1. Returns None if the perturbation violates any
    constraint.

    Args:
        params: Baseline parameter set.
        param_name: Name of the parameter to perturb.
        factor: Multiplicative perturbation factor. The new value is
                current * (1 + factor).

    Returns:
        A new ParameterSet with the perturbed parameter, or None if
        constraints are violated.
    """
    current = getattr(params, param_name)
    new_value = current * (1.0 + factor)

    if param_name in _WEIGHT_NAMES:
        return _perturb_weight(params, param_name, new_value)

    # Build kwargs from all fields
    kwargs = {name: getattr(params, name) for name in _PARAM_NAMES}
    kwargs[param_name] = new_value

    # Co-perturbation for cross-constrained parameters
    if param_name == "s_max":
        kwargs["s0"] = min(kwargs["s0"], new_value * 0.99)
    elif param_name == "novelty_start":
        kwargs["survival_threshold"] = min(
            kwargs["survival_threshold"], new_value * 0.99
        )

    try:
        return ParameterSet(**kwargs)
    except ValueError:
        return None


def _perturb_weight(
    params: ParameterSet,
    weight_name: str,
    new_value: float,
) -> Optional[ParameterSet]:
    """Perturb a scoring weight and renormalize others to maintain sum=1.

    Returns None if constraints are violated.
    """
    weight_names = ["w1", "w2", "w3", "w4"]
    weights = {name: getattr(params, name) for name in weight_names}

    # Set the perturbed weight
    weights[weight_name] = new_value

    # Check bounds for the perturbed weight
    if new_value < 0:
        return None
    if weight_name == "w2" and new_value >= 0.4:
        return None

    # Compute residual for other weights
    residual = 1.0 - new_value
    if residual < 0:
        return None

    # Sum of other weights (for proportional distribution)
    other_names = [n for n in weight_names if n != weight_name]
    other_sum = sum(weights[n] for n in other_names)

    if other_sum <= 0:
        # Cannot distribute proportionally
        return None

    # Distribute residual proportionally among other weights
    for n in other_names:
        weights[n] = weights[n] * (residual / other_sum)

    # Build kwargs
    kwargs = {name: getattr(params, name) for name in _PARAM_NAMES}
    for n in weight_names:
        kwargs[n] = weights[n]

    try:
        return ParameterSet(**kwargs)
    except ValueError:
        return None


# ============================================================
# 5. Kendall tau-b (pure implementation, no scipy)
# ============================================================


def _kendall_tau_b(ranking_a: list[int], ranking_b: list[int]) -> float:
    """Compute Kendall's tau-b between two rankings.

    Both rankings are lists of memory indices in rank order (first element
    is ranked highest). For permutations (no ties), tau-b simplifies to
    (concordant - discordant) / (n*(n-1)/2).

    For a single-element ranking, returns 1.0 (trivially identical).
    """
    n = len(ranking_a)
    if n <= 1:
        return 1.0

    # Convert rank orders to position maps: index -> rank position
    pos_a = {idx: pos for pos, idx in enumerate(ranking_a)}
    pos_b = {idx: pos for pos, idx in enumerate(ranking_b)}

    # All memory indices (union of both rankings)
    indices = list(pos_a.keys())

    concordant = 0
    discordant = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_i = indices[i]
            idx_j = indices[j]
            diff_a = pos_a[idx_i] - pos_a[idx_j]
            diff_b = pos_b[idx_i] - pos_b[idx_j]
            product = diff_a * diff_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return 1.0
    return (concordant - discordant) / total_pairs


# ============================================================
# 6. Classification logic
# ============================================================


def _classify(
    kendall_taus: list[float],
    scenarios_correct_counts: list[int],
    contraction_margins: list[float],
    skipped: list[float],
    perturbation_levels: list[float],
    total_scenarios: int,
    baseline_contraction_margin: float = 0.0,
) -> str:
    """Classify a parameter as critical, sensitive, or insensitive.

    Classification rules are evaluated in order (first match wins).
    Ranking-based rules (tau, scenarios_correct) apply only to non-skipped
    levels. Contraction margin rules apply to ALL levels including skipped,
    because contraction violation is a structural property independent of
    whether the full ParameterSet is constructible.

    The baseline_contraction_margin is used to detect significant margin
    erosion for contraction-affecting parameters (dynamics parameters that
    do not appear in the scoring function).
    """
    non_skipped_indices = [
        i for i, level in enumerate(perturbation_levels)
        if level not in skipped
    ]
    all_indices = list(range(len(perturbation_levels)))

    # --- CRITICAL ---
    # ANY level (including skipped) has contraction_margin < 0
    for i in all_indices:
        if not math.isnan(contraction_margins[i]) and contraction_margins[i] < 0:
            return "critical"

    # Any non-skipped level flips more than 50% of scenario winners
    if total_scenarios > 0:
        for i in non_skipped_indices:
            if scenarios_correct_counts[i] < total_scenarios / 2:
                return "critical"

    # --- SENSITIVE ---
    # Any non-skipped level with |factor| >= 0.25 flips more than 25% of winners
    if total_scenarios > 0:
        for i in non_skipped_indices:
            level = perturbation_levels[i]
            if abs(level) >= 0.25:
                if scenarios_correct_counts[i] < 0.75 * total_scenarios:
                    return "sensitive"

    # Any non-skipped level has kendall_tau < 0.8
    for i in non_skipped_indices:
        tau = kendall_taus[i]
        if not math.isnan(tau) and tau < 0.8:
            return "sensitive"

    # Contraction margin erosion: if the minimum margin at any level
    # (including skipped) drops below 50% of the baseline margin, the
    # parameter significantly affects system stability. This catches
    # dynamics-only parameters (alpha, beta, delta_t, s_max, temperature)
    # that don't affect static scoring but do affect contraction.
    if baseline_contraction_margin > 0:
        for i in all_indices:
            margin = contraction_margins[i]
            if not math.isnan(margin):
                if margin < baseline_contraction_margin * 0.5:
                    return "sensitive"

    # --- INSENSITIVE ---
    # All non-skipped levels have scenarios_correct > 0.75 * total AND tau > 0.9
    if not non_skipped_indices:
        return "insensitive"

    all_correct_high = True
    all_tau_high = True

    for i in non_skipped_indices:
        if total_scenarios > 0 and scenarios_correct_counts[i] <= 0.75 * total_scenarios:
            all_correct_high = False
        tau = kendall_taus[i]
        if not math.isnan(tau) and tau <= 0.9:
            all_tau_high = False

    if all_correct_high and all_tau_high:
        return "insensitive"

    # Fallback: does not clearly match any category
    # Per the spec, if not critical and not clearly insensitive, classify sensitive
    return "sensitive"


# ============================================================
# 7. Single-parameter analysis
# ============================================================

_DEFAULT_PERTURBATION_LEVELS = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]


def analyze_parameter(
    params: ParameterSet,
    scenarios: list[Scenario],
    param_name: str,
    perturbation_levels: Optional[list[float]] = None,
) -> SensitivityResult:
    """Run sensitivity analysis for a single parameter.

    Computes baseline rankings, then evaluates each perturbation level
    to measure Kendall tau, score change, contraction margin, and
    scenario correctness.

    Args:
        params: Baseline parameter set.
        scenarios: List of scenarios to evaluate.
        param_name: Name of the parameter to perturb.
        perturbation_levels: Override default perturbation levels.

    Returns:
        SensitivityResult with per-level metrics and classification.
    """
    if perturbation_levels is None:
        perturbation_levels = list(_DEFAULT_PERTURBATION_LEVELS)

    # Compute baselines for non-stability scenarios
    non_stability_scenarios = [s for s in scenarios if s.expected_winner != -1]
    total_scenarios = len(non_stability_scenarios)

    baseline_rankings: dict[str, list[tuple[int, float]]] = {}
    baseline_scores: dict[str, list[float]] = {}

    for scenario in non_stability_scenarios:
        ranked = rank_memories(params, scenario.memories, scenario.current_time)
        baseline_rankings[scenario.name] = ranked
        baseline_scores[scenario.name] = [
            score_memory(params, m, scenario.current_time)
            for m in scenario.memories
        ]

    # Evaluate each perturbation level
    kendall_taus: list[float] = []
    score_changes: list[float] = []
    contraction_margins: list[float] = []
    scenarios_correct_counts: list[int] = []
    scenario_flipped: list[list[str]] = []
    skipped: list[float] = []

    # Baseline contraction margin for erosion detection
    baseline_margin = params.contraction_margin()

    # Hypothetical contraction margins for ALL levels (including skipped),
    # used by the classifier to detect contraction violations even when
    # the ParameterSet cannot be constructed. The public
    # contraction_margin_per_level uses NaN for skipped levels per the spec.
    hypothetical_margins: list[float] = []

    for factor in perturbation_levels:
        perturbed = perturb_parameter(params, param_name, factor)

        # Always compute hypothetical margin (classifier needs this)
        hyp_margin = _hypothetical_contraction_margin(
            params, param_name, factor
        )
        hypothetical_margins.append(hyp_margin)

        if perturbed is None:
            skipped.append(factor)
            kendall_taus.append(float("nan"))
            score_changes.append(float("nan"))
            contraction_margins.append(float("nan"))
            scenarios_correct_counts.append(0)
            scenario_flipped.append([])
            continue

        contraction_margins.append(perturbed.contraction_margin())

        # Evaluate each non-stability scenario
        all_taus: list[float] = []
        all_score_deltas: list[float] = []
        correct_count = 0
        flipped_names: list[str] = []

        for scenario in non_stability_scenarios:
            perturbed_ranked = rank_memories(
                perturbed, scenario.memories, scenario.current_time
            )
            perturbed_scores = [
                score_memory(perturbed, m, scenario.current_time)
                for m in scenario.memories
            ]

            # Kendall tau between baseline and perturbed ranking orders
            baseline_order = [idx for idx, _ in baseline_rankings[scenario.name]]
            perturbed_order = [idx for idx, _ in perturbed_ranked]
            tau = _kendall_tau_b(baseline_order, perturbed_order)
            all_taus.append(tau)

            # Score change (mean absolute change per memory)
            base_sc = baseline_scores[scenario.name]
            delta = sum(
                abs(a - b) for a, b in zip(base_sc, perturbed_scores)
            ) / max(len(base_sc), 1)
            all_score_deltas.append(delta)

            # Winner check
            if perturbed_ranked[0][0] == scenario.expected_winner:
                correct_count += 1
            else:
                flipped_names.append(scenario.name)

        if all_taus:
            kendall_taus.append(sum(all_taus) / len(all_taus))
        else:
            kendall_taus.append(float("nan"))

        if all_score_deltas:
            score_changes.append(sum(all_score_deltas) / len(all_score_deltas))
        else:
            score_changes.append(float("nan"))

        scenarios_correct_counts.append(correct_count)
        scenario_flipped.append(flipped_names)

    # Classify using hypothetical margins (which include values for
    # skipped levels) so contraction violations are always detected
    classification = _classify(
        kendall_taus, scenarios_correct_counts, hypothetical_margins,
        skipped, perturbation_levels, total_scenarios,
        baseline_contraction_margin=baseline_margin,
    )

    is_weight = param_name in _WEIGHT_NAMES

    return SensitivityResult(
        parameter_name=param_name,
        perturbation_levels=perturbation_levels,
        kendall_tau_per_level=kendall_taus,
        score_change_per_level=score_changes,
        contraction_margin_per_level=contraction_margins,
        scenarios_correct_per_level=scenarios_correct_counts,
        skipped_levels=skipped,
        scenario_flipped_per_level=scenario_flipped,
        is_weight=is_weight,
        classification=classification,
    )


# ============================================================
# 8. Full sensitivity analysis
# ============================================================


def run_sensitivity_analysis(
    params: ParameterSet,
    perturbation_levels: Optional[list[float]] = None,
) -> dict[str, SensitivityResult]:
    """Run OAT sensitivity analysis across all 14 parameters.

    Args:
        params: Baseline parameter set.
        perturbation_levels: Override default perturbation levels.

    Returns:
        Dict mapping parameter name to SensitivityResult.
    """
    scenarios = build_standard_scenarios(params)
    results: dict[str, SensitivityResult] = {}

    for name in _PARAM_NAMES:
        results[name] = analyze_parameter(
            params, scenarios, name, perturbation_levels
        )

    return results


# ============================================================
# 9. Report generation
# ============================================================

# Parameters that only affect dynamics/contraction, not static scoring
_DYNAMICS_ONLY_PARAMS = {
    "alpha", "beta", "delta_t", "s_max", "temperature",
    "feedback_sensitivity", "survival_threshold", "s0",
}


def generate_report(results: dict[str, SensitivityResult]) -> str:
    """Generate a human-readable markdown sensitivity analysis report.

    Args:
        results: Dict mapping parameter name to SensitivityResult.

    Returns:
        Markdown-formatted report string.
    """
    lines: list[str] = []
    lines.append("# Sensitivity Analysis Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Parameter | Classification | Min Kendall Tau | "
        "Min Scenarios Correct | Min Contraction Margin | Weight | "
        "Dynamics-Only |"
    )
    lines.append(
        "|-----------|---------------|-----------------|"
        "----------------------|----------------------|--------|"
        "---------------|"
    )

    # Sort by classification severity: critical > sensitive > insensitive
    classification_order = {"critical": 0, "sensitive": 1, "insensitive": 2}
    sorted_names = sorted(
        results.keys(),
        key=lambda n: (classification_order.get(results[n].classification, 3), n),
    )

    for name in sorted_names:
        result = results[name]
        # Compute min values (ignoring NaN)
        valid_taus = [
            t for t in result.kendall_tau_per_level if not math.isnan(t)
        ]
        min_tau = min(valid_taus) if valid_taus else float("nan")
        tau_str = f"{min_tau:.2f}" if not math.isnan(min_tau) else "N/A"

        valid_correct = [
            c for i, c in enumerate(result.scenarios_correct_per_level)
            if result.perturbation_levels[i] not in result.skipped_levels
        ]
        total_sc = max(valid_correct) if valid_correct else 0
        min_correct = min(valid_correct) if valid_correct else 0
        correct_str = f"{min_correct}/{total_sc}" if valid_correct else "N/A"

        valid_margins = [
            m for m in result.contraction_margin_per_level
            if not math.isnan(m)
        ]
        min_margin = min(valid_margins) if valid_margins else float("nan")
        margin_str = f"{min_margin:.4f}" if not math.isnan(min_margin) else "N/A"

        weight_str = "*" if result.is_weight else ""
        dynamics_str = "yes" if name in _DYNAMICS_ONLY_PARAMS else ""

        lines.append(
            f"| {name} | {result.classification} | {tau_str} | "
            f"{correct_str} | {margin_str} | {weight_str} | {dynamics_str} |"
        )

    lines.append("")
    lines.append(
        "\\* Weight parameters are renormalized when perturbed "
        "(OAT assumption violated). "
        "Results reflect sensitivity to shifting weight balance, "
        "not isolated parameter change."
    )
    lines.append("")

    # Dynamics-only note
    lines.append(
        "Dynamics-only parameters (alpha, beta, delta_t, s_max, s0, "
        "temperature, feedback_sensitivity, survival_threshold) do not "
        "appear in the static scoring function. Their sensitivity is "
        "detected only through the contraction margin. Rankings and "
        "score changes for these parameters will show tau=1.0 and "
        "score_change=0.0, which reflects their invisibility to static "
        "ranking analysis, not actual insensitivity."
    )
    lines.append("")

    # Critical parameters section
    critical = [n for n in sorted_names if results[n].classification == "critical"]
    if critical:
        lines.append("## Critical Parameters")
        lines.append("")
        for name in critical:
            result = results[name]
            lines.append(f"### {name}")
            lines.append("")
            for i, level in enumerate(result.perturbation_levels):
                if level in result.skipped_levels:
                    lines.append(f"- factor={level}: SKIPPED (constraint violation)")
                else:
                    tau = result.kendall_tau_per_level[i]
                    margin = result.contraction_margin_per_level[i]
                    correct = result.scenarios_correct_per_level[i]
                    flipped = result.scenario_flipped_per_level[i]
                    lines.append(
                        f"- factor={level}: tau={tau:.3f}, "
                        f"margin={margin:.4f}, "
                        f"correct={correct}, "
                        f"flipped={flipped}"
                    )
            lines.append("")

    # Sensitive parameters section
    sensitive = [n for n in sorted_names if results[n].classification == "sensitive"]
    if sensitive:
        lines.append("## Sensitive Parameters")
        lines.append("")
        for name in sensitive:
            result = results[name]
            lines.append(f"### {name}")
            lines.append("")
            for i, level in enumerate(result.perturbation_levels):
                if level in result.skipped_levels:
                    lines.append(f"- factor={level}: SKIPPED")
                else:
                    tau = result.kendall_tau_per_level[i]
                    correct = result.scenarios_correct_per_level[i]
                    flipped = result.scenario_flipped_per_level[i]
                    lines.append(
                        f"- factor={level}: tau={tau:.3f}, "
                        f"correct={correct}, "
                        f"flipped={flipped}"
                    )
            lines.append("")

    # Insensitive parameters section
    insensitive = [
        n for n in sorted_names if results[n].classification == "insensitive"
    ]
    if insensitive:
        lines.append("## Insensitive Parameters")
        lines.append("")
        for name in insensitive:
            lines.append(f"- {name}")
        lines.append("")

    return "\n".join(lines)
