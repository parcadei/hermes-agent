"""Hermes Memory System -- N-memory Retrieval Engine.

Pure-math retrieval engine that composes core.py primitives into a usable
scoring/ranking/dynamics pipeline. Bridges the gap between the formal
2-memory mean-field proofs and a practical multi-memory retrieval system.

All mathematical operations delegate to core.py to inherit formally
verified properties. This module adds:
  - ParameterSet: validated parameter container
  - MemoryState: immutable memory snapshot
  - Scoring pipeline: score_memory, rank_memories, select_memory
  - Dynamics pipeline: step_dynamics, simulate
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from hermes_memory.core import (
    retention,
    strength_update,
    strength_decay,
    sigmoid,
    ScoringWeights,
    score,
    clamp01,
    importance_update,
    soft_select,
    novelty_bonus,
    boosted_score,
    exploration_window,
    composed_contraction_factor,
)


# ============================================================
# 1. ParameterSet -- Validated parameter container
# ============================================================


@dataclass(frozen=True)
class ParameterSet:
    """Complete parameter set for the Hermes memory dynamics.

    Frozen and validated at construction time. Every parameter carries
    domain constraints derived from the Lean formalization.

    Attributes -- Dynamics:
        alpha:               Learning rate for strength update. Domain: (0, 1).
        beta:                Strength decay rate. Domain: (0, +inf).
        delta_t:             Time step duration. Domain: (0, +inf).
        s_max:               Maximum strength. Domain: (0, +inf).
        s0:                  Initial strength for new memories. Domain: (0, s_max).
        temperature:         Soft selection temperature. Domain: (0, +inf).
        novelty_start:       Initial novelty bonus N0. Domain: (0, +inf).
        novelty_decay:       Novelty decay rate gamma. Domain: (0, +inf).
        survival_threshold:  Minimum score for survival epsilon. Domain: (0, novelty_start).
        feedback_sensitivity: Importance update step size delta. Domain: (0, +inf).

    Attributes -- Scoring weights:
        w1:                  Relevance weight. Domain: [0, 1].
        w2:                  Recency weight. Domain: [0, 0.4).
        w3:                  Importance weight. Domain: [0, 1].
        w4:                  Activation weight. Domain: [0, 1].
                             w1 + w2 + w3 + w4 = 1.0 (convex combination).
    """

    # Dynamics
    alpha: float
    beta: float
    delta_t: float
    s_max: float
    s0: float
    temperature: float
    novelty_start: float
    novelty_decay: float
    survival_threshold: float
    feedback_sensitivity: float

    # Scoring weights
    w1: float
    w2: float
    w3: float
    w4: float

    def __post_init__(self) -> None:
        """Validate all parameter constraints.

        Violations raise ValueError with a descriptive message.
        Constraints are checked in declaration order.
        """
        if not (0 < self.alpha < 1):
            raise ValueError(
                f"alpha must be in (0, 1), got {self.alpha}"
            )
        if self.beta <= 0:
            raise ValueError(
                f"beta must be > 0, got {self.beta}"
            )
        if self.delta_t <= 0:
            raise ValueError(
                f"delta_t must be > 0, got {self.delta_t}"
            )
        if self.s_max <= 0:
            raise ValueError(
                f"s_max must be > 0, got {self.s_max}"
            )
        if not (0 < self.s0 < self.s_max):
            raise ValueError(
                f"s0 must be in (0, s_max={self.s_max}), got {self.s0}"
            )
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {self.temperature}"
            )
        if self.novelty_start <= 0:
            raise ValueError(
                f"novelty_start must be > 0, got {self.novelty_start}"
            )
        if self.novelty_decay <= 0:
            raise ValueError(
                f"novelty_decay must be > 0, got {self.novelty_decay}"
            )
        if not (0 < self.survival_threshold < self.novelty_start):
            raise ValueError(
                f"survival_threshold must be in (0, novelty_start={self.novelty_start}), "
                f"got {self.survival_threshold}"
            )
        if self.feedback_sensitivity <= 0:
            raise ValueError(
                f"feedback_sensitivity must be > 0, got {self.feedback_sensitivity}"
            )
        if self.w1 < 0:
            raise ValueError(
                f"w1 must be >= 0, got {self.w1}"
            )
        if self.w2 < 0 or self.w2 >= 0.4:
            raise ValueError(
                f"w2 must be in [0, 0.4), got {self.w2}"
            )
        if self.w3 < 0:
            raise ValueError(
                f"w3 must be >= 0, got {self.w3}"
            )
        if self.w4 < 0:
            raise ValueError(
                f"w4 must be >= 0, got {self.w4}"
            )
        if abs(self.w1 + self.w2 + self.w3 + self.w4 - 1.0) >= 1e-10:
            raise ValueError(
                f"weights must sum to 1.0, got {self.w1 + self.w2 + self.w3 + self.w4}"
            )
        # Contraction condition: K < 1
        if not self.satisfies_contraction():
            margin = self.contraction_margin()
            raise ValueError(
                f"contraction condition violated: margin={margin:.6f}. "
                f"K = exp(-beta*delta_t) + L*alpha*s_max >= 1 "
                f"where L = 0.25/temperature"
            )

    def satisfies_contraction(self) -> bool:
        """Check whether the contraction condition K < 1 holds.

        K = exp(-beta * delta_t) + L * alpha * s_max
        where L = 0.25 / temperature (sigmoid Lipschitz constant).
        """
        L = 0.25 / self.temperature
        K = math.exp(-self.beta * self.delta_t) + L * self.alpha * self.s_max
        return K < 1.0

    def contraction_margin(self) -> float:
        """Return 1 - K, where K is the contraction factor.

        Positive margin means contraction holds. Zero or negative means
        it does not.
        """
        L = 0.25 / self.temperature
        K = math.exp(-self.beta * self.delta_t) + L * self.alpha * self.s_max
        return 1.0 - K

    def to_scoring_weights(self) -> ScoringWeights:
        """Extract a ScoringWeights instance for use with core.score()."""
        return ScoringWeights(w1=self.w1, w2=self.w2, w3=self.w3, w4=self.w4)


# ============================================================
# 2. MemoryState -- Immutable memory snapshot
# ============================================================


@dataclass(frozen=True)
class MemoryState:
    """Immutable snapshot of a single memory evaluated against a specific query.

    All fields carry domain constraints that the engine preserves
    across dynamics steps (forward-invariance).

    The relevance field is not an intrinsic memory property -- it is
    the cosine similarity to the query that produced this snapshot.

    Attributes:
        relevance:         Cosine similarity to the current query. Domain: [0, 1].
        last_access_time:  Time steps since last accessed. Domain: [0, +inf).
        importance:        Learned importance from feedback. Domain: [0, 1].
        access_count:      Total number of accesses. Domain: non-negative integer.
        strength:          Memory strength. Domain: [0, +inf).
        creation_time:     Time steps since creation. Domain: [0, +inf).
    """

    relevance: float
    last_access_time: float
    importance: float
    access_count: int
    strength: float
    creation_time: float

    def __post_init__(self) -> None:
        """Validate all field constraints with ValueError."""
        if not (0.0 <= self.relevance <= 1.0):
            raise ValueError(
                f"relevance must be in [0, 1], got {self.relevance}"
            )
        if self.last_access_time < 0.0:
            raise ValueError(
                f"last_access_time must be >= 0, got {self.last_access_time}"
            )
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(
                f"importance must be in [0, 1], got {self.importance}"
            )
        if self.access_count < 0:
            raise ValueError(
                f"access_count must be >= 0, got {self.access_count}"
            )
        if self.strength < 0.0:
            raise ValueError(
                f"strength must be >= 0, got {self.strength}"
            )
        if self.creation_time < 0.0:
            raise ValueError(
                f"creation_time must be >= 0, got {self.creation_time}"
            )


# ============================================================
# 3. SimulationResult -- Output of multi-step simulation
# ============================================================


@dataclass(frozen=True)
class SimulationResult:
    """Output of a multi-step simulation.

    Attributes:
        rankings_per_step:  For each step, memory indices in descending score order.
        scores_per_step:    For each step, score of each memory (original index order).
        strengths_per_step: For each step, strength of each memory.
        final_memories:     The MemoryState list after the last step.
    """

    rankings_per_step: list[list[int]]
    scores_per_step: list[list[float]]
    strengths_per_step: list[list[float]]
    final_memories: list[MemoryState]


# ============================================================
# 4. Scoring functions (pure -- no state mutation)
# ============================================================


def score_memory(
    params: ParameterSet,
    memory: MemoryState,
    current_time: float,
) -> float:
    """Compute the composite score for a single memory.

    The current_time parameter is accepted for API consistency but is not
    used in the current implementation. All time-dependent computations
    use the MemoryState's own fields.

    Returns a value in [0, 1 + params.novelty_start]. The return value
    is NOT clamped to [0, 1] -- the novelty bonus is additive and unclamped.
    """
    # 1. Recency via core.retention, guarding against division by zero
    safe_strength = max(memory.strength, 1e-12)
    rec = retention(memory.last_access_time, safe_strength)

    # 2. Activation: pass access_count as the pre-sigmoid value.
    #    core.score() applies sigmoid internally.
    act = float(memory.access_count)

    # 3. Base score via core.score (applies sigmoid to act internally)
    weights = params.to_scoring_weights()
    base = score(weights, memory.relevance, rec, memory.importance, act)

    # 4. Novelty bonus based on creation_time (age of memory)
    age = memory.creation_time
    if age >= 0:
        bonus = novelty_bonus(params.novelty_start, params.novelty_decay, age)
    else:
        bonus = params.novelty_start

    # 5. Return unclamped sum (adversarial finding 3: DO NOT clamp)
    return base + bonus


def rank_memories(
    params: ParameterSet,
    memories: list[MemoryState],
    current_time: float,
) -> list[tuple[int, float]]:
    """Rank all memories by descending score.

    Returns a list of (index, score) tuples sorted by (-score, index)
    for deterministic tie-breaking (lower index wins ties).
    """
    if not memories:
        return []

    scored = [
        (i, score_memory(params, m, current_time))
        for i, m in enumerate(memories)
    ]
    # Sort by descending score, then ascending index for tie-breaking
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored


def select_memory(
    params: ParameterSet,
    memories: list[MemoryState],
    current_time: float,
    rng: Optional[random.Random] = None,
) -> int:
    """Probabilistic soft selection using pairwise tournament.

    Sequential tournament: start with the highest-ranked memory as
    champion, then challenge with each subsequent memory using
    soft_select for pairwise comparison.

    Args:
        params: Parameter set (temperature controls selectivity).
        memories: List of memories to select from.
        current_time: Current time for scoring.
        rng: Random number generator for reproducibility.

    Returns:
        Index of the selected memory.

    Raises:
        ValueError: If memories list is empty.
    """
    if not memories:
        raise ValueError("Cannot select from empty memory list")
    if len(memories) == 1:
        return 0

    if rng is None:
        rng = random.Random()

    ranked = rank_memories(params, memories, current_time)

    winner_idx, winner_score = ranked[0]
    for challenger_idx, challenger_score in ranked[1:]:
        p_winner = soft_select(winner_score, challenger_score, params.temperature)
        u = rng.random()
        if u >= p_winner:
            winner_idx = challenger_idx
            winner_score = challenger_score

    return winner_idx


# ============================================================
# 5. Dynamics functions (state mutation via new instances)
# ============================================================


def step_dynamics(
    params: ParameterSet,
    memories: list[MemoryState],
    accessed_idx: Optional[int],
    feedback_signal: float,
    current_time: float,
) -> list[MemoryState]:
    """Advance the memory system by one time step.

    For all memories: decay strength by exp(-beta * delta_t).
    For accessed memory: additionally reinforce strength and update importance.
    Returns a new list of MemoryState instances (immutable).

    The ordering is: decay first, then reinforce (algebraically equivalent
    to the Lean formalization's single-step formula).
    """
    if not memories:
        return []

    if accessed_idx is not None and not (0 <= accessed_idx < len(memories)):
        raise IndexError(
            f"accessed_idx {accessed_idx} out of range for {len(memories)} memories"
        )

    result = []
    for i, m in enumerate(memories):
        # 1. Strength decay (all memories)
        decayed_strength = strength_decay(params.beta, m.strength, params.delta_t)
        decayed_strength = max(0.0, min(decayed_strength, params.s_max))

        if i == accessed_idx:
            # 2a. Strength reinforcement (accessed only)
            new_strength = strength_update(params.alpha, decayed_strength, params.s_max)
            new_strength = max(0.0, min(new_strength, params.s_max))

            # 2b. Importance update
            new_importance = importance_update(
                m.importance, params.feedback_sensitivity, feedback_signal
            )

            # 2c. Access time reset
            new_last_access_time = 0.0

            # 2d. Access count increment
            new_access_count = m.access_count + 1
        else:
            # 3. Non-accessed memory: only decay
            new_strength = decayed_strength
            new_importance = m.importance
            new_last_access_time = m.last_access_time + params.delta_t
            new_access_count = m.access_count

        # 4. Creation time update (all memories)
        new_creation_time = m.creation_time + params.delta_t

        # 5. Relevance unchanged (exogenous)
        new_relevance = m.relevance

        result.append(MemoryState(
            relevance=new_relevance,
            last_access_time=new_last_access_time,
            importance=new_importance,
            access_count=new_access_count,
            strength=new_strength,
            creation_time=new_creation_time,
        ))

    return result


def simulate(
    params: ParameterSet,
    memories: list[MemoryState],
    n_steps: int,
    access_pattern: list[Optional[int]],
    feedback_signals: Optional[list[float]] = None,
    rng: Optional[random.Random] = None,
) -> SimulationResult:
    """Run a multi-step simulation with a prescribed access pattern.

    Args:
        params: Parameter set for scoring and dynamics.
        memories: Initial memory states.
        n_steps: Number of simulation steps.
        access_pattern: List of length n_steps. Each element is an integer
            index (which memory to access), None (no access), or -1
            (use select_memory to choose).
        feedback_signals: Optional list of length n_steps with feedback
            signal per step. Defaults to 1.0 for all steps.
        rng: Random number generator for select_memory calls.

    Returns:
        SimulationResult with per-step rankings, scores, strengths,
        and final memory states.
    """
    if len(access_pattern) != n_steps:
        raise ValueError(
            f"access_pattern length {len(access_pattern)} != n_steps {n_steps}"
        )
    if feedback_signals is not None and len(feedback_signals) != n_steps:
        raise ValueError(
            f"feedback_signals length {len(feedback_signals)} != n_steps {n_steps}"
        )

    current_memories = list(memories)
    rankings_per_step: list[list[int]] = []
    scores_per_step: list[list[float]] = []
    strengths_per_step: list[list[float]] = []

    for step in range(n_steps):
        current_time = float(step)

        # 1. Score and rank
        ranked = rank_memories(params, current_memories, current_time)
        rankings_per_step.append([idx for idx, _ in ranked])
        scores_per_step.append([
            score_memory(params, m, current_time)
            for m in current_memories
        ])
        strengths_per_step.append([m.strength for m in current_memories])

        # 2. Determine access
        access_idx = access_pattern[step]
        if access_idx == -1:
            access_idx = select_memory(
                params, current_memories, current_time, rng=rng
            )

        # 3. Get feedback signal
        signal = feedback_signals[step] if feedback_signals is not None else 1.0

        # 4. Advance dynamics
        current_memories = step_dynamics(
            params, current_memories, access_idx, signal, current_time
        )

    return SimulationResult(
        rankings_per_step=rankings_per_step,
        scores_per_step=scores_per_step,
        strengths_per_step=strengths_per_step,
        final_memories=current_memories,
    )
