# Behavioral Specification: Sensitivity Analysis Engine

Created: 2026-02-27
Author: architect-agent
Status: COMPLETE — ready for implementation
Reviewed: 2026-02-27 (adversarial pass 1/3 — Brenner method, 14 findings applied)

---

## Overview

Two new modules for the Hermes memory system's Python verification suite:

1. **`engine.py`** — A pure-math N-memory retrieval engine that composes the 18
   core.py primitives into a usable scoring/ranking/dynamics pipeline. It bridges
   the gap between the formal 2-memory mean-field proofs and a practical
   multi-memory retrieval system.

2. **`sensitivity.py`** — One-at-a-time (OAT) sensitivity analysis that
   systematically perturbs each parameter, measures the impact on retrieval
   rankings, score distributions, and contraction safety, and classifies each
   parameter as critical/sensitive/insensitive.

Together they answer: "Which parameters control the system's behavior, and how
much can each one move before correctness guarantees break?"

---

## Part 1: `engine.py` — Pure-Math Retrieval Engine

File: `proofs/hermes-memory/python/hermes_memory/engine.py`

### 1.1 Imports and Dependencies

```python
from __future__ import annotations

import math
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
```

**Hard rule:** engine.py MUST NOT reimplement any function from core.py. Every
mathematical operation (retention, sigmoid, score, soft_select, strength_update,
strength_decay, importance_update, novelty_bonus, boosted_score, clamp01) MUST
be a direct call to the core.py function. This ensures the engine inherits the
formally verified properties.

### 1.2 ParameterSet Dataclass

```python
@dataclass(frozen=True)
class ParameterSet:
    """Complete parameter set for the Hermes memory dynamics.

    Frozen and validated at construction time. Every parameter carries
    domain constraints derived from the Lean formalization.

    Attributes — Dynamics:
        alpha:               Learning rate for strength update. Domain: (0, 1).
                             Controls how fast strength approaches s_max on access.
        beta:                Strength decay rate. Domain: (0, +inf).
                             Anti-lock-in: dS/dt = -beta * S between accesses.
        delta_t:             Time step duration. Domain: (0, +inf).
                             Period between discrete dynamics steps.
        s_max:               Maximum strength. Domain: (0, +inf).
                             Upper bound of the forward-invariant domain [0, s_max].
        s0:                  Initial strength for new memories. Domain: (0, s_max).
                             Must be strictly less than s_max (Lean: strengthUpdate
                             increases S only when S < Smax).
                             NOTE: s0 is a CONFIGURATION parameter, used only at
                             memory creation time. It is not consumed by any engine
                             dynamics function (step_dynamics, score_memory, etc.).
                             It participates in ParameterSet validation to ensure
                             initial conditions are consistent with s_max.
        temperature:         Soft selection temperature. Domain: (0, +inf).
                             Controls selectivity: low T -> near-deterministic,
                             high T -> near-uniform. Affects Lipschitz constant
                             L = 0.25 / T of soft_select.
        novelty_start:       Initial novelty bonus N0. Domain: (0, +inf).
                             Lean: coldStart_survival requires N0 > 0.
        novelty_decay:       Novelty decay rate gamma. Domain: (0, +inf).
                             Lean: explorationWindow = ln(N0/epsilon) / gamma.
        survival_threshold:  Minimum score for survival epsilon. Domain: (0, novelty_start).
                             Lean: epsilon < N0 required by coldStart_survival.
                             NOTE: survival_threshold is a CONFIGURATION parameter,
                             not consumed by any engine dynamics function. It
                             participates in validation to ensure cold-start
                             guarantees are satisfiable.
        feedback_sensitivity: Importance update step size delta. Domain: (0, +inf).
                             Controls imp' = clamp(imp + delta * signal, 0, 1).

    Attributes — Scoring weights:
        w1:                  Relevance weight. Domain: [0, 1].
        w2:                  Recency weight. Domain: [0, 0.4).
                             Capped at 0.4 — extended contraction condition
                             (MEMORY_SYSTEM.md Section 4.4.1). When w2 >= 0.5
                             and S is near zero, dR/dS diverges and contraction
                             breaks.
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
```

**Validation (__post_init__):**

The `__post_init__` method MUST enforce every constraint. Violations raise
`ValueError` with a descriptive message (not `AssertionError` — these are
data-validation errors, not programming errors).

| Constraint | Source | Check |
|------------|--------|-------|
| 0 < alpha < 1 | Lean: hα₀, hα₁ | `not (0 < alpha < 1)` |
| beta > 0 | Lean: hβ | `beta <= 0` |
| delta_t > 0 | Lean: hΔ | `delta_t <= 0` |
| s_max > 0 | Lean: hSmax | `s_max <= 0` |
| 0 < s0 < s_max | Lean: S₀ ∈ (0, Smax) | `not (0 < s0 < s_max)` |
| temperature > 0 | Lean: hT (soft_select) | `temperature <= 0` |
| novelty_start > 0 | Lean: hN₀ | `novelty_start <= 0` |
| novelty_decay > 0 | Lean: hγ | `novelty_decay <= 0` |
| 0 < survival_threshold < novelty_start | Lean: hε, hεN | `not (0 < survival_threshold < novelty_start)` |
| feedback_sensitivity > 0 | Design constraint | `feedback_sensitivity <= 0` |
| w1 >= 0 | Lean: score_nonneg | `w1 < 0` |
| 0 <= w2 < 0.4 | MEMORY_SYSTEM.md §4.4.1 | `w2 < 0 or w2 >= 0.4` |
| w3 >= 0 | Lean: score_nonneg | `w3 < 0` |
| w4 >= 0 | Lean: score_nonneg | `w4 < 0` |
| w1 + w2 + w3 + w4 = 1 | Lean: ScoringWeights.__post_init__ | `abs(w1 + w2 + w3 + w4 - 1.0) >= 1e-10` |

Validation MUST check constraints in the order listed. First failing constraint
determines the error message.

**Parameter-function coupling matrix** *(adversarial finding 5):*

This table shows which parameters are consumed by which engine functions. It
makes the parameter-function coupling explicit and aids sensitivity analysis
interpretation (a parameter that only affects one function cannot cause failures
in the others).

| Parameter | score_memory | step_dynamics | select_memory | simulate |
|-----------|:---:|:---:|:---:|:---:|
| w1, w2, w3, w4 | yes | - | via score | yes |
| novelty_start, novelty_decay | yes | - | via score | yes |
| temperature | - | - | yes | yes |
| alpha | - | yes | - | yes |
| beta | - | yes | - | yes |
| delta_t | - | yes | - | yes |
| s_max | - | yes (clamp) | - | yes |
| s0 | - | - | - | - |
| feedback_sensitivity | - | yes | - | yes |
| survival_threshold | - | - | - | - |

**Method: `satisfies_contraction() -> bool`:**

```
L = 0.25 / self.temperature    # sigmoid Lipschitz constant
K = exp(-beta * delta_t) + L * alpha * s_max
return K < 1.0
```

This checks the composed contraction condition from Lean theorem
`composedContractionFactor_lt_one` (ComposedSystem.lean:196):
`L * alpha * Smax < 1 - exp(-beta * delta_t)`.

The Lipschitz constant L = 0.25/T comes from:
- sigmoid has maximum derivative 0.25 (at x=0)
- soft_select(s1, s2, T) = sigmoid((s1 - s2) / T)
- By chain rule, d/ds1 soft_select = (1/T) * sigmoid'((s1-s2)/T) <= 0.25/T

**Method: `contraction_margin() -> float`:**

```
L = 0.25 / self.temperature
K = exp(-beta * delta_t) + L * alpha * s_max
return 1.0 - K
```

Positive margin means contraction holds. Zero or negative means it does not.
This gives a continuous measure of "how far inside the safe region" the
parameters are, which is critical for sensitivity analysis (a parameter
perturbation that flips the sign of the margin crosses the stability boundary).

**Method: `to_scoring_weights() -> ScoringWeights`:**

```
return ScoringWeights(w1=self.w1, w2=self.w2, w3=self.w3, w4=self.w4)
```

Convenience method to extract a `ScoringWeights` instance for use with
`core.score()`. The core ScoringWeights validates sum=1 and non-negativity
independently.

> **ADVERSARIAL NOTE (finding 9):** core.py's `ScoringWeights` uses `assert`
> for validation (raising `AssertionError`), not `ValueError`. ParameterSet
> validates the same constraints with `ValueError` before calling
> `to_scoring_weights()`. The ParameterSet validation is the authoritative
> check; the ScoringWeights assert is a defense-in-depth backstop from the Lean
> translation. Tests MUST NOT rely on ScoringWeights raising a specific exception
> type — test ParameterSet validation directly.

### 1.3 MemoryState Dataclass

```python
@dataclass(frozen=True)
class MemoryState:
    """Immutable snapshot of a single memory evaluated against a specific query.

    All fields carry domain constraints that the engine MUST preserve
    across dynamics steps (forward-invariance).

    IMPORTANT: MemoryState represents a memory *evaluated against a specific
    query*. The relevance field is not an intrinsic memory property — it is
    the cosine similarity to the query that produced this snapshot. Sensitivity
    analysis results are conditional on fixed-query evaluation; parameter
    sensitivity under query variation requires a separate analysis.

    Fields are divided into two categories:
    - INTRINSIC (strength, importance, access_count, creation_time,
      last_access_time): Properties of the memory itself, evolving under
      dynamics.
    - QUERY-DEPENDENT (relevance): A function of the current query and
      this memory's embedding. Set at query time, not modified by dynamics.

    Attributes:
        relevance:         Cosine similarity to the current query. Domain: [0, 1].
                           Exogenous — set at query time, not modified by dynamics.
                           (QUERY-DEPENDENT)
        last_access_time:  Time steps since this memory was last accessed.
                           Domain: [0, +inf). Reset to 0 on access. (INTRINSIC)
        importance:        Learned importance from feedback. Domain: [0, 1].
                           Updated by importance_update() on access. (INTRINSIC)
        access_count:      Total number of times this memory has been accessed.
                           Domain: non-negative integer. Incremented on access.
                           (INTRINSIC)
        strength:          Memory strength (controls retention decay rate).
                           Domain: [0, s_max]. Updated by strength_update on
                           access, decayed by strength_decay between accesses.
                           (INTRINSIC)
        creation_time:     Time steps since this memory was created.
                           Domain: [0, +inf). Monotonically increasing. Used
                           for novelty_bonus computation. (INTRINSIC)
    """

    relevance: float
    last_access_time: float
    importance: float
    access_count: int
    strength: float
    creation_time: float
```

**Validation (__post_init__):**

| Field | Constraint | Action on violation |
|-------|-----------|---------------------|
| relevance | 0.0 <= x <= 1.0 | `ValueError` |
| last_access_time | x >= 0.0 | `ValueError` |
| importance | 0.0 <= x <= 1.0 | `ValueError` |
| access_count | x >= 0 | `ValueError` |
| strength | x >= 0.0 | `ValueError` |
| creation_time | x >= 0.0 | `ValueError` |

Note: strength upper bound (strength <= s_max) is NOT checked in MemoryState
because MemoryState does not know s_max. The engine functions that create
MemoryState instances MUST clamp strength to [0, s_max].

### 1.4 SimulationResult Dataclass

```python
@dataclass(frozen=True)
class SimulationResult:
    """Output of a multi-step simulation.

    Attributes:
        rankings_per_step:  For each step, a list of memory indices in
                            descending score order. Length: n_steps.
                            Each inner list has length = number of memories.
        scores_per_step:    For each step, the score of each memory (indexed
                            by original position). Length: n_steps.
                            Each inner list has length = number of memories.
        strengths_per_step: For each step, the strength of each memory.
                            Length: n_steps. Same indexing as scores.
        final_memories:     The MemoryState list after the last step.
    """

    rankings_per_step: list[list[int]]
    scores_per_step: list[list[float]]
    strengths_per_step: list[list[float]]
    final_memories: list[MemoryState]
```

### 1.5 Engine Functions

#### `score_memory(params: ParameterSet, memory: MemoryState, current_time: float) -> float`

Compute the composite score for a single memory at the given time.

> **ADVERSARIAL NOTE (finding 11):** The `current_time` parameter is accepted
> for API consistency (future extensibility for time-varying queries) but is NOT
> used in the current implementation. All time-dependent computations use the
> MemoryState's own fields (creation_time for novelty, last_access_time for
> retention). This parameter exists so that future extensions (e.g., time-varying
> relevance) do not require API changes.

**Algorithm:**
1. Compute recency: `rec = retention(memory.last_access_time, max(memory.strength, 1e-12))`
   - Use max(strength, 1e-12) to avoid division by zero in retention when
     strength has decayed to zero. This is the IEEE754 guard — in exact math,
     strength > 0 is guaranteed by the domain invariant, but floating-point
     decay can reach zero.
   - **ADVERSARIAL NOTE (finding 13):** The IEEE754 guard `max(strength, 1e-12)`
     prevents NaN/Inf from `retention(0, 0)` but creates a discontinuity:
     `retention(epsilon, 1e-12)` ~ 0 for any `epsilon > 0` while
     `retention(0, 1e-12)` = 1.0. This is physically correct (t=0 means 'just
     accessed') but numerically discontinuous. Scenarios should avoid memories
     with both `strength ~ 0` and `last_access_time ~ 0`.
2. Compute activation: `act = float(memory.access_count)`
   - The Lean formalization uses sigmoid(activation) where activation is a
     real number. We use access_count directly as the pre-sigmoid activation
     value. core.score() applies sigmoid internally.
3. Compute base score: `base = score(params.to_scoring_weights(), memory.relevance, rec, memory.importance, act)`
4. Compute novelty boost: `boost = novelty_bonus(params.novelty_start, params.novelty_decay, memory.creation_time)`
5. Return: `base + boost`
   - Note: the return value CAN exceed 1.0 due to the novelty bonus addition.
     The base score is in [0,1] (Lean: score_mem_Icc), and the novelty bonus
     is in [0, N0]. The total is in [0, 1 + N0].
   - This matches `boosted_score` from core.py (which is base + novelty_bonus).
   - **DO NOT clamp the return value to [0, 1].** The novelty bonus MUST be
     additive and unclamped. Clamping would violate the cold-start survival
     guarantee (`coldStart_survival` requires `boosted_score >= survival_threshold`
     even when `base_score = 0`). For memories with high base scores, the
     boosted score exceeds 1.0 — this is correct and intentional.
     *(adversarial finding 3)*
   - The Lipschitz bound L = 0.25/T for soft_select holds regardless of input
     magnitude because sigmoid' <= 0.25 globally. Scores exceeding 1.0 do not
     break the Lipschitz analysis. *(adversarial finding 3)*

**Postconditions:**
- Return value >= 0.0 (base >= 0 by score_nonneg, boost >= 0 by noveltyBonus_pos)
- Return value <= 1.0 + params.novelty_start (score_le_one + noveltyBonus_le_init)

#### `rank_memories(params: ParameterSet, memories: list[MemoryState], current_time: float) -> list[tuple[int, float]]`

Rank all memories by descending score.

**Algorithm:**
1. Compute `scores = [(i, score_memory(params, m, current_time)) for i, m in enumerate(memories)]`
2. Sort by `(-score, index)` — descending score, ascending index for tie-breaking
3. Return the sorted list of `(index, score)` tuples

**Tie-breaking policy:** When two memories have identical scores (within
floating-point equality, i.e., `score_a == score_b` exactly), the memory with
the lower index wins. This produces a strict total order on every call.

**Edge cases:**
- Empty list: return `[]`
- Single memory: return `[(0, score)]`

**Postconditions:**
- Length of result == length of memories
- Indices in result are a permutation of `range(len(memories))`
- Scores are monotonically non-increasing in result order
- For adjacent entries with equal scores, indices are strictly increasing

#### `select_memory(params: ParameterSet, memories: list[MemoryState], current_time: float) -> int`

Probabilistic soft selection of a single memory using pairwise tournament.

**Algorithm (sequential tournament):**
1. Compute `ranked = rank_memories(params, memories, current_time)`
2. If 0 memories: raise `ValueError("Cannot select from empty memory list")`
3. If 1 memory: return `ranked[0][0]`
4. Start with `winner_idx, winner_score = ranked[0]`
5. For each subsequent `(challenger_idx, challenger_score)` in `ranked[1:]`:
   - Compute `p_winner = soft_select(winner_score, challenger_score, params.temperature)`
   - Draw uniform random `u` in [0, 1)
   - If `u >= p_winner`: set `winner_idx, winner_score = challenger_idx, challenger_score`
6. Return `winner_idx`

**Rationale for tournament design:** The Lean formalization proves soft_select
for a 2-memory system. The sequential tournament extends to N memories by
iterating pairwise comparisons. The highest-ranked memory starts as the
incumbent, giving it a natural advantage (it has the highest score, so
soft_select favors it against each challenger). This preserves the anti-thrashing
property: small score changes cause small probability changes in each pairwise
comparison.

> **ADVERSARIAL CAVEAT (finding 4):** The contraction factor
> `K = exp(-beta*dt) + L*alpha*s_max` is PROVEN for the 2-memory system only.
> For N>2, the sequential tournament's composed Lipschitz constant with respect
> to any single strength coordinate is bounded by `L * (N-1)` in the worst case
> (each pairwise comparison contributes at most L). However, in practice the
> tournament structure means later challengers face increasingly unfavorable
> odds, dampening the amplification. The `contraction_margin()` method reports
> the 2-memory bound; users should treat it as a NECESSARY (not sufficient)
> condition for N>2 stability. N>2 stability is validated empirically by the
> stability scenarios (STAB-1, STAB-2).

**Note on determinism:** This function uses a random draw (via Python's
`random.random()` by default). For reproducibility in tests, functions that call
`select_memory` should accept an optional `rng: random.Random` parameter.
Add `rng: Optional[random.Random] = None` to the function signature. If `rng` is
None, use the module-level `random` (matching markov_chain.py convention).

**Edge cases:**
- 0 memories: raise ValueError
- 1 memory: deterministic return of index 0
- All memories with identical scores: each pairwise soft_select gives 0.5
  probability, so any memory can win (uniform in expectation)

#### `step_dynamics(params: ParameterSet, memories: list[MemoryState], accessed_idx: Optional[int], feedback_signal: float, current_time: float) -> list[MemoryState]`

Advance the memory system by one time step.

**Algorithm:**
For each memory `m` at index `i`:

1. **Strength decay** (all memories): `decayed_strength = strength_decay(params.beta, m.strength, params.delta_t)`
   - Clamp to [0, s_max]: `decayed_strength = max(0.0, min(decayed_strength, params.s_max))`

2. **If `i == accessed_idx`** (this memory was accessed):
   a. Strength reinforcement: `new_strength = strength_update(params.alpha, decayed_strength, params.s_max)`
      - Clamp: `new_strength = max(0.0, min(new_strength, params.s_max))`
   b. Importance update: `new_importance = importance_update(m.importance, params.feedback_sensitivity, feedback_signal)`
      - importance_update already clamps to [0, 1]
   c. Access time reset: `new_last_access_time = 0.0`
   d. Access count increment: `new_access_count = m.access_count + 1`

3. **If `i != accessed_idx`** (not accessed):
   a. `new_strength = decayed_strength`
   b. `new_importance = m.importance` (unchanged)
   c. `new_last_access_time = m.last_access_time + params.delta_t`
   d. `new_access_count = m.access_count`

4. **Creation time update** (all memories): `new_creation_time = m.creation_time + params.delta_t`

5. **Relevance** (all memories): `new_relevance = m.relevance` (unchanged — relevance is exogenous, set at query time, not modified by dynamics)

6. Construct new MemoryState with updated fields.

> **ADVERSARIAL NOTE (finding 6 — ordering proof):** The sequential application
> "decay first, then reinforce" produces the same result as the Lean
> formalization's single-step formula `(1-alpha)*exp(-beta*dt)*S + alpha*S_max`
> (for q=1). Proof: `strength_update(alpha, decay(beta, S, dt), S_max)`
> = `(1-alpha) * S*exp(-beta*dt) + alpha*S_max` = Lean's
> `expectedStrengthUpdate(alpha, beta, dt, S_max, 1, S)`. This equivalence is
> algebraic. **WARNING:** The reverse order "reinforce then decay" is NOT
> equivalent and MUST NOT be used.

**When `accessed_idx is None`:** No memory was accessed this step. Only strength
decay and time advancement occur. This represents a timestep where the memory
system ticks but no retrieval happened.

**Postconditions (forward-invariance):**
- For every output memory: `0.0 <= strength <= params.s_max`
- For every output memory: `0.0 <= importance <= 1.0`
- For every output memory: `last_access_time >= 0.0`
- For every output memory: `creation_time >= 0.0`
- For every output memory: `access_count >= 0`
- For every output memory: `relevance == input_memory.relevance` (unchanged)

**Edge cases:**
- Empty memory list: return `[]`
- `accessed_idx` out of range: raise `IndexError`
- `accessed_idx is None`: pure decay step, no reinforcement

#### `simulate(params: ParameterSet, memories: list[MemoryState], n_steps: int, access_pattern: list[Optional[int]], feedback_signals: Optional[list[float]] = None, rng: Optional[random.Random] = None) -> SimulationResult`

Run a multi-step simulation with a prescribed access pattern.

**Parameters:**
- `access_pattern`: A list of length `n_steps`. Each element is either an
  integer index (which memory to access at that step) or `None` (no access).
  If an element is the sentinel value `-1`, the engine performs soft selection
  to choose which memory to access (using `select_memory`).
- `feedback_signals`: A list of length `n_steps` with the feedback signal for
  each step. If `None`, all feedback signals default to `1.0` (positive
  reinforcement).
- `rng`: Random number generator for `select_memory` calls.

**Algorithm:**
```
current_memories = list(memories)    # defensive copy
rankings_per_step = []
scores_per_step = []
strengths_per_step = []

for step in range(n_steps):
    current_time = float(step)  # normalize time to step count

    # 1. Score and rank
    ranked = rank_memories(params, current_memories, current_time)
    rankings_per_step.append([idx for idx, _ in ranked])
    scores_per_step.append([score_memory(params, m, current_time) for m in current_memories])
    strengths_per_step.append([m.strength for m in current_memories])

    # 2. Determine access
    access_idx = access_pattern[step]
    if access_idx == -1:
        access_idx = select_memory(params, current_memories, current_time, rng=rng)

    # 3. Get feedback signal
    signal = feedback_signals[step] if feedback_signals is not None else 1.0

    # 4. Advance dynamics
    current_memories = step_dynamics(params, current_memories, access_idx, signal, current_time)

return SimulationResult(
    rankings_per_step=rankings_per_step,
    scores_per_step=scores_per_step,
    strengths_per_step=strengths_per_step,
    final_memories=current_memories,
)
```

**Validation:**
- `len(access_pattern) == n_steps` — raise `ValueError` if not
- `feedback_signals is None or len(feedback_signals) == n_steps` — raise `ValueError` if not
- `n_steps >= 0` — 0 steps is valid (returns empty lists, final_memories = input)

**Edge cases:**
- 0 steps: return empty result lists, final_memories = input memories
- 0 memories: return empty result lists at each step
- All access_pattern entries are None: pure decay simulation

---

## Part 2: `sensitivity.py` — Sensitivity Analysis

File: `proofs/hermes-memory/python/hermes_memory/sensitivity.py`

### 2.1 Imports

```python
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
```

### 2.2 Scenario Dataclass

```python
@dataclass(frozen=True)
class Scenario:
    """A test scenario for evaluating memory system behavior.

    Each scenario defines a set of memories, a query time, and the
    expected winner — the memory index that SHOULD rank first under
    correct behavior.

    Attributes:
        name:             Human-readable scenario name.
        memories:         List of MemoryState instances in the scenario.
        current_time:     The time at which to evaluate rankings.
        expected_winner:  Index into `memories` of the memory that should
                          rank first. Use -1 as a sentinel for stability
                          scenarios that have no single expected winner.
                          The sensitivity analysis checks whether
                          perturbations flip this winner.
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
                f"competency must be one of {valid_competencies}, got {self.competency!r}"
            )
        # ADVERSARIAL FIX (finding 10): Allow expected_winner=-1 for stability
        # scenarios. The sentinel -1 means "no single expected winner — use
        # stability-specific evaluation instead."
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
```

### 2.3 Standard Scenario Suite

`build_standard_scenarios(params: ParameterSet) -> list[Scenario]`

Returns a list of 10 scenarios, 2-3 per competency. The scenarios use
`params.s_max` and `params.s0` to construct memory states that are within the
valid domain.

> **ADVERSARIAL NOTE (finding 8 — competency overlap):** The four competency
> categories are not orthogonal experimental conditions. They represent
> different qualitative aspects of system behavior that share underlying
> mechanisms (scoring weights, strength decay, soft selection). A parameter
> that appears "insensitive" within one category may still indirectly affect
> that category through shared mechanisms. The classification should be
> interpreted as "primary sensitivity channel," not "exclusively affects this
> category."

#### Accurate Retrieval Scenarios (3)

**AR-1: "High relevance beats high recency"**
- Memory 0: relevance=0.95, last_access_time=50.0, importance=0.3, access_count=2, strength=s0, creation_time=100.0
- Memory 1: relevance=0.4, last_access_time=1.0, importance=0.3, access_count=10, strength=s_max*0.8, creation_time=100.0
- Expected winner: 0 (relevance dominates when w1 > w2, which is enforced by w2 < 0.4 constraint)
- current_time: 0.0

**AR-2: "Relevance tiebreaker uses importance"**
- Memory 0: relevance=0.7, last_access_time=10.0, importance=0.9, access_count=5, strength=s0, creation_time=50.0
- Memory 1: relevance=0.7, last_access_time=10.0, importance=0.2, access_count=5, strength=s0, creation_time=50.0
- Expected winner: 0 (higher importance breaks the relevance tie)
- current_time: 0.0

**AR-3: "High relevance beats high activation"**
- Memory 0: relevance=0.9, last_access_time=20.0, importance=0.5, access_count=1, strength=s0, creation_time=80.0
- Memory 1: relevance=0.3, last_access_time=5.0, importance=0.5, access_count=100, strength=s_max*0.9, creation_time=80.0
- Expected winner: 0
- current_time: 0.0

#### Test-Time Learning Scenarios (2)

**TTL-1: "New high-relevance memory enters established pool"**
- Memory 0: relevance=0.9, last_access_time=0.0, importance=0.5, access_count=0, strength=s0, creation_time=0.0 (brand new)
- Memory 1: relevance=0.5, last_access_time=5.0, importance=0.7, access_count=20, strength=s_max*0.7, creation_time=200.0
- Memory 2: relevance=0.4, last_access_time=3.0, importance=0.6, access_count=15, strength=s_max*0.6, creation_time=150.0
- Expected winner: 0 (high relevance + novelty bonus for creation_time=0)
- current_time: 0.0
- *NOTE (adversarial finding 8):* This scenario tests "novelty bonus + relevance
  synergy," not novelty in isolation. Memory 0 would likely win on relevance
  alone (0.9 vs 0.5 and 0.4). TTL-2 below tests novelty bonus in isolation.

**TTL-2: "Novelty bonus saves zero-base-score new memory"**
- Memory 0: relevance=0.0, last_access_time=0.0, importance=0.0, access_count=0, strength=s0, creation_time=0.0 (brand new, zero relevance)
- Memory 1: relevance=0.3, last_access_time=20.0, importance=0.3, access_count=5, strength=s0*0.5, creation_time=100.0
- Expected winner: 0 (novelty_bonus(N0, gamma, 0) = N0 > survival_threshold > base_score of memory 1, assuming N0 is large enough relative to base scores — this is the cold-start survival theorem)
- current_time: 0.0
- Note: This scenario is only valid when novelty_start is sufficiently large
  (N0 > the base score of memory 1). build_standard_scenarios MUST verify this
  precondition and skip the scenario if the params make it degenerate.

#### Selective Forgetting Scenarios (3)

**SF-1: "Stale heavily-accessed memory loses to fresh relevant one"**
- Memory 0: relevance=0.8, last_access_time=5.0, importance=0.6, access_count=3, strength=s_max*0.5, creation_time=20.0
- Memory 1: relevance=0.5, last_access_time=200.0, importance=0.8, access_count=100, strength=s_max*0.3, creation_time=500.0 (old, decayed)
- Expected winner: 0 (strength decay has reduced memory 1's recency to near-zero)
- current_time: 0.0

**SF-2: "Anti-lock-in: established memory does not dominate forever"**
- Memory 0: relevance=0.7, last_access_time=2.0, importance=0.5, access_count=5, strength=s0, creation_time=10.0
- Memory 1: relevance=0.65, last_access_time=1.0, importance=0.9, access_count=1000, strength=s_max*0.95, creation_time=1000.0
- Expected winner: 0 (even though memory 1 has massive access_count and high
  strength, memory 0 has meaningfully higher relevance and recency. The
  anti-lock-in mechanism via strength decay and the w2 < 0.4 cap prevent memory
  1 from winning on recency alone.)
- current_time: 0.0

**SF-3: "Low-importance memory with low relevance loses"**
- Memory 0: relevance=0.6, last_access_time=10.0, importance=0.5, access_count=3, strength=s0, creation_time=30.0
- Memory 1: relevance=0.15, last_access_time=100.0, importance=0.1, access_count=50, strength=s_max*0.2, creation_time=500.0
- Expected winner: 0
- current_time: 0.0

#### Stability Scenarios (2)

**STAB-1: "No monopolization over 100 steps"**
- This is a SIMULATION scenario, not a single-point ranking scenario.
- 4 memories with moderate, varied relevances (0.5, 0.55, 0.6, 0.45), similar
  strengths (s0), importance (0.5), creation_time (50.0), last_access_time (10.0),
  access_count (5).
- Run simulate() for 100 steps with access_pattern = [-1] * 100 (soft selection
  each step).
- Expected winner: -1 (no single expected winner — instead, check that no
  single memory appears at rank 0 in more than 60% of steps)
- Stability criterion: max_fraction_at_rank_0 < 0.60 for any single memory
- current_time: 0.0

**STAB-2: "Strength convergence to steady state"**
- 2 memories with equal relevance (0.5), equal everything else.
- Run simulate() for 200 steps with access_pattern = [-1] * 200.
- Expected winner: -1 (check that final strengths are within 20% of
  steady_state_strength for the given params)
- current_time: 0.0

**Note on stability scenarios:** These two scenarios use a different evaluation
mechanism than the "expected_winner" check. The `analyze_parameter` function
MUST handle `expected_winner == -1` as a signal to apply the stability-specific
evaluation criteria instead of the simple "who ranks first" check.

### 2.4 SensitivityResult Dataclass

```python
@dataclass(frozen=True)
class SensitivityResult:
    """Result of sensitivity analysis for a single parameter.

    NOTE (adversarial finding 7): kendall_tau_per_level and
    score_change_per_level are AVERAGED across non-stability scenarios.
    Individual scenario breakdowns are available in
    scenario_flipped_per_level. Averaging can mask scenario-specific
    effects — a parameter that dominates one scenario but is neutral
    elsewhere will show moderate average metrics.

    NOTE (adversarial finding 12): Weight parameters (w1-w4) are subject
    to OAT violation due to sum-to-1 renormalization. Their results
    reflect the combined effect of renormalization, not isolated
    parameter sensitivity. Marked by is_weight=True.

    Attributes:
        parameter_name:              Name of the perturbed parameter.
        perturbation_levels:         List of multiplicative perturbation factors
                                     applied to the parameter. E.g., [-0.5, -0.25,
                                     -0.1, 0.1, 0.25, 0.5] means the parameter
                                     was multiplied by [0.5, 0.75, 0.9, 1.1,
                                     1.25, 1.5].
        kendall_tau_per_level:       Kendall's tau-b rank correlation between the
                                     baseline ranking and the perturbed ranking,
                                     for each perturbation level. Range: [-1, 1].
                                     1.0 = identical ranking. Computed over all
                                     non-stability scenarios.
        score_change_per_level:      Mean absolute score change across all memories
                                     and all non-stability scenarios, for each
                                     perturbation level.
        contraction_margin_per_level: The contraction_margin() of the perturbed
                                     ParameterSet, for each perturbation level.
                                     Negative means contraction is violated.
        scenarios_correct_per_level: Count of non-stability scenarios where the
                                     expected_winner still ranks first after
                                     perturbation, for each perturbation level.
        skipped_levels:              List of perturbation levels that were skipped
                                     because the perturbed parameter violated
                                     domain constraints (e.g., alpha would go
                                     negative or exceed 1). These levels appear
                                     in perturbation_levels but have NaN in all
                                     metric lists.
        scenario_flipped_per_level:  For each perturbation level, a list of
                                     scenario names whose expected_winner was
                                     flipped (no longer ranks first). Empty list
                                     for skipped levels.
                                     (adversarial finding 7)
        is_weight:                   True if this parameter is a scoring weight
                                     (w1, w2, w3, w4) subject to OAT violation
                                     from renormalization. (adversarial finding 12)
        classification:              One of "critical", "sensitive", "insensitive".
    """

    parameter_name: str
    perturbation_levels: list[float]
    kendall_tau_per_level: list[float]
    score_change_per_level: list[float]
    contraction_margin_per_level: list[float]
    scenarios_correct_per_level: list[int]
    skipped_levels: list[float]
    scenario_flipped_per_level: list[list[str]]
    is_weight: bool
    classification: str
```

### 2.5 Classification Rules

Given a SensitivityResult, the classification is determined by these rules
evaluated in order (first match wins):

> **ADVERSARIAL FIX (finding 14):** ALL classification rules now explicitly
> require "non-skipped" qualification. Skipped levels (where the perturbation
> violated domain constraints) have `scenarios_correct=0` and `NaN` metrics.
> Without the "non-skipped" filter, parameters near constraint boundaries would
> be misclassified as "sensitive" or "critical" due to invalid perturbation
> levels, not actual behavioral impact.

> **ADVERSARIAL NOTE (finding 12):** Weight parameters (w1-w4) are subject to
> OAT violation due to sum-to-1 renormalization. Their classifications should be
> interpreted as "sensitivity of the system to shifting weight balance
> toward/away from this component," not "sensitivity to this specific weight
> value."

**"critical"** — Any of:
- Any **non-skipped** perturbation level has `contraction_margin < 0` (contraction
  violated)
- Any **non-skipped** perturbation level has `scenarios_correct < total_scenarios / 2`
  (flips more than 50% of scenario winners)

**"sensitive"** — Any of:
- Any **non-skipped** perturbation level with `|factor| >= 0.25` has `scenarios_correct < 0.75 * total_scenarios` (flips more than 25% of winners at 25%+ perturbation)
- Any **non-skipped** perturbation level has `kendall_tau < 0.8`

**"insensitive"** — All of:
- All **non-skipped** perturbation levels have `scenarios_correct > 0.75 * total_scenarios`
- All **non-skipped** perturbation levels have `kendall_tau > 0.9`

Note: `total_scenarios` is the count of non-stability scenarios (those with
`expected_winner != -1`).

### 2.6 Functions

#### `perturb_parameter(params: ParameterSet, param_name: str, factor: float) -> Optional[ParameterSet]`

Create a new ParameterSet with one parameter scaled by `(1 + factor)`.

**Algorithm:**
1. Get current value: `current = getattr(params, param_name)`
2. Compute new value: `new_value = current * (1.0 + factor)`
3. Construct kwargs from all params fields, replacing `param_name` with
   `new_value`
4. Try `ParameterSet(**kwargs)` — if it raises `ValueError` (constraint
   violation), return `None`
5. Special handling for weight parameters (w1, w2, w3, w4):
   - When perturbing a weight, the other weights must be renormalized to
     maintain sum=1.
   - Algorithm: set `new_wi = current_wi * (1 + factor)`. Clamp to
     `[0, max_for_wi]` where max is 0.4 for w2, 1.0 for others. Distribute
     the residual `1.0 - new_wi` proportionally among the other three weights.
     If proportional distribution fails (other weights are all zero), return
     None.
   - Validate the resulting ParameterSet. If invalid, return None.
6. **Co-perturbation for cross-constrained parameters** *(adversarial finding 1):*
   - When perturbing `s_max`: also adjust `s0` to `min(s0, new_s_max * 0.99)`
     to keep the initial-condition constraint `s0 < s_max` satisfied.
   - When perturbing `novelty_start`: also adjust `survival_threshold` to
     `min(survival_threshold, new_novelty_start * 0.99)` to keep the
     `survival_threshold < novelty_start` constraint satisfied.
   - When perturbing `s0`: if `new_s0 >= s_max`, return None (do NOT
     co-adjust s_max, because that would change the dynamics domain).
   - Rationale: `s0` and `survival_threshold` are configuration parameters, not
     dynamics parameters. Adjusting them to maintain cross-constraints does not
     change the system's behavior — it only keeps the configuration consistent.
     This prevents false "skipped" entries when perturbing dynamics parameters
     like `s_max`.

**Parameters that can be perturbed:**
All 14 fields of ParameterSet: alpha, beta, delta_t, s_max, s0, temperature,
novelty_start, novelty_decay, survival_threshold, feedback_sensitivity, w1, w2,
w3, w4.

#### `analyze_parameter(params: ParameterSet, scenarios: list[Scenario], param_name: str, perturbation_levels: Optional[list[float]] = None) -> SensitivityResult`

Run sensitivity analysis for a single parameter.

**Default perturbation levels:** `[-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]`

**Algorithm:**
```
baseline_rankings = {}    # scenario_name -> list of (idx, score)
baseline_scores = {}      # scenario_name -> list of scores (one per memory)

# 1. Compute baselines
for scenario in scenarios:
    if scenario.expected_winner == -1:
        continue    # stability scenarios handled separately
    ranked = rank_memories(params, scenario.memories, scenario.current_time)
    baseline_rankings[scenario.name] = ranked
    baseline_scores[scenario.name] = [
        score_memory(params, m, scenario.current_time)
        for m in scenario.memories
    ]

non_stability_scenarios = [s for s in scenarios if s.expected_winner != -1]
total_scenarios = len(non_stability_scenarios)

# 2. For each perturbation level
kendall_taus = []
score_changes = []
contraction_margins = []
scenarios_correct_counts = []
scenario_flipped = []          # ADVERSARIAL ADDITION (finding 7)
skipped = []

for factor in perturbation_levels:
    perturbed = perturb_parameter(params, param_name, factor)

    if perturbed is None:
        # Constraint violated — mark as skipped
        skipped.append(factor)
        kendall_taus.append(float('nan'))
        score_changes.append(float('nan'))
        contraction_margins.append(float('nan'))
        scenarios_correct_counts.append(0)
        scenario_flipped.append([])    # ADVERSARIAL ADDITION (finding 7)
        continue

    contraction_margins.append(perturbed.contraction_margin())

    # 2a. Evaluate each non-stability scenario
    all_taus = []
    all_score_deltas = []
    correct_count = 0
    flipped_names = []             # ADVERSARIAL ADDITION (finding 7)

    for scenario in non_stability_scenarios:
        perturbed_ranked = rank_memories(perturbed, scenario.memories, scenario.current_time)
        perturbed_scores = [
            score_memory(perturbed, m, scenario.current_time)
            for m in scenario.memories
        ]

        # Kendall tau between baseline ranking order and perturbed ranking order
        baseline_order = [idx for idx, _ in baseline_rankings[scenario.name]]
        perturbed_order = [idx for idx, _ in perturbed_ranked]
        tau = _kendall_tau_b(baseline_order, perturbed_order)
        all_taus.append(tau)

        # Score change
        base_sc = baseline_scores[scenario.name]
        delta = sum(abs(a - b) for a, b in zip(base_sc, perturbed_scores)) / len(base_sc)
        all_score_deltas.append(delta)

        # Winner check
        if perturbed_ranked[0][0] == scenario.expected_winner:
            correct_count += 1
        else:
            flipped_names.append(scenario.name)   # ADVERSARIAL ADDITION

    kendall_taus.append(sum(all_taus) / len(all_taus) if all_taus else float('nan'))
    score_changes.append(sum(all_score_deltas) / len(all_score_deltas) if all_score_deltas else float('nan'))
    scenarios_correct_counts.append(correct_count)
    scenario_flipped.append(flipped_names)         # ADVERSARIAL ADDITION

# 3. Classify
classification = _classify(kendall_taus, scenarios_correct_counts,
                           contraction_margins, skipped, perturbation_levels,
                           total_scenarios)

# 4. Determine is_weight flag (adversarial finding 12)
is_weight = param_name in {"w1", "w2", "w3", "w4"}

return SensitivityResult(...)
```

**Kendall's tau-b implementation** (`_kendall_tau_b`):
Implement directly (avoid scipy dependency for this single function). Kendall's
tau-b counts concordant and discordant pairs:
- For two rankings r1 and r2 (both permutations of 0..n-1):
- concordant: pair (i,j) where r1 ranks i before j AND r2 ranks i before j
- discordant: pair (i,j) where r1 ranks i before j AND r2 ranks j before i
- tau_b = (concordant - discordant) / sqrt((n*(n-1)/2) * (n*(n-1)/2))
  = (concordant - discordant) / (n*(n-1)/2) for permutations (no ties)

For 2-element rankings, tau is either +1 (same order) or -1 (flipped).
For 1-element rankings, tau is defined as 1.0 (trivially identical).

#### `run_sensitivity_analysis(params: ParameterSet, perturbation_levels: Optional[list[float]] = None) -> dict[str, SensitivityResult]`

Run OAT sensitivity analysis across all parameters.

**Algorithm:**
1. `scenarios = build_standard_scenarios(params)`
2. `results = {}`
3. For each parameter name in ParameterSet's fields (in declaration order):
   `results[name] = analyze_parameter(params, scenarios, name, perturbation_levels)`
4. Return `results`

**Parameters analyzed:** All 14: alpha, beta, delta_t, s_max, s0, temperature,
novelty_start, novelty_decay, survival_threshold, feedback_sensitivity, w1, w2,
w3, w4.

#### `generate_report(results: dict[str, SensitivityResult]) -> str`

Generate a human-readable markdown report.

**Format:**
```markdown
# Sensitivity Analysis Report

## Summary

| Parameter | Classification | Min Kendall Tau | Min Scenarios Correct | Min Contraction Margin | Weight* |
|-----------|---------------|-----------------|----------------------|----------------------|---------|
| alpha     | critical      | 0.67            | 3/8                  | -0.02                |         |
| beta      | insensitive   | 0.95            | 8/8                  | 0.15                 |         |
| w1        | sensitive     | 0.80            | 6/8                  | 0.04                 | *       |
| ...       | ...           | ...             | ...                  | ...                  |         |

\* Weight parameters are renormalized when perturbed (OAT assumption violated).
Results reflect sensitivity to shifting weight balance, not isolated parameter change.

## Critical Parameters
[Details for each critical parameter: which scenarios flip, at what perturbation]
[Include scenario_flipped_per_level breakdown]

## Sensitive Parameters
[Details for each sensitive parameter]
[Include scenario_flipped_per_level breakdown]

## Insensitive Parameters
[Brief listing]

## Contraction Safety
[Which parameters can push the system past the contraction boundary, and at what perturbation level]
```

---

## Part 3: Hard Constraints Reference

This section consolidates ALL constraints from the Lean formalization and
MEMORY_SYSTEM.md that the implementation MUST respect. These are not suggestions
— violations produce mathematically undefined behavior.

### 3.1 Parameter Domain Constraints

| Parameter | Domain | Lean Source | Theorem(s) |
|-----------|--------|-------------|------------|
| alpha (α) | (0, 1) | hα₀ : 0 < α, hα₁ : α < 1 | strength_increases, combinedFactor_lt_one |
| beta (β) | (0, +inf) | hβ : 0 < β | strengthDecay_pos, steadyState_lt_Smax |
| delta_t (Δ) | (0, +inf) | hΔ : 0 < Δ | combinedFactor_lt_one, steadyState_lt_Smax |
| s_max (Smax) | (0, +inf) | hSmax : 0 < Smax | strength_le_max, fixedPoint_lt_Smax |
| s0 (S₀) | (0, s_max) | 0 ≤ S₀ ≤ Smax | composedDomain_contains |
| temperature (T) | (0, +inf) | hT : 0 < T, hT : T ≠ 0 | softSelect_pos, softSelect_complementary |
| novelty_start (N₀) | (0, +inf) | hN₀ : 0 < N₀ | noveltyBonus_pos, coldStart_survival |
| novelty_decay (γ) | (0, +inf) | hγ : 0 < γ | explorationWindow_pos |
| survival_threshold (ε) | (0, N₀) | hε : 0 < ε, hεN : ε < N₀ | coldStart_survival |
| w1 | [0, 1] | w_i ≥ 0 | score_nonneg |
| w2 | [0, 0.4) | w_rec < 0.5, extended to < 0.4 | §4.4.1 composed scoring contraction |
| w3 | [0, 1] | w_i ≥ 0 | score_nonneg |
| w4 | [0, 1] | w_i ≥ 0 | score_nonneg |
| Σwi | = 1 | Σw_i = 1 | ScoringWeights.__post_init__ |

### 3.2 Contraction Condition

The central parameter constraint for system stability:

```
L * α * Smax < 1 - exp(-β * Δ)
```

where `L = 0.25 / T` is the Lipschitz constant of soft_select w.r.t. score.

**Equivalently:** `K = exp(-β * Δ) + L * α * Smax < 1`

**Derivation (from MEMORY_SYSTEM.md §4.3-4.4):**
1. The composed expected map T has two Lipschitz contributions:
   - In strength S: factor `exp(-β*Δ)` (from `expectedUpdate_lipschitz_in_S`)
   - In selection probability q: factor `α * Smax` (from `expectedUpdate_lipschitz_in_q`)
2. Soft selection q = σ((s1-s2)/T) has Lipschitz constant L w.r.t. each
   strength coordinate, where L ≤ 0.25/T (since σ' ≤ 0.25 and the argument
   is scaled by 1/T).
3. By triangle inequality: full Lipschitz constant K = exp(-β*Δ) + L * α * Smax
4. Banach fixed-point theorem requires K < 1 for contraction.

**Lean theorem:** `composedContractionFactor_lt_one` (ComposedSystem.lean:196)

**Interpretation:** The "gap" created by strength decay (`1 - exp(-β*Δ)`) must
exceed the "amplification" from selection feedback (`L * α * Smax`). When this
holds:
- Unique fixed point S* exists in [0, Smax]^2
- All trajectories converge to S*
- S* satisfies all three safety guarantees

**Extended contraction (§4.4.1):** When selection uses the composite score
(not raw strength), the effective Lipschitz constant picks up chain-rule
factors from the scoring weights. The w2 < 0.4 constraint ensures this
extended K still < 1.

> **ADVERSARIAL NOTE (finding 4 — N>2 caveat):** The formal contraction proof
> (Banach fixed-point with K < 1) applies to the 2-memory system. Extension
> to N>2 memories via the sequential tournament is NOT formally proven. The
> tournament's effective Lipschitz constant in the N-memory space is at most
> `K_N = exp(-beta*dt) + L*(N-1)*alpha*s_max` in the worst case, but the
> dampening effect of the tournament (later challengers face worse odds) likely
> makes the effective constant much smaller. The stability scenarios (STAB-1,
> STAB-2) validate N>2 behavior empirically. `contraction_margin()` reports the
> 2-memory bound as a necessary condition.

### 3.3 Forward-Invariance Constraints

The domain [0, Smax] for strength and [0, 1] for importance are
forward-invariant under the dynamics:

| Invariant | Lean Source | Scope |
|-----------|-------------|-------|
| S' >= 0 when S >= 0 | expectedStrengthUpdate_nonneg | strength domain |
| S' <= Smax when S <= Smax | expectedStrengthUpdate_le_Smax | strength domain |
| imp' in [0, 1] | importanceUpdate_mem_Icc (via clamp01) | importance domain |
| base_score in [0, 1] | score_mem_Icc | **base** score only |
| boosted_score in [0, 1 + N0] | engine.score_memory | **boosted** score (base + novelty) |

> **ADVERSARIAL FIX (finding 3):** The previous version listed "score in [0, 1]"
> without distinguishing base score from boosted score. The Lean theorem
> `score_mem_Icc` proves the **base** score (w1*rel + w2*rec + w3*imp + w4*sig(act))
> is in [0, 1]. The engine's `score_memory` returns `base + novelty_bonus`, which
> is in [0, 1+N0]. These are different functions with different ranges. DO NOT
> clamp the boosted score to [0, 1].

The engine's `step_dynamics` MUST preserve these invariants by clamping outputs.

### 3.4 Monotonicity / Structural Constraints

| Property | Lean Source | Engine implication |
|----------|-------------|-------------------|
| sigmoid is monotone | sigmoid_monotone | Higher scores -> higher selection prob |
| soft_select is monotone in s1 | softSelect_monotone_score | Anti-thrashing |
| soft_select(s,s,T) = 0.5 | softSelect_equal_is_half | Equal memories get equal probability |
| P(A) + P(B) = 1 | softSelect_complementary | Selection is a proper probability |
| novelty decays to 0 | noveltyBonus_tendsto_zero | Bonus is transient, not permanent |
| retention decays with time | retention_antitone | Older access -> lower recency |
| retention increases with strength | retention_mono_strength | Higher strength -> slower forgetting |

---

## Part 4: Edge Cases

### 4.1 Zero Memories

| Function | Behavior |
|----------|----------|
| `score_memory` | N/A (single memory input) |
| `rank_memories([])` | Return `[]` |
| `select_memory([])` | Raise `ValueError("Cannot select from empty memory list")` |
| `step_dynamics([], ...)` | Return `[]` |
| `simulate([], ...)` | Return SimulationResult with empty lists at each step |
| `build_standard_scenarios` | All scenarios have >= 2 memories (never empty) |
| `analyze_parameter` with empty scenarios | Return result with NaN metrics |

### 4.2 Single Memory

| Function | Behavior |
|----------|----------|
| `rank_memories([m])` | Return `[(0, score)]` |
| `select_memory([m])` | Return `0` (deterministic, no tournament needed) |
| `step_dynamics([m], 0, ...)` | Normal dynamics on single memory |
| Kendall tau on 1-element ranking | 1.0 (trivially identical) |

### 4.3 All Identical Scores

When all memories produce identical scores:

| Function | Behavior |
|----------|----------|
| `rank_memories` | Return indices in ascending order (0, 1, 2, ...) per tie-breaking policy |
| `select_memory` | Each pairwise soft_select gives 0.5 (equal_is_half), so selection is approximately uniform via random draws |
| Kendall tau | 1.0 if tie-breaking produces same order, which it will (deterministic tie-breaking by index) |

### 4.4 Parameters at Constraint Boundaries

| Boundary | Parameter | Behavior |
|----------|-----------|----------|
| alpha -> 0+ | alpha | Combined factor -> exp(-beta*dt), contraction easier. Strength barely increases on access. |
| alpha -> 1- | alpha | Strength jumps to s_max on access. Combined factor -> 0. |
| beta -> 0+ | beta | Decay vanishes, contraction harder (exp(-beta*dt) -> 1). Approaches lock-in. |
| temperature -> 0+ | temperature | L = 0.25/T -> inf, contraction breaks. Selection becomes deterministic. |
| temperature -> inf | temperature | L -> 0, contraction easy. Selection becomes uniform (all 0.5). |
| w2 -> 0.4- | w2 | Near the composed-scoring contraction boundary. |
| s_max -> 0+ | s_max | Shrinks the domain. All memories have near-zero strength. |
| novelty_start -> threshold+ | novelty_start | Exploration window -> 0 (ln(N0/eps)/gamma -> 0). Cold-start barely helps. |

### 4.5 Perturbation Constraint Violations

When `perturb_parameter` would produce an invalid parameter:

| Example | Result |
|---------|--------|
| alpha * 1.5 when alpha = 0.8 -> 1.2 (> 1) | Return None |
| alpha * 0.5 when alpha = -0.1 (impossible, but factor=-1.5 on alpha=0.1 -> -0.05) | Return None |
| w2 * 1.5 when w2 = 0.3 -> 0.45 (>= 0.4) | Return None |
| beta * -0.5 when beta = 0.1 -> beta=0.15 (valid, factor is multiplicative: beta*(1-0.5)=0.05) | Actually: factor=-0.5 means multiply by 0.5, so beta=0.05. Still valid (> 0). Return ParameterSet. |
| temperature * -0.99 -> temperature * 0.01 | Valid but L = 0.25/0.01 = 25.0, contraction almost certainly violated |
| survival_threshold perturbed above novelty_start | Return None |
| s0 perturbed above s_max | Return None |
| s_max * 0.5 when s0 = 1.0, s_max = 10.0 -> s_max=5.0, s0 co-adjusted to min(1.0, 4.95) = 1.0 | Return valid ParameterSet (co-perturbation rule, adversarial finding 1) |

**Policy:** perturb_parameter MUST never crash. It catches ValueError from
ParameterSet construction and returns None. The analyze_parameter function
records skipped levels and continues.

---

## Part 5: Test Plan

File: `proofs/hermes-memory/python/tests/test_engine.py`
File: `proofs/hermes-memory/python/tests/test_sensitivity.py`

### 5.1 Engine Tests (`test_engine.py`)

#### ParameterSet validation
- `test_valid_params_construct`: Known-good params construct without error
- `test_alpha_out_of_range`: alpha <= 0 or >= 1 raises ValueError
- `test_beta_nonpositive`: beta <= 0 raises ValueError
- `test_w2_too_large`: w2 >= 0.4 raises ValueError
- `test_weights_not_sum_one`: w1+w2+w3+w4 != 1 raises ValueError
- `test_s0_exceeds_smax`: s0 >= s_max raises ValueError
- `test_survival_exceeds_novelty`: survival_threshold >= novelty_start raises ValueError
- `test_satisfies_contraction`: Known contracting params return True
- `test_violates_contraction`: Known non-contracting params return False
- `test_contraction_margin_positive`: Margin matches manual computation

#### MemoryState validation
- `test_valid_memory_construct`: Known-good memory constructs
- `test_negative_strength`: strength < 0 raises ValueError
- `test_relevance_out_of_range`: relevance > 1 or < 0 raises ValueError
- `test_negative_access_count`: access_count < 0 raises ValueError

#### score_memory
- `test_score_memory_range`: Score in [0, 1 + novelty_start]
- `test_score_memory_uses_core_score` (property): For memories with creation_time large enough that novelty_bonus ~ 0, score_memory ~ core.score
- `test_score_memory_novelty_boost`: Brand-new memory (creation_time=0) gets full novelty_start added
- `test_score_memory_zero_strength_guard`: Memory with strength=0 does not crash (division by zero guard)

#### rank_memories
- `test_rank_empty`: Empty list returns empty list
- `test_rank_single`: Single memory returns [(0, score)]
- `test_rank_descending`: Returned scores are non-increasing
- `test_rank_permutation`: Indices are a permutation of range(n)
- `test_rank_tiebreak_by_index`: Equal scores produce ascending indices
- `test_rank_higher_relevance_first`: Memory with higher relevance ranks higher (all else equal)

#### select_memory
- `test_select_empty_raises`: Empty list raises ValueError
- `test_select_single_deterministic`: Single memory always returns 0
- `test_select_favors_higher_score` (statistical): Over 1000 trials, higher-scored memory selected more often
- `test_select_reproducible_with_rng`: Same seed produces same selection

#### step_dynamics
- `test_step_preserves_strength_bounds`: 0 <= strength <= s_max after step
- `test_step_preserves_importance_bounds`: 0 <= importance <= 1 after step
- `test_step_accessed_memory_strength_increases`: Accessed memory's strength >= its decayed strength
- `test_step_unaccessed_strength_decays`: Non-accessed memory's strength <= original
- `test_step_accessed_resets_access_time`: last_access_time = 0 for accessed memory
- `test_step_increments_access_count`: access_count += 1 for accessed
- `test_step_advances_creation_time`: creation_time increases by delta_t
- `test_step_none_access_pure_decay`: No access -> all strengths decay
- `test_step_empty_list`: Empty input returns empty output

#### simulate
- `test_simulate_zero_steps`: Returns empty lists, final_memories = input
- `test_simulate_length_consistency`: All per-step lists have length n_steps
- `test_simulate_scores_bounded`: All scores in [0, 1 + novelty_start]
- `test_simulate_strengths_bounded`: All strengths in [0, s_max]
- `test_simulate_access_pattern_length_mismatch`: Raises ValueError

### 5.2 Sensitivity Tests (`test_sensitivity.py`)

#### Scenario construction
- `test_build_standard_scenarios_count`: Returns ~10 scenarios
- `test_build_standard_scenarios_valid`: All scenarios have valid MemoryState, valid competency
- `test_build_standard_scenarios_covers_competencies`: All 4 competencies represented
- `test_scenario_invalid_competency`: Invalid competency raises ValueError
- `test_scenario_invalid_winner`: expected_winner out of range raises ValueError
- `test_scenario_stability_sentinel_accepted`: Scenario with expected_winner=-1 and competency="stability" constructs without error
- `test_scenario_sentinel_wrong_competency_raises`: Scenario with expected_winner=-1 and non-stability competency raises ValueError

#### perturb_parameter
- `test_perturb_valid`: Valid perturbation produces new ParameterSet
- `test_perturb_violating_returns_none`: Factor that breaks constraint returns None
- `test_perturb_weight_renormalization`: Perturbing w1 renormalizes w2,w3,w4 to sum=1
- `test_perturb_zero_factor`: factor=0.0 produces identical params
- `test_perturb_all_parameters`: Every parameter name is accepted
- `test_perturb_smax_coadjusts_s0`: Perturbing s_max downward co-adjusts s0 to stay valid
- `test_perturb_novelty_start_coadjusts_threshold`: Perturbing novelty_start downward co-adjusts survival_threshold

#### analyze_parameter
- `test_analyze_returns_correct_structure`: Result has expected fields, correct lengths
- `test_analyze_skips_invalid_perturbations`: Skipped levels marked with NaN
- `test_analyze_baseline_is_perfect`: factor=0 (if included) gives tau=1.0, all scenarios correct
- `test_analyze_scenario_flipped_populated`: scenario_flipped_per_level contains correct scenario names
- `test_analyze_weight_marked`: Weight parameters have is_weight=True

#### Classification
- `test_classify_critical_contraction`: Params near contraction boundary classify temperature as critical
- `test_classify_insensitive`: For well-separated params, beta (large) is insensitive
- `test_classify_sensitive`: Intermediate perturbation impact -> sensitive
- `test_classify_skipped_not_counted`: Skipped levels do not trigger classification thresholds

#### run_sensitivity_analysis
- `test_full_analysis_returns_all_params`: Result dict has all 14 parameter names
- `test_full_analysis_classifications_valid`: All classifications in {"critical", "sensitive", "insensitive"}

#### generate_report
- `test_report_is_markdown`: Output contains markdown table headers
- `test_report_mentions_all_params`: Every parameter appears in the report
- `test_report_flags_weight_parameters`: Weight parameters marked with footnote

### 5.3 Reference Parameter Set for Tests

Tests should use a known-good parameter set that satisfies contraction:

```python
REFERENCE_PARAMS = ParameterSet(
    alpha=0.1,
    beta=0.1,
    delta_t=1.0,
    s_max=10.0,
    s0=1.0,
    temperature=5.0,
    novelty_start=0.3,
    novelty_decay=0.1,
    survival_threshold=0.05,
    feedback_sensitivity=0.1,
    w1=0.35,
    w2=0.20,
    w3=0.25,
    w4=0.20,
)
# K = exp(-0.1) + (0.25/5)*0.1*10 = 0.9048 + 0.05 = 0.9548 < 1
# Contraction margin: 1 - 0.9548 = 0.0452
```

This matches the parameters used in the existing stochastic verification suite
(MEMORY_SYSTEM.md §6.2), ensuring consistency across the test infrastructure.

---

## Part 6: Implementation Phases

### Phase 1: ParameterSet + MemoryState (Foundation)

**Files to create:**
- `proofs/hermes-memory/python/hermes_memory/engine.py` — dataclass definitions only

**Acceptance:**
- [ ] ParameterSet validates all 16 constraints from §3.1
- [ ] ParameterSet.satisfies_contraction() matches manual K computation
- [ ] ParameterSet.contraction_margin() = 1 - K
- [ ] MemoryState validates all field constraints
- [ ] Both are frozen dataclasses
- [ ] All ValueError messages are descriptive

**Estimated effort:** Small

### Phase 2: Engine Functions (Core Logic)

**Files to modify:**
- `proofs/hermes-memory/python/hermes_memory/engine.py` — add functions

**Dependencies:** Phase 1

**Acceptance:**
- [ ] score_memory calls core.py primitives (no reimplementation)
- [ ] score_memory does NOT clamp boosted score to [0, 1] (adversarial finding 3)
- [ ] rank_memories produces strict total order
- [ ] select_memory uses soft_select pairwise tournament
- [ ] step_dynamics preserves all forward-invariance invariants
- [ ] step_dynamics uses decay-then-reinforce order (adversarial finding 6)
- [ ] simulate orchestrates correctly with access_pattern sentinel

**Estimated effort:** Medium

### Phase 3: Sensitivity Framework (Analysis)

**Files to create:**
- `proofs/hermes-memory/python/hermes_memory/sensitivity.py`

**Dependencies:** Phase 2

**Acceptance:**
- [ ] Scenario allows expected_winner=-1 for stability (adversarial finding 10)
- [ ] build_standard_scenarios returns 10 valid scenarios
- [ ] perturb_parameter handles constraint violations gracefully
- [ ] perturb_parameter co-adjusts s0/survival_threshold (adversarial finding 1)
- [ ] analyze_parameter produces correct Kendall tau and winner counts
- [ ] analyze_parameter populates scenario_flipped_per_level (adversarial finding 7)
- [ ] Classification rules use non-skipped filter consistently (adversarial finding 14)
- [ ] run_sensitivity_analysis covers all 14 parameters

**Estimated effort:** Medium

### Phase 4: Report + Integration

**Files to modify:**
- `proofs/hermes-memory/python/hermes_memory/sensitivity.py` — add generate_report

**Files to modify:**
- `proofs/hermes-memory/python/hermes_memory/__init__.py` — export new modules

**Dependencies:** Phase 3

**Acceptance:**
- [ ] generate_report produces valid markdown
- [ ] Report correctly groups by classification
- [ ] Report flags weight parameters (adversarial finding 12)
- [ ] Full pipeline: params -> scenarios -> analysis -> report runs end-to-end

**Estimated effort:** Small

### Phase 5: Tests

**Files to create:**
- `proofs/hermes-memory/python/tests/test_engine.py`
- `proofs/hermes-memory/python/tests/test_sensitivity.py`

**Dependencies:** Phase 4

**Acceptance:**
- [ ] All tests from §5.1 and §5.2 pass
- [ ] Property-based tests use Hypothesis where appropriate
- [ ] No test reimplements core.py functions
- [ ] Tests use REFERENCE_PARAMS as baseline
- [ ] Tests cover adversarial findings (sentinel -1, skipped classification, co-perturbation)

**Estimated effort:** Medium

---

## Part 7: Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Floating-point divergence from Lean exact math | Score comparisons produce wrong rankings at edge cases | Use epsilon tolerances in tie-breaking; document IEEE754 guards (following existing test pattern with `assume(beta*t < 700)`) |
| Novelty bonus makes all scores > 1 | Breaks intuition about "score in [0,1]" | Document clearly: base score in [0,1], boosted score in [0, 1+N0]. Do not clamp the boost. (adversarial finding 3) |
| Sequential tournament for N memories introduces ordering bias | Higher-ranked memories have structural advantage as incumbent | This is by design (highest score starts as winner). Document explicitly. Alternative: round-robin tournament is O(n^2). |
| Weight perturbation renormalization changes multiple parameters at once | OAT assumption (one-at-a-time) is violated for weights | Document that weight sensitivity results reflect the combined effect of renormalization. Mark weight analyses with is_weight flag and report footnote. (adversarial finding 12) |
| Stability scenarios need different evaluation than winner-check | Classification logic becomes complex with two evaluation modes | Use expected_winner == -1 sentinel cleanly; keep evaluation paths separate and well-documented. Scenario validation allows -1 only for stability competency. (adversarial finding 10) |
| Kendall tau undefined for 0 or 1 element rankings | Division by zero | Define tau = 1.0 for n <= 1 (trivially identical) |
| Skipped perturbation levels corrupt classification | Parameters near constraint boundaries falsely classified critical/sensitive | All classification rules use "non-skipped" filter explicitly. (adversarial finding 14) |
| ParameterSet conflates config (s0, survival_threshold) with dynamics | Perturbing s_max may invalidate s0, producing false skips | Co-perturbation rule: adjust s0 and survival_threshold when perturbing their dependent parameters. (adversarial finding 1) |
| N>2 contraction guarantee is empirical, not proven | contraction_margin() may give false sense of security for N>2 | Document 2-memory caveat. Validate N>2 empirically via stability scenarios. (adversarial finding 4) |
| retention(t, S~0) is discontinuous | Score jumps from 0 to 1 at t=0 when S~0 | Document the discontinuity. Scenarios avoid pathological edge case. (adversarial finding 13) |

---

## Part 8: Open Questions

- [ ] **Q1:** Should the engine support time-varying relevance (re-querying)?
  Current design: relevance is fixed at construction. For sensitivity analysis
  this is sufficient. For production use, a `with_relevance(new_rel)` method
  on MemoryState could support re-querying. **Decision: defer to production
  integration. Sensitivity analysis uses fixed relevance.**
  *(adversarial finding 2 adds: sensitivity results are conditional on
  fixed-query evaluation; this should be stated in any report that cites
  these results.)*

- [ ] **Q2:** Should the tournament in select_memory use a random permutation
  of challengers instead of rank-order? This would remove the structural
  advantage of high-ranked memories. **Decision: use rank-order. The advantage
  is intentional — it makes the tournament more likely to select the top-ranked
  memory, which aligns with the soft-selection design intent.**

- [ ] **Q3:** Should stability scenarios count toward classification metrics?
  Current design: no — stability scenarios use expected_winner=-1 and have
  separate evaluation criteria. They are not included in scenarios_correct
  counts or Kendall tau averages. **Decision: correct. Stability is a
  qualitative check, not a ranking comparison.**

---

## Success Criteria

1. `ParameterSet` enforces all 16 constraints from the Lean formalization,
   with no false accepts and no false rejects.
2. `engine.py` uses ONLY core.py primitives — verified by grep showing no
   `math.exp`, `math.log`, or manual sigmoid in engine.py (except in
   `contraction_margin` and `satisfies_contraction` which compute the
   contraction factor directly).
3. `step_dynamics` preserves forward-invariance: property-based test with
   Hypothesis over random params and memory states shows strength always
   in [0, s_max] and importance always in [0, 1].
4. `rank_memories` produces a strict total order on every call (no ties
   in the output).
5. `run_sensitivity_analysis` with REFERENCE_PARAMS completes without error
   and classifies `temperature` as critical or sensitive (since it directly
   controls the Lipschitz constant L = 0.25/T in the contraction condition).
6. `generate_report` produces a valid markdown document that a human can
   read and act on.
7. All existing 127 tests continue to pass (engine.py and sensitivity.py
   are additive — no modifications to core.py or markov_chain.py).
8. Scenario with `expected_winner=-1` constructs without error for stability
   competency *(adversarial finding 10)*.
9. Classification rules correctly ignore skipped levels — verified by test
   with a parameter near its constraint boundary *(adversarial finding 14)*.
10. Weight parameter sensitivity results are flagged with `is_weight=True`
    and the report includes a renormalization caveat *(adversarial finding 12)*.

---

## Appendix A: Adversarial Analysis Changelog

Applied from `adversarial-pass1.md` (Brenner method, 14 findings):

| Finding | Operator | Change | Sections Modified |
|---------|----------|--------|-------------------|
| 1 | Level-Split | Co-perturbation rule for cross-constrained params | 1.2, 2.6, 4.5, 5.2 |
| 2 | Level-Split | MemoryState docstring: query-dependent vs intrinsic | 1.3 |
| 3 | Paradox-Hunt | Explicit [0,1] vs [0,1+N0] range + no-clamp rule | 1.5, 3.3 |
| 4 | Paradox-Hunt | N>2 contraction caveat | 1.5, 3.2 |
| 5 | Level-Split | Parameter-function coupling matrix | 1.2 |
| 6 | Paradox-Hunt | Decay-then-reinforce ordering proof + warning | 1.5 |
| 7 | Level-Split | scenario_flipped_per_level field | 2.4, 2.6 |
| 8 | Level-Split | Competency overlap note + TTL-1 clarification | 2.3 |
| 9 | Paradox-Hunt | ScoringWeights assert vs ValueError note | 1.2 |
| 10 | Paradox-Hunt | **BUG FIX:** Scenario validation allows -1 sentinel | 2.2, 5.2 |
| 11 | Level-Split | current_time parameter documentation | 1.5 |
| 12 | Paradox-Hunt | is_weight flag + OAT caveat for weights | 2.4, 2.5, 2.6 |
| 13 | Paradox-Hunt | retention discontinuity documentation | 1.5 |
| 14 | Paradox-Hunt | **BUG FIX:** Non-skipped filter in all classification rules | 2.5 |

---

## Addendum B: Adversarial Pass 3 Corrections (Brenner Kernel)

Added: 2026-02-27
Source: `thoughts/shared/plans/memory-system/sensitivity/adversarial-pass3.md`
Operators: Exclusion-Test, Object-Transpose

This addendum documents corrections identified by Brenner Kernel adversarial
analysis (Pass 3/3). All items are mandatory for implementation.

### B.1 Scenario Corrections (CRITICAL -- blocks correctness)

Two scenarios have incorrect expected winners under REFERENCE_PARAMS. The
narrative intent is correct but the memory parameters do not instantiate it.

#### AR-1 Fix: "High relevance beats high recency"

**Problem:** Memory 0 has strength=s0=1.0 and last_access_time=50.0.
retention(50, 1) = exp(-50) ~ 0, so the recency component is dead. Memory 1
with retention(1, 8) = 0.88 wins by 0.008.

**Corrected parameters (REPLACE the original AR-1 in Section 2.3):**
- Memory 0: relevance=0.95, **last_access_time=5.0**, importance=0.3,
  access_count=2, **strength=s_max*0.3**, creation_time=100.0
- Memory 1: (unchanged) relevance=0.4, last_access_time=1.0, importance=0.3,
  access_count=10, strength=s_max*0.8, creation_time=100.0
- Expected winner: 0
- Verification: M0=0.6214, M1=0.5915, gap=+0.030
- Discriminative: flipped by w1 -50%, w2 +50%

**Rationale:** Reducing M0 last_access_time to 5.0 and increasing M0 strength
to s_max*0.3 gives retention(5, 3) = exp(-5/3) = 0.189, a non-negligible
recency contribution. The scenario now correctly tests that high relevance
dominates recency, with a margin small enough to be sensitive to weight
perturbations.

#### SF-2 Fix: "Anti-lock-in: established memory does not dominate forever"

**Problem:** Memory 1 has last_access_time=1.0 and strength=s_max*0.95=9.5.
retention(1, 9.5) = exp(-0.105) = 0.900 -- the memory was accessed extremely
recently and has near-maximum strength, so its recency is excellent. The spec
comment said "strength decay has reduced memory 1's recency to near-zero" but
this is incorrect: last_access_time controls recency, not creation_time.

**Corrected parameters (REPLACE the original SF-2 in Section 2.3):**
- Memory 0: (unchanged) relevance=0.7, last_access_time=2.0, importance=0.5,
  access_count=5, strength=s0, creation_time=10.0
- Memory 1: relevance=0.65, **last_access_time=20.0**, importance=0.9,
  access_count=1000, strength=s_max*0.95, creation_time=1000.0
- Expected winner: 0
- Verification: M0=0.7061, M1=0.6769, gap=+0.029
- Discriminative: flipped by w3 +50%

**Rationale:** Increasing M1 last_access_time to 20.0 gives retention(20, 9.5)
= exp(-2.1) = 0.122, properly representing a memory whose high strength cannot
compensate for staleness.

### B.2 Scenario Discriminativeness Enhancement

**Problem:** Of the 6 correct non-stability scenarios (after B.1 fixes), most
have score gaps so large that no +/-50% perturbation can flip the winner:

| Scenario | Gap | Flippable perturbations |
|----------|-----|------------------------|
| AR-2 | 26.1% | 0/36 |
| AR-3 | 7.1% | 3/36 |
| SF-1 | 21.7% | 0/36 |
| SF-3 | 48.7% | 0/36 |
| TTL-2 | ~58% | 0/36 |

**Mitigation:** Add 3 razor-thin margin scenarios where memories derive their
advantage from DIFFERENT scoring components, making the outcome weight-sensitive.
All verified numerically under REFERENCE_PARAMS.

**RT-1: "Relevance barely beats importance" (accurate_retrieval)**
- Memory 0: relevance=0.65, last_access_time=8.0, importance=0.40,
  access_count=5, strength=s_max*0.3, creation_time=200.0
- Memory 1: relevance=0.55, last_access_time=8.0, importance=0.52,
  access_count=5, strength=s_max*0.3, creation_time=200.0
- Expected winner: 0 (w1*(0.65-0.55) = 0.035 > w3*(0.52-0.40) = 0.030)
- Gap: +0.005
- Flipped by: w1 -50%, w1 -25%, w3 +25%, w3 +50%
- 2-memory scenario exercising w1-vs-w3 tradeoff

**RT-2: "Relevance barely beats recency" (accurate_retrieval)**
- Memory 0: relevance=0.65, last_access_time=12.0, importance=0.5,
  access_count=5, strength=s_max*0.8, creation_time=200.0
- Memory 1: relevance=0.55, last_access_time=8.0, importance=0.5,
  access_count=5, strength=s_max*0.8, creation_time=200.0
- Expected winner: 0 (w1*(0.10) > w2*(rec_8 - rec_12) where rec computed with
  retention at strength=s_max*0.8)
- Gap: ~+0.008
- Flipped by: w1 -50%, w1 -25%, w2 +50%
- 2-memory scenario exercising w1-vs-w2 tradeoff

**RT-3: "Novelty barely saves low-relevance new memory" (test_time_learning)**
- Memory 0: relevance=0.15, last_access_time=5.0, importance=0.3,
  access_count=2, strength=s0, creation_time=0.0 (brand new)
- Memory 1: relevance=0.90, last_access_time=5.0, importance=0.3,
  access_count=2, strength=s0, creation_time=200.0 (old)
- Expected winner: 0 (novelty_bonus(0.3, 0.1, 0) = 0.3 compensates for
  w1*(0.90-0.15) = 0.2625 relevance deficit, net +0.0375)
- Gap: +0.038
- Flipped by: w1 +25%, w1 +50%, w3 -50%, novelty_start -50%, novelty_start -25%
- Exercises novelty_start and novelty_decay sensitivity

**Note:** build_standard_scenarios MUST verify that each razor-thin scenario's
expected winner actually wins under the given params. If a razor-thin scenario
is invalid (winner flipped by construction), emit a warning and exclude it.

The standard suite now has **13 scenarios**: 3 AR + 3 RT + 2 TTL + 3 SF + 2 STAB.
Non-stability scenarios: 11.

### B.3 Scoring-Irrelevant Parameter Awareness

**Problem:** 8 of 14 parameters (alpha, beta, delta_t, s_max, s0, temperature,
feedback_sensitivity, survival_threshold) do not appear in the scoring formula.
For non-stability scenarios, these produce tau=1.0 and score_change=0.0 at
all perturbation levels. Their only signal is contraction_margin.

**Mitigations:**

1. `generate_report` MUST split the summary into three sections:
   - **Ranking Sensitivity:** w1, w2, w3, w4, novelty_start, novelty_decay.
     Evaluated by tau, score_change, and scenarios_correct.
   - **Stability Sensitivity:** alpha, beta, delta_t, s_max, temperature.
     Evaluated by contraction_margin_per_level.
   - **Inert Parameters:** s0, feedback_sensitivity, survival_threshold.
     Noted with explanation that they only affect multi-step dynamics.

2. `run_sensitivity_analysis` MUST emit a diagnostic warning if more than 8
   of 14 parameters have `max(score_change_per_level) < 1e-10` for all
   non-skipped levels.

3. Classification rules: a parameter with max(score_change) < 1e-10 and
   tau = 1.0 at all levels MUST be annotated "insensitive (not exercised by
   scenario suite)" rather than "insensitive (confirmed robust)". These are
   epistemically different.

### B.4 Parameter Correlation Detection

**Problem:** alpha and s_max produce identical contraction_margin_per_level
curves (both appear as product L*alpha*s_max in K). Similarly beta and delta_t
produce identical curves (both appear in exp(-beta*delta_t)).

**Mitigation:** Add a `detect_correlated_parameters` function:

```python
def detect_correlated_parameters(
    results: dict[str, SensitivityResult],
    threshold: float = 0.99,
) -> list[tuple[str, str, float]]:
    """Find parameter pairs with highly correlated contraction margins.

    Returns list of (param_a, param_b, pearson_r) tuples where |r| > threshold.
    Only considers non-skipped, non-NaN margin values.
    """
```

`generate_report` MUST call this and include a "Correlated Parameters" section
listing any pairs with |r| > 0.99.

### B.5 Classification Rule Refinement

When classifying based on contraction_margin < 0, the report MUST note that
this is a **stability** criticality, not a **ranking** criticality. A parameter
can be critical for stability but insensitive for ranking (e.g., temperature).

### B.6 Test Plan Additions

Add these tests to the test plan (Section 5.2):

- `test_ar1_corrected_winner`: Verify corrected AR-1 M0 wins (regression guard)
- `test_sf2_corrected_winner`: Verify corrected SF-2 M0 wins (regression guard)
- `test_razor_thin_scenarios_discriminative`: At least 2 RT scenarios are
  flipped by some weight perturbation at +/-50%
- `test_correlated_parameters_detected`: alpha/s_max flagged as correlated
- `test_scoring_irrelevant_params_zero_score_change`: alpha, beta, delta_t,
  s_max, feedback_sensitivity, survival_threshold show score_change < 1e-10
- `test_report_splits_ranking_stability`: Report contains "Ranking Sensitivity"
  and "Stability Sensitivity" sections

### B.7 Updated Scenario Count

build_standard_scenarios returns **13 scenarios**:

| Category | Count | Scenarios |
|----------|-------|-----------|
| Accurate Retrieval | 3 | AR-1 (fixed), AR-2, AR-3 |
| Razor-Thin Margin | 3 | RT-1, RT-2, RT-3 |
| Test-Time Learning | 2 | TTL-1, TTL-2 |
| Selective Forgetting | 3 | SF-1, SF-2 (fixed), SF-3 |
| Stability | 2 | STAB-1, STAB-2 |

Non-stability scenarios: 11. Of these, at least 5 should be flippable by some
weight perturbation at +/-50% (AR-1 fixed, SF-2 fixed, RT-1, RT-2, RT-3).

### B.8 Object-Transpose Assessments (No Spec Changes)

These were evaluated and determined to NOT require changes:

- **OAT vs Morris screening:** Keep OAT. Add correlation detection instead.
- **Kendall tau vs "did winner change":** For 2-memory scenarios, they are
  mathematically equivalent. Keep both metrics for now; RT-1 could use 3 memories
  if intermediate tau values are desired in a future revision.
- **Perturbation in margin coordinates:** Useful idea but adds complexity.
  Defer to follow-up.
- **Skip soft selection for sensitivity:** Already the case -- rank_memories
  uses sort, not tournament.
- **ParameterSet split:** Keep unified. Add scoring_params()/dynamics_params()
  accessor methods if needed.
- **Tournament vs sort for N>2:** Architecturally moot -- different functions
  for different purposes.
