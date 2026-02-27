# Adversarial Analysis Pass 1/3 — Brenner Epistemological Method

**Spec under review:** `thoughts/shared/plans/memory-system/sensitivity/spec.md`
**Date:** 2026-02-27
**Operators:** ⊘ Level-Split, ◊ Paradox-Hunt

---

## Finding 1: ⊘ Level-Split — ParameterSet conflates configuration and runtime state

### Confusion

`ParameterSet` is declared `frozen=True` and described as "complete parameter set
for the Hermes memory dynamics." It includes `s0` (initial strength for new memories),
which is a **configuration** value used only at memory creation time. It also includes
`alpha`, `beta`, `temperature`, etc., which are **runtime dynamics** parameters used
at every step.

The conflation becomes dangerous in `step_dynamics`: the function receives a
`ParameterSet` containing `s0`, but `s0` is never used inside `step_dynamics`. It is
only consumed when constructing initial `MemoryState` instances (in scenarios). Yet
`s0` participates in ParameterSet validation (`0 < s0 < s_max`), meaning a
perturbation of `s_max` can invalidate `s0` even though `s_max` perturbation has
nothing to do with initial conditions.

### Why it matters

When `perturb_parameter` scales `s_max` by `(1 - 0.5) = 0.5`, the new `s_max` might
become less than the unchanged `s0`, causing ParameterSet construction to fail. This
is recorded as a "skipped level" even though the perturbation is perfectly valid for
the dynamics — only the initial-condition constraint is violated. This produces false
"skipped" entries that artificially inflate the apparent sensitivity of `s_max`.

### Mitigation

**In `perturb_parameter`:** When perturbing `s_max`, also adjust `s0` to
`min(s0, new_s_max * 0.99)` to keep the initial-condition constraint satisfied.
Similarly, when perturbing `novelty_start`, also adjust `survival_threshold` to
`min(survival_threshold, new_novelty_start * 0.99)`. Document this as a
**co-perturbation rule** in the spec: "Parameters with cross-constraints are
co-adjusted to maintain validity of the configuration-level invariants."

**Added to spec section 2.6 (perturb_parameter).**

---

## Finding 2: ⊘ Level-Split — MemoryState conflates query-dependent and intrinsic fields

### Confusion

`MemoryState` contains both:
- **Intrinsic fields** (strength, importance, access_count, creation_time) — properties
  of the memory itself, evolving under dynamics
- **Query-dependent fields** (relevance) — a function of the *current query* and this
  memory's embedding, not an intrinsic property of the memory

The spec says "relevance is exogenous — set at query time, not modified by dynamics"
(section 1.3). Yet `MemoryState` is `frozen=True`, and `step_dynamics` copies
relevance unchanged into the new state. This means a `MemoryState` is really a
"(memory, query) pair snapshot" rather than a "memory snapshot."

### Why it matters

In the sensitivity analysis, all scenarios use fixed relevances. This is fine for
single-ranking tests. But the stability scenarios (STAB-1, STAB-2) run 100-200 step
simulations where relevance never changes. In a real system, relevance would change
per query. The sensitivity analysis measures "parameter sensitivity under fixed
queries" but calls it "parameter sensitivity" without qualification. A parameter that
is insensitive under fixed queries could be critical when relevance fluctuates.

Additionally, `step_dynamics` has a postcondition "relevance == input_memory.relevance"
which seems like an invariant but is actually a design choice (no re-querying). If
someone extends the engine to support re-querying, this postcondition becomes wrong.

### Mitigation

**Documentation clarification:** Add a note to section 1.3 and the MemoryState
docstring: "MemoryState represents a memory *evaluated against a specific query*.
The relevance field is not an intrinsic memory property — it is the cosine similarity
to the query that produced this snapshot. Sensitivity analysis results are conditional
on fixed-query evaluation; parameter sensitivity under query variation requires a
separate analysis." This is a scope-narrowing clarification, not a code change.

**Added to spec section 1.3.**

---

## Finding 3: ◊ Paradox-Hunt — Score range: [0,1] vs. [0, 1+N0]

### Contradiction

The spec states in three places:
1. Section 1.5 (score_memory): "the return value CAN exceed 1.0 due to the novelty
   bonus addition ... total is in [0, 1 + N0]"
2. Section 3.3 (Forward-Invariance): "score in [0, 1]" citing Lean `score_mem_Icc`
3. Section 1.2 (ParameterSet): "w1 + w2 + w3 + w4 = 1.0 (convex combination)"

The Lean theorem `score_mem_Icc` proves the *base score* is in [0, 1]. The engine's
`score_memory` function returns `base + novelty_bonus`, which is in [0, 1+N0]. These
are two different functions with different ranges, but the spec's "Forward-Invariance
Constraints" table (section 3.3) lists "score in [0, 1]" without distinguishing which
score.

### Why it matters

If an implementor reads section 3.3 and adds a clamp to [0,1] on `score_memory`'s
output (to enforce the "invariant"), the novelty bonus is killed for any memory whose
base score is already near 1.0. This is exactly the scenario described in the
Paradox-Hunt prompt: "clamping kills the bonus for already-high-scoring memories."

The soft_select function receives these scores as `s1, s2`. The Lipschitz constant
L = 0.25/T is derived assuming the inputs are scores. If scores can be in [0, 1+N0],
the argument to sigmoid is `(s1-s2)/T` where the maximum `|s1-s2|` is now `1+N0`
rather than `1`. This does not break the Lipschitz bound (sigmoid' <= 0.25 everywhere),
but it does mean soft_select can be more extreme than the [0,1] analysis suggests.

### Mitigation

1. **Section 3.3:** Rename the row to "base_score in [0, 1]" and add a separate row:
   "boosted_score in [0, 1 + N0] | engine.score_memory | Novelty bonus is additive,
   not clamped."
2. **Section 1.5 (score_memory):** Add explicit prohibition: "DO NOT clamp the return
   value to [0, 1]. The novelty bonus MUST be additive and unclamped. Clamping would
   violate the cold-start survival guarantee (coldStart_survival requires
   boosted_score >= survival_threshold even when base_score = 0)."
3. **Section 1.5 (soft_select usage):** Note that the Lipschitz bound L = 0.25/T
   holds regardless of input magnitude because sigmoid' <= 0.25 globally.

**Added to spec sections 1.5 and 3.3.**

---

## Finding 4: ◊ Paradox-Hunt — Sequential tournament extension for N>2 memories

### Contradiction

The Lean formalization proves soft_select properties for a 2-memory system:
- `softSelect_complementary`: P(A) + P(B) = 1
- `softSelect_monotone_score`: higher score -> higher P
- Anti-thrashing via Lipschitz continuity

The engine's `select_memory` extends this to N memories via a sequential tournament
(section 1.5). But the tournament is **not** a proper probability distribution over
N memories in the same sense as the 2-memory case:

- In the 2-memory case, P(A) + P(B) = 1 exactly.
- In the N-memory tournament, the probabilities do not sum to 1 through the
  soft_select mechanism. They sum to 1 trivially (it is a valid selection
  procedure), but the individual probabilities are complex products of conditional
  pairwise comparisons, not a single softmax.
- The anti-thrashing property (small score change -> small probability change) holds
  pairwise but the composed effect through the tournament is NOT proven to be
  Lipschitz with constant L = 0.25/T. Each round multiplies Lipschitz constants.

### Why it matters

The contraction analysis uses `K = exp(-beta*dt) + L*alpha*s_max` where `L = 0.25/T`.
This L comes from the 2-memory soft_select. With N>2 memories and a sequential
tournament, the effective Lipschitz constant of the selection probability with
respect to any single memory's strength could be larger than 0.25/T due to
composition of N-1 pairwise comparisons.

For the sensitivity analysis (which evaluates ranking stability, not contraction
convergence), this means the contraction margin reported by `satisfies_contraction()`
is an **optimistic** bound when N>2 memories are used in scenarios.

### Mitigation

1. **Section 1.5 (select_memory, Rationale):** Add: "CAVEAT: The contraction factor
   K = exp(-beta*dt) + L*alpha*s_max is proven for the 2-memory system only. For N>2,
   the sequential tournament's composed Lipschitz constant with respect to any single
   strength coordinate is bounded by L * (N-1) in the worst case (each pairwise
   comparison contributes at most L). However, in practice the tournament structure
   means later challengers face increasingly unfavorable odds, dampening the
   amplification. The contraction_margin() method reports the 2-memory bound; users
   should treat it as a necessary (not sufficient) condition for N>2 stability."
2. **Section 3.2 (Contraction Condition):** Add a subsection "N>2 caveat" documenting
   that the formal contraction proof applies to the 2-memory system and the extension
   to N>2 is empirical, validated by the stability scenarios (STAB-1, STAB-2) rather
   than proven.

**Added to spec sections 1.5 and 3.2.**

---

## Finding 5: ⊘ Level-Split — Engine conflates scoring (pure) with dynamics (mutation)

### Confusion

The engine module contains both:
- **Pure scoring functions**: `score_memory`, `rank_memories` — given a state, compute
  a number. No side effects, no state changes.
- **State mutation functions**: `step_dynamics`, `simulate` — advance the system,
  create new MemoryState instances.

These are logically separate concerns but live in the same module and share the same
`ParameterSet`. The scoring functions need `w1-w4`, `novelty_start`, `novelty_decay`,
`temperature`. The dynamics functions need `alpha`, `beta`, `delta_t`, `s_max`,
`feedback_sensitivity`. Only `simulate` needs both.

### Why it matters

For the sensitivity analysis, this conflation means "perturbing alpha" affects
dynamics but should NOT affect scoring. The spec correctly separates them (alpha
does not appear in `score_memory`). But the shared ParameterSet makes it easy to
accidentally leak dynamics parameters into scoring or vice versa. More concretely:
if someone adds a new scoring component that depends on `delta_t`, the sensitivity
analysis would not catch the coupling because it perturbs one parameter at a time
(OAT), not interaction effects.

This is a **design smell**, not a bug. The spec is correct as written. But the
finding should be documented.

### Mitigation

**Documentation only:** Add to section 1.2 a table showing which parameters are
consumed by which functions:

| Parameter | score_memory | step_dynamics | select_memory | simulate |
|-----------|:---:|:---:|:---:|:---:|
| w1-w4 | yes | - | via score | yes |
| novelty_start/decay | yes | - | via score | yes |
| temperature | - | - | yes | yes |
| alpha | - | yes | - | yes |
| beta | - | yes | - | yes |
| delta_t | - | yes | - | yes |
| s_max | - | yes | - | yes |
| s0 | - | - | - | - |
| feedback_sensitivity | - | yes | - | yes |
| survival_threshold | - | - | - | - |

This makes the parameter-function coupling explicit and helps sensitivity analysis
interpretation.

**Added to spec section 1.2.**

---

## Finding 6: ◊ Paradox-Hunt — step_dynamics ordering: decay THEN reinforce vs. reinforce THEN decay

### Contradiction

The spec (section 1.5, step_dynamics algorithm) says:
1. Strength decay (all memories): `decayed_strength = strength_decay(beta, m.strength, delta_t)`
2. If accessed: `new_strength = strength_update(alpha, decayed_strength, s_max)`

So the order is: **decay first, then reinforce** (for the accessed memory).

The Lean formalization's `expected_strength_update` (core.py line 166-171):
```
E[S'] = (1 - q*alpha) * exp(-beta*dt) * S + q*alpha * S_max
```

Expanding for q=1 (accessed with certainty):
```
E[S'] = (1 - alpha) * exp(-beta*dt) * S + alpha * S_max
```

But the spec's sequential application gives:
```
decayed = S * exp(-beta*dt)
new = decayed + alpha * (S_max - decayed)
    = S*exp(-beta*dt) + alpha * (S_max - S*exp(-beta*dt))
    = S*exp(-beta*dt) + alpha*S_max - alpha*S*exp(-beta*dt)
    = (1 - alpha) * exp(-beta*dt) * S + alpha * S_max
```

These are **algebraically identical**. The paradox resolves: the sequential
"decay then reinforce" in the engine produces exactly the same result as the
Lean formalization's single-step formula. This is because `strength_update` is
`S + alpha*(S_max - S)` which is `(1-alpha)*S + alpha*S_max`, and composing with
prior decay `S*exp(-beta*dt)` gives the Lean formula.

### Why it matters

The paradox resolves cleanly, but the equivalence is non-obvious. If someone reorders
the operations (reinforce then decay), the result would be different:
```
reinforced = S + alpha*(S_max - S)
new = reinforced * exp(-beta*dt)
    = ((1-alpha)*S + alpha*S_max) * exp(-beta*dt)
    = (1-alpha)*exp(-beta*dt)*S + alpha*S_max*exp(-beta*dt)
```

This is NOT the same — the `alpha*S_max` term is also decayed. So the ordering matters.

### Mitigation

**Add a proof-of-equivalence note to section 1.5 (step_dynamics):** "The sequential
application 'decay then reinforce' produces the same result as the Lean formalization's
single-step formula `(1-alpha)*exp(-beta*dt)*S + alpha*S_max` (for q=1). This
equivalence is algebraic. WARNING: The reverse order 'reinforce then decay' is NOT
equivalent. The implementation MUST decay first, then reinforce."

**Added to spec section 1.5.**

---

## Finding 7: ⊘ Level-Split — Sensitivity analysis conflates parameter sensitivity and scenario sensitivity

### Confusion

The `analyze_parameter` function computes metrics across all non-stability scenarios,
then averages them:
- `kendall_tau`: mean across scenarios
- `score_change`: mean across scenarios
- `scenarios_correct`: count across scenarios

This averaging conflates two distinct sensitivities:
1. **Parameter sensitivity**: "How much does this parameter affect the system's
   behavior?" (answer: the mean tau and mean score change)
2. **Scenario sensitivity**: "Which scenarios are fragile to perturbation?" (answer:
   which scenarios flip their winner under perturbation)

The `scenarios_correct` count captures scenario sensitivity partially, but it is
mixed into the classification along with the parameter-level metrics.

### Why it matters

Consider a parameter that flips the winner in exactly one scenario at every
perturbation level, but has tau=0.99 and tiny score changes in all other scenarios.
The `scenarios_correct` count would be `total_scenarios - 1`, and the tau average
would be near 1.0. This parameter would be classified "insensitive" even though
it completely controls one scenario.

The classification rules (section 2.5) partially address this with the
`scenarios_correct < total_scenarios / 2` threshold for "critical", but a parameter
that flips 1 out of 8 scenarios (12.5%) at every perturbation level would be
classified "insensitive" even though it is the sole determinant of that scenario's
outcome.

### Mitigation

1. **Add per-scenario breakdown to SensitivityResult:** Add a field
   `scenario_flipped_per_level: list[list[str]]` that records which scenario names
   had their winner flipped at each perturbation level. This preserves scenario-level
   information that the averaging destroys.
2. **Add to generate_report:** In the "Critical Parameters" and "Sensitive Parameters"
   sections, list which specific scenarios are affected and at what perturbation level.
3. **Clarify in section 2.4 docstring:** "kendall_tau_per_level and
   score_change_per_level are AVERAGED across non-stability scenarios. Individual
   scenario breakdowns are available in scenario_flipped_per_level."

**Added to spec sections 2.4 and 2.6.**

---

## Finding 8: ⊘ Level-Split — Are the 4 competency categories orthogonal?

### Confusion

The spec defines 4 competencies:
- **accurate_retrieval**: "relevance dominates"
- **test_time_learning**: "new memories can compete" (novelty bonus)
- **selective_forgetting**: "old memories lose" (strength decay)
- **stability**: "no monopolization" (contraction/anti-thrashing)

Are these orthogonal?

- **accurate_retrieval vs. selective_forgetting**: Both test "who wins." AR tests
  that high-relevance wins; SF tests that stale memories lose. These are
  complementary perspectives on the same ranking mechanism. A parameter perturbation
  that degrades AR could also degrade SF (e.g., increasing w2 makes recency more
  important, hurting AR scenarios where the relevant memory is old, while potentially
  helping SF scenarios where the stale memory has low recency).

- **test_time_learning vs. accurate_retrieval**: TTL-1 has a new memory with
  relevance=0.9. It wins partly because of relevance AND partly because of novelty
  bonus. If you remove the novelty bonus, it still wins on relevance alone (0.9 vs
  0.5 and 0.4). This means TTL-1 does not isolate the novelty bonus — it conflates
  novelty with relevance advantage.

- **stability vs. selective_forgetting**: STAB-1 checks that no memory monopolizes.
  SF-2 checks anti-lock-in. These test the same underlying mechanism (strength decay
  prevents lock-in) but from different angles. A parameter change that affects one
  likely affects both.

### Why it matters

When the sensitivity report says "parameter X affects accurate_retrieval scenarios
but not selective_forgetting scenarios," this should be interpreted carefully. The
categories are not independent experimental conditions — they are overlapping
projections of the same scoring/dynamics system.

For TTL-2 specifically: the scenario is designed to test novelty bonus in isolation
(relevance=0.0), but TTL-1 does not. If the sensitivity analysis finds that
`novelty_start` only matters for TTL-2 but not TTL-1, it is because TTL-1's relevance
advantage masks the novelty effect — not because novelty is unimportant.

### Mitigation

1. **Add to spec section 2.3 (Standard Scenario Suite):** "OVERLAP NOTE: The four
   competency categories are not orthogonal experimental conditions. They represent
   different qualitative aspects of system behavior that share underlying mechanisms.
   A parameter that appears 'insensitive' within one category may still indirectly
   affect that category through shared mechanisms. The classification should be
   interpreted as 'primary sensitivity channel,' not 'exclusively affects this
   category.'"
2. **Strengthen TTL-1 isolation:** Change TTL-1 memory 0's relevance from 0.9 to 0.6
   so that the novelty bonus is the decisive factor, not just an additive bonus on
   top of an already-winning relevance advantage. Alternatively, add a note that
   TTL-1 tests "novelty bonus + relevance synergy" while TTL-2 tests "novelty bonus
   in isolation."

**Added to spec section 2.3. TTL-1 kept as-is with a clarifying note (changing
relevance would alter the scenario's character).**

---

## Finding 9: ◊ Paradox-Hunt — ScoringWeights uses `assert` but ParameterSet uses `ValueError`

### Contradiction

The spec (section 1.2) says: "Violations raise ValueError with a descriptive message
(not AssertionError — these are data-validation errors, not programming errors)."

But `core.py` ScoringWeights (line 79) uses `assert`:
```python
assert getattr(self, name) >= 0, f"{name} must be non-negative"
```

ParameterSet's `to_scoring_weights()` creates a ScoringWeights from ParameterSet's
w1-w4. Since ParameterSet already validates these constraints with ValueError, the
ScoringWeights assert should never fire under normal use. But:

1. If someone constructs ScoringWeights directly (bypassing ParameterSet), they get
   AssertionError instead of ValueError.
2. If Python is run with `-O` (optimizations), asserts are stripped. ScoringWeights
   validation silently disappears.

### Why it matters

This is a pre-existing issue in core.py (not in the spec), but the spec builds on it
without acknowledging the inconsistency. The spec says "the engine MUST NOT
reimplement any function from core.py" but core.py's ScoringWeights validation uses a
different error type than the spec requires for ParameterSet.

If a test checks `pytest.raises(ValueError)` on ScoringWeights construction, it will
fail — the actual exception is AssertionError.

### Mitigation

**The spec cannot change core.py** (it is a translation of the Lean formalization).
Add a note to section 1.2 (to_scoring_weights): "NOTE: core.py's ScoringWeights uses
`assert` for validation (raising AssertionError). ParameterSet validates the same
constraints with ValueError before calling to_scoring_weights(). The ParameterSet
validation is the authoritative check; the ScoringWeights assert is a defense-in-depth
backstop from the Lean translation."

**Added to spec section 1.2.**

---

## Finding 10: ◊ Paradox-Hunt — Stability scenarios use expected_winner=-1 but Scenario validation requires valid index

### Contradiction

The `Scenario.__post_init__` (section 2.2) validates:
```python
if not (0 <= self.expected_winner < len(self.memories)):
    raise ValueError(...)
```

But stability scenarios use `expected_winner=-1` as a sentinel. The value `-1` fails
the validation check `0 <= -1` -> False -> ValueError raised.

The tests in `test_sensitivity.py` (line 108-113) check for this:
```python
if scenario.expected_winner == -1:
    assert scenario.competency == "stability"
```

But `Scenario(-1)` would never be constructible under the current validation.

### Why it matters

This is a **spec bug**. `build_standard_scenarios` cannot create stability scenarios
with `expected_winner=-1` because the Scenario constructor would reject them.

### Mitigation

**Fix the Scenario validation to allow -1 for stability scenarios:**
```python
def __post_init__(self) -> None:
    ...
    if self.expected_winner == -1:
        if self.competency != "stability":
            raise ValueError(
                "expected_winner=-1 (sentinel) is only valid for stability scenarios"
            )
    elif not (0 <= self.expected_winner < len(self.memories)):
        raise ValueError(...)
```

**Updated in spec section 2.2.**

---

## Finding 11: ⊘ Level-Split — simulate() conflates time model with step count

### Confusion

The `simulate` function uses `current_time = float(step)` (section 1.5), normalizing
time to step count. But `MemoryState` has `creation_time` and `last_access_time` as
floats in arbitrary time units. If memories are constructed with creation_time=100.0
and the simulation runs from step 0 (current_time=0.0), then creation_time > current_time.

More specifically: `step_dynamics` advances `creation_time += delta_t` and
`last_access_time += delta_t` for non-accessed memories. But `simulate` passes
`current_time = float(step)` to `rank_memories` and `score_memory`. The
`score_memory` function uses `memory.creation_time` (the cumulative time) for novelty
bonus calculation, NOT `current_time - memory.creation_time`. So `current_time` is
passed to `score_memory` and `rank_memories` but never actually used inside them
for novelty computation — the novelty computation uses `memory.creation_time` directly.

Looking at the spec's `score_memory` (section 1.5):
```
boost = novelty_bonus(params.novelty_start, params.novelty_decay, memory.creation_time)
```

This uses `memory.creation_time` as the `t` argument. The `core.novelty_bonus` computes
`N0 * exp(-gamma * t)`. So `creation_time` is being used as "age of the memory" —
but `step_dynamics` *increments* creation_time by delta_t each step. So after 100 steps
with delta_t=1.0, a memory that started with creation_time=50.0 would have
creation_time=150.0.

This means creation_time is actually "age of the memory at creation + elapsed time
since simulation start" — a monotonically increasing value. The novelty bonus thus
decays over time as intended.

But the `current_time` parameter to `score_memory` is **unused**. It is passed through
to `rank_memories` and `select_memory` but none of the scoring logic consumes it.

### Why it matters

The `current_time` parameter is dead weight in the function signatures. This is
confusing — a reader might think score depends on absolute time. It does not; it
depends only on the state fields within `MemoryState`.

### Mitigation

**Add to spec section 1.5 (score_memory):** "NOTE: The `current_time` parameter is
accepted for API consistency (future extensibility for time-varying queries) but is
NOT used in the current implementation. All time-dependent computations use the
MemoryState's own fields (creation_time for novelty, last_access_time for retention).
This parameter exists so that future extensions (e.g., time-varying relevance) do not
require API changes."

**Added to spec section 1.5.**

---

## Finding 12: ◊ Paradox-Hunt — Weight perturbation violates OAT assumption

### Contradiction

The sensitivity analysis is described as "One-at-a-time (OAT)" (section 2). OAT means:
perturb one parameter, hold all others constant, measure the effect.

But when perturbing a weight (w1, w2, w3, w4), the `perturb_parameter` function
(section 2.6) renormalizes the other weights: "set new_wi = current_wi * (1 + factor).
Distribute the residual 1.0 - new_wi proportionally among the other three weights."

This means perturbing w1 changes w2, w3, and w4 simultaneously. This is not OAT —
it is a multi-parameter perturbation. The spec acknowledges this in the Risks table
(section 7): "Weight perturbation renormalization changes multiple parameters at once.
OAT assumption violated for weights."

But the classification rules (section 2.5) do not account for this. A weight parameter
might be classified "critical" not because it is individually critical, but because
the renormalization of the other weights amplifies the effect.

### Why it matters

This produces misleading classifications. If w1 is perturbed +50% and the
renormalization reduces w3 (importance weight) significantly, scenarios where
importance is the tiebreaker will flip — but the flip is caused by w3 reduction,
not w1 increase. The analysis would blame w1 for a w3 effect.

### Mitigation

1. **Add a renormalization-aware caveat to the classification:** In section 2.5, add:
   "Weight parameters (w1-w4) are subject to OAT violation due to sum-to-1
   renormalization. Their classifications should be interpreted as 'sensitivity of
   the system to shifting weight balance toward/away from this component,' not
   'sensitivity to this specific weight value.'"
2. **In generate_report:** Flag weight parameters with a footnote: "* Weight
   parameters are renormalized, see methodology notes."
3. **In SensitivityResult:** Add an optional `is_weight: bool` field (default False)
   to mark weight parameters for distinct reporting.

**Added to spec sections 2.4, 2.5, and 2.6.**

---

## Finding 13: ◊ Paradox-Hunt — retention(t=0, S=0) is undefined but can occur

### Contradiction

`core.retention(t, S)` computes `exp(-t/S)`. When `S=0`:
- If `t=0`: `exp(-0/0)` = `exp(NaN)` = NaN
- If `t>0`: `exp(-inf)` = 0.0

The spec (section 1.5, score_memory) guards against this: "Use max(strength, 1e-12)
to avoid division by zero." But this guard is in the engine's `score_memory`, not in
core.py's `retention`. The spec says "engine.py MUST NOT reimplement any function
from core.py." The guard `max(strength, 1e-12)` is not reimplementation — it is an
argument transformation before calling `retention`.

But what about the case `t=0, S=0`? With the guard: `retention(0, 1e-12)` =
`exp(-0/1e-12)` = `exp(0)` = 1.0. This means a memory with zero strength and zero
time-since-access gets recency = 1.0 (maximum recency). Is this correct?

A memory with zero strength should have decayed completely. Its recency should
arguably be low, not maximum. But the retention formula `exp(-t/S)` at t=0 is
always 1.0 regardless of S (since t=0 means "just accessed").

### Why it matters

The guard produces mathematically correct behavior for the formula at t=0: if you
just accessed the memory, recency is 1.0. But it produces a discontinuity at
`S -> 0+, t -> 0+`. With S=1e-12 and t=1e-6: `exp(-1e-6/1e-12)` = `exp(-1e6)` = 0.
So the guard creates a cliff: recency jumps from 0 to 1 exactly at t=0.

For the sensitivity analysis, this edge case is unlikely to be triggered (scenarios
have sensible strengths), but it should be documented.

### Mitigation

**Add to section 1.5 (score_memory, Algorithm step 1):** "The IEEE754 guard
`max(strength, 1e-12)` prevents NaN/Inf from `retention(0, 0)` but creates a
discontinuity: `retention(epsilon, 1e-12)` ~ 0 for any `epsilon > 0` while
`retention(0, 1e-12)` = 1.0. This is physically correct (t=0 means 'just accessed')
but numerically discontinuous. Scenarios should avoid memories with both
`strength ~ 0` and `last_access_time ~ 0`."

**Added to spec section 1.5.**

---

## Finding 14: ◊ Paradox-Hunt — Skipped perturbation levels scored as 0 in scenarios_correct

### Contradiction

Section 2.6 (analyze_parameter algorithm) says that when a perturbation is skipped
(constraint violation), it records:
```python
scenarios_correct_counts.append(0)
```

But section 2.5 (classification rules) says:
- "critical" if "any non-skipped perturbation level has scenarios_correct < total/2"

The classification rules check for "non-skipped" in the contraction margin condition
but the `scenarios_correct` check says "any non-skipped perturbation level." If
skipped levels have scenarios_correct=0, and the classification checks for
"non-skipped" levels, then skipped levels are excluded from the critical check.
Good — this is consistent.

BUT: the "sensitive" classification says "any perturbation level with |factor| >= 0.25
has scenarios_correct < 0.75 * total." This does NOT say "non-skipped." If a skipped
level at factor=0.5 has scenarios_correct=0, this condition fires and classifies the
parameter as "sensitive" even though the perturbation was invalid.

### Why it matters

Parameters near constraint boundaries (like alpha near 1.0, where factor=0.5 would
push it to 1.5 and get skipped) would be classified "sensitive" due to the skipped
level's scenarios_correct=0, not due to any actual behavioral impact.

### Mitigation

**Fix the classification rules in section 2.5:**
- "sensitive" condition: "Any **non-skipped** perturbation level with |factor| >= 0.25
  has scenarios_correct < 0.75 * total_scenarios"
- "insensitive" condition: "All **non-skipped** perturbation levels have
  scenarios_correct > 0.75 * total_scenarios"

Add explicit "non-skipped" qualification to ALL classification rules.

**Updated in spec section 2.5.**

---

## Summary of Findings

| # | Operator | Finding | Severity | Spec Section Modified |
|---|----------|---------|----------|----------------------|
| 1 | ⊘ | ParameterSet conflates config/runtime | Medium | 1.2, 2.6 |
| 2 | ⊘ | MemoryState conflates query-dependent/intrinsic | Low | 1.3 |
| 3 | ◊ | Score range [0,1] vs [0,1+N0] | High | 1.5, 3.3 |
| 4 | ◊ | Tournament extension N>2 unproven | Medium | 1.5, 3.2 |
| 5 | ⊘ | Scoring vs dynamics in same module | Low | 1.2 |
| 6 | ◊ | Decay-then-reinforce ordering (resolves cleanly) | Low | 1.5 |
| 7 | ⊘ | Parameter vs scenario sensitivity conflation | Medium | 2.4, 2.6 |
| 8 | ⊘ | Competency categories not orthogonal | Low | 2.3 |
| 9 | ◊ | assert vs ValueError in ScoringWeights | Low | 1.2 |
| 10 | ◊ | expected_winner=-1 fails Scenario validation | **BUG** | 2.2 |
| 11 | ⊘ | current_time parameter is dead weight | Low | 1.5 |
| 12 | ◊ | Weight perturbation violates OAT | Medium | 2.4, 2.5, 2.6 |
| 13 | ◊ | retention(0, ~0) discontinuity | Low | 1.5 |
| 14 | ◊ | Skipped levels scored as 0 corrupts classification | **BUG** | 2.5 |

**Critical bugs found:** 2 (Findings 10 and 14)
**High-severity design issues:** 1 (Finding 3)
**Medium-severity design issues:** 4 (Findings 1, 4, 7, 12)
