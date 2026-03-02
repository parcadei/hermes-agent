# Adversarial Pass 1 -- Level-Split + Paradox-Hunt

Created: 2026-02-27
Author: architect-agent (Opus 4.6)
Target: `thoughts/shared/plans/memory-system/recall/spec.md`
Context: `hermes_memory/engine.py`, `thoughts/shared/plans/memory-system/architecture.md`

---

## Method

Two Brenner operators applied:

- **Level-Split** identifies where the spec conflates two distinct concerns
  that should be separated to prevent category errors in implementation.
- **Paradox-Hunt** identifies where two parts of the spec produce contradictory
  behavior for the same input, requiring resolution before implementation.

Findings are ordered by severity. Each finding includes a concrete mitigation
that has been propagated into the spec.

---

## Finding 1: budget_constrain min_k guarantee vs total_budget invariant

**Severity: CRITICAL**
**Operator: Paradox-Hunt**

### The Contradiction

Section 10 states two invariants simultaneously:

1. `len(output[0]) >= min(config.min_k, len(assignments))` -- never drop below min_k
2. "Total estimated tokens of output formatted strings <= config.total_budget
   (unless min_k prevents further dropping)"

The parenthetical "unless min_k prevents further dropping" acknowledges the
contradiction but does not resolve it. Consider:

- `min_k = 3`, `total_budget = 50` (tokens)
- 5 memories, each at LOW tier produces ~20 chars = 5 tokens each
- 3 LOW-tier memories = 15 tokens, fits
- But with content, each HIGH memory is ~400 chars = 100 tokens
- 3 memories even at LOW with content=None still produce metadata strings

The edge case "Single memory over budget" (row 4 of the table) says: "Demoted to
LOW, never dropped (min_k=1)." But what if a single LOW-tier metadata string is
60 chars = 15 tokens, and `total_budget = 10`? The invariant "total estimated
tokens <= budget" is violated, but min_k prevents dropping.

More critically, `RecallResult.budget_exceeded` is True, but the consumer has
no way to distinguish "slightly over due to min_k" from "massively over due to
min_k." The result says budget_exceeded=True but the context could be 10x the
budget.

### Why This Matters

The downstream consumer (PromptAssembler) injects `RecallResult.context` into a
prompt. If context is 10x the budget, the prompt may exceed the model's context
window. The consumer cannot trust the budget as an actual bound.

### Mitigation (applied to spec)

1. The invariant is restated as a **priority hierarchy**: min_k > budget.
   The budget is a best-effort constraint, not a hard guarantee.
2. A new field `budget_overflow: int` is added to `RecallResult` to report
   by how many tokens the budget was exceeded (0 when within budget).
3. The consumer contract (Section 16) is updated to note that callers MUST
   handle `budget_overflow > 0` by truncating the context string if the
   prompt would exceed limits.
4. A new test category is added: "min_k vs budget conflict scenarios."

---

## Finding 2: Score semantics crossing the engine/recall boundary

**Severity: HIGH**
**Operator: Level-Split**

### The Conflation

The spec uses "score" to mean three different things without distinguishing them:

1. **Engine composite score**: `score_memory()` returns `base + novelty_bonus`,
   range [0, 1 + novelty_start]. This is an absolute score in engine-space.
2. **Ranked score**: The float in `rank_memories()` tuples. Same value as (1),
   but now it carries positional semantics (it was used for sorting).
3. **Normalized score**: The [0,1]-mapped value in `TierAssignment.normalized_score`,
   produced by within-k min-max normalization.

The spec's `adaptive_k` operates on (2), while `assign_tiers` normalizes to (3).
But `TierAssignment.score` stores (2) and `TierAssignment.normalized_score`
stores (3). A reader cannot tell from the field name `score` which semantic
space it occupies, and the spec does not define the relationship between the
raw engine score space and the tier threshold space explicitly.

### Specific Confusion Point

Section 7 says epsilon=0.01 should catch "genuinely flat score distributions."
But engine scores range [0, ~1.3]. A gap of 0.01 in [0, 1.3] space is ~0.77%
of the range. If the optimizer changes weights such that all scores compress
into [0.8, 0.85], then 0.01 is 20% of the effective range -- not "flat" at all.
The epsilon threshold is defined in absolute engine-score space but its semantic
meaning depends on the score distribution.

### Mitigation (applied to spec)

1. Section 4 (`TierAssignment`) field docs explicitly label `score` as
   "engine-space composite score" and `normalized_score` as "within-k
   normalized presentation score."
2. A new subsection in Section 7 documents the epsilon sensitivity to score
   range, noting that epsilon should be recalibrated if the optimizer
   significantly changes the score distribution.
3. Section 3 adds a comment on `epsilon` noting its coupling to engine score
   magnitude.

---

## Finding 3: Tier assignment when all scores identical vs when k=1

**Severity: HIGH**
**Operator: Paradox-Hunt**

### The Contradiction

Section 8 defines two special cases:

- **k=1**: s_max == s_min, normalization defaults to 1.0, tier = HIGH.
- **All k scores identical (k>1)**: s_max == s_min, all get normalized 1.0,
  all tier = HIGH.

The second case is stated as "reasonable: if the system cannot distinguish them,
present them all at maximum detail and let the LLM sort it out."

But this contradicts the budget system's assumptions. If k=8 and all are HIGH,
each gets `high_max_chars` (400), totaling ~3200 chars = 800 tokens. That
exactly equals the default budget -- but only if every HIGH memory uses its
full allocation. If any content exceeds the allocation, budget_constrain must
demote, but ALL have identical scores, so the "lowest-scored" demotion order
is arbitrary (implementation-dependent, likely last-in-list).

More fundamentally: if the engine cannot distinguish 8 memories, presenting
all 8 at HIGH detail is wasteful. The tier system exists to allocate scarce
budget across unequal memories. When all are equal, the tier system should
distribute the budget evenly, not pretend all are maximally important.

### Mitigation (applied to spec)

1. Section 8 adds a new rule: when all k scores are identical AND k > 1,
   tier assignment uses a **positional fallback**: the first ceil(k/3) get
   HIGH, the next ceil(k/3) get MEDIUM, the rest get LOW. This distributes
   the budget across tiers even when scores provide no discrimination signal.
2. The k=1 case remains: single memory always gets HIGH.
3. The normalization "all 1.0" edge case documentation is updated to explain
   that the positional fallback overrides the threshold-based assignment.
4. Test cases updated to cover this.

---

## Finding 4: format_memory content=None fallback vs budget_constrain demotion

**Severity: HIGH**
**Operator: Paradox-Hunt**

### The Contradiction

Section 9 says: when `content=None`, format_memory falls back to metadata-only
rendering for all tiers. The HIGH-without-content format is:

```
[Memory] relevance=0.80 strength=3.0 importance=0.7
  Accessed 5 times, age=100 steps
```

The MEDIUM-without-content format is:

```
[Memory] relevance=0.80 importance=0.7
```

The LOW format is:

```
[Memory] rel=0.80 imp=0.70 age=100
```

Section 10 says budget_constrain demotes HIGH->MEDIUM->LOW and re-formats.
When content=None, the HIGH format (~70 chars) demotes to MEDIUM (~40 chars)
then LOW (~35 chars). The savings from HIGH->MEDIUM demotion without content
are minimal (~30 chars = ~7 tokens). For budget_constrain to be effective
without content, it may need to demote many memories for a small gain.

But the larger issue: the spec's `budget_constrain` calls `format_memory`
with the memory's content. When `contents` list is `None` (the parameter to
`budget_constrain`), the re-format call passes `content=None`. But
`budget_constrain`'s Final Signature accepts `contents: list[str | None] | None`.
There is a double-None ambiguity:

- `contents=None` means "no content list at all" (all memories get content=None)
- `contents=[None, "hello", None]` means "some memories lack content"

Both produce content=None for some memories, but the first is a blanket opt-out
while the second is per-memory. The demotion logic treats them identically, but
the test contracts differ: testing with `contents=None` exercises a different
path than testing with `contents=[None, None, None]`.

### Mitigation (applied to spec)

1. Section 10 explicitly documents the double-None semantics: `contents=None`
   is equivalent to `[None] * len(assignments)` for all demotion purposes.
   Implementation MUST normalize `contents=None` to the explicit list form
   before entering the demotion loop.
2. Section 14 (test categories) adds explicit test cases for both None forms.
3. Section 9 adds a note that demotion savings are minimal without content,
   and budget_constrain should prioritize dropping over demoting when
   content=None (since demotion without content barely saves tokens).

---

## Finding 5: Gating vs empty-memory-list ordering

**Severity: MEDIUM**
**Operator: Paradox-Hunt**

### The Contradiction

Section 11 (recall algorithm) shows:

1. Gate check: if not should_recall -> return gated result
2. Rank memories
3. adaptive_k (returns 0 for empty list)
4. ...

If memories is empty AND the message is not gated, the pipeline reaches step 3
and adaptive_k returns 0, producing `RecallResult(context="", gated=False, k=0)`.

But should the gate have intercepted this? The gate does not check whether
memories exist -- it only checks the message. This means a non-gated message
with an empty memory DB produces a result that says "I decided to recall
(gated=False) but found nothing (k=0)." This is semantically correct but
raises a design question: should `recall()` check for empty memories BEFORE
the gate, to avoid the wasted scoring call?

More importantly, the edge case table says:
- "Empty memories list: RecallResult(context="", gated=False, k=0, ...)"

But the invariant says: "If memories is non-empty and not gated: k >= 1"

The contrapositive is: "If k == 0, then either memories is empty or gated."
This is satisfied. But the result's `gated=False, k=0` is an ambiguous signal
to the consumer: was there nothing to recall, or was the query bad? The
consumer cannot distinguish "empty DB" from "all memories scored so poorly
that adaptive_k returned 0" (which cannot actually happen, since adaptive_k
returns >= min_k >= 1 when scores is non-empty).

### Mitigation (applied to spec)

1. Section 11 adds a fast-path: if `len(memories) == 0`, return immediately
   with a dedicated result (gated=False, k=0) BEFORE the gate check. This
   avoids running the gate on a query that cannot produce results anyway.
2. The RecallResult documentation adds a note: `gated=False, k=0` means
   "the memory database was empty." `gated=True, k=0` means "the gate
   decided not to recall."
3. The empty-memories fast-path is placed AFTER the gate check is removed
   for this case -- actually, it is placed BEFORE the gate check. Rationale:
   gating is about avoiding unnecessary work; if there are no memories,
   there is no work to avoid, and the gate's decision is moot.

---

## Finding 6: Configuration vs runtime state in RecallConfig

**Severity: MEDIUM**
**Operator: Level-Split**

### The Conflation

RecallConfig mixes two distinct concerns:

**A. Algorithmic parameters** (affect recall decisions):
- min_k, max_k, gap_buffer, epsilon (adaptive-k)
- high_threshold, mid_threshold (tier assignment)
- total_budget, *_budget_share (budget constraint)

**B. Presentation parameters** (affect output formatting):
- high_max_chars, medium_max_chars, low_max_chars (format_memory)
- gate_min_length, gate_trivial_patterns (gating)

These have different change frequencies and different consumers. Algorithmic
parameters are tuned alongside the optimizer and should change rarely.
Presentation parameters depend on the LLM model (context window size,
tokenizer efficiency) and may change when switching models.

Bundling them in one frozen dataclass means you cannot change presentation
without also re-specifying algorithmic parameters (or accepting defaults).
More subtly, the *_budget_share fields look like algorithmic parameters but
actually control presentation (how much space each tier gets), creating a
third implicit category.

### Mitigation (applied to spec)

1. RecallConfig remains a single dataclass (splitting it adds complexity
   without enough benefit for v1). However, the field documentation in
   Section 3 is reorganized into labeled groups with comments explaining
   the different concerns.
2. A note is added to Section 15 (Open Decisions) recording this as OD5
   for potential future splitting.
3. The budget_share fields are explicitly documented as "presentation-layer
   budget allocation" distinct from "algorithmic budget enforcement."

---

## Finding 7: assign_tiers threshold boundaries are ambiguous

**Severity: MEDIUM**
**Operator: Paradox-Hunt**

### The Contradiction

Section 8 algorithm step 4 says:
- `normalized > config.high_threshold` -> Tier.HIGH
- `config.mid_threshold < normalized <= config.high_threshold` -> Tier.MEDIUM
- `normalized <= config.mid_threshold` -> Tier.LOW

Section 4 (TierAssignment invariants) says:
- `normalized_score > config.high_threshold` implies `tier == Tier.HIGH`
- `config.mid_threshold < normalized_score <= config.high_threshold` implies `tier == Tier.MEDIUM`
- `normalized_score <= config.mid_threshold` implies `tier == Tier.LOW`

These are consistent with each other but produce a surprising result:
a normalized score of exactly `high_threshold` (e.g., 0.7) is assigned
Tier.MEDIUM, not Tier.HIGH. The threshold name "high_threshold" suggests
"the threshold for HIGH tier," but a score exactly at the threshold does
NOT get HIGH.

This is a spec-implementation mismatch waiting to happen. A developer reading
"high_threshold = 0.7" would reasonably expect score 0.7 to be HIGH.

Similarly, a score of exactly `mid_threshold` (0.4) is assigned LOW, not
MEDIUM.

### Mitigation (applied to spec)

1. Section 3 field documentation for high_threshold and mid_threshold is
   updated to explicitly state: "Score must be STRICTLY ABOVE this
   threshold to qualify for the tier."
2. Section 8 adds a boundary-value edge case table.
3. Test categories in Section 14 add boundary-value tests at exactly
   the threshold values.

---

## Finding 8: Testing contracts vs production contracts for content

**Severity: MEDIUM**
**Operator: Level-Split**

### The Conflation

The spec repeatedly distinguishes "testing without content" from "production
with content" but treats them as the same code path with content=None:

- format_memory: "When content is None (e.g., in pure-math testing)"
- budget_constrain: "When reformat_fn is None (testing without content)"

This conflates two legitimate runtime scenarios:

**A. Pure-math testing**: No storage layer, no content, MemoryState only.
   The content parameter is absent because the test infrastructure does not
   have a storage layer.

**B. Production metadata-only recall**: The storage layer exists but a
   particular memory's content was deleted/corrupted/unavailable. content=None
   is a runtime condition, not a test convenience.

In scenario A, content=None means "I don't care about content, just test
the algorithm." In scenario B, content=None means "this memory is damaged
and I should handle it gracefully." The code path is identical but the
semantic meaning and the appropriate behavior differ.

### Mitigation (applied to spec)

1. Section 9 explicitly documents both scenarios and notes that the
   content=None path serves double duty intentionally.
2. A warning is added: if a future version needs to distinguish "no content
   available" from "content not requested," the None sentinel must be
   replaced with an explicit union type (e.g., `Content | NotRequested | Missing`).
3. Section 14 test categories add at least one test that exercises the
   "production metadata-only" scenario (content=None for one memory in a
   list where others have content).

---

## Finding 9: adaptive_k epsilon fallback produces min_k + gap_buffer, not min_k

**Severity: LOW**
**Operator: Paradox-Hunt**

### The Nuance

Section 7, step 5 says:
"All scores are nearly equal. Fall back to `min(config.min_k + config.gap_buffer, len(scores), config.max_k)`"

The edge case table confirms: `[0.5, 0.5, 0.5]` with default config -> 3
(which is `min(1+2, 3, 10) = 3`).

But the "fallback" description says "all scores are nearly equal," implying
the system cannot distinguish them. If the system cannot distinguish them,
why return min_k + gap_buffer instead of just min_k? The gap_buffer is
defined as "extra memories past the score gap to include" -- but there IS
no gap. Adding gap_buffer to the fallback is an unexplained design choice.

This is not a contradiction (the algorithm is well-defined) but it is a
potential source of confusion. A developer might expect the flat-score
fallback to return min_k (the minimum), not min_k + gap_buffer.

### Mitigation (applied to spec)

1. Section 7 adds a rationale note for the fallback formula: "When all
   scores are equal, we cannot identify a natural cutoff. The gap_buffer
   acts as a 'benefit of the doubt' factor, returning more memories
   rather than fewer. This biases toward recall (showing context) over
   silence (dropping potentially useful memories)."

---

## Finding 10: budget_share fields are unused in budget_constrain

**Severity: LOW**
**Operator: Level-Split**

### The Conflation

Section 3 defines `high_budget_share`, `medium_budget_share`, `low_budget_share`
(summing to 1.0). These suggest that the budget is partitioned by tier: 55%
for HIGH, 30% for MEDIUM, 15% for LOW.

But Section 10 (budget_constrain) never references these fields. The algorithm
simply checks total token count against total_budget. The per-tier budget
shares are defined, validated (must sum to 1.0), but never consumed.

This is dead configuration: fields that exist in the config, pass validation,
but have no effect on behavior. A user tuning `high_budget_share` would see
no change in output.

### Mitigation (applied to spec)

1. Section 10 adds a subsection "Per-Tier Budget Allocation" that uses
   the budget_share fields. The demotion pass checks per-tier budgets:
   if HIGH-tier memories exceed `total_budget * high_budget_share` tokens,
   the lowest-scored HIGH memories are demoted to MEDIUM even if total
   budget is not exceeded. This gives the fields a purpose.
2. Alternatively, if per-tier budgets add too much complexity for v1,
   the fields are removed from RecallConfig and the validation rule
   (sum to 1.0) is removed. The spec documents this as OD6.
3. Decision: Keep the fields and implement per-tier enforcement in the
   demotion pass. This makes the budget system more granular and prevents
   a scenario where all budget goes to HIGH memories, starving MEDIUM
   and LOW of their minimum allocations.

---

## Summary

| # | Finding | Severity | Operator | Key Issue |
|---|---------|----------|----------|-----------|
| 1 | min_k vs budget invariant | CRITICAL | Paradox-Hunt | Priority hierarchy undefined; consumer cannot handle overflow |
| 2 | Score semantic spaces | HIGH | Level-Split | Three meanings of "score" conflated |
| 3 | All-identical tier assignment | HIGH | Paradox-Hunt | All-HIGH wastes budget; no positional fallback |
| 4 | content=None double semantics | HIGH | Paradox-Hunt | Demotion with no content barely saves tokens; double-None ambiguity |
| 5 | Gate vs empty-list ordering | MEDIUM | Paradox-Hunt | Ambiguous k=0 signal; wasted gate evaluation |
| 6 | Config vs presentation params | MEDIUM | Level-Split | Different change frequencies bundled |
| 7 | Threshold boundary ambiguity | MEDIUM | Paradox-Hunt | "high_threshold" does not include the threshold value |
| 8 | Test vs production content=None | MEDIUM | Level-Split | Two scenarios share one path with different semantics |
| 9 | Epsilon fallback includes gap_buffer | LOW | Paradox-Hunt | Unexplained design choice in flat-score case |
| 10 | budget_share fields unused | LOW | Level-Split | Dead configuration with no behavioral effect |

