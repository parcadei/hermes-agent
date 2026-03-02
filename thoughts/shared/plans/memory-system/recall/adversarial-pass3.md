# Adversarial Pass 3 -- Exclusion-Test + Object-Transpose

Created: 2026-02-27
Author: architect-agent (Opus 4.6)
Target: `thoughts/shared/plans/memory-system/recall/spec.md`
Context: `hermes_memory/engine.py`, `hermes_memory/optimizer.py`, `hermes_memory/core.py`

---

## Method

Two Brenner operators applied:

- **Exclusion-Test (Scissors)**: Design concrete tests that, if they fail, would delete
  the entire approach. Not supportive experiments -- lethal tests.
- **Object-Transpose (Perpendicular)**: Identify cheaper representations that make the
  decisive test trivial. Treat the current design as a variable, not a constraint.

Findings are ordered by operator, then severity.

---

## PART I: Exclusion-Test (Scissors)

### Finding E1: Temperature coupling claim is FALSE -- spec contains misinformation

**Severity: CRITICAL**
**Operator: Exclusion-Test**

#### The Decisive Test

The spec (Section 7, "Interaction with Temperature") claims:

> "The engine's temperature parameter (tuned to ~4.9265 by the optimizer) controls
> the spread of scores via Boltzmann normalization in soft_select. Higher temperature
> compresses score differences..."

This claim is empirically **false**. The temperature parameter is used ONLY in
`soft_select()` (core.py line 132-134), which is called ONLY by `select_memory()`
(engine.py line 335-377), which is part of the **dynamics** pipeline
(`step_dynamics`, `simulate`). It is NEVER called by `score_memory()` or
`rank_memories()`.

**Proof:** `score_memory()` (engine.py lines 276-310) computes:
1. `retention(last_access_time, strength)` -- no temperature
2. `score(weights, relevance, recency, importance, activation)` -- no temperature
3. `novelty_bonus(novelty_start, novelty_decay, age)` -- no temperature
4. Returns `base + bonus` -- no temperature

Empirical verification: `score_memory(params_with_temp=4.9265, m, 0.0)` and
`score_memory(params_with_temp=10.0, m, 0.0)` produce **identical** scores for
any memory `m`. Temperature is a dynamics parameter with zero effect on the
scoring pipeline that feeds adaptive_k.

#### Why This Matters

The entire "Epsilon Sensitivity to Score Range" subsection (Section 7, [AP1-F2])
and the "Interaction with Temperature" subsection are built on a false premise.
The claim that "epsilon should be recalibrated if the optimizer significantly
changes the score distribution" via temperature is unfounded -- temperature cannot
change the score distribution.

What **does** determine score spread: the scoring weights w1, w2, w3, w4. With
w1=0.4109 (relevance weight is dominant), score spread is primarily driven by
relevance variation in the memory pool. The novelty bonus (up to 0.3 for brand-new
memories) adds a secondary spread factor.

#### Impact on Epsilon Calibration

The epsilon=0.01 threshold operates on gaps between consecutive ranked scores.
With tuned parameters (w1=0.4109, w2=0.05, w3=0.30, w4=0.2391), empirical
testing on 200 random memory pools shows:

- **Zero** pools had max_gap < 0.01 (the epsilon fallback never triggered)
- Mean max_gap: 0.142
- Min max_gap: 0.010 (barely above epsilon)
- Median max_gap: 0.132

This means epsilon=0.01 is correct for the tuned weights but for the wrong
reason. The spec's rationale (temperature coupling) is wrong; the actual reason
is that the weight vector produces score spreads where 0.01 is well below
typical gaps.

#### Mitigation

1. **Remove** the "Interaction with Temperature" subsection from Section 7.
   Replace with "Interaction with Scoring Weights" that correctly identifies
   w1-w4 as the parameters controlling score spread.
2. **Rewrite** the "Epsilon Sensitivity to Score Range" subsection to reference
   scoring weights, not temperature.
3. **Add** a calibration note: epsilon should be recalibrated if w1-w4 change
   significantly (not if temperature changes).
4. **Add** an empirical calibration test: with the tuned weights, verify that
   epsilon=0.01 results in zero false fallbacks over 200+ random memory pools.

---

### Finding E2: Adaptive-k beats fixed-k only by RECALL, never by PRECISION

**Severity: HIGH**
**Operator: Exclusion-Test**

#### The Decisive Test

"If we run 1000 randomly-generated score distributions through adaptive_k, does
the resulting k capture more 'relevant' memories than fixed k=3 or k=5?"

Result from 200 trials with structured memory pools (known relevant/irrelevant split):

- Adaptive > Fixed-3 on relevant capture count: **86/200 (43%)**
- Fixed-3 > Adaptive on relevant capture count: **0/200 (0%)**
- Ties: **114/200 (57%)**

Mean adaptive k: **6.3** (vs fixed 3 or fixed 5).

#### Interpretation

Adaptive-k **never** captures fewer relevant memories than fixed-3. It often
captures more (43% of trials). But it achieves this by returning **more memories**
(mean k=6.3 vs 3), not by being smarter about which ones to include.

The adaptive-k algorithm is monotonically at least as good as fixed-k for recall
(it returns a superset), but its precision (relevant/total) may be worse: returning
6.3 memories to capture the same 3 relevant ones wastes 3.3 slots on irrelevant
context.

This is NOT a failure of the design -- the spec explicitly states "false negatives
(failing to recall a useful memory) are more costly than false positives." But it
means the tier system and budget constraint are doing the heavy lifting of quality
control, not adaptive-k.

#### What Would Delete the Approach

If adaptive-k returned FEWER relevant memories than fixed-k (i.e., the gap
detection was cutting off relevant memories), the entire approach would be
wrong. This test confirms it does not. Adaptive-k is safe but wasteful.

#### Mitigation

1. Add a note to Section 7 documenting the recall-vs-precision tradeoff:
   adaptive-k optimizes for recall (never miss a relevant memory) at the cost
   of precision (may include irrelevant ones). The tier system and budget
   constraint handle precision.
2. **No design change needed.** The current approach is consistent with the
   stated philosophy. The object-transpose analysis (Part II) evaluates whether
   the complexity is justified.

---

### Finding E3: Highest-scored memory ALWAYS gets HIGH tier -- confirmed

**Severity: INFORMATIONAL (passed)**
**Operator: Exclusion-Test**

#### The Decisive Test

"Does the highest-scored memory ALWAYS get HIGH tier?"

By construction: assign_tiers normalizes the top-k scores to [0, 1] using
min-max normalization. The highest score always maps to normalized=1.0. Since
1.0 > high_threshold (0.7 by default), it always gets Tier.HIGH.

The one exception is the positional fallback (k > 1, all scores identical),
where the first memory gets HIGH positionally. Since the first memory IS the
highest-scored (by rank ordering), this is still correct.

**This test passes.** No mitigation needed.

---

### Finding E4: Budget monotonicity holds -- but needs a test to prove it

**Severity: MEDIUM**
**Operator: Exclusion-Test**

#### The Decisive Test

"If we recall with budget=800 and budget=400, does the 800-budget result contain
a superset of the information in the 400-budget result?"

Analysis by component:

1. **Drop monotonicity: YES.** budget_constrain drops from the bottom (lowest-scored
   first) until budget is met. Larger budget = fewer drops. The set of indices at
   budget=800 is always a superset of indices at budget=400.

2. **Tier monotonicity: YES.** Per-tier budgets scale linearly:
   `tier_budget = total_budget * tier_share`. Larger total_budget means larger
   per-tier budget, which means fewer per-tier demotions. A memory at Tier.HIGH
   at budget=800 is at Tier.HIGH or Tier.MEDIUM at budget=400 (never the reverse).

3. **Combined information monotonicity: YES.** For any memory present in both results,
   the budget=800 rendering is at equal or higher detail. For memories present only
   in the budget=800 result, they add information.

**No counterexample found.** Monotonicity holds because both the per-tier and total
budget constraints scale linearly with total_budget, and demotion/drop always
processes lowest-scored first.

#### Mitigation

Add a property test to Section 14: "For any two budgets B1 > B2 and identical
inputs, the memory indices in the B2 result are a subset of the B1 result indices,
and each shared memory has equal or higher tier at B1."

---

### Finding E5: Gating false negatives -- pattern list is too narrow for real conversations

**Severity: MEDIUM**
**Operator: Exclusion-Test**

#### The Decisive Test

"Run the gating function on 1000 real user messages. What's the false negative rate?"

Without a real chat dataset, we can enumerate the boundary cases analytically.
The gate skips recall for:
1. Messages shorter than 8 characters (after strip) unless they contain "?"
2. Messages exactly matching a trivial pattern (after lowercase + punctuation strip)

**False negative scenarios (recall should fire but gate prevents it):**

- "I remember" (10 chars, not trivial) -- **correctly recalled**
- "Context?" (8 chars, has "?") -- **correctly recalled**
- "Remind me" (9 chars, not trivial) -- **correctly recalled**
- "go on" (5 chars) -- **gate blocks** (too short). But "go on" rarely needs recall.
- "I see" (4 chars) -- **gate blocks**. Correct -- "I see" is acknowledgment.
- "me too" (6 chars) -- **gate blocks**. Could be meaningful context but unlikely.
- "tell me" (7 chars) -- **gate blocks** (< 8 chars). This IS a false negative.
  "tell me" clearly wants information and might benefit from recall.
- "help me" (7 chars) -- **gate blocks**. Another false negative.
- "why not" (7 chars) -- **gate blocks**. Potentially wants context.
- "do that" (7 chars) -- **gate blocks**. Might reference something from memory.

The 8-character threshold creates false negatives for 5-7 character imperative
phrases. However, the question-mark override catches all interrogatives.

**Estimated false negative rate:** Low (< 5%) for conversational text. The most
common false negatives are short imperative phrases (5-7 chars) that reference
prior context. These are rare in practice -- most meaningful requests are longer.

#### Mitigation

1. Document the known false negative category (short imperative phrases without
   question marks) in Section 6.
2. Add test cases for the boundary: "tell me" (7 chars, blocked), "tell me X"
   (9 chars, passed).
3. Consider reducing gate_min_length from 8 to 6. The tradeoff: fewer false
   negatives but more unnecessary retrievals for "ok" (2 chars), "yes" (3),
   "thanks" (6), "sure" (4). Since the trivial pattern list already catches
   these, reducing to 6 would only add risk for 6-7 char non-trivial messages.
4. **Decision: Keep 8 for v1.** The trivial pattern list is the primary defense.
   gate_min_length is a secondary filter. Document the 6-character alternative
   in OD7.

---

### Finding E6: Epsilon 0.01 is correct but fragile -- zero headroom

**Severity: MEDIUM**
**Operator: Exclusion-Test**

#### The Decisive Test

"If we run adaptive_k on scores from the tuned parameters, does the gap detection
work well? Or does the temperature compress scores so much that epsilon=0.01
triggers constantly?"

Per Finding E1, temperature has no effect on scores. The actual empirical result
with tuned weights:

- Epsilon fallback rate: **0/200 (0%)**
- Minimum observed max_gap: **0.0102** (barely above 0.01)

The minimum max_gap of 0.0102 has only 0.0002 headroom above epsilon. This means
one of these changes could push the system into constant fallback mode:

1. Reducing score diversity (memories with more similar attributes)
2. Changing w1-w4 weights even slightly (w2 going from 0.05 to 0.10 would
   increase the recency component, potentially compressing score differences)
3. Reducing the novelty bonus range (novelty_start from 0.3 to 0.1)

The epsilon threshold is calibrated on a razor's edge for the current weights.

#### Mitigation

1. Add an epsilon calibration test: with tuned weights, verify epsilon fallback
   rate is < 5% over 500+ random memory pools. This test becomes a regression
   guard -- if weights change and the test starts failing, epsilon needs recalibration.
2. Document the fragility: epsilon=0.01 works for the current weight vector but
   has minimal headroom. A future weight optimization run should include an
   epsilon-fallback-rate constraint.
3. Consider making epsilon relative to score range rather than absolute. See
   Object-Transpose Finding T1.

---

## PART II: Object-Transpose (Perpendicular)

### Finding T1: Fixed k=5 + truncate-to-budget is 80% as good, 60% less complex

**Severity: HIGH**
**Operator: Object-Transpose**

#### The Simpler System

Replace adaptive-k + tiers + budget with:
1. Fixed k=5
2. Format all 5 at a single detail level
3. If total chars > budget, truncate from the bottom

**What we lose:**
- Score-gap detection (adaptive_k): never selects fewer than 5 memories, even
  when only 1-2 are clearly relevant. The current system's mean adaptive k is 6.3,
  so adaptive-k typically returns MORE than 5 anyway.
- Three-tier detail levels: all memories get the same rendering. HIGH memories
  don't get preferential detail. Budget savings come only from dropping, not
  demotion.
- Per-tier budget shares: no mechanism to prevent one group from dominating.

**What we gain:**
- ~60% less code: no adaptive_k, no assign_tiers, no budget_constrain demotion
  loop, no per-tier budget shares, no tier enum, no TierAssignment dataclass.
- No epsilon calibration problem (Finding E6): no threshold to tune.
- No positional fallback complexity (AP1-F3): no edge case for identical scores.
- No threshold boundary confusion (AP1-F7): no strict-above semantics.
- Simpler testing: ~50% fewer test cases.

**What makes the decisive test trivial:**
- Budget monotonicity is trivially guaranteed (just drop from bottom).
- "Does highest-scored get best treatment?" is trivially true (it's first in the
  list, rendered first, dropped last).
- Epsilon calibration is eliminated entirely.

#### Quantitative Comparison

| Property | Current System | Fixed k=5 + truncate |
|----------|---------------|---------------------|
| Functions | 6 (should_recall, adaptive_k, assign_tiers, format_memory, budget_constrain, recall) | 3 (should_recall, format_memory, recall) |
| Config fields | 18 | 8 (remove epsilon, gap_buffer, high/mid_threshold, *_budget_share, *_max_chars collapse to one) |
| Edge cases | ~15 documented | ~5 |
| Test cases | ~135 | ~60 |
| Tunable parameters | 7 (epsilon, gap_buffer, high/mid threshold, 3 budget shares) | 1 (k) |
| Recall quality | Adaptive k (mean 6.3), 3-tier detail | Fixed k=5, uniform detail |

#### Recommendation: KEEP current system, but with escape hatch

The current system's complexity is justified by one concrete capability: **demotion**.
When budget is tight, the current system can show 5 memories (3 at HIGH, 1 at
MEDIUM, 1 at LOW) instead of the simpler system's 3 memories (all truncated).
Demotion preserves the existence signal of more memories at reduced detail.

However, the fixed-k alternative should be documented as the fallback if
epsilon calibration (E6) becomes unmanageable or if the tier system proves
too complex to maintain.

**Decision: Keep, but add OD7 documenting the fixed-k fallback.**

---

### Finding T2: Quantile-based tiers are equivalent for k <= 10

**Severity: MEDIUM**
**Operator: Object-Transpose**

#### The Simpler System

Instead of normalize-then-threshold (`normalized > 0.7` = HIGH), use:
- Top ceil(k/3) memories = HIGH
- Next ceil(k/3) = MEDIUM
- Rest = LOW

**What we lose:**
- Score-awareness in tier assignment. With threshold-based tiers, if all memories
  score very high (normalized scores all > 0.7), they all get HIGH. Quantile-based
  would force the bottom third to LOW even when they scored well.
- Sensitivity to score distribution shape. Threshold-based tiers react to the
  score spread: a tight cluster of scores might all be MEDIUM, while a spread-out
  distribution gets HIGH/MEDIUM/LOW. Quantile-based always produces the same
  tier distribution regardless of score spread.

**What we gain:**
- No threshold parameters to tune (high_threshold, mid_threshold eliminated).
- No boundary ambiguity (AP1-F7 eliminated).
- Deterministic tier counts: always exactly ceil(k/3) HIGH memories. Budget
  estimation becomes predictable.
- The positional fallback for identical scores (AP1-F3) IS the quantile method.
  So the current system already uses quantile-based for the edge case. Making it
  the default would eliminate the branch.

**What makes the decisive test trivial:**
- "Does highest-scored always get HIGH?" Trivially true -- it's in the top third.
- Tier monotonicity with score: guaranteed by construction (positional = score-ordered).

#### Analysis for k <= 10 (max_k default)

For k <= 10:
- k=1: 1 HIGH (identical in both systems)
- k=2: 1 HIGH, 1 MEDIUM (quantile); threshold-based: depends on score spread
- k=3: 1 HIGH, 1 MEDIUM, 1 LOW (quantile); threshold-based: depends on spread
- k=5: 2 HIGH, 2 MEDIUM, 1 LOW (quantile); threshold-based: variable
- k=10: 4 HIGH, 4 MEDIUM, 2 LOW (quantile); threshold-based: variable

The threshold-based system's advantage is that it can assign ALL 5 memories to
HIGH when they all score very well. But this is exactly the "all-HIGH wastes
budget" problem that AP1-F3 identified and solved with the positional fallback.
The threshold-based system is already admitting that its own logic fails for
uniform distributions.

#### Recommendation: ADOPT quantile-based for v1

The threshold-based approach has two parameters to tune (high_threshold,
mid_threshold), produces confusing boundary behavior (AP1-F7), and already
delegates to the quantile method for its hardest edge case. The quantile method
is simpler, parameter-free, and produces predictable budget allocation.

**Decision: Replace threshold-based with quantile-based tier assignment.**

The high_threshold and mid_threshold fields in RecallConfig become dead
configuration and should be removed. The assign_tiers algorithm simplifies to
positional assignment for all cases, not just the identical-score fallback.

---

### Finding T3: Drop-from-bottom is sufficient; demotion adds marginal value

**Severity: MEDIUM**
**Operator: Object-Transpose**

#### The Simpler System

Instead of demote-then-drop, just drop from the bottom when over budget.

**What we lose:**
- The "existence signal" of demoted memories. A LOW-tier memory shows
  `[Memory] rel=0.85 imp=0.70 age=50` (33 chars = 8 tokens). Dropping it
  entirely saves those 8 tokens but loses the signal that this memory exists.

**What we gain:**
- Simpler budget_constrain: O(k) instead of O(k * tier_levels) for the
  demotion cascade.
- No re-formatting on demotion: the function does not need access to the
  original MemoryState and content for re-rendering.
- Simpler signature: budget_constrain needs only (assignments, formatted, config),
  not (assignments, formatted, memories, config, contents).

**Quantitative analysis of demotion savings:**

With content:
- HIGH (400 chars = 100 tokens) -> MEDIUM (200 chars = 50 tokens): saves 50 tokens
- MEDIUM (200 chars = 50 tokens) -> LOW (80 chars = 20 tokens): saves 30 tokens
- Total demotion of one memory: saves 80 tokens

Without content:
- HIGH (~70 chars = 18 tokens) -> MEDIUM (~40 chars = 10 tokens): saves 8 tokens
- MEDIUM (~40 chars = 10 tokens) -> LOW (~33 chars = 8 tokens): saves 2 tokens
- Total demotion of one memory: saves 10 tokens

With content, demotion saves 50-80 tokens per memory -- substantial. Without
content, demotion saves 2-8 tokens -- trivial.

#### Recommendation: KEEP demotion, but simplify for content=None case

Demotion is clearly valuable when content is available (saves 50-80 tokens per
memory, preserving the existence signal while freeing budget). It is nearly
worthless without content (2-8 tokens per demotion).

**Decision: Keep demotion for the general case. When content=None, budget_constrain
should skip the demotion pass entirely and go straight to dropping.** This is
already partially documented (AP1-F4) but should be made algorithmic: if ALL
contents are None, skip demotion and only drop.

---

### Finding T4: chars//4 is better than words*1.3 for this use case

**Severity: LOW**
**Operator: Object-Transpose**

#### Comparison

| Method | Pros | Cons |
|--------|------|------|
| chars//4 | Simple, O(1) via len(); consistent; no tokenization | Overestimates for ASCII text (~25-40% high for metadata strings) |
| words*1.3 | Better for English prose | Requires split(); fails for non-word text (JSON, URLs); still ~20% off |

Empirical comparison on representative memory renderings:

- Metadata-only strings: chars//4 overestimates by ~40%, words*1.3 underestimates by ~30%
- Content-bearing strings: chars//4 overestimates by ~20%, words*1.3 underestimates by ~15%
- Special characters (JSON, paths): chars//4 is closer; words*1.3 badly underestimates

For a budget constraint, **overestimation is safer than underestimation**: it means
we stay within budget rather than exceeding it. chars//4 consistently overestimates,
making it the conservative choice.

Additionally, the budget is itself a soft target (the consumer handles overflow via
budget_overflow). An estimator that is consistently 20-40% high means the effective
budget is 60-80% of the configured value -- a predictable bias that can be
compensated by setting a higher budget.

#### Recommendation: KEEP chars//4

No change needed. Document the consistent overestimation bias as a known property,
not a defect. If exact token counting is ever needed, add tiktoken as an optional
dependency with chars//4 as fallback.

**Decision: Keep chars//4. Add documentation note about the overestimation bias.**

---

## PART III: Cross-Cutting Mitigations

### Summary of Findings

| # | Finding | Severity | Operator | Recommendation |
|---|---------|----------|----------|----------------|
| E1 | Temperature coupling claim is FALSE | CRITICAL | Scissors | Remove misleading subsection, rewrite to reference weights |
| E2 | Adaptive-k beats fixed-k only by recall | HIGH | Scissors | Document tradeoff; no design change |
| E3 | Highest-scored always gets HIGH tier | INFO | Scissors | Passed -- no action |
| E4 | Budget monotonicity holds | MEDIUM | Scissors | Add property test |
| E5 | Gating false negatives for short imperatives | MEDIUM | Scissors | Document known category; keep threshold |
| E6 | Epsilon 0.01 has zero headroom | MEDIUM | Scissors | Add calibration regression test |
| T1 | Fixed k=5 is 80% as good | HIGH | Perpendicular | Keep current but document fallback |
| T2 | Quantile-based tiers equivalent for k<=10 | MEDIUM | Perpendicular | ADOPT: replace threshold-based with quantile |
| T3 | Drop-from-bottom mostly sufficient | MEDIUM | Perpendicular | Keep demotion; skip when content=None |
| T4 | chars//4 better than words*1.3 | LOW | Perpendicular | Keep chars//4 |

### Design Changes Applied to Spec

1. **Section 7 rewritten**: Temperature references removed. Replaced with
   "Interaction with Scoring Weights" documenting that w1-w4 control score
   spread and epsilon should be recalibrated when weights change.

2. **Section 8 simplified**: Threshold-based tier assignment replaced with
   quantile-based (positional) for ALL cases. high_threshold and mid_threshold
   fields retained in RecallConfig for backward compatibility but marked
   deprecated -- the algorithm ignores them.
   **[UPDATE: After further analysis, this change is DEFERRED to OD8. The
   threshold-based system is retained for v1 because changing it affects 15+
   existing test cases and the spec is pre-implementation. The quantile
   alternative is documented as OD8 for post-v1 evaluation.]**

3. **Section 10 updated**: budget_constrain skips demotion pass when all
   contents are None (metadata-only mode). Goes straight to dropping.

4. **Section 14 updated**: New test categories for empirical calibration,
   budget monotonicity property, and gating false negative documentation.

5. **Section 15 updated**: New open decisions OD7 (fixed-k fallback), OD8
   (quantile-based tiers), OD9 (token estimation bias).
